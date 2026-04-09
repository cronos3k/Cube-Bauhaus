//! OGZ map loader — verified against Sauerbraten src/engine/worldio.cpp (v33).
//!
//! # Exact on-disk layout (v29–v33, "new header")
//!
//!  Bytes   Field
//!  ------  -------------------------------------------------------
//!  4       magic "OCTA"
//!  4       version       (i32 LE)
//!  4       headersize    (i32 LE) — sizeof(octaheader) = 40 for v29–v33
//!  4       worldsize     (i32 LE) — actual world size, e.g. 1024
//!  4       numents       (i32 LE)
//!  4       numpvs        (i32 LE)
//!  4       lightmaps     (i32 LE)
//!  4       blendmap      (i32 LE)   \  v30–v33
//!  4       numvars       (i32 LE)   |
//!  4       numvslots     (i32 LE)   /  v30+ (v29 has no numvslots here → 0)
//!  ---     [variables block]        numvars entries
//!  ---     gametype+extras          v16+: u8 len, (len+1) bytes, u16 eif, u16 esz, esz bytes
//!  ---     texmru                   v14+: u16 n, n×u16;  v<14: 256 bytes
//!  ---     [entities]               numents entries
//!  ---     [vslots]                 numvslots entries (loadvslots)
//!  ---     [octree]                 8 root cubes (loadchildren style)
//!  ---     [lightmaps, pvs, blendmap — ignored]
//!
//! For v≤28 the header is the old `compatheader` (196 bytes total).
//! We skip everything in the compat tail and force numvars/numvslots = 0.

use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use tracing::{info, warn};

use crate::octree::{Cube, MapEntity, OctreeWorld, EDGES_EMPTY, EDGES_SOLID};

// ── Version thresholds ──────────────────────────────────────────────────────────

const VER_OLD_GEOM:      i32 = 6;   // v≤6: 3-byte skip + 1-byte textures
const VER_MATERIAL_MASK: i32 = 7;   // v7–v31: mask byte after textures; 0x80=material,0x3F=surfs,0x40=norms
const VER_FLOAT_ENTS:    i32 = 14;  // v14+: entity positions stored as floats
const VER_2BYTE_TEX:     i32 = 14;  // v14+: texture ids are u16 (not u8)
const VER_GAMETYPE:      i32 = 16;  // v16+: gametype/extras block
const VER_MERGED_OLD:    i32 = 20;  // v20–v31: merged in octsav&0x80 (old cube path)
const VER_NEW_CUBE_FMT:  i32 = 32;  // v32+: new cube format (octsav flags, u16 material v33+)
const VER_USHORT_MAT:    i32 = 33;  // v33+: material stored as u16
const VER_NEW_HEADER:    i32 = 29;  // v29+: new-style header with blendmap/numvars/numvslots
const VER_NUMVSLOTS_HDR: i32 = 30;  // v30+: numvslots in header (v29 has it zero)
const VER_VSLOTS:        i32 = 29;  // v29+: vslot table in data stream

// ── compatheader size ──────────────────────────────────────────────────────────
// sizeof(compatheader) = 196; first 28 bytes already read as 7 ints.
const COMPAT_HEADER_TAIL: u64 = 196 - 28;

// ── Error type ─────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum OgzError {
    Io(std::io::Error),
    BadMagic([u8; 4]),
    UnsupportedVersion(i32),
    Truncated,
}

impl std::fmt::Display for OgzError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e)                 => write!(f, "IO error: {e}"),
            Self::BadMagic(m)           => write!(f, "Bad magic: {m:?}"),
            Self::UnsupportedVersion(v) => write!(f, "Unsupported OGZ version {v}"),
            Self::Truncated             => write!(f, "OGZ file truncated"),
        }
    }
}

impl From<std::io::Error> for OgzError { fn from(e: std::io::Error) -> Self { Self::Io(e) } }

// ── Public entry points ─────────────────────────────────────────────────────────

pub fn load_ogz(path: &str) -> Result<OctreeWorld, OgzError> {
    let compressed = std::fs::read(path)?;
    load_ogz_bytes(&compressed)
}

pub fn load_ogz_bytes(compressed: &[u8]) -> Result<OctreeWorld, OgzError> {
    let mut decoder = GzDecoder::new(compressed);
    let mut raw = Vec::new();
    decoder.read_to_end(&mut raw).map_err(OgzError::Io)?;
    info!("OGZ decompressed: {} bytes raw", raw.len());
    parse_ogz(&mut Cursor::new(raw))
}

// ── Save ────────────────────────────────────────────────────────────────────

pub fn save_ogz(path: &str, world: &OctreeWorld) -> Result<(), OgzError> {
    let raw = save_ogz_bytes(world)?;
    std::fs::write(path, &raw)?;
    Ok(())
}

pub fn save_ogz_bytes(world: &OctreeWorld) -> Result<Vec<u8>, OgzError> {
    let mut buf: Vec<u8> = Vec::new();

    // ── Header (40 bytes) ────────────────────────────────────────────────
    buf.extend_from_slice(b"OCTA");
    buf.extend_from_slice(&33i32.to_le_bytes());                    // version
    buf.extend_from_slice(&40i32.to_le_bytes());                    // headersize
    buf.extend_from_slice(&world.world_size().to_le_bytes());       // worldsize
    buf.extend_from_slice(&(world.entities.len() as i32).to_le_bytes()); // numents
    buf.extend_from_slice(&0i32.to_le_bytes());                     // numpvs
    buf.extend_from_slice(&0i32.to_le_bytes());                     // lightmaps
    buf.extend_from_slice(&0i32.to_le_bytes());                     // blendmap
    buf.extend_from_slice(&0i32.to_le_bytes());                     // numvars
    buf.extend_from_slice(&0i32.to_le_bytes());                     // numvslots

    // ── Gametype (v16+): empty string ────────────────────────────────────
    buf.push(0u8);                    // gametype string length = 0
    buf.push(0u8);                    // null terminator
    buf.extend_from_slice(&0u16.to_le_bytes()); // eif
    buf.extend_from_slice(&0u16.to_le_bytes()); // extrasize

    // ── TexMRU: none ─────────────────────────────────────────────────────
    buf.extend_from_slice(&0u16.to_le_bytes()); // nummru = 0

    // ── Entities ─────────────────────────────────────────────────────────
    for ent in &world.entities {
        buf.extend_from_slice(&ent.pos[0].to_le_bytes());
        buf.extend_from_slice(&ent.pos[1].to_le_bytes());
        buf.extend_from_slice(&ent.pos[2].to_le_bytes());
        for i in 0..5 {
            buf.extend_from_slice(&ent.attr[i].to_le_bytes());
        }
        buf.push(ent.etype);
        buf.push(0u8); // reserved
    }

    // ── VSlots: none (numvslots=0) ───────────────────────────────────────

    // ── Octree: 8 root cubes ─────────────────────────────────────────────
    for cube in world.root.iter() {
        write_cube(&mut buf, cube);
    }

    // ── Gzip compress ────────────────────────────────────────────────────
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&buf).map_err(OgzError::Io)?;
    let compressed = encoder.finish().map_err(OgzError::Io)?;

    info!("OGZ save: {} bytes raw, {} bytes compressed", buf.len(), compressed.len());
    Ok(compressed)
}

fn write_cube(buf: &mut Vec<u8>, cube: &Cube) {
    if let Some(children) = &cube.children {
        buf.push(0u8); // OCTSAV_CHILDREN
        for child in children.iter() {
            write_cube(buf, child);
        }
        return;
    }

    // Leaf node: determine type and flags
    let base_type: u8;
    let write_edges: bool;

    if cube.edges == EDGES_EMPTY {
        base_type = 1; // OCTSAV_EMPTY
        write_edges = false;
    } else if cube.edges == EDGES_SOLID {
        base_type = 2; // OCTSAV_SOLID
        write_edges = false;
    } else {
        base_type = 3; // OCTSAV_NORMAL
        write_edges = true;
    }

    // Compute octsav byte with flags OR'd in
    let mut octsav = base_type;
    if cube.material != 0 { octsav |= 0x40; }
    if cube.merged   != 0 { octsav |= 0x80; }

    buf.push(octsav);

    // Edge data for NORMAL cubes
    if write_edges {
        buf.extend_from_slice(&cube.edges);
    }

    // 6 texture indices (u16 each)
    for i in 0..6 {
        buf.extend_from_slice(&cube.texture[i].to_le_bytes());
    }

    // Material (v33: u16)
    if cube.material != 0 {
        buf.extend_from_slice(&(cube.material as u16).to_le_bytes());
    }

    // Merged
    if cube.merged != 0 {
        buf.push(cube.merged);
    }
}

// ── Main parser ─────────────────────────────────────────────────────────────────

fn parse_ogz(cur: &mut Cursor<Vec<u8>>) -> Result<OctreeWorld, OgzError> {

    // ── First 7 i32s (same layout for ALL versions) ──────────────────────────
    let mut magic = [0u8; 4];
    cur.read_exact(&mut magic)?;
    if &magic != b"OCTA" { return Err(OgzError::BadMagic(magic)); }

    let version    = read_i32(cur)?;
    let headersize = read_i32(cur)?;  // sizeof(octaheader); typically 40 for v29–v33
    let worldsize  = read_i32(cur)?;  // actual world size, e.g. 1024
    let numents    = read_i32(cur)?;
    let _numpvs    = read_i32(cur)?;
    let _lightmaps = read_i32(cur)?;
    // 28 bytes consumed so far (magic=4 + 6 ints)

    if version < 1 { return Err(OgzError::UnsupportedVersion(version)); }
    info!("OGZ version={version} headersize={headersize} worldsize={worldsize} numents={numents}");

    // ── Remaining header fields ───────────────────────────────────────────────
    let numvars:   i32;
    let numvslots: i32;

    if version >= VER_NEW_HEADER {
        // v29+: blendmap(i32), numvars(i32), [numvslots(i32) v30+]
        let _blendmap = read_i32(cur)?;
        numvars       = read_i32(cur)?;
        numvslots     = if version >= VER_NUMVSLOTS_HDR { read_i32(cur)? } else { 0 };
        // Skip any extra header bytes (headersize may exceed what we read)
        let bytes_consumed: i64 = 4 + 6*4 + if version >= VER_NUMVSLOTS_HDR { 3*4 } else { 2*4 };
        let skip = (headersize as i64 - bytes_consumed).max(0);
        if skip > 0 { skip_bytes(cur, skip as u64)?; }
    } else {
        // v1–v28: compatheader — skip the tail we don't need
        skip_bytes(cur, COMPAT_HEADER_TAIL)?;
        numvars   = 0;
        numvslots = 0;
    }

    // ── world_scale ───────────────────────────────────────────────────────────
    let mut world_scale = 0u32;
    while (1i32 << world_scale) < worldsize { world_scale += 1; }
    if world_scale < 1 || world_scale > 16 {
        warn!("worldsize={worldsize} → world_scale={world_scale} clamped");
        world_scale = world_scale.clamp(1, 16);
    }

    // ── Variables (numvars from header) ───────────────────────────────────────
    for _ in 0..numvars {
        let vtype  = read_u8(cur)?;
        let namelen = read_u16(cur)?;
        skip_bytes(cur, namelen as u64)?;
        match vtype {
            0 => { read_i32(cur)?; }                   // ID_VAR
            1 => { read_f32(cur)?; }                   // ID_FVAR
            2 => { let sl = read_u16(cur)?; skip_bytes(cur, sl as u64)?; } // ID_SVAR
            _ => { warn!("Unknown var type {vtype}"); }
        }
    }

    // ── Gametype + extras (v16+) ──────────────────────────────────────────────
    if version >= VER_GAMETYPE {
        let len = read_u8(cur)? as u64;
        skip_bytes(cur, len + 1)?;          // gametype string + null terminator
        let _eif      = read_u16(cur)?;
        let extrasize = read_u16(cur)? as u64;
        skip_bytes(cur, extrasize)?;
    }

    // ── Texture MRU ──────────────────────────────────────────────────────────
    if version < VER_2BYTE_TEX {
        skip_bytes(cur, 256)?;              // old 256-byte MRU list
    } else {
        let nummru = read_u16(cur)? as u64;
        skip_bytes(cur, nummru * 2)?;       // u16 per entry
    }

    // ── Entities ──────────────────────────────────────────────────────────────
    let entities = read_entities(cur, numents, version)?;

    // ── vslots (after entities) ───────────────────────────────────────────────
    if version >= VER_VSLOTS && numvslots > 0 {
        skip_vslots(cur, numvslots)?;
    }

    // ── Octree ────────────────────────────────────────────────────────────────
    let mut root_cubes: [Cube; 8] = Default::default();
    for cube in root_cubes.iter_mut() {
        *cube = read_cube(cur, version)?;
    }

    info!("OGZ load OK: {} entities, world_scale={world_scale}", entities.len());

    Ok(OctreeWorld {
        root:        Box::new(root_cubes),
        world_scale: world_scale as u32,
        entities,
        ogz_version: version,
    })
}

// ── Entity reading ─────────────────────────────────────────────────────────────

fn read_entities(cur: &mut Cursor<Vec<u8>>, count: i32, version: i32)
    -> Result<Vec<MapEntity>, OgzError>
{
    let mut ents = Vec::with_capacity(count.max(0) as usize);
    for _ in 0..count {
        let ent = if version >= VER_FLOAT_ENTS {
            let x  = read_f32(cur)?; let y = read_f32(cur)?; let z = read_f32(cur)?;
            let a1 = read_i16(cur)?; let a2 = read_i16(cur)?; let a3 = read_i16(cur)?;
            let a4 = read_i16(cur)?; let a5 = read_i16(cur)?;
            let et = read_u8(cur)?;  let _r = read_u8(cur)?;
            MapEntity { pos: [x,y,z], etype: fix_entity_type(et, version), attr: [a1,a2,a3,a4,a5] }
        } else {
            let x  = read_i32(cur)? as f32; let y = read_i32(cur)? as f32; let z = read_i32(cur)? as f32;
            let a1 = read_i16(cur)?; let a2 = read_i16(cur)?;
            let et = read_u8(cur)?;
            MapEntity { pos: [x,y,z], etype: fix_entity_type(et, version), attr: [a1,a2,0,0,0] }
        };
        ents.push(ent);
    }
    Ok(ents)
}

fn fix_entity_type(t: u8, version: i32) -> u8 {
    let mut t = t;
    if version <= 10 && t >= 7  { t += 1; }
    if version <= 12 && t >= 8  { t += 1; }
    if version <= 14 && t == 16 { t = 8; }
    else if version <= 14 && t >= 8 { t += 1; }
    t
}

// ── vslot table ───────────────────────────────────────────────────────────────
//
// Format (from worldio.cpp loadvslot/loadvslots):
//   while numvslots > 0:
//     changed (i32)  — if < 0: run of -changed unchanged slots (no data)
//                    — if ≥ 0: bitmask of VSLOT_* flags
//     if changed ≥ 0:
//       prev  (i32)
//       VSLOT_SHPARAM (1<<0): u16 numparams, each: u16 namelen, namelen bytes, 4×f32
//       VSLOT_SCALE   (1<<1): f32
//       VSLOT_ROTATION(1<<2): i32
//       VSLOT_OFFSET  (1<<3): 2×i32
//       VSLOT_SCROLL  (1<<4): 2×f32
//       VSLOT_LAYER   (1<<5): i32
//       VSLOT_ALPHA   (1<<6): 2×f32
//       VSLOT_COLOR   (1<<7): 3×f32

fn skip_vslots(cur: &mut Cursor<Vec<u8>>, mut numvslots: i32) -> Result<(), OgzError> {
    if numvslots < 0 || numvslots > 262144 {
        warn!("Suspicious numvslots={numvslots}, skipping");
        return Ok(());
    }
    while numvslots > 0 {
        let changed = read_i32(cur)?;
        if changed < 0 {
            numvslots += changed;   // skip a run of unchanged slots (no data)
            continue;
        }
        let _prev = read_i32(cur)?; // chain index

        // VSLOT_SHPARAM = 1<<0
        if changed & (1 << 0) != 0 {
            let numparams = read_u16(cur)? as u64;
            for _ in 0..numparams {
                let nlen = read_u16(cur)? as u64;
                skip_bytes(cur, nlen)?;      // name bytes (no null)
                skip_bytes(cur, 4 * 4)?;     // 4 × f32 vals
            }
        }
        if changed & (1 << 1) != 0 { read_f32(cur)?; }          // VSLOT_SCALE
        if changed & (1 << 2) != 0 { read_i32(cur)?; }          // VSLOT_ROTATION
        if changed & (1 << 3) != 0 { skip_bytes(cur, 8)?; }     // VSLOT_OFFSET: 2×i32
        if changed & (1 << 4) != 0 { skip_bytes(cur, 8)?; }     // VSLOT_SCROLL: 2×f32
        if changed & (1 << 5) != 0 { read_i32(cur)?; }          // VSLOT_LAYER
        if changed & (1 << 6) != 0 { skip_bytes(cur, 8)?; }     // VSLOT_ALPHA: 2×f32
        if changed & (1 << 7) != 0 { skip_bytes(cur, 12)?; }    // VSLOT_COLOR: 3×f32

        numvslots -= 1;
    }
    Ok(())
}

// ── Octree reading ─────────────────────────────────────────────────────────────

fn read_cube(cur: &mut Cursor<Vec<u8>>, version: i32) -> Result<Cube, OgzError> {
    let octsav = read_u8(cur)?;
    let node_type = octsav & 0x07;

    // CHILDREN: no leaf data, just recurse
    if node_type == 0 {
        let mut children: [Cube; 8] = Default::default();
        for c in children.iter_mut() { *c = read_cube(cur, version)?; }
        let mut cube = Cube::empty();
        cube.children = Some(Box::new(children));
        return Ok(cube);
    }

    // Leaf: build geometry
    let mut cube = match node_type {
        1 => Cube { edges: EDGES_EMPTY, ..Default::default() },  // EMPTY
        2 => Cube { edges: EDGES_SOLID, ..Default::default() },  // SOLID
        3 => {                                                     // NORMAL
            let mut c = Cube { edges: EDGES_EMPTY, ..Default::default() };
            cur.read_exact(&mut c.edges)?;
            c
        }
        4 => Cube { edges: EDGES_SOLID, ..Default::default() },  // LODCUBE (treat as solid)
        _ => {
            warn!("Unknown OCTSAV type {node_type:#04x} @ {}", cur.position());
            Cube::empty()
        }
    };

    // ── Textures ─────────────────────────────────────────────────────────────
    for i in 0..6 {
        cube.texture[i] = if version < VER_2BYTE_TEX {
            read_u8(cur)? as u16
        } else {
            read_u16(cur)?
        };
    }

    // ── Post-texture data: two formats ──────────────────────────────────────

    if version < VER_MATERIAL_MASK + 1 {
        // v1–v6: skip 3 bytes (old UV/light data, from Sauerbraten source `f->seek(3)`)
        skip_bytes(cur, 3)?;
    } else if version <= 31 {
        // v7–v31: mask byte format
        let mask = read_u8(cur)?;

        // Material (bit 0x80 in mask)
        if mask & 0x80 != 0 {
            let _mat = read_u8(cur)?;
            // We don't convert old materials — just note it
        }

        // Surface + normals data (bits 0x3F are per-face, bit 0x40 = has normals per-face)
        if mask & 0x3F != 0 {
            skip_old_surfaces_v7_v31(cur, mask)?;
        }

        // Merged (v20–v31): bit 0x80 of octsav
        if version >= VER_MERGED_OLD && (octsav & 0x80) != 0 {
            let merged_byte = read_u8(cur)?;
            cube.merged = merged_byte & 0x3F;
            // If bit 0x80 of merged_byte is set, there are also mergecompat entries
            if merged_byte & 0x80 != 0 {
                let mmask = read_u8(cur)?;
                for i in 0..6 {
                    if mmask & (1 << i) != 0 {
                        skip_bytes(cur, 8)?; // mergecompat: 4 × u16
                    }
                }
            }
        }
    } else {
        // v32+: octsav flags format
        if octsav & 0x40 != 0 {
            if version < VER_USHORT_MAT {
                cube.material = read_u8(cur)?;   // v32: 1 byte (old material index)
            } else {
                cube.material = (read_u16(cur)? & 0xFF) as u8; // v33+: u16
            }
        }
        if octsav & 0x80 != 0 {
            cube.merged = read_u8(cur)?;
        }
        if octsav & 0x20 != 0 {
            skip_new_surfaces(cur)?;
        }
    }

    // LODCUBE: read children but discard (game uses the LOD cube as the leaf)
    if node_type == 4 {
        let mut tmp: [Cube; 8] = Default::default();
        for c in tmp.iter_mut() { *c = read_cube(cur, version)?; }
        // Discard children — we just render this as solid
    }

    Ok(cube)
}

// ── Surface skip: v7–v31 (mask-byte format) ────────────────────────────────────
//
// mask bits 0–5: which faces have surfacecompat data (16 bytes each)
// mask bit 6 (0x40): those faces also have normalscompat data (12 bytes each)
// Additional layer faces (surfacecompat.layer & 2) add one more surfacecompat.

fn skip_old_surfaces_v7_v31(cur: &mut Cursor<Vec<u8>>, mask: u8) -> Result<(), OgzError> {
    let has_norms = (mask & 0x40) != 0;
    let _num_surfs = 6i32;
    let mut extra_surfs = 0i32;

    for i in 0..6 {
        if mask & (1 << i) != 0 {
            // surfacecompat = uchar texcoords[8], w, h, ushort x, ushort y, lmid, layer = 16 bytes
            let mut surf = [0u8; 16];
            cur.read_exact(&mut surf).map_err(|_| OgzError::Truncated)?;
            let layer = surf[15];
            if layer & 2 != 0 { extra_surfs += 1; }  // LAYER_DUP → extra blend face
            if has_norms {
                skip_bytes(cur, 12)?; // normalscompat: bvec normals[4] = 4×3 bytes
            }
        }
    }
    // Extra blend-layer faces (their surfacecompat, no normals)
    for _ in 0..extra_surfs {
        skip_bytes(cur, 16)?;
    }

    Ok(())
}

// ── Surface skip: v32+ (new format) ───────────────────────────────────────────
//
// surfmask (u8) + totalverts (u8)
// for each face bit set in surfmask:
//   surfaceinfo (4 bytes: lmid[2], verts/vertmask, numverts)
//   then compressed vertex data (variable, see savec in worldio.cpp)

fn skip_new_surfaces(cur: &mut Cursor<Vec<u8>>) -> Result<(), OgzError> {
    let surfmask   = read_u8(cur)?;
    let _totalverts = read_u8(cur)?;

    for i in 0..6u8 {
        if (surfmask >> i) & 1 == 0 { continue; }

        // surfaceinfo: lmid[2], vertmask, numverts_byte
        let _lmid0       = read_u8(cur)?;
        let _lmid1       = read_u8(cur)?;
        let vertmask     = read_u8(cur)?;
        let numverts_byte = read_u8(cur)?;

        // MAXFACEVERTS = 15 (0x0F), LAYER_DUP = 1<<7
        let layerverts = (numverts_byte & 0x0F) as i64;
        let layer_dup  = (numverts_byte & 0x80) != 0;

        if layerverts == 0 { continue; }

        let mut hasxyz  = (vertmask & 0x04) != 0;
        let mut hasuv   = (vertmask & 0x40) != 0;
        let mut hasnorm = (vertmask & 0x80) != 0;
        let mut skip: i64 = 0;

        if layerverts == 4 {
            if hasxyz && (vertmask & 0x01) != 0 {
                skip += 8;   // 4 × u16 (two opposite corners)
                hasxyz = false;
            }
            if hasuv && (vertmask & 0x02) != 0 {
                skip += 8;   // 4 × u16 (two opposite UVs)
                if layer_dup { skip += 8; } // duplicate layer UVs
                hasuv = false;
            }
        }
        if hasnorm && (vertmask & 0x08) != 0 {
            skip += 2;       // 1 × u16 shared normal
            hasnorm = false;
        }
        // Per-vertex data
        for _ in 0..layerverts {
            if hasxyz  { skip += 4; }   // 2 × u16
            if hasuv   { skip += 4; }   // 2 × u16
            if hasnorm { skip += 2; }   // 1 × u16
        }
        // LAYER_DUP duplicate per-vertex UV (using modified hasuv)
        if layer_dup {
            for _ in 0..layerverts {
                if hasuv { skip += 4; }
            }
        }

        if skip > 0 { skip_bytes(cur, skip as u64)?; }
    }
    Ok(())
}

// ── Low-level I/O helpers ──────────────────────────────────────────────────────

#[inline]
fn read_u8(cur: &mut Cursor<Vec<u8>>) -> Result<u8, OgzError> {
    let mut b = [0u8; 1];
    cur.read_exact(&mut b).map_err(|_| OgzError::Truncated)?;
    Ok(b[0])
}
#[inline]
fn read_u16(cur: &mut Cursor<Vec<u8>>) -> Result<u16, OgzError> {
    let mut b = [0u8; 2]; cur.read_exact(&mut b).map_err(|_| OgzError::Truncated)?;
    Ok(u16::from_le_bytes(b))
}
#[inline]
fn read_i16(cur: &mut Cursor<Vec<u8>>) -> Result<i16, OgzError> {
    let mut b = [0u8; 2]; cur.read_exact(&mut b).map_err(|_| OgzError::Truncated)?;
    Ok(i16::from_le_bytes(b))
}
#[inline]
fn read_i32(cur: &mut Cursor<Vec<u8>>) -> Result<i32, OgzError> {
    let mut b = [0u8; 4]; cur.read_exact(&mut b).map_err(|_| OgzError::Truncated)?;
    Ok(i32::from_le_bytes(b))
}
#[inline]
fn read_f32(cur: &mut Cursor<Vec<u8>>) -> Result<f32, OgzError> {
    let mut b = [0u8; 4]; cur.read_exact(&mut b).map_err(|_| OgzError::Truncated)?;
    Ok(f32::from_le_bytes(b))
}
#[inline]
fn skip_bytes(cur: &mut Cursor<Vec<u8>>, n: u64) -> Result<(), OgzError> {
    cur.seek(SeekFrom::Current(n as i64)).map_err(|_| OgzError::Truncated)?;
    Ok(())
}
