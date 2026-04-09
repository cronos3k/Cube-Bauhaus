//! GLB (binary glTF 2.0) exporter for the Cube2 octree world.
//!
//! Exports the entire octree mesh as a single `.glb` file with:
//! - Per-texture-slot materials and embedded texture images
//! - Optimized mesh (deduplicated vertices, no degenerate triangles)
//! - Standard PBR metallic-roughness materials
//!
//! No external glTF crate required -- the binary format is simple enough
//! to emit directly with `std::io::Write`.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use crate::geometry::MeshVertex;
use crate::octree::OctreeWorld;
use crate::texture::{TextureRegistry, TexType};

// ── Mesh optimisation ────────────────────────────────────────────────────────

/// Spatial hash key for vertex deduplication.
/// Quantises position, normal, and UV into integer cells so that vertices
/// within epsilon land in the same bucket.
fn vertex_key(v: &MeshVertex) -> (i64, i64, i64, i32, i32, i32, i32, i32) {
    const POS_QUANT: f64 = 1000.0;  // 1/epsilon for position (0.001)
    const NRM_QUANT: f64 = 100.0;   // 1/epsilon for normal   (0.01)
    const UV_QUANT:  f64 = 1000.0;  // 1/epsilon for UV       (0.001)

    (
        (v.position[0] as f64 * POS_QUANT).round() as i64,
        (v.position[1] as f64 * POS_QUANT).round() as i64,
        (v.position[2] as f64 * POS_QUANT).round() as i64,
        (v.normal[0] as f64 * NRM_QUANT).round() as i32,
        (v.normal[1] as f64 * NRM_QUANT).round() as i32,
        (v.normal[2] as f64 * NRM_QUANT).round() as i32,
        (v.uv[0] as f64 * UV_QUANT).round() as i32,
        (v.uv[1] as f64 * UV_QUANT).round() as i32,
    )
}

/// Optimise a triangle mesh by deduplicating vertices and removing degenerate
/// triangles.
///
/// 1. Builds a spatial hash of vertices and merges any whose position, normal,
///    and UV all fall within a small epsilon (pos < 0.001, normal < 0.01,
///    uv < 0.001).
/// 2. Removes degenerate triangles where two or more indices coincide or the
///    triangle has zero area.
///
/// Returns `(deduplicated_vertices, remapped_indices)`.
pub fn optimize_mesh(verts: &[MeshVertex], indices: &[u32]) -> (Vec<MeshVertex>, Vec<u32>) {
    // --- Pass 1: vertex deduplication via spatial hash ---
    let mut canonical: HashMap<(i64, i64, i64, i32, i32, i32, i32, i32), u32> = HashMap::new();
    let mut new_verts: Vec<MeshVertex> = Vec::new();
    let mut remap: Vec<u32> = Vec::with_capacity(verts.len());

    for v in verts {
        let key = vertex_key(v);
        let idx = canonical.entry(key).or_insert_with(|| {
            let i = new_verts.len() as u32;
            new_verts.push(*v);
            i
        });
        remap.push(*idx);
    }

    // --- Pass 2: remap indices and cull degenerate triangles ---
    let mut new_indices: Vec<u32> = Vec::with_capacity(indices.len());

    let tri_count = indices.len() / 3;
    for t in 0..tri_count {
        let i0 = remap[indices[t * 3]     as usize];
        let i1 = remap[indices[t * 3 + 1] as usize];
        let i2 = remap[indices[t * 3 + 2] as usize];

        // Skip if any two indices are the same (collapsed edge).
        if i0 == i1 || i1 == i2 || i0 == i2 {
            continue;
        }

        // Skip zero-area triangles (cross product magnitude check).
        let p0 = new_verts[i0 as usize].position;
        let p1 = new_verts[i1 as usize].position;
        let p2 = new_verts[i2 as usize].position;

        let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let cx = e1[1] * e2[2] - e1[2] * e2[1];
        let cy = e1[2] * e2[0] - e1[0] * e2[2];
        let cz = e1[0] * e2[1] - e1[1] * e2[0];
        let area_sq = cx * cx + cy * cy + cz * cz;

        if area_sq < 1e-12 {
            continue;
        }

        new_indices.push(i0);
        new_indices.push(i1);
        new_indices.push(i2);
    }

    (new_verts, new_indices)
}

// ── GLB export ───────────────────────────────────────────────────────────────

/// Magic number for glTF binary container.
const GLB_MAGIC: u32 = 0x46546C67; // "glTF"
/// Chunk type for JSON.
const CHUNK_JSON: u32 = 0x4E4F534A; // "JSON"
/// Chunk type for BIN.
const CHUNK_BIN: u32 = 0x004E4942; // "BIN\0"

/// Export the octree world as a binary glTF 2.0 (`.glb`) file.
///
/// - Generates mesh via `build_mesh_with_textures`, then optimises it.
/// - Groups triangles by texture slot and creates one glTF primitive per group.
/// - Embeds referenced texture images (PNG/JPEG) directly in the binary buffer
///   when `texture_base_path` is provided and files can be found on disk.
/// - Materials use `pbrMetallicRoughness` with metallicFactor 0 and
///   roughnessFactor 0.8.
pub fn export_glb(
    path: &str,
    world: &OctreeWorld,
    registry: Option<&TextureRegistry>,
    texture_base_path: Option<&str>,
) -> Result<(), String> {
    // ── 1. Generate and optimise mesh ────────────────────────────────────
    let (raw_verts, raw_indices) = crate::geometry::build_mesh_with_textures(world, registry);
    if raw_verts.is_empty() || raw_indices.is_empty() {
        return Err("World produces an empty mesh — nothing to export.".into());
    }
    let (verts, indices) = optimize_mesh(&raw_verts, &raw_indices);
    if indices.is_empty() {
        return Err("Mesh is fully degenerate after optimisation.".into());
    }

    // ── 2. Group triangles by texture slot ───────────────────────────────
    // The texture slot is encoded in `color.a`:
    //   >= 0.0  → valid layer index (cast to usize)
    //   < 0.0   → material volume (-2=glass, -3=water, -4=lava, -5=clip)
    //             or -1 for untextured debug color
    // We use i32 as the key, preserving distinct material codes.

    struct PrimGroup {
        slot_key: i32,
        indices: Vec<u32>,
    }

    let mut group_map: HashMap<i32, usize> = HashMap::new();
    let mut groups: Vec<PrimGroup> = Vec::new();

    let tri_count = indices.len() / 3;
    for t in 0..tri_count {
        let vi = indices[t * 3] as usize;
        let layer = verts[vi].color[3];
        let key = if layer >= 0.0 { layer.round() as i32 } else { layer.round() as i32 };

        let g = group_map.entry(key).or_insert_with(|| {
            let idx = groups.len();
            groups.push(PrimGroup { slot_key: key, indices: Vec::new() });
            idx
        });
        groups[*g].indices.push(indices[t * 3]);
        groups[*g].indices.push(indices[t * 3 + 1]);
        groups[*g].indices.push(indices[t * 3 + 2]);
    }

    // ── 2b. Remove sky-slot geometry (slot 0 / layer 0) ────────────────
    groups.retain(|g| g.slot_key != 0);

    // ── 2c. Separate material volumes into their own files ─────────────
    let mut mat_groups: Vec<PrimGroup> = Vec::new();
    let mut solid_groups: Vec<PrimGroup> = Vec::new();
    for g in groups {
        if g.slot_key <= -2 {
            mat_groups.push(g);
        } else {
            solid_groups.push(g);
        }
    }
    let groups = solid_groups;
    if !mat_groups.is_empty() {
        let base = std::path::Path::new(path);
        let stem = base.file_stem().and_then(|s| s.to_str()).unwrap_or("export");
        let ext = base.extension().and_then(|s| s.to_str()).unwrap_or("glb");
        let dir = base.parent().unwrap_or(std::path::Path::new("."));
        for mg in &mat_groups {
            let mat_name = match mg.slot_key {
                -2 => "glass",
                -3 => "water",
                -4 => "lava",
                -5 => "clip",
                _  => "material",
            };
            let mat_path = dir.join(format!("{}_{}.{}", stem, mat_name, ext));
            eprintln!("  Material volume: {} → {} ({} tris)",
                mat_name, mat_path.display(), mg.indices.len() / 3);
            let mat_color: [f32; 4] = match mg.slot_key {
                -2 => [0.6, 0.8, 1.0, 0.5],  // glass: light blue, semi-transparent
                -3 => [0.2, 0.4, 0.9, 0.5],  // water: blue
                -4 => [1.0, 0.4, 0.1, 0.8],  // lava: orange
                -5 => [1.0, 0.2, 0.2, 0.3],  // clip: red, mostly transparent
                _  => [0.5, 0.5, 0.5, 0.5],
            };
            if let Err(e) = write_material_volume_glb(
                &mat_path.to_string_lossy(), &verts, &mg.indices, mat_color
            ) {
                eprintln!("  Warning: failed to write {}: {}", mat_name, e);
            }
        }
    }

    if groups.is_empty() {
        return Err("All geometry is sky or material — nothing to export.".into());
    }

    // ── 3. Collect texture image data ────────────────────────────────────

    struct EmbeddedImage {
        data: Vec<u8>,
        mime: &'static str,
    }

    // Maps slot_key -> (material_index, optional image index)
    // We will build these as we iterate groups.

    let mut images: Vec<EmbeddedImage> = Vec::new();
    // slot_key -> image index in `images`
    let mut slot_image_map: HashMap<i32, usize> = HashMap::new();

    if let (Some(reg), Some(base)) = (registry, texture_base_path) {
        for g in &groups {
            if g.slot_key < 0 { continue; }
            if slot_image_map.contains_key(&g.slot_key) { continue; }

            // The slot_key is a layer index assigned by the renderer.  To find
            // the corresponding Slot we need to search for a SlotTex whose
            // `layer` field matches.  Alternatively, the more reliable path is
            // to look at the vslot -> slot chain.  Since `color.a` is set from
            // `slot.textures[0].layer`, we scan slots for a matching layer.
            // But the layer is a GPU index that may not be set before rendering.
            //
            // A simpler and more robust approach: walk the *verts* to find which
            // vslot index (cube.texture[orient]) produced each layer value.
            // Since we already lost that mapping, we do a best-effort scan of
            // all slots' diffuse textures.

            'slot_search: for slot in &reg.slots {
                if let Some(dtex) = slot.textures.iter().find(|t| t.tex_type == TexType::Diffuse) {
                    if dtex.layer as i32 == g.slot_key && !dtex.path.is_empty() {
                        // Try to load the file.
                        let tex_path = Path::new(base).join(&dtex.path);
                        if let Ok(data) = std::fs::read(&tex_path) {
                            let mime = guess_image_mime(&dtex.path);
                            let idx = images.len();
                            images.push(EmbeddedImage { data, mime });
                            slot_image_map.insert(g.slot_key, idx);
                            break 'slot_search;
                        }
                        // Also try with common alternative extensions.
                        for alt in try_alternative_paths(&tex_path) {
                            if let Ok(data) = std::fs::read(&alt) {
                                let mime = guess_image_mime(alt.to_str().unwrap_or(""));
                                let idx = images.len();
                                images.push(EmbeddedImage { data, mime });
                                slot_image_map.insert(g.slot_key, idx);
                                break 'slot_search;
                            }
                        }
                    }
                }
            }
        }
    }

    // ── 4. Build binary buffer ───────────────────────────────────────────

    let vert_count = verts.len();
    let idx_count  = indices.len();

    // Positions: vec3<f32>, 12 bytes each
    let pos_byte_len  = vert_count * 12;
    // Normals: vec3<f32>, 12 bytes each
    let nrm_byte_len  = vert_count * 12;
    // UVs: vec2<f32>, 8 bytes each
    let uv_byte_len   = vert_count * 8;
    // Indices: u32, 4 bytes each
    let idx_byte_len  = idx_count * 4;

    // Each section must start at a 4-byte aligned offset.
    // Since all component sizes are multiples of 4, offsets are naturally aligned.

    let pos_offset = 0usize;
    let nrm_offset = pos_offset + pos_byte_len;
    let uv_offset  = nrm_offset + nrm_byte_len;
    let idx_offset = uv_offset + uv_byte_len;
    let mut img_data_offset = idx_offset + idx_byte_len;

    // Compute image offsets (each must be 4-byte aligned).
    let mut image_offsets: Vec<usize> = Vec::new();
    for img in &images {
        img_data_offset = align4(img_data_offset);
        image_offsets.push(img_data_offset);
        img_data_offset += img.data.len();
    }

    let total_bin_len = align4(img_data_offset);

    // Write the binary buffer into a Vec<u8>.
    let mut bin = vec![0u8; total_bin_len];

    // Positions
    for (i, v) in verts.iter().enumerate() {
        let off = pos_offset + i * 12;
        bin[off..off + 4].copy_from_slice(&v.position[0].to_le_bytes());
        bin[off + 4..off + 8].copy_from_slice(&v.position[1].to_le_bytes());
        bin[off + 8..off + 12].copy_from_slice(&v.position[2].to_le_bytes());
    }

    // Normals
    for (i, v) in verts.iter().enumerate() {
        let off = nrm_offset + i * 12;
        bin[off..off + 4].copy_from_slice(&v.normal[0].to_le_bytes());
        bin[off + 4..off + 8].copy_from_slice(&v.normal[1].to_le_bytes());
        bin[off + 8..off + 12].copy_from_slice(&v.normal[2].to_le_bytes());
    }

    // UVs
    for (i, v) in verts.iter().enumerate() {
        let off = uv_offset + i * 8;
        bin[off..off + 4].copy_from_slice(&v.uv[0].to_le_bytes());
        bin[off + 4..off + 8].copy_from_slice(&v.uv[1].to_le_bytes());
    }

    // Indices
    for (i, &idx) in indices.iter().enumerate() {
        let off = idx_offset + i * 4;
        bin[off..off + 4].copy_from_slice(&idx.to_le_bytes());
    }

    // Embedded images
    for (i, img) in images.iter().enumerate() {
        let off = image_offsets[i];
        bin[off..off + img.data.len()].copy_from_slice(&img.data);
    }

    // ── 5. Compute bounding box for POSITION accessor ────────────────────

    let mut pos_min = [f32::MAX; 3];
    let mut pos_max = [f32::MIN; 3];
    for v in &verts {
        for k in 0..3 {
            if v.position[k] < pos_min[k] { pos_min[k] = v.position[k]; }
            if v.position[k] > pos_max[k] { pos_max[k] = v.position[k]; }
        }
    }

    // ── 6. Build glTF JSON ───────────────────────────────────────────────

    // Buffer views:
    //   0 = positions
    //   1 = normals
    //   2 = UVs
    //   3 = indices
    //   4.. = images

    let mut buffer_views = Vec::new();
    // 0: positions
    buffer_views.push(format!(
        r#"{{"buffer":0,"byteOffset":{},"byteLength":{},"target":34962}}"#,
        pos_offset, pos_byte_len
    ));
    // 1: normals
    buffer_views.push(format!(
        r#"{{"buffer":0,"byteOffset":{},"byteLength":{},"target":34962}}"#,
        nrm_offset, nrm_byte_len
    ));
    // 2: UVs
    buffer_views.push(format!(
        r#"{{"buffer":0,"byteOffset":{},"byteLength":{},"target":34962}}"#,
        uv_offset, uv_byte_len
    ));
    // 3: indices
    buffer_views.push(format!(
        r#"{{"buffer":0,"byteOffset":{},"byteLength":{},"target":34963}}"#,
        idx_offset, idx_byte_len
    ));
    // 4+: images
    for (i, img) in images.iter().enumerate() {
        buffer_views.push(format!(
            r#"{{"buffer":0,"byteOffset":{},"byteLength":{}}}"#,
            image_offsets[i], img.data.len()
        ));
    }

    // Accessors:
    //   0 = POSITION (whole mesh)
    //   1 = NORMAL   (whole mesh)
    //   2 = TEXCOORD_0 (whole mesh)
    //   3 + i = indices for each primitive group

    let mut accessors = Vec::new();

    // 0: POSITION
    accessors.push(format!(
        r#"{{"bufferView":0,"componentType":5126,"count":{},"type":"VEC3","min":[{},{},{}],"max":[{},{},{}]}}"#,
        vert_count,
        format_f32(pos_min[0]), format_f32(pos_min[1]), format_f32(pos_min[2]),
        format_f32(pos_max[0]), format_f32(pos_max[1]), format_f32(pos_max[2]),
    ));
    // 1: NORMAL
    accessors.push(format!(
        r#"{{"bufferView":1,"componentType":5126,"count":{},"type":"VEC3"}}"#,
        vert_count,
    ));
    // 2: TEXCOORD_0
    accessors.push(format!(
        r#"{{"bufferView":2,"componentType":5126,"count":{},"type":"VEC2"}}"#,
        vert_count,
    ));

    // Per-group index accessors (accessor index = 3 + group_index)
    // Each group's indices are a sub-range of the global index buffer.
    // We need to compute the byte offset and min/max for each group.

    // First, build a flat index buffer per group that references the global
    // vertex array.  The group indices already do that; we just need the
    // byte offset into the global index bufferView.
    //
    // Since all groups' indices are concatenated into the single `indices`
    // array, we track the running offset.

    // Actually, the groups hold their own index lists which are subsets of
    // the global `indices`.  We need to write them in order into the binary
    // buffer.  But we already wrote the full `indices` array.  The groups'
    // index lists are subsets, but they may not be contiguous in the global
    // array because we built them by scanning triangles.
    //
    // We need to re-lay-out the index buffer so that each group's indices
    // are contiguous.  Let's rebuild the index portion of the binary buffer.

    // Rebuild index buffer in group order.
    {
        let mut off = idx_offset;
        for g in &groups {
            for &idx in &g.indices {
                bin[off..off + 4].copy_from_slice(&idx.to_le_bytes());
                off += 4;
            }
        }
    }

    // Now compute per-group accessor entries.
    let mut running_idx_offset = 0usize; // in elements, not bytes
    for g in &groups {
        let count = g.indices.len();
        let byte_off = idx_offset + running_idx_offset * 4;

        // Compute min/max index value for this group (required by some validators).
        let mut imin = u32::MAX;
        let mut imax = 0u32;
        for &idx in &g.indices {
            if idx < imin { imin = idx; }
            if idx > imax { imax = idx; }
        }

        accessors.push(format!(
            r#"{{"bufferView":3,"byteOffset":{},"componentType":5125,"count":{},"type":"SCALAR","min":[{}],"max":[{}]}}"#,
            byte_off - idx_offset,
            count,
            imin,
            imax,
        ));

        running_idx_offset += count;
    }

    // ── Materials, textures, images (glTF entries) ───────────────────────

    let mut materials_json: Vec<String> = Vec::new();
    let mut textures_json: Vec<String> = Vec::new();
    let mut images_json: Vec<String> = Vec::new();
    let has_sampler = !images.is_empty();

    for (gi, g) in groups.iter().enumerate() {
        if let Some(&img_idx) = slot_image_map.get(&g.slot_key) {
            // Material with texture.
            let tex_idx = textures_json.len();
            let img_json_idx = images_json.len();

            // bufferView for this image = 4 + img_idx
            let bv_idx = 4 + img_idx;

            images_json.push(format!(
                r#"{{"bufferView":{},"mimeType":"{}"}}"#,
                bv_idx, images[img_idx].mime
            ));

            textures_json.push(format!(
                r#"{{"sampler":0,"source":{}}}"#,
                img_json_idx
            ));

            // Use vslot color_scale as base color factor if available.
            let vi = groups[gi].indices.first().copied().unwrap_or(0) as usize;
            let col = &verts[vi].color;
            let r = col[0].min(1.0);
            let gc = col[1].min(1.0);
            let b = col[2].min(1.0);

            materials_json.push(format!(
                r#"{{"pbrMetallicRoughness":{{"baseColorTexture":{{"index":{}}},"baseColorFactor":[{},{},{},1.0],"metallicFactor":0.0,"roughnessFactor":0.8}}}}"#,
                tex_idx,
                format_f32(r), format_f32(gc), format_f32(b),
            ));
        } else {
            // Material without texture — solid colour.
            let vi = g.indices.first().copied().unwrap_or(0) as usize;
            let col = &verts[vi].color;
            let r = col[0].min(1.0).max(0.0);
            let gc = col[1].min(1.0).max(0.0);
            let b = col[2].min(1.0).max(0.0);

            materials_json.push(format!(
                r#"{{"pbrMetallicRoughness":{{"baseColorFactor":[{},{},{},1.0],"metallicFactor":0.0,"roughnessFactor":0.8}}}}"#,
                format_f32(r), format_f32(gc), format_f32(b),
            ));
        }
    }

    // ── Primitives ───────────────────────────────────────────────────────

    let mut primitives: Vec<String> = Vec::new();
    for (gi, _g) in groups.iter().enumerate() {
        let idx_accessor = 3 + gi;
        primitives.push(format!(
            r#"{{"attributes":{{"POSITION":0,"NORMAL":1,"TEXCOORD_0":2}},"indices":{},"material":{}}}"#,
            idx_accessor, gi,
        ));
    }

    // ── Assemble top-level JSON ──────────────────────────────────────────

    let samplers_json = if has_sampler {
        r#","samplers":[{"magFilter":9729,"minFilter":9987,"wrapS":10497,"wrapT":10497}]"#
    } else {
        ""
    };

    let textures_section = if !textures_json.is_empty() {
        format!(r#","textures":[{}]"#, textures_json.join(","))
    } else {
        String::new()
    };

    let images_section = if !images_json.is_empty() {
        format!(r#","images":[{}]"#, images_json.join(","))
    } else {
        String::new()
    };

    let json_str = format!(
        r#"{{"asset":{{"version":"2.0","generator":"BBC Cube2 Editor"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],"meshes":[{{"primitives":[{}]}}],"accessors":[{}],"bufferViews":[{}],"buffers":[{{"byteLength":{}}}],"materials":[{}]{}{}{}}}"#,
        primitives.join(","),
        accessors.join(","),
        buffer_views.join(","),
        total_bin_len,
        materials_json.join(","),
        samplers_json,
        textures_section,
        images_section,
    );

    // ── 7. Write GLB ─────────────────────────────────────────────────────

    // JSON chunk must be padded to 4-byte boundary with spaces (0x20).
    let json_bytes = json_str.as_bytes();
    let json_padded_len = align4(json_bytes.len());

    // BIN chunk is already padded (total_bin_len is aligned).
    let bin_padded_len = total_bin_len;

    // GLB total length = 12 (header) + 8 (json chunk header) + json_padded_len
    //                                 + 8 (bin chunk header)  + bin_padded_len
    let total_length = 12 + 8 + json_padded_len + 8 + bin_padded_len;

    let out_path = Path::new(path);
    if let Some(parent) = out_path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create output directory: {}", e))?;
        }
    }

    let file = std::fs::File::create(out_path)
        .map_err(|e| format!("Failed to create GLB file '{}': {}", path, e))?;
    let mut w = std::io::BufWriter::new(file);

    // Header: magic, version, length
    w.write_all(&GLB_MAGIC.to_le_bytes()).map_err(io_err)?;
    w.write_all(&2u32.to_le_bytes()).map_err(io_err)?;
    w.write_all(&(total_length as u32).to_le_bytes()).map_err(io_err)?;

    // JSON chunk
    w.write_all(&(json_padded_len as u32).to_le_bytes()).map_err(io_err)?;
    w.write_all(&CHUNK_JSON.to_le_bytes()).map_err(io_err)?;
    w.write_all(json_bytes).map_err(io_err)?;
    // Pad with spaces
    let json_pad = json_padded_len - json_bytes.len();
    if json_pad > 0 {
        w.write_all(&vec![0x20u8; json_pad]).map_err(io_err)?;
    }

    // BIN chunk
    w.write_all(&(bin_padded_len as u32).to_le_bytes()).map_err(io_err)?;
    w.write_all(&CHUNK_BIN.to_le_bytes()).map_err(io_err)?;
    w.write_all(&bin).map_err(io_err)?;

    w.flush().map_err(io_err)?;

    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Round `n` up to the next multiple of 4.
#[inline]
fn align4(n: usize) -> usize {
    (n + 3) & !3
}

/// Format an f32 for JSON without unnecessary trailing zeros but always with
/// at least one decimal digit so it parses as a float.
fn format_f32(v: f32) -> String {
    if v == v.floor() && v.abs() < 1e15 {
        format!("{:.1}", v)
    } else {
        format!("{}", v)
    }
}

/// Map `std::io::Error` to `String`.
fn io_err(e: std::io::Error) -> String {
    format!("I/O error writing GLB: {}", e)
}

/// Write a single material volume as a minimal GLB file.
///
/// The mesh contains only position and normal attributes with a single flat-color
/// PBR material. Alpha < 1.0 uses `BLEND` mode so the volume appears semi-transparent
/// in standard glTF viewers.
fn write_material_volume_glb(
    path: &str,
    all_verts: &[MeshVertex],
    group_indices: &[u32],
    color: [f32; 4],
) -> Result<(), String> {
    // Collect only the vertices referenced by this group
    let mut remap: HashMap<u32, u32> = HashMap::new();
    let mut verts: Vec<MeshVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    for &idx in group_indices {
        let new_idx = remap.entry(idx).or_insert_with(|| {
            let i = verts.len() as u32;
            verts.push(all_verts[idx as usize]);
            i
        });
        indices.push(*new_idx);
    }

    if verts.is_empty() || indices.is_empty() {
        return Ok(()); // nothing to write
    }

    // Binary buffer: positions (vec3) + normals (vec3) + indices (u32)
    let pos_byte_len = verts.len() * 12;
    let nrm_byte_len = verts.len() * 12;
    let idx_byte_len = indices.len() * 4;
    let total_bin = align4(pos_byte_len) + align4(nrm_byte_len) + align4(idx_byte_len);

    let mut bin = Vec::with_capacity(total_bin);

    // Positions
    let mut pos_min = [f32::MAX; 3];
    let mut pos_max = [f32::MIN; 3];
    for v in &verts {
        for i in 0..3 {
            if v.position[i] < pos_min[i] { pos_min[i] = v.position[i]; }
            if v.position[i] > pos_max[i] { pos_max[i] = v.position[i]; }
        }
        bin.extend_from_slice(&v.position[0].to_le_bytes());
        bin.extend_from_slice(&v.position[1].to_le_bytes());
        bin.extend_from_slice(&v.position[2].to_le_bytes());
    }
    while bin.len() < align4(pos_byte_len) { bin.push(0); }

    let nrm_offset = bin.len();
    // Normals
    for v in &verts {
        bin.extend_from_slice(&v.normal[0].to_le_bytes());
        bin.extend_from_slice(&v.normal[1].to_le_bytes());
        bin.extend_from_slice(&v.normal[2].to_le_bytes());
    }
    while bin.len() < nrm_offset + align4(nrm_byte_len) { bin.push(0); }

    let idx_offset = bin.len();
    // Indices
    let mut idx_min = u32::MAX;
    let mut idx_max = 0u32;
    for &i in &indices {
        if i < idx_min { idx_min = i; }
        if i > idx_max { idx_max = i; }
        bin.extend_from_slice(&i.to_le_bytes());
    }
    while bin.len() < idx_offset + align4(idx_byte_len) { bin.push(0); }

    // Alpha mode
    let alpha_mode = if color[3] < 1.0 { r#","alphaMode":"BLEND""# } else { "" };

    let json_str = format!(
        concat!(
            r#"{{"asset":{{"version":"2.0","generator":"BBC Cube2 Editor"}},"#,
            r#""scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],"#,
            r#""meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1}},"indices":2,"material":0}}]}}],"#,
            r#""accessors":["#,
            r#"{{"bufferView":0,"componentType":5126,"count":{},"type":"VEC3","min":[{},{},{}],"max":[{},{},{}]}},"#,
            r#"{{"bufferView":1,"componentType":5126,"count":{},"type":"VEC3"}},"#,
            r#"{{"bufferView":2,"componentType":5125,"count":{},"type":"SCALAR","min":[{}],"max":[{}]}}"#,
            r#"],"bufferViews":["#,
            r#"{{"buffer":0,"byteOffset":0,"byteLength":{}}},"#,
            r#"{{"buffer":0,"byteOffset":{},"byteLength":{}}},"#,
            r#"{{"buffer":0,"byteOffset":{},"byteLength":{}}}"#,
            r#"],"buffers":[{{"byteLength":{}}}],"#,
            r#""materials":[{{"pbrMetallicRoughness":{{"baseColorFactor":[{},{},{},{}],"metallicFactor":0.0,"roughnessFactor":0.8}}{}}}]}}"#,
        ),
        verts.len(),
        format_f32(pos_min[0]), format_f32(pos_min[1]), format_f32(pos_min[2]),
        format_f32(pos_max[0]), format_f32(pos_max[1]), format_f32(pos_max[2]),
        verts.len(),
        indices.len(), idx_min, idx_max,
        align4(pos_byte_len),
        nrm_offset, align4(nrm_byte_len),
        idx_offset, align4(idx_byte_len),
        total_bin,
        format_f32(color[0]), format_f32(color[1]), format_f32(color[2]), format_f32(color[3]),
        alpha_mode,
    );

    // Write GLB
    let json_bytes = json_str.as_bytes();
    let json_padded = align4(json_bytes.len());
    let total_length = 12 + 8 + json_padded + 8 + total_bin;

    let out_path = Path::new(path);
    if let Some(parent) = out_path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create dir: {}", e))?;
        }
    }

    let file = std::fs::File::create(out_path)
        .map_err(|e| format!("Failed to create '{}': {}", path, e))?;
    let mut w = std::io::BufWriter::new(file);

    w.write_all(&GLB_MAGIC.to_le_bytes()).map_err(io_err)?;
    w.write_all(&2u32.to_le_bytes()).map_err(io_err)?;
    w.write_all(&(total_length as u32).to_le_bytes()).map_err(io_err)?;

    w.write_all(&(json_padded as u32).to_le_bytes()).map_err(io_err)?;
    w.write_all(&CHUNK_JSON.to_le_bytes()).map_err(io_err)?;
    w.write_all(json_bytes).map_err(io_err)?;
    if json_padded > json_bytes.len() {
        w.write_all(&vec![0x20u8; json_padded - json_bytes.len()]).map_err(io_err)?;
    }

    w.write_all(&(total_bin as u32).to_le_bytes()).map_err(io_err)?;
    w.write_all(&CHUNK_BIN.to_le_bytes()).map_err(io_err)?;
    w.write_all(&bin).map_err(io_err)?;

    w.flush().map_err(io_err)?;

    eprintln!("  ✓ Wrote material volume: {} ({} verts, {} tris)",
        path, verts.len(), indices.len() / 3);
    Ok(())
}

/// Map TexType to a unique channel offset for FBX ID generation.
/// Each channel gets its own million-range to avoid ID collisions.
fn textype_channel_offset(tt: TexType) -> i64 {
    match tt {
        TexType::Diffuse => 0,
        TexType::Normal  => 1,
        TexType::Spec    => 2,
        TexType::Glow    => 3,
        TexType::Alpha   => 4,
        _                => 5,
    }
}

/// Map TexType to a human-readable label for FBX object naming.
fn textype_label(tt: TexType) -> &'static str {
    match tt {
        TexType::Diffuse => "Diffuse",
        TexType::Normal  => "Normal",
        TexType::Spec    => "Specular",
        TexType::Glow    => "Emissive",
        TexType::Alpha   => "Alpha",
        _                => "Other",
    }
}

/// Map TexType to the FBX material property name that Unreal Engine
/// recognizes for auto-material creation on import.
///
/// Key mappings (confirmed from FBX SDK + Unreal source):
///   Diffuse  → "DiffuseColor"       → UE Base Color (auto-connected)
///   Normal   → "NormalMap"           → UE Normal (auto-connected)
///   Spec     → "SpecularFactor"      → UE Specular
///   Glow     → "EmissiveColor"       → UE Emissive
///   Alpha    → "TransparencyFactor"  → UE Opacity
fn textype_to_fbx_property(tt: TexType) -> &'static str {
    match tt {
        TexType::Diffuse => "DiffuseColor",
        TexType::Normal  => "NormalMap",
        TexType::Spec    => "SpecularFactor",
        TexType::Glow    => "EmissiveColor",
        TexType::Alpha   => "TransparencyFactor",
        _                => "DiffuseColor",
    }
}

/// Guess MIME type from a file path extension.
fn guess_image_mime(path: &str) -> &'static str {
    let lower = path.to_ascii_lowercase();
    if lower.ends_with(".png") {
        "image/png"
    } else if lower.ends_with(".jpg") || lower.ends_with(".jpeg") {
        "image/jpeg"
    } else if lower.ends_with(".webp") {
        "image/webp"
    } else {
        // Default to PNG for unknown formats.
        "image/png"
    }
}

/// Try common alternative file extensions for a texture path.
fn try_alternative_paths(original: &Path) -> Vec<std::path::PathBuf> {
    let mut alts = Vec::new();
    if let Some(stem) = original.file_stem() {
        if let Some(parent) = original.parent() {
            let stem = stem.to_string_lossy();
            for ext in &["png", "jpg", "jpeg", "tga", "bmp"] {
                let alt = parent.join(format!("{}.{}", stem, ext));
                if alt != original {
                    alts.push(alt);
                }
            }
        }
    }
    alts
}

// ── FBX ASCII 7.4 export ────────────────────────────────────────────────────

/// Export the octree world as an FBX 7.4 ASCII file.
///
/// - Generates mesh via `build_mesh_with_textures`, then optimises it.
/// - Groups triangles by texture slot; each group becomes a separate Geometry+Model.
/// - Embeds texture file references (relative paths) in Materials.
/// - Compatible with Blender, Unity, Unreal, Godot, and most DCC tools.
///
/// The FBX ASCII format is verbose but universally supported and much simpler
/// to emit than the binary format.  For very large meshes the resulting file
/// can be large; use GLB for more compact output.
/// A box collision primitive for Unreal Engine export.
/// Represents an axis-aligned box with 8 vertices.
struct CollisionBox {
    /// 8 corners of the box in renderer coords (Y-up, already swapped from Cube2 Z-up)
    corners: [[f32; 3]; 8],
}

/// Generate UBX collision boxes from the octree.
/// Each solid non-empty leaf cube becomes one collision box.
/// Returns boxes in renderer coordinate space (Y-up, Y↔Z swapped).
fn generate_collision_boxes(world: &OctreeWorld) -> Vec<CollisionBox> {
    let mut boxes = Vec::new();

    world.for_each_leaf(|cube, (ox, oy, oz), size| {
        // Skip empty cubes
        if cube.is_empty() { return; }
        // Include normal solids (MAT_AIR=0x00), alpha (0x80), and clip (0x20) as collision
        // Skip water (0x04), lava (0x08), glass (0x10) — they're passable
        let m = cube.material;
        if m == 0x04 || m == 0x08 || m == 0x10 {
            return;
        }

        let s = size as f32;
        let fx = ox as f32;
        let fy = oy as f32;
        let fz = oz as f32;

        // For solid cubes (all edges at 0x88), the box is exact: [ox..ox+size] in each dim.
        // For deformed cubes, compute the tight AABB of the actual corner positions.
        if cube.is_solid() {
            // Cube2 coords: (x, y, z) → renderer: (x, z, y) [Y↔Z swap]
            let corners = [
                [fx,     fz,     fy    ],  // 0: (0,0,0)
                [fx + s, fz,     fy    ],  // 1: (1,0,0)
                [fx,     fz,     fy + s],  // 2: (0,1,0)
                [fx + s, fz,     fy + s],  // 3: (1,1,0)
                [fx,     fz + s, fy    ],  // 4: (0,0,1)
                [fx + s, fz + s, fy    ],  // 5: (1,0,1)
                [fx,     fz + s, fy + s],  // 6: (0,1,1)
                [fx + s, fz + s, fy + s],  // 7: (1,1,1)
            ];
            boxes.push(CollisionBox { corners });
        } else {
            // Deformed cube: compute AABB from actual corner positions
            let mut min = [f32::MAX; 3];
            let mut max = [f32::MIN; 3];

            for ci in 0..8 {
                let c = cube.corner(ci);
                // Local coords 0-8 → world space, then Y↔Z swap
                let wx = fx + c[0] as f32 * s / 8.0;
                let wy = fy + c[1] as f32 * s / 8.0;
                let wz = fz + c[2] as f32 * s / 8.0;
                // Renderer coords: (wx, wz, wy)
                let rx = wx;
                let ry = wz;
                let rz = wy;
                if rx < min[0] { min[0] = rx; }
                if ry < min[1] { min[1] = ry; }
                if rz < min[2] { min[2] = rz; }
                if rx > max[0] { max[0] = rx; }
                if ry > max[1] { max[1] = ry; }
                if rz > max[2] { max[2] = rz; }
            }

            // Skip zero-volume boxes
            if (max[0] - min[0]).abs() < 0.001
                || (max[1] - min[1]).abs() < 0.001
                || (max[2] - min[2]).abs() < 0.001 {
                return;
            }

            let corners = [
                [min[0], min[1], min[2]],
                [max[0], min[1], min[2]],
                [min[0], min[1], max[2]],
                [max[0], min[1], max[2]],
                [min[0], max[1], min[2]],
                [max[0], max[1], min[2]],
                [min[0], max[1], max[2]],
                [max[0], max[1], max[2]],
            ];
            boxes.push(CollisionBox { corners });
        }
    });

    boxes
}

pub fn export_fbx(
    path: &str,
    world: &OctreeWorld,
    registry: Option<&TextureRegistry>,
    texture_base_path: Option<&str>,
) -> Result<(), String> {
    export_fbx_inner(path, world, registry, texture_base_path, false)
}

/// Export FBX with optional Unreal Engine collision geometry.
pub fn export_fbx_unreal(
    path: &str,
    world: &OctreeWorld,
    registry: Option<&TextureRegistry>,
    texture_base_path: Option<&str>,
) -> Result<(), String> {
    export_fbx_inner(path, world, registry, texture_base_path, true)
}

fn export_fbx_inner(
    path: &str,
    world: &OctreeWorld,
    registry: Option<&TextureRegistry>,
    texture_base_path: Option<&str>,
    unreal_collision: bool,
) -> Result<(), String> {
    // ── 1. Generate and optimise mesh ────────────────────────────────────
    let (raw_verts, raw_indices) = crate::geometry::build_mesh_with_textures(world, registry);
    if raw_verts.is_empty() || raw_indices.is_empty() {
        return Err("World produces an empty mesh — nothing to export.".into());
    }
    let (verts, indices) = optimize_mesh(&raw_verts, &raw_indices);
    if indices.is_empty() {
        return Err("Mesh is fully degenerate after optimisation.".into());
    }

    // ── 2. Group triangles by texture slot ───────────────────────────────
    struct FbxGroup {
        slot_key: i32,
        indices: Vec<u32>,
    }

    let mut group_map: HashMap<i32, usize> = HashMap::new();
    let mut groups: Vec<FbxGroup> = Vec::new();

    let tri_count = indices.len() / 3;
    for t in 0..tri_count {
        let vi = indices[t * 3] as usize;
        let layer = verts[vi].color[3];
        let key = if layer >= 0.0 { layer.round() as i32 } else { layer.round() as i32 };

        let g = group_map.entry(key).or_insert_with(|| {
            let idx = groups.len();
            groups.push(FbxGroup { slot_key: key, indices: Vec::new() });
            idx
        });
        groups[*g].indices.push(indices[t * 3]);
        groups[*g].indices.push(indices[t * 3 + 1]);
        groups[*g].indices.push(indices[t * 3 + 2]);
    }

    // ── 2b. Remove sky + separate material volumes ──────────────────────
    groups.retain(|g| g.slot_key != 0);  // remove sky

    // Separate material volumes — write as separate GLB files, keep only solid geometry
    let mut solid_groups: Vec<FbxGroup> = Vec::new();
    let base = std::path::Path::new(path);
    let stem_fbx = base.file_stem().and_then(|s| s.to_str()).unwrap_or("export");
    let dir_fbx = base.parent().unwrap_or(std::path::Path::new("."));
    for g in groups {
        if g.slot_key <= -2 {
            let mat_name = match g.slot_key {
                -2 => "glass", -3 => "water", -4 => "lava", -5 => "clip", _ => "material",
            };
            let mat_path = dir_fbx.join(format!("{}_{}.glb", stem_fbx, mat_name));
            eprintln!("  Material volume: {} → {} ({} tris)",
                mat_name, mat_path.display(), g.indices.len() / 3);
            let mat_color: [f32; 4] = match g.slot_key {
                -2 => [0.6, 0.8, 1.0, 0.5],
                -3 => [0.2, 0.4, 0.9, 0.5],
                -4 => [1.0, 0.4, 0.1, 0.8],
                -5 => [1.0, 0.2, 0.2, 0.3],
                _  => [0.5, 0.5, 0.5, 0.5],
            };
            if let Err(e) = write_material_volume_glb(
                &mat_path.to_string_lossy(), &verts, &g.indices, mat_color
            ) {
                eprintln!("  Warning: failed to write {}: {}", mat_name, e);
            }
        } else {
            solid_groups.push(g);
        }
    }
    let groups = solid_groups;

    if groups.is_empty() {
        return Err("All geometry is sky or material — nothing to export.".into());
    }

    // ── 3. Resolve ALL texture paths per group (diffuse, normal, spec, glow, etc.)
    //
    // Maps slot_key → Vec<(TexType, resolved_path)>.
    // Each entry is one texture channel that exists on disk.
    let mut slot_textures: HashMap<i32, Vec<(TexType, String)>> = HashMap::new();
    if let (Some(reg), Some(base)) = (registry, texture_base_path) {
        for g in &groups {
            if g.slot_key < 0 { continue; }
            if slot_textures.contains_key(&g.slot_key) { continue; }

            // Find the Slot whose diffuse layer matches this group key
            for slot in &reg.slots {
                let diffuse_match = slot.textures.iter()
                    .find(|t| t.tex_type == TexType::Diffuse && t.layer as i32 == g.slot_key);
                if diffuse_match.is_none() { continue; }

                let mut resolved: Vec<(TexType, String)> = Vec::new();

                for stex in &slot.textures {
                    if stex.path.is_empty() { continue; }
                    // Skip types that have no FBX equivalent
                    match stex.tex_type {
                        TexType::Unknown | TexType::Decal | TexType::Depth | TexType::Envmap => continue,
                        _ => {}
                    }

                    let tex_path = Path::new(base).join(&stex.path);
                    if tex_path.exists() {
                        resolved.push((stex.tex_type, tex_path.display().to_string()));
                    } else {
                        // Try alternative extensions
                        let mut found = false;
                        for alt in try_alternative_paths(&tex_path) {
                            if alt.exists() {
                                resolved.push((stex.tex_type, alt.display().to_string()));
                                found = true;
                                break;
                            }
                        }
                        if !found {
                            // Still record the path even if not found — Unreal can locate it
                            // if the textures are copied alongside the FBX
                            if stex.tex_type == TexType::Diffuse {
                                // Only insist on diffuse existing
                            } else {
                                // For secondary channels, include path even if missing
                                resolved.push((stex.tex_type, tex_path.display().to_string()));
                            }
                        }
                    }
                }

                if !resolved.is_empty() {
                    slot_textures.insert(g.slot_key, resolved);
                }
                break;
            }
        }
    }

    // Log texture channel stats
    {
        let total_channels: usize = slot_textures.values().map(|v| v.len()).sum();
        let slots_with_normal = slot_textures.values()
            .filter(|v| v.iter().any(|(t, _)| *t == TexType::Normal)).count();
        let slots_with_spec = slot_textures.values()
            .filter(|v| v.iter().any(|(t, _)| *t == TexType::Spec)).count();
        eprintln!("  FBX textures: {} channels across {} slots ({} with normals, {} with spec)",
            total_channels, slot_textures.len(), slots_with_normal, slots_with_spec);
    }

    // ── 4. Generate collision boxes if Unreal mode ────────────────────────
    let collision_boxes = if unreal_collision {
        let boxes = generate_collision_boxes(world);
        eprintln!("  Unreal collision: {} UBX boxes generated from octree leaves", boxes.len());
        boxes
    } else {
        Vec::new()
    };

    // Derive mesh name for Unreal naming convention
    let mesh_stem = Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("export");
    let sm_name = if unreal_collision {
        format!("SM_{}", mesh_stem)
    } else {
        mesh_stem.to_string()
    };

    // ── 5. Write FBX ASCII ───────────────────────────────────────────────
    let out_path = Path::new(path);
    if let Some(parent) = out_path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create output directory: {}", e))?;
        }
    }

    let file = std::fs::File::create(out_path)
        .map_err(|e| format!("Failed to create FBX file '{}': {}", path, e))?;
    let mut w = std::io::BufWriter::new(file);

    // FBX header
    writeln!(w, "; FBX 7.4.0 project file").map_err(io_err)?;
    writeln!(w, "; Exported by BBC Cube2 Editor").map_err(io_err)?;
    if unreal_collision {
        writeln!(w, "; Unreal Engine collision mode: {} UBX boxes", collision_boxes.len()).map_err(io_err)?;
    }
    writeln!(w, "; -------------------------------------------").map_err(io_err)?;
    writeln!(w).map_err(io_err)?;

    // FBXHeaderExtension
    writeln!(w, "FBXHeaderExtension:  {{").map_err(io_err)?;
    writeln!(w, "\tFBXHeaderVersion: 1003").map_err(io_err)?;
    writeln!(w, "\tFBXVersion: 7400").map_err(io_err)?;
    writeln!(w, "\tCreator: \"BBC Cube2 Editor\"").map_err(io_err)?;
    writeln!(w, "}}").map_err(io_err)?;
    writeln!(w).map_err(io_err)?;

    // GlobalSettings
    writeln!(w, "GlobalSettings:  {{").map_err(io_err)?;
    writeln!(w, "\tVersion: 1000").map_err(io_err)?;
    writeln!(w, "\tProperties70:  {{").map_err(io_err)?;
    writeln!(w, "\t\tP: \"UpAxis\", \"int\", \"Integer\", \"\",2").map_err(io_err)?;
    writeln!(w, "\t\tP: \"UpAxisSign\", \"int\", \"Integer\", \"\",1").map_err(io_err)?;
    writeln!(w, "\t\tP: \"FrontAxis\", \"int\", \"Integer\", \"\",1").map_err(io_err)?;
    writeln!(w, "\t\tP: \"FrontAxisSign\", \"int\", \"Integer\", \"\",1").map_err(io_err)?;
    writeln!(w, "\t\tP: \"CoordAxis\", \"int\", \"Integer\", \"\",0").map_err(io_err)?;
    writeln!(w, "\t\tP: \"CoordAxisSign\", \"int\", \"Integer\", \"\",1").map_err(io_err)?;
    writeln!(w, "\t\tP: \"UnitScaleFactor\", \"double\", \"Number\", \"\",1.0").map_err(io_err)?;
    writeln!(w, "\t}}").map_err(io_err)?;
    writeln!(w, "}}").map_err(io_err)?;
    writeln!(w).map_err(io_err)?;

    // Definitions
    let collision_count = collision_boxes.len();
    let model_count = groups.len() + collision_count;  // render groups + collision boxes
    let geom_count = groups.len() + collision_count;
    let mat_count = groups.len();  // collision boxes don't need materials
    // Count total texture objects (each channel = 1 Texture + 1 Video)
    let tex_obj_count: usize = slot_textures.values().map(|v| v.len()).sum();
    let total_obj_count = geom_count + model_count + mat_count + tex_obj_count * 2;

    writeln!(w, "Definitions:  {{").map_err(io_err)?;
    writeln!(w, "\tVersion: 100").map_err(io_err)?;
    writeln!(w, "\tCount: {}", total_obj_count + 1).map_err(io_err)?;
    writeln!(w, "\tObjectType: \"GlobalSettings\" {{").map_err(io_err)?;
    writeln!(w, "\t\tCount: 1").map_err(io_err)?;
    writeln!(w, "\t}}").map_err(io_err)?;
    writeln!(w, "\tObjectType: \"Geometry\" {{").map_err(io_err)?;
    writeln!(w, "\t\tCount: {}", geom_count).map_err(io_err)?;
    writeln!(w, "\t}}").map_err(io_err)?;
    writeln!(w, "\tObjectType: \"Model\" {{").map_err(io_err)?;
    writeln!(w, "\t\tCount: {}", model_count).map_err(io_err)?;
    writeln!(w, "\t}}").map_err(io_err)?;
    writeln!(w, "\tObjectType: \"Material\" {{").map_err(io_err)?;
    writeln!(w, "\t\tCount: {}", mat_count).map_err(io_err)?;
    writeln!(w, "\t}}").map_err(io_err)?;
    if tex_obj_count > 0 {
        writeln!(w, "\tObjectType: \"Texture\" {{").map_err(io_err)?;
        writeln!(w, "\t\tCount: {}", tex_obj_count).map_err(io_err)?;
        writeln!(w, "\t}}").map_err(io_err)?;
        writeln!(w, "\tObjectType: \"Video\" {{").map_err(io_err)?;
        writeln!(w, "\t\tCount: {}", tex_obj_count).map_err(io_err)?;
        writeln!(w, "\t}}").map_err(io_err)?;
    }
    writeln!(w, "}}").map_err(io_err)?;
    writeln!(w).map_err(io_err)?;

    // Objects
    writeln!(w, "Objects:  {{").map_err(io_err)?;

    // Generate deterministic IDs:
    // Geometry: 100_000_000 + gi
    // Model:    200_000_000 + gi
    // Material: 300_000_000 + gi
    // Texture:  400_000_000 + gi
    // Video:    500_000_000 + gi
    // Collision Geometry: 600_000_000 + ci
    // Collision Model:    700_000_000 + ci

    for (gi, g) in groups.iter().enumerate() {
        let geom_id  = 100_000_000i64 + gi as i64;
        let model_id = 200_000_000i64 + gi as i64;
        let mat_id   = 300_000_000i64 + gi as i64;

        // In Unreal mode: render meshes get SM_ prefix
        let name = if unreal_collision {
            if g.slot_key < 0 {
                format!("{}_Untextured_{}", sm_name, gi)
            } else {
                format!("{}_{}", sm_name, gi)
            }
        } else if g.slot_key < 0 {
            format!("Untextured_{}", gi)
        } else {
            format!("Slot_{}", g.slot_key)
        };

        // ── Geometry ─────────────────────────────────────────────────────
        writeln!(w, "\tGeometry: {}, \"Geometry::{}\", \"Mesh\" {{", geom_id, name).map_err(io_err)?;

        // Collect unique vertex indices used by this group
        let mut local_vert_set: Vec<u32> = g.indices.clone();
        local_vert_set.sort_unstable();
        local_vert_set.dedup();

        // Build local remap: global vertex index -> local index
        let mut global_to_local: HashMap<u32, usize> = HashMap::new();
        for (li, &gvi) in local_vert_set.iter().enumerate() {
            global_to_local.insert(gvi, li);
        }

        let local_vert_count = local_vert_set.len();
        let local_tri_count = g.indices.len() / 3;

        // Vertices
        write!(w, "\t\tVertices: *{} {{", local_vert_count * 3).map_err(io_err)?;
        write!(w, "\n\t\t\ta: ").map_err(io_err)?;
        for (i, &gvi) in local_vert_set.iter().enumerate() {
            let v = &verts[gvi as usize];
            if i > 0 { write!(w, ",").map_err(io_err)?; }
            write!(w, "{},{},{}", v.position[0], v.position[1], v.position[2]).map_err(io_err)?;
        }
        writeln!(w, "\n\t\t}}").map_err(io_err)?;

        // PolygonVertexIndex (FBX convention: last index of each polygon is bitwise-negated: -(idx+1))
        write!(w, "\t\tPolygonVertexIndex: *{} {{", local_tri_count * 3).map_err(io_err)?;
        write!(w, "\n\t\t\ta: ").map_err(io_err)?;
        for t in 0..local_tri_count {
            let li0 = global_to_local[&g.indices[t * 3]] as i32;
            let li1 = global_to_local[&g.indices[t * 3 + 1]] as i32;
            let li2 = global_to_local[&g.indices[t * 3 + 2]] as i32;
            if t > 0 { write!(w, ",").map_err(io_err)?; }
            write!(w, "{},{},{}", li0, li1, -(li2 + 1)).map_err(io_err)?;
        }
        writeln!(w, "\n\t\t}}").map_err(io_err)?;

        // LayerElementNormal
        writeln!(w, "\t\tLayerElementNormal: 0 {{").map_err(io_err)?;
        writeln!(w, "\t\t\tVersion: 101").map_err(io_err)?;
        writeln!(w, "\t\t\tName: \"\"").map_err(io_err)?;
        writeln!(w, "\t\t\tMappingInformationType: \"ByVertice\"").map_err(io_err)?;
        writeln!(w, "\t\t\tReferenceInformationType: \"Direct\"").map_err(io_err)?;
        write!(w, "\t\t\tNormals: *{} {{", local_vert_count * 3).map_err(io_err)?;
        write!(w, "\n\t\t\t\ta: ").map_err(io_err)?;
        for (i, &gvi) in local_vert_set.iter().enumerate() {
            let v = &verts[gvi as usize];
            if i > 0 { write!(w, ",").map_err(io_err)?; }
            write!(w, "{},{},{}", v.normal[0], v.normal[1], v.normal[2]).map_err(io_err)?;
        }
        writeln!(w, "\n\t\t\t}}").map_err(io_err)?;
        writeln!(w, "\t\t}}").map_err(io_err)?;

        // LayerElementUV
        writeln!(w, "\t\tLayerElementUV: 0 {{").map_err(io_err)?;
        writeln!(w, "\t\t\tVersion: 101").map_err(io_err)?;
        writeln!(w, "\t\t\tName: \"UVChannel_1\"").map_err(io_err)?;
        writeln!(w, "\t\t\tMappingInformationType: \"ByVertice\"").map_err(io_err)?;
        writeln!(w, "\t\t\tReferenceInformationType: \"Direct\"").map_err(io_err)?;
        write!(w, "\t\t\tUV: *{} {{", local_vert_count * 2).map_err(io_err)?;
        write!(w, "\n\t\t\t\ta: ").map_err(io_err)?;
        for (i, &gvi) in local_vert_set.iter().enumerate() {
            let v = &verts[gvi as usize];
            if i > 0 { write!(w, ",").map_err(io_err)?; }
            write!(w, "{},{}", v.uv[0], v.uv[1]).map_err(io_err)?;
        }
        writeln!(w, "\n\t\t\t}}").map_err(io_err)?;
        writeln!(w, "\t\t}}").map_err(io_err)?;

        // LayerElementMaterial (single material per geometry)
        writeln!(w, "\t\tLayerElementMaterial: 0 {{").map_err(io_err)?;
        writeln!(w, "\t\t\tVersion: 101").map_err(io_err)?;
        writeln!(w, "\t\t\tName: \"\"").map_err(io_err)?;
        writeln!(w, "\t\t\tMappingInformationType: \"AllSame\"").map_err(io_err)?;
        writeln!(w, "\t\t\tReferenceInformationType: \"IndexToDirect\"").map_err(io_err)?;
        writeln!(w, "\t\t\tMaterials: *1 {{").map_err(io_err)?;
        writeln!(w, "\t\t\t\ta: 0").map_err(io_err)?;
        writeln!(w, "\t\t\t}}").map_err(io_err)?;
        writeln!(w, "\t\t}}").map_err(io_err)?;

        // Layer
        writeln!(w, "\t\tLayer: 0 {{").map_err(io_err)?;
        writeln!(w, "\t\t\tVersion: 100").map_err(io_err)?;
        writeln!(w, "\t\t\tLayerElement:  {{").map_err(io_err)?;
        writeln!(w, "\t\t\t\tType: \"LayerElementNormal\"").map_err(io_err)?;
        writeln!(w, "\t\t\t\tTypedIndex: 0").map_err(io_err)?;
        writeln!(w, "\t\t\t}}").map_err(io_err)?;
        writeln!(w, "\t\t\tLayerElement:  {{").map_err(io_err)?;
        writeln!(w, "\t\t\t\tType: \"LayerElementUV\"").map_err(io_err)?;
        writeln!(w, "\t\t\t\tTypedIndex: 0").map_err(io_err)?;
        writeln!(w, "\t\t\t}}").map_err(io_err)?;
        writeln!(w, "\t\t\tLayerElement:  {{").map_err(io_err)?;
        writeln!(w, "\t\t\t\tType: \"LayerElementMaterial\"").map_err(io_err)?;
        writeln!(w, "\t\t\t\tTypedIndex: 0").map_err(io_err)?;
        writeln!(w, "\t\t\t}}").map_err(io_err)?;
        writeln!(w, "\t\t}}").map_err(io_err)?;

        writeln!(w, "\t}}").map_err(io_err)?;  // end Geometry

        // ── Model ────────────────────────────────────────────────────────
        writeln!(w, "\tModel: {}, \"Model::{}\", \"Mesh\" {{", model_id, name).map_err(io_err)?;
        writeln!(w, "\t\tVersion: 232").map_err(io_err)?;
        writeln!(w, "\t\tProperties70:  {{").map_err(io_err)?;
        writeln!(w, "\t\t\tP: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\",0,0,0").map_err(io_err)?;
        writeln!(w, "\t\t}}").map_err(io_err)?;
        writeln!(w, "\t}}").map_err(io_err)?;

        // ── Material ─────────────────────────────────────────────────────
        let vi = g.indices.first().copied().unwrap_or(0) as usize;
        let col = &verts[vi].color;
        let r = (col[0].min(1.0).max(0.0) as f64).min(1.0);
        let gc = (col[1].min(1.0).max(0.0) as f64).min(1.0);
        let b = (col[2].min(1.0).max(0.0) as f64).min(1.0);

        writeln!(w, "\tMaterial: {}, \"Material::{}\", \"\" {{", mat_id, name).map_err(io_err)?;
        writeln!(w, "\t\tVersion: 102").map_err(io_err)?;
        writeln!(w, "\t\tShadingModel: \"phong\"").map_err(io_err)?;
        writeln!(w, "\t\tProperties70:  {{").map_err(io_err)?;
        writeln!(w, "\t\t\tP: \"DiffuseColor\", \"Color\", \"\", \"A\",{},{},{}", r, gc, b).map_err(io_err)?;
        writeln!(w, "\t\t\tP: \"SpecularFactor\", \"Number\", \"\", \"A\",0.0").map_err(io_err)?;
        writeln!(w, "\t\t\tP: \"Shininess\", \"Number\", \"\", \"A\",20.0").map_err(io_err)?;
        writeln!(w, "\t\t}}").map_err(io_err)?;
        writeln!(w, "\t}}").map_err(io_err)?;
    }

    // ── Texture & Video objects for ALL texture channels ────────────────
    // ID scheme per channel: base + channel_offset * 1_000_000 + group_index
    //   Diffuse: 400_000_000, Normal: 401_000_000, Spec: 402_000_000,
    //   Glow: 403_000_000, Alpha: 404_000_000
    //   Video IDs mirror at 500_xxx_xxx
    for (gi, g) in groups.iter().enumerate() {
        if let Some(textures) = slot_textures.get(&g.slot_key) {
            for (tex_type, tex_path) in textures {
                let channel_offset = textype_channel_offset(*tex_type);
                let tex_id   = 400_000_000i64 + channel_offset * 1_000_000 + gi as i64;
                let video_id = 500_000_000i64 + channel_offset * 1_000_000 + gi as i64;

                let filename = Path::new(tex_path)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("texture.jpg");

                let channel_label = textype_label(*tex_type);

                // Texture
                writeln!(w, "\tTexture: {}, \"Texture::{}_{}\", \"\" {{",
                    tex_id, channel_label, filename).map_err(io_err)?;
                writeln!(w, "\t\tType: \"TextureVideoClip\"").map_err(io_err)?;
                writeln!(w, "\t\tVersion: 202").map_err(io_err)?;
                writeln!(w, "\t\tTextureName: \"Texture::{}\"", filename).map_err(io_err)?;
                writeln!(w, "\t\tFileName: \"{}\"", tex_path.replace('\\', "/")).map_err(io_err)?;
                writeln!(w, "\t\tRelativeFilename: \"{}\"", filename).map_err(io_err)?;
                writeln!(w, "\t\tProperties70:  {{").map_err(io_err)?;
                writeln!(w, "\t\t\tP: \"UVSet\", \"KString\", \"\", \"\", \"UVChannel_1\"").map_err(io_err)?;
                writeln!(w, "\t\t}}").map_err(io_err)?;
                writeln!(w, "\t}}").map_err(io_err)?;

                // Video (media clip)
                writeln!(w, "\tVideo: {}, \"Video::{}_{}\", \"Clip\" {{",
                    video_id, channel_label, filename).map_err(io_err)?;
                writeln!(w, "\t\tType: \"Clip\"").map_err(io_err)?;
                writeln!(w, "\t\tFileName: \"{}\"", tex_path.replace('\\', "/")).map_err(io_err)?;
                writeln!(w, "\t\tRelativeFilename: \"{}\"", filename).map_err(io_err)?;
                writeln!(w, "\t}}").map_err(io_err)?;
            }
        }
    }

    // ── Collision box geometry (Unreal UBX_ convention) ────────────────
    if unreal_collision && !collision_boxes.is_empty() {
        for (ci, cbox) in collision_boxes.iter().enumerate() {
            let col_geom_id  = 600_000_000i64 + ci as i64;
            let col_model_id = 700_000_000i64 + ci as i64;
            let col_name = format!("UBX_{}_{:04}", sm_name, ci);

            // Geometry: 8 vertices, 12 triangles (6 faces × 2 tris)
            writeln!(w, "\tGeometry: {}, \"Geometry::{}\", \"Mesh\" {{", col_geom_id, col_name).map_err(io_err)?;

            // Vertices (8 corners)
            write!(w, "\t\tVertices: *24 {{").map_err(io_err)?;
            write!(w, "\n\t\t\ta: ").map_err(io_err)?;
            for (i, c) in cbox.corners.iter().enumerate() {
                if i > 0 { write!(w, ",").map_err(io_err)?; }
                write!(w, "{},{},{}", c[0], c[1], c[2]).map_err(io_err)?;
            }
            writeln!(w, "\n\t\t}}").map_err(io_err)?;

            // PolygonVertexIndex: 12 triangles for a box (6 faces × 2 tris)
            // Face winding must be consistent. Using right-hand rule.
            // Corners layout:
            //   0=(min,min,min) 1=(max,min,min) 2=(min,min,max) 3=(max,min,max)
            //   4=(min,max,min) 5=(max,max,min) 6=(min,max,max) 7=(max,max,max)
            let box_tris: [[i32; 3]; 12] = [
                // -Y face (bottom): 0,1,3,2
                [0, 1, 3], [0, 3, 2],
                // +Y face (top): 4,6,7,5
                [4, 6, 7], [4, 7, 5],
                // -X face (left): 0,2,6,4
                [0, 2, 6], [0, 6, 4],
                // +X face (right): 1,5,7,3
                [1, 5, 7], [1, 7, 3],
                // -Z face (front): 0,4,5,1
                [0, 4, 5], [0, 5, 1],
                // +Z face (back): 2,3,7,6
                [2, 3, 7], [2, 7, 6],
            ];

            write!(w, "\t\tPolygonVertexIndex: *36 {{").map_err(io_err)?;
            write!(w, "\n\t\t\ta: ").map_err(io_err)?;
            for (ti, tri) in box_tris.iter().enumerate() {
                if ti > 0 { write!(w, ",").map_err(io_err)?; }
                write!(w, "{},{},{}", tri[0], tri[1], -(tri[2] + 1)).map_err(io_err)?;
            }
            writeln!(w, "\n\t\t}}").map_err(io_err)?;

            writeln!(w, "\t}}").map_err(io_err)?;  // end Geometry

            // Model
            writeln!(w, "\tModel: {}, \"Model::{}\", \"Mesh\" {{", col_model_id, col_name).map_err(io_err)?;
            writeln!(w, "\t\tVersion: 232").map_err(io_err)?;
            writeln!(w, "\t\tProperties70:  {{").map_err(io_err)?;
            writeln!(w, "\t\t\tP: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\",0,0,0").map_err(io_err)?;
            writeln!(w, "\t\t}}").map_err(io_err)?;
            writeln!(w, "\t}}").map_err(io_err)?;
        }
    }

    writeln!(w, "}}").map_err(io_err)?;  // end Objects
    writeln!(w).map_err(io_err)?;

    // Connections
    writeln!(w, "Connections:  {{").map_err(io_err)?;

    for (gi, g) in groups.iter().enumerate() {
        let geom_id  = 100_000_000i64 + gi as i64;
        let model_id = 200_000_000i64 + gi as i64;
        let mat_id   = 300_000_000i64 + gi as i64;

        // Model -> Root (0)
        writeln!(w, "\tC: \"OO\",{},0", model_id).map_err(io_err)?;
        // Geometry -> Model
        writeln!(w, "\tC: \"OO\",{},{}", geom_id, model_id).map_err(io_err)?;
        // Material -> Model
        writeln!(w, "\tC: \"OO\",{},{}", mat_id, model_id).map_err(io_err)?;

        // Texture channels -> Material properties
        if let Some(textures) = slot_textures.get(&g.slot_key) {
            for (tex_type, _) in textures {
                let channel_offset = textype_channel_offset(*tex_type);
                let tex_id   = 400_000_000i64 + channel_offset * 1_000_000 + gi as i64;
                let video_id = 500_000_000i64 + channel_offset * 1_000_000 + gi as i64;
                let fbx_prop = textype_to_fbx_property(*tex_type);

                // Texture -> Material (specific property)
                writeln!(w, "\tC: \"OP\",{},{},\"{}\"", tex_id, mat_id, fbx_prop).map_err(io_err)?;
                // Video -> Texture
                writeln!(w, "\tC: \"OO\",{},{}", video_id, tex_id).map_err(io_err)?;
            }
        }
    }

    // Collision box connections
    if unreal_collision {
        for ci in 0..collision_boxes.len() {
            let col_geom_id  = 600_000_000i64 + ci as i64;
            let col_model_id = 700_000_000i64 + ci as i64;

            // Collision Model -> Root (0)
            writeln!(w, "\tC: \"OO\",{},0", col_model_id).map_err(io_err)?;
            // Collision Geometry -> Collision Model
            writeln!(w, "\tC: \"OO\",{},{}", col_geom_id, col_model_id).map_err(io_err)?;
        }
    }

    writeln!(w, "}}").map_err(io_err)?;

    w.flush().map_err(io_err)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align4() {
        assert_eq!(align4(0), 0);
        assert_eq!(align4(1), 4);
        assert_eq!(align4(4), 4);
        assert_eq!(align4(5), 8);
        assert_eq!(align4(13), 16);
    }

    #[test]
    fn test_optimize_mesh_dedup() {
        let v = MeshVertex {
            position: [1.0, 2.0, 3.0],
            normal: [0.0, 1.0, 0.0],
            uv: [0.5, 0.5],
            color: [1.0, 1.0, 1.0, 0.0],
        };
        // Two identical vertices should be merged.
        let verts = vec![v, v, MeshVertex { position: [4.0, 5.0, 6.0], ..v }];
        let indices = vec![0, 1, 2]; // v0 == v1 after dedup -> degenerate

        let (opt_v, opt_i) = optimize_mesh(&verts, &indices);
        // v0 and v1 merge, so the triangle becomes degenerate and is removed.
        assert_eq!(opt_v.len(), 2);
        assert_eq!(opt_i.len(), 0);
    }

    #[test]
    fn test_optimize_mesh_keeps_valid() {
        let v0 = MeshVertex {
            position: [0.0, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            uv: [0.0, 0.0],
            color: [1.0, 1.0, 1.0, 0.0],
        };
        let v1 = MeshVertex { position: [1.0, 0.0, 0.0], ..v0 };
        let v2 = MeshVertex { position: [0.0, 0.0, 1.0], ..v0 };

        let verts = vec![v0, v1, v2];
        let indices = vec![0, 1, 2];

        let (opt_v, opt_i) = optimize_mesh(&verts, &indices);
        assert_eq!(opt_v.len(), 3);
        assert_eq!(opt_i.len(), 3);
    }

    #[test]
    fn test_format_f32() {
        assert_eq!(format_f32(0.0), "0.0");
        assert_eq!(format_f32(1.0), "1.0");
        assert_eq!(format_f32(0.5), "0.5");
        assert_eq!(format_f32(-3.0), "-3.0");
    }

    #[test]
    fn test_guess_mime() {
        assert_eq!(guess_image_mime("foo/bar.png"), "image/png");
        assert_eq!(guess_image_mime("foo/bar.jpg"), "image/jpeg");
        assert_eq!(guess_image_mime("foo/bar.JPEG"), "image/jpeg");
        assert_eq!(guess_image_mime("foo/bar.tga"), "image/png");
    }

    #[test]
    fn test_collision_boxes_from_solid_world() {
        use crate::octree::{Cube, OctreeWorld, EDGES_SOLID, newcubes, F_SOLID, MAT_AIR};
        // Create a small world with one solid cube
        let mut root = newcubes(F_SOLID, MAT_AIR);
        root[0].edges = EDGES_SOLID;
        let world = OctreeWorld {
            root,
            world_scale: 4,  // 16x16x16
            entities: Vec::new(),
            ogz_version: 33,
        };
        let boxes = generate_collision_boxes(&world);
        // Should produce at least 1 collision box for the solid leaf
        assert!(!boxes.is_empty(), "Expected collision boxes from solid geometry, got 0");
        // Each box should have 8 corners
        for b in &boxes {
            assert_eq!(b.corners.len(), 8);
        }
    }
}
