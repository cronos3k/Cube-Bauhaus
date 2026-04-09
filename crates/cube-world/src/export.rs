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
    //   < 0.0   → no texture loaded (debug colour)
    // We use i32 as the key: -1 for untextured, otherwise the layer index.

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
        let key = if layer < 0.0 { -1 } else { layer.round() as i32 };

        let g = group_map.entry(key).or_insert_with(|| {
            let idx = groups.len();
            groups.push(PrimGroup { slot_key: key, indices: Vec::new() });
            idx
        });
        groups[*g].indices.push(indices[t * 3]);
        groups[*g].indices.push(indices[t * 3 + 1]);
        groups[*g].indices.push(indices[t * 3 + 2]);
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
pub fn export_fbx(
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
        let key = if layer < 0.0 { -1 } else { layer.round() as i32 };

        let g = group_map.entry(key).or_insert_with(|| {
            let idx = groups.len();
            groups.push(FbxGroup { slot_key: key, indices: Vec::new() });
            idx
        });
        groups[*g].indices.push(indices[t * 3]);
        groups[*g].indices.push(indices[t * 3 + 1]);
        groups[*g].indices.push(indices[t * 3 + 2]);
    }

    // ── 3. Resolve texture paths per group ───────────────────────────────
    let mut slot_tex_path: HashMap<i32, String> = HashMap::new();
    if let (Some(reg), Some(base)) = (registry, texture_base_path) {
        for g in &groups {
            if g.slot_key < 0 { continue; }
            if slot_tex_path.contains_key(&g.slot_key) { continue; }
            for slot in &reg.slots {
                if let Some(dtex) = slot.textures.iter().find(|t| t.tex_type == TexType::Diffuse) {
                    if dtex.layer as i32 == g.slot_key && !dtex.path.is_empty() {
                        let tex_path = Path::new(base).join(&dtex.path);
                        if tex_path.exists() {
                            slot_tex_path.insert(g.slot_key, tex_path.display().to_string());
                        } else {
                            // Try alternatives
                            for alt in try_alternative_paths(&tex_path) {
                                if alt.exists() {
                                    slot_tex_path.insert(g.slot_key, alt.display().to_string());
                                    break;
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }
    }

    // ── 4. Write FBX ASCII ───────────────────────────────────────────────
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
    let model_count = groups.len();
    let mat_count = groups.len();
    let tex_count = slot_tex_path.len();
    let total_obj_count = model_count * 2 + mat_count + tex_count; // Geometry+Model per group, Material per group, Texture per textured group

    writeln!(w, "Definitions:  {{").map_err(io_err)?;
    writeln!(w, "\tVersion: 100").map_err(io_err)?;
    writeln!(w, "\tCount: {}", total_obj_count + 1).map_err(io_err)?;
    writeln!(w, "\tObjectType: \"GlobalSettings\" {{").map_err(io_err)?;
    writeln!(w, "\t\tCount: 1").map_err(io_err)?;
    writeln!(w, "\t}}").map_err(io_err)?;
    writeln!(w, "\tObjectType: \"Geometry\" {{").map_err(io_err)?;
    writeln!(w, "\t\tCount: {}", model_count).map_err(io_err)?;
    writeln!(w, "\t}}").map_err(io_err)?;
    writeln!(w, "\tObjectType: \"Model\" {{").map_err(io_err)?;
    writeln!(w, "\t\tCount: {}", model_count).map_err(io_err)?;
    writeln!(w, "\t}}").map_err(io_err)?;
    writeln!(w, "\tObjectType: \"Material\" {{").map_err(io_err)?;
    writeln!(w, "\t\tCount: {}", mat_count).map_err(io_err)?;
    writeln!(w, "\t}}").map_err(io_err)?;
    if tex_count > 0 {
        writeln!(w, "\tObjectType: \"Texture\" {{").map_err(io_err)?;
        writeln!(w, "\t\tCount: {}", tex_count).map_err(io_err)?;
        writeln!(w, "\t}}").map_err(io_err)?;
        writeln!(w, "\tObjectType: \"Video\" {{").map_err(io_err)?;
        writeln!(w, "\t\tCount: {}", tex_count).map_err(io_err)?;
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

    for (gi, g) in groups.iter().enumerate() {
        let geom_id  = 100_000_000i64 + gi as i64;
        let model_id = 200_000_000i64 + gi as i64;
        let mat_id   = 300_000_000i64 + gi as i64;

        let name = if g.slot_key < 0 {
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

    // ── Texture & Video objects for textured slots ────────────────────────
    for (gi, g) in groups.iter().enumerate() {
        if let Some(tex_path) = slot_tex_path.get(&g.slot_key) {
            let tex_id   = 400_000_000i64 + gi as i64;
            let video_id = 500_000_000i64 + gi as i64;

            let filename = Path::new(tex_path)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("texture.jpg");

            // Texture
            writeln!(w, "\tTexture: {}, \"Texture::{}\", \"\" {{", tex_id, filename).map_err(io_err)?;
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
            writeln!(w, "\tVideo: {}, \"Video::{}\", \"Clip\" {{", video_id, filename).map_err(io_err)?;
            writeln!(w, "\t\tType: \"Clip\"").map_err(io_err)?;
            writeln!(w, "\t\tFileName: \"{}\"", tex_path.replace('\\', "/")).map_err(io_err)?;
            writeln!(w, "\t\tRelativeFilename: \"{}\"", filename).map_err(io_err)?;
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

        // Texture -> Material (DiffuseColor property)
        if slot_tex_path.contains_key(&g.slot_key) {
            let tex_id   = 400_000_000i64 + gi as i64;
            let video_id = 500_000_000i64 + gi as i64;
            writeln!(w, "\tC: \"OP\",{},{},\"DiffuseColor\"", tex_id, mat_id).map_err(io_err)?;
            writeln!(w, "\tC: \"OO\",{},{}", video_id, tex_id).map_err(io_err)?;
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
}
