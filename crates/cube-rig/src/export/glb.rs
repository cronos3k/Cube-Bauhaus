//! Binary glTF 2.0 (`.glb`) exporter with skinning.
//!
//! Writes a single skinned primitive plus a `skin` whose joints mirror the
//! [`Skeleton`] bone order one-to-one (so bone index == joint index == the value
//! stored in `JOINTS_0`). No external glTF crate is used on the write side — the
//! container is simple and the octree exporter already hand-rolls it the same
//! way. Vertices with no influence are bound rigidly to the root bone so the
//! output stays a valid skinned mesh.

use std::path::Path;

use glam::Mat4;

use crate::mesh::SkinnedMesh;
use crate::skeleton::Skeleton;

use super::ExportError;

const GLB_MAGIC: u32 = 0x4654_6C67; // "glTF"
const CHUNK_JSON: u32 = 0x4E4F_534A; // "JSON"
const CHUNK_BIN: u32 = 0x004E_4942; // "BIN\0"

pub fn export_glb(path: &Path, mesh: &SkinnedMesh, skeleton: &Skeleton) -> Result<(), ExportError> {
    let bytes = build_glb(mesh, skeleton)?;
    std::fs::write(path, bytes).map_err(|e| ExportError::Io(e.to_string()))
}

/// Build the in-memory `.glb` byte stream.
pub fn build_glb(mesh: &SkinnedMesh, skeleton: &Skeleton) -> Result<Vec<u8>, ExportError> {
    if mesh.vertices.is_empty() {
        return Err(ExportError::Empty);
    }
    if skeleton.len() > u16::MAX as usize {
        return Err(ExportError::TooManyBones(skeleton.len()));
    }

    let vcount = mesh.vertices.len();
    let (bmin, bmax) = mesh.bounds();

    // ── Binary buffer ────────────────────────────────────────────────────────
    let mut bin: Vec<u8> = Vec::new();
    let mut views: Vec<(usize, usize, Option<u32>)> = Vec::new(); // (offset, len, target)
    const ARRAY_BUFFER: u32 = 34962;
    const ELEMENT_ARRAY: u32 = 34963;

    // 0: positions (VEC3 f32)
    let pos_view = push_view(&mut bin, &mut views, Some(ARRAY_BUFFER), |b| {
        for v in &mesh.vertices {
            push_vec3(b, v.position);
        }
    });
    // 1: normals
    let nrm_view = push_view(&mut bin, &mut views, Some(ARRAY_BUFFER), |b| {
        for v in &mesh.vertices {
            push_vec3(b, v.normal);
        }
    });
    // 2: uvs (VEC2 f32)
    let uv_view = push_view(&mut bin, &mut views, Some(ARRAY_BUFFER), |b| {
        for v in &mesh.vertices {
            push_f32(b, v.uv[0]);
            push_f32(b, v.uv[1]);
        }
    });
    // 3: joints (VEC4 u16)
    let joint_view = push_view(&mut bin, &mut views, Some(ARRAY_BUFFER), |b| {
        for inf in &mesh.influences {
            let (joints, _) = packed_influences(inf);
            for j in joints {
                push_u16(b, j);
            }
        }
    });
    // 4: weights (VEC4 f32)
    let weight_view = push_view(&mut bin, &mut views, Some(ARRAY_BUFFER), |b| {
        for inf in &mesh.influences {
            let (_, weights) = packed_influences(inf);
            for w in weights {
                push_f32(b, w);
            }
        }
    });
    // 5: indices (SCALAR u32)
    let idx_view = push_view(&mut bin, &mut views, Some(ELEMENT_ARRAY), |b| {
        for &i in &mesh.indices {
            push_u32(b, i);
        }
    });
    // 6: inverse bind matrices (MAT4 f32) — empty if no skeleton
    let ibm_view = push_view(&mut bin, &mut views, None, |b| {
        for m in skeleton.inverse_binds() {
            push_mat4(b, m);
        }
    });

    // ── Accessors ────────────────────────────────────────────────────────────
    let mut accessors = Vec::new();
    let pos_acc = accessors.len();
    accessors.push(format!(
        r#"{{"bufferView":{pos_view},"componentType":5126,"count":{vcount},"type":"VEC3","min":[{},{},{}],"max":[{},{},{}]}}"#,
        bmin.x, bmin.y, bmin.z, bmax.x, bmax.y, bmax.z
    ));
    let nrm_acc = accessors.len();
    accessors.push(format!(
        r#"{{"bufferView":{nrm_view},"componentType":5126,"count":{vcount},"type":"VEC3"}}"#
    ));
    let uv_acc = accessors.len();
    accessors.push(format!(
        r#"{{"bufferView":{uv_view},"componentType":5126,"count":{vcount},"type":"VEC2"}}"#
    ));
    let joint_acc = accessors.len();
    accessors.push(format!(
        r#"{{"bufferView":{joint_view},"componentType":5123,"count":{vcount},"type":"VEC4"}}"#
    ));
    let weight_acc = accessors.len();
    accessors.push(format!(
        r#"{{"bufferView":{weight_view},"componentType":5126,"count":{vcount},"type":"VEC4"}}"#
    ));
    let idx_acc = accessors.len();
    accessors.push(format!(
        r#"{{"bufferView":{idx_view},"componentType":5125,"count":{},"type":"SCALAR"}}"#,
        mesh.indices.len()
    ));
    let ibm_acc = accessors.len();
    accessors.push(format!(
        r#"{{"bufferView":{ibm_view},"componentType":5126,"count":{},"type":"MAT4"}}"#,
        skeleton.len()
    ));

    // ── Nodes / skin ─────────────────────────────────────────────────────────
    // node 0 = mesh node; nodes 1..=N = joints (bone i → node i+1).
    let joint_node = |bone: usize| bone + 1;
    let mut nodes = Vec::new();
    nodes.push(r#"{"name":"rigged_mesh","mesh":0,"skin":0}"#.to_string());
    for (i, b) in skeleton.bones.iter().enumerate() {
        let children: Vec<String> = skeleton
            .children(i)
            .into_iter()
            .map(|c| joint_node(c).to_string())
            .collect();
        let child_field = if children.is_empty() {
            String::new()
        } else {
            format!(r#","children":[{}]"#, children.join(","))
        };
        nodes.push(format!(
            r#"{{"name":{},"matrix":[{}]{}}}"#,
            json_string(&b.name),
            mat4_json(b.local_bind),
            child_field
        ));
    }

    let joints_list: Vec<String> = (0..skeleton.len()).map(|i| joint_node(i).to_string()).collect();
    let skeleton_root = skeleton.roots().first().map(|&r| joint_node(r)).unwrap_or(1);
    let skin = format!(
        r#"{{"inverseBindMatrices":{ibm_acc},"skeleton":{skeleton_root},"joints":[{}]}}"#,
        joints_list.join(",")
    );

    // Scene roots: the mesh node plus the skeleton root joints.
    let mut scene_nodes = vec!["0".to_string()];
    for r in skeleton.roots() {
        scene_nodes.push(joint_node(r).to_string());
    }

    // ── Buffer views JSON ─────────────────────────────────────────────────────
    let views_json: Vec<String> = views
        .iter()
        .map(|(off, len, target)| match target {
            Some(t) => format!(
                r#"{{"buffer":0,"byteOffset":{off},"byteLength":{len},"target":{t}}}"#
            ),
            None => format!(r#"{{"buffer":0,"byteOffset":{off},"byteLength":{len}}}"#),
        })
        .collect();

    let primitive = format!(
        r#"{{"attributes":{{"POSITION":{pos_acc},"NORMAL":{nrm_acc},"TEXCOORD_0":{uv_acc},"JOINTS_0":{joint_acc},"WEIGHTS_0":{weight_acc}}},"indices":{idx_acc}}}"#
    );

    let json = format!(
        r#"{{"asset":{{"version":"2.0","generator":"Cube Bauhaus rig bay"}},"scene":0,"scenes":[{{"nodes":[{}]}}],"nodes":[{}],"meshes":[{{"primitives":[{}]}}],"skins":[{}],"accessors":[{}],"bufferViews":[{}],"buffers":[{{"byteLength":{}}}]}}"#,
        scene_nodes.join(","),
        nodes.join(","),
        primitive,
        skin,
        accessors.join(","),
        views_json.join(","),
        bin.len()
    );

    Ok(assemble_glb(json.into_bytes(), bin))
}

/// Pack a vertex's influences into glTF `(JOINTS_0, WEIGHTS_0)` vec4s, falling
/// back to a rigid root bind for unweighted vertices so the glTF stays valid.
fn packed_influences(inf: &crate::mesh::VertexInfluences) -> ([u16; 4], [f32; 4]) {
    let mut joints = [0u16; 4];
    let mut weights = [0.0f32; 4];
    let mut total = 0.0;
    for (k, slot) in inf.iter().enumerate() {
        if slot.weight > 0.0 {
            joints[k] = slot.bone;
            weights[k] = slot.weight;
            total += slot.weight;
        }
    }
    if total <= 0.0 {
        // unweighted → rigid to root (joint 0)
        joints = [0; 4];
        weights = [1.0, 0.0, 0.0, 0.0];
    } else if (total - 1.0).abs() > 1e-4 {
        for w in &mut weights {
            *w /= total;
        }
    }
    (joints, weights)
}

// ── binary helpers ───────────────────────────────────────────────────────────

fn push_f32(b: &mut Vec<u8>, v: f32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn push_u32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn push_u16(b: &mut Vec<u8>, v: u16) {
    b.extend_from_slice(&v.to_le_bytes());
}
fn push_vec3(b: &mut Vec<u8>, v: [f32; 3]) {
    for c in v {
        push_f32(b, c);
    }
}
fn push_mat4(b: &mut Vec<u8>, m: Mat4) {
    for c in m.to_cols_array() {
        push_f32(b, c);
    }
}

/// Append a 4-byte-aligned buffer view, record it, and return its index.
fn push_view(
    bin: &mut Vec<u8>,
    views: &mut Vec<(usize, usize, Option<u32>)>,
    target: Option<u32>,
    fill: impl FnOnce(&mut Vec<u8>),
) -> usize {
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }
    let off = bin.len();
    fill(bin);
    let len = bin.len() - off;
    views.push((off, len, target));
    views.len() - 1
}

fn mat4_json(m: Mat4) -> String {
    m.to_cols_array()
        .iter()
        .map(|f| f.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn json_string(s: &str) -> String {
    let escaped: String = s
        .chars()
        .flat_map(|c| match c {
            '"' => vec!['\\', '"'],
            '\\' => vec!['\\', '\\'],
            c => vec![c],
        })
        .collect();
    format!("\"{escaped}\"")
}

/// Wrap JSON + BIN chunks into the GLB container.
fn assemble_glb(mut json: Vec<u8>, mut bin: Vec<u8>) -> Vec<u8> {
    while !json.len().is_multiple_of(4) {
        json.push(b' ');
    }
    while !bin.len().is_multiple_of(4) {
        bin.push(0);
    }
    let total = 12 + 8 + json.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&GLB_MAGIC.to_le_bytes());
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    // JSON chunk
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(&CHUNK_JSON.to_le_bytes());
    out.extend_from_slice(&json);
    // BIN chunk
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(&CHUNK_BIN.to_le_bytes());
    out.extend_from_slice(&bin);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::RigVertex;
    use crate::skeleton::Bone;

    fn rig() -> (SkinnedMesh, Skeleton) {
        let v = |x: f32| RigVertex { position: [x, 0.0, 0.0], normal: [0.0, 0.0, 1.0], uv: [0.0, 0.0] };
        let mut mesh = SkinnedMesh::new(vec![v(0.0), v(1.0), v(2.0)], vec![0, 1, 2]);
        let mut sk = Skeleton::new();
        let r = sk.add(Bone::root("root"));
        sk.add(Bone::new("tip", Some(r), Mat4::from_translation(glam::Vec3::X)));
        mesh.set_rigid(0, 0);
        mesh.set_rigid(1, 1);
        (mesh, sk)
    }

    #[test]
    fn glb_has_valid_header_and_chunks() {
        let (mesh, sk) = rig();
        let bytes = build_glb(&mesh, &sk).unwrap();

        // header
        assert_eq!(&bytes[0..4], &GLB_MAGIC.to_le_bytes());
        let total = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        assert_eq!(total, bytes.len());

        // first chunk is JSON
        let json_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        assert_eq!(&bytes[16..20], &CHUNK_JSON.to_le_bytes());
        let json = std::str::from_utf8(&bytes[20..20 + json_len]).unwrap();
        assert!(json.contains("\"JOINTS_0\""));
        assert!(json.contains("\"skins\""));
        assert!(json.contains("inverseBindMatrices"));

        // 4-byte alignment everywhere
        assert_eq!(bytes.len() % 4, 0);
    }

    #[test]
    fn unweighted_vertex_falls_back_to_root() {
        let inf = [crate::mesh::Influence::NONE; 4];
        let (joints, weights) = packed_influences(&inf);
        assert_eq!(joints, [0, 0, 0, 0]);
        assert_eq!(weights, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn empty_mesh_errors() {
        let sk = Skeleton::new();
        assert!(build_glb(&SkinnedMesh::default(), &sk).is_err());
    }
}
