//! ASCII FBX 7.4 exporter with skinning, for Unreal/Maya skeletal import.
//!
//! Emits the standard FBX object graph for a skinned mesh:
//!
//! ```text
//!   Geometry(Mesh) ──OO──> Model(Mesh) ──OO──> RootNode
//!        ▲                                          ▲
//!        │ OO                                       │ OO
//!   Deformer(Skin)                            Model(LimbNode) … (bone hierarchy)
//!        ▲                                          │
//!        │ OO                                       │ OO (limb → its cluster)
//!   SubDeformer(Cluster) per bone <────────────────┘
//! ```
//!
//! Each [`Skeleton`] bone becomes a `LimbNode` model (TRS from its `local_bind`)
//! and a skin `Cluster` carrying that bone's vertex indices + weights, its
//! `TransformLink` (global bind) and the mesh `Transform` (identity). A
//! `BindPose` lists every limb. IDs are stable per object so the `Connections`
//! block resolves cleanly.
//!
//! The output is validated structurally (object/connection markers); it is not
//! round-tripped through a full FBX SDK here.

use std::fmt::Write as _;
use std::path::Path;

use glam::{EulerRot, Mat4};

use crate::mesh::SkinnedMesh;
use crate::skeleton::Skeleton;

use super::ExportError;

pub fn export_fbx(path: &Path, mesh: &SkinnedMesh, skeleton: &Skeleton) -> Result<(), ExportError> {
    let text = build_fbx(mesh, skeleton)?;
    std::fs::write(path, text).map_err(|e| ExportError::Io(e.to_string()))
}

/// IDs are assigned deterministically from a base so connections resolve.
const GEOM_ID: i64 = 1_000_000;
const MODEL_ID: i64 = 1_000_001;
const SKIN_ID: i64 = 1_000_002;
const LIMB_BASE: i64 = 2_000_000; // limb i  = LIMB_BASE + i
const CLUSTER_BASE: i64 = 3_000_000; // cluster i = CLUSTER_BASE + i
const POSE_ID: i64 = 4_000_000;

pub fn build_fbx(mesh: &SkinnedMesh, skeleton: &Skeleton) -> Result<String, ExportError> {
    if mesh.vertices.is_empty() {
        return Err(ExportError::Empty);
    }

    let limb_id = |i: usize| LIMB_BASE + i as i64;
    let cluster_id = |i: usize| CLUSTER_BASE + i as i64;

    // Per-bone vertex influence lists.
    let mut bone_indices: Vec<Vec<i32>> = vec![Vec::new(); skeleton.len()];
    let mut bone_weights: Vec<Vec<f64>> = vec![Vec::new(); skeleton.len()];
    for (vi, inf) in mesh.influences.iter().enumerate() {
        for slot in inf {
            if slot.weight > 0.0 && (slot.bone as usize) < skeleton.len() {
                bone_indices[slot.bone as usize].push(vi as i32);
                bone_weights[slot.bone as usize].push(slot.weight as f64);
            }
        }
    }

    let mut s = String::new();
    write_header(&mut s);
    write_definitions(&mut s, skeleton.len());

    let _ = writeln!(s, "Objects:  {{");
    write_geometry(&mut s, mesh);
    write_mesh_model(&mut s);
    for i in 0..skeleton.len() {
        write_limb_model(&mut s, skeleton, i, limb_id(i));
    }
    write_skin_deformer(&mut s);
    for i in 0..skeleton.len() {
        write_cluster(
            &mut s,
            skeleton,
            i,
            cluster_id(i),
            &bone_indices[i],
            &bone_weights[i],
        );
    }
    write_bind_pose(&mut s, skeleton, &limb_id);
    let _ = writeln!(s, "}}");
    let _ = writeln!(s);

    write_connections(&mut s, skeleton, &limb_id, &cluster_id);

    Ok(s)
}

fn write_header(s: &mut String) {
    let _ = writeln!(s, "; FBX 7.4.0 project file");
    let _ = writeln!(s, "; Exported by Cube Bauhaus rig bay");
    let _ = writeln!(s, "; -------------------------------------------");
    let _ = writeln!(s);
    let _ = writeln!(s, "FBXHeaderExtension:  {{");
    let _ = writeln!(s, "\tFBXHeaderVersion: 1003");
    let _ = writeln!(s, "\tFBXVersion: 7400");
    let _ = writeln!(s, "\tCreator: \"Cube Bauhaus rig bay\"");
    let _ = writeln!(s, "}}");
    let _ = writeln!(s);
    let _ = writeln!(s, "GlobalSettings:  {{");
    let _ = writeln!(s, "\tVersion: 1000");
    let _ = writeln!(s, "\tProperties70:  {{");
    let _ = writeln!(s, "\t\tP: \"UpAxis\", \"int\", \"Integer\", \"\",1");
    let _ = writeln!(s, "\t\tP: \"UpAxisSign\", \"int\", \"Integer\", \"\",1");
    let _ = writeln!(s, "\t\tP: \"FrontAxis\", \"int\", \"Integer\", \"\",2");
    let _ = writeln!(s, "\t\tP: \"FrontAxisSign\", \"int\", \"Integer\", \"\",1");
    let _ = writeln!(s, "\t\tP: \"CoordAxis\", \"int\", \"Integer\", \"\",0");
    let _ = writeln!(s, "\t\tP: \"CoordAxisSign\", \"int\", \"Integer\", \"\",1");
    let _ = writeln!(s, "\t\tP: \"UnitScaleFactor\", \"double\", \"Number\", \"\",1.0");
    let _ = writeln!(s, "\t}}");
    let _ = writeln!(s, "}}");
    let _ = writeln!(s);
}

fn write_definitions(s: &mut String, bone_count: usize) {
    let model_count = 1 + bone_count; // mesh + limbs
    let deformer_count = 1 + bone_count; // skin + clusters
    let _ = writeln!(s, "Definitions:  {{");
    let _ = writeln!(s, "\tVersion: 100");
    let _ = writeln!(s, "\tCount: {}", 2 + model_count + deformer_count + 1);
    let _ = writeln!(s, "\tObjectType: \"Geometry\" {{ Count: 1 }}");
    let _ = writeln!(s, "\tObjectType: \"Model\" {{ Count: {model_count} }}");
    let _ = writeln!(s, "\tObjectType: \"Deformer\" {{ Count: {deformer_count} }}");
    let _ = writeln!(s, "\tObjectType: \"Pose\" {{ Count: 1 }}");
    let _ = writeln!(s, "}}");
    let _ = writeln!(s);
}

fn write_geometry(s: &mut String, mesh: &SkinnedMesh) {
    let _ = writeln!(s, "\tGeometry: {GEOM_ID}, \"Geometry::mesh\", \"Mesh\" {{");

    // Vertices
    let _ = write!(s, "\t\tVertices: *{} {{\n\t\t\ta: ", mesh.vertices.len() * 3);
    let coords: Vec<String> = mesh
        .vertices
        .iter()
        .flat_map(|v| v.position.iter().map(|c| format_f(*c as f64)))
        .collect();
    let _ = write!(s, "{}", coords.join(","));
    let _ = writeln!(s, "\n\t\t}}");

    // PolygonVertexIndex: FBX flags the last index of each polygon by negating
    // it and subtracting 1 (i.e. ~i). Triangles only here.
    let _ = write!(s, "\t\tPolygonVertexIndex: *{} {{\n\t\t\ta: ", mesh.indices.len());
    let poly: Vec<String> = mesh
        .indices
        .chunks_exact(3)
        .flat_map(|t| [t[0] as i32, t[1] as i32, !(t[2] as i32)])
        .map(|i| i.to_string())
        .collect();
    let _ = write!(s, "{}", poly.join(","));
    let _ = writeln!(s, "\n\t\t}}");

    // Normals (by vertex)
    let _ = writeln!(s, "\t\tLayerElementNormal: 0 {{");
    let _ = writeln!(s, "\t\t\tVersion: 101");
    let _ = writeln!(s, "\t\t\tName: \"\"");
    let _ = writeln!(s, "\t\t\tMappingInformationType: \"ByVertice\"");
    let _ = writeln!(s, "\t\t\tReferenceInformationType: \"Direct\"");
    let normals: Vec<String> = mesh
        .vertices
        .iter()
        .flat_map(|v| v.normal.iter().map(|c| format_f(*c as f64)))
        .collect();
    let _ = writeln!(
        s,
        "\t\t\tNormals: *{} {{\n\t\t\t\ta: {}\n\t\t\t}}",
        mesh.vertices.len() * 3,
        normals.join(",")
    );
    let _ = writeln!(s, "\t\t}}");

    // UVs (by vertex, direct)
    let _ = writeln!(s, "\t\tLayerElementUV: 0 {{");
    let _ = writeln!(s, "\t\t\tVersion: 101");
    let _ = writeln!(s, "\t\t\tName: \"map1\"");
    let _ = writeln!(s, "\t\t\tMappingInformationType: \"ByVertice\"");
    let _ = writeln!(s, "\t\t\tReferenceInformationType: \"Direct\"");
    let uvs: Vec<String> = mesh
        .vertices
        .iter()
        .flat_map(|v| v.uv.iter().map(|c| format_f(*c as f64)))
        .collect();
    let _ = writeln!(
        s,
        "\t\t\tUV: *{} {{\n\t\t\t\ta: {}\n\t\t\t}}",
        mesh.vertices.len() * 2,
        uvs.join(",")
    );
    let _ = writeln!(s, "\t\t}}");

    let _ = writeln!(s, "\t\tLayer: 0 {{");
    let _ = writeln!(s, "\t\t\tVersion: 100");
    let _ = writeln!(s, "\t\t\tLayerElement:  {{ Type: \"LayerElementNormal\" TypedIndex: 0 }}");
    let _ = writeln!(s, "\t\t\tLayerElement:  {{ Type: \"LayerElementUV\" TypedIndex: 0 }}");
    let _ = writeln!(s, "\t\t}}");

    let _ = writeln!(s, "\t}}");
}

fn write_mesh_model(s: &mut String) {
    let _ = writeln!(s, "\tModel: {MODEL_ID}, \"Model::mesh\", \"Mesh\" {{");
    let _ = writeln!(s, "\t\tVersion: 232");
    let _ = writeln!(s, "\t\tProperties70:  {{");
    let _ = writeln!(s, "\t\t\tP: \"DefaultAttributeIndex\", \"int\", \"Integer\", \"\",0");
    let _ = writeln!(s, "\t\t}}");
    let _ = writeln!(s, "\t\tShading: T");
    let _ = writeln!(s, "\t\tCulling: \"CullingOff\"");
    let _ = writeln!(s, "\t}}");
}

fn write_limb_model(s: &mut String, skeleton: &Skeleton, bone: usize, id: i64) {
    let b = &skeleton.bones[bone];
    let (scale, rot, trans) = b.local_bind.to_scale_rotation_translation();
    let (rx, ry, rz) = rot.to_euler(EulerRot::XYZ);
    let deg = |r: f32| (r as f64).to_degrees();

    let _ = writeln!(s, "\tModel: {id}, \"Model::{}\", \"LimbNode\" {{", b.name);
    let _ = writeln!(s, "\t\tVersion: 232");
    let _ = writeln!(s, "\t\tProperties70:  {{");
    let _ = writeln!(
        s,
        "\t\t\tP: \"Lcl Translation\", \"Lcl Translation\", \"\", \"A\",{},{},{}",
        format_f(trans.x as f64),
        format_f(trans.y as f64),
        format_f(trans.z as f64)
    );
    let _ = writeln!(
        s,
        "\t\t\tP: \"Lcl Rotation\", \"Lcl Rotation\", \"\", \"A\",{},{},{}",
        format_f(deg(rx)),
        format_f(deg(ry)),
        format_f(deg(rz))
    );
    let _ = writeln!(
        s,
        "\t\t\tP: \"Lcl Scaling\", \"Lcl Scaling\", \"\", \"A\",{},{},{}",
        format_f(scale.x as f64),
        format_f(scale.y as f64),
        format_f(scale.z as f64)
    );
    let _ = writeln!(s, "\t\t}}");
    let _ = writeln!(s, "\t}}");
}

fn write_skin_deformer(s: &mut String) {
    let _ = writeln!(s, "\tDeformer: {SKIN_ID}, \"Deformer::skin\", \"Skin\" {{");
    let _ = writeln!(s, "\t\tVersion: 101");
    let _ = writeln!(s, "\t\tLink_DeformAcuracy: 50");
    let _ = writeln!(s, "\t}}");
}

fn write_cluster(
    s: &mut String,
    skeleton: &Skeleton,
    bone: usize,
    id: i64,
    indices: &[i32],
    weights: &[f64],
) {
    let b = &skeleton.bones[bone];
    let _ = writeln!(s, "\tDeformer: {id}, \"SubDeformer::{}\", \"Cluster\" {{", b.name);
    let _ = writeln!(s, "\t\tVersion: 100");
    let _ = writeln!(s, "\t\tMode: \"Total1\"");

    if !indices.is_empty() {
        let idx: Vec<String> = indices.iter().map(|i| i.to_string()).collect();
        let _ = writeln!(
            s,
            "\t\tIndexes: *{} {{\n\t\t\ta: {}\n\t\t}}",
            indices.len(),
            idx.join(",")
        );
        let wts: Vec<String> = weights.iter().map(|w| format_f(*w)).collect();
        let _ = writeln!(
            s,
            "\t\tWeights: *{} {{\n\t\t\ta: {}\n\t\t}}",
            weights.len(),
            wts.join(",")
        );
    }

    // Transform = mesh global at bind (identity); TransformLink = bone global bind.
    let _ = writeln!(s, "\t\tTransform: *16 {{\n\t\t\ta: {}\n\t\t}}", mat_row(Mat4::IDENTITY));
    let _ = writeln!(
        s,
        "\t\tTransformLink: *16 {{\n\t\t\ta: {}\n\t\t}}",
        mat_row(skeleton.global_bind(bone))
    );
    let _ = writeln!(s, "\t}}");
}

fn write_bind_pose(s: &mut String, skeleton: &Skeleton, limb_id: &impl Fn(usize) -> i64) {
    let _ = writeln!(s, "\tPose: {POSE_ID}, \"Pose::BIND_POSES\", \"BindPose\" {{");
    let _ = writeln!(s, "\t\tType: \"BindPose\"");
    let _ = writeln!(s, "\t\tVersion: 100");
    let _ = writeln!(s, "\t\tNbPoseNodes: {}", skeleton.len() + 1);
    // mesh node at identity
    let _ = writeln!(s, "\t\tPoseNode:  {{");
    let _ = writeln!(s, "\t\t\tNode: {MODEL_ID}");
    let _ = writeln!(s, "\t\t\tMatrix: *16 {{\n\t\t\t\ta: {}\n\t\t\t}}", mat_row(Mat4::IDENTITY));
    let _ = writeln!(s, "\t\t}}");
    for i in 0..skeleton.len() {
        let _ = writeln!(s, "\t\tPoseNode:  {{");
        let _ = writeln!(s, "\t\t\tNode: {}", limb_id(i));
        let _ = writeln!(
            s,
            "\t\t\tMatrix: *16 {{\n\t\t\t\ta: {}\n\t\t\t}}",
            mat_row(skeleton.global_bind(i))
        );
        let _ = writeln!(s, "\t\t}}");
    }
    let _ = writeln!(s, "\t}}");
}

fn write_connections(
    s: &mut String,
    skeleton: &Skeleton,
    limb_id: &impl Fn(usize) -> i64,
    cluster_id: &impl Fn(usize) -> i64,
) {
    let _ = writeln!(s, "Connections:  {{");
    // mesh model → scene root
    let _ = writeln!(s, "\t;Model::mesh, Model::RootNode");
    let _ = writeln!(s, "\tC: \"OO\",{MODEL_ID},0");
    // geometry → mesh model
    let _ = writeln!(s, "\t;Geometry::mesh, Model::mesh");
    let _ = writeln!(s, "\tC: \"OO\",{GEOM_ID},{MODEL_ID}");
    // skin → geometry
    let _ = writeln!(s, "\t;Deformer::skin, Geometry::mesh");
    let _ = writeln!(s, "\tC: \"OO\",{SKIN_ID},{GEOM_ID}");

    // bone hierarchy: each limb → its parent limb (or scene root)
    for i in 0..skeleton.len() {
        let parent = skeleton.bones[i].parent.map(|p| limb_id(p)).unwrap_or(0);
        let _ = writeln!(s, "\t;Model::{} -> parent", skeleton.bones[i].name);
        let _ = writeln!(s, "\tC: \"OO\",{},{}", limb_id(i), parent);
    }

    // clusters → skin, and limb → cluster
    for i in 0..skeleton.len() {
        let _ = writeln!(s, "\t;SubDeformer::{} -> Deformer::skin", skeleton.bones[i].name);
        let _ = writeln!(s, "\tC: \"OO\",{},{}", cluster_id(i), SKIN_ID);
        let _ = writeln!(s, "\t;Model::{} -> SubDeformer", skeleton.bones[i].name);
        let _ = writeln!(s, "\tC: \"OO\",{},{}", limb_id(i), cluster_id(i));
    }
    let _ = writeln!(s, "}}");
}

// ── formatting helpers ────────────────────────────────────────────────────────

/// Format a float with enough precision and no scientific notation, trimming to
/// a plain decimal (FBX ASCII dislikes `1e-7`).
fn format_f(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    let mut s = format!("{:.6}", v);
    // trim trailing zeros but keep at least one digit after the dot
    if s.contains('.') {
        while s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.push('0');
        }
    }
    s
}

/// A 4×4 matrix as 16 row-major-style FBX doubles (FBX stores column-major, the
/// same layout glam uses for `to_cols_array`).
fn mat_row(m: Mat4) -> String {
    m.to_cols_array()
        .iter()
        .map(|c| format_f(*c as f64))
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::RigVertex;
    use crate::skeleton::Bone;
    use glam::Vec3;

    fn rig() -> (SkinnedMesh, Skeleton) {
        let v = |x: f32| RigVertex { position: [x, 0.0, 0.0], normal: [0.0, 0.0, 1.0], uv: [0.0, 0.0] };
        let mut mesh = SkinnedMesh::new(vec![v(0.0), v(1.0), v(2.0)], vec![0, 1, 2]);
        let mut sk = Skeleton::new();
        let r = sk.add(Bone::root("root"));
        sk.add(Bone::new("tip", Some(r), Mat4::from_translation(Vec3::X)));
        mesh.set_rigid(0, 0);
        mesh.set_rigid(1, 1);
        mesh.set_rigid(2, 1);
        (mesh, sk)
    }

    #[test]
    fn emits_skin_objects_and_connections() {
        let (mesh, sk) = rig();
        let fbx = build_fbx(&mesh, &sk).unwrap();
        assert!(fbx.contains("\"Skin\""));
        assert!(fbx.contains("\"Cluster\""));
        assert!(fbx.contains("\"LimbNode\""));
        assert!(fbx.contains("BindPose"));
        assert!(fbx.contains("Connections:"));
        // one cluster per bone references the skin deformer
        let cluster_links = fbx.matches(&format!(",{SKIN_ID}\n")).count();
        assert_eq!(cluster_links, sk.len());
    }

    #[test]
    fn cluster_carries_vertex_indices() {
        let (mesh, sk) = rig();
        let fbx = build_fbx(&mesh, &sk).unwrap();
        // bone 1 (tip) has verts 1 and 2
        assert!(fbx.contains("Indexes: *2"));
    }

    #[test]
    fn format_f_no_scientific() {
        assert_eq!(format_f(0.0), "0");
        assert!(!format_f(0.0000001).contains('e'));
        assert_eq!(format_f(1.5), "1.5");
    }
}
