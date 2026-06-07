//! End-to-end round-trip tests: build a rig → export GLB → re-import it with the
//! `gltf` crate. Because the importer parses our hand-rolled GLB with an
//! independent library, a passing round-trip also proves the exported container
//! is spec-conformant.

use cube_rig::export::glb::export_glb;
use cube_rig::import::import_model;
use cube_rig::mesh::{RigVertex, SkinnedMesh};
use cube_rig::skeleton::{Bone, Skeleton};
use glam::{Mat4, Vec3};

fn sample_rig() -> (SkinnedMesh, Skeleton) {
    // a small strip of 4 verts, 2 triangles
    let v = |x: f32, y: f32| RigVertex {
        position: [x, y, 0.0],
        normal: [0.0, 0.0, 1.0],
        uv: [x, y],
    };
    let verts = vec![v(0.0, 0.0), v(1.0, 0.0), v(0.0, 1.0), v(1.0, 1.0)];
    let idx = vec![0, 1, 2, 1, 3, 2];
    let mut mesh = SkinnedMesh::new(verts, idx);

    let mut sk = Skeleton::new();
    let root = sk.add(Bone::root("root"));
    let tip = sk.add(Bone::new("tip", Some(root), Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0))));

    // left edge to root, right edge blended root/tip
    mesh.set_rigid(0, root as u16);
    mesh.set_rigid(2, root as u16);
    mesh.add_weight(1, root as u16, 0.5);
    mesh.add_weight(1, tip as u16, 0.5);
    mesh.add_weight(3, tip as u16, 1.0);

    (mesh, sk)
}

#[test]
fn glb_export_reimports_with_skeleton_and_weights() {
    let (mesh, sk) = sample_rig();
    let path = std::env::temp_dir().join("cube_rig_roundtrip.glb");
    export_glb(&path, &mesh, &sk).expect("export");

    let imported = import_model(&path).expect("reimport");

    // geometry survived
    assert_eq!(imported.mesh.vertex_count(), 4);
    assert_eq!(imported.mesh.triangle_count(), 2);

    // skeleton survived with the right names and hierarchy
    let skel = imported.skeleton.expect("skeleton present");
    assert_eq!(skel.len(), 2);
    assert_eq!(skel.bones[0].name, "root");
    assert_eq!(skel.bones[1].name, "tip");
    assert_eq!(skel.bones[1].parent, Some(0));

    // weights survived: vertex 3 is bound fully to "tip" (bone 1)
    assert_eq!(imported.mesh.dominant_bone(3), Some(1));
    // vertex 1 is a ~50/50 blend
    let infl = imported.mesh.influences[1];
    let active: Vec<_> = infl.iter().filter(|i| i.weight > 0.0).collect();
    assert_eq!(active.len(), 2);
    let sum: f32 = infl.iter().map(|i| i.weight).sum();
    assert!((sum - 1.0).abs() < 1e-4);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn obj_export_reimports_geometry() {
    let (mesh, _sk) = sample_rig();
    let path = std::env::temp_dir().join("cube_rig_roundtrip.obj");
    cube_rig::export::obj::export_obj(&path, &mesh).expect("export obj");

    let imported = import_model(&path).expect("reimport obj");
    assert_eq!(imported.mesh.vertex_count(), 4);
    assert_eq!(imported.mesh.triangle_count(), 2);
    assert!(imported.skeleton.is_none());

    let _ = std::fs::remove_file(&path);
}
