//! Wavefront OBJ export — geometry, normals and UVs only. OBJ has no concept of
//! a skeleton or skin weights, so this drops the rig (the skeleton should go out
//! via GLB/FBX); it's here for round-tripping the plain mesh.

use std::fmt::Write as _;
use std::path::Path;

use crate::mesh::SkinnedMesh;

use super::ExportError;

pub fn export_obj(path: &Path, mesh: &SkinnedMesh) -> Result<(), ExportError> {
    if mesh.vertices.is_empty() {
        return Err(ExportError::Empty);
    }
    let mut s = String::new();
    let _ = writeln!(s, "# Exported by Cube Bauhaus rig bay");
    let _ = writeln!(s, "o rigged_mesh");

    for v in &mesh.vertices {
        let _ = writeln!(s, "v {} {} {}", v.position[0], v.position[1], v.position[2]);
    }
    for v in &mesh.vertices {
        let _ = writeln!(s, "vt {} {}", v.uv[0], v.uv[1]);
    }
    for v in &mesh.vertices {
        let _ = writeln!(s, "vn {} {} {}", v.normal[0], v.normal[1], v.normal[2]);
    }
    // OBJ indices are 1-based and reference v/vt/vn together (single-index mesh).
    for tri in mesh.indices.chunks_exact(3) {
        let a = tri[0] + 1;
        let b = tri[1] + 1;
        let c = tri[2] + 1;
        let _ = writeln!(s, "f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}");
    }

    std::fs::write(path, s).map_err(|e| ExportError::Io(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::RigVertex;

    #[test]
    fn writes_obj_with_faces() {
        let v = |x: f32| RigVertex { position: [x, 0.0, 0.0], normal: [0.0, 0.0, 1.0], uv: [0.0, 0.0] };
        let mesh = SkinnedMesh::new(vec![v(0.0), v(1.0), v(0.0)], vec![0, 1, 2]);
        let path = std::env::temp_dir().join("cube_rig_out.obj");
        export_obj(&path, &mesh).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("f 1/1/1 2/2/2 3/3/3"));
        let _ = std::fs::remove_file(&path);
    }
}
