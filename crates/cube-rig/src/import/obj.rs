//! Wavefront OBJ import (geometry only — OBJ has no skeleton or skin).

use std::path::Path;

use crate::mesh::{RigVertex, SkinnedMesh};

use super::{append_mesh, ImportError, Imported};

pub fn import_obj(path: &Path) -> Result<Imported, ImportError> {
    let opts = tobj::LoadOptions {
        triangulate: true,
        single_index: true,
        ..Default::default()
    };
    let (models, _materials) =
        tobj::load_obj(path, &opts).map_err(|e| ImportError::Parse(e.to_string()))?;

    let mut mesh = SkinnedMesh::default();
    for model in models {
        let m = &model.mesh;
        if m.positions.is_empty() {
            continue;
        }
        let vcount = m.positions.len() / 3;
        let mut verts = Vec::with_capacity(vcount);
        for i in 0..vcount {
            let position = [m.positions[i * 3], m.positions[i * 3 + 1], m.positions[i * 3 + 2]];
            let normal = if m.normals.len() >= (i + 1) * 3 {
                [m.normals[i * 3], m.normals[i * 3 + 1], m.normals[i * 3 + 2]]
            } else {
                [0.0, 0.0, 0.0]
            };
            let uv = if m.texcoords.len() >= (i + 1) * 2 {
                [m.texcoords[i * 2], m.texcoords[i * 2 + 1]]
            } else {
                [0.0, 0.0]
            };
            verts.push(RigVertex { position, normal, uv });
        }
        let part = SkinnedMesh::new(verts, m.indices.clone());
        append_mesh(&mut mesh, part);
    }

    if mesh.vertices.is_empty() {
        return Err(ImportError::NoGeometry);
    }
    if mesh.vertices.iter().all(|v| v.normal == [0.0, 0.0, 0.0]) {
        recompute_normals(&mut mesh);
    }

    Ok(Imported {
        mesh,
        skeleton: None,
        source: path.display().to_string(),
    })
}

/// Area-weighted vertex normals, for OBJ files exported without them.
pub(crate) fn recompute_normals(mesh: &mut SkinnedMesh) {
    use glam::Vec3;
    let mut accum = vec![Vec3::ZERO; mesh.vertices.len()];
    for tri in mesh.indices.chunks_exact(3) {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let pa = mesh.vertices[a].pos();
        let pb = mesh.vertices[b].pos();
        let pc = mesh.vertices[c].pos();
        let n = (pb - pa).cross(pc - pa); // length ∝ 2×area, so area-weighted
        accum[a] += n;
        accum[b] += n;
        accum[c] += n;
    }
    for (v, n) in mesh.vertices.iter_mut().zip(accum) {
        v.normal = n.normalize_or_zero().to_array();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn imports_a_triangle_and_recomputes_normals() {
        let dir = std::env::temp_dir();
        let path = dir.join("cube_rig_test_tri.obj");
        let mut f = std::fs::File::create(&path).unwrap();
        // a single triangle in the z=0 plane, no normals
        writeln!(f, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3").unwrap();
        drop(f);

        let imp = import_obj(&path).unwrap();
        assert_eq!(imp.mesh.vertex_count(), 3);
        assert_eq!(imp.mesh.triangle_count(), 1);
        assert!(imp.skeleton.is_none());
        // recomputed normal should point along +Z
        let n = imp.mesh.vertices[0].normal;
        assert!((n[2].abs() - 1.0).abs() < 1e-4);

        let _ = std::fs::remove_file(&path);
    }
}
