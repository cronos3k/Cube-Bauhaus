//! Best-effort FBX import (geometry + skeleton + skin weights).
//!
//! FBX has no mature, fully-featured pure-Rust reader, so this is a pragmatic
//! importer built on `fbxcel-dom`'s v7400 DOM:
//!
//!   * **Geometry** — control points become vertices (one per control point, so
//!     skin-cluster indices map straight onto them); polygons are fan-triangulated;
//!     normals are recomputed from the triangles (FBX stores them per
//!     polygon-vertex, which doesn't map cleanly onto shared control points).
//!   * **Skeleton** — reconstructed from the skin clusters: each cluster's
//!     `TransformLink` gives its bone's global bind transform, and the bone
//!     hierarchy is recovered from the limb-node model connections.
//!   * **Weights** — each cluster's `Indexes`/`Weights` arrays are splatted onto
//!     the matching control-point vertices, then normalised.
//!
//! Limitations (best-effort): only the first mesh is imported, axis/unit
//! conversion from `GlobalSettings` is not applied, and bones not referenced by
//! any cluster are omitted. This path is **unverified against real FBX fixtures**
//! in CI and is gated behind the opt-in `fbx-import` feature.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use fbxcel_dom::any::AnyDocument;
use fbxcel_dom::v7400::object::{model, TypedObjectHandle};
use glam::Mat4;

use crate::mesh::{Influence, RigVertex, SkinnedMesh, VertexInfluences, MAX_INFLUENCES};
use crate::skeleton::{Bone, Skeleton};

use super::{ImportError, Imported};

pub fn import_fbx(path: &Path) -> Result<Imported, ImportError> {
    let file = File::open(path).map_err(|e| ImportError::Io(e.to_string()))?;
    let reader = BufReader::new(file);
    let doc = match AnyDocument::from_seekable_reader(reader)
        .map_err(|e| ImportError::Parse(e.to_string()))?
    {
        AnyDocument::V7400(_ver, doc) => doc,
        _ => return Err(ImportError::UnsupportedFormat("unsupported FBX version".into())),
    };

    // Find the first mesh geometry.
    let mesh_geom = doc
        .objects()
        .filter_map(|obj| match obj.get_typed() {
            TypedObjectHandle::Geometry(
                fbxcel_dom::v7400::object::geometry::TypedGeometryHandle::Mesh(m),
            ) => Some(m),
            _ => None,
        })
        .next()
        .ok_or(ImportError::NoGeometry)?;

    let mut mesh = read_geometry(&mesh_geom)?;
    let skeleton = read_skin(&mesh_geom, &mut mesh);

    Ok(Imported {
        mesh,
        skeleton,
        source: path.display().to_string(),
    })
}

/// Read control points + fan-triangulated indices, then recompute normals.
fn read_geometry(
    mesh: &fbxcel_dom::v7400::object::geometry::MeshHandle,
) -> Result<SkinnedMesh, ImportError> {
    let pv = mesh
        .polygon_vertices()
        .map_err(|e| ImportError::Parse(e.to_string()))?;

    // Vertices = control points (so cluster indices line up directly).
    let mut vertices: Vec<RigVertex> = Vec::new();
    for p in pv
        .raw_control_points()
        .map_err(|e| ImportError::Parse(e.to_string()))?
    {
        vertices.push(RigVertex {
            position: [p.x as f32, p.y as f32, p.z as f32],
            normal: [0.0, 0.0, 0.0],
            uv: [0.0, 0.0],
        });
    }
    if vertices.is_empty() {
        return Err(ImportError::NoGeometry);
    }

    // Fan-triangulate polygons; collect control-point indices per triangle.
    let tris = pv
        .triangulate_each(|_pvs, poly, out| {
            for i in 1..poly.len().saturating_sub(1) {
                out.push([poly[0], poly[i], poly[i + 1]]);
            }
            Ok(())
        })
        .map_err(|e| ImportError::Parse(e.to_string()))?;

    let mut indices: Vec<u32> = Vec::new();
    for tvi in tris.triangle_vertex_indices() {
        if let Some(cpi) = tris.control_point_index(tvi) {
            indices.push(cpi.to_u32());
        }
    }

    let mut out = SkinnedMesh::new(vertices, indices);
    super::obj::recompute_normals(&mut out);
    Ok(out)
}

/// Reconstruct the skeleton from the mesh's first skin and splat its cluster
/// weights onto the mesh vertices. Returns `None` if there's no skin.
fn read_skin(
    mesh: &fbxcel_dom::v7400::object::geometry::MeshHandle,
    out: &mut SkinnedMesh,
) -> Option<Skeleton> {
    let skin = mesh.skins().next()?;

    // Gather one record per cluster: limb id, name, global bind, and the
    // (control-point index, weight) pairs.
    struct ClusterRec {
        limb_id: i64,
        name: String,
        global: Mat4,
        parent_id: Option<i64>,
        weights: Vec<(u32, f32)>,
    }

    impl HasParent for ClusterRec {
        fn id(&self) -> i64 {
            self.limb_id
        }
        fn parent(&self) -> Option<i64> {
            self.parent_id
        }
    }

    let mut recs: Vec<ClusterRec> = Vec::new();
    for cluster in skin.clusters() {
        // The limb node (bone) this cluster drives.
        let limb = cluster
            .source_objects()
            .filter(|c| c.label().is_none())
            .filter_map(|c| c.object_handle())
            .find_map(|o| match o.get_typed() {
                TypedObjectHandle::Model(model::TypedModelHandle::LimbNode(l)) => Some(l),
                _ => None,
            });
        let Some(limb) = limb else { continue };
        let limb_id = limb.object_id().raw();
        let name = limb.name().unwrap_or("bone").to_string();

        // Parent limb (best-effort): a destination object that is a LimbNode.
        let parent_id = limb
            .destination_objects()
            .filter(|c| c.label().is_none())
            .filter_map(|c| c.object_handle())
            .find_map(|o| match o.get_typed() {
                TypedObjectHandle::Model(model::TypedModelHandle::LimbNode(p)) => {
                    Some(p.object_id().raw())
                }
                _ => None,
            });

        let node = cluster.node();
        let global = node
            .first_child_by_name("TransformLink")
            .and_then(|n| n.attributes().first().and_then(|a| a.get_arr_f64()))
            .map(mat4_from_slice)
            .unwrap_or(Mat4::IDENTITY);

        let idx = node
            .first_child_by_name("Indexes")
            .and_then(|n| n.attributes().first().and_then(|a| a.get_arr_i32()))
            .map(|s| s.to_vec())
            .unwrap_or_default();
        let wts = node
            .first_child_by_name("Weights")
            .and_then(|n| n.attributes().first().and_then(|a| a.get_arr_f64()))
            .map(|s| s.to_vec())
            .unwrap_or_default();
        let weights: Vec<(u32, f32)> = idx
            .iter()
            .zip(wts.iter())
            .filter(|(_, &w)| w > 0.0)
            .map(|(&i, &w)| (i as u32, w as f32))
            .collect();

        recs.push(ClusterRec { limb_id, name, global, parent_id, weights });
    }

    if recs.is_empty() {
        return None;
    }

    // Order bones so parents precede children, then build the skeleton.
    let order = topo_order_clusters(&recs);
    let id_to_bone: HashMap<i64, u16> =
        order.iter().enumerate().map(|(bone, &ri)| (recs[ri].limb_id, bone as u16)).collect();
    let global_of: HashMap<i64, Mat4> = recs.iter().map(|r| (r.limb_id, r.global)).collect();

    let mut skeleton = Skeleton::new();
    for &ri in &order {
        let r = &recs[ri];
        let parent_bone = r.parent_id.and_then(|p| id_to_bone.get(&p)).map(|&b| b as usize);
        let local_bind = match r.parent_id.and_then(|p| global_of.get(&p)) {
            Some(pg) => pg.inverse() * r.global,
            None => r.global,
        };
        skeleton.add(Bone::new(r.name.clone(), parent_bone, local_bind));
    }

    // Splat weights onto vertices (accumulate, then keep top-N normalised).
    let mut accum: Vec<Vec<(u16, f32)>> = vec![Vec::new(); out.vertex_count()];
    for r in &recs {
        let bone = id_to_bone[&r.limb_id];
        for &(cp, w) in &r.weights {
            if let Some(slot) = accum.get_mut(cp as usize) {
                slot.push((bone, w));
            }
        }
    }
    for (vi, mut list) in accum.into_iter().enumerate() {
        if list.is_empty() {
            continue;
        }
        list.sort_by(|a, b| b.1.total_cmp(&a.1));
        list.truncate(MAX_INFLUENCES);
        let mut inf: VertexInfluences = [Influence::NONE; MAX_INFLUENCES];
        for (slot, (bone, weight)) in inf.iter_mut().zip(list) {
            *slot = Influence { bone, weight };
        }
        crate::mesh::normalize_one(&mut inf);
        out.influences[vi] = inf;
    }

    Some(skeleton)
}

/// Topologically order cluster records so each parent precedes its children.
fn topo_order_clusters(recs: &[impl HasParent]) -> Vec<usize> {
    let id_to_idx: HashMap<i64, usize> = recs.iter().enumerate().map(|(i, r)| (r.id(), i)).collect();
    let mut placed = vec![false; recs.len()];
    let mut order = Vec::with_capacity(recs.len());
    while order.len() < recs.len() {
        let before = order.len();
        for (i, r) in recs.iter().enumerate() {
            if placed[i] {
                continue;
            }
            let ready = match r.parent().and_then(|p| id_to_idx.get(&p)) {
                Some(&pi) => placed[pi],
                None => true, // root, or parent not in our set
            };
            if ready {
                order.push(i);
                placed[i] = true;
            }
        }
        if order.len() == before {
            for (i, _) in recs.iter().enumerate() {
                if !placed[i] {
                    order.push(i);
                    placed[i] = true;
                }
            }
        }
    }
    order
}

/// Minimal interface so the topo sort can be unit-tested without the FBX DOM.
trait HasParent {
    fn id(&self) -> i64;
    fn parent(&self) -> Option<i64>;
}

fn mat4_from_slice(s: &[f64]) -> Mat4 {
    let mut a = [0.0f32; 16];
    for (i, v) in s.iter().take(16).enumerate() {
        a[i] = *v as f32;
    }
    Mat4::from_cols_array(&a)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rec {
        id: i64,
        parent: Option<i64>,
    }
    impl HasParent for Rec {
        fn id(&self) -> i64 {
            self.id
        }
        fn parent(&self) -> Option<i64> {
            self.parent
        }
    }

    #[test]
    fn topo_orders_parents_first() {
        // child (10) listed before its parent (20)
        let recs = vec![
            Rec { id: 10, parent: Some(20) },
            Rec { id: 20, parent: None },
            Rec { id: 30, parent: Some(10) },
        ];
        let order = topo_order_clusters(&recs);
        let pos = |id: i64| order.iter().position(|&i| recs[i].id == id).unwrap();
        assert!(pos(20) < pos(10));
        assert!(pos(10) < pos(30));
    }

    #[test]
    fn mat4_from_slice_reads_columns() {
        let id: Vec<f64> = Mat4::IDENTITY.to_cols_array().iter().map(|f| *f as f64).collect();
        assert!(mat4_from_slice(&id).abs_diff_eq(Mat4::IDENTITY, 1e-6));
    }
}
