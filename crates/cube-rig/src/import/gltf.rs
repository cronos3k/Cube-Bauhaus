//! glTF 2.0 / GLB import: geometry, skeleton and (if present) skin weights.
//!
//! The skeleton is rebuilt from the skin's joint nodes. Rather than trusting the
//! joint array's order or assuming joints are direct parents of each other, we
//! compute every joint's *global* bind transform by walking the full node tree,
//! then derive each bone's parent-relative `local_bind` from its nearest
//! ancestor joint. This stays correct even when non-joint nodes sit between
//! joints, and we topologically sort so parents always precede children.

use std::collections::HashMap;
use std::path::Path;

use glam::Mat4;

use crate::mesh::{Influence, RigVertex, SkinnedMesh, VertexInfluences, MAX_INFLUENCES};
use crate::skeleton::{Bone, Skeleton};

use super::{append_mesh, ImportError, Imported};

pub fn import_gltf(path: &Path) -> Result<Imported, ImportError> {
    let (doc, buffers, _images) =
        gltf::import(path).map_err(|e| ImportError::Parse(e.to_string()))?;

    // Local transform of every node, indexed by node index.
    let node_count = doc.nodes().count();
    let mut local = vec![Mat4::IDENTITY; node_count];
    let mut parent = vec![None; node_count];
    for node in doc.nodes() {
        local[node.index()] = Mat4::from_cols_array_2d(&node.transform().matrix());
        for child in node.children() {
            parent[child.index()] = Some(node.index());
        }
    }
    let global_of = |mut n: usize| -> Mat4 {
        let mut m = local[n];
        while let Some(p) = parent[n] {
            m = local[p] * m;
            n = p;
        }
        m
    };

    // Build the skeleton from the first skin, if any, plus a node→bone map for
    // remapping per-vertex joint indices.
    let (skeleton, orig_joint_to_bone) = match doc.skins().next() {
        Some(skin) => build_skeleton(&skin, &parent, &global_of),
        None => (None, HashMap::new()),
    };

    // Merge all mesh primitives into a single skinned mesh.
    let mut mesh = SkinnedMesh::default();
    for gmesh in doc.meshes() {
        for prim in gmesh.primitives() {
            let part = read_primitive(&prim, &buffers, &orig_joint_to_bone)?;
            append_mesh(&mut mesh, part);
        }
    }

    if mesh.vertices.is_empty() {
        return Err(ImportError::NoGeometry);
    }

    Ok(Imported {
        mesh,
        skeleton,
        source: path.display().to_string(),
    })
}

/// Load *only* the skeleton from a glTF/GLB file (for a standalone skeleton
/// file that may carry no mesh). Returns `Ok(None)` if the file has no skin.
pub fn import_gltf_skeleton(path: &Path) -> Result<Option<Skeleton>, ImportError> {
    let (doc, _buffers, _images) =
        gltf::import(path).map_err(|e| ImportError::Parse(e.to_string()))?;

    let node_count = doc.nodes().count();
    let mut local = vec![Mat4::IDENTITY; node_count];
    let mut parent = vec![None; node_count];
    for node in doc.nodes() {
        local[node.index()] = Mat4::from_cols_array_2d(&node.transform().matrix());
        for child in node.children() {
            parent[child.index()] = Some(node.index());
        }
    }
    let global_of = |mut n: usize| -> Mat4 {
        let mut m = local[n];
        while let Some(p) = parent[n] {
            m = local[p] * m;
            n = p;
        }
        m
    };

    match doc.skins().next() {
        Some(skin) => Ok(build_skeleton(&skin, &parent, &global_of).0),
        None => Ok(None),
    }
}

/// Build a [`Skeleton`] from a glTF skin. Returns the skeleton and a map from
/// the skin's original joint-array index to the final bone index (for remapping
/// vertex joint references).
fn build_skeleton(
    skin: &gltf::Skin,
    parent: &[Option<usize>],
    global_of: &impl Fn(usize) -> Mat4,
) -> (Option<Skeleton>, HashMap<u16, u16>) {
    let joint_nodes: Vec<usize> = skin.joints().map(|n| n.index()).collect();
    if joint_nodes.is_empty() {
        return (None, HashMap::new());
    }
    let joint_set: HashMap<usize, ()> = joint_nodes.iter().map(|&n| (n, ())).collect();

    // Nearest ancestor node that is itself a joint.
    let parent_joint = |node: usize| -> Option<usize> {
        let mut p = parent[node];
        while let Some(pp) = p {
            if joint_set.contains_key(&pp) {
                return Some(pp);
            }
            p = parent[pp];
        }
        None
    };

    // Topologically order joint nodes (parents before children).
    let order = topo_order(&joint_nodes, &parent_joint);
    let node_to_bone: HashMap<usize, u16> =
        order.iter().enumerate().map(|(i, &n)| (n, i as u16)).collect();

    let mut skeleton = Skeleton::new();
    for (bi, &node) in order.iter().enumerate() {
        let name = skin
            .joints()
            .nth(joint_nodes.iter().position(|&n| n == node).unwrap())
            .and_then(|n| n.name().map(str::to_owned))
            .unwrap_or_else(|| format!("bone{bi}"));
        let global = global_of(node);
        let local_bind = match parent_joint(node) {
            Some(pj) => global_of(pj).inverse() * global,
            None => global,
        };
        let parent_bone = parent_joint(node).map(|pj| node_to_bone[&pj] as usize);
        skeleton.add(Bone::new(name, parent_bone, local_bind));
    }

    // Map original joint-array index → final bone index.
    let mut orig_to_bone = HashMap::new();
    for (orig, &node) in joint_nodes.iter().enumerate() {
        orig_to_bone.insert(orig as u16, node_to_bone[&node]);
    }

    (Some(skeleton), orig_to_bone)
}

/// Order nodes so each parent precedes its children (Kahn-style, stable).
fn topo_order(nodes: &[usize], parent_joint: &impl Fn(usize) -> Option<usize>) -> Vec<usize> {
    let in_set: HashMap<usize, ()> = nodes.iter().map(|&n| (n, ())).collect();
    let mut placed: HashMap<usize, ()> = HashMap::new();
    let mut order = Vec::with_capacity(nodes.len());
    // Repeatedly place any node whose parent joint is absent or already placed.
    while order.len() < nodes.len() {
        let before = order.len();
        for &n in nodes {
            if placed.contains_key(&n) {
                continue;
            }
            let ready = match parent_joint(n) {
                Some(p) => !in_set.contains_key(&p) || placed.contains_key(&p),
                None => true,
            };
            if ready {
                order.push(n);
                placed.insert(n, ());
            }
        }
        if order.len() == before {
            // Cycle (shouldn't happen in valid glTF) — append the rest as-is.
            for &n in nodes {
                if !placed.contains_key(&n) {
                    order.push(n);
                    placed.insert(n, ());
                }
            }
        }
    }
    order
}

fn read_primitive(
    prim: &gltf::Primitive,
    buffers: &[gltf::buffer::Data],
    orig_joint_to_bone: &HashMap<u16, u16>,
) -> Result<SkinnedMesh, ImportError> {
    let reader = prim.reader(|b| buffers.get(b.index()).map(|d| d.0.as_slice()));

    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .ok_or(ImportError::NoGeometry)?
        .collect();
    let vcount = positions.len();

    let normals: Vec<[f32; 3]> = reader
        .read_normals()
        .map(|it| it.collect())
        .unwrap_or_else(|| vec![[0.0, 0.0, 0.0]; vcount]);

    let uvs: Vec<[f32; 2]> = reader
        .read_tex_coords(0)
        .map(|tc| tc.into_f32().collect())
        .unwrap_or_else(|| vec![[0.0, 0.0]; vcount]);

    let mut verts = Vec::with_capacity(vcount);
    for i in 0..vcount {
        verts.push(RigVertex {
            position: positions[i],
            normal: *normals.get(i).unwrap_or(&[0.0, 0.0, 0.0]),
            uv: *uvs.get(i).unwrap_or(&[0.0, 0.0]),
        });
    }

    let indices: Vec<u32> = match reader.read_indices() {
        Some(idx) => idx.into_u32().collect(),
        None => (0..vcount as u32).collect(),
    };

    let mut mesh = SkinnedMesh::new(verts, indices);

    // Skin weights, if present.
    if let (Some(joints), Some(weights)) = (reader.read_joints(0), reader.read_weights(0)) {
        let joints: Vec<[u16; 4]> = joints.into_u16().collect();
        let weights: Vec<[f32; 4]> = weights.into_f32().collect();
        for (vi, (js, ws)) in joints.into_iter().zip(weights).enumerate().take(vcount) {
            let mut inf: VertexInfluences = [Influence::NONE; MAX_INFLUENCES];
            for k in 0..MAX_INFLUENCES {
                if ws[k] > 0.0 {
                    let bone = orig_joint_to_bone.get(&js[k]).copied().unwrap_or(js[k]);
                    inf[k] = Influence { bone, weight: ws[k] };
                }
            }
            crate::mesh::normalize_one(&mut inf);
            mesh.influences[vi] = inf;
        }
    }

    Ok(mesh)
}
