//! glTF 2.0 / GLB import: geometry, skeleton and (if present) skin weights.
//!
//! The skeleton is rebuilt from the skin's joint nodes. Rather than trusting the
//! joint array's order or assuming joints are direct parents of each other, we
//! compute every joint's *global* bind transform by walking the full node tree,
//! then derive each bone's parent-relative `local_bind` from its nearest
//! ancestor joint. This stays correct even when non-joint nodes sit between
//! joints, and we topologically sort so parents always precede children.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use glam::{Mat4, Quat, Vec3};

use crate::anim::{AnimationClip, BoneTrack};
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
    Ok(skeleton_with_node_map(&doc).map(|(sk, _)| sk))
}

/// Build the skeleton from a document's first skin and return it together with
/// the node-index → bone-index map, sharing the exact joint ordering used by
/// [`import_gltf_skeleton`] (so animation channels remap consistently).
fn skeleton_with_node_map(doc: &gltf::Document) -> Option<(Skeleton, HashMap<usize, u16>)> {
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

    let skin = doc.skins().next()?;
    let (skeleton, node_to_bone) = build_skeleton_mapped(&skin, &parent, &global_of);
    skeleton.map(|sk| (sk, node_to_bone))
}

/// Import all animations from a glTF/GLB file as [`AnimationClip`]s, keyed to the
/// same joint ordering as [`import_gltf_skeleton`].
///
/// Channels targeting non-joint nodes are skipped. All samplers are treated as
/// linear keyframe tracks (the interpolation our [`crate::anim`] sampler
/// expects); unsupported interpolation modes are read as plain keyframes.
pub fn import_gltf_animations(path: &Path) -> Result<Vec<AnimationClip>, ImportError> {
    let (doc, buffers, _images) =
        gltf::import(path).map_err(|e| ImportError::Parse(e.to_string()))?;

    // Node→bone mapping identical to the skeleton importer. If the file has no
    // skin, there are no joints to key animations against.
    let node_to_bone = match skeleton_with_node_map(&doc) {
        Some((_, map)) => map,
        None => return Ok(Vec::new()),
    };

    let mut clips = Vec::new();
    for (ai, animation) in doc.animations().enumerate() {
        let name = animation
            .name()
            .map(str::to_owned)
            .unwrap_or_else(|| format!("animation{ai}"));

        // Accumulate one BoneTrack per bone, gathering its T/R/S channels.
        let mut tracks: HashMap<u16, BoneTrack> = HashMap::new();
        for channel in animation.channels() {
            let target_node = channel.target().node().index();
            let Some(&bone) = node_to_bone.get(&target_node) else {
                continue; // channel targets a non-joint node
            };

            let reader = channel.reader(|b| Some(&buffers[b.index()].0));
            let Some(times) = reader.read_inputs() else {
                continue;
            };
            let times: Vec<f32> = times.collect();

            let Some(outputs) = reader.read_outputs() else {
                continue;
            };

            let track = tracks.entry(bone).or_insert_with(|| BoneTrack::new(bone));
            match outputs {
                gltf::animation::util::ReadOutputs::Translations(it) => {
                    track.translation = times
                        .iter()
                        .copied()
                        .zip(it.map(Vec3::from_array))
                        .collect();
                }
                gltf::animation::util::ReadOutputs::Rotations(it) => {
                    track.rotation = times
                        .iter()
                        .copied()
                        .zip(it.into_f32().map(Quat::from_array))
                        .collect();
                }
                gltf::animation::util::ReadOutputs::Scales(it) => {
                    track.scale = times
                        .iter()
                        .copied()
                        .zip(it.map(Vec3::from_array))
                        .collect();
                }
                // Morph-target weights are not part of a BoneTrack.
                gltf::animation::util::ReadOutputs::MorphTargetWeights(_) => {}
            }
        }

        // Emit tracks in stable bone order for determinism.
        let mut tracks: Vec<BoneTrack> = tracks.into_values().collect();
        tracks.sort_by_key(|t| t.bone);
        clips.push(AnimationClip { name, tracks });
    }

    Ok(clips)
}

/// Build a [`Skeleton`] from a glTF skin. Returns the skeleton and a map from
/// the skin's original joint-array index to the final bone index (for remapping
/// vertex joint references).
fn build_skeleton(
    skin: &gltf::Skin,
    parent: &[Option<usize>],
    global_of: &impl Fn(usize) -> Mat4,
) -> (Option<Skeleton>, HashMap<u16, u16>) {
    let (skeleton, node_to_bone) = build_skeleton_mapped(skin, parent, global_of);
    let orig_to_bone = if skeleton.is_some() {
        orig_joint_map(skin, &node_to_bone)
    } else {
        HashMap::new()
    };
    (skeleton, orig_to_bone)
}

/// Map each skin joint-array index → final bone index, using the node→bone map.
fn orig_joint_map(skin: &gltf::Skin, node_to_bone: &HashMap<usize, u16>) -> HashMap<u16, u16> {
    let mut orig_to_bone = HashMap::new();
    for (orig, node) in skin.joints().enumerate() {
        if let Some(&bone) = node_to_bone.get(&node.index()) {
            orig_to_bone.insert(orig as u16, bone);
        }
    }
    orig_to_bone
}

/// Build a [`Skeleton`] from a glTF skin and return the node-index → bone-index
/// map. This is the single source of truth for joint ordering shared by the
/// skeleton and animation importers.
fn build_skeleton_mapped(
    skin: &gltf::Skin,
    parent: &[Option<usize>],
    global_of: &impl Fn(usize) -> Mat4,
) -> (Option<Skeleton>, HashMap<usize, u16>) {
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

    (Some(skeleton), node_to_bone)
}

/// Order nodes so each parent precedes its children (Kahn-style, stable).
fn topo_order(nodes: &[usize], parent_joint: &impl Fn(usize) -> Option<usize>) -> Vec<usize> {
    let in_set: HashSet<usize> = nodes.iter().copied().collect();
    let mut placed: HashSet<usize> = HashSet::new();
    let mut order = Vec::with_capacity(nodes.len());
    // Repeatedly place any node whose parent joint is absent or already placed.
    while order.len() < nodes.len() {
        let before = order.len();
        for &n in nodes {
            if placed.contains(&n) {
                continue;
            }
            let ready = match parent_joint(n) {
                Some(p) => !in_set.contains(&p) || placed.contains(&p),
                None => true,
            };
            if ready {
                order.push(n);
                placed.insert(n);
            }
        }
        if order.len() == before {
            // Cycle (shouldn't happen in valid glTF) — append the rest as-is.
            for &n in nodes {
                if placed.insert(n) {
                    order.push(n);
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
    for (i, &position) in positions.iter().enumerate() {
        verts.push(RigVertex {
            position,
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
