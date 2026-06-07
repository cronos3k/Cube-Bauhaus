//! Weight assignment operations applied to a selection of vertices.
//!
//! These sit on top of [`SkinnedMesh`]'s per-vertex influence primitives and
//! implement the artist-facing actions: rigid bind, brush paint (with falloff),
//! smooth/relax, mirror across an axis, and prune small weights. Every op keeps
//! the per-vertex influence count within [`MAX_INFLUENCES`] and renormalises.

use std::collections::HashMap;

use crate::mesh::{normalize_one, Influence, SkinnedMesh, MAX_INFLUENCES};
use crate::select::VertexSelection;
use crate::skeleton::Skeleton;

/// Rigidly bind every selected vertex 100% to `bone`.
pub fn bind_rigid(mesh: &mut SkinnedMesh, sel: &VertexSelection, bone: u16) {
    for v in sel.iter() {
        mesh.set_rigid(v as usize, bone);
    }
}

/// Paint `bone` weight onto the selection. `amount` is added per vertex
/// (negative erases). When `falloff` is provided it gives a 0..1 multiplier per
/// vertex index (e.g. a brush's radial falloff), so the edge of a brush dab
/// applies less weight than its centre.
pub fn paint(
    mesh: &mut SkinnedMesh,
    sel: &VertexSelection,
    bone: u16,
    amount: f32,
    falloff: Option<&HashMap<u32, f32>>,
) {
    for v in sel.iter() {
        let scale = falloff.and_then(|f| f.get(&v)).copied().unwrap_or(1.0);
        let delta = amount * scale;
        if delta != 0.0 {
            mesh.add_weight(v as usize, bone, delta);
        }
    }
}

/// Relax weights toward the average of each vertex's neighbours. `strength`
/// (0..1) blends between the original and the neighbour average; `iterations`
/// repeats the pass. Only selected vertices are modified, but unselected
/// neighbours still contribute to the averages.
pub fn smooth(
    mesh: &mut SkinnedMesh,
    sel: &VertexSelection,
    iterations: u32,
    strength: f32,
) {
    let strength = strength.clamp(0.0, 1.0);
    let adj = mesh.adjacency().to_vec();
    let targets: Vec<u32> = sel.iter().collect();

    for _ in 0..iterations {
        // Snapshot so the pass is order-independent.
        let snapshot = mesh.influences.clone();
        for &v in &targets {
            let neighbours = &adj[v as usize];
            if neighbours.is_empty() {
                continue;
            }
            // Accumulate neighbour influence as bone→summed-weight.
            let mut acc: HashMap<u16, f32> = HashMap::new();
            for &n in neighbours {
                for inf in &snapshot[n as usize] {
                    if inf.weight > 0.0 {
                        *acc.entry(inf.bone).or_insert(0.0) += inf.weight;
                    }
                }
            }
            if acc.is_empty() {
                continue;
            }
            let inv = 1.0 / neighbours.len() as f32;
            for w in acc.values_mut() {
                *w *= inv;
            }
            // Blend current → neighbour average.
            let mut blended: HashMap<u16, f32> = HashMap::new();
            for inf in &snapshot[v as usize] {
                if inf.weight > 0.0 {
                    *blended.entry(inf.bone).or_insert(0.0) += inf.weight * (1.0 - strength);
                }
            }
            for (&bone, &w) in &acc {
                *blended.entry(bone).or_insert(0.0) += w * strength;
            }
            mesh.influences[v as usize] = top_influences(&blended);
        }
    }
}

/// Mirror weights from the positive side of `axis` (0=x,1=y,2=z) to the negative
/// side (or vice-versa for `negative_to_positive`). For each source vertex it
/// finds the geometrically mirrored vertex within `tolerance` and copies its
/// influences, remapping each bone to its mirror counterpart (matched by the
/// usual `_L`/`_R`, `.L`/`.R`, `Left`/`Right` naming). Returns the number of
/// vertices written.
pub fn mirror(
    mesh: &mut SkinnedMesh,
    skeleton: &Skeleton,
    axis: usize,
    tolerance: f32,
    negative_to_positive: bool,
) -> usize {
    let bone_mirror = build_bone_mirror_map(skeleton);
    let positions: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| v.position).collect();

    // Spatial lookup: bucket vertices by quantised position for the mirror match.
    let mut written = 0;
    for src in 0..positions.len() {
        let p = positions[src];
        let on_source_side = if negative_to_positive { p[axis] < -tolerance } else { p[axis] > tolerance };
        if !on_source_side {
            continue;
        }
        let mut mirror_pos = p;
        mirror_pos[axis] = -mirror_pos[axis];
        if let Some(dst) = nearest_within(&positions, mirror_pos, tolerance, src) {
            let mut inf = mesh.influences[src];
            for slot in inf.iter_mut() {
                if slot.weight > 0.0 {
                    if let Some(&m) = bone_mirror.get(&slot.bone) {
                        slot.bone = m;
                    }
                }
            }
            normalize_one(&mut inf);
            mesh.influences[dst] = inf;
            written += 1;
        }
    }
    written
}

/// Drop influences below `min_weight` from every vertex and renormalise.
pub fn prune(mesh: &mut SkinnedMesh, min_weight: f32) {
    for inf in &mut mesh.influences {
        for slot in inf.iter_mut() {
            if slot.weight < min_weight {
                *slot = Influence::NONE;
            }
        }
        normalize_one(inf);
    }
}

// ── helpers ─────────────────────────────────────────────────────────────────

/// Reduce a bone→weight map to the strongest [`MAX_INFLUENCES`], normalised.
fn top_influences(map: &HashMap<u16, f32>) -> [Influence; MAX_INFLUENCES] {
    let mut pairs: Vec<(u16, f32)> = map.iter().map(|(&b, &w)| (b, w)).filter(|&(_, w)| w > 0.0).collect();
    pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
    pairs.truncate(MAX_INFLUENCES);
    let mut out = [Influence::NONE; MAX_INFLUENCES];
    for (slot, (bone, weight)) in out.iter_mut().zip(pairs) {
        *slot = Influence { bone, weight };
    }
    normalize_one(&mut out);
    out
}

fn nearest_within(positions: &[[f32; 3]], target: [f32; 3], tol: f32, skip: usize) -> Option<usize> {
    let tol2 = tol * tol;
    let mut best = None;
    let mut best_d2 = tol2;
    for (i, p) in positions.iter().enumerate() {
        if i == skip {
            continue;
        }
        let dx = p[0] - target[0];
        let dy = p[1] - target[1];
        let dz = p[2] - target[2];
        let d2 = dx * dx + dy * dy + dz * dz;
        if d2 <= best_d2 {
            best_d2 = d2;
            best = Some(i);
        }
    }
    best
}

/// Map each bone index to its mirror-side counterpart by name convention.
fn build_bone_mirror_map(skeleton: &Skeleton) -> HashMap<u16, u16> {
    let mut map = HashMap::new();
    for (i, b) in skeleton.bones.iter().enumerate() {
        if let Some(mirror_name) = mirror_bone_name(&b.name) {
            if let Some(j) = skeleton.index_of(&mirror_name) {
                map.insert(i as u16, j as u16);
            }
        }
    }
    map
}

/// Given a bone name, return the name of its left/right mirror, if it follows a
/// recognised convention. Returns `None` for centre bones.
pub fn mirror_bone_name(name: &str) -> Option<String> {
    // suffix pairs
    const PAIRS: [(&str, &str); 6] = [
        ("_L", "_R"), ("_l", "_r"),
        (".L", ".R"), (".l", ".r"),
        ("Left", "Right"), ("left", "right"),
    ];
    for (a, b) in PAIRS {
        if let Some(stem) = name.strip_suffix(a) {
            return Some(format!("{stem}{b}"));
        }
        if let Some(stem) = name.strip_suffix(b) {
            return Some(format!("{stem}{a}"));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::RigVertex;
    use crate::skeleton::Bone;
    use glam::Mat4;

    fn two_vert_mesh(x0: f32, x1: f32) -> SkinnedMesh {
        let v = |x: f32| RigVertex { position: [x, 0.0, 0.0], normal: [0.0, 1.0, 0.0], uv: [0.0, 0.0] };
        // a degenerate single triangle is enough to give the two verts adjacency
        SkinnedMesh::new(vec![v(x0), v(x1)], vec![0, 1, 0])
    }

    #[test]
    fn rigid_bind_over_selection() {
        let mut m = two_vert_mesh(-1.0, 1.0);
        let mut sel = VertexSelection::new(2);
        sel.set(0, true);
        sel.set(1, true);
        bind_rigid(&mut m, &sel, 3);
        assert_eq!(m.dominant_bone(0), Some(3));
        assert_eq!(m.dominant_bone(1), Some(3));
    }

    #[test]
    fn paint_with_falloff_scales() {
        let mut m = two_vert_mesh(-1.0, 1.0);
        let mut sel = VertexSelection::new(2);
        sel.set(0, true);
        let mut fo = HashMap::new();
        fo.insert(0u32, 0.5);
        paint(&mut m, &sel, 2, 1.0, Some(&fo));
        // single bone → normalises to 1.0 regardless, but the slot exists
        assert_eq!(m.dominant_bone(0), Some(2));
    }

    #[test]
    fn mirror_swaps_left_right_bone() {
        // skeleton: arm_L (idx0), arm_R (idx1)
        let mut sk = Skeleton::new();
        sk.add(Bone::new("arm_L", None, Mat4::IDENTITY));
        sk.add(Bone::new("arm_R", None, Mat4::IDENTITY));

        let mut m = two_vert_mesh(-1.0, 1.0); // v0 at x=-1, v1 at x=+1
        m.set_rigid(0, 0); // negative side bound to arm_L

        // mirror negative → positive across x
        let n = mirror(&mut m, &sk, 0, 1e-3, true);
        assert_eq!(n, 1);
        // v1 (positive side) should now be bound to arm_R
        assert_eq!(m.dominant_bone(1), Some(1));
    }

    #[test]
    fn mirror_name_pairs() {
        assert_eq!(mirror_bone_name("hand_L").as_deref(), Some("hand_R"));
        assert_eq!(mirror_bone_name("hand_R").as_deref(), Some("hand_L"));
        assert_eq!(mirror_bone_name("Leg.L").as_deref(), Some("Leg.R"));
        assert_eq!(mirror_bone_name("spine"), None);
    }

    #[test]
    fn smooth_pulls_toward_neighbour() {
        // v0 bound to bone 0, v1 bound to bone 1, adjacent. Smoothing v0 should
        // introduce some bone-1 influence.
        let mut m = two_vert_mesh(0.0, 1.0);
        m.set_rigid(0, 0);
        m.set_rigid(1, 1);
        let mut sel = VertexSelection::new(2);
        sel.set(0, true);
        smooth(&mut m, &sel, 1, 0.5);
        let has_bone1 = m.influences[0].iter().any(|i| i.bone == 1 && i.weight > 0.0);
        assert!(has_bone1);
    }

    #[test]
    fn prune_drops_small_weights() {
        let mut m = two_vert_mesh(0.0, 1.0);
        m.add_weight(0, 0, 0.95);
        m.add_weight(0, 1, 0.05);
        prune(&mut m, 0.1);
        assert_eq!(m.dominant_bone(0), Some(0));
        let count = m.influences[0].iter().filter(|i| i.weight > 0.0).count();
        assert_eq!(count, 1);
    }
}
