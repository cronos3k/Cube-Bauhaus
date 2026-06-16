//! Per-bone capsule collision (§8).
//!
//! Each bone is approximated by a [`Capsule`] — a swept sphere along the segment
//! from the bone's head to its first child's head. This module provides three
//! layers built on that primitive:
//!
//!   * the geometry core — [`closest_segment_points`], [`capsule_distance`] and
//!     [`capsule_penetration`] — exact, engine-agnostic, and heavily tested;
//!   * a pose query — [`bone_capsules`], [`self_collisions`] and
//!     [`is_self_colliding`] — which lift the core onto a [`Pose`] and report
//!     overlapping non-adjacent bone pairs;
//!   * a resolver — [`avoid_obstacles`] — which rotates an articulated chain out
//!     of static obstacle capsules using the same weighted damped-least-squares
//!     step that drives [`solve_dls`](crate::ik::solve_dls).
//!
//! The resolver targets the canonical "arm vs torso" case: the torso is a set of
//! static capsules and the arm chain is pushed out of them by joint rotation.

use glam::{Mat3, Mat4, Quat, Vec3};

use crate::ik::Pose;
use crate::skeleton::Skeleton;

/// A capsule: the set of points within `radius` of the segment `a..b`.
#[derive(Debug, Clone, Copy)]
pub struct Capsule {
    /// Segment start.
    pub a: Vec3,
    /// Segment end.
    pub b: Vec3,
    /// Sweep radius.
    pub radius: f32,
}

/// A penetration contact between two capsules.
#[derive(Debug, Clone, Copy)]
pub struct Contact {
    /// Unit vector off capsule A's surface toward the free space on B's side
    /// (i.e. the direction to push A to separate the pair).
    pub normal: Vec3,
    /// Penetration depth: `(a.radius + b.radius) - segment_distance`, `> 0`.
    pub depth: f32,
    /// A point roughly midway inside the overlap region.
    pub point: Vec3,
}

/// Closest points between two segments `p1..q1` and `p2..q2` (Ericson, *Real-Time
/// Collision Detection*, §5.1.9). Returns `(c1, c2, distance)` where `c1` lies on
/// the first segment, `c2` on the second, and `distance == (c2 - c1).length()`.
///
/// Degenerate cases (either segment a point, parallel segments) are handled with
/// the standard epsilon guards.
pub fn closest_segment_points(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (Vec3, Vec3, f32) {
    const EPS: f32 = 1e-10;
    let d1 = q1 - p1; // direction of segment 1
    let d2 = q2 - p2; // direction of segment 2
    let r = p1 - p2;
    let a = d1.dot(d1); // squared length of segment 1
    let e = d2.dot(d2); // squared length of segment 2
    let f = d2.dot(r);

    let (mut s, mut t);
    if a <= EPS && e <= EPS {
        // Both segments degenerate to points.
        s = 0.0;
        t = 0.0;
    } else if a <= EPS {
        // First segment is a point.
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e <= EPS {
            // Second segment is a point.
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            // General non-degenerate case.
            let b = d1.dot(d2);
            let denom = a * e - b * b; // always >= 0
            // If segments are not parallel, compute closest point on L1 to L2 and
            // clamp to segment 1. Otherwise pick s = 0.
            s = if denom > EPS {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            // Closest point on segment 2 to s, clamped.
            t = (b * s + f) / e;
            // If t fell outside [0,1], clamp it and recompute s for the new t.
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
        }
    }

    let c1 = p1 + d1 * s;
    let c2 = p2 + d2 * t;
    (c1, c2, (c2 - c1).length())
}

/// Distance between two capsules' surfaces: the segment distance minus both
/// radii. Negative when the capsules overlap.
pub fn capsule_distance(a: &Capsule, b: &Capsule) -> f32 {
    let (_, _, seg) = closest_segment_points(a.a, a.b, b.a, b.b);
    seg - a.radius - b.radius
}

/// Penetration contact between two capsules, or `None` if they do not overlap.
///
/// Returns `Some` iff the segment distance is strictly less than the sum of the
/// radii. The `normal` points from `a` toward `b` (off A's surface), `depth` is
/// the overlap of the inflated radii, and `point` is the midpoint of the two
/// surface points along the normal.
pub fn capsule_penetration(a: &Capsule, b: &Capsule) -> Option<Contact> {
    let (c1, c2, seg) = closest_segment_points(a.a, a.b, b.a, b.b);
    let sum_r = a.radius + b.radius;
    if seg >= sum_r {
        return None;
    }
    let delta = c2 - c1;
    let len = delta.length();
    let normal = if len > 1e-6 {
        delta / len
    } else {
        // Centers coincide: pick a stable arbitrary axis.
        Vec3::X
    };
    let depth = sum_r - seg;
    // Surface points along the normal, then their midpoint.
    let surf_a = c1 + normal * a.radius;
    let surf_b = c2 - normal * b.radius;
    let point = (surf_a + surf_b) * 0.5;
    Some(Contact { normal, depth, point })
}

/// Per-bone capsules for a pose: bone `i`'s capsule spans from its head to the
/// head of its first child, with the given `radius`. Bones with no children, or
/// whose segment is shorter than `1e-5`, get `None`. The result is parallel to
/// `skeleton.bones`.
pub fn bone_capsules(pose: &Pose, skeleton: &Skeleton, radius: f32) -> Vec<Option<Capsule>> {
    (0..skeleton.len())
        .map(|i| {
            let children = skeleton.children(i);
            let child = match children.first() {
                Some(&c) => c,
                None => return None,
            };
            let a = pose.head(skeleton, i);
            let b = pose.head(skeleton, child);
            if (b - a).length() < 1e-5 {
                return None;
            }
            Some(Capsule { a, b, radius })
        })
        .collect()
}

/// All penetrating bone pairs `(i, j, contact)` with `i < j`, excluding adjacent
/// (direct parent/child) pairs, which legitimately touch.
pub fn self_collisions(
    pose: &Pose,
    skeleton: &Skeleton,
    radius: f32,
) -> Vec<(usize, usize, Contact)> {
    let caps = bone_capsules(pose, skeleton, radius);
    let mut out = Vec::new();
    for i in 0..caps.len() {
        let ca = match &caps[i] {
            Some(c) => c,
            None => continue,
        };
        for (j, cbj) in caps.iter().enumerate().skip(i + 1) {
            let cb = match cbj {
                Some(c) => c,
                None => continue,
            };
            // Skip direct parent/child pairs.
            if skeleton.bones[j].parent == Some(i) || skeleton.bones[i].parent == Some(j) {
                continue;
            }
            if let Some(contact) = capsule_penetration(ca, cb) {
                out.push((i, j, contact));
            }
        }
    }
    out
}

/// Whether the pose has any non-adjacent self-collision.
pub fn is_self_colliding(pose: &Pose, skeleton: &Skeleton, radius: f32) -> bool {
    !self_collisions(pose, skeleton, radius).is_empty()
}

/// Tuning parameters for [`avoid_obstacles`].
#[derive(Debug, Clone, Copy)]
pub struct AvoidParams {
    /// Maximum number of push-out iterations.
    pub iterations: usize,
    /// Step scale applied to each rotational update.
    pub step: f32,
    /// Penetrations shallower than this depth are tolerated (not resolved).
    pub margin: f32,
}

impl Default for AvoidParams {
    fn default() -> Self {
        Self { iterations: 12, step: 1.0, margin: 0.0 }
    }
}

/// Rotation part of a transform matrix as a [`Quat`].
fn rotation_of(m: Mat4) -> Quat {
    m.to_scale_rotation_translation().1
}

/// Apply a world-space delta rotation `dq` to chain joint `bone` (at chain slot
/// `slot`), converting back to local exactly as `solve_dls` does: compose with
/// the current world rotation, strip the parent's world rotation, optionally
/// clamp relative to bind against `limits[slot]`, and rebuild the local matrix
/// preserving its translation.
fn apply_world_delta(
    pose: &mut Pose,
    skeleton: &Skeleton,
    bone: usize,
    slot: usize,
    dq: Quat,
    limits: Option<&[Option<crate::limits::JointLimit>]>,
) {
    let parent_world_rot = match skeleton.bones[bone].parent {
        Some(p) => rotation_of(pose.global(skeleton, p)),
        None => Quat::IDENTITY,
    };
    let world_rot_old = rotation_of(pose.global(skeleton, bone));
    let world_rot_new = dq * world_rot_old;
    let mut local_rot_new = parent_world_rot.inverse() * world_rot_new;

    if let Some(limit) = limits.and_then(|ls| ls.get(slot)).and_then(|l| l.as_ref()) {
        let bind_rot = rotation_of(skeleton.bones[bone].local_bind);
        let rel = bind_rot.inverse() * local_rot_new;
        let rel = crate::limits::clamp(rel, limit);
        local_rot_new = bind_rot * rel;
    }

    let local_translation = pose.local[bone].w_axis.truncate();
    pose.local[bone] = Mat4::from_rotation_translation(local_rot_new, local_translation);
}

/// Count the (chain-bone, obstacle) penetrations deeper than `margin` for the
/// current pose, returning the count and the deepest one (slot in `chain`, the
/// obstacle's contact).
fn count_penetrations(
    pose: &Pose,
    skeleton: &Skeleton,
    chain: &[usize],
    radius: f32,
    obstacles: &[Capsule],
    margin: f32,
) -> (usize, Option<(usize, Contact)>) {
    let mut count = 0usize;
    let mut deepest: Option<(usize, Contact)> = None;
    for (k, &bone) in chain.iter().enumerate() {
        let children = skeleton.children(bone);
        let child = match children.first() {
            Some(&c) => c,
            None => continue,
        };
        let a = pose.head(skeleton, bone);
        let b = pose.head(skeleton, child);
        if (b - a).length() < 1e-5 {
            continue;
        }
        let cap = Capsule { a, b, radius };
        for obs in obstacles {
            if let Some(contact) = capsule_penetration(&cap, obs) {
                if contact.depth > margin {
                    count += 1;
                    if deepest.as_ref().map(|(_, c)| contact.depth > c.depth).unwrap_or(true) {
                        deepest = Some((k, contact));
                    }
                }
            }
        }
    }
    (count, deepest)
}

/// Rotate the joints in `chain` (root→tip) to push the chain's bone capsules out
/// of the static `obstacles`. Returns the number of remaining (bone, obstacle)
/// penetrations after resolution. `limits[i]` optionally clamps chain joint `i`.
///
/// Each iteration finds the deepest penetration and takes one weighted
/// damped-least-squares step (the same construction as
/// [`solve_dls`](crate::ik::solve_dls)) that drives the contact point out along
/// the contact normal, rotating the ancestor-or-self joints `chain[0..=k]`.
pub fn avoid_obstacles(
    pose: &mut Pose,
    skeleton: &Skeleton,
    chain: &[usize],
    radius: f32,
    obstacles: &[Capsule],
    limits: Option<&[Option<crate::limits::JointLimit>]>,
    params: &AvoidParams,
) -> usize {
    if chain.is_empty() || obstacles.is_empty() {
        return count_penetrations(pose, skeleton, chain, radius, obstacles, params.margin).0;
    }

    let lambda = 0.5_f32;
    let lambda2 = lambda * lambda;
    let axes = [Vec3::X, Vec3::Y, Vec3::Z];

    for _ in 0..params.iterations {
        let (_, deepest) =
            count_penetrations(pose, skeleton, chain, radius, obstacles, params.margin);
        let (k, contact) = match deepest {
            Some(d) => d,
            None => break,
        };

        // Effector point (the colliding material) and the desired displacement.
        // The contact `normal` points from the chain bone (capsule A) toward the
        // obstacle (capsule B); to *separate* the chain we push it the opposite
        // way, along `-normal`, by the penetration depth plus margin.
        let pt = contact.point;
        let err = -contact.normal * (contact.depth + params.margin);

        // Jacobian columns for each ancestor-or-self joint chain[0..=k].
        let mut columns: Vec<[Vec3; 3]> = Vec::with_capacity(k + 1);
        for &j in &chain[..=k] {
            let p_j = pose.head(skeleton, j);
            columns.push([
                axes[0].cross(pt - p_j),
                axes[1].cross(pt - p_j),
                axes[2].cross(pt - p_j),
            ]);
        }

        // A = Σ (c ⊗ c) + λ² I.
        let mut a = Mat3::IDENTITY * lambda2;
        for cols in &columns {
            for c in cols {
                a += Mat3::from_cols(*c * c.x, *c * c.y, *c * c.z);
            }
        }
        let a_inv = a.inverse();
        let bvec = a_inv * err;

        // Per-joint DOF increments, applied root→tip so children see updated
        // parents within this pass.
        for (slot_in_sub, &j) in chain[..=k].iter().enumerate() {
            let cols = &columns[slot_in_sub];
            let dtheta = Vec3::new(cols[0].dot(bvec), cols[1].dot(bvec), cols[2].dot(bvec));
            let dq = Quat::from_scaled_axis(params.step * dtheta);
            apply_world_delta(pose, skeleton, j, slot_in_sub, dq, limits);
        }
    }

    count_penetrations(pose, skeleton, chain, radius, obstacles, params.margin).0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::Bone;
    use approx::assert_relative_eq;

    #[test]
    fn closest_points_crossing_perpendicular_segments() {
        // Segment 1 along X through origin, segment 2 along Y crossing it at x=0.
        let (c1, c2, dist) = closest_segment_points(
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        );
        assert_relative_eq!(dist, 0.0, epsilon = 1e-5);
        assert_relative_eq!((c1 - c2).length(), 0.0, epsilon = 1e-5);
        assert_relative_eq!(c1.x, 0.0, epsilon = 1e-5);
        assert_relative_eq!(c2.y, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn closest_points_parallel_offset_segments() {
        // Two parallel segments along X offset by 2 on Y → distance == offset.
        let (_, _, dist) = closest_segment_points(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(2.0, 2.0, 0.0),
        );
        assert_relative_eq!(dist, 2.0, epsilon = 1e-5);
    }

    #[test]
    fn closest_points_zero_length_segment_vs_segment() {
        // A point at (1, 3, 0) vs a segment along X from 0..2 → nearest is (1,0,0).
        let p = Vec3::new(1.0, 3.0, 0.0);
        let (c1, c2, dist) = closest_segment_points(
            p,
            p,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        );
        assert_relative_eq!(dist, 3.0, epsilon = 1e-5);
        assert_relative_eq!(c1.x, 1.0, epsilon = 1e-5);
        assert_relative_eq!(c2.x, 1.0, epsilon = 1e-5);
        assert_relative_eq!(c2.y, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn capsule_penetration_overlapping_parallel() {
        // Two parallel capsules along X, offset 1.0 on Y, each radius 0.75.
        // Segment gap = 1.0, sum radii = 1.5 → depth ≈ 0.5, normal ≈ +Y.
        let a = Capsule { a: Vec3::ZERO, b: Vec3::new(2.0, 0.0, 0.0), radius: 0.75 };
        let b = Capsule {
            a: Vec3::new(0.0, 1.0, 0.0),
            b: Vec3::new(2.0, 1.0, 0.0),
            radius: 0.75,
        };
        let c = capsule_penetration(&a, &b).expect("should penetrate");
        assert_relative_eq!(c.depth, 0.5, epsilon = 1e-5);
        // Normal points off A toward B (+Y).
        assert!(c.normal.y > 0.9, "normal should be roughly +Y, got {:?}", c.normal);
    }

    #[test]
    fn capsule_penetration_far_apart_is_none() {
        let a = Capsule { a: Vec3::ZERO, b: Vec3::new(2.0, 0.0, 0.0), radius: 0.25 };
        let b = Capsule {
            a: Vec3::new(0.0, 5.0, 0.0),
            b: Vec3::new(2.0, 5.0, 0.0),
            radius: 0.25,
        };
        assert!(capsule_penetration(&a, &b).is_none());
    }

    #[test]
    fn capsule_distance_overlap_is_negative() {
        let a = Capsule { a: Vec3::ZERO, b: Vec3::new(2.0, 0.0, 0.0), radius: 0.75 };
        let b = Capsule {
            a: Vec3::new(0.0, 1.0, 0.0),
            b: Vec3::new(2.0, 1.0, 0.0),
            radius: 0.75,
        };
        assert_relative_eq!(capsule_distance(&a, &b), -0.5, epsilon = 1e-5);
    }

    /// Build a skeleton with two parallel arms that overlap; the two "lower"
    /// bones are non-adjacent (they have different parents). Layout:
    ///
    /// ```text
    ///   root(0) ─ a1(1) ─ a2(2)        upper arm at y=0
    ///           └ b1(3) ─ b2(4)        lower arm at y=0.5 (overlaps a1)
    /// ```
    fn overlapping_arms() -> Skeleton {
        let mut sk = Skeleton::new();
        let r = sk.add(Bone::root("root"));
        // Arm A: along +X at y=0.
        let a1 = sk.add(Bone::new("a1", Some(r), Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0))));
        sk.add(Bone::new("a2", Some(a1), Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0))));
        // Arm B: parallel, offset +0.5 on Y, slightly forward so it overlaps a1.
        let b1 = sk.add(Bone::new("b1", Some(r), Mat4::from_translation(Vec3::new(1.0, 0.5, 0.0))));
        sk.add(Bone::new("b2", Some(b1), Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0))));
        sk
    }

    #[test]
    fn self_collisions_reports_nonadjacent_overlap_only() {
        let sk = overlapping_arms();
        let pose = Pose::from_bind(&sk);
        // radius large enough that a1 (bone 1) and b1 (bone 3) overlap. Their
        // segments are offset 0.5 on Y; radius 0.4 each → gap 0.5 < 0.8.
        let pairs = self_collisions(&pose, &sk, 0.4);
        // Exactly one non-adjacent pair: (1, 3).
        assert_eq!(pairs.len(), 1, "got pairs {:?}", pairs.iter().map(|(i, j, _)| (*i, *j)).collect::<Vec<_>>());
        assert_eq!((pairs[0].0, pairs[0].1), (1, 3));
        assert!(is_self_colliding(&pose, &sk, 0.4));
    }

    #[test]
    fn self_collisions_excludes_adjacent_parent_child() {
        // A straight 2-bone chain whose consecutive capsules necessarily touch at
        // the shared joint. With a large radius the adjacent capsules overlap,
        // but the pair must NOT be reported.
        let mut sk = Skeleton::new();
        let r = sk.add(Bone::root("root"));
        let m = sk.add(Bone::new("mid", Some(r), Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0))));
        sk.add(Bone::new("tip", Some(m), Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0))));
        let pose = Pose::from_bind(&sk);
        // root capsule (0→1) and mid capsule (1→2) are adjacent (mid.parent == root).
        let pairs = self_collisions(&pose, &sk, 1.0);
        assert!(pairs.is_empty(), "adjacent parent/child must be excluded, got {:?}", pairs);
    }

    /// Build a planar chain: root at origin, then `n` unit bones along +X, plus a
    /// final tip bone one unit past the last joint. Returns the skeleton and the
    /// chain joint indices (root→tip). Mirrors the helper in `ik.rs` tests.
    fn planar_chain(n: usize) -> (Skeleton, Vec<usize>) {
        let mut sk = Skeleton::new();
        let mut chain = Vec::new();
        let r = sk.add(Bone::root("j0"));
        chain.push(r);
        let mut prev = r;
        for i in 1..n {
            prev = sk.add(Bone::new(
                format!("j{i}"),
                Some(prev),
                Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)),
            ));
            chain.push(prev);
        }
        sk.add(Bone::new(
            "effector",
            Some(prev),
            Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)),
        ));
        (sk, chain)
    }

    #[test]
    fn avoid_obstacles_pushes_chain_out() {
        // Straight 3-joint chain along +X (bones at x=0,1,2; tip at x=3).
        let (sk, chain) = planar_chain(3);
        let mut pose = Pose::from_bind(&sk);
        let radius = 0.3;

        // Static obstacle straddling the chain near its far end: a short capsule
        // perpendicular to the chain (along Z) sitting just above the chain line
        // around x=2.0, so the outer bones penetrate it. The chain can resolve by
        // bending down (-Y) to swing clear.
        let obstacle = Capsule {
            a: Vec3::new(2.0, 0.25, -0.5),
            b: Vec3::new(2.0, 0.25, 0.5),
            radius: 0.4,
        };
        let obstacles = [obstacle];

        // Sanity: at least one bone penetrates before resolving.
        let before =
            count_penetrations(&pose, &sk, &chain, radius, &obstacles, 0.0).0;
        assert!(before > 0, "expected initial penetration, got {before}");

        let remaining = avoid_obstacles(
            &mut pose,
            &sk,
            &chain,
            radius,
            &obstacles,
            None,
            &AvoidParams::default(),
        );
        assert_eq!(remaining, 0, "chain should end collision-free, {remaining} remain");

        // Pose must remain finite.
        for i in 0..sk.len() {
            assert!(pose.head(&sk, i).is_finite(), "bone {i} head must be finite");
        }
    }
}
