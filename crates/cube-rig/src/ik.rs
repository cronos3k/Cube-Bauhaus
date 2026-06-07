//! Procedural inverse kinematics over a skeleton pose.
//!
//! A [`Pose`] is a mutable layer of per-bone *local* transforms that starts from
//! the skeleton's bind pose and can be driven by animation or IK. The solvers
//! here are positional and engine-agnostic:
//!
//!   * [`fabrik`] — Forward-And-Backward-Reaching IK for chains of any length
//!     (tails, spines, tentacles): fast, stable, length-preserving.
//!   * [`two_bone`] — analytic law-of-cosines IK for the common 2-bone case
//!     (arms, legs) with a pole vector for elbow/knee direction.
//!
//! Both return joint world positions; [`Pose::aim_chain`] reconstructs bone
//! rotations from a solved position chain so the result can be skinned/exported.
//!
//! This is built fresh (no external IK code); if a reference implementation is
//! provided later it can be reconciled against these primitives.

use glam::{Mat4, Quat, Vec3};

use crate::skeleton::Skeleton;

/// A mutable per-bone local-transform layer over a [`Skeleton`]'s bind pose.
#[derive(Debug, Clone)]
pub struct Pose {
    /// Local transform per bone (parallel to `skeleton.bones`). Starts equal to
    /// each bone's `local_bind`.
    pub local: Vec<Mat4>,
}

impl Pose {
    /// A pose identical to the skeleton's bind pose.
    pub fn from_bind(skeleton: &Skeleton) -> Self {
        Self { local: skeleton.bones.iter().map(|b| b.local_bind).collect() }
    }

    /// Global (model-space) transform of bone `i` under this pose.
    pub fn global(&self, skeleton: &Skeleton, i: usize) -> Mat4 {
        match skeleton.bones[i].parent {
            Some(p) => self.global(skeleton, p) * self.local[i],
            None => self.local[i],
        }
    }

    /// Model-space position of bone `i`'s head under this pose.
    pub fn head(&self, skeleton: &Skeleton, i: usize) -> Vec3 {
        self.global(skeleton, i).w_axis.truncate()
    }

    /// Skinning matrices for this pose: `global_pose(i) * inverse_bind(i)`,
    /// ready to multiply bind-space vertices. Parallel to bones.
    pub fn skinning_matrices(&self, skeleton: &Skeleton) -> Vec<Mat4> {
        (0..skeleton.len())
            .map(|i| self.global(skeleton, i) * skeleton.inverse_bind(i))
            .collect()
    }

    /// Re-orient each bone in `chain` so it points at the next solved position,
    /// writing the rotation into the pose's local transforms. `chain` is a list
    /// of bone indices from root→tip where each bone is the parent of the next;
    /// `solved` holds the desired world position of each chain joint head plus a
    /// final tip position (so `solved.len() == chain.len() + 1`).
    pub fn aim_chain(&mut self, skeleton: &Skeleton, chain: &[usize], solved: &[Vec3]) {
        debug_assert_eq!(solved.len(), chain.len() + 1);
        for (k, &bone) in chain.iter().enumerate() {
            // Parent global under the *current* pose (already updated upstream).
            let parent_global = match skeleton.bones[bone].parent {
                Some(p) => self.global(skeleton, p),
                None => Mat4::IDENTITY,
            };
            let bone_global = parent_global * self.local[bone];

            // Current tip direction (toward the child's bind head) and the
            // desired direction (toward the solved next joint), both in world.
            let head = bone_global.w_axis.truncate();
            let current_dir = {
                let child_local_head = self.local.get(chain.get(k + 1).copied().unwrap_or(bone))
                    .map(|m| m.w_axis.truncate())
                    .unwrap_or(Vec3::Y);
                (bone_global.transform_point3(child_local_head) - head).normalize_or_zero()
            };
            let desired_dir = (solved[k + 1] - solved[k]).normalize_or_zero();
            if current_dir.length_squared() < 1e-12 || desired_dir.length_squared() < 1e-12 {
                continue;
            }
            let delta = Quat::from_rotation_arc(current_dir, desired_dir);
            // Apply the world-space delta rotation on the bone's local transform.
            let parent_rot = Quat::from_mat4(&parent_global);
            let local_delta = parent_rot.inverse() * delta * parent_rot;
            self.local[bone] *= Mat4::from_quat(local_delta);
        }
    }
}

/// Solve a positional chain so its tip reaches `target` using FABRIK.
///
/// `joints` are the current world positions from root (`joints[0]`, held fixed)
/// to tip. Segment lengths are taken from the initial joint spacing. Returns the
/// solved joint positions (same length as `joints`).
pub fn fabrik(joints: &[Vec3], target: Vec3, iterations: usize, tolerance: f32) -> Vec<Vec3> {
    let n = joints.len();
    if n < 2 {
        return joints.to_vec();
    }
    let lengths: Vec<f32> = (0..n - 1).map(|i| (joints[i + 1] - joints[i]).length()).collect();
    let total: f32 = lengths.iter().sum();
    let root = joints[0];

    let mut p = joints.to_vec();

    // Target out of reach → fully extend straight toward it.
    if (target - root).length() >= total {
        let dir = (target - root).normalize_or_zero();
        for i in 1..n {
            p[i] = p[i - 1] + dir * lengths[i - 1];
        }
        return p;
    }

    for _ in 0..iterations.max(1) {
        // Backward: set tip to target, work toward root.
        p[n - 1] = target;
        for i in (0..n - 1).rev() {
            let dir = (p[i] - p[i + 1]).normalize_or_zero();
            p[i] = p[i + 1] + dir * lengths[i];
        }
        // Forward: pin root, work toward tip.
        p[0] = root;
        for i in 0..n - 1 {
            let dir = (p[i + 1] - p[i]).normalize_or_zero();
            p[i + 1] = p[i] + dir * lengths[i];
        }
        if (p[n - 1] - target).length() < tolerance {
            break;
        }
    }
    p
}

/// Analytic two-bone IK (law of cosines). Given a fixed `root`, an upper-bone
/// length and lower-bone length, a `target`, and a `pole` hint for the joint
/// (elbow/knee) direction, returns `(joint_position, end_position)`.
///
/// If the target is unreachable the limb is fully extended toward it.
pub fn two_bone(
    root: Vec3,
    upper_len: f32,
    lower_len: f32,
    target: Vec3,
    pole: Vec3,
) -> (Vec3, Vec3) {
    let to_target = target - root;
    let dist = to_target.length().clamp(1e-6, upper_len + lower_len);
    let dir = to_target / to_target.length().max(1e-6);

    // Angle at the root between the upper bone and the root→target line.
    let cos_root = ((upper_len * upper_len + dist * dist - lower_len * lower_len)
        / (2.0 * upper_len * dist))
        .clamp(-1.0, 1.0);
    let root_angle = cos_root.acos();

    // Build a bend axis perpendicular to the limb, biased by the pole vector.
    let pole_dir = {
        let projected = pole - root - dir * (pole - root).dot(dir);
        if projected.length_squared() < 1e-10 {
            // pole is colinear; pick any perpendicular
            let any = if dir.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
            (any - dir * any.dot(dir)).normalize_or_zero()
        } else {
            projected.normalize_or_zero()
        }
    };
    let bend_axis = dir.cross(pole_dir).normalize_or_zero();

    // Rotate the target direction up by root_angle around the bend axis to find
    // the upper bone's direction, then place the joint.
    let upper_dir = Quat::from_axis_angle(bend_axis, root_angle) * dir;
    let joint = root + upper_dir * upper_len;
    let end = joint + (target - joint).normalize_or_zero() * lower_len;
    (joint, end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::Bone;
    use approx::assert_relative_eq;

    fn approx_len(a: Vec3, b: Vec3) -> f32 {
        (b - a).length()
    }

    #[test]
    fn fabrik_reaches_target_in_range() {
        let joints = vec![Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)];
        let target = Vec3::new(1.0, 1.0, 0.0);
        let solved = fabrik(&joints, target, 32, 1e-4);
        // tip reaches the target
        assert!((solved[2] - target).length() < 1e-3);
        // root stays pinned
        assert_relative_eq!(approx_len(solved[0], Vec3::ZERO), 0.0, epsilon = 1e-5);
        // segment lengths preserved
        assert_relative_eq!(approx_len(solved[0], solved[1]), 1.0, epsilon = 1e-3);
        assert_relative_eq!(approx_len(solved[1], solved[2]), 1.0, epsilon = 1e-3);
    }

    #[test]
    fn fabrik_out_of_reach_extends_straight() {
        let joints = vec![Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 0.0, 0.0)];
        let target = Vec3::new(10.0, 0.0, 0.0);
        let solved = fabrik(&joints, target, 16, 1e-4);
        // fully extended toward target along +X, total length 2
        assert_relative_eq!(solved[2].x, 2.0, epsilon = 1e-4);
        assert_relative_eq!(solved[2].y, 0.0, epsilon = 1e-4);
    }

    #[test]
    fn two_bone_reaches_and_preserves_lengths() {
        let root = Vec3::ZERO;
        let (joint, end) = two_bone(root, 1.0, 1.0, Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        assert_relative_eq!(approx_len(root, joint), 1.0, epsilon = 1e-4);
        assert_relative_eq!(approx_len(joint, end), 1.0, epsilon = 1e-4);
        assert!((end - Vec3::new(1.0, 1.0, 0.0)).length() < 1e-3);
    }

    #[test]
    fn pose_from_bind_matches_skeleton() {
        let mut sk = Skeleton::new();
        let r = sk.add(Bone::root("root"));
        sk.add(Bone::new("c", Some(r), Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0))));
        let pose = Pose::from_bind(&sk);
        assert_eq!(pose.head(&sk, 1), sk.head_position(1));
    }

    #[test]
    fn aim_chain_points_bone_at_target() {
        // two-bone chain along +Y; aim it along +X
        let mut sk = Skeleton::new();
        let r = sk.add(Bone::root("root"));
        let m = sk.add(Bone::new("mid", Some(r), Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0))));
        sk.add(Bone::new("tip", Some(m), Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0))));

        let mut pose = Pose::from_bind(&sk);
        let chain = [r, m];
        let solved = [
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
        ];
        pose.aim_chain(&sk, &chain, &solved);
        // root bone head stays at origin, its child (mid head) should move toward +X
        let mid_head = pose.head(&sk, m);
        assert!(mid_head.x > 0.7, "mid head x = {}", mid_head.x);
    }
}
