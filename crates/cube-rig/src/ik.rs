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
//!   * [`solve_dls`] — a constrained weighted damped-least-squares solver (§5.1)
//!     that steers the solution with anisotropic per-axis compliance from a
//!     [`MotionPrior`](crate::prior::MotionPrior), relaxes toward the prior's
//!     preferred pose via a posture nullspace term, and clamps each step against
//!     hard [`JointLimit`](crate::limits::JointLimit)s.
//!
//! Both return joint world positions; [`Pose::aim_chain`] reconstructs bone
//! rotations from a solved position chain so the result can be skinned/exported.
//!
//! This is built fresh (no external IK code); if a reference implementation is
//! provided later it can be reconciled against these primitives.

use glam::{Mat3, Mat4, Quat, Vec3};

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

/// Tuning parameters for [`solve_dls`].
#[derive(Debug, Clone, Copy)]
pub struct DlsParams {
    /// Number of outer Gauss-Newton iterations.
    pub iterations: usize,
    /// Damping factor `λ` for the damped-least-squares inverse (stabilizes the
    /// solve near singularities; larger ⇒ slower but more robust).
    pub damping: f32,
    /// Step scale in `(0, 1]` applied to each iteration's joint update.
    pub step: f32,
    /// Secondary posture weight: how strongly the nullspace term relaxes the
    /// chain toward the prior's preferred pose. `0` disables the posture term.
    pub posture_gain: f32,
    /// Convergence tolerance: stop once the effector is within this distance of
    /// the target.
    pub tolerance: f32,
}

impl Default for DlsParams {
    fn default() -> Self {
        Self { iterations: 16, damping: 0.5, step: 1.0, posture_gain: 0.1, tolerance: 1e-3 }
    }
}

/// Rotation part of a transform matrix as a [`Quat`].
fn rotation_of(m: Mat4) -> Quat {
    m.to_scale_rotation_translation().1
}

/// Drive `effector_bone`'s head to `target` by rotating the joints in `chain`
/// using a weighted damped-least-squares (DLS) solver with a posture nullspace
/// term (§5.1).
///
/// `chain` lists joint bone indices in root→tip ancestor order; `effector_bone`
/// is typically a descendant of the last chain joint (so the chain steers a tip
/// it does not itself contain). Each joint exposes three rotational DOFs about
/// the world X/Y/Z axes.
///
/// `prior` supplies per-joint anisotropic compliance (`axis_weight`) and a
/// preferred pose to relax toward; when `None`, all DOF weights are `1.0` and
/// the preferred pose is identity. `limits[i]`, if present, clamps chain joint
/// `i` against a hard [`JointLimit`](crate::limits::JointLimit) every step
/// (relative to that bone's bind rotation).
///
/// Returns the final effector→target distance.
///
/// # Compliance frame (v1 simplification)
///
/// v1 treats `axis_weight` as weights on the world X/Y/Z DOFs directly. This is
/// a deliberate simplification: the prior's `axis_weight` is keyed loosely by
/// swing-twist role, not by world axis. v2 aligns the DOF frame with the
/// swing-twist axes the prior was built from before applying the weights.
// The full IK problem genuinely needs all of these inputs (pose, skeleton,
// chain, effector, target, prior, limits, params); grouping them into a struct
// would only obscure the call site, so the arity lint is silenced here.
#[allow(clippy::too_many_arguments)]
pub fn solve_dls(
    pose: &mut Pose,
    skeleton: &Skeleton,
    chain: &[usize],
    effector_bone: usize,
    target: Vec3,
    prior: Option<&crate::prior::MotionPrior>,
    limits: Option<&[Option<crate::limits::JointLimit>]>,
    params: &DlsParams,
) -> f32 {
    use crate::prior::GoalDescriptor;

    if chain.is_empty() {
        return (target - pose.head(skeleton, effector_bone)).length();
    }

    // Build the goal descriptor once for prior lookups.
    let root_head = pose.head(skeleton, chain[0]);
    let goal = GoalDescriptor {
        target_local: target,
        approach_local: (target - root_head).normalize_or_zero(),
        action_tag: None,
    };

    let lambda = params.damping;
    let lambda2 = lambda * lambda;
    let axes = [Vec3::X, Vec3::Y, Vec3::Z];

    for _ in 0..params.iterations {
        let e = pose.head(skeleton, effector_bone);
        let err = target - e;
        if err.length() < params.tolerance {
            break;
        }

        // Per-joint Jacobian columns (3 per joint) and DOF weights.
        let mut columns: Vec<[Vec3; 3]> = Vec::with_capacity(chain.len());
        let mut weights: Vec<Vec3> = Vec::with_capacity(chain.len());
        for &j in chain {
            let p_j = pose.head(skeleton, j);
            let cols = [
                axes[0].cross(e - p_j),
                axes[1].cross(e - p_j),
                axes[2].cross(e - p_j),
            ];
            let w = match prior {
                Some(pr) => pr.bias(j, &goal).axis_weight,
                None => Vec3::ONE,
            };
            columns.push(cols);
            weights.push(w);
        }

        // A = Σ_i w_i (c_i ⊗ c_i) + λ² I, as a 3×3.
        let mut a = Mat3::IDENTITY * lambda2;
        for (cols, w) in columns.iter().zip(weights.iter()) {
            let ws = [w.x, w.y, w.z];
            for (c, &wi) in cols.iter().zip(ws.iter()) {
                let outer = Mat3::from_cols(*c * c.x, *c * c.y, *c * c.z);
                a += outer * wi;
            }
        }
        let a_inv = a.inverse();

        // Posture nullspace term: relax toward the prior's preferred pose.
        //
        // We assemble the per-DOF posture targets `z_i` and the aggregate
        // `Jz = Σ_i c_i z_i` up front because the primary task needs the
        // posture's residual task-space coupling to stay task-orthogonal (see
        // the leak-compensation note below).
        let mut z: Vec<Vec3> = Vec::new();
        let mut jz = Vec3::ZERO;
        let posture_on = params.posture_gain > 0.0;
        if posture_on {
            z.reserve(chain.len());
            for (k, &j) in chain.iter().enumerate() {
                let preferred = match prior {
                    Some(pr) => pr.bias(j, &goal).preferred,
                    None => Quat::IDENTITY,
                };
                let local_rot = rotation_of(pose.local[j]);
                let delta = preferred * local_rot.inverse();
                let zj = params.posture_gain * delta.to_scaled_axis();
                let w = weights[k];
                let zi = w * zj; // (w_x z_x, w_y z_y, w_z z_z)
                z.push(zi);
                jz += columns[k][0] * zi.x + columns[k][1] * zi.y + columns[k][2] * zi.z;
            }
        }

        // Primary task: dtheta_i = w_i * c_i · (A⁻¹ err).
        //
        // The spec's nullspace projector `z_i - w_i c_i·(A⁻¹ Jz)` realizes
        // `(I - J⁺_W J) z` with the *damped* weighted pseudo-inverse, which
        // leaves a residual effector motion of exactly `λ² A⁻¹ Jz` (the damped
        // projector is not an exact orthogonal projector). Left uncompensated
        // this puts a `~λ²`-sized floor on the effector error whenever the
        // posture target disagrees with the reach. We cancel that leak — keeping
        // the posture term truly secondary, per the design intent — by folding
        // it into the primary target, using the *same* 3×3 inverse:
        //   err_eff := err − λ² A⁻¹ Jz.
        let err_eff = if posture_on { err - lambda2 * (a_inv * jz) } else { err };
        let b = a_inv * err_eff;

        // Per-joint accumulated DOF increments.
        let mut dtheta: Vec<Vec3> = columns
            .iter()
            .zip(weights.iter())
            .map(|(cols, w)| {
                Vec3::new(
                    w.x * cols[0].dot(b),
                    w.y * cols[1].dot(b),
                    w.z * cols[2].dot(b),
                )
            })
            .collect();

        if posture_on {
            // dtheta_i += z_i - w_i * c_i · (A⁻¹ Jz)   ⇒  (I - J⁺_W J) z.
            let aj = a_inv * jz;
            for (k, dt) in dtheta.iter_mut().enumerate() {
                let cols = &columns[k];
                let w = weights[k];
                *dt += z[k]
                    - Vec3::new(
                        w.x * cols[0].dot(aj),
                        w.y * cols[1].dot(aj),
                        w.z * cols[2].dot(aj),
                    );
            }
        }

        // Apply per joint root→tip so children see updated parents this pass.
        for (k, &j) in chain.iter().enumerate() {
            let dq = Quat::from_scaled_axis(params.step * dtheta[k]);

            let parent_world_rot = match skeleton.bones[j].parent {
                Some(p) => rotation_of(pose.global(skeleton, p)),
                None => Quat::IDENTITY,
            };
            let world_rot_old = rotation_of(pose.global(skeleton, j));
            let world_rot_new = dq * world_rot_old;
            let mut local_rot_new = parent_world_rot.inverse() * world_rot_new;

            // Clamp relative to the bind rotation against the hard limit, if any.
            if let Some(limit) = limits.and_then(|ls| ls.get(k)).and_then(|l| l.as_ref()) {
                let bind_rot = rotation_of(skeleton.bones[j].local_bind);
                let rel = bind_rot.inverse() * local_rot_new;
                let rel = crate::limits::clamp(rel, limit);
                local_rot_new = bind_rot * rel;
            }

            // Preserve the bone's local translation (its rest offset).
            let local_translation = pose.local[j].w_axis.truncate();
            pose.local[j] = Mat4::from_rotation_translation(local_rot_new, local_translation);
        }
    }

    (target - pose.head(skeleton, effector_bone)).length()
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

    /// Build a planar chain: root at origin, then `n` unit bones along +X, plus
    /// a final tip ("effector") bone one unit past the last joint. Returns the
    /// skeleton, the chain joint indices (root→tip), and the effector index.
    fn planar_chain(n: usize) -> (Skeleton, Vec<usize>, usize) {
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
        let effector = sk.add(Bone::new(
            "effector",
            Some(prev),
            Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)),
        ));
        (sk, chain, effector)
    }

    #[test]
    fn dls_converges_to_reachable_target() {
        // 3 joints + tip ⇒ reach up to length 3 from the origin.
        let (sk, chain, effector) = planar_chain(3);
        let mut pose = Pose::from_bind(&sk);
        let target = Vec3::new(1.5, 1.0, 0.0);
        let dist = solve_dls(
            &mut pose,
            &sk,
            &chain,
            effector,
            target,
            None,
            None,
            &DlsParams::default(),
        );
        assert!(dist < 1e-2, "expected convergence, got distance {dist}");
    }

    #[test]
    fn dls_unreachable_is_stable_and_extends() {
        let (sk, chain, effector) = planar_chain(3);
        let mut pose = Pose::from_bind(&sk);
        let start_x = pose.head(&sk, effector).x;
        let target = Vec3::new(100.0, 0.0, 0.0);
        let dist = solve_dls(
            &mut pose,
            &sk,
            &chain,
            effector,
            target,
            None,
            None,
            &DlsParams::default(),
        );
        // Finite, no NaN/panic.
        assert!(dist.is_finite(), "distance must be finite, got {dist}");
        let end = pose.head(&sk, effector);
        assert!(end.is_finite(), "effector must be finite, got {end:?}");
        // Chain extends roughly toward the target (+X).
        assert!(end.x >= start_x, "effector x {} should not shrink from {}", end.x, start_x);
    }

    /// Build a one-bone prior whose joint `joint` has a strong `axis_weight`
    /// dominated by the `which` swing-twist component (0=swing-u, 1=twist=Y,
    /// 2=swing-v), by feeding spread rotations about the matching axis.
    fn prior_biased(num_bones: usize, joint: usize, which: usize) -> crate::prior::MotionPrior {
        use crate::clip::{Clip, Frame};
        use crate::prior::MotionPriorBuilder;
        let axis = match which {
            0 => Vec3::X, // swing-u
            1 => Vec3::Y, // twist
            _ => Vec3::Z, // swing-v
        };
        let mut frames = Vec::new();
        for i in 0..9 {
            let angle = (i as f32 - 4.0) * 0.3;
            let mut rots = vec![Quat::IDENTITY; num_bones];
            rots[joint] = Quat::from_axis_angle(axis, angle);
            frames.push(Frame { root: Mat4::IDENTITY, local_rotations: rots });
        }
        let clip = Clip { name: "spread".into(), frame_rate: 30.0, frames };
        let mut builder = MotionPriorBuilder::new(num_bones);
        builder.add_clip(&clip);
        builder.build()
    }

    #[test]
    fn dls_anisotropy_steers_solution() {
        // Same reach; compare a uniform prior against one that makes the middle
        // joint highly compliant about Z (the planar DOF that actually bends the
        // chain toward an out-of-plane-free target). The high-Z-weight joint
        // should rotate more.
        let (sk, chain, effector) = planar_chain(3);
        let target = Vec3::new(1.2, 1.2, 0.0);
        let params = DlsParams { posture_gain: 0.0, ..DlsParams::default() };

        // Middle joint = chain index 1.
        let mid = chain[1];
        let num = sk.len();

        // Uniform: no prior.
        let mut pose_uniform = Pose::from_bind(&sk);
        solve_dls(&mut pose_uniform, &sk, &chain, effector, target, None, None, &params);
        let mid_angle_uniform = rotation_of(pose_uniform.local[mid]).to_scaled_axis().z.abs();

        // Biased: strong Z (swing-v) compliance on the middle joint.
        let prior = prior_biased(num, mid, 2);
        let mut pose_biased = Pose::from_bind(&sk);
        solve_dls(&mut pose_biased, &sk, &chain, effector, target, Some(&prior), None, &params);
        let mid_angle_biased = rotation_of(pose_biased.local[mid]).to_scaled_axis().z.abs();

        assert!(
            mid_angle_biased > mid_angle_uniform + 1e-3,
            "high-weight middle joint should rotate more: biased {mid_angle_biased} vs uniform {mid_angle_uniform}"
        );
    }

    #[test]
    fn dls_respects_hinge_limit() {
        let (sk, chain, effector) = planar_chain(3);
        let mut pose = Pose::from_bind(&sk);
        let target = Vec3::new(1.0, 1.5, 0.0);

        // Tight hinge about Z on the middle joint (chain index 1).
        let (lo, hi) = (-0.2_f32, 0.2_f32);
        let mut limits: Vec<Option<crate::limits::JointLimit>> = vec![None; chain.len()];
        limits[1] = Some(crate::limits::JointLimit::Hinge { axis: Vec3::Z, min: lo, max: hi });

        solve_dls(
            &mut pose,
            &sk,
            &chain,
            effector,
            target,
            None,
            Some(&limits),
            &DlsParams::default(),
        );

        // Recover the middle joint's angle relative to bind about the hinge axis.
        let mid = chain[1];
        let bind_rot = rotation_of(sk.bones[mid].local_bind);
        let local_rot = rotation_of(pose.local[mid]);
        let rel = bind_rot.inverse() * local_rot;
        // Signed angle about Z.
        let r = Vec3::new(rel.x, rel.y, rel.z);
        let angle = 2.0 * r.dot(Vec3::Z).atan2(rel.w);
        assert!(
            angle >= lo - 1e-3 && angle <= hi + 1e-3,
            "hinge angle {angle} out of [{lo}, {hi}]"
        );
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
