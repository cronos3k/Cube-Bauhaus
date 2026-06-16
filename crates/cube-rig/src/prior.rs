//! Goal-conditioned motion prior — the steering field skeleton (§6–7).
//!
//! A [`MotionPrior`] is a learned-by-statistics field that tells the IK solver
//! *how* a joint prefers to move: a per-axis compliance weight (how freely the
//! joint may deviate from its preferred pose along each axis) plus a preferred
//! rotation to relax toward. It is built from [`Clip`](crate::clip::Clip)s by
//! [`MotionPriorBuilder`], which accumulates per-bone rotation statistics.
//!
//! The lookup is **goal-conditioned by design**: [`MotionPrior::bias`] takes a
//! [`GoalDescriptor`] (a body-relative target/approach/action triple). v2 keeps a
//! `global` fallback bucket AND a [`HashMap`](std::collections::HashMap) of
//! goal-conditioned cells: the same joint is steered differently depending on
//! where the effector goal sits relative to the body. See [`goal_cell`] for the
//! quantization scheme. A v1-baked prior (global only, empty `cells`) keeps
//! working unchanged — `bias` falls back to `global`.

use std::collections::HashMap;

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

/// Decompose `rot` into `(swing, twist)` about `twist_axis`, where
/// `swing * twist ≈ rot` and `twist` is a pure rotation about `twist_axis`.
///
/// Standard swing-twist decomposition: project the quaternion's vector part onto
/// the axis to isolate the twist, then `swing = rot * twist⁻¹`.
pub fn swing_twist(rot: Quat, twist_axis: Vec3) -> (Quat, Quat) {
    let axis = twist_axis.normalize_or_zero();
    let r = Vec3::new(rot.x, rot.y, rot.z);
    let proj = axis * r.dot(axis);
    let mut twist = Quat::from_xyzw(proj.x, proj.y, proj.z, rot.w);
    if twist.length_squared() < 1e-12 {
        // 180° swing degeneracy: no well-defined twist.
        twist = Quat::IDENTITY;
    } else {
        twist = twist.normalize();
    }
    let swing = rot * twist.inverse();
    (swing, twist)
}

/// Body-relative conditioning variable: where the goal is and how to approach it.
///
/// All vectors are expressed in the body's local frame so the prior is
/// translation/orientation invariant. `action_tag` optionally selects a discrete
/// behavior bucket (reach, grasp, step, …) once goal-conditioned cells exist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalDescriptor {
    pub target_local: Vec3,
    pub approach_local: Vec3,
    pub action_tag: Option<u16>,
}

/// Number of radial distance shells used by [`goal_cell`].
const NUM_SHELLS: u32 = 3;
/// Upper bound (exclusive) of shell 0, in body-local units. Below this the goal
/// is "near"; from here to [`SHELL_FAR`] it is "mid"; beyond it is "far".
const SHELL_NEAR: f32 = 0.5;
const SHELL_FAR: f32 = 1.5;
/// The cell id reserved for a zero-length (undefined-direction) target.
pub const FALLBACK_CELL: u32 = u32::MAX;

/// Quantize a [`GoalDescriptor`] into a coarse, deterministic cell id.
///
/// The scheme buckets the body-relative target [`GoalDescriptor::target_local`]
/// along two coarse axes:
///
/// * **Direction** → one of 6 sectors by the dominant signed axis of the
///   *direction* (the component with the largest magnitude): `+X,-X,+Y,-Y,+Z,-Z`
///   mapped to sector ids `0..6`.
/// * **Distance** → one of [`NUM_SHELLS`] radial shells by `target_local.length()`
///   against the fixed thresholds [`SHELL_NEAR`] and [`SHELL_FAR`]: shell `0` is
///   `len < SHELL_NEAR`, shell `1` is `SHELL_NEAR..SHELL_FAR`, shell `2` is
///   `len >= SHELL_FAR`.
///
/// The two are combined as `sector * NUM_SHELLS + shell`, yielding `0..18`.
///
/// A zero-length target (no well-defined direction) maps to [`FALLBACK_CELL`].
pub fn goal_cell(goal: &GoalDescriptor) -> u32 {
    let t = goal.target_local;
    let len = t.length();
    if len < 1e-6 {
        return FALLBACK_CELL;
    }

    // Dominant signed axis → sector 0..6.
    let (ax, ay, az) = (t.x.abs(), t.y.abs(), t.z.abs());
    let sector = if ax >= ay && ax >= az {
        if t.x >= 0.0 { 0 } else { 1 }
    } else if ay >= az {
        if t.y >= 0.0 { 2 } else { 3 }
    } else if t.z >= 0.0 {
        4
    } else {
        5
    };

    // Radial shell 0..NUM_SHELLS.
    let shell = if len < SHELL_NEAR {
        0
    } else if len < SHELL_FAR {
        1
    } else {
        2
    };

    sector * NUM_SHELLS + shell
}

/// Per-bone steering bias: how compliant the joint is per axis, plus the pose it
/// relaxes toward.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointBias {
    /// Per-axis compliance weight (larger ⇒ more freely the joint may move along
    /// that axis). Defaults to `Vec3::ONE`.
    pub axis_weight: Vec3,
    /// Preferred (mean) local rotation the joint relaxes toward.
    pub preferred: Quat,
}

impl Default for JointBias {
    fn default() -> Self {
        Self { axis_weight: Vec3::ONE, preferred: Quat::IDENTITY }
    }
}

/// A goal-conditioned motion prior.
///
/// Holds a `global` per-bone bucket (the goal-agnostic fallback) plus a map of
/// goal-conditioned `cells` keyed by [`goal_cell`]. The
/// [`bias`](MotionPrior::bias) lookup prefers the cell matching the goal and
/// falls back to `global` (then to [`JointBias::default`]) when a cell lacks data
/// for the requested bone.
///
/// A v1-baked prior has data only in `global` and an empty `cells` map; it
/// behaves exactly as before. The `cells` field is `#[serde(default)]` so older
/// JSON without it still deserializes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MotionPrior {
    /// Per-bone bias for the goal-agnostic global bucket. Bones outside this
    /// vector (or with no observations) fall back to [`JointBias::default`].
    global: Vec<JointBias>,
    /// Goal-conditioned cells: cell id (from [`goal_cell`]) → per-bone bias.
    /// A cell only holds bones that had enough samples; missing bones fall back
    /// to `global`.
    #[serde(default)]
    cells: HashMap<u32, Vec<JointBias>>,
}

impl MotionPrior {
    /// Steering bias for `bone` under `goal`.
    ///
    /// Resolves the goal's cell via [`goal_cell`] and returns the cell's per-bone
    /// bias when that cell has recorded data for `bone`; otherwise falls back to
    /// the `global` bucket, then to the graceful default ([`Vec3::ONE`] weights,
    /// identity preferred) when no data exists at all.
    pub fn bias(&self, bone: usize, goal: &GoalDescriptor) -> JointBias {
        let cell = goal_cell(goal);
        if let Some(bias) = self.cells.get(&cell).and_then(|v| v.get(bone)) {
            return bias.clone();
        }
        self.global.get(bone).cloned().unwrap_or_default()
    }

    /// Number of bones with recorded bias data in the global bucket.
    pub fn len(&self) -> usize {
        self.global.len()
    }

    /// Whether the prior holds no global bone data.
    pub fn is_empty(&self) -> bool {
        self.global.is_empty()
    }

    /// Number of goal-conditioned cells with recorded data.
    pub fn num_cells(&self) -> usize {
        self.cells.len()
    }
}

/// Running per-bone swing-twist statistics for one bucket (the global bucket or
/// a single goal-conditioned cell).
///
/// For each observed local rotation we decompose it (about a per-bone twist
/// axis) and accumulate a running mean quaternion plus per-axis angle variance.
/// On [`finalize`](BoneAccumulator::finalize) the variance becomes a normalized
/// inverse-stiffness proxy: high-variance axes are deemed more compliant and get
/// a larger `axis_weight`.
#[derive(Clone)]
struct BoneAccumulator {
    num_bones: usize,
    /// Sample count per bone.
    count: Vec<u32>,
    /// Running mean rotation per bone (sign-aligned average, renormalized).
    mean: Vec<Quat>,
    /// Running sum and sum-of-squares of per-axis signed angles, for variance.
    sum_angles: Vec<Vec3>,
    sum_sq_angles: Vec<Vec3>,
}

impl BoneAccumulator {
    fn new(num_bones: usize) -> Self {
        Self {
            num_bones,
            count: vec![0; num_bones],
            mean: vec![Quat::IDENTITY; num_bones],
            sum_angles: vec![Vec3::ZERO; num_bones],
            sum_sq_angles: vec![Vec3::ZERO; num_bones],
        }
    }

    /// Fold one rotation observation into bone `bone`'s running statistics,
    /// decomposed about `axis`.
    fn observe(&mut self, bone: usize, rot: Quat, axis: Vec3) {
        let n = self.count[bone];

        // Running mean quaternion with sign alignment: flip `rot` into the same
        // hemisphere as the current mean before averaging, then renormalize.
        if n == 0 {
            self.mean[bone] = rot.normalize();
        } else {
            let mean = self.mean[bone];
            let aligned = if mean.dot(rot) < 0.0 { -rot } else { rot };
            let k = 1.0 / (n as f32 + 1.0);
            // Lerp toward the new sample, then renormalize to stay a unit quat.
            let blended = Quat::from_xyzw(
                mean.x + (aligned.x - mean.x) * k,
                mean.y + (aligned.y - mean.y) * k,
                mean.z + (aligned.z - mean.z) * k,
                mean.w + (aligned.w - mean.w) * k,
            );
            self.mean[bone] = blended.normalize();
        }

        // Per-axis angle samples from the swing-twist decomposition: the twist
        // angle about the bone axis, and two swing angles in the ⟂ plane.
        let (swing, twist) = swing_twist(rot, axis);
        let twist_angle = signed_angle_about(twist, axis);
        let (sx, sy) = swing_components(swing, axis);
        // Map to a Vec3 keyed loosely by axis role: (swing-u, twist, swing-v).
        let sample = Vec3::new(sx, twist_angle, sy);

        self.sum_angles[bone] += sample;
        self.sum_sq_angles[bone] += sample * sample;
        self.count[bone] = n + 1;
    }

    /// Per-bone per-axis variance (`Vec3::ZERO` for bones with < 2 samples).
    fn variances(&self) -> Vec<Vec3> {
        (0..self.num_bones)
            .map(|bone| {
                let n = self.count[bone];
                if n < 2 {
                    Vec3::ZERO
                } else {
                    let nf = n as f32;
                    let mean = self.sum_angles[bone] / nf;
                    // E[x²] − E[x]² per component, clamped non-negative.
                    let mean_sq = self.sum_sq_angles[bone] / nf;
                    (mean_sq - mean * mean).max(Vec3::ZERO)
                }
            })
            .collect()
    }

    /// Finalize into a per-bone bias vector spanning all bones (for index
    /// stability). Bones with `count < min_samples` get [`JointBias::default`],
    /// so cell lookups for those bones fall back to the global bucket.
    fn finalize(&self, min_samples: u32) -> Vec<JointBias> {
        let variances = self.variances();
        // Normalize variance into a compliance weight: higher variance ⇒ larger
        // weight. We scale by the max so weights land in a tame range and floor
        // at 1.0 so the default (no data) stays comparable.
        let max_var = variances
            .iter()
            .flat_map(|v| [v.x, v.y, v.z])
            .fold(0.0_f32, f32::max);

        variances
            .iter()
            .enumerate()
            .map(|(bone, &v)| {
                if self.count[bone] < min_samples {
                    return JointBias::default();
                }
                // 1.0 baseline + variance-proportional bonus (normalized).
                let axis_weight = if max_var > 1e-9 {
                    Vec3::ONE + v / max_var
                } else {
                    Vec3::ONE
                };
                JointBias { axis_weight, preferred: self.mean[bone] }
            })
            .collect()
    }

    /// Whether any bone has at least `min_samples` samples.
    fn any_dense(&self, min_samples: u32) -> bool {
        self.count.iter().any(|&c| c >= min_samples)
    }
}

/// Running per-bone statistics accumulator over one or more clips.
///
/// Keeps a goal-agnostic `global` accumulator (fed by [`add_clip`] and
/// [`add_clip_with_goal`]) plus per-cell accumulators keyed by [`goal_cell`]
/// (fed only by [`add_clip_with_goal`]). On [`build`](MotionPriorBuilder::build)
/// the global bucket finalizes as before and each cell with enough samples
/// finalizes into the prior's `cells` map.
///
/// [`add_clip`]: MotionPriorBuilder::add_clip
/// [`add_clip_with_goal`]: MotionPriorBuilder::add_clip_with_goal
pub struct MotionPriorBuilder {
    num_bones: usize,
    /// Twist axis per bone (default `Vec3::Y`).
    twist_axis: Vec<Vec3>,
    /// Goal-agnostic accumulator.
    global: BoneAccumulator,
    /// Per-cell accumulators, created lazily as cells are observed.
    cells: HashMap<u32, BoneAccumulator>,
}

/// Minimum samples a (cell, bone) pair needs before its cell-specific bias is
/// emitted. Below this the cell omits the bone and [`MotionPrior::bias`] falls
/// back to `global`.
const MIN_CELL_SAMPLES: u32 = 2;

impl MotionPriorBuilder {
    /// A fresh builder for a skeleton of `num_bones` bones.
    pub fn new(num_bones: usize) -> Self {
        Self {
            num_bones,
            twist_axis: vec![Vec3::Y; num_bones],
            global: BoneAccumulator::new(num_bones),
            cells: HashMap::new(),
        }
    }

    /// Override the twist axis used to decompose `bone`'s rotations.
    pub fn set_twist_axis(&mut self, bone: usize, axis: Vec3) {
        if bone < self.num_bones {
            self.twist_axis[bone] = axis.normalize_or_zero();
        }
    }

    /// Accumulate statistics from every frame of `clip` into the global bucket.
    pub fn add_clip(&mut self, clip: &crate::clip::Clip) {
        for frame in &clip.frames {
            for (bone, &rot) in frame.local_rotations.iter().enumerate() {
                if bone >= self.num_bones {
                    break;
                }
                let axis = self.twist_axis[bone];
                self.global.observe(bone, rot, axis);
            }
        }
    }

    /// Accumulate statistics from `clip` into BOTH the global bucket and the
    /// goal-conditioned cell each frame falls into.
    ///
    /// For every frame we build a temporary [`Pose`](crate::ik::Pose) from the
    /// frame's local rotations (preserving each bone's bind translation),
    /// forward-kinematic the `effector_bone`'s head into the root bone's local
    /// frame, and use that body-relative position as a [`GoalDescriptor`] to find
    /// the frame's cell via [`goal_cell`]. The frame's per-bone swing-twist stats
    /// then feed the global accumulator and that cell's accumulator.
    pub fn add_clip_with_goal(
        &mut self,
        clip: &crate::clip::Clip,
        skeleton: &crate::skeleton::Skeleton,
        effector_bone: usize,
    ) {
        let root_bone = skeleton.roots().first().copied().unwrap_or(0);
        for frame in &clip.frames {
            let goal = frame_goal(frame, skeleton, root_bone, effector_bone);
            let cell = goal_cell(&goal);
            let cell_acc = self
                .cells
                .entry(cell)
                .or_insert_with(|| BoneAccumulator::new(self.num_bones));
            for (bone, &rot) in frame.local_rotations.iter().enumerate() {
                if bone >= self.num_bones {
                    break;
                }
                let axis = self.twist_axis[bone];
                self.global.observe(bone, rot, axis);
                cell_acc.observe(bone, rot, axis);
            }
        }
    }

    /// Finalize the accumulated statistics into a [`MotionPrior`].
    ///
    /// The global bucket finalizes with no minimum-sample gate (matching v1).
    /// Each cell finalizes with a [`MIN_CELL_SAMPLES`] gate per bone; cells with
    /// no qualifying bone are dropped so `bias` falls back to `global`.
    pub fn build(self) -> MotionPrior {
        let global = self.global.finalize(1);

        let cells = self
            .cells
            .iter()
            .filter(|(_, acc)| acc.any_dense(MIN_CELL_SAMPLES))
            .map(|(&id, acc)| (id, acc.finalize(MIN_CELL_SAMPLES)))
            .collect();

        MotionPrior { global, cells }
    }
}

/// Build a body-relative [`GoalDescriptor`] for one frame: forward-kinematic the
/// effector's head into the root bone's local frame.
fn frame_goal(
    frame: &crate::clip::Frame,
    skeleton: &crate::skeleton::Skeleton,
    root_bone: usize,
    effector_bone: usize,
) -> GoalDescriptor {
    use crate::ik::Pose;
    use glam::Mat4;

    // Temporary pose: each bone's local transform is its bind translation with
    // the frame's local rotation applied (rotation about the bind origin).
    let mut pose = Pose::from_bind(skeleton);
    for (bone, &rot) in frame.local_rotations.iter().enumerate() {
        if bone >= pose.local.len() {
            break;
        }
        let translation = skeleton.bones[bone].local_bind.w_axis.truncate();
        pose.local[bone] = Mat4::from_rotation_translation(rot, translation);
    }

    let effector_world = pose.head(skeleton, effector_bone);
    let root_world = pose.global(skeleton, root_bone);
    // Effector head expressed in the root bone's local frame (body-relative).
    let target_local = root_world.inverse().transform_point3(effector_world);

    GoalDescriptor {
        target_local,
        approach_local: target_local.normalize_or_zero(),
        action_tag: None,
    }
}

/// Signed rotation angle of `q` about `axis`, in `[-π, π]`.
fn signed_angle_about(q: Quat, axis: Vec3) -> f32 {
    let axis = axis.normalize_or_zero();
    let r = Vec3::new(q.x, q.y, q.z);
    let s = r.dot(axis);
    2.0 * s.atan2(q.w)
}

/// The two swing angles of `swing` in the plane perpendicular to `axis`,
/// returned as `(u_component, v_component)` of the axis-angle vector.
fn swing_components(swing: Quat, axis: Vec3) -> (f32, f32) {
    let (sa, angle) = swing.to_axis_angle();
    let sv = sa.normalize_or_zero() * angle;
    let t = axis.normalize_or_zero();
    let u = if t.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let u = (u - t * u.dot(t)).normalize_or_zero();
    let v = t.cross(u);
    (sv.dot(u), sv.dot(v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{Clip, Frame};
    use glam::Mat4;

    #[test]
    fn swing_twist_recomposes() {
        let rot = Quat::from_rotation_y(0.7) * Quat::from_rotation_x(0.3);
        let (swing, twist) = swing_twist(rot, Vec3::Y);
        let recombined = swing * twist;
        assert!(
            recombined.abs_diff_eq(rot, 1e-4) || recombined.abs_diff_eq(-rot, 1e-4),
            "swing*twist {recombined:?} did not recompose {rot:?}"
        );
    }

    fn frame_with_bone1(rot: Quat) -> Frame {
        Frame { root: Mat4::IDENTITY, local_rotations: vec![Quat::IDENTITY, rot] }
    }

    #[test]
    fn builder_weights_high_variance_axis_higher() {
        // Bone 1 rotates a lot about Y (the twist axis) and never about X.
        let mut frames = Vec::new();
        for i in 0..9 {
            let angle = (i as f32 - 4.0) * 0.3; // spread of twist angles about Y
            frames.push(frame_with_bone1(Quat::from_rotation_y(angle)));
        }
        let clip = Clip { name: "y_spin".into(), frame_rate: 30.0, frames };

        let mut builder = MotionPriorBuilder::new(2);
        builder.add_clip(&clip);
        let prior = builder.build();

        let goal = GoalDescriptor {
            target_local: Vec3::ZERO,
            approach_local: Vec3::Z,
            action_tag: None,
        };
        let bias = prior.bias(1, &goal);

        // axis_weight maps to (swing-u, twist=Y, swing-v). Y varies a lot;
        // X (swing-u) is static, so the Y component should dominate.
        assert!(
            bias.axis_weight.y > bias.axis_weight.x,
            "expected high-variance Y weight {} > static X weight {}",
            bias.axis_weight.y,
            bias.axis_weight.x
        );
    }

    #[test]
    fn missing_bone_returns_default_bias() {
        let prior = MotionPrior::default();
        let goal = GoalDescriptor {
            target_local: Vec3::ZERO,
            approach_local: Vec3::Z,
            action_tag: None,
        };
        let bias = prior.bias(42, &goal);
        assert_eq!(bias.axis_weight, Vec3::ONE);
        assert!(bias.preferred.abs_diff_eq(Quat::IDENTITY, 1e-6));
    }

    // ── Goal quantization ────────────────────────────────────────────────────

    fn goal_at(target: Vec3) -> GoalDescriptor {
        GoalDescriptor {
            target_local: target,
            approach_local: target.normalize_or_zero(),
            action_tag: None,
        }
    }

    #[test]
    fn goal_cell_is_stable_and_distinguishes_direction_and_distance() {
        // Stability: same input → same id.
        let x = goal_at(Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(goal_cell(&x), goal_cell(&goal_at(Vec3::new(1.0, 0.0, 0.0))));

        // +X vs +Y differ (different sector).
        let y = goal_at(Vec3::new(0.0, 1.0, 0.0));
        assert_ne!(goal_cell(&x), goal_cell(&y));

        // Near vs far along +X differ (different shell), same sector.
        let near = goal_at(Vec3::new(0.2, 0.0, 0.0)); // len 0.2 < SHELL_NEAR
        let far = goal_at(Vec3::new(3.0, 0.0, 0.0)); // len 3.0 >= SHELL_FAR
        assert_ne!(goal_cell(&near), goal_cell(&far));

        // Zero-length target → fallback cell.
        assert_eq!(goal_cell(&goal_at(Vec3::ZERO)), FALLBACK_CELL);
    }

    // ── v1 compatibility ─────────────────────────────────────────────────────

    #[test]
    fn v1_built_prior_falls_back_to_global() {
        // Built the old way (add_clip + build): no cells, only global.
        let mut frames = Vec::new();
        for i in 0..9 {
            frames.push(frame_with_bone1(Quat::from_rotation_y((i as f32 - 4.0) * 0.3)));
        }
        let clip = Clip { name: "y".into(), frame_rate: 30.0, frames };
        let mut builder = MotionPriorBuilder::new(2);
        builder.add_clip(&clip);
        let prior = builder.build();

        assert_eq!(prior.num_cells(), 0);
        // A non-fallback goal still resolves via global (no cell data exists).
        let bias = prior.bias(1, &goal_at(Vec3::new(1.0, 0.0, 0.0)));
        assert!(bias.axis_weight.y > bias.axis_weight.x);
    }

    #[test]
    fn global_only_json_deserializes_without_cells() {
        // Simulate a v1-baked prior on disk: object with only `global`, no
        // `cells` key. `#[serde(default)]` must fill the empty map.
        let json = r#"{"global":[{"axis_weight":[1.0,2.0,1.0],"preferred":[0.0,0.0,0.0,1.0]}]}"#;
        let prior: MotionPrior = serde_json::from_str(json).expect("deserialize v1 json");
        assert_eq!(prior.len(), 1);
        assert_eq!(prior.num_cells(), 0);
        let bias = prior.bias(0, &goal_at(Vec3::new(0.0, 1.0, 0.0)));
        assert!((bias.axis_weight.y - 2.0).abs() < 1e-6);
    }

    // ── Goal-conditioning ────────────────────────────────────────────────────

    /// A 3-bone arm: fixed root, a steerable shoulder (bone 1), and an effector
    /// tip (bone 2) offset along +X by `arm` from the shoulder. The shoulder's
    /// rotation steers where the effector's FK head lands, so the shoulder's
    /// posture and the body-relative goal are naturally coupled.
    fn arm_skeleton(arm: f32) -> crate::skeleton::Skeleton {
        use crate::skeleton::{Bone, Skeleton};
        let mut sk = Skeleton::new();
        let r = sk.add(Bone::root("root"));
        let s = sk.add(Bone::new("shoulder", Some(r), Mat4::IDENTITY));
        sk.add(Bone::new("effector", Some(s), Mat4::from_translation(Vec3::new(arm, 0.0, 0.0))));
        sk
    }

    #[test]
    fn goal_conditioning_separates_postures_by_goal() {
        // Effector arm of length 2 from the shoulder along +X. Rotating the
        // shoulder (bone 1) about +Z by +90° swings the effector head from ≈ +X
        // to ≈ +Y. Each clip holds a distinct shoulder posture, so the +X-goal
        // cell and the +Y-goal cell record different preferred rotations.
        let sk = arm_skeleton(2.0);
        let effector = 2usize;

        let posture_a = Quat::IDENTITY; // clip A: shoulder unrotated → effector +X
        let posture_b = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2); // → +Y

        let make = |posture: Quat| -> Clip {
            let frames = (0..6)
                .map(|_| Frame {
                    root: Mat4::IDENTITY,
                    local_rotations: vec![Quat::IDENTITY, posture, Quat::IDENTITY],
                })
                .collect();
            Clip { name: "c".into(), frame_rate: 30.0, frames }
        };
        let clip_a = make(posture_a);
        let clip_b = make(posture_b);

        let mut builder = MotionPriorBuilder::new(3);
        builder.add_clip_with_goal(&clip_a, &sk, effector);
        builder.add_clip_with_goal(&clip_b, &sk, effector);
        let prior = builder.build();

        // Two distinct cells should have formed (+X shell, +Y shell).
        assert!(prior.num_cells() >= 2, "expected ≥2 cells, got {}", prior.num_cells());

        // Sanity: the cells the two goals resolve to are different.
        let goal_x = goal_at(Vec3::new(2.0, 0.0, 0.0));
        let goal_y = goal_at(Vec3::new(0.0, 2.0, 0.0));
        assert_ne!(goal_cell(&goal_x), goal_cell(&goal_y));

        let bias_x = prior.bias(1, &goal_x);
        let bias_y = prior.bias(1, &goal_y);

        // The +X-goal cell recorded posture_a; the +Y-goal cell posture_b.
        let d = |q: Quat, p: Quat| (q.dot(p)).abs(); // 1.0 == identical
        assert!(
            d(bias_x.preferred, posture_a) > d(bias_x.preferred, posture_b),
            "X-goal preferred {:?} should be nearer posture_a",
            bias_x.preferred
        );
        assert!(
            d(bias_y.preferred, posture_b) > d(bias_y.preferred, posture_a),
            "Y-goal preferred {:?} should be nearer posture_b",
            bias_y.preferred
        );
        // And the two cells' biases differ meaningfully.
        assert!(
            d(bias_x.preferred, bias_y.preferred) < 0.99,
            "cells should differ: {:?} vs {:?}",
            bias_x.preferred,
            bias_y.preferred
        );
    }
}
