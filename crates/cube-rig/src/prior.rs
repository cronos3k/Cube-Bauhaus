//! Goal-conditioned motion prior — the steering field skeleton (§6–7).
//!
//! A [`MotionPrior`] is a learned-by-statistics field that tells the IK solver
//! *how* a joint prefers to move: a per-axis compliance weight (how freely the
//! joint may deviate from its preferred pose along each axis) plus a preferred
//! rotation to relax toward. It is built from [`Clip`](crate::clip::Clip)s by
//! [`MotionPriorBuilder`], which accumulates per-bone rotation statistics.
//!
//! The lookup is **goal-conditioned by design**: [`MotionPrior::bias`] already
//! takes a [`GoalDescriptor`] (a body-relative target/approach/action triple) so
//! callers wire against the final API now. The current v1 prior stores a single
//! global bucket and ignores the goal; v2 will index a goal-conditioned grid.

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
/// v1: one global [`JointBias`] per bone. The [`bias`](MotionPrior::bias) lookup
/// already accepts a [`GoalDescriptor`] but ignores it for now.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MotionPrior {
    /// Per-bone bias for the single global goal bucket. Bones outside this vector
    /// (or with no observations) fall back to [`JointBias::default`].
    per_bone: Vec<JointBias>,
}

impl MotionPrior {
    /// Steering bias for `bone` under `goal`.
    ///
    /// Returns the graceful default ([`Vec3::ONE`] weights, identity preferred)
    /// when the bone has no recorded data.
    pub fn bias(&self, bone: usize, _goal: &GoalDescriptor) -> JointBias {
        // TODO: goal-conditioned cells — index a per-goal grid using `goal`.
        self.per_bone.get(bone).cloned().unwrap_or_default()
    }

    /// Number of bones with recorded bias data.
    pub fn len(&self) -> usize {
        self.per_bone.len()
    }

    /// Whether the prior holds no bone data.
    pub fn is_empty(&self) -> bool {
        self.per_bone.is_empty()
    }
}

/// Running per-bone statistics accumulator over one or more clips.
///
/// For each observed local rotation we decompose it (about a per-bone twist
/// axis, default `Vec3::Y` for v1) and accumulate a running mean quaternion plus
/// per-axis angle variance. On [`build`](MotionPriorBuilder::build) the variance
/// becomes a normalized inverse-stiffness proxy: high-variance axes are deemed
/// more compliant and get a larger `axis_weight`.
///
/// v1 keeps a single scalar weight per axis; v2 would replace this with a
/// swing-twist grid keyed by goal.
pub struct MotionPriorBuilder {
    num_bones: usize,
    /// Twist axis per bone (default `Vec3::Y`).
    twist_axis: Vec<Vec3>,
    /// Sample count per bone.
    count: Vec<u32>,
    /// Running mean rotation per bone (sign-aligned average, renormalized).
    mean: Vec<Quat>,
    /// Running sum and sum-of-squares of per-axis signed angles, for variance.
    sum_angles: Vec<Vec3>,
    sum_sq_angles: Vec<Vec3>,
}

impl MotionPriorBuilder {
    /// A fresh builder for a skeleton of `num_bones` bones.
    pub fn new(num_bones: usize) -> Self {
        Self {
            num_bones,
            twist_axis: vec![Vec3::Y; num_bones],
            count: vec![0; num_bones],
            mean: vec![Quat::IDENTITY; num_bones],
            sum_angles: vec![Vec3::ZERO; num_bones],
            sum_sq_angles: vec![Vec3::ZERO; num_bones],
        }
    }

    /// Override the twist axis used to decompose `bone`'s rotations.
    pub fn set_twist_axis(&mut self, bone: usize, axis: Vec3) {
        if bone < self.num_bones {
            self.twist_axis[bone] = axis.normalize_or_zero();
        }
    }

    /// Accumulate statistics from every frame of `clip`.
    pub fn add_clip(&mut self, clip: &crate::clip::Clip) {
        for frame in &clip.frames {
            for (bone, &rot) in frame.local_rotations.iter().enumerate() {
                if bone >= self.num_bones {
                    break;
                }
                self.observe(bone, rot);
            }
        }
    }

    /// Fold one rotation observation into bone `bone`'s running statistics.
    fn observe(&mut self, bone: usize, rot: Quat) {
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
        let axis = self.twist_axis[bone];
        let (swing, twist) = swing_twist(rot, axis);
        let twist_angle = signed_angle_about(twist, axis);
        let (sx, sy) = swing_components(swing, axis);
        // Map to a Vec3 keyed loosely by axis role: (swing-u, twist, swing-v).
        let sample = Vec3::new(sx, twist_angle, sy);

        self.sum_angles[bone] += sample;
        self.sum_sq_angles[bone] += sample * sample;
        self.count[bone] = n + 1;
    }

    /// Finalize the accumulated statistics into a [`MotionPrior`].
    pub fn build(self) -> MotionPrior {
        let mut per_bone = Vec::with_capacity(self.num_bones);

        // First pass: per-bone per-axis variance.
        let mut variances = Vec::with_capacity(self.num_bones);
        for bone in 0..self.num_bones {
            let n = self.count[bone];
            let var = if n < 2 {
                Vec3::ZERO
            } else {
                let nf = n as f32;
                let mean = self.sum_angles[bone] / nf;
                // E[x²] − E[x]² per component, clamped non-negative.
                let mean_sq = self.sum_sq_angles[bone] / nf;
                (mean_sq - mean * mean).max(Vec3::ZERO)
            };
            variances.push(var);
        }

        // Normalize variance into a compliance weight: higher variance ⇒ larger
        // weight. We scale by the global max so weights land in a tame range and
        // floor at 1.0 so the default (no data) stays comparable.
        let max_var = variances
            .iter()
            .flat_map(|v| [v.x, v.y, v.z])
            .fold(0.0_f32, f32::max);

        for (bone, &v) in variances.iter().enumerate() {
            if self.count[bone] == 0 {
                per_bone.push(JointBias::default());
                continue;
            }
            // 1.0 baseline + variance-proportional bonus (normalized by max_var).
            let axis_weight = if max_var > 1e-9 {
                Vec3::ONE + v / max_var
            } else {
                Vec3::ONE
            };
            per_bone.push(JointBias { axis_weight, preferred: self.mean[bone] });
        }

        MotionPrior { per_bone }
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
}
