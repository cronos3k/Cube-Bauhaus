//! Hard joint limits (§4).
//!
//! Two limit representations cover the rigs we care about:
//!
//!   * [`JointLimit::Hinge`] — a single-DOF revolute joint about a fixed axis,
//!     as used by robot rigs (e.g. the Unitree G1's knee or elbow).
//!   * [`JointLimit::SwingTwist`] — a ball joint decomposed into *twist* about a
//!     primary axis plus a *swing* cone, as used by character ball-joints
//!     (shoulders, hips).
//!
//! [`clamp`] projects an arbitrary rotation back into the feasible set. All
//! angles are in radians.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

/// A hard joint limit. Angles are radians.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JointLimit {
    /// Single-DOF revolute joint about `axis`, angle clamped to `[min, max]`.
    Hinge { axis: Vec3, min: f32, max: f32 },
    /// Ball joint: twist about `twist_axis` clamped to `twist`, and swing
    /// clamped to an elliptical cone with per-component half-angles `swing`.
    SwingTwist {
        twist_axis: Vec3,
        /// `(min, max)` twist angle about `twist_axis`.
        twist: (f32, f32),
        /// `(x_half_angle, y_half_angle)` of the elliptical swing cone, applied
        /// to the swing rotation's axis-angle components perpendicular to twist.
        swing: (f32, f32),
    },
}

/// Decompose `rot` into `(swing, twist)` about `axis` such that
/// `swing * twist == rot` and `twist` is a rotation purely about `axis`.
///
/// This is the standard swing-twist decomposition: project the quaternion's
/// vector part onto `axis` to recover the twist, then `swing = rot * twist⁻¹`.
fn swing_twist(rot: Quat, axis: Vec3) -> (Quat, Quat) {
    let axis = axis.normalize_or_zero();
    // Vector part of the quaternion (the rotation axis scaled by sin(θ/2)).
    let r = Vec3::new(rot.x, rot.y, rot.z);
    // Project onto the twist axis; this is the component that twists about it.
    let proj = axis * r.dot(axis);
    let mut twist = Quat::from_xyzw(proj.x, proj.y, proj.z, rot.w);
    // Degenerate (180° swing): the projection vanishes; fall back to identity.
    if twist.length_squared() < 1e-12 {
        twist = Quat::IDENTITY;
    } else {
        twist = twist.normalize();
    }
    let swing = rot * twist.inverse();
    (swing, twist)
}

/// Signed rotation angle of `q` about `axis`, in `[-π, π]`.
fn signed_angle_about(q: Quat, axis: Vec3) -> f32 {
    let axis = axis.normalize_or_zero();
    let r = Vec3::new(q.x, q.y, q.z);
    // sin(θ/2) along the axis, cos(θ/2) in w → atan2 recovers the signed angle.
    let s = r.dot(axis);
    2.0 * s.atan2(q.w)
}

/// Project `rot` into the feasible region of `limit`.
///
/// `Hinge` extracts the signed angle about the hinge axis, clamps it to
/// `[min, max]`, and rebuilds a pure-axis rotation (any off-axis component is
/// discarded). `SwingTwist` decomposes about the twist axis, clamps the twist
/// angle to its range and the swing to an elliptical cone, then recombines.
pub fn clamp(rot: Quat, limit: &JointLimit) -> Quat {
    match *limit {
        JointLimit::Hinge { axis, min, max } => {
            let angle = signed_angle_about(rot, axis).clamp(min, max);
            Quat::from_axis_angle(axis.normalize_or_zero(), angle)
        }
        JointLimit::SwingTwist { twist_axis, twist, swing } => {
            let (swing_q, twist_q) = swing_twist(rot, twist_axis);

            // Clamp the twist angle to its range and rebuild.
            let twist_angle = signed_angle_about(twist_q, twist_axis).clamp(twist.0, twist.1);
            let clamped_twist = Quat::from_axis_angle(twist_axis.normalize_or_zero(), twist_angle);

            // The swing is a rotation whose axis lies in the plane ⟂ twist_axis.
            // Represent it as an axis-angle vector and clamp each perpendicular
            // component against the elliptical cone half-angles.
            let (swing_axis, swing_angle) = swing_q.to_axis_angle();
            let mut sv = swing_axis.normalize_or_zero() * swing_angle;
            // Build an orthonormal basis (u, v) spanning the plane ⟂ twist_axis.
            let t = twist_axis.normalize_or_zero();
            let u = if t.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
            let u = (u - t * u.dot(t)).normalize_or_zero();
            let v = t.cross(u);
            // Decompose the swing vector onto (u, v) and clamp elliptically.
            let su = sv.dot(u);
            let sv_comp = sv.dot(v);
            let cu = su.clamp(-swing.0, swing.0);
            let cv = sv_comp.clamp(-swing.1, swing.1);
            sv = u * cu + v * cv;
            let clamped_swing = if sv.length() < 1e-9 {
                Quat::IDENTITY
            } else {
                Quat::from_axis_angle(sv.normalize(), sv.length())
            };

            clamped_swing * clamped_twist
        }
    }
}

/// Joint-limit table for the Unitree G1 29-DOF humanoid.
///
/// Values transcribed from the Unitree G1 MuJoCo model distributed with NVIDIA
/// GR00T-WholeBodyControl. Each entry is a 1-DOF [`JointLimit::Hinge`] about the
/// joint's actuation axis, in radians.
// Some table values (0.5236 ≈ π/6, 1.0472 ≈ π/3) are literal transcriptions
// from the source model and are intentionally kept as-is, not const-folded.
#[allow(clippy::approx_constant)]
pub fn g1_29dof_limits() -> Vec<(&'static str, JointLimit)> {
    const X: Vec3 = Vec3::X;
    const Y: Vec3 = Vec3::Y;
    const Z: Vec3 = Vec3::Z;
    fn h(axis: Vec3, min: f32, max: f32) -> JointLimit {
        JointLimit::Hinge { axis, min, max }
    }
    vec![
        ("left_hip_pitch_joint", h(Y, -2.5307, 2.8798)),
        ("left_hip_roll_joint", h(X, -0.5236, 2.9671)),
        ("left_hip_yaw_joint", h(Z, -2.7576, 2.7576)),
        ("left_knee_joint", h(Y, -0.087267, 2.8798)),
        ("left_ankle_pitch_joint", h(Y, -0.87267, 0.5236)),
        ("left_ankle_roll_joint", h(X, -0.2618, 0.2618)),
        ("right_hip_pitch_joint", h(Y, -2.5307, 2.8798)),
        ("right_hip_roll_joint", h(X, -2.9671, 0.5236)),
        ("right_hip_yaw_joint", h(Z, -2.7576, 2.7576)),
        ("right_knee_joint", h(Y, -0.087267, 2.8798)),
        ("right_ankle_pitch_joint", h(Y, -0.87267, 0.5236)),
        ("right_ankle_roll_joint", h(X, -0.2618, 0.2618)),
        ("waist_yaw_joint", h(Z, -2.618, 2.618)),
        ("waist_roll_joint", h(X, -0.52, 0.52)),
        ("waist_pitch_joint", h(Y, -0.52, 0.52)),
        ("left_shoulder_pitch_joint", h(Y, -3.0892, 2.6704)),
        ("left_shoulder_roll_joint", h(X, -1.5882, 2.2515)),
        ("left_shoulder_yaw_joint", h(Z, -2.618, 2.618)),
        ("left_elbow_joint", h(Y, -1.0472, 2.0944)),
        ("left_wrist_roll_joint", h(X, -1.9722, 1.9722)),
        ("left_wrist_pitch_joint", h(Y, -1.6144, 1.6144)),
        ("left_wrist_yaw_joint", h(Z, -1.6144, 1.6144)),
        ("right_shoulder_pitch_joint", h(Y, -3.0892, 2.6704)),
        ("right_shoulder_roll_joint", h(X, -2.2515, 1.5882)),
        ("right_shoulder_yaw_joint", h(Z, -2.618, 2.618)),
        ("right_elbow_joint", h(Y, -1.0472, 2.0944)),
        ("right_wrist_roll_joint", h(X, -1.9722, 1.9722)),
        ("right_wrist_pitch_joint", h(Y, -1.6144, 1.6144)),
        ("right_wrist_yaw_joint", h(Z, -1.6144, 1.6144)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn hinge_clamps_rotation_beyond_max() {
        let limit = JointLimit::Hinge { axis: Vec3::Z, min: -0.5, max: 1.0 };
        // 2 rad about Z exceeds the 1.0 max → should clamp to 1.0.
        let rot = Quat::from_rotation_z(2.0);
        let clamped = clamp(rot, &limit);
        let angle = signed_angle_about(clamped, Vec3::Z);
        assert_relative_eq!(angle, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn hinge_leaves_in_range_rotation_unchanged() {
        let limit = JointLimit::Hinge { axis: Vec3::Z, min: -2.0, max: 2.0 };
        let rot = Quat::from_rotation_z(0.7);
        let clamped = clamp(rot, &limit);
        assert!(
            clamped.abs_diff_eq(rot, 1e-5) || clamped.abs_diff_eq(-rot, 1e-5),
            "in-range rotation should be unchanged"
        );
    }

    #[test]
    fn swing_twist_clamps_twist() {
        let limit = JointLimit::SwingTwist {
            twist_axis: Vec3::Y,
            twist: (-0.3, 0.3),
            swing: (1.0, 1.0),
        };
        let rot = Quat::from_rotation_y(1.2);
        let clamped = clamp(rot, &limit);
        let twist_angle = signed_angle_about(clamped, Vec3::Y);
        assert_relative_eq!(twist_angle, 0.3, epsilon = 1e-4);
    }

    #[test]
    fn g1_table_has_29_entries_and_asymmetric_knee() {
        let table = g1_29dof_limits();
        assert_eq!(table.len(), 29);
        let (name, knee) = &table[3];
        assert_eq!(*name, "left_knee_joint");
        match knee {
            JointLimit::Hinge { min, max, .. } => {
                assert_relative_eq!(*min, -0.087267, epsilon = 1e-6);
                assert_relative_eq!(*max, 2.8798, epsilon = 1e-6);
            }
            _ => panic!("knee should be a hinge"),
        }
    }
}
