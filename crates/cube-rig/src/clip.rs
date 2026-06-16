//! Source-agnostic motion clips.
//!
//! Motion can come from many places — optical mocap, hand-keyed animation, or
//! the *output* of a generative model. To keep everything downstream (limits,
//! priors, IK retargeting) oblivious to provenance, every source normalizes into
//! a single [`Clip`]: a flat list of [`Frame`]s, each holding a root world
//! transform plus one **local** rotation per skeleton bone.
//!
//! The adapter [`Clip::from_animation`] proves the abstraction against a real
//! in-crate source by baking an [`AnimationClip`](crate::anim::AnimationClip)
//! into evenly-spaced frames. A generative source would implement an analogous
//! adapter, and downstream code would not be able to tell the difference.

use glam::{Mat4, Quat};
use serde::{Deserialize, Serialize};

/// A single baked frame of motion: the root's world transform plus one local
/// rotation per bone (parallel to `skeleton.bones`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frame {
    /// World-space transform of the skeleton root for this frame.
    #[serde(with = "crate::skeleton::mat4_serde")]
    pub root: Mat4,
    /// Per-bone **local** rotation, parallel to the skeleton's bones. `Quat`
    /// serializes via glam's `serde` feature (enabled in `Cargo.toml`).
    pub local_rotations: Vec<Quat>,
}

/// A named sequence of frames sampled at a fixed rate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Clip {
    pub name: String,
    /// Frames per second the clip was baked at.
    pub frame_rate: f32,
    pub frames: Vec<Frame>,
}

impl Clip {
    /// Number of frames.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether the clip has no frames.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Wall-clock length in seconds, from frame count and rate. An empty or
    /// single-frame clip has zero duration.
    pub fn duration(&self) -> f32 {
        if self.frames.len() < 2 || self.frame_rate <= 0.0 {
            0.0
        } else {
            (self.frames.len() - 1) as f32 / self.frame_rate
        }
    }

    /// Bake an [`AnimationClip`](crate::anim::AnimationClip) into a source-agnostic
    /// clip by sampling it at `fps` over `[0, duration]`.
    ///
    /// Each frame extracts every bone's local rotation from the sampled
    /// [`Pose`](crate::ik::Pose) via `Mat4::to_scale_rotation_translation`, and
    /// the root world transform from the root bone — the one whose `parent` is
    /// `None` (if there are several, bone 0 is used).
    pub fn from_animation(
        clip: &crate::anim::AnimationClip,
        skeleton: &crate::skeleton::Skeleton,
        fps: f32,
    ) -> Clip {
        // The root bone whose world transform we record per frame.
        let root_bone = skeleton.roots().first().copied().unwrap_or(0);

        let duration = clip.duration();
        // At least one frame; otherwise ceil(duration * fps) + 1 evenly-spaced
        // samples so both endpoints are represented.
        let frame_count = if duration <= 0.0 || fps <= 0.0 {
            1
        } else {
            (duration * fps).ceil() as usize + 1
        };

        let mut frames = Vec::with_capacity(frame_count);
        for f in 0..frame_count {
            let t = if fps > 0.0 { f as f32 / fps } else { 0.0 };
            let pose = clip.sample(skeleton, t);

            // Local rotation per bone (drop scale/translation; keep orientation).
            let local_rotations = pose
                .local
                .iter()
                .map(|m| m.to_scale_rotation_translation().1)
                .collect();

            // Root world transform under this pose.
            let root = pose.global(skeleton, root_bone);

            frames.push(Frame { root, local_rotations });
        }

        Clip { name: clip.name.clone(), frame_rate: fps, frames }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anim::{AnimationClip, BoneTrack};
    use crate::skeleton::{Bone, Skeleton};
    use approx::assert_relative_eq;
    use glam::Vec3;

    fn two_bone_skeleton() -> Skeleton {
        let mut sk = Skeleton::new();
        let r = sk.add(Bone::root("root"));
        sk.add(Bone::new("child", Some(r), Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0))));
        sk
    }

    #[test]
    fn from_animation_frame_count_and_rotation_roundtrip() {
        let sk = two_bone_skeleton();

        // Animate the child bone: rotate it 90° about Z over 1 second.
        let angle = std::f32::consts::FRAC_PI_2;
        let mut clip = AnimationClip::new("spin");
        let mut tr = BoneTrack::new(1);
        tr.rotation = vec![(0.0, Quat::IDENTITY), (1.0, Quat::from_rotation_z(angle))];
        clip.tracks.push(tr);

        let fps = 10.0;
        let baked = Clip::from_animation(&clip, &sk, fps);

        // duration 1s at 10 fps → 11 frames (both endpoints included).
        assert_eq!(baked.len(), 11);
        assert_relative_eq!(baked.frame_rate, fps);
        assert_relative_eq!(baked.duration(), 1.0, epsilon = 1e-5);

        // The final frame's child rotation should match the keyframed 90° turn.
        let last = baked.frames.last().unwrap();
        let child_rot = last.local_rotations[1];
        let expected = Quat::from_rotation_z(angle);
        assert!(
            child_rot.abs_diff_eq(expected, 1e-4) || child_rot.abs_diff_eq(-expected, 1e-4),
            "child rotation {child_rot:?} did not round-trip to {expected:?}"
        );

        // The first frame should still be identity for the child.
        let first = baked.frames.first().unwrap();
        assert!(first.local_rotations[1].abs_diff_eq(Quat::IDENTITY, 1e-4));
    }

    #[test]
    fn empty_animation_bakes_single_frame() {
        let sk = two_bone_skeleton();
        let clip = AnimationClip::new("static");
        let baked = Clip::from_animation(&clip, &sk, 30.0);
        assert_eq!(baked.len(), 1);
        assert!(!baked.is_empty());
        assert_relative_eq!(baked.duration(), 0.0);
    }
}
