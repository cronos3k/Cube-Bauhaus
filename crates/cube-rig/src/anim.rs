//! Skeletal animation: keyframed TRS tracks sampled into a [`Pose`].
//!
//! An [`AnimationClip`] holds one [`BoneTrack`] per animated bone, each with
//! independent translation / rotation / scale keyframes. Sampling produces a
//! [`Pose`] (a local-transform layer over the skeleton's bind pose): animated
//! channels override the bind TRS for their bone, un-animated bones keep their
//! bind transform. Translation/scale interpolate linearly; rotation slerps.
//!
//! This drives both the editor's timeline preview and glTF animation export.

use glam::{Quat, Vec3};

use crate::ik::Pose;
use crate::skeleton::Skeleton;

/// A keyframed TRS track for a single bone. Each channel is a list of
/// `(time_seconds, value)` samples sorted by time; any channel may be empty.
#[derive(Debug, Clone, Default)]
pub struct BoneTrack {
    pub bone: u16,
    pub translation: Vec<(f32, Vec3)>,
    pub rotation: Vec<(f32, Quat)>,
    pub scale: Vec<(f32, Vec3)>,
}

impl BoneTrack {
    pub fn new(bone: u16) -> Self {
        Self { bone, ..Default::default() }
    }

    fn last_time(&self) -> f32 {
        let t = self.translation.last().map(|k| k.0).unwrap_or(0.0);
        let r = self.rotation.last().map(|k| k.0).unwrap_or(0.0);
        let s = self.scale.last().map(|k| k.0).unwrap_or(0.0);
        t.max(r).max(s)
    }
}

/// A named animation: a set of per-bone tracks.
#[derive(Debug, Clone, Default)]
pub struct AnimationClip {
    pub name: String,
    pub tracks: Vec<BoneTrack>,
}

impl AnimationClip {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), tracks: Vec::new() }
    }

    /// Clip length = latest keyframe time across all tracks.
    pub fn duration(&self) -> f32 {
        self.tracks.iter().map(|t| t.last_time()).fold(0.0, f32::max)
    }

    /// Sample the clip at `time` (seconds, clamped to `[0, duration]`) into a new
    /// [`Pose`] over `skeleton`'s bind pose.
    pub fn sample(&self, skeleton: &Skeleton, time: f32) -> Pose {
        let mut pose = Pose::from_bind(skeleton);
        let t = time.clamp(0.0, self.duration());
        for track in &self.tracks {
            let bone = track.bone as usize;
            if bone >= skeleton.len() {
                continue;
            }
            // Start from the bind TRS so partially-animated bones stay sane.
            let (mut scale, mut rot, mut trans) =
                skeleton.bones[bone].local_bind.to_scale_rotation_translation();
            if let Some(v) = sample_vec3(&track.translation, t) {
                trans = v;
            }
            if let Some(q) = sample_quat(&track.rotation, t) {
                rot = q;
            }
            if let Some(v) = sample_vec3(&track.scale, t) {
                scale = v;
            }
            pose.local[bone] = glam::Mat4::from_scale_rotation_translation(scale, rot, trans);
        }
        pose
    }
}

/// Linearly sample a `(time, Vec3)` track, clamping outside the range. Returns
/// `None` if the track is empty.
pub fn sample_vec3(keys: &[(f32, Vec3)], t: f32) -> Option<Vec3> {
    match keys {
        [] => None,
        [single] => Some(single.1),
        _ => {
            if t <= keys[0].0 {
                return Some(keys[0].1);
            }
            if t >= keys[keys.len() - 1].0 {
                return Some(keys[keys.len() - 1].1);
            }
            let i = upper_index(keys, t);
            let (t0, v0) = keys[i - 1];
            let (t1, v1) = keys[i];
            let f = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
            Some(v0.lerp(v1, f))
        }
    }
}

/// Spherically sample a `(time, Quat)` track, clamping outside the range.
pub fn sample_quat(keys: &[(f32, Quat)], t: f32) -> Option<Quat> {
    match keys {
        [] => None,
        [single] => Some(single.1),
        _ => {
            if t <= keys[0].0 {
                return Some(keys[0].1);
            }
            if t >= keys[keys.len() - 1].0 {
                return Some(keys[keys.len() - 1].1);
            }
            let i = upper_index(keys, t);
            let (t0, q0) = keys[i - 1];
            let (t1, q1) = keys[i];
            let f = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
            Some(q0.slerp(q1, f))
        }
    }
}

/// Index of the first key whose time is strictly greater than `t` (callers
/// guarantee `t` is within the interior range).
fn upper_index<V>(keys: &[(f32, V)], t: f32) -> usize {
    keys.partition_point(|k| k.0 <= t).clamp(1, keys.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::Bone;
    use glam::Mat4;

    fn one_bone() -> Skeleton {
        let mut sk = Skeleton::new();
        sk.add(Bone::root("root"));
        sk
    }

    #[test]
    fn sample_vec3_interpolates_and_clamps() {
        let keys = vec![(0.0, Vec3::ZERO), (1.0, Vec3::new(2.0, 0.0, 0.0))];
        assert_eq!(sample_vec3(&keys, -1.0), Some(Vec3::ZERO));
        assert_eq!(sample_vec3(&keys, 0.5), Some(Vec3::new(1.0, 0.0, 0.0)));
        assert_eq!(sample_vec3(&keys, 5.0), Some(Vec3::new(2.0, 0.0, 0.0)));
        assert_eq!(sample_vec3(&[], 0.0), None);
    }

    #[test]
    fn quat_slerp_halfway() {
        let keys = vec![
            (0.0, Quat::IDENTITY),
            (1.0, Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
        ];
        let mid = sample_quat(&keys, 0.5).unwrap();
        let expected = Quat::from_rotation_z(std::f32::consts::FRAC_PI_4);
        assert!(mid.abs_diff_eq(expected, 1e-4) || mid.abs_diff_eq(-expected, 1e-4));
    }

    #[test]
    fn clip_duration_is_latest_key() {
        let mut clip = AnimationClip::new("a");
        let mut tr = BoneTrack::new(0);
        tr.translation = vec![(0.0, Vec3::ZERO), (2.5, Vec3::X)];
        clip.tracks.push(tr);
        assert_eq!(clip.duration(), 2.5);
    }

    #[test]
    fn sample_moves_bone() {
        let sk = one_bone();
        let mut clip = AnimationClip::new("slide");
        let mut tr = BoneTrack::new(0);
        tr.translation = vec![(0.0, Vec3::ZERO), (1.0, Vec3::new(0.0, 3.0, 0.0))];
        clip.tracks.push(tr);

        let p0 = clip.sample(&sk, 0.0);
        assert_eq!(p0.head(&sk, 0), Vec3::ZERO);
        let p1 = clip.sample(&sk, 1.0);
        assert_eq!(p1.head(&sk, 0), Vec3::new(0.0, 3.0, 0.0));
        let ph = clip.sample(&sk, 0.5);
        assert!((ph.head(&sk, 0) - Vec3::new(0.0, 1.5, 0.0)).length() < 1e-5);
    }

    #[test]
    fn unanimated_bone_keeps_bind() {
        let mut sk = Skeleton::new();
        let r = sk.add(Bone::root("root"));
        sk.add(Bone::new("c", Some(r), Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0))));
        // clip only animates root
        let mut clip = AnimationClip::new("a");
        let mut tr = BoneTrack::new(0);
        tr.translation = vec![(0.0, Vec3::ZERO), (1.0, Vec3::ZERO)];
        clip.tracks.push(tr);
        let pose = clip.sample(&sk, 1.0);
        // child keeps its bind offset relative to root
        assert_eq!(pose.local[1], sk.bones[1].local_bind);
    }
}
