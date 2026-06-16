//! Offline motion-prior extraction (§9): bake a [`MotionPrior`] from a directory
//! of motion clips, source-agnostically.
//!
//! Clips arrive in two flavors: pre-baked [`Clip`]s as `.json`, or raw motion as
//! `.gltf`/`.glb` (a skeleton plus one or more animations). Both normalize into
//! [`Clip`]s and feed a single [`MotionPriorBuilder`], so the resulting prior is
//! oblivious to where the motion came from. The [`extract_prior`](crate) binary
//! is a thin wrapper over [`bake_prior_from_dir`].

use std::path::Path;

use crate::clip::Clip;
use crate::import::gltf::{import_gltf_animations, import_gltf_skeleton};
use crate::import::ImportError;
use crate::prior::{MotionPrior, MotionPriorBuilder};
use crate::skeleton::Skeleton;

/// Outcome stats for a bake, for logging.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExtractStats {
    /// Number of source files that yielded at least one clip.
    pub files: usize,
    /// Total clips baked across all files.
    pub clips: usize,
    /// Total frames across all clips.
    pub frames: usize,
    /// Canonical bone count (max `local_rotations` length seen).
    pub bones: usize,
}

/// Errors raised while loading clips or baking a prior.
#[derive(Debug)]
pub enum ExtractError {
    /// Filesystem error (reading a file, walking a directory).
    Io(String),
    /// JSON (de)serialization error for a `.json` clip.
    Json(String),
    /// Underlying model/animation import error for glTF/GLB.
    Import(ImportError),
}

impl std::fmt::Display for ExtractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtractError::Io(s) => write!(f, "I/O error: {s}"),
            ExtractError::Json(s) => write!(f, "JSON error: {s}"),
            ExtractError::Import(e) => write!(f, "import error: {e}"),
        }
    }
}

impl std::error::Error for ExtractError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ExtractError::Import(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ExtractError {
    fn from(e: std::io::Error) -> Self {
        ExtractError::Io(e.to_string())
    }
}

impl From<serde_json::Error> for ExtractError {
    fn from(e: serde_json::Error) -> Self {
        ExtractError::Json(e.to_string())
    }
}

impl From<ImportError> for ExtractError {
    fn from(e: ImportError) -> Self {
        ExtractError::Import(e)
    }
}

/// Load motion clip(s) from one path.
///
/// `.json` → a single [`Clip`] via `serde_json`; `.gltf`/`.glb` → import the
/// skeleton plus animations and bake each animation into a [`Clip`] with
/// [`Clip::from_animation`] at `fps`. Any other extension yields an empty vec.
///
/// A glTF/GLB with no skin (and thus no skeleton) yields an empty vec.
pub fn clips_from_path(path: &Path, fps: f32) -> Result<Vec<Clip>, ExtractError> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "json" => {
            let bytes = std::fs::read(path)?;
            let clip: Clip = serde_json::from_slice(&bytes)?;
            Ok(vec![clip])
        }
        "gltf" | "glb" => {
            let Some(skeleton) = import_gltf_skeleton(path)? else {
                return Ok(Vec::new());
            };
            let animations = import_gltf_animations(path)?;
            Ok(animations
                .iter()
                .map(|anim| Clip::from_animation(anim, &skeleton, fps))
                .collect())
        }
        _ => Ok(Vec::new()),
    }
}

/// The canonical bone count: the max `local_rotations` length across all frames.
fn canonical_bone_count(clips: &[Clip]) -> usize {
    clips
        .iter()
        .flat_map(|c| &c.frames)
        .map(|f| f.local_rotations.len())
        .max()
        .unwrap_or(0)
}

/// Bake a [`MotionPrior`] from clips.
///
/// `num_bones` is the max `local_rotations` length across all clips (the
/// canonical bone count); clips are accumulated in order into one builder.
pub fn bake_prior(clips: &[Clip]) -> MotionPrior {
    let num_bones = canonical_bone_count(clips);
    let mut builder = MotionPriorBuilder::new(num_bones);
    for clip in clips {
        builder.add_clip(clip);
    }
    builder.build()
}

/// Bake a **goal-conditioned** [`MotionPrior`] from clips.
///
/// Like [`bake_prior`] but additionally indexes each frame by the body-relative
/// position of `effector_bone`, feeding a goal-conditioned cell grid (see
/// [`goal_cell`](crate::prior::goal_cell)). Every frame still contributes to the
/// `global` fallback bucket, so the result degrades gracefully to v1 behavior
/// for goals whose cell lacks data.
///
/// `skeleton` supplies the bind pose used for the per-frame forward kinematics;
/// it must match the clips' bone ordering. `effector_bone` is the bone whose FK
/// head defines the goal (e.g. a hand or foot tip).
///
/// To get goal-conditioning from a directory of glTF/GLB clips, import the
/// skeleton (via [`import_gltf_skeleton`]) and pass it here together with the
/// clips from [`clips_from_path`]; the binary's default
/// [`bake_prior_from_dir`] path stays goal-agnostic.
pub fn bake_prior_with_goals(
    clips: &[Clip],
    skeleton: &Skeleton,
    effector_bone: usize,
) -> MotionPrior {
    let num_bones = canonical_bone_count(clips).max(skeleton.len());
    let mut builder = MotionPriorBuilder::new(num_bones);
    for clip in clips {
        builder.add_clip_with_goal(clip, skeleton, effector_bone);
    }
    builder.build()
}

/// Walk `dir` **recursively**, load every clip, bake them into one prior, and
/// return the prior plus [`ExtractStats`].
///
/// Files are visited in a deterministic order (sorted by path at each directory
/// level). Files that yield no clips (unsupported extensions, skin-less glTF)
/// are not counted toward `stats.files`.
pub fn bake_prior_from_dir(dir: &Path, fps: f32) -> Result<(MotionPrior, ExtractStats), ExtractError> {
    let mut clips = Vec::new();
    let mut files = 0usize;
    collect_clips(dir, fps, &mut clips, &mut files)?;

    let stats = ExtractStats {
        files,
        clips: clips.len(),
        frames: clips.iter().map(Clip::len).sum(),
        bones: canonical_bone_count(&clips),
    };
    Ok((bake_prior(&clips), stats))
}

/// Recursively gather clips under `dir`, counting source files that contributed.
fn collect_clips(
    dir: &Path,
    fps: f32,
    clips: &mut Vec<Clip>,
    files: &mut usize,
) -> Result<(), ExtractError> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)?
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(|e| e.path())
        .collect();
    entries.sort();

    for path in entries {
        if path.is_dir() {
            collect_clips(&path, fps, clips, files)?;
        } else {
            let loaded = clips_from_path(&path, fps)?;
            if !loaded.is_empty() {
                *files += 1;
                clips.extend(loaded);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anim::{AnimationClip, BoneTrack};
    use crate::clip::Frame;
    use crate::export::glb::export_glb_animated;
    use crate::mesh::{RigVertex, SkinnedMesh};
    use crate::prior::GoalDescriptor;
    use crate::skeleton::{Bone, Skeleton};
    use glam::{Mat4, Quat, Vec3};

    /// A unique temp path with the given suffix (so parallel tests don't clash).
    fn temp_path(stem: &str, ext: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id();
        std::env::temp_dir().join(format!("cube_rig_extract_{stem}_{pid}_{n}.{ext}"))
    }

    fn two_bone_rig() -> (SkinnedMesh, Skeleton) {
        let v = |x: f32| RigVertex {
            position: [x, 0.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
        };
        let mut mesh = SkinnedMesh::new(vec![v(0.0), v(1.0), v(2.0)], vec![0, 1, 2]);
        let mut sk = Skeleton::new();
        let r = sk.add(Bone::root("root"));
        sk.add(Bone::new("tip", Some(r), Mat4::from_translation(Vec3::X)));
        mesh.set_rigid(0, 0);
        mesh.set_rigid(1, 1);
        (mesh, sk)
    }

    // ── Test 1: glTF animation round-trip (the key gate for Part A) ──────────

    #[test]
    fn gltf_animation_roundtrips_through_export_import() {
        let (mesh, sk) = two_bone_rig();

        // Known translation on the root, known rotation on the tip.
        let mut clip = AnimationClip::new("walk");
        let mut root_tr = BoneTrack::new(0);
        let root_t0 = Vec3::ZERO;
        let root_t1 = Vec3::new(0.0, 0.5, 0.25);
        root_tr.translation = vec![(0.0, root_t0), (1.0, root_t1)];
        let mut tip_tr = BoneTrack::new(1);
        let tip_q0 = Quat::IDENTITY;
        let tip_q1 = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        tip_tr.rotation = vec![(0.0, tip_q0), (0.5, tip_q1)];
        clip.tracks.push(root_tr);
        clip.tracks.push(tip_tr);

        let path = temp_path("anim", "glb");
        export_glb_animated(&path, &mesh, &sk, std::slice::from_ref(&clip)).expect("export");

        let recovered = import_gltf_animations(&path).expect("import animations");
        let _ = std::fs::remove_file(&path);

        assert_eq!(recovered.len(), 1, "expected exactly one animation");
        let anim = &recovered[0];
        assert_eq!(anim.name, "walk");

        // Find recovered tracks by bone index (the node→bone mapping must be
        // exact: bone 0 = root, bone 1 = tip).
        let root = anim
            .tracks
            .iter()
            .find(|t| t.bone == 0)
            .expect("root track");
        let tip = anim.tracks.iter().find(|t| t.bone == 1).expect("tip track");

        // Translation keyframe times and values survive.
        assert_eq!(root.translation.len(), 2);
        assert!((root.translation[0].0 - 0.0).abs() < 1e-4);
        assert!((root.translation[1].0 - 1.0).abs() < 1e-4);
        assert!((root.translation[0].1 - root_t0).length() < 1e-4);
        assert!((root.translation[1].1 - root_t1).length() < 1e-4);

        // Rotation keyframe times and values survive (allow quat sign flip).
        assert_eq!(tip.rotation.len(), 2);
        assert!((tip.rotation[0].0 - 0.0).abs() < 1e-4);
        assert!((tip.rotation[1].0 - 0.5).abs() < 1e-4);
        let q0 = tip.rotation[0].1;
        let q1 = tip.rotation[1].1;
        assert!(q0.abs_diff_eq(tip_q0, 1e-4) || q0.abs_diff_eq(-tip_q0, 1e-4));
        assert!(q1.abs_diff_eq(tip_q1, 1e-4) || q1.abs_diff_eq(-tip_q1, 1e-4));
    }

    // ── Test 2: clips_from_path on JSON ──────────────────────────────────────

    #[test]
    fn clips_from_path_loads_json() {
        let rot = Quat::from_rotation_y(0.6);
        let frames = vec![
            Frame {
                root: Mat4::IDENTITY,
                local_rotations: vec![Quat::IDENTITY, rot],
            },
            Frame {
                root: Mat4::IDENTITY,
                local_rotations: vec![Quat::IDENTITY, rot],
            },
        ];
        let clip = Clip {
            name: "hand_built".into(),
            frame_rate: 24.0,
            frames,
        };

        let path = temp_path("clip", "json");
        let json = serde_json::to_vec_pretty(&clip).unwrap();
        std::fs::write(&path, json).unwrap();

        let loaded = clips_from_path(&path, 30.0).expect("load json clip");
        let _ = std::fs::remove_file(&path);

        assert_eq!(loaded.len(), 1);
        let c = &loaded[0];
        assert_eq!(c.name, "hand_built");
        assert_eq!(c.frames.len(), 2);
        let recovered_rot = c.frames[0].local_rotations[1];
        assert!(recovered_rot.abs_diff_eq(rot, 1e-5));
    }

    #[test]
    fn clips_from_path_ignores_unknown_extension() {
        let path = temp_path("misc", "txt");
        std::fs::write(&path, b"not a clip").unwrap();
        let loaded = clips_from_path(&path, 30.0).expect("ignore unknown");
        let _ = std::fs::remove_file(&path);
        assert!(loaded.is_empty());
    }

    // ── Test 3: bake_prior weights animated bone over static ─────────────────

    #[test]
    fn bake_prior_weights_animated_bone_higher() {
        // Bone 0 static; bone 1 spins about Y across frames.
        let mut frames = Vec::new();
        for i in 0..9 {
            let angle = (i as f32 - 4.0) * 0.3;
            frames.push(Frame {
                root: Mat4::IDENTITY,
                local_rotations: vec![Quat::IDENTITY, Quat::from_rotation_y(angle)],
            });
        }
        let clip = Clip {
            name: "spin".into(),
            frame_rate: 30.0,
            frames,
        };

        let prior = bake_prior(std::slice::from_ref(&clip));
        assert_eq!(prior.len(), 2);

        let goal = GoalDescriptor {
            target_local: Vec3::ZERO,
            approach_local: Vec3::Z,
            action_tag: None,
        };
        let static_bias = prior.bias(0, &goal);
        let animated_bias = prior.bias(1, &goal);

        // The animated bone's twist (Y) axis should be more compliant than the
        // static bone's same component.
        assert!(
            animated_bias.axis_weight.y > static_bias.axis_weight.y,
            "animated bone Y weight {} should exceed static bone Y weight {}",
            animated_bias.axis_weight.y,
            static_bias.axis_weight.y
        );
    }

    // ── Test 4: bake_prior_from_dir over a directory of JSON clips ───────────

    #[test]
    fn bake_prior_from_dir_counts_and_bakes() {
        let dir = std::env::temp_dir().join(format!(
            "cube_rig_extract_dir_{}_{}",
            std::process::id(),
            {
                use std::sync::atomic::{AtomicU64, Ordering};
                static C: AtomicU64 = AtomicU64::new(0);
                C.fetch_add(1, Ordering::Relaxed)
            }
        ));
        std::fs::create_dir_all(&dir).unwrap();

        let make_clip = |name: &str, angle: f32| Clip {
            name: name.into(),
            frame_rate: 30.0,
            frames: (0..5)
                .map(|i| Frame {
                    root: Mat4::IDENTITY,
                    local_rotations: vec![
                        Quat::IDENTITY,
                        Quat::from_rotation_y(angle * i as f32),
                    ],
                })
                .collect(),
        };

        for (name, angle) in [("a", 0.1), ("b", 0.2)] {
            let c = make_clip(name, angle);
            std::fs::write(
                dir.join(format!("{name}.json")),
                serde_json::to_vec_pretty(&c).unwrap(),
            )
            .unwrap();
        }

        let (prior, stats) = bake_prior_from_dir(&dir, 30.0).expect("bake from dir");
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(stats.files, 2);
        assert_eq!(stats.clips, 2);
        assert_eq!(stats.frames, 10);
        assert_eq!(stats.bones, 2);
        assert!(!prior.is_empty());
        assert_eq!(prior.len(), 2);
    }

    // ── Test 5: bake_prior_with_goals forms goal-conditioned cells ───────────

    #[test]
    fn bake_prior_with_goals_forms_cells() {
        // 3-bone arm: root, steerable shoulder (bone 1), effector tip (bone 2).
        let mut sk = Skeleton::new();
        let r = sk.add(Bone::root("root"));
        let s = sk.add(Bone::new("shoulder", Some(r), Mat4::IDENTITY));
        sk.add(Bone::new("tip", Some(s), Mat4::from_translation(Vec3::new(2.0, 0.0, 0.0))));

        let make = |posture: Quat| Clip {
            name: "c".into(),
            frame_rate: 30.0,
            frames: (0..6)
                .map(|_| Frame {
                    root: Mat4::IDENTITY,
                    local_rotations: vec![Quat::IDENTITY, posture, Quat::IDENTITY],
                })
                .collect(),
        };
        // Shoulder unrotated → tip toward +X; +90° about Z → tip toward +Y.
        let clip_x = make(Quat::IDENTITY);
        let clip_y = make(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2));

        let prior = bake_prior_with_goals(&[clip_x, clip_y], &sk, 2);

        assert!(prior.num_cells() >= 2, "expected ≥2 cells, got {}", prior.num_cells());
        assert!(!prior.is_empty(), "global fallback must still be populated");
    }
}
