//! End-to-end pipeline test for the goal-conditioned constrained-IK stack.
//!
//! Exercises every layer together through the public crate API, in the order the
//! design lays them out:
//!
//!   animation clip → source-agnostic `Clip` → baked `MotionPrior`
//!     → constrained DLS solve (hard limits, incl. the real Unitree G1 table)
//!     → capsule self-collision detection and obstacle resolution.
//!
//! These complement the per-module unit tests by proving the pieces compose.

use cube_rig::anim::{AnimationClip, BoneTrack};
use cube_rig::clip::Clip;
use cube_rig::collision::{avoid_obstacles, bone_capsules, capsule_penetration, AvoidParams, Capsule};
use cube_rig::ik::{solve_dls, DlsParams, Pose};
use cube_rig::limits::{self, g1_29dof_limits, JointLimit};
use cube_rig::prior::{GoalDescriptor, MotionPrior, MotionPriorBuilder};
use cube_rig::skeleton::{Bone, Skeleton};
use glam::{Mat4, Quat, Vec3};

/// A planar arm: a shoulder at the origin, then `n-1` unit bones along +X, plus a
/// final "hand" effector bone one unit past the last joint. Returns the skeleton,
/// the chain joint indices (root→tip) and the effector bone index.
fn arm_chain(n: usize) -> (Skeleton, Vec<usize>, usize) {
    let mut sk = Skeleton::new();
    let mut chain = Vec::new();
    let r = sk.add(Bone::root("shoulder"));
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
    let effector =
        sk.add(Bone::new("hand", Some(prev), Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0))));
    (sk, chain, effector)
}

fn dummy_goal() -> GoalDescriptor {
    GoalDescriptor { target_local: Vec3::ZERO, approach_local: Vec3::X, action_tag: None }
}

/// Bake a prior from a clip that swings `joint` about Z, leaving the rest at bind.
fn swing_prior(sk: &Skeleton, joint: usize) -> MotionPrior {
    let mut clip = AnimationClip::new("swing");
    let mut tr = BoneTrack::new(joint as u16);
    tr.rotation = vec![
        (0.0, Quat::from_rotation_z(-0.6)),
        (0.5, Quat::IDENTITY),
        (1.0, Quat::from_rotation_z(0.6)),
    ];
    clip.tracks.push(tr);

    let baked = Clip::from_animation(&clip, sk, 30.0);
    let mut builder = MotionPriorBuilder::new(sk.len());
    builder.add_clip(&baked);
    builder.build()
}

/// Count chain-bone capsules that penetrate a single static obstacle.
fn obstacle_penetrations(pose: &Pose, sk: &Skeleton, radius: f32, obs: &Capsule) -> usize {
    bone_capsules(pose, sk, radius)
        .iter()
        .flatten()
        .filter(|c| capsule_penetration(c, obs).is_some())
        .count()
}

#[test]
fn clip_bakes_into_anisotropic_prior() {
    let (sk, chain, _eff) = arm_chain(3);

    // Animate the middle joint swinging about Z; leave everything else at bind.
    let mut clip = AnimationClip::new("swing");
    let mut tr = BoneTrack::new(chain[1] as u16);
    tr.rotation = vec![
        (0.0, Quat::from_rotation_z(-0.6)),
        (0.5, Quat::IDENTITY),
        (1.0, Quat::from_rotation_z(0.6)),
    ];
    clip.tracks.push(tr);

    // animation clip → source-agnostic Clip (exercises clip.rs adapter).
    let baked = Clip::from_animation(&clip, &sk, 30.0);
    assert!(baked.len() > 10, "expected a sampled clip, got {} frames", baked.len());

    // Clip → MotionPrior (exercises prior.rs accumulation).
    let mut builder = MotionPriorBuilder::new(sk.len());
    builder.add_clip(&baked);
    let prior = builder.build();
    assert!(!prior.is_empty());

    let g = dummy_goal();
    let moved = prior.bias(chain[1], &g).axis_weight;
    let still = prior.bias(chain[0], &g).axis_weight;

    // The animated joint learns clear compliance on its swung axis; the static
    // root joint stays at the unit baseline.
    assert!(moved.max_element() > 1.2, "animated joint should learn compliance, got {moved:?}");
    assert!(
        still.max_element() <= 1.0 + 1e-3,
        "static joint should stay at baseline, got {still:?}"
    );
}

#[test]
fn solve_reaches_reachable_target_with_prior() {
    let (sk, chain, eff) = arm_chain(3);
    let prior = swing_prior(&sk, chain[1]);

    let mut pose = Pose::from_bind(&sk);
    let target = Vec3::new(1.5, 1.0, 0.0); // distance ~1.8, well within reach 3
    let dist =
        solve_dls(&mut pose, &sk, &chain, eff, target, Some(&prior), None, &DlsParams::default());

    assert!(dist < 1e-2, "prior-steered solve should reach the target, got distance {dist}");
}

#[test]
fn g1_elbow_limit_clamps_via_public_table() {
    // Pull the real Unitree G1 elbow limit out of the baked table and clamp a
    // beyond-range rotation against it.
    let table = g1_29dof_limits();
    assert_eq!(table.len(), 29);

    let elbow = table
        .iter()
        .find(|(n, _)| *n == "left_elbow_joint")
        .map(|(_, l)| l.clone())
        .expect("left_elbow_joint should be in the G1 table");

    let (axis, max) = match elbow {
        JointLimit::Hinge { axis, max, .. } => (axis, max),
        _ => panic!("the G1 elbow is a hinge"),
    };
    assert!(axis.abs_diff_eq(Vec3::Y, 1e-6), "G1 elbow hinges about Y");

    // 3 rad exceeds the +2.0944 max → clamps to the max.
    let clamped = limits::clamp(Quat::from_rotation_y(3.0), &elbow);
    let angle = clamped.to_scaled_axis().y;
    approx::assert_relative_eq!(angle, max, epsilon = 1e-4);
}

#[test]
fn solve_respects_tight_elbow_hinge() {
    let (sk, chain, eff) = arm_chain(3);
    let mut pose = Pose::from_bind(&sk);

    // A tight planar hinge on the middle joint.
    let (lo, hi) = (-0.25_f32, 0.25_f32);
    let mut lims: Vec<Option<JointLimit>> = vec![None; chain.len()];
    lims[1] = Some(JointLimit::Hinge { axis: Vec3::Z, min: lo, max: hi });

    // A target that demands a large bend the hinge must refuse.
    let target = Vec3::new(1.0, 1.8, 0.0);
    solve_dls(&mut pose, &sk, &chain, eff, target, None, Some(&lims), &DlsParams::default());

    // The bind rotation is identity, so the local rotation *is* the bind-relative
    // one. It must stay within the hinge range and on the hinge axis.
    let v = (pose.local[chain[1]].to_scale_rotation_translation().1).to_scaled_axis();
    assert!(v.z >= lo - 1e-3 && v.z <= hi + 1e-3, "elbow angle {} out of [{lo}, {hi}]", v.z);
    assert!(v.x.abs() < 1e-2 && v.y.abs() < 1e-2, "hinge should stay about Z, got {v:?}");
}

#[test]
fn straight_arm_resolves_against_obstacle() {
    let (sk, chain, _eff) = arm_chain(3);
    let mut pose = Pose::from_bind(&sk);
    let radius = 0.3;

    // A small obstacle sitting just above the arm near its far end, that the arm
    // can swing clear of by bending down. (A full vertical "wall" would be
    // geometrically unresolvable for a short three-bone arm.)
    let obstacle =
        Capsule { a: Vec3::new(2.0, 0.25, -0.5), b: Vec3::new(2.0, 0.25, 0.5), radius: 0.4 };

    let before = obstacle_penetrations(&pose, &sk, radius, &obstacle);
    assert!(before > 0, "the straight arm should penetrate the obstacle, got {before}");

    let remaining =
        avoid_obstacles(&mut pose, &sk, &chain, radius, &[obstacle], None, &AvoidParams::default());
    assert_eq!(remaining, 0, "the arm should be pushed clear, {remaining} penetrations remain");

    for i in 0..sk.len() {
        assert!(pose.head(&sk, i).is_finite(), "bone {i} head must stay finite");
    }
}
