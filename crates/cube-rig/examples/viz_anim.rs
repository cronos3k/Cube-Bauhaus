//! Animation demo: 320 frames (~5 s) of the real solver tracking a moving target
//! while a drifting obstacle pushes through its workspace. Emits per-frame joint
//! positions as JSON for an external renderer. Planar (XY).
//!
//! Run: `cargo run -p cube-rig --example viz_anim`

use cube_rig::collision::{avoid_obstacles, AvoidParams, Capsule};
use cube_rig::ik::{solve_dls, DlsParams, Pose};
use cube_rig::skeleton::{Bone, Skeleton};
use glam::{Mat4, Vec3};

fn arm(n: usize) -> (Skeleton, Vec<usize>, usize) {
    let mut sk = Skeleton::new();
    let mut chain = Vec::new();
    let r = sk.add(Bone::root("shoulder"));
    chain.push(r);
    let mut prev = r;
    for i in 1..n {
        prev = sk.add(Bone::new(format!("j{i}"), Some(prev), Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0))));
        chain.push(prev);
    }
    let hand = sk.add(Bone::new("hand", Some(prev), Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0))));
    (sk, chain, hand)
}

fn pts(pose: &Pose, sk: &Skeleton, chain: &[usize], eff: usize) -> String {
    let mut v: Vec<String> = chain
        .iter()
        .map(|&b| { let h = pose.head(sk, b); format!("[{:.3},{:.3}]", h.x, h.y) })
        .collect();
    let h = pose.head(sk, eff);
    v.push(format!("[{:.3},{:.3}]", h.x, h.y));
    format!("[{}]", v.join(","))
}

fn main() {
    let (sk, chain, eff) = arm(4);
    let mut pose = Pose::from_bind(&sk);
    let params = DlsParams::default();
    let bone_r = 0.26_f32;

    let frames = 320;
    let mut out: Vec<String> = Vec::new();
    for f in 0..frames {
        let t = f as f32 / frames as f32 * std::f32::consts::TAU;
        // Target: a Lissajous loop kept inside the arm's reach.
        let target = Vec3::new(1.7 + 1.45 * t.cos(), 1.55 * (2.0 * t).sin(), 0.0);
        // Obstacle: bobs vertically near x = 2.1, drifting through the arm.
        let oc = Vec3::new(2.1, 1.35 * (t * 0.7 + 1.0).sin(), 0.0);
        let obstacle = Capsule { a: oc + Vec3::new(0.0, 0.0, -0.6), b: oc + Vec3::new(0.0, 0.0, 0.6), radius: 0.42 };

        // Continuous: reach for the target, then dodge the obstacle.
        solve_dls(&mut pose, &sk, &chain, eff, target, None, None, &params);
        avoid_obstacles(&mut pose, &sk, &chain, bone_r, &[obstacle], None, &AvoidParams::default());

        out.push(format!(
            "{{\"j\":{},\"tgt\":[{:.3},{:.3}],\"obs\":[{:.3},{:.3},{:.3}]}}",
            pts(&pose, &sk, &chain, eff), target.x, target.y, oc.x, oc.y, obstacle.radius
        ));
    }

    println!("{{\"reach\":4.0,\"bone_r\":{:.3},\"frames\":[{}]}}", bone_r, out.join(","));
}
