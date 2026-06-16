//! Visualization demo: run the real solver/collision over scripted scenarios and
//! emit joint positions as JSON for an external renderer. Planar (XY) for clarity.
//!
//! Run: `cargo run -p cube-rig --example viz_demo`

use cube_rig::collision::{avoid_obstacles, bone_capsules, capsule_penetration, AvoidParams, Capsule};
use cube_rig::ik::{solve_dls, DlsParams, Pose};
use cube_rig::skeleton::{Bone, Skeleton};
use glam::{Mat4, Vec3};

/// A planar arm: shoulder at origin + `n` unit bones along +X, plus a hand tip.
fn arm(n: usize) -> (Skeleton, Vec<usize>, usize) {
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
    let hand = sk.add(Bone::new("hand", Some(prev), Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0))));
    (sk, chain, hand)
}

/// XY positions of every chain joint head plus the effector head.
fn points(pose: &Pose, sk: &Skeleton, chain: &[usize], eff: usize) -> Vec<[f32; 2]> {
    let mut p: Vec<[f32; 2]> = chain
        .iter()
        .map(|&b| {
            let h = pose.head(sk, b);
            [h.x, h.y]
        })
        .collect();
    let h = pose.head(sk, eff);
    p.push([h.x, h.y]);
    p
}

fn json_pts(p: &[[f32; 2]]) -> String {
    let inner: Vec<String> = p.iter().map(|q| format!("[{:.4},{:.4}]", q[0], q[1])).collect();
    format!("[{}]", inner.join(","))
}

fn main() {
    // ── Scenario A: track a target looping in the reachable area ──────────────
    let (sk, chain, eff) = arm(4); // reach 4
    let mut pose = Pose::from_bind(&sk);
    let params = DlsParams::default();

    let frames = 48;
    let mut track_poses = Vec::new();
    let mut track_targets = Vec::new();
    for f in 0..frames {
        let t = (f as f32 / frames as f32) * std::f32::consts::TAU;
        // A leaning figure-eight the hand chases, kept within reach.
        let target = Vec3::new(2.0 + 1.1 * t.cos(), 1.3 * (2.0 * t).sin(), 0.0);
        solve_dls(&mut pose, &sk, &chain, eff, target, None, None, &params);
        track_poses.push(json_pts(&points(&pose, &sk, &chain, eff)));
        track_targets.push(format!("[{:.4},{:.4}]", target.x, target.y));
    }

    // ── Scenario B: reach past an obstacle, naive vs collision-resolved ───────
    let (sk2, chain2, eff2) = arm(4);
    let radius = 0.28;
    let target_b = Vec3::new(3.3, 0.05, 0.0);

    let mut naive = Pose::from_bind(&sk2);
    solve_dls(&mut naive, &sk2, &chain2, eff2, target_b, None, None, &params);
    let naive_pts = points(&naive, &sk2, &chain2, eff2);

    // Plant the obstacle squarely on a mid-arm bone of the *naive* solution, so
    // the naive pose genuinely penetrates it and the resolver has real work to do
    // (rather than staging a collision that never happens).
    let mid = [
        (naive_pts[1][0] + naive_pts[2][0]) * 0.5,
        (naive_pts[1][1] + naive_pts[2][1]) * 0.5,
    ];
    let obstacle =
        Capsule { a: Vec3::new(mid[0], mid[1], -0.7), b: Vec3::new(mid[0], mid[1], 0.7), radius: 0.4 };
    let naive_pen = bone_capsules(&naive, &sk2, radius)
        .iter()
        .flatten()
        .filter(|c| capsule_penetration(c, &obstacle).is_some())
        .count();

    let mut resolved = naive.clone();
    let remaining = avoid_obstacles(
        &mut resolved,
        &sk2,
        &chain2,
        radius,
        &[obstacle],
        None,
        &AvoidParams::default(),
    );
    let resolved_pts = points(&resolved, &sk2, &chain2, eff2);

    // ── Emit JSON ─────────────────────────────────────────────────────────────
    println!("{{");
    println!("  \"reach\": 4.0,");
    println!("  \"track_poses\": [{}],", track_poses.join(","));
    println!("  \"track_targets\": [{}],", track_targets.join(","));
    println!("  \"obstacle\": {{\"a\":[{:.3},{:.3}],\"b\":[{:.3},{:.3}],\"r\":{:.3},\"bone_r\":{:.3}}},",
        obstacle.a.x, obstacle.a.y, obstacle.b.x, obstacle.b.y, obstacle.radius, radius);
    println!("  \"target_b\": [{:.3},{:.3}],", target_b.x, target_b.y);
    println!("  \"naive\": {},", json_pts(&naive_pts));
    println!("  \"naive_penetrations\": {},", naive_pen);
    println!("  \"resolved\": {},", json_pts(&resolved_pts));
    println!("  \"resolved_penetrations\": {}", remaining);
    println!("}}");
}
