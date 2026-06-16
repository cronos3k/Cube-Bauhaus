//! `extract_prior` — bake a [`MotionPrior`](cube_rig::MotionPrior) from a
//! directory of motion clips (§9's offline, source-agnostic extractor).
//!
//! Usage: `extract_prior <input_dir> <output.json> [--fps N]` (default fps 30).
//!
//! Walks `input_dir` recursively, loading every `.json` clip and every animation
//! in any `.gltf`/`.glb` file, bakes them into one prior, and writes it as pretty
//! JSON to `output.json`.

use std::path::Path;
use std::process::ExitCode;

use cube_rig::extract::bake_prior_from_dir;
use cube_rig::JointBias;

fn usage() -> ! {
    eprintln!("usage: extract_prior <input_dir> <output.json> [--fps N]");
    std::process::exit(2);
}

fn main() -> ExitCode {
    let mut positionals: Vec<String> = Vec::new();
    let mut fps: f32 = 30.0;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--fps" => {
                let Some(v) = args.next() else { usage() };
                match v.parse::<f32>() {
                    Ok(f) if f > 0.0 => fps = f,
                    _ => {
                        eprintln!("error: --fps expects a positive number, got {v:?}");
                        usage();
                    }
                }
            }
            "-h" | "--help" => usage(),
            other => positionals.push(other.to_string()),
        }
    }

    if positionals.len() != 2 {
        usage();
    }
    let input_dir = Path::new(&positionals[0]);
    let output = Path::new(&positionals[1]);

    let (prior, stats) = match bake_prior_from_dir(input_dir, fps) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: failed to bake prior from {}: {e}", input_dir.display());
            return ExitCode::FAILURE;
        }
    };

    let file = match std::fs::File::create(output) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: cannot create {}: {e}", output.display());
            return ExitCode::FAILURE;
        }
    };
    let writer = std::io::BufWriter::new(file);
    if let Err(e) = serde_json::to_writer_pretty(writer, &prior) {
        eprintln!("error: failed to write prior JSON: {e}");
        return ExitCode::FAILURE;
    }

    // Count bones whose bias differs from the graceful default.
    let default = JointBias::default();
    let non_default = (0..prior.len())
        .filter(|&b| {
            // bias() ignores the goal in v1; any goal works as a probe.
            let goal = cube_rig::GoalDescriptor {
                target_local: glam::Vec3::ZERO,
                approach_local: glam::Vec3::Z,
                action_tag: None,
            };
            let bias = prior.bias(b, &goal);
            !bias.axis_weight.abs_diff_eq(default.axis_weight, 1e-6)
        })
        .count();

    eprintln!(
        "baked prior: {} files, {} clips, {} frames, {} bones ({} with non-default axis_weight) → {}",
        stats.files,
        stats.clips,
        stats.frames,
        stats.bones,
        non_default,
        output.display(),
    );

    ExitCode::SUCCESS
}
