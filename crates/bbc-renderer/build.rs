//! Build script: compile GLSL shaders to SPIR-V via glslc (Vulkan SDK).

use std::process::Command;
use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir  = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // (source file, stage flag for glslc, output spv name)
    let shaders = [
        ("cube.vert.glsl", "vertex",   "cube.vert.spv"),
        ("cube.frag.glsl", "fragment", "cube.frag.spv"),
        ("wire.vert.glsl", "vertex",   "wire.vert.spv"),
        ("wire.frag.glsl", "fragment", "wire.frag.spv"),
        ("egui.vert.glsl", "vertex",   "egui.vert.spv"),
        ("egui.frag.glsl", "fragment", "egui.frag.spv"),
    ];

    let glslc = find_glslc();

    for (src_name, stage, dst_name) in &shaders {
        let src = manifest.join("src/shaders").join(src_name);
        let dst = out_dir.join(dst_name);

        println!("cargo:rerun-if-changed={}", src.display());

        let stage_flag = format!("-fshader-stage={stage}");
        let status = Command::new(&glslc)
            .args([
                stage_flag.as_str(),
                src.to_str().unwrap(),
                "-o",
                dst.to_str().unwrap(),
            ])
            .status()
            .unwrap_or_else(|e| panic!("Failed to run glslc for {src_name}: {e}"));

        if !status.success() {
            panic!("Shader compilation failed for {src_name}");
        }
        println!("cargo:warning=Compiled shader: {src_name} → {dst_name}");
    }
}

fn find_glslc() -> String {
    // 1. Check PATH
    let test = if cfg!(windows) { "glslc.exe" } else { "glslc" };
    if Command::new(test).arg("--version").output().is_ok() {
        return test.to_string();
    }

    // 2. VULKAN_SDK environment variable (set by Vulkan SDK installer)
    if let Ok(sdk) = std::env::var("VULKAN_SDK") {
        let candidate = if cfg!(windows) {
            format!("{sdk}/Bin/glslc.exe")
        } else {
            format!("{sdk}/bin/glslc")
        };
        if std::path::Path::new(&candidate).exists() {
            return candidate;
        }
    }

    // 3. Common Windows Vulkan SDK paths
    #[cfg(windows)]
    {
        for drive in ["C", "D", "E", "F"] {
            for version in ["1.4.304.1", "1.4.304.0", "1.3.296.0", "1.3.283.0", "1.3.268.0", "1.3.261.1"] {
                let path = format!("{drive}:\\VulkanSDK\\{version}\\Bin\\glslc.exe");
                if std::path::Path::new(&path).exists() {
                    return path;
                }
            }
        }
    }

    panic!(
        "glslc not found.\n\
         Install the Vulkan SDK from https://vulkan.lunarg.com/sdk/home\n\
         or set the VULKAN_SDK environment variable."
    );
}
