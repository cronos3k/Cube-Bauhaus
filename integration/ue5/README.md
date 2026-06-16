# Cube Rig IK — Unreal Engine 5 integration

This directory bridges the engine-agnostic `cube-rig` IK solver into Unreal
Engine 5 via the `cube-rig-ffi` C-ABI crate.

> **Status: uncompiled-here scaffold.** The Rust side (`crates/cube-rig-ffi`)
> compiles and is unit-tested in this repository. The UE5 plugin under
> `CubeRigIK/` **cannot be compiled in this repo** — it requires the Unreal
> Engine headers and build system. It is written to be correct and idiomatic so
> you can drop it into a UE 5.4 project and build it there. Everything below is
> the step-by-step runbook for doing exactly that.

---

## 0. Layout

```
integration/ue5/CubeRigIK/
  CubeRigIK.uplugin                       # UE 5.4 plugin descriptor (Runtime module)
  Source/CubeRigIK/
    CubeRigIK.Build.cs                     # links cube_rig_ffi, adds the include dir
    Public/CubeRigIKModule.h               # IModuleInterface (loads the DLL on Win64)
    Public/CubeRigIKBlueprintLibrary.h     # UBlueprintFunctionLibrary: SolveArmIK(...)
    Private/CubeRigIKModule.cpp
    Private/CubeRigIKBlueprintLibrary.cpp  # marshals UE types -> C arrays -> crf_solve_chain
  ThirdParty/CubeRigFFI/
    include/cube_rig.h                     # copy of crates/cube-rig-ffi/include/cube_rig.h
    lib/Win64/   lib/Linux/   lib/Mac/     # <- drop the built libraries here (see step 2)
```

## 1. Build the Rust FFI library

From the repository root:

```sh
cargo build -p cube-rig-ffi --release
```

The artifacts land in `target/release/` (workspace target dir):

| Platform | Shared library         | Companion / import lib        | Static library         |
|----------|------------------------|-------------------------------|------------------------|
| Linux    | `libcube_rig_ffi.so`   | —                             | `libcube_rig_ffi.a`    |
| macOS    | `libcube_rig_ffi.dylib`| —                             | `libcube_rig_ffi.a`    |
| Windows  | `cube_rig_ffi.dll`     | `cube_rig_ffi.dll.lib`        | `cube_rig_ffi.lib`     |

Build on the **same OS/arch you will run UE on** (cross-compiling Rust for UE
targets is possible but out of scope here). On Windows, build from a
`x86_64-pc-windows-msvc` toolchain so the import lib is MSVC-compatible.

> The `[lib] crate-type` is `["cdylib", "staticlib", "rlib"]`, so a single
> `cargo build` produces both the shared and static libraries. Choose one to
> link (the `Build.cs` defaults to the shared/DLL path; static-lib lines are
> present but commented).

## 2. Stage the library + header in the plugin

Copy the built files into the plugin's `ThirdParty/CubeRigFFI/lib/<platform>/`:

```sh
# Linux
cp target/release/libcube_rig_ffi.so \
   integration/ue5/CubeRigIK/ThirdParty/CubeRigFFI/lib/Linux/

# macOS
cp target/release/libcube_rig_ffi.dylib \
   integration/ue5/CubeRigIK/ThirdParty/CubeRigFFI/lib/Mac/

# Windows (from a shell with the files present)
copy target\release\cube_rig_ffi.dll     integration\ue5\CubeRigIK\ThirdParty\CubeRigFFI\lib\Win64\
copy target\release\cube_rig_ffi.dll.lib integration\ue5\CubeRigIK\ThirdParty\CubeRigFFI\lib\Win64\
```

The header `ThirdParty/CubeRigFFI/include/cube_rig.h` is already a copy of
`crates/cube-rig-ffi/include/cube_rig.h`. If you change the FFI signatures,
re-copy it:

```sh
cp crates/cube-rig-ffi/include/cube_rig.h \
   integration/ue5/CubeRigIK/ThirdParty/CubeRigFFI/include/cube_rig.h
```

## 3. Add the plugin to a UE 5.4 project

1. Copy (or symlink) `integration/ue5/CubeRigIK/` into your project's
   `Plugins/` directory: `<YourProject>/Plugins/CubeRigIK/`.
2. Enable it in `<YourProject>.uproject` (or via **Edit → Plugins** in-editor):

   ```json
   "Plugins": [
       { "Name": "CubeRigIK", "Enabled": true }
   ]
   ```

3. Regenerate project files (right-click the `.uproject` → *Generate Visual
   Studio / Xcode project files*, or `Build/BatchFiles/<Platform>/GenerateProjectFiles`).
4. Build the project (the editor target). UBT picks up `CubeRigIK.Build.cs`,
   adds the include dir and links the staged library. On Win64 the DLL is staged
   as a `RuntimeDependency` and delay-loaded by `FCubeRigIKModule::StartupModule`.

## 4. Minimal in-editor test recipe

The goal: drive a SkeletalMesh arm chain's effector to a moving target and watch
the arm track it.

### Inputs you provide (component space)
`SolveArmIK` is space-agnostic; feed everything in one consistent space —
component space of your `USkeletalMeshComponent` is the natural choice.

- **BoneBinds** — one `FTransform` per bone (the rest/bind pose, component space).
  You can pull these from `GetBoneSpaceTransforms` composed up the hierarchy, or
  author a small fixed arm chain for a first test.
- **Parents** — parent index per bone (`-1` for the root). Must be topologically
  ordered (every parent index `<` its child index — reorder if your skeleton
  isn't already).
- **Chain** — the root→tip bone indices to rotate (e.g. shoulder, elbow).
- **Effector** — usually the hand/wrist bone (a descendant of the last chain bone).
- **Target** — an `FVector` you move each tick (e.g. a target actor's location
  converted into the mesh's component space).

### A) Blueprint recipe
1. Add an Actor with a `SkeletalMeshComponent` for an arm (or a simple 2–3 bone
   test skeleton).
2. On **Event Tick**:
   - Build the `BoneBinds` / `Parents` / `Chain` arrays (you can cache the static
     ones once on BeginPlay; only `Target` changes per tick).
   - Convert your moving target actor's world location into the mesh's component
     space (`Transform Location` with the inverse of the component transform).
   - Call **Solve Arm IK** (category *Cube Rig IK*) with those inputs and an
     empty *Prior Json Path*.
   - For each chain/effector bone, apply `OutRotations[i]` to the pose — the
     simplest path is a **Transform (Modify) Bone** node per chain bone in an
     Animation Blueprint's *AnimGraph*, fed the matching `OutRotations` quat in
     **Bone Space = Component Space**, replacing rotation. (For a quick smoke
     test you can instead drive a chain of debug arrows / scene components from
     `OutRotations` to confirm the solve before wiring the AnimBP.)
3. Press Play and move the target actor — the arm should follow.

### B) C++ AActor recipe (equivalent, fewer nodes)
In an `AActor::Tick`:

```cpp
TArray<FTransform> BoneBinds = /* component-space bind transforms */;
TArray<int32>      Parents   = /* -1 for root, topologically ordered */;
TArray<int32>      Chain     = { ShoulderIdx, ElbowIdx };
int32              Effector  = WristIdx;
FVector            Target    = MeshComp->GetComponentTransform()
                                 .InverseTransformPosition(TargetActor->GetActorLocation());

TArray<FQuat> OutRot;
if (UCubeRigIKBlueprintLibrary::SolveArmIK(
        BoneBinds, Parents, Chain, Effector, Target,
        /*PriorJsonPath*/ TEXT(""), /*Iterations*/ 32, OutRot))
{
    // Apply OutRot[ChainBone] in component space (e.g. via a control rig /
    // skeletal mesh modify-bone, or your own pose buffer).
}
```

### C) Add a motion prior (steering)
1. Bake a prior from motion clips with the `extract_prior` binary:

   ```sh
   cargo run -p cube-rig --bin extract_prior -- <clips_dir> <out.json>
   ```

   (See `crates/cube-rig/src/bin/extract_prior.rs` for its exact CLI.) The
   output is a `MotionPrior` JSON.
2. Place `out.json` somewhere readable (e.g. your project's `Content/` or a
   `Saved/` path) and pass its **absolute** path as `PriorJsonPath` to
   `SolveArmIK`.
3. Re-run the same moving-target test. With the prior, joints whose recorded
   motion had higher variance become more compliant, so the arm now reaches the
   same targets with a posture biased toward the clips it was baked from — you
   should see the elbow/shoulder favor the trained range. An empty path keeps the
   uniform (unsteered) behavior for an A/B comparison.

## 5. Conventions & gotchas

- **Matrix layout:** `cube_rig.h` documents column-major `Mat4`. The plugin's
  `PackColumnMajor` converts UE's `FMatrix` (row-major storage, row-vector math)
  into that layout. If you bypass `SolveArmIK` and call the C ABI directly, mind
  the transpose.
- **Quaternions** are `(x, y, z, w)` on both sides — `FQuat` matches directly.
- **Topological order** is mandatory: `crf_solve_chain` returns
  `CRF_ERR_BAD_SKELETON` if any parent index is `>=` its child's index.
- **Handedness / units:** cube-rig does no coordinate conversion. UE is
  left-handed, Z-up, centimeters; if your clips/targets came from a different
  convention, convert before calling.
- **Thread-safety:** `crf_solve_chain` is reentrant and allocates only locally;
  it is safe to call from worker threads. A `CrfPrior*` handle is read-only once
  loaded and may be shared across threads, but free it exactly once.
- **Panic safety:** every C entry point is wrapped in `catch_unwind`, so a Rust
  panic surfaces as an error code / null handle, never an unwind across the ABI.

## 6. Platform library names (quick reference)

| Platform | Stage into `ThirdParty/CubeRigFFI/lib/...` |
|----------|--------------------------------------------|
| Win64    | `Win64/cube_rig_ffi.dll` + `Win64/cube_rig_ffi.dll.lib` (or `Win64/cube_rig_ffi.lib` for static) |
| Linux    | `Linux/libcube_rig_ffi.so` (or `Linux/libcube_rig_ffi.a` for static) |
| macOS    | `Mac/libcube_rig_ffi.dylib` (or `Mac/libcube_rig_ffi.a` for static) |
