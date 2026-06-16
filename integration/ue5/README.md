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
  CubeRigIK.uplugin                       # UE 5.4 plugin descriptor (Runtime + Editor modules)
  Source/CubeRigIK/                        # RUNTIME module
    CubeRigIK.Build.cs                     # links cube_rig_ffi; deps: AnimationCore, AnimGraphRuntime
    Public/CubeRigIKModule.h               # IModuleInterface (loads the DLL on Win64)
    Public/CubeRigIKMarshal.h              # shared FMatrix->column-major / capsule packing helpers
    Public/CubeRigIKBlueprintLibrary.h     # UBlueprintFunctionLibrary: SolveArmIK(...) (one-shot)
    Public/CubeRigIKComponent.h            # UActorComponent: live per-frame driver (easiest path)
    Public/CubeRigWorldAdapter.h           # world primitives -> CrfCapsule obstacles
    Public/CubeRigGoalLibrary.h            # goal hooks: foot trace / actor reach
    Public/AnimNode_CubeRigIK.h            # FAnimNode_SkeletalControlBase (production apply path)
    Private/*.cpp                          # implementations of the above
  Source/CubeRigIKEditor/                  # EDITOR module (UncookedOnly)
    CubeRigIKEditor.Build.cs               # deps: AnimGraph, BlueprintGraph, UnrealEd, CubeRigIK
    Public/AnimGraphNode_CubeRigIK.h       # UAnimGraphNode_SkeletalControlBase wrapper
    Private/AnimGraphNode_CubeRigIK.cpp
    Private/CubeRigIKEditorModule.cpp      # IMPLEMENT_MODULE(FDefaultModuleImpl, ...)
  ThirdParty/CubeRigFFI/
    include/cube_rig.h                     # copy of crates/cube-rig-ffi/include/cube_rig.h
                                           #   (now includes CrfCapsule + crf_solve_chain_avoid)
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

## 4b. Live integration in a level

Section 4 shows the one-shot `SolveArmIK` call. This section is the **live**
path: components/nodes that feed gameplay + world data into the solver every
frame and apply the result, so a character actually tracks moving targets and
avoids obstacles in a running level. All of it is built in **UE 5.4** (not in
this repo).

### One space, one unit system (read this first)
Everything below works in the `SkeletalMeshComponent`'s **component space**, UE
units (**centimeters, left-handed, Z-up**). The solver does **no** conversion:
binds, the effector target, and obstacle capsules must all be in that one space.
The provided code converts world-space goals/obstacles into component space for
you via the mesh's component transform — if you feed your own data, match that.
If your prior/clips came from a different handedness or unit, convert first.

### A) `UCubeRigIKComponent` — drop-on-a-Character (easiest to run)
Data flow each tick: **harvest skeleton** (parents + bind locals + chain +
effector from the mesh) → **resolve goal** (world target → component space) →
**gather obstacles** (`UCubeRigWorldAdapter` → `CrfCapsule[]`) → **solve**
(`crf_solve_chain_avoid`) → **apply** (write solved local rotations back).

1. Add a **CubeRigIK** component to your Character (Add Component, or
   `CreateDefaultSubobject<UCubeRigIKComponent>` in C++).
2. Leave **Target Mesh** empty to auto-find the Character's mesh, or assign one.
3. Set **Chain Bone Names** to an arm chain root→tip, e.g.
   `upperarm_l, lowerarm_l`, and **Effector Bone Name** to `hand_l`.
4. Pick a goal: set **Target Actor** (+ optional **Target Socket**) to a moving
   actor, or drive **Effector Target World** each frame from Blueprint/C++.
5. Enable **Gather World Obstacles**, tune **Obstacle Query Radius** (cm) and
   **Obstacle Channel** (e.g. `WorldStatic`). Optionally list **Self Collision
   Bones** (e.g. spine/torso) so the arm avoids the body.
6. Bake a prior (section 4C) and put its absolute path in **Prior Json Path**.
7. Press Play and move the target — the arm tracks it and pushes off obstacles.

**Apply path / trade-off.** The component returns per-bone local rotations. With
**Apply To Mesh Directly = true** it writes the chain bones straight onto the
mesh each tick — simplest to see working, but it runs *after* the anim update,
is overwritten next frame, and does not blend with an AnimBP. For shipping, set
it **false** and consume `GetSolvedLocalRotation(BoneIndex)` from a post-process
`AnimInstance`, or use the AnimGraph node (B). **Bind Source** chooses reference
pose (stable, recommended) vs. the live current pose as the solve's starting
binds.

### B) `FAnimNode_CubeRigIK` — AnimGraph node (production path)
The proper home for the apply: it runs inside animation evaluation, so it blends
with the rest of the AnimBP and is thread-safe.

1. Open your character's **Animation Blueprint → AnimGraph**.
2. Add the **Cube Rig IK** node (category *Cube Rig IK*) into the pose stream.
3. Set its **Chain Bones** (`FBoneReference` pins) + **Effector Bone**, the
   **Prior Json Path**, **Iterations/Avoid Iterations/Bone Radius**.
4. Feed **Effector Target (Component)** — convert your world goal into the mesh's
   component space upstream (e.g. in the AnimBP's *Update*, take the goal from
   the goal helpers below and `InverseTransformPosition` with the component
   transform), then plug it into the pin.
5. (Optional) Fill the node's `Obstacles` array from native code in a
   game-thread step (the component or an AnimInstance NativeUpdate calling
   `UCubeRigWorldAdapter`), already in component space. The node itself cannot
   query the world (worker thread); empty obstacles = plain IK.

Thread-safety: `EvaluateSkeletalControl_AnyThread` touches only the pose + POD
inputs and the reentrant C ABI; the `CrfPrior*` is loaded once and read-only.

### C) Goal hooks — `UCubeRigGoalLibrary`
Helpers that turn live gameplay/world data into an effector goal (world space;
convert to component space before solving):

- **Trace Foot Goal** — line-traces from above the animated foot down to the
  ground; use its hit point as the effector target so feet plant on terrain.
- **Reach Goal From Actor** — world location of a socket/actor (a lever, a held
  prop, an NPC hand) to use as a hand-reach target.

Wire either into the component's **Effector Target World** or the AnimNode's
target pin (after converting to component space).

### D) World → capsule adapter — `UCubeRigWorldAdapter`
Static helpers the component uses, callable directly from C++:
- `GatherObstacles(World, SelfMesh, Origin, Radius, Channel, Out)` overlaps a
  sphere and converts hits to `CrfCapsule` in component space (capsule
  components map exactly; other primitives are approximated from bounds).
- `GatherSelfCollisionCapsules(Mesh, BoneNames, Radius, Out)` builds bone-segment
  capsules from the current pose for self-collision (arm vs torso).

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
