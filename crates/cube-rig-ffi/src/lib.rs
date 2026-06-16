//! C-ABI bridge that makes the `cube-rig` IK solver callable from native
//! engines (Unreal Engine 5, Unity native plugins, raw C/C++).
//!
//! # Design rules
//!
//! * Every exported function is `#[no_mangle] pub extern "C"` and is
//!   **panic-safe**: its body runs inside [`std::panic::catch_unwind`] so a Rust
//!   panic can never unwind across the FFI boundary (which is undefined
//!   behavior). On panic the function returns an error code (or a null pointer).
//! * Arrays are passed as raw pointer + length pairs; the caller owns the
//!   memory. We never take ownership of caller arrays and never hand back Rust
//!   `Vec`s — outputs are written into caller-provided buffers.
//! * Owned Rust objects (the motion prior) are exposed as **opaque boxed
//!   handles**: the C side only sees a `CrfPrior*` and must release it with the
//!   matching `crf_*_free` function.
//!
//! The companion hand-authored header is `include/cube_rig.h`; keep the two in
//! lock-step when changing any signature.

use std::ffi::{c_char, c_int, CStr};
use std::os::raw::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};

use cube_rig::ik::{solve_dls, DlsParams, Pose};
use cube_rig::limits::JointLimit;
use cube_rig::prior::MotionPrior;
use cube_rig::skeleton::{Bone, Skeleton};
use glam::{Mat4, Quat, Vec3};

/// Sentinel parent index meaning "this bone is a root". Mirrors `UINT64_MAX` in
/// the C header.
pub const CRF_NO_PARENT: u64 = u64::MAX;

// ── Error codes (kept in sync with cube_rig.h) ──────────────────────────────

/// Success.
pub const CRF_OK: c_int = 0;
/// A required pointer argument was null.
pub const CRF_ERR_NULL_POINTER: c_int = 1;
/// A length / index argument was inconsistent or out of range.
pub const CRF_ERR_INVALID_ARG: c_int = 2;
/// The skeleton or chain failed validation (bad topology, bad indices).
pub const CRF_ERR_BAD_SKELETON: c_int = 3;
/// A Rust panic was caught at the FFI boundary.
pub const CRF_ERR_PANIC: c_int = 4;

// ── Opaque prior handle ─────────────────────────────────────────────────────

/// Opaque handle wrapping an owned [`MotionPrior`]. C code only ever sees a
/// `*mut CrfPrior`; the layout is private.
pub struct CrfPrior {
    inner: MotionPrior,
}

/// Load a [`MotionPrior`] from a JSON file produced by `extract_prior`.
///
/// Returns a heap-allocated handle on success, or null on any failure (null
/// path, bad UTF-8, missing file, malformed JSON, or panic). Free the result
/// with [`crf_prior_free`].
///
/// # Safety
/// `path` must be a valid, NUL-terminated C string or null.
#[no_mangle]
pub unsafe extern "C" fn crf_prior_load_json(path: *const c_char) -> *mut CrfPrior {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if path.is_null() {
            return std::ptr::null_mut();
        }
        let c_path = unsafe { CStr::from_ptr(path) };
        let Ok(path_str) = c_path.to_str() else {
            return std::ptr::null_mut();
        };
        let Ok(bytes) = std::fs::read(path_str) else {
            return std::ptr::null_mut();
        };
        prior_from_bytes(&bytes)
    }));
    result.unwrap_or(std::ptr::null_mut())
}

/// Load a [`MotionPrior`] from an in-memory JSON buffer (same format as
/// [`crf_prior_load_json`]). Returns null on failure. Free with
/// [`crf_prior_free`].
///
/// # Safety
/// `ptr` must point to at least `len` readable bytes, or be null (returns null).
#[no_mangle]
pub unsafe extern "C" fn crf_prior_load_json_bytes(ptr: *const u8, len: usize) -> *mut CrfPrior {
    let result = catch_unwind(AssertUnwindSafe(|| {
        if ptr.is_null() {
            return std::ptr::null_mut();
        }
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        prior_from_bytes(bytes)
    }));
    result.unwrap_or(std::ptr::null_mut())
}

/// Parse a `MotionPrior` from JSON bytes into a boxed handle, or null on error.
fn prior_from_bytes(bytes: &[u8]) -> *mut CrfPrior {
    match serde_json::from_slice::<MotionPrior>(bytes) {
        Ok(inner) => Box::into_raw(Box::new(CrfPrior { inner })),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Release a prior handle previously returned by `crf_prior_load_json*`.
/// Passing null is a no-op. Double-free is undefined behavior.
///
/// # Safety
/// `prior` must be a handle from this library that has not already been freed,
/// or null.
#[no_mangle]
pub unsafe extern "C" fn crf_prior_free(prior: *mut CrfPrior) {
    if prior.is_null() {
        return;
    }
    // Catch a panic in Drop so it never crosses the boundary.
    let _ = catch_unwind(AssertUnwindSafe(|| {
        drop(unsafe { Box::from_raw(prior) });
    }));
}

// ── Core solve entrypoint ───────────────────────────────────────────────────

/// Solve an IK chain so `effector_bone`'s head reaches `target`, writing every
/// bone's solved local rotation (xyzw) into `out_local_rotations`.
///
/// All matrices are column-major `Mat4` (glam / glTF / UE convention via
/// `Mat4::from_cols_array`). Returns [`CRF_OK`] on success or a nonzero error
/// code otherwise (see the `CRF_ERR_*` constants).
///
/// `prior` may be null (uniform compliance). `hinge_limits` may be null (no
/// limits — the v1 supported path). When non-null it is `chain_len * 4` floats:
/// per chain joint `[axis_x, axis_y, axis_z, half_range]`, applied as a hinge
/// about the (normalized) axis with `min = -half_range`, `max = +half_range`
/// relative to that bone's bind rotation. A non-finite or zero axis disables the
/// limit for that joint. Richer (asymmetric / swing-twist) limits are a
/// follow-up; the correct, fully-tested path is `hinge_limits == null`.
///
/// # Safety
/// Every non-null pointer must point to a buffer of the documented length:
/// `parents` and `local_bind` of `bone_count` (×16 for `local_bind`), `chain`
/// of `chain_len`, `target` of 3, `hinge_limits` of `chain_len*4` (or null),
/// and `out_local_rotations` of `bone_count*4`.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn crf_solve_chain(
    parents: *const u64,
    local_bind: *const f32,
    bone_count: usize,
    chain: *const u64,
    chain_len: usize,
    effector_bone: u64,
    target: *const f32,
    prior: *const CrfPrior,
    hinge_limits: *const f32,
    iterations: c_int,
    out_local_rotations: *mut f32,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        solve_chain_inner(
            parents,
            local_bind,
            bone_count,
            chain,
            chain_len,
            effector_bone,
            target,
            prior,
            hinge_limits,
            iterations,
            out_local_rotations,
        )
    }));
    result.unwrap_or(CRF_ERR_PANIC)
}

/// The actual solve, factored out so `crf_solve_chain` is just the panic guard.
///
/// # Safety
/// Same contract as [`crf_solve_chain`].
#[allow(clippy::too_many_arguments)]
unsafe fn solve_chain_inner(
    parents: *const u64,
    local_bind: *const f32,
    bone_count: usize,
    chain: *const u64,
    chain_len: usize,
    effector_bone: u64,
    target: *const f32,
    prior: *const CrfPrior,
    hinge_limits: *const f32,
    iterations: c_int,
    out_local_rotations: *mut f32,
) -> c_int {
    if out_local_rotations.is_null() {
        return CRF_ERR_NULL_POINTER;
    }
    // Run the shared validate + rebuild + solve path.
    let (skeleton, pose, _chain_idx, _limits) = match unsafe {
        solve_into(
            parents,
            local_bind,
            bone_count,
            chain,
            chain_len,
            effector_bone,
            target,
            prior,
            hinge_limits,
            iterations,
        )
    } {
        Ok(v) => v,
        Err(code) => return code,
    };

    // Write every bone's solved local rotation (xyzw) into the output buffer.
    unsafe { write_local_rotations(&pose, &skeleton, out_local_rotations) };

    CRF_OK
}

/// The solved data shared between the plain and obstacle-aware entrypoints: the
/// rebuilt skeleton, the solved pose, the translated chain indices, and the
/// decoded optional hinge limits.
type SolveResult = (Skeleton, Pose, Vec<usize>, Option<Vec<Option<JointLimit>>>);

/// Shared core for [`crf_solve_chain`] and [`crf_solve_chain_avoid`]: validate
/// the flat inputs, rebuild the [`Skeleton`], decode the optional hinge limits,
/// run [`solve_dls`], and hand back the solved `(Skeleton, Pose)` together with
/// the translated chain indices and decoded limits so the caller can do further
/// work (e.g. obstacle avoidance) against the exact same data.
///
/// Returns `Err(code)` with the appropriate `CRF_ERR_*` on any bad input so both
/// entrypoints reject identically and cannot drift.
///
/// # Safety
/// All non-null pointers must satisfy the buffer-length contract documented on
/// [`crf_solve_chain`].
#[allow(clippy::too_many_arguments)]
unsafe fn solve_into(
    parents: *const u64,
    local_bind: *const f32,
    bone_count: usize,
    chain: *const u64,
    chain_len: usize,
    effector_bone: u64,
    target: *const f32,
    prior: *const CrfPrior,
    hinge_limits: *const f32,
    iterations: c_int,
) -> Result<SolveResult, c_int> {
    // Null checks for all required buffers.
    if parents.is_null() || local_bind.is_null() || chain.is_null() || target.is_null() {
        return Err(CRF_ERR_NULL_POINTER);
    }
    if bone_count == 0 || chain_len == 0 {
        return Err(CRF_ERR_INVALID_ARG);
    }

    let parents = unsafe { std::slice::from_raw_parts(parents, bone_count) };
    let local_bind = unsafe { std::slice::from_raw_parts(local_bind, bone_count * 16) };
    let chain_raw = unsafe { std::slice::from_raw_parts(chain, chain_len) };
    let target_raw = unsafe { std::slice::from_raw_parts(target, 3) };

    // Rebuild a Skeleton from flat arrays.
    let mut skeleton = Skeleton::new();
    for (i, &p) in parents.iter().enumerate() {
        let parent = if p == CRF_NO_PARENT {
            None
        } else {
            let pu = p as usize;
            // Topological order is required by the solver's global() recursion.
            if pu >= bone_count || pu >= i {
                return Err(CRF_ERR_BAD_SKELETON);
            }
            Some(pu)
        };
        let m = Mat4::from_cols_array(
            <&[f32; 16]>::try_from(&local_bind[i * 16..i * 16 + 16]).unwrap(),
        );
        skeleton.add(Bone::new(format!("b{i}"), parent, m));
    }
    if skeleton.validate().is_err() {
        return Err(CRF_ERR_BAD_SKELETON);
    }

    // Translate the chain indices, bounds-checking each.
    let mut chain_idx: Vec<usize> = Vec::with_capacity(chain_len);
    for &c in chain_raw {
        let ci = c as usize;
        if c == CRF_NO_PARENT || ci >= bone_count {
            return Err(CRF_ERR_INVALID_ARG);
        }
        chain_idx.push(ci);
    }
    let effector = effector_bone as usize;
    if effector_bone == CRF_NO_PARENT || effector >= bone_count {
        return Err(CRF_ERR_INVALID_ARG);
    }

    let target_v = Vec3::new(target_raw[0], target_raw[1], target_raw[2]);

    // Optional hinge limits, parallel to the chain.
    let limits: Option<Vec<Option<JointLimit>>> = if hinge_limits.is_null() {
        None
    } else {
        let raw = unsafe { std::slice::from_raw_parts(hinge_limits, chain_len * 4) };
        let mut v = Vec::with_capacity(chain_len);
        for k in 0..chain_len {
            let axis = Vec3::new(raw[k * 4], raw[k * 4 + 1], raw[k * 4 + 2]);
            let half = raw[k * 4 + 3];
            if !axis.is_finite() || !half.is_finite() || axis.length_squared() < 1e-12 {
                v.push(None);
            } else {
                v.push(Some(JointLimit::Hinge {
                    axis: axis.normalize(),
                    min: -half.abs(),
                    max: half.abs(),
                }));
            }
        }
        Some(v)
    };

    // Resolve the optional prior handle.
    let prior_ref: Option<&MotionPrior> = if prior.is_null() {
        None
    } else {
        Some(unsafe { &(*prior).inner })
    };

    // Solve.
    let mut pose = Pose::from_bind(&skeleton);
    let params = DlsParams {
        iterations: iterations.max(1) as usize,
        ..DlsParams::default()
    };
    let _final_dist = solve_dls(
        &mut pose,
        &skeleton,
        &chain_idx,
        effector,
        target_v,
        prior_ref,
        limits.as_deref(),
        &params,
    );

    Ok((skeleton, pose, chain_idx, limits))
}

/// Write every bone's solved local rotation (xyzw) from `pose` into the
/// caller-provided `out` buffer (`skeleton.len() * 4` floats).
///
/// # Safety
/// `out` must point to a writable buffer of at least `skeleton.len() * 4` floats.
unsafe fn write_local_rotations(pose: &Pose, skeleton: &Skeleton, out: *mut f32) {
    let out = unsafe { std::slice::from_raw_parts_mut(out, skeleton.len() * 4) };
    for (i, m) in pose.local.iter().enumerate() {
        let q: Quat = m.to_scale_rotation_translation().1;
        out[i * 4] = q.x;
        out[i * 4 + 1] = q.y;
        out[i * 4 + 2] = q.z;
        out[i * 4 + 3] = q.w;
    }
}

// ── Obstacle-aware solve entrypoint ─────────────────────────────────────────

/// A capsule obstacle in model space: the set of points within `radius` of the
/// segment from `a` to `b`. Mirrors `cube_rig::collision::Capsule` and the
/// `CrfCapsule` struct in `cube_rig.h`.
#[repr(C)]
pub struct CrfCapsule {
    /// Segment start (x, y, z).
    pub a: [f32; 3],
    /// Segment end (x, y, z).
    pub b: [f32; 3],
    /// Sweep radius.
    pub radius: f32,
}

/// Solve an IK chain to `target` exactly like [`crf_solve_chain`], THEN push the
/// chain's bone capsules out of the static `obstacles` using the collision
/// resolver, writing every bone's final local rotation (xyzw) into
/// `out_local_rotations`.
///
/// The solve half shares the *exact* same code path as [`crf_solve_chain`] (both
/// call the private `solve_into` helper), so with `obstacle_count == 0` and a
/// null `obstacles` pointer the result is identical to `crf_solve_chain`.
///
/// After solving, each bone in `chain` is approximated by a capsule of radius
/// `bone_radius` and rotated out of the `obstacles` via
/// `cube_rig::collision::avoid_obstacles` (running `avoid_iterations`, clamped to
/// `>= 1`). The optional `hinge_limits` are decoded once and applied to both the
/// solve and the avoidance step.
///
/// Returns [`CRF_OK`] on success or a nonzero `CRF_ERR_*` code. `obstacles` may
/// be null only when `obstacle_count == 0`.
///
/// # Safety
/// Same buffer-length contract as [`crf_solve_chain`], plus: when
/// `obstacle_count > 0`, `obstacles` must point to `obstacle_count` `CrfCapsule`s.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn crf_solve_chain_avoid(
    parents: *const u64,
    local_bind: *const f32,
    bone_count: usize,
    chain: *const u64,
    chain_len: usize,
    effector_bone: u64,
    target: *const f32,
    prior: *const CrfPrior,
    hinge_limits: *const f32,
    iterations: c_int,
    obstacles: *const CrfCapsule,
    obstacle_count: usize,
    bone_radius: f32,
    avoid_iterations: c_int,
    out_local_rotations: *mut f32,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| unsafe {
        solve_chain_avoid_inner(
            parents,
            local_bind,
            bone_count,
            chain,
            chain_len,
            effector_bone,
            target,
            prior,
            hinge_limits,
            iterations,
            obstacles,
            obstacle_count,
            bone_radius,
            avoid_iterations,
            out_local_rotations,
        )
    }));
    result.unwrap_or(CRF_ERR_PANIC)
}

/// The actual obstacle-aware solve, factored out so `crf_solve_chain_avoid` is
/// just the panic guard.
///
/// # Safety
/// Same contract as [`crf_solve_chain_avoid`].
#[allow(clippy::too_many_arguments)]
unsafe fn solve_chain_avoid_inner(
    parents: *const u64,
    local_bind: *const f32,
    bone_count: usize,
    chain: *const u64,
    chain_len: usize,
    effector_bone: u64,
    target: *const f32,
    prior: *const CrfPrior,
    hinge_limits: *const f32,
    iterations: c_int,
    obstacles: *const CrfCapsule,
    obstacle_count: usize,
    bone_radius: f32,
    avoid_iterations: c_int,
    out_local_rotations: *mut f32,
) -> c_int {
    if out_local_rotations.is_null() {
        return CRF_ERR_NULL_POINTER;
    }
    // Obstacles may only be null when there are none.
    if obstacles.is_null() && obstacle_count != 0 {
        return CRF_ERR_NULL_POINTER;
    }

    // Run the same validate + rebuild + solve path as crf_solve_chain.
    let (skeleton, mut pose, chain_idx, limits) = match unsafe {
        solve_into(
            parents,
            local_bind,
            bone_count,
            chain,
            chain_len,
            effector_bone,
            target,
            prior,
            hinge_limits,
            iterations,
        )
    } {
        Ok(v) => v,
        Err(code) => return code,
    };

    // Push the chain out of the obstacles. With no obstacles this is a no-op and
    // the result matches crf_solve_chain exactly.
    if obstacle_count != 0 {
        let raw = unsafe { std::slice::from_raw_parts(obstacles, obstacle_count) };
        let obstacles_vec: Vec<cube_rig::collision::Capsule> = raw
            .iter()
            .map(|c| cube_rig::collision::Capsule {
                a: Vec3::new(c.a[0], c.a[1], c.a[2]),
                b: Vec3::new(c.b[0], c.b[1], c.b[2]),
                radius: c.radius,
            })
            .collect();
        let _remaining = cube_rig::collision::avoid_obstacles(
            &mut pose,
            &skeleton,
            &chain_idx,
            bone_radius,
            &obstacles_vec,
            limits.as_deref(),
            &cube_rig::collision::AvoidParams {
                iterations: avoid_iterations.max(1) as usize,
                ..cube_rig::collision::AvoidParams::default()
            },
        );
    }

    // Write every bone's final local rotation (xyzw) into the output buffer.
    unsafe { write_local_rotations(&pose, &skeleton, out_local_rotations) };

    CRF_OK
}

// ── Version ─────────────────────────────────────────────────────────────────

/// Library version as a static, NUL-terminated C string (`"<semver>\0"`). The
/// returned pointer is valid for the lifetime of the program; do not free it.
#[no_mangle]
pub extern "C" fn crf_version() -> *const c_char {
    // A compile-time NUL-terminated string with no embedded NULs.
    const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");
    VERSION.as_ptr() as *const c_char
}

// Keep `c_void` referenced so the import is not flagged unused if the API grows;
// some downstream headers expect an opaque `void*`-shaped handle.
#[doc(hidden)]
pub type _CrfOpaque = *mut c_void;

#[cfg(test)]
mod tests {
    use super::*;
    use cube_rig::clip::{Clip, Frame};
    use cube_rig::prior::MotionPriorBuilder;
    use std::ffi::CString;

    /// Build a planar `n`-joint chain plus a tip effector as the flat arrays the
    /// FFI expects. Returns `(parents, local_bind, bone_count, chain, effector)`.
    fn planar_chain_arrays(n: usize) -> (Vec<u64>, Vec<f32>, usize, Vec<u64>, u64) {
        // bone 0 root at origin; bones 1..n each +1 on X from parent; final tip.
        let bone_count = n + 1;
        let mut parents = vec![CRF_NO_PARENT; bone_count];
        let mut local_bind = vec![0.0f32; bone_count * 16];
        let mut chain = Vec::new();

        let write = |buf: &mut [f32], i: usize, m: Mat4| {
            buf[i * 16..i * 16 + 16].copy_from_slice(&m.to_cols_array());
        };

        // root
        parents[0] = CRF_NO_PARENT;
        write(&mut local_bind, 0, Mat4::IDENTITY);
        chain.push(0u64);
        for (i, parent) in parents.iter_mut().enumerate().take(n).skip(1) {
            *parent = (i - 1) as u64;
            write(&mut local_bind, i, Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)));
            chain.push(i as u64);
        }
        // effector / tip, child of last joint
        let eff = n;
        parents[eff] = (n - 1) as u64;
        write(&mut local_bind, eff, Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)));

        (parents, local_bind, bone_count, chain, eff as u64)
    }

    /// Tiny forward-kinematics: recompute the effector head position from the
    /// returned local rotations + the bind translations, so we can verify the
    /// pose actually reaches the target.
    fn fk_effector(
        parents: &[u64],
        local_bind: &[f32],
        out_rot: &[f32],
        effector: usize,
    ) -> Vec3 {
        fn global(parents: &[u64], local_bind: &[f32], out_rot: &[f32], i: usize) -> Mat4 {
            let translation = Mat4::from_cols_array(
                <&[f32; 16]>::try_from(&local_bind[i * 16..i * 16 + 16]).unwrap(),
            )
            .w_axis
            .truncate();
            let q = Quat::from_xyzw(
                out_rot[i * 4],
                out_rot[i * 4 + 1],
                out_rot[i * 4 + 2],
                out_rot[i * 4 + 3],
            );
            let local = Mat4::from_rotation_translation(q, translation);
            match parents[i] {
                p if p == CRF_NO_PARENT => local,
                p => global(parents, local_bind, out_rot, p as usize) * local,
            }
        }
        global(parents, local_bind, out_rot, effector).w_axis.truncate()
    }

    #[test]
    fn solve_chain_reaches_reachable_target() {
        let (parents, local_bind, bone_count, chain, effector) = planar_chain_arrays(3);
        let target = [1.5f32, 1.0, 0.0];
        let mut out = vec![0.0f32; bone_count * 4];

        let code = unsafe {
            crf_solve_chain(
                parents.as_ptr(),
                local_bind.as_ptr(),
                bone_count,
                chain.as_ptr(),
                chain.len(),
                effector,
                target.as_ptr(),
                std::ptr::null(),       // uniform prior
                std::ptr::null(),       // no limits
                64,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(code, CRF_OK, "solve must succeed");
        // All outputs finite and unit-ish quaternions.
        for chunk in out.chunks(4) {
            for &c in chunk {
                assert!(c.is_finite(), "rotation component must be finite");
            }
            let len = (chunk[0] * chunk[0]
                + chunk[1] * chunk[1]
                + chunk[2] * chunk[2]
                + chunk[3] * chunk[3])
                .sqrt();
            assert!((len - 1.0).abs() < 1e-3, "quaternion must be unit, got {len}");
        }
        // FK from the solved pose lands the effector on the target.
        let e = fk_effector(&parents, &local_bind, &out, effector as usize);
        let dist = (e - Vec3::new(target[0], target[1], target[2])).length();
        assert!(dist < 1e-2, "effector should reach target, dist = {dist}, e = {e:?}");
    }

    #[test]
    fn solve_chain_with_hinge_limits_is_finite() {
        let (parents, local_bind, bone_count, chain, effector) = planar_chain_arrays(3);
        let target = [1.0f32, 1.5, 0.0];
        let mut out = vec![0.0f32; bone_count * 4];

        // Hinge about Z, half-range 0.2 rad on each chain joint.
        let mut limits = vec![0.0f32; chain.len() * 4];
        for k in 0..chain.len() {
            limits[k * 4 + 2] = 1.0; // axis = +Z
            limits[k * 4 + 3] = 0.2; // half range
        }

        let code = unsafe {
            crf_solve_chain(
                parents.as_ptr(),
                local_bind.as_ptr(),
                bone_count,
                chain.as_ptr(),
                chain.len(),
                effector,
                target.as_ptr(),
                std::ptr::null(),
                limits.as_ptr(),
                32,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(code, CRF_OK);
        assert!(out.iter().all(|c| c.is_finite()));
    }

    #[test]
    fn solve_chain_rejects_null_and_bad_args() {
        let (parents, local_bind, bone_count, chain, effector) = planar_chain_arrays(2);
        let target = [1.0f32, 0.0, 0.0];
        let mut out = vec![0.0f32; bone_count * 4];

        // Null required pointer → NULL_POINTER.
        let code = unsafe {
            crf_solve_chain(
                std::ptr::null(),
                local_bind.as_ptr(),
                bone_count,
                chain.as_ptr(),
                chain.len(),
                effector,
                target.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                16,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(code, CRF_ERR_NULL_POINTER);

        // Zero bone_count → INVALID_ARG.
        let code = unsafe {
            crf_solve_chain(
                parents.as_ptr(),
                local_bind.as_ptr(),
                0,
                chain.as_ptr(),
                chain.len(),
                effector,
                target.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                16,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(code, CRF_ERR_INVALID_ARG);

        // Out-of-range effector → INVALID_ARG.
        let code = unsafe {
            crf_solve_chain(
                parents.as_ptr(),
                local_bind.as_ptr(),
                bone_count,
                chain.as_ptr(),
                chain.len(),
                999,
                target.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                16,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(code, CRF_ERR_INVALID_ARG);
    }

    /// FK head position of an arbitrary bone `i` from returned local rotations.
    fn fk_head(parents: &[u64], local_bind: &[f32], out_rot: &[f32], i: usize) -> Vec3 {
        fk_effector(parents, local_bind, out_rot, i)
    }

    #[test]
    fn solve_chain_avoid_no_obstacles_matches_solve_chain() {
        let (parents, local_bind, bone_count, chain, effector) = planar_chain_arrays(3);
        let target = [1.5f32, 1.0, 0.0];

        let mut out_plain = vec![0.0f32; bone_count * 4];
        let code = unsafe {
            crf_solve_chain(
                parents.as_ptr(),
                local_bind.as_ptr(),
                bone_count,
                chain.as_ptr(),
                chain.len(),
                effector,
                target.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                64,
                out_plain.as_mut_ptr(),
            )
        };
        assert_eq!(code, CRF_OK);

        // No obstacles: null ptr + count 0 must behave identically.
        let mut out_avoid = vec![0.0f32; bone_count * 4];
        let code = unsafe {
            crf_solve_chain_avoid(
                parents.as_ptr(),
                local_bind.as_ptr(),
                bone_count,
                chain.as_ptr(),
                chain.len(),
                effector,
                target.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                64,
                std::ptr::null(), // obstacles null
                0,                // obstacle_count
                0.28,
                12,
                out_avoid.as_mut_ptr(),
            )
        };
        assert_eq!(code, CRF_OK);

        for (a, b) in out_plain.iter().zip(out_avoid.iter()) {
            assert!(a.is_finite() && b.is_finite());
            assert!((a - b).abs() < 1e-6, "no-obstacle avoid must match plain: {a} vs {b}");
        }
    }

    #[test]
    fn solve_chain_avoid_pushes_off_obstacle() {
        // 3-joint planar chain reaching past an obstacle planted on the naive
        // solution's path (mirrors viz_demo's scenario B).
        let (parents, local_bind, bone_count, chain, effector) = planar_chain_arrays(3);
        let target = [2.6f32, 0.05, 0.0];

        // Naive solve (no avoidance) to find where the arm lands.
        let mut naive = vec![0.0f32; bone_count * 4];
        let code = unsafe {
            crf_solve_chain(
                parents.as_ptr(),
                local_bind.as_ptr(),
                bone_count,
                chain.as_ptr(),
                chain.len(),
                effector,
                target.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                64,
                naive.as_mut_ptr(),
            )
        };
        assert_eq!(code, CRF_OK);

        // Plant the obstacle on the midpoint of a mid-arm bone of the naive pose
        // (heads of bones 1 and 2), swept through Z so the bone capsule truly
        // penetrates it.
        let h1 = fk_head(&parents, &local_bind, &naive, 1);
        let h2 = fk_head(&parents, &local_bind, &naive, 2);
        let mid = (h1 + h2) * 0.5;
        let bone_radius = 0.28f32;
        let obstacle = CrfCapsule {
            a: [mid.x, mid.y, -0.7],
            b: [mid.x, mid.y, 0.7],
            radius: 0.4,
        };

        // Sanity: the naive pose actually penetrates the obstacle, otherwise the
        // test would be vacuous. Rebuild a cube_rig pose to query collision.
        let (sk, naive_pose) = rebuild_pose(&parents, &local_bind, &naive, bone_count);
        let obs_cr = cube_rig::collision::Capsule {
            a: Vec3::new(obstacle.a[0], obstacle.a[1], obstacle.a[2]),
            b: Vec3::new(obstacle.b[0], obstacle.b[1], obstacle.b[2]),
            radius: obstacle.radius,
        };
        let naive_pen = cube_rig::collision::bone_capsules(&naive_pose, &sk, bone_radius)
            .iter()
            .flatten()
            .filter(|c| cube_rig::collision::capsule_penetration(c, &obs_cr).is_some())
            .count();
        assert!(naive_pen > 0, "obstacle must actually penetrate the naive arm");

        // Now solve with avoidance.
        let obstacles = [obstacle];
        let mut out = vec![0.0f32; bone_count * 4];
        let code = unsafe {
            crf_solve_chain_avoid(
                parents.as_ptr(),
                local_bind.as_ptr(),
                bone_count,
                chain.as_ptr(),
                chain.len(),
                effector,
                target.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                64,
                obstacles.as_ptr(),
                obstacles.len(),
                bone_radius,
                64, // avoid_iterations
                out.as_mut_ptr(),
            )
        };
        assert_eq!(code, CRF_OK);

        // All outputs finite.
        assert!(out.iter().all(|c| c.is_finite()), "all rotations must be finite");

        // Rebuild the solved pose and assert ZERO bone capsules penetrate.
        let (sk, solved_pose) = rebuild_pose(&parents, &local_bind, &out, bone_count);
        let remaining = cube_rig::collision::bone_capsules(&solved_pose, &sk, bone_radius)
            .iter()
            .flatten()
            .filter(|c| cube_rig::collision::capsule_penetration(c, &obs_cr).is_some())
            .count();
        assert_eq!(remaining, 0, "no bone capsule may penetrate the obstacle after avoidance");
    }

    #[test]
    fn solve_chain_avoid_rejects_null_and_bad_args() {
        let (parents, local_bind, bone_count, chain, effector) = planar_chain_arrays(2);
        let target = [1.0f32, 0.0, 0.0];
        let mut out = vec![0.0f32; bone_count * 4];

        // Null out buffer → NULL_POINTER.
        let code = unsafe {
            crf_solve_chain_avoid(
                parents.as_ptr(),
                local_bind.as_ptr(),
                bone_count,
                chain.as_ptr(),
                chain.len(),
                effector,
                target.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                16,
                std::ptr::null(),
                0,
                0.28,
                12,
                std::ptr::null_mut(),
            )
        };
        assert_eq!(code, CRF_ERR_NULL_POINTER);

        // Null obstacles with a nonzero count → NULL_POINTER.
        let code = unsafe {
            crf_solve_chain_avoid(
                parents.as_ptr(),
                local_bind.as_ptr(),
                bone_count,
                chain.as_ptr(),
                chain.len(),
                effector,
                target.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                16,
                std::ptr::null(), // null obstacles
                3,                // but count > 0
                0.28,
                12,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(code, CRF_ERR_NULL_POINTER);

        // Null required input pointer (parents) → NULL_POINTER.
        let code = unsafe {
            crf_solve_chain_avoid(
                std::ptr::null(),
                local_bind.as_ptr(),
                bone_count,
                chain.as_ptr(),
                chain.len(),
                effector,
                target.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                16,
                std::ptr::null(),
                0,
                0.28,
                12,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(code, CRF_ERR_NULL_POINTER);

        // Out-of-range effector → INVALID_ARG.
        let code = unsafe {
            crf_solve_chain_avoid(
                parents.as_ptr(),
                local_bind.as_ptr(),
                bone_count,
                chain.as_ptr(),
                chain.len(),
                999,
                target.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                16,
                std::ptr::null(),
                0,
                0.28,
                12,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(code, CRF_ERR_INVALID_ARG);
    }

    /// Rebuild a `cube_rig` `Skeleton` + `Pose` from the flat FFI arrays and the
    /// returned per-bone local rotations, so tests can query collision exactly as
    /// the engine would on the data the FFI handed back.
    fn rebuild_pose(
        parents: &[u64],
        local_bind: &[f32],
        out_rot: &[f32],
        bone_count: usize,
    ) -> (Skeleton, Pose) {
        let mut sk = Skeleton::new();
        for (i, &p) in parents.iter().enumerate() {
            let parent = if p == CRF_NO_PARENT { None } else { Some(p as usize) };
            let m = Mat4::from_cols_array(
                <&[f32; 16]>::try_from(&local_bind[i * 16..i * 16 + 16]).unwrap(),
            );
            sk.add(Bone::new(format!("b{i}"), parent, m));
        }
        let mut pose = Pose::from_bind(&sk);
        for i in 0..bone_count {
            let translation = pose.local[i].w_axis.truncate();
            let q = Quat::from_xyzw(
                out_rot[i * 4],
                out_rot[i * 4 + 1],
                out_rot[i * 4 + 2],
                out_rot[i * 4 + 3],
            );
            pose.local[i] = Mat4::from_rotation_translation(q, translation);
        }
        (sk, pose)
    }

    #[test]
    fn prior_load_json_roundtrips_via_tempfile() {
        // Bake a small prior the same way `extract_prior` does, serialize it,
        // and load it back through the FFI.
        let mut frames = Vec::new();
        for i in 0..9 {
            let angle = (i as f32 - 4.0) * 0.3;
            frames.push(Frame {
                root: Mat4::IDENTITY,
                local_rotations: vec![Quat::IDENTITY, Quat::from_rotation_y(angle)],
            });
        }
        let clip = Clip { name: "spin".into(), frame_rate: 30.0, frames };
        let mut builder = MotionPriorBuilder::new(2);
        builder.add_clip(&clip);
        let prior = builder.build();

        let json = serde_json::to_vec_pretty(&prior).unwrap();

        // Round-trip through a temp file.
        let pid = std::process::id();
        let path = std::env::temp_dir().join(format!("crf_ffi_prior_{pid}.json"));
        std::fs::write(&path, &json).unwrap();
        let c_path = CString::new(path.to_str().unwrap()).unwrap();

        let handle = unsafe { crf_prior_load_json(c_path.as_ptr()) };
        assert!(!handle.is_null(), "valid prior file should load");
        // Use it in a solve to prove it dereferences cleanly.
        let (parents, local_bind, bone_count, chain, effector) = planar_chain_arrays(2);
        let target = [1.0f32, 0.5, 0.0];
        let mut out = vec![0.0f32; bone_count * 4];
        let code = unsafe {
            crf_solve_chain(
                parents.as_ptr(),
                local_bind.as_ptr(),
                bone_count,
                chain.as_ptr(),
                chain.len(),
                effector,
                target.as_ptr(),
                handle, // non-null prior
                std::ptr::null(),
                32,
                out.as_mut_ptr(),
            )
        };
        assert_eq!(code, CRF_OK);
        unsafe { crf_prior_free(handle) };
        let _ = std::fs::remove_file(&path);

        // Also round-trip through the bytes loader.
        let handle2 = unsafe { crf_prior_load_json_bytes(json.as_ptr(), json.len()) };
        assert!(!handle2.is_null());
        unsafe { crf_prior_free(handle2) };
    }

    #[test]
    fn prior_load_json_bad_path_returns_null() {
        let c_path = CString::new("/nonexistent/path/does/not/exist.json").unwrap();
        let handle = unsafe { crf_prior_load_json(c_path.as_ptr()) };
        assert!(handle.is_null(), "missing file must return null, not panic");

        // Null path is also handled gracefully.
        let handle = unsafe { crf_prior_load_json(std::ptr::null()) };
        assert!(handle.is_null());

        // Malformed JSON → null.
        let bad = b"{ not valid json";
        let handle = unsafe { crf_prior_load_json_bytes(bad.as_ptr(), bad.len()) };
        assert!(handle.is_null());
    }

    #[test]
    fn prior_free_null_is_noop() {
        unsafe { crf_prior_free(std::ptr::null_mut()) };
    }

    #[test]
    fn version_is_nonempty_c_string() {
        let ptr = crf_version();
        assert!(!ptr.is_null());
        let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
        assert!(!s.is_empty());
        assert_eq!(s, env!("CARGO_PKG_VERSION"));
    }
}
