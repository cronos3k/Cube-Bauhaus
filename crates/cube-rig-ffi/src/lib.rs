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
    // Null checks for all required buffers.
    if parents.is_null()
        || local_bind.is_null()
        || chain.is_null()
        || target.is_null()
        || out_local_rotations.is_null()
    {
        return CRF_ERR_NULL_POINTER;
    }
    if bone_count == 0 || chain_len == 0 {
        return CRF_ERR_INVALID_ARG;
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
                return CRF_ERR_BAD_SKELETON;
            }
            Some(pu)
        };
        let m = Mat4::from_cols_array(
            <&[f32; 16]>::try_from(&local_bind[i * 16..i * 16 + 16]).unwrap(),
        );
        skeleton.add(Bone::new(format!("b{i}"), parent, m));
    }
    if skeleton.validate().is_err() {
        return CRF_ERR_BAD_SKELETON;
    }

    // Translate the chain indices, bounds-checking each.
    let mut chain_idx: Vec<usize> = Vec::with_capacity(chain_len);
    for &c in chain_raw {
        let ci = c as usize;
        if c == CRF_NO_PARENT || ci >= bone_count {
            return CRF_ERR_INVALID_ARG;
        }
        chain_idx.push(ci);
    }
    let effector = effector_bone as usize;
    if effector_bone == CRF_NO_PARENT || effector >= bone_count {
        return CRF_ERR_INVALID_ARG;
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

    // Write every bone's solved local rotation (xyzw) into the output buffer.
    let out = unsafe { std::slice::from_raw_parts_mut(out_local_rotations, bone_count * 4) };
    for (i, m) in pose.local.iter().enumerate() {
        let q: Quat = m.to_scale_rotation_translation().1;
        out[i * 4] = q.x;
        out[i * 4 + 1] = q.y;
        out[i * 4 + 2] = q.z;
        out[i * 4 + 3] = q.w;
    }

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
