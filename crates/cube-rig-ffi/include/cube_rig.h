/*
 * cube_rig.h — C-ABI for the cube-rig IK solver (cube-rig-ffi crate).
 *
 * Hand-authored to match crates/cube-rig-ffi/src/lib.rs exactly. Keep the two
 * in lock-step when changing any signature.
 *
 * Conventions
 * -----------
 *  - All 4x4 matrices are COLUMN-MAJOR (glam / glTF / Unreal FMatrix layout via
 *    Mat4::from_cols_array): 16 floats, columns 0..3 contiguous.
 *  - Quaternions are (x, y, z, w).
 *  - Parent indices use UINT64_MAX (CRF_NO_PARENT) to mark a root bone.
 *  - The skeleton must be in TOPOLOGICAL ORDER: every parent index is strictly
 *    less than its child's index. (crf_solve_chain validates and rejects otherwise.)
 *  - Functions are panic-safe: a Rust panic is caught and reported as an error
 *    code / null pointer; it never unwinds across this boundary.
 */

#ifndef CUBE_RIG_FFI_H
#define CUBE_RIG_FFI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Sentinel parent index: this bone is a root. */
#define CRF_NO_PARENT ((uint64_t)UINT64_MAX)

/* Return codes for crf_solve_chain. */
#define CRF_OK                 0  /* success                                   */
#define CRF_ERR_NULL_POINTER   1  /* a required pointer argument was null      */
#define CRF_ERR_INVALID_ARG    2  /* a length/index argument was out of range  */
#define CRF_ERR_BAD_SKELETON   3  /* bad topology / parent indices             */
#define CRF_ERR_PANIC          4  /* a Rust panic was caught at the boundary   */

/* Opaque handle to an owned MotionPrior. Allocate via crf_prior_load_json*,
 * release via crf_prior_free. The layout is private. */
typedef struct CrfPrior CrfPrior;

/*
 * Load a MotionPrior from a JSON file produced by `extract_prior`.
 * Returns a handle, or NULL on any failure (null/invalid path, missing file,
 * malformed JSON). Free the result with crf_prior_free.
 */
CrfPrior* crf_prior_load_json(const char* path);

/*
 * Load a MotionPrior from an in-memory JSON buffer (same format).
 * Returns NULL on failure. Free with crf_prior_free.
 */
CrfPrior* crf_prior_load_json_bytes(const uint8_t* ptr, size_t len);

/*
 * Release a prior handle. Passing NULL is a no-op. Double-free is UB.
 */
void crf_prior_free(CrfPrior* prior);

/*
 * Solve an IK chain so `effector_bone`'s head reaches `target`, writing every
 * bone's solved local rotation (xyzw) into `out_local_rotations`.
 *
 * Parameters
 *  parents             len = bone_count; parent index, or CRF_NO_PARENT for a root.
 *  local_bind          len = bone_count*16; column-major Mat4 per bone (bind pose).
 *  bone_count          number of bones.
 *  chain               len = chain_len; root->tip bone indices to rotate.
 *  chain_len           number of chain joints (>= 1).
 *  effector_bone       bone whose head is driven to the target (often a descendant
 *                      of the last chain joint).
 *  target              float[3], model-space target position.
 *  prior               optional CrfPrior* (NULL => uniform compliance).
 *  hinge_limits        optional; NULL => no limits (the fully-supported v1 path).
 *                      When non-NULL: chain_len*4 floats, per chain joint
 *                      [axis_x, axis_y, axis_z, half_range]; applied as a hinge
 *                      about the normalized axis with min=-half_range, max=+half_range
 *                      relative to the bone's bind rotation. A zero/non-finite axis
 *                      disables the limit for that joint. Asymmetric/swing-twist
 *                      limits are a follow-up.
 *  iterations          DLS iterations (clamped to >= 1).
 *  out_local_rotations len = bone_count*4; receives the solved local rotation
 *                      (x,y,z,w) of EVERY bone (non-chain bones keep bind).
 *
 * Returns CRF_OK (0) on success, or a nonzero CRF_ERR_* code.
 */
int crf_solve_chain(
    const uint64_t* parents,
    const float*    local_bind,
    size_t          bone_count,
    const uint64_t* chain,
    size_t          chain_len,
    uint64_t        effector_bone,
    const float     target[3],
    const CrfPrior* prior,
    const float*    hinge_limits,
    int             iterations,
    float           out_local_rotations[]
);

/*
 * Library version as a static, NUL-terminated C string. The pointer is valid for
 * the lifetime of the program; do not free it.
 */
const char* crf_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CUBE_RIG_FFI_H */
