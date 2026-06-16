# Goal-Conditioned Motion Priors for Constrained IK

**A data-driven steering, limiting, and collision-avoidance layer for skeletal inverse kinematics**

| | |
|---|---|
| **Project** | Cube-Bauhaus / `cube-rig` |
| **Status** | Design proposal (pre-implementation) |
| **Date** | 2026-06-09 |
| **Scope** | Skeletal IK solver, joint constraints, learned motion bias, self-collision |

---

## Abstract

This document specifies a system that steers a skeletal inverse-kinematics (IK)
solver toward **statistically plausible, real-world joint motion** while
guaranteeing physical validity. It combines four cooperating layers: (1) **hard
per-axis joint limits**, (2) a learned **soft gradient steering field** that
biases each joint toward its habitual rotation directions, (3) **goal-conditioned
context filtering** so the steering applied depends on *what the character is
trying to do*, and (4) **per-bone capsule collision** resolution evaluated as a
multi-pass stage before a solver result is committed.

The learned bias is harvested offline from a large motion corpus (~9000 clips).
Crucially, the field is conditioned on a **body-relative goal descriptor** that
is computed identically at training time and at solve time, so the runtime
solver can always retrieve the gradients relevant to its current goal.

---

## 1. Motivation

A geometric IK solver (FABRIK, CCD, analytic two-bone) finds *a* solution that
places the end effector on its goal. But the solution space is enormous and most
of it looks wrong: elbows that hyperextend, shoulders that roll into unnatural
twists, limbs that pass through the torso. Hard joint limits remove the
*impossible* poses but do nothing about the difference between a *valid* pose and
a *natural* one.

Real motion-capture data encodes that difference. If 4000 of 9000 arm clips spend
most of their rotation about one shoulder axis and almost none about another, that
anisotropy is information: it tells us which solutions humans actually use. This
system turns that statistic into a **steering force** on the solver.

Three observations shape the design:

1. **Limits are hard; preference is soft.** Limits clamp; preference nudges
   *within* the limits. They are separate layers.
2. **Natural motion is self-avoiding.** Biasing toward habitual real poses
   *implicitly* steers away from self-intersecting configurations — but a hard
   collision backstop is still required for correctness.
3. **Preference is context-dependent.** The same shoulder behaves completely
   differently reaching up to a shelf versus pulling a lever to the chest. An
   unconditional average smears these into mush. The bias must be **conditioned
   on the goal.**

---

## 2. Current State of `cube-rig`

The crate today provides:

- `skeleton::Bone { name, parent, local_bind }` — no per-axis limits, no
  collision volumes.
- `ik::fabrik(...)` — positional FABRIK.
- `ik::two_bone(...)` — analytic law-of-cosines two-bone IK.
- `anim`, `mesh`, `weights`, `import` (OBJ/glTF/FBX), `export`, `viz`, `state`.

**There are no joint limits, capsules, or collision handling in the codebase
today.** All four layers described here are greenfield.

A key architectural fact: **FABRIK and `two_bone` solve in terms of joint
*positions*, not per-joint rotations.** "Bias this joint about axis X" and "clamp
axis Y to ±30°" have no natural place in a positional solver. The system
therefore introduces a rotation-space solver as its home.

---

## 3. Architecture Overview

Four layers, stacked from hardest to softest, wrapped in a multi-pass
solve→validate→commit pipeline:

```
                 ┌─────────────────────────────────────────────┐
                 │            IK request (goal pose)            │
                 └───────────────────────┬─────────────────────┘
                                         │
              ┌──────────────────────────▼──────────────────────────┐
              │  Layer 3b: GOAL-CONDITIONED CONTEXT SELECTION        │
              │  body-relative goal descriptor → matching gradients  │
              └──────────────────────────┬──────────────────────────┘
                                         │  (per-joint bias field for this goal)
              ┌──────────────────────────▼──────────────────────────┐
              │  CCD SOLVER (rotation space), per iteration:         │
              │   • Layer 1: clamp to hard per-axis joint limits     │
              │   • Layer 2: apply soft gradient steering bias       │
              └──────────────────────────┬──────────────────────────┘
                                         │  candidate pose
              ┌──────────────────────────▼──────────────────────────┐
              │  Layer 4: CAPSULE COLLISION                          │
              │   broad-phase → capsule-capsule test → push-out /    │
              │   reject → re-solve if needed                        │
              └──────────────────────────┬──────────────────────────┘
                                         │  validated pose
                                         ▼
                                     COMMIT
```

The soft layer reduces how often the hard collision layer must fire; the
collision layer guarantees correctness when it does.

---

## 4. Layer 1 — Hard Joint Limits

### 4.1 Representation

Each joint's local rotation relative to bind is decomposed into **swing-twist**:

- **Twist**: rotation about the bone's primary axis.
- **Swing**: the remaining rotation, a 2-DOF tilt of the bone axis.

Limits are stored per joint as bounds on this decomposition:

```rust
pub struct JointLimits {
    /// Twist bound about the primary axis, radians: (min, max).
    pub twist: (f32, f32),
    /// Swing cone half-angles about the two perpendicular axes, radians.
    /// An elliptical cone: (x_half_angle, y_half_angle).
    pub swing: (f32, f32),
}
```

This cleanly expresses both ball-and-socket joints (shoulder/hip: wide swing,
limited twist) and hinges (elbow/knee: near-zero swing on one axis, a single
flexion range on the other).

### 4.2 Clamping

A `clamp_to_limits(local_rot, &JointLimits) -> Quat` function projects any
rotation onto the admissible set. This is unit-testable in isolation, with no
solver or corpus dependency.

---

## 5. The Solver — CCD in Rotation Space

**Decision: introduce a Cyclic-Coordinate-Descent (CCD) solver as the home for
constrained IK.** CCD iterates joint-by-joint *in rotation space*, which is
exactly where per-axis limits, the gradient bias, and a per-iteration collision
check all slot in naturally.

FABRIK and `two_bone` remain as the **fast path** for the unconstrained case.

Per CCD iteration, for each joint from tip to root:

1. Compute the rotation that best aligns the effector with the goal.
2. **Layer 2:** modulate that rotation by the joint's gradient bias.
3. **Layer 1:** clamp the result to the joint's hard limits.
4. (Optionally) cheap-check capsule collision for early rejection.

---

## 6. Layer 2 — Soft Gradient Steering Field

### 6.1 Concept

For each joint, the corpus tells us *which rotation directions are habitual*.
We encode this as **anisotropic, data-driven joint compliance**:

- Directions the joint uses often → **low stiffness / high compliance** → the
  solver freely spends rotation there.
- Directions the joint rarely uses → **high stiffness** → the solver avoids them.

This is *not* a generative density that we sample from. It is a **bias / weighting
field** that steers an otherwise geometric solve toward natural motion within the
hard limits.

### 6.2 Minimal representation (v1)

```rust
pub struct JointBias {
    /// Per-axis usage weight (anisotropic compliance), in swing-x / swing-y /
    /// twist. Higher = more habitual = more compliant.
    pub axis_weight: Vec3,
    /// Preferred resting rotation the joint is gently pulled toward when the
    /// goal under-constrains it.
    pub preferred: Quat,
}
```

In the CCD step, `axis_weight` scales how much of the corrective rotation is
applied per axis, and `preferred` supplies a weak restoring nudge along the
null space of the goal constraint.

### 6.3 Richer representation (v2)

Replace the single weight vector with a **low-resolution weight grid over the
swing-twist space**, giving a true gradient `∇` that can vary across the joint's
range (e.g. an elbow that prefers different twists at different flexion angles).
The v1 type is a degenerate (1-cell) case of v2, so v2 is a non-breaking
extension.

---

## 7. Layer 3b — Goal-Conditioned Context Filtering

### 7.1 The conditioning variable

The bias is conditioned on a **goal descriptor** so the steering applied depends
on the task. The governing constraint:

> **The descriptor must be computable identically at training time and at solve
> time.** At solve time the IK system reliably knows only *goal geometry* — the
> effector, target position/orientation, and approach direction. It does **not**
> know semantic intent ("the player means to pull") unless the game explicitly
> supplies it.

Therefore the descriptor is a **body-relative goal descriptor**:

```rust
pub struct GoalDescriptor {
    /// Effector target position in a body-anchored frame (root or chest).
    pub target_local: Vec3,
    /// Approach / pull direction in the same frame (captures "pulling toward
    /// the chest" purely geometrically).
    pub approach_local: Vec3,
    /// Optional discrete action tag (reach / pull / throw / ...), used ONLY
    /// when the game supplies it. Never depended upon.
    pub action_tag: Option<u16>,
}
```

Advantages:

- **No train/run mismatch** — every clip's end-effector trajectory yields the
  descriptor directly, and the solver computes the same thing live.
- **Fully unsupervised** — needs no labels, appropriate for a flat 9000-clip dump.
- **"Pulling" is just an approach vector** — captured with no semantics.

The optional `action_tag` channel filters further *when available*, but the
system never requires it.

### 7.2 Indexing and lookup

The conditioned field is **an indexing layer on top of the same per-joint
machinery** — it does not change layers 1, 2, or 4.

- Quantize the body-relative goal into coarse **spherical shells × direction
  sectors**, store a per-cell, per-joint `JointBias`.
- At solve time, map the live `GoalDescriptor` to its cell, blend the nearest
  cells for smoothness — **O(1)** lookup.
- v1 ships with a **single global cell** (unconditioned) so the steering can be
  validated end-to-end before the field is split by goal. Splitting the cell is
  non-breaking.

---

## 8. Layer 4 — Capsule Collision

### 8.1 Volumes

Each bone carries a **capsule** (a line segment plus a radius):

```rust
pub struct Capsule {
    pub a: Vec3,      // segment start (joint)
    pub b: Vec3,      // segment end (child joint)
    pub radius: f32,
}
```

### 8.2 Multi-pass resolution

Collision is evaluated as a **validation/resolution stage before committing** the
solver result:

1. **Broad-phase**: cheap bounding test to cull non-adjacent bone pairs (adjacent
   bones legitimately touch and are excluded).
2. **Narrow-phase**: capsule–capsule closest-distance test; penetration when
   distance < r₁ + r₂.
3. **Resolve**: either
   - **soft** — add a penetration penalty term to the CCD objective and
     re-iterate, or
   - **hard** — push the offending joint out along the contact normal and
     re-solve the affected chain.
4. **Commit** only once the pose is collision-free.

The soft gradient layer (6) makes collisions rare; this layer makes correctness
guaranteed.

---

## 9. Offline Corpus Extraction

A standalone binary bakes the field from the motion corpus (run locally; ~9000
clips is too large for a CI sandbox):

1. **Walk** a directory of clips → `import` (OBJ/glTF/FBX; **add BVH** if the
   corpus is mocap-standard, since `import/` does not cover it today).
2. **Retarget** each source skeleton to a canonical bone set via a name-alias
   table (e.g. `mixamorig:LeftArm` ↔ `L_Shoulder`), expressing every rotation
   **bone-local, relative to bind** so rest-pose differences cancel. (Assume
   **mixed rigs**; for uniform corpora this is an identity pass.)
3. **Sample** each `AnimationClip` at a fixed rate; convert each bone's local
   quaternion to swing-twist; compute each frame's `GoalDescriptor`.
4. **Accumulate** per-(goal-cell, joint) swing-twist usage histograms.
5. **Bake** to a compact `MotionPrior` table — per cell, per joint a `JointBias`
   (v1) or weight grid (v2) — serialized alongside the app.

Joints with insufficient samples fall back to the global cell, and ultimately to
pure geometric IK (graceful degradation).

---

## 10. Data Types Summary

| Type | Module | Role |
|---|---|---|
| `JointLimits` | `limits.rs` | Hard per-axis swing-twist bounds |
| `JointBias` | `prior.rs` | Soft per-joint anisotropic steering bias |
| `GoalDescriptor` | `prior.rs` | Body-relative conditioning variable |
| `MotionPrior` | `prior.rs` | Goal-cell → joint → bias lookup table |
| `Capsule` | `collision.rs` | Per-bone collision volume |
| `ik::ccd` | `ik.rs` | Rotation-space solver honoring all layers |

---

## 11. Build Order

Each step compiles and is unit-testable **without** the corpus; only step 5's
*baking* needs the real clips.

1. **Joint limits** — `JointLimits` + `clamp_to_limits`. Testable alone.
2. **CCD solver** — `ik::ccd` honoring limits; A/B against FABRIK.
3. **Gradient field** — `JointBias` wired as CCD step weighting, with a single
   **global** goal cell.
4. **Goal conditioning** — `GoalDescriptor` + cell indexing; split the global
   cell into goal-conditioned cells (non-breaking).
5. **Corpus extractor** — offline binary that bakes the real `MotionPrior`
   (run locally).
6. **Capsule collision** — `Capsule` + capsule-capsule distance + the
   resolve-before-commit pass.

**Steps 1–2 are invariant to every open design decision** and are the
recommended starting point: layers 3, 3b, and 4 all bolt onto CCD and are inert
without it.

---

## 12. Relationship to Generative Backbones (MotionBricks)

Large-scale **neural generative** motion systems target the same end goal as this
design — natural, controllable, real-time, constraint-respecting motion — but by
the opposite mechanism. The clearest contemporary example is **MotionBricks**
[Wang et al. 2026]: a single latent-token model trained on ~350k clips
(700 hours, 27 joints) running at ~15,900 FPS / 2 ms. High-level "smart
primitives" emit **target keyframes** through a unified interface; a coarse-to-fine
backbone (root module → multi-head discrete pose-token module → decoder) generates
the in-between motion. Plausibility is **learned implicitly**; constraints are
**soft keyframe conditioning**, and the authors deliberately avoid strict
enforcement because "strict enforcement would produce unnatural, over-constrained
motion."

The two approaches are **complementary, operating at different layers**, not
competitors:

| | Generative backbone (MotionBricks) | This design |
|---|---|---|
| Role | *What* skill, *where* to go (bulk full-body motion) | *Exact* effector placement + hard validity |
| Plausibility | Learned from the corpus, implicit | Explicit hard limits + capsule collision + soft learned bias |
| Contact | Soft / approximate by design | Precise, solved to the goal |
| Self-penetration | Not guaranteed (model "should" avoid) | **Guaranteed** by capsule resolve-before-commit |
| Joint limits | None explicit | Explicit per-axis swing-twist clamps |
| Footprint | Large model, GPU | Microseconds, CPU, KB-scale baked table |

**Architectural implication.** A generative backbone keeps contact soft and offers
no hard guarantee against self-penetration or joint-limit violation — precisely
the gap this system fills. The natural production stack is therefore:

```
  smart primitives / generative backbone  →  bulk full-body motion (approximate)
                          │
                          ▼
  THIS SYSTEM (constrained IK pass)        →  precise contact + hard guarantees
   • snap effector exactly onto the goal (hand on handle)
   • clamp to hard joint limits
   • resolve capsule self-collision before commit
                          │
                          ▼
                       final pose
```

In this arrangement the goal-conditioned gradient field (§6–7) and the generative
backbone draw on the *same* statistical insight — that learned motion priors beat
hand-authored graphs — but apply it at different cost/guarantee points: the
backbone generates richly and stochastically; this layer corrects deterministically
and cheaply, with guarantees the learned model only approximates. The two compose:
neither subsumes the other.

This also clarifies scope. Building this system does **not** require adopting a
generative backbone; the analytical core stands alone for any rig-driving,
retargeting, or contact task. But it is designed to slot *underneath* one when a
generative motion source is present.

> **Reference.** T. Wang, O. Dionne, M. De Ruyter, D. Minor, D. Rempe,
> K. Zhao, M. Petrovich, Y. Yuan, C. Li, Z. Luo, B. Robison, X. Blackwell,
> B. Antoniazzi, X. B. Peng, Y. Zhu, S. Yuen. *MotionBricks: Scalable Real-Time
> Motions with Modular Latent Generative Model and Smart Primitives.* ACM Trans.
> Graph. 45(4), July 2026. arXiv:2604.24833. DOI 10.1145/3811334.

---

## 13. Open Questions

1. **Corpus labels** — is the corpus a flat unlabeled dump, or grouped/named by
   action (`reach/`, `pull/`, …)? This decides whether the optional `action_tag`
   channel is worth wiring early.
2. **Corpus format** — if it is predominantly **BVH**, a BVH importer is a
   prerequisite for extraction.
3. **Collision resolution mode** — soft penalty term vs. hard push-out + re-solve
   (or both, soft-then-hard).
4. **Bias representation** — ship v1 (single weight vector) first, or go straight
   to v2 (swing-twist weight grid)?
