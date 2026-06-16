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

### 5.1 Reconsideration: damped least-squares with a weighted posture term

CCD is the simplest rotation-space option, but dissecting a shipping system
(NVIDIA GR00T-WBC's upper-body IK, §12.1) shows the soft bias has a *more
natural* home: a **Jacobian / damped-least-squares (DLS) solver with the bias as
a weighted secondary ("posture") task.** There, anisotropic compliance is a
per-joint weight matrix on the secondary objective — exactly our `JointBias` —
rather than something bolted onto a per-joint sweep. Their solver is literally a
`FrameTask` (effector goal) + a `WeightedPostureTask` (per-joint weights) under
joint-limit constraints, with Levenberg–Marquardt damping for unreachable
targets.

**Updated recommendation:** target a hand-rolled **DLS solver with a weighted
posture nullspace** as the primary constrained solver (no heavy dependency, same
shape as the production reference); keep CCD as the dependency-free fallback and
FABRIK/`two_bone` for the fast unconstrained path. The layering above is
unchanged — only the inner update rule differs.

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

## 9. Offline Extraction — a Source-Agnostic Pipeline

The field is baked offline (run locally; a large corpus is too big for a CI
sandbox). The central design decision is that **motion enters through one
abstraction regardless of where it came from** — captured mocap, hand-keyed
animation, or the *output* of a generative model. The extractor never knows or
cares about the source.

### 9.1 The `Clip` abstraction

```rust
pub struct Frame { pub root: Mat4, pub local_rotations: Vec<Quat> }   // per bone
pub struct Clip  { pub name: String, pub frame_rate: f32, pub frames: Vec<Frame> }
```

Every source normalizes into `Clip`. From this single stream we derive
*everything*: per-joint swing-twist → bias field (§6); empirical per-joint min/max
→ data-tightened limits validating the physical ones (§4); forward kinematics →
effector trajectories → goal descriptors (§7); inter-limb closest distances →
which bones actually near-collide → selective capsule placement and radii (§8).
None of it requires anything but the clip.

### 9.2 The bake

1. **Ingest** clips from any source into the `Clip` form (see §9.3). For external
   rigs, **retarget** to a canonical bone set via a name-alias table
   (`mixamorig:LeftArm` ↔ `L_Shoulder`), expressing rotations **bone-local,
   relative to bind** so rest-pose differences cancel. (Assume **mixed rigs**;
   for uniform corpora this is an identity pass.)
2. **Decompose** each bone's local rotation to swing-twist; compute each frame's
   `GoalDescriptor` by FK on the effector.
3. **Accumulate** per-(goal-cell, joint) swing-twist usage statistics.
4. **Bake** to a compact `MotionPrior` table — per cell, per joint a `JointBias`
   (v1) or weight grid (v2) — serialized alongside the app.

Joints with insufficient samples fall back to the global cell, and ultimately to
pure geometric IK (graceful degradation).

### 9.3 Motion sources — and the generative model as an *oracle*

Because the pipeline is source-agnostic, a trained generative motion model
(e.g. a MotionBricks/SONIC-style backbone, §12) becomes **just another clip
source — used through its output, never its weights.** We run the model, take the
per-frame joint configuration `q(t)` it emits, normalize it into `Clip`, and
distill exactly as we would from mocap. The runtime IK keeps **zero** neural
dependency; the model only ever appears offline, as data.

This is strictly better than a fixed corpus in one important way — **active
sampling.** Raw mocap gives whatever motions happen to exist; a generator can be
*queried*. We enumerate the goal-conditioned cells the IK will actually face
(reach-high-left, pull-to-chest, vault, low-crawl…), drive the model's high-level
commands to **each**, and synthesize as many clips per cell as we want. That turns
§7's conditioning from "hope the corpus covers this cell" into "**synthesize
exactly the data each cell needs**, uniformly, on demand." Passive corpus →
active oracle.

Practical notes: model output is typically already retargeted to a known skeleton
and label-free, so ingestion is an identity pass; standing up a model's runtime
has a setup cost (weigh it against the active-sampling payoff); and a model's
*outputs* may be license-governed (e.g. NVIDIA Open Model License) — a one-time
check before a distilled prior ships in a product, not a runtime concern.

---

## 10. Data Types Summary

| Type | Module | Role |
|---|---|---|
| `Clip` / `Frame` | `clip.rs` | Source-agnostic motion (mocap / keyed / model output) |
| `JointLimit` | `limits.rs` | Hard limits: `Hinge{axis,min,max}` **or** `SwingTwist` cone |
| `JointBias` | `prior.rs` | Soft per-joint anisotropic steering bias |
| `GoalDescriptor` | `prior.rs` | Body-relative conditioning variable |
| `MotionPriorBuilder` | `prior.rs` | Accumulates clip statistics → bakes a `MotionPrior` |
| `MotionPrior` | `prior.rs` | Goal-cell → joint → bias lookup table |
| `Capsule` | `collision.rs` | Per-bone collision volume |
| `ik::solve` | `ik.rs` | Constrained solver honoring all layers (see §5) |

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

### 12.1 Production evidence: the decoupled RL + IK split (GR00T-WBC)

The complementary-layer framing is not hypothetical — it is how a shipping
humanoid platform is built. NVIDIA's **GR00T Whole-Body Control** stack houses
three subsystems (`gear_sonic/`, `motionbricks/`, and the **decoupled WBC** used
in GR00T N1.5/N1.6). The decoupled controller is described verbatim as
**"RL for lower body, and IK for upper body"**: a learned policy handles
locomotion and balance (the bulk, dynamic, hard-to-author part), while
**analytical inverse kinematics handles precise upper-body tasks** (reaching,
manipulation, contact).

This is the same division of labour proposed here, with the bulk-motion source
swapped for an RL policy instead of a generative model:

- **Bulk / balance layer** — learned (RL policy *or* generative backbone).
- **Precision + guarantee layer** — analytical IK (this system) for exact
  effector placement, hard joint limits, and self-collision.

Our contribution sits squarely in that second slot, and adds what the GR00T-WBC
README does not surface as explicit features: **data-driven goal-conditioned
steering** of the IK (§6–7) and **capsule self-collision guarantees** (§8). The
broader stack is trained on the **Bones-SEED** dataset (142K+ human motions,
~288 hours, retargeted to the Unitree G1) — the same scale of corpus our offline
extractor (§9) is designed to consume.

> **Reference.** NVIDIA GEAR. *GR00T Whole-Body Control* (GEAR-SONIC,
> MotionBricks, decoupled WBC). https://github.com/NVlabs/GR00T-WholeBodyControl
> — decoupled controller: "RL for lower body, and IK for upper body."

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
