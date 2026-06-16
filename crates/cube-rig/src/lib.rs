//! Skeletal rigging & skinning for Cube Bauhaus.
//!
//! Cube Bauhaus' level geometry is an octree whose render mesh is regenerated on
//! every edit, so its vertices have no persistent identity to bind weights to.
//! This crate provides the parallel, *stable* world needed for rigging:
//!
//!   * [`SkinnedMesh`] — imported geometry whose vertices keep fixed indices and
//!     carry per-vertex bone influences.
//!   * [`Skeleton`] — a flat, topologically ordered bone hierarchy that maps
//!     directly onto glTF/FBX joint arrays.
//!   * [`select`] — the "prep" vertex-selection toolkit (marquee, lasso, brush,
//!     grow/shrink, linked, by-bone) that runs before weight assignment.
//!   * [`weights`] — rigid bind, brush paint, smooth, mirror, prune.
//!   * [`import`] / [`export`] — OBJ, glTF/GLB and (best-effort) FBX in, skinned
//!     GLB/FBX and plain OBJ out, plus standalone skeleton files (glTF + JSON).
//!   * [`RigState`] — the editor-side session tying a mesh, a skeleton, the
//!     active bone and the current selection together.
//!
//! The whole crate is engine-agnostic and free of any GPU dependency, so it is
//! fully unit-testable headless; the editor layer drives it with a camera and an
//! egui tool palette.

pub mod anim;
pub mod clip;
pub mod collision;
pub mod controller;
pub mod export;
pub mod extract;
pub mod ik;
pub mod import;
pub mod limits;
pub mod mesh;
pub mod prior;
pub mod select;
pub mod skeleton;
pub mod skeleton_io;
pub mod state;
pub mod viz;
pub mod weights;

pub use clip::{Clip, Frame};
pub use collision::{
    avoid_obstacles, bone_capsules, capsule_distance, capsule_penetration, closest_segment_points,
    is_self_colliding, self_collisions, AvoidParams, Capsule, Contact,
};
pub use controller::{RenderVertex, RigController};
pub use limits::{g1_29dof_limits, JointLimit};
pub use mesh::{Influence, RigVertex, SkinnedMesh, MAX_INFLUENCES};
pub use prior::{GoalDescriptor, JointBias, MotionPrior, MotionPriorBuilder};
pub use select::{SelectMode, VertexSelection};
pub use skeleton::{Bone, Skeleton};
pub use state::{RigState, Tool};
pub use viz::{weight_colors, WeightView};
