//! Exporters for skinned meshes.
//!
//!   * [`glb`] — binary glTF 2.0 with a full `skin` (joints, inverse-bind
//!     matrices, `JOINTS_0`/`WEIGHTS_0`). The reliable skin round-trip format.
//!   * [`fbx`] — ASCII FBX 7.4 with skin `Deformer`/`SubDeformer` clusters and a
//!     bind `Pose`, for Unreal/Maya import.
//!   * [`obj`] — Wavefront OBJ, geometry only (no skin; OBJ can't carry one).
//!
//! All writers hand-roll their formats (matching the octree exporter's style)
//! and take a [`SkinnedMesh`] + [`Skeleton`].

pub mod fbx;
pub mod glb;
pub mod obj;

/// Errors shared by the exporters.
#[derive(Debug)]
pub enum ExportError {
    Io(String),
    Empty,
    TooManyBones(usize),
}

impl std::fmt::Display for ExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExportError::Io(s) => write!(f, "I/O error: {s}"),
            ExportError::Empty => write!(f, "nothing to export (empty mesh)"),
            ExportError::TooManyBones(n) => {
                write!(f, "{n} bones exceeds the glTF joint limit for u16 indices")
            }
        }
    }
}

impl std::error::Error for ExportError {}
