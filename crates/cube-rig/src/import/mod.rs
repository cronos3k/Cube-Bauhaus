//! Mesh & skeleton importers.
//!
//! OBJ carries geometry only; glTF/GLB can carry geometry, a skeleton and an
//! existing skin; FBX import is best-effort (see [`fbx`]). Each loader returns an
//! [`Imported`] — a [`SkinnedMesh`] plus an optional [`Skeleton`]. When the file
//! has no skeleton the caller can load one from a separate file via
//! [`crate::skeleton_io`].

use std::path::Path;

use crate::mesh::SkinnedMesh;
use crate::skeleton::Skeleton;

pub mod gltf;
pub mod obj;

#[cfg(feature = "fbx-import")]
pub mod fbx;

/// The result of importing a model file.
pub struct Imported {
    pub mesh: SkinnedMesh,
    /// Present only if the file contained a skeleton.
    pub skeleton: Option<Skeleton>,
    /// Where it came from, for UI/logging.
    pub source: String,
}

#[derive(Debug)]
pub enum ImportError {
    Io(String),
    UnsupportedFormat(String),
    Parse(String),
    NoGeometry,
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImportError::Io(s) => write!(f, "I/O error: {s}"),
            ImportError::UnsupportedFormat(s) => write!(f, "unsupported format: {s}"),
            ImportError::Parse(s) => write!(f, "parse error: {s}"),
            ImportError::NoGeometry => write!(f, "file contained no geometry"),
        }
    }
}

impl std::error::Error for ImportError {}

/// Import any supported model by file extension.
pub fn import_model(path: impl AsRef<Path>) -> Result<Imported, ImportError> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "obj" => obj::import_obj(path),
        "gltf" | "glb" => gltf::import_gltf(path),
        #[cfg(feature = "fbx-import")]
        "fbx" => fbx::import_fbx(path),
        #[cfg(not(feature = "fbx-import"))]
        "fbx" => Err(ImportError::UnsupportedFormat(
            "FBX import not built (enable the `fbx-import` feature)".into(),
        )),
        other => Err(ImportError::UnsupportedFormat(other.into())),
    }
}

/// Append `src` onto `dst`, offsetting indices and influence slots. Used by
/// loaders to merge multiple primitives/objects into one mesh.
pub(crate) fn append_mesh(dst: &mut SkinnedMesh, src: SkinnedMesh) {
    let base = dst.vertices.len() as u32;
    dst.vertices.extend(src.vertices);
    dst.influences.extend(src.influences);
    dst.indices.extend(src.indices.into_iter().map(|i| i + base));
    dst.invalidate_adjacency();
}
