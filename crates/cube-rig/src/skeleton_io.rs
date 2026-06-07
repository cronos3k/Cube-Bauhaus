//! Loading and saving standalone skeleton files.
//!
//! A rig's skeleton can come from a different file than its mesh (e.g. a shared
//! character rig reused across many meshes). Two formats are supported:
//!
//!   * **glTF / GLB** — reuses the model importer's skin→skeleton logic.
//!   * **JSON** — a small, human-editable format (the `serde` representation of
//!     [`Skeleton`]): bone name, parent index and a 16-float column-major bind
//!     matrix per bone.

use std::path::Path;

use crate::import::ImportError;
use crate::skeleton::Skeleton;

/// Load a skeleton from any supported file, dispatched by extension.
pub fn load_skeleton(path: impl AsRef<Path>) -> Result<Skeleton, ImportError> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "json" => load_skeleton_json(path),
        "gltf" | "glb" => crate::import::gltf::import_gltf_skeleton(path)?
            .ok_or_else(|| ImportError::Parse("glTF file contains no skeleton".into())),
        other => Err(ImportError::UnsupportedFormat(other.into())),
    }
}

/// Load a skeleton from the JSON format. Validates the hierarchy on load.
pub fn load_skeleton_json(path: &Path) -> Result<Skeleton, ImportError> {
    let text = std::fs::read_to_string(path).map_err(|e| ImportError::Io(e.to_string()))?;
    let skeleton: Skeleton =
        serde_json::from_str(&text).map_err(|e| ImportError::Parse(e.to_string()))?;
    skeleton.validate().map_err(ImportError::Parse)?;
    Ok(skeleton)
}

/// Write a skeleton to the JSON format (pretty-printed).
pub fn save_skeleton_json(path: &Path, skeleton: &Skeleton) -> Result<(), ImportError> {
    let text = serde_json::to_string_pretty(skeleton).map_err(|e| ImportError::Parse(e.to_string()))?;
    std::fs::write(path, text).map_err(|e| ImportError::Io(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::Bone;
    use glam::{Mat4, Vec3};

    #[test]
    fn json_roundtrip_preserves_hierarchy() {
        let mut sk = Skeleton::new();
        let r = sk.add(Bone::root("root"));
        sk.add(Bone::new("child", Some(r), Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0))));

        let path = std::env::temp_dir().join("cube_rig_skel.json");
        save_skeleton_json(&path, &sk).unwrap();
        let loaded = load_skeleton(&path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.bones[1].name, "child");
        assert_eq!(loaded.bones[1].parent, Some(0));
        assert_eq!(loaded.head_position(1), Vec3::new(1.0, 2.0, 3.0));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_invalid_hierarchy_on_load() {
        // child references a parent that comes after it
        let json = r#"{"bones":[
            {"name":"a","parent":1,"local_bind":[1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]},
            {"name":"b","parent":null,"local_bind":[1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]}
        ]}"#;
        let path = std::env::temp_dir().join("cube_rig_bad_skel.json");
        std::fs::write(&path, json).unwrap();
        assert!(load_skeleton(&path).is_err());
        let _ = std::fs::remove_file(&path);
    }
}
