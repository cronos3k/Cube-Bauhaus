//! Skeleton data model: a flat array of bones with parent indices.
//!
//! A flat array (rather than a pointer tree) is chosen deliberately — it maps
//! one-to-one onto the glTF `skin.joints` array and FBX limb-node list, so
//! export is a direct copy with no re-indexing. Bones must be stored in
//! topological order (every parent precedes its children); [`Skeleton::validate`]
//! enforces this and rejects cycles.

use glam::Mat4;
use serde::{Deserialize, Serialize};

/// A single joint in the skeleton.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bone {
    /// Human-readable name (used for export, mirroring by `_L`/`_R` suffix, UI).
    pub name: String,
    /// Index into [`Skeleton::bones`] of the parent, or `None` for a root.
    pub parent: Option<usize>,
    /// Bind-pose transform **relative to the parent** (the rest pose). The
    /// global bind transform is the product of this and all ancestors.
    #[serde(with = "mat4_serde")]
    pub local_bind: Mat4,
}

impl Bone {
    pub fn new(name: impl Into<String>, parent: Option<usize>, local_bind: Mat4) -> Self {
        Self { name: name.into(), parent, local_bind }
    }

    /// A root bone at the origin with identity orientation.
    pub fn root(name: impl Into<String>) -> Self {
        Self::new(name, None, Mat4::IDENTITY)
    }
}

/// A hierarchy of bones in topological order.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Skeleton {
    pub bones: Vec<Bone>,
}

impl Skeleton {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.bones.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bones.is_empty()
    }

    /// Append a bone, returning its index. The parent (if any) must already
    /// exist and have a lower index to preserve topological order.
    pub fn add(&mut self, bone: Bone) -> usize {
        if let Some(p) = bone.parent {
            debug_assert!(p < self.bones.len(), "parent must precede child");
        }
        self.bones.push(bone);
        self.bones.len() - 1
    }

    /// Find a bone index by exact name.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.bones.iter().position(|b| b.name == name)
    }

    /// Global (model-space) bind transform of bone `i`, composing up the chain.
    pub fn global_bind(&self, i: usize) -> Mat4 {
        let bone = &self.bones[i];
        match bone.parent {
            Some(p) => self.global_bind(p) * bone.local_bind,
            None => bone.local_bind,
        }
    }

    /// Inverse-bind matrix of bone `i` — transforms a model-space vertex into
    /// the bone's local space. This is what glTF stores in `inverseBindMatrices`
    /// and what FBX stores per skin cluster (`TransformLink`⁻¹).
    pub fn inverse_bind(&self, i: usize) -> Mat4 {
        self.global_bind(i).inverse()
    }

    /// All global bind matrices, indexed by bone.
    pub fn global_binds(&self) -> Vec<Mat4> {
        (0..self.bones.len()).map(|i| self.global_bind(i)).collect()
    }

    /// All inverse-bind matrices, indexed by bone.
    pub fn inverse_binds(&self) -> Vec<Mat4> {
        (0..self.bones.len()).map(|i| self.inverse_bind(i)).collect()
    }

    /// World-space position of a bone's origin (head), from its global bind.
    pub fn head_position(&self, i: usize) -> glam::Vec3 {
        self.global_bind(i).w_axis.truncate()
    }

    /// Direct children of bone `i`.
    pub fn children(&self, i: usize) -> Vec<usize> {
        self.bones
            .iter()
            .enumerate()
            .filter(|(_, b)| b.parent == Some(i))
            .map(|(j, _)| j)
            .collect()
    }

    /// The root bones (no parent).
    pub fn roots(&self) -> Vec<usize> {
        self.bones
            .iter()
            .enumerate()
            .filter(|(_, b)| b.parent.is_none())
            .map(|(i, _)| i)
            .collect()
    }

    /// Check the hierarchy is well-formed: parents in range, no cycles, and
    /// every parent precedes its child (topological order).
    pub fn validate(&self) -> Result<(), String> {
        for (i, b) in self.bones.iter().enumerate() {
            if let Some(p) = b.parent {
                if p >= self.bones.len() {
                    return Err(format!("bone {} ('{}') has out-of-range parent {}", i, b.name, p));
                }
                if p == i {
                    return Err(format!("bone {} ('{}') is its own parent", i, b.name));
                }
                if p > i {
                    return Err(format!(
                        "bone {} ('{}') has parent {} after it; skeleton not in topological order",
                        i, b.name, p
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Serialize a `Mat4` as a flat 16-element column-major array for JSON.
mod mat4_serde {
    use glam::Mat4;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(m: &Mat4, s: S) -> Result<S::Ok, S::Error> {
        m.to_cols_array().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Mat4, D::Error> {
        let a = <[f32; 16]>::deserialize(d)?;
        Ok(Mat4::from_cols_array(&a))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Vec3};

    fn chain() -> Skeleton {
        // root at origin → child translated +2 on X → grandchild +3 on X
        let mut s = Skeleton::new();
        let r = s.add(Bone::root("root"));
        let c = s.add(Bone::new("child", Some(r), Mat4::from_translation(Vec3::new(2.0, 0.0, 0.0))));
        s.add(Bone::new("grand", Some(c), Mat4::from_translation(Vec3::new(3.0, 0.0, 0.0))));
        s
    }

    #[test]
    fn global_bind_composes_chain() {
        let s = chain();
        assert_eq!(s.head_position(0), Vec3::ZERO);
        assert_eq!(s.head_position(1), Vec3::new(2.0, 0.0, 0.0));
        assert_eq!(s.head_position(2), Vec3::new(5.0, 0.0, 0.0));
    }

    #[test]
    fn inverse_bind_is_inverse_of_global() {
        let s = chain();
        for i in 0..s.len() {
            let prod = s.inverse_bind(i) * s.global_bind(i);
            assert!(prod.abs_diff_eq(Mat4::IDENTITY, 1e-5));
        }
    }

    #[test]
    fn children_and_roots() {
        let s = chain();
        assert_eq!(s.roots(), vec![0]);
        assert_eq!(s.children(0), vec![1]);
        assert_eq!(s.children(1), vec![2]);
        assert!(s.children(2).is_empty());
    }

    #[test]
    fn validate_rejects_forward_parent() {
        let mut s = Skeleton::new();
        s.bones.push(Bone::new("a", Some(1), Mat4::IDENTITY));
        s.bones.push(Bone::root("b"));
        assert!(s.validate().is_err());
    }

    #[test]
    fn validate_accepts_topological() {
        assert!(chain().validate().is_ok());
    }
}
