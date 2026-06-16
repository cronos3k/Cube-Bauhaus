//! The editor-side rigging session.
//!
//! [`RigState`] bundles the imported mesh, its skeleton, the active bone and the
//! current vertex selection, plus the active tool and a small undo stack of
//! influence snapshots. The editor (egui + viewport) reads and drives this; the
//! state itself stays GPU-free and engine-agnostic.

use glam::Vec3;

use crate::mesh::{SkinnedMesh, VertexInfluences};
use crate::select::VertexSelection;
use crate::skeleton::Skeleton;

/// Which interaction tool is active in the rig bay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(Default)]
pub enum Tool {
    /// Pick / parent bones in the viewport.
    #[default]
    BoneSelect,
    /// Rectangle vertex selection.
    Marquee,
    /// Free-hand polygon vertex selection.
    Lasso,
    /// Radial brush vertex selection.
    BrushSelect,
    /// Paint weight for the active bone.
    WeightPaint,
}


/// A reversible edit to per-vertex influences (the only thing weight/selection
/// ops mutate that's worth undoing). Selection changes are cheap and not
/// snapshotted.
#[derive(Clone)]
struct InfluenceSnapshot {
    label: &'static str,
    influences: Vec<VertexInfluences>,
}

/// Everything the rigging editor needs for one session.
pub struct RigState {
    pub mesh: SkinnedMesh,
    pub skeleton: Skeleton,
    pub selection: VertexSelection,
    /// The bone targeted by bind/paint actions and highlighted in the viewport.
    pub active_bone: Option<u16>,
    pub tool: Tool,
    /// Brush radius in pixels for the brush-select and weight-paint tools.
    pub brush_radius: f32,
    /// Weight delta applied per paint dab (negative erases).
    pub paint_strength: f32,
    /// Only affect camera-facing vertices in screen-space selection tools.
    pub front_facing_only: bool,

    undo: Vec<InfluenceSnapshot>,
    redo: Vec<InfluenceSnapshot>,
    max_undo: usize,
}

impl RigState {
    /// Start a session from an imported mesh and skeleton.
    pub fn new(mesh: SkinnedMesh, skeleton: Skeleton) -> Self {
        let selection = VertexSelection::new(mesh.vertex_count());
        let active_bone = if skeleton.is_empty() { None } else { Some(0) };
        Self {
            mesh,
            skeleton,
            selection,
            active_bone,
            tool: Tool::default(),
            brush_radius: 24.0,
            paint_strength: 0.25,
            front_facing_only: true,
            undo: Vec::new(),
            redo: Vec::new(),
            max_undo: 64,
        }
    }

    /// An empty session (nothing imported yet).
    pub fn empty() -> Self {
        Self::new(SkinnedMesh::default(), Skeleton::new())
    }

    /// Replace the mesh (e.g. after importing a new file), resizing selection
    /// and clearing undo history.
    pub fn set_mesh(&mut self, mesh: SkinnedMesh) {
        self.selection = VertexSelection::new(mesh.vertex_count());
        self.mesh = mesh;
        self.undo.clear();
        self.redo.clear();
    }

    /// Replace the skeleton, keeping the mesh and weights. Clamps the active
    /// bone to the new range.
    pub fn set_skeleton(&mut self, skeleton: Skeleton) {
        self.active_bone = if skeleton.is_empty() {
            None
        } else {
            Some(self.active_bone.unwrap_or(0).min(skeleton.len() as u16 - 1))
        };
        self.skeleton = skeleton;
    }

    pub fn select_bone(&mut self, bone: u16) {
        if (bone as usize) < self.skeleton.len() {
            self.active_bone = Some(bone);
        }
    }

    /// Pick the bone whose projected head is closest to a screen point, within
    /// `max_pixels`. Returns the chosen bone, if any. (Pure helper; the editor
    /// supplies the view-projection.)
    pub fn pick_bone(
        &self,
        view_proj: &glam::Mat4,
        viewport: glam::Vec2,
        screen: glam::Vec2,
        max_pixels: f32,
    ) -> Option<u16> {
        let mut best: Option<(u16, f32)> = None;
        for i in 0..self.skeleton.len() {
            let head = self.skeleton.head_position(i);
            if let Some(p) = crate::select::project(view_proj, head, viewport) {
                let d = (p - screen).length();
                if d <= max_pixels && best.is_none_or(|(_, bd)| d < bd) {
                    best = Some((i as u16, d));
                }
            }
        }
        best.map(|(b, _)| b)
    }

    /// World-space line segments for drawing the bones (head → each child head,
    /// or head → a short stub for leaf bones).
    pub fn bone_segments(&self) -> Vec<(Vec3, Vec3)> {
        let mut segs = Vec::new();
        for i in 0..self.skeleton.len() {
            let head = self.skeleton.head_position(i);
            let children = self.skeleton.children(i);
            if children.is_empty() {
                // stub along the bone's local +Y so leaf bones are visible
                let tip = self.skeleton.global_bind(i).transform_point3(Vec3::new(0.0, 0.1, 0.0));
                segs.push((head, tip));
            } else {
                for c in children {
                    segs.push((head, self.skeleton.head_position(c)));
                }
            }
        }
        segs
    }

    // ── undo / redo ──────────────────────────────────────────────────────────

    /// Snapshot the current influences before a mutating op. Call this *before*
    /// any [`weights`](crate::weights) operation you want to be undoable.
    pub fn push_undo(&mut self, label: &'static str) {
        self.redo.clear();
        self.undo.push(InfluenceSnapshot { label, influences: self.mesh.influences.clone() });
        if self.undo.len() > self.max_undo {
            self.undo.remove(0);
        }
    }

    /// The label of the next undo step, for menu display.
    pub fn peek_undo(&self) -> Option<&'static str> {
        self.undo.last().map(|s| s.label)
    }

    pub fn undo(&mut self) -> bool {
        if let Some(snap) = self.undo.pop() {
            self.redo.push(InfluenceSnapshot {
                label: snap.label,
                influences: std::mem::replace(&mut self.mesh.influences, snap.influences),
            });
            true
        } else {
            false
        }
    }

    pub fn redo(&mut self) -> bool {
        if let Some(snap) = self.redo.pop() {
            self.undo.push(InfluenceSnapshot {
                label: snap.label,
                influences: std::mem::replace(&mut self.mesh.influences, snap.influences),
            });
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::RigVertex;
    use crate::skeleton::Bone;
    use glam::{Mat4, Vec3};

    fn tri_mesh() -> SkinnedMesh {
        let v = |x: f32| RigVertex { position: [x, 0.0, 0.0], normal: [0.0, 0.0, 1.0], uv: [0.0, 0.0] };
        SkinnedMesh::new(vec![v(0.0), v(1.0), v(2.0)], vec![0, 1, 2])
    }

    #[test]
    fn undo_redo_restores_influences() {
        let mut sk = Skeleton::new();
        sk.add(Bone::root("root"));
        let mut st = RigState::new(tri_mesh(), sk);

        st.push_undo("bind");
        st.mesh.set_rigid(0, 0);
        assert_eq!(st.mesh.dominant_bone(0), Some(0));

        assert!(st.undo());
        assert_eq!(st.mesh.dominant_bone(0), None);

        assert!(st.redo());
        assert_eq!(st.mesh.dominant_bone(0), Some(0));
    }

    #[test]
    fn pick_nearest_bone() {
        let mut sk = Skeleton::new();
        sk.add(Bone::new("a", None, Mat4::from_translation(Vec3::new(-1.0, 0.0, 0.0))));
        sk.add(Bone::new("b", None, Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0))));
        let st = RigState::new(tri_mesh(), sk);

        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0);
        let vp = proj * view;
        let viewport = glam::Vec2::new(100.0, 100.0);

        // bone "a" projects to the left half; click left of centre
        let left = crate::select::project(&vp, Vec3::new(-1.0, 0.0, 0.0), viewport).unwrap();
        assert_eq!(st.pick_bone(&vp, viewport, left, 50.0), Some(0));
    }

    #[test]
    fn set_mesh_resizes_selection() {
        let mut st = RigState::empty();
        let m = tri_mesh();
        st.set_mesh(m);
        assert_eq!(st.selection.len(), 3);
    }
}
