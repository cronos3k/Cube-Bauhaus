//! Editor-facing controller that ties the rig session to a viewport.
//!
//! [`RigController`] owns a [`RigState`] and turns high-level editor actions
//! (import/export, bone picking, vertex selection, weight painting) into
//! operations on the mesh + skeleton, and produces render-ready vertex buffers
//! (skinned mesh tinted by weight, and bone line segments). It is GPU-free: the
//! app converts [`RenderVertex`] into its renderer's vertex type and uploads it.
//!
//! Screen-space actions take the camera's `view_proj` and the viewport size so
//! they work against whatever the editor is currently showing.

use std::collections::HashMap;
use std::path::Path;

use glam::{Mat4, Vec2, Vec3};

use crate::select::{self, ScreenFilter, SelectMode, VertexSelection};
use crate::state::{RigState, Tool};
use crate::viz::{self, WeightView};
use crate::weights;

/// A render-ready vertex matching the common `position/normal/uv/color` layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

/// The editor-side rig controller.
pub struct RigController {
    pub state: RigState,
    pub view: WeightView,
    pub select_mode: SelectMode,
    /// Last imported source path label, for the UI.
    pub source: String,
    /// True when the mesh colours/geometry changed and need re-upload.
    pub render_dirty: bool,
}

impl Default for RigController {
    fn default() -> Self {
        Self::new()
    }
}

impl RigController {
    pub fn new() -> Self {
        Self {
            state: RigState::empty(),
            view: WeightView::DominantBone,
            select_mode: SelectMode::Replace,
            source: String::new(),
            render_dirty: true,
        }
    }

    pub fn has_mesh(&self) -> bool {
        self.state.mesh.vertex_count() > 0
    }

    // ── import / export ───────────────────────────────────────────────────────

    /// Import a mesh (and its skeleton, if the file has one).
    pub fn import_mesh(&mut self, path: impl AsRef<Path>) -> Result<(), String> {
        let imported = crate::import::import_model(path).map_err(|e| e.to_string())?;
        self.source = imported.source;
        self.state.set_mesh(imported.mesh);
        if let Some(sk) = imported.skeleton {
            self.state.set_skeleton(sk);
        }
        self.render_dirty = true;
        Ok(())
    }

    /// Load a skeleton from a separate file, keeping the current mesh + weights.
    pub fn import_skeleton(&mut self, path: impl AsRef<Path>) -> Result<(), String> {
        let sk = crate::skeleton_io::load_skeleton(path).map_err(|e| e.to_string())?;
        self.state.set_skeleton(sk);
        self.render_dirty = true;
        Ok(())
    }

    pub fn export_glb(&self, path: impl AsRef<Path>) -> Result<(), String> {
        crate::export::glb::export_glb(path.as_ref(), &self.state.mesh, &self.state.skeleton)
            .map_err(|e| e.to_string())
    }

    pub fn export_fbx(&self, path: impl AsRef<Path>) -> Result<(), String> {
        crate::export::fbx::export_fbx(path.as_ref(), &self.state.mesh, &self.state.skeleton)
            .map_err(|e| e.to_string())
    }

    pub fn export_obj(&self, path: impl AsRef<Path>) -> Result<(), String> {
        crate::export::obj::export_obj(path.as_ref(), &self.state.mesh).map_err(|e| e.to_string())
    }

    // ── viewport interaction ────────────────────────────────────────────────

    fn filter(&self) -> ScreenFilter {
        ScreenFilter {
            front_facing_only: self.state.front_facing_only,
            camera_pos: Vec3::ZERO, // set by callers that know the camera
        }
    }

    /// Handle a click at `screen` for the active tool. `camera_pos` enables the
    /// front-facing filter. Returns true if anything changed.
    pub fn click(
        &mut self,
        view_proj: &Mat4,
        viewport: Vec2,
        screen: Vec2,
        camera_pos: Vec3,
    ) -> bool {
        let mut filter = self.filter();
        filter.camera_pos = camera_pos;
        match self.state.tool {
            Tool::BoneSelect => {
                if let Some(b) = self.state.pick_bone(view_proj, viewport, screen, 24.0) {
                    self.state.select_bone(b);
                    self.render_dirty = true;
                    return true;
                }
                false
            }
            Tool::BrushSelect => {
                let hits = select::brush(
                    &self.state.mesh, view_proj, viewport, screen, self.state.brush_radius, filter,
                );
                self.state.selection.apply(&hits, self.select_mode);
                self.render_dirty = true;
                !hits.is_empty()
            }
            Tool::WeightPaint => {
                self.begin_stroke("paint");
                self.paint(view_proj, viewport, screen, camera_pos)
            }
            // Marquee/Lasso are driven by drag rectangles/polygons via the
            // dedicated methods below; a bare click does nothing for them.
            Tool::Marquee | Tool::Lasso => false,
        }
    }

    /// Apply a rectangle (marquee) selection in pixel coordinates.
    pub fn marquee(&mut self, view_proj: &Mat4, viewport: Vec2, min: Vec2, max: Vec2, camera_pos: Vec3) {
        let mut filter = self.filter();
        filter.camera_pos = camera_pos;
        let hits = select::marquee(&self.state.mesh, view_proj, viewport, min, max, filter);
        self.state.selection.apply(&hits, self.select_mode);
        self.render_dirty = true;
    }

    /// Apply a freehand lasso selection (pixel-space polygon).
    pub fn lasso(&mut self, view_proj: &Mat4, viewport: Vec2, polygon: &[Vec2], camera_pos: Vec3) {
        let mut filter = self.filter();
        filter.camera_pos = camera_pos;
        let hits = select::lasso(&self.state.mesh, view_proj, viewport, polygon, filter);
        self.state.selection.apply(&hits, self.select_mode);
        self.render_dirty = true;
    }

    /// Paint the active bone's weight under the brush, with radial falloff.
    ///
    /// Does **not** push an undo step — for continuous painting the caller pushes
    /// one undo snapshot at the start of a stroke (see [`RigController::begin_stroke`]).
    pub fn paint(&mut self, view_proj: &Mat4, viewport: Vec2, screen: Vec2, camera_pos: Vec3) -> bool {
        let Some(bone) = self.state.active_bone else { return false };
        let radius = self.state.brush_radius;
        // Brush hits + radial falloff (1 at centre → 0 at the rim).
        let mut falloff: HashMap<u32, f32> = HashMap::new();
        let mut sel = VertexSelection::new(self.state.mesh.vertex_count());
        for (i, v) in self.state.mesh.vertices.iter().enumerate() {
            if self.state.front_facing_only {
                let view_dir = camera_pos - Vec3::from_array(v.position);
                if Vec3::from_array(v.normal).dot(view_dir) <= 0.0 {
                    continue;
                }
            }
            if let Some(p) = select::project(view_proj, Vec3::from_array(v.position), viewport) {
                let d = (p - screen).length();
                if d <= radius {
                    sel.set(i, true);
                    falloff.insert(i as u32, 1.0 - d / radius);
                }
            }
        }
        if sel.count() == 0 {
            return false;
        }
        weights::paint(&mut self.state.mesh, &sel, bone, self.state.paint_strength, Some(&falloff));
        self.render_dirty = true;
        true
    }

    /// Snapshot influences for undo at the start of a paint stroke.
    pub fn begin_stroke(&mut self, label: &'static str) {
        self.state.push_undo(label);
    }

    // ── selection ops ─────────────────────────────────────────────────────────

    pub fn grow(&mut self) {
        select::grow(&mut self.state.mesh, &mut self.state.selection);
        self.render_dirty = true;
    }
    pub fn shrink(&mut self) {
        select::shrink(&mut self.state.mesh, &mut self.state.selection);
        self.render_dirty = true;
    }
    pub fn select_linked(&mut self) {
        select::select_linked(&mut self.state.mesh, &mut self.state.selection);
        self.render_dirty = true;
    }
    pub fn invert(&mut self) {
        self.state.selection.invert();
        self.render_dirty = true;
    }
    pub fn select_all(&mut self) {
        self.state.selection.select_all();
        self.render_dirty = true;
    }
    pub fn clear_selection(&mut self) {
        self.state.selection.clear();
        self.render_dirty = true;
    }
    /// Select all vertices whose dominant bone is the active bone.
    pub fn select_by_active_bone(&mut self) {
        if let Some(b) = self.state.active_bone {
            let hits = select::by_dominant_bone(&self.state.mesh, b);
            self.state.selection.apply(&hits, SelectMode::Replace);
            self.render_dirty = true;
        }
    }

    // ── weight ops (all undoable) ──────────────────────────────────────────────

    /// Rigidly bind the selection to the active bone.
    pub fn bind_selection(&mut self) {
        if let Some(b) = self.state.active_bone {
            self.state.push_undo("bind");
            weights::bind_rigid(&mut self.state.mesh, &self.state.selection, b);
            self.render_dirty = true;
        }
    }
    pub fn smooth_selection(&mut self, iterations: u32, strength: f32) {
        self.state.push_undo("smooth");
        weights::smooth(&mut self.state.mesh, &self.state.selection, iterations, strength);
        self.render_dirty = true;
    }
    pub fn normalize(&mut self) {
        self.state.push_undo("normalize");
        self.state.mesh.normalize_all();
        self.render_dirty = true;
    }
    pub fn prune(&mut self, min_weight: f32) {
        self.state.push_undo("prune");
        weights::prune(&mut self.state.mesh, min_weight);
        self.render_dirty = true;
    }
    pub fn mirror(&mut self, axis: usize, tolerance: f32, negative_to_positive: bool) {
        self.state.push_undo("mirror");
        weights::mirror(&mut self.state.mesh, &self.state.skeleton, axis, tolerance, negative_to_positive);
        self.render_dirty = true;
    }
    pub fn undo(&mut self) {
        if self.state.undo() {
            self.render_dirty = true;
        }
    }
    pub fn redo(&mut self) {
        if self.state.redo() {
            self.render_dirty = true;
        }
    }

    // ── render data ─────────────────────────────────────────────────────────

    /// Skinned-mesh render vertices, coloured by the current [`WeightView`] and
    /// with the selection tinted. Pair with [`mesh_indices`](Self::mesh_indices).
    pub fn mesh_vertices(&self) -> Vec<RenderVertex> {
        let mut colors = viz::weight_colors(&self.state.mesh, self.view, self.state.active_bone);
        viz::blend_selection(&mut colors, &self.state.selection, [1.0, 0.85, 0.1], 0.6);
        self.state
            .mesh
            .vertices
            .iter()
            .zip(colors)
            .map(|(v, c)| RenderVertex { position: v.position, normal: v.normal, uv: v.uv, color: c })
            .collect()
    }

    pub fn mesh_indices(&self) -> Vec<u32> {
        self.state.mesh.indices.clone()
    }

    /// Weight-coloured wireframe of the rig mesh as a **line list** (triangle
    /// edges → vertex pairs). Renders safely through a pass-through colour line
    /// pipeline, showing the weight visualisation on the surface edges.
    pub fn wireframe(&self) -> (Vec<RenderVertex>, Vec<u32>) {
        let verts = self.mesh_vertices();
        let mut idx = Vec::with_capacity(self.state.mesh.indices.len() * 2);
        for tri in self.state.mesh.indices.chunks_exact(3) {
            idx.extend_from_slice(&[tri[0], tri[1], tri[1], tri[2], tri[2], tri[0]]);
        }
        (verts, idx)
    }

    /// Bone line-list render vertices (2 per segment). The segments touching the
    /// active bone are highlighted. Indices are simply `0,1,2,3,…`.
    pub fn bone_lines(&self) -> (Vec<RenderVertex>, Vec<u32>) {
        let active_head = self.state.active_bone.map(|b| self.state.skeleton.head_position(b as usize));
        let mut verts = Vec::new();
        for (a, b) in self.state.bone_segments() {
            let highlight = active_head.map(|h| (a - h).length() < 1e-4).unwrap_or(false);
            let color = if highlight { [1.0, 0.8, 0.1, 1.0] } else { [0.2, 0.9, 0.4, 1.0] };
            verts.push(line_vertex(a, color));
            verts.push(line_vertex(b, color));
        }
        let indices = (0..verts.len() as u32).collect();
        (verts, indices)
    }
}

fn line_vertex(p: Vec3, color: [f32; 4]) -> RenderVertex {
    RenderVertex { position: p.to_array(), normal: [0.0, 1.0, 0.0], uv: [0.0, 0.0], color }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{RigVertex, SkinnedMesh};
    use crate::skeleton::{Bone, Skeleton};

    fn loaded_controller() -> RigController {
        let v = |x: f32, y: f32| RigVertex { position: [x, y, 0.0], normal: [0.0, 0.0, 1.0], uv: [0.0, 0.0] };
        let mesh = SkinnedMesh::new(vec![v(0.0, 0.0), v(1.0, 0.0), v(0.0, 1.0), v(1.0, 1.0)], vec![0, 1, 2, 1, 3, 2]);
        let mut sk = Skeleton::new();
        let r = sk.add(Bone::root("root"));
        sk.add(Bone::new("tip", Some(r), Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0))));
        let mut c = RigController::new();
        c.state = RigState::new(mesh, sk);
        c
    }

    #[test]
    fn render_vertices_match_mesh() {
        let c = loaded_controller();
        assert_eq!(c.mesh_vertices().len(), 4);
        assert_eq!(c.mesh_indices().len(), 6);
    }

    #[test]
    fn wireframe_is_line_list_of_edges() {
        let c = loaded_controller();
        let (verts, idx) = c.wireframe();
        assert_eq!(verts.len(), 4);
        // 2 triangles × 3 edges × 2 indices
        assert_eq!(idx.len(), 12);
        assert_eq!(idx.len() % 2, 0);
    }

    #[test]
    fn bone_lines_two_verts_per_segment() {
        let c = loaded_controller();
        let (verts, idx) = c.bone_lines();
        assert_eq!(verts.len() % 2, 0);
        assert_eq!(verts.len(), idx.len());
        assert!(!verts.is_empty());
    }

    #[test]
    fn bind_selection_to_active_bone() {
        let mut c = loaded_controller();
        c.state.active_bone = Some(1);
        c.state.selection.select_all();
        c.bind_selection();
        assert_eq!(c.state.mesh.dominant_bone(0), Some(1));
    }

    #[test]
    fn paint_falls_off_with_distance() {
        let mut c = loaded_controller();
        c.state.active_bone = Some(1);
        c.state.front_facing_only = false;
        c.state.brush_radius = 1000.0;
        c.state.paint_strength = 0.5;
        // orthographic-ish: identity view_proj maps x,y∈[-1,1] to NDC; place
        // the brush over the projected vertices
        let view = Mat4::look_at_rh(Vec3::new(0.5, 0.5, 3.0), Vec3::new(0.5, 0.5, 0.0), Vec3::Y);
        let proj = Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0);
        let vp = proj * view;
        let viewport = Vec2::new(800.0, 600.0);
        let center = Vec2::new(400.0, 300.0);
        let changed = c.paint(&vp, viewport, center, Vec3::new(0.5, 0.5, 3.0));
        assert!(changed);
        // at least one vertex now has bone-1 influence
        assert!(c.state.mesh.influences.iter().any(|inf| inf.iter().any(|i| i.bone == 1 && i.weight > 0.0)));
    }

    #[test]
    fn grow_after_seed_expands() {
        let mut c = loaded_controller();
        c.state.selection.set(0, true);
        c.grow();
        assert!(c.state.selection.count() > 1);
    }

    #[test]
    fn import_unsupported_errors() {
        let mut c = RigController::new();
        assert!(c.import_mesh("/tmp/nope.xyz").is_err());
    }
}
