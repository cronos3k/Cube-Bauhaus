//! Editor integration for the rigging bay.
//!
//! [`RigBay`] wraps a [`cube_rig::RigController`] and bridges it to the live app:
//! it draws the egui control panel, handles import/export file dialogs (rfd),
//! maps viewport clicks to bone-pick / vertex-select / weight-paint, and builds
//! the renderer meshes (a weight-coloured wireframe of the rig mesh plus bone
//! line segments) for the line pipeline.
//!
//! GPU mesh lifetime is owned by the caller (`main`), which rebuilds the meshes
//! from [`RigBay::take_render_data`] when the controller is dirty, reusing the
//! editor's deferred-deletion queue.

use bbc_renderer::{FlyCamera, Vertex};
use cube_rig::controller::RenderVertex;
use cube_rig::{RigController, SelectMode, Tool, WeightView};
use glam::{Mat4, Vec2, Vec3};

/// Rigging-bay editor state layered over the [`RigController`].
pub struct RigBay {
    pub enabled: bool,
    pub ctrl: RigController,
    pub status: String,

    // Deferred file-dialog requests (handled outside the egui closure).
    req_import_mesh: bool,
    req_import_skeleton: bool,
    req_export_glb: bool,
    req_export_fbx: bool,
    req_export_obj: bool,

    // Drag state for marquee / lasso / continuous paint.
    drag_start: Option<Vec2>,
    lasso_points: Vec<Vec2>,
    stroking: bool,
}

impl Default for RigBay {
    fn default() -> Self {
        Self::new()
    }
}

impl RigBay {
    pub fn new() -> Self {
        Self {
            enabled: false,
            ctrl: RigController::new(),
            status: String::new(),
            req_import_mesh: false,
            req_import_skeleton: false,
            req_export_glb: false,
            req_export_fbx: false,
            req_export_obj: false,
            drag_start: None,
            lasso_points: Vec::new(),
            stroking: false,
        }
    }

    /// Standard RH view-projection (no Vulkan clip flip) for screen-space picking
    /// — matches what `cube_rig::select::project` expects.
    fn pick_view_proj(camera: &FlyCamera, width: u32, height: u32) -> Mat4 {
        let aspect = width as f32 / height.max(1) as f32;
        let view = Mat4::look_at_rh(camera.pos, camera.pos + camera.forward(), Vec3::Y);
        let proj = Mat4::perspective_rh(60_f32.to_radians(), aspect, 0.5, 65536.0);
        proj * view
    }

    /// Draw the Rig Bay egui window. Sets deferred request flags; mutates the
    /// controller's tool/view/brush settings directly.
    pub fn ui(&mut self, ctx: &egui::Context) {
        if !self.enabled {
            return;
        }
        let mut open = true;
        egui::Window::new("Rig Bay")
            .default_width(260.0)
            .open(&mut open)
            .show(ctx, |ui| self.window_contents(ui));
        if !open {
            self.enabled = false;
        }
    }

    fn window_contents(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button("Import Mesh…").clicked() {
                self.req_import_mesh = true;
            }
            if ui.button("Import Skeleton…").clicked() {
                self.req_import_skeleton = true;
            }
        });
        if !self.ctrl.source.is_empty() {
            ui.label(format!("Source: {}", short_path(&self.ctrl.source)));
        }
        ui.separator();

        if !self.ctrl.has_mesh() {
            ui.label("Import a mesh to begin rigging.");
            return;
        }

        // ── Tools ──────────────────────────────────────────────────────────
        ui.label("Tool");
        ui.horizontal_wrapped(|ui| {
            for (tool, label) in [
                (Tool::BoneSelect, "Bone"),
                (Tool::Marquee, "Marquee"),
                (Tool::Lasso, "Lasso"),
                (Tool::BrushSelect, "Brush"),
                (Tool::WeightPaint, "Paint"),
            ] {
                ui.selectable_value(&mut self.ctrl.state.tool, tool, label);
            }
        });

        ui.horizontal(|ui| {
            ui.label("Combine");
            for (m, l) in [
                (SelectMode::Replace, "Replace"),
                (SelectMode::Add, "Add"),
                (SelectMode::Subtract, "Sub"),
            ] {
                ui.selectable_value(&mut self.ctrl.select_mode, m, l);
            }
        });
        ui.checkbox(&mut self.ctrl.state.front_facing_only, "Front-facing only");
        ui.add(egui::Slider::new(&mut self.ctrl.state.brush_radius, 2.0..=200.0).text("Brush px"));
        ui.add(egui::Slider::new(&mut self.ctrl.state.paint_strength, -1.0..=1.0).text("Paint strength"));

        ui.separator();
        // ── Selection ops ────────────────────────────────────────────────────
        ui.label(format!("Selected: {} verts", self.ctrl.state.selection.count()));
        ui.horizontal_wrapped(|ui| {
            if ui.button("All").clicked() { self.ctrl.select_all(); }
            if ui.button("None").clicked() { self.ctrl.clear_selection(); }
            if ui.button("Invert").clicked() { self.ctrl.invert(); }
            if ui.button("Grow").clicked() { self.ctrl.grow(); }
            if ui.button("Shrink").clicked() { self.ctrl.shrink(); }
            if ui.button("Linked").clicked() { self.ctrl.select_linked(); }
            if ui.button("By Bone").clicked() { self.ctrl.select_by_active_bone(); }
        });

        ui.separator();
        // ── Bones ──────────────────────────────────────────────────────────
        ui.label("Bones");
        egui::ScrollArea::vertical().max_height(140.0).show(ui, |ui| {
            for i in 0..self.ctrl.state.skeleton.len() {
                let name = self.ctrl.state.skeleton.bones[i].name.clone();
                let selected = self.ctrl.state.active_bone == Some(i as u16);
                if ui.selectable_label(selected, format!("{i}: {name}")).clicked() {
                    self.ctrl.state.select_bone(i as u16);
                    self.ctrl.render_dirty = true;
                }
            }
        });

        ui.separator();
        // ── Weight ops ───────────────────────────────────────────────────────
        ui.horizontal_wrapped(|ui| {
            if ui.button("Bind→Bone").clicked() { self.ctrl.bind_selection(); }
            if ui.button("Smooth").clicked() { self.ctrl.smooth_selection(2, 0.5); }
            if ui.button("Normalize").clicked() { self.ctrl.normalize(); }
            if ui.button("Prune").clicked() { self.ctrl.prune(0.01); }
            if ui.button("Mirror X").clicked() { self.ctrl.mirror(0, 1e-3, false); }
            if ui.button("Undo").clicked() { self.ctrl.undo(); }
            if ui.button("Redo").clicked() { self.ctrl.redo(); }
        });

        ui.separator();
        // ── Weight view ──────────────────────────────────────────────────────
        ui.horizontal(|ui| {
            ui.label("View");
            for (v, l) in [
                (WeightView::DominantBone, "Bones"),
                (WeightView::Gradient, "Gradient"),
                (WeightView::Off, "Off"),
            ] {
                if ui.selectable_label(self.ctrl.view == v, l).clicked() {
                    self.ctrl.view = v;
                    self.ctrl.render_dirty = true;
                }
            }
        });

        ui.separator();
        // ── Export ───────────────────────────────────────────────────────────
        ui.horizontal(|ui| {
            if ui.button("Export GLB…").clicked() { self.req_export_glb = true; }
            if ui.button("Export FBX…").clicked() { self.req_export_fbx = true; }
            if ui.button("Export OBJ…").clicked() { self.req_export_obj = true; }
        });
        if !self.status.is_empty() {
            ui.label(&self.status);
        }
    }

    /// Process deferred file-dialog requests (called once per frame, outside the
    /// egui closure). Uses blocking rfd dialogs, matching the rest of the editor.
    pub fn process_requests(&mut self) {
        if std::mem::take(&mut self.req_import_mesh) {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Mesh", &["glb", "gltf", "obj", "fbx"])
                .pick_file()
            {
                self.status = match self.ctrl.import_mesh(&path) {
                    Ok(()) => format!("Imported {}", short_path(&path.display().to_string())),
                    Err(e) => format!("Import failed: {e}"),
                };
            }
        }
        if std::mem::take(&mut self.req_import_skeleton) {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Skeleton", &["glb", "gltf", "json"])
                .pick_file()
            {
                self.status = match self.ctrl.import_skeleton(&path) {
                    Ok(()) => "Skeleton loaded".into(),
                    Err(e) => format!("Skeleton load failed: {e}"),
                };
            }
        }
        self.process_export();
    }

    fn process_export(&mut self) {
        let want = if std::mem::take(&mut self.req_export_glb) {
            Some(("glb", "GLB"))
        } else if std::mem::take(&mut self.req_export_fbx) {
            Some(("fbx", "FBX"))
        } else if std::mem::take(&mut self.req_export_obj) {
            Some(("obj", "OBJ"))
        } else {
            None
        };
        let Some((ext, label)) = want else { return };
        if let Some(path) = rfd::FileDialog::new().add_filter(label, &[ext]).save_file() {
            let res = match ext {
                "glb" => self.ctrl.export_glb(&path),
                "fbx" => self.ctrl.export_fbx(&path),
                _ => self.ctrl.export_obj(&path),
            };
            self.status = match res {
                Ok(()) => format!("Exported {label}"),
                Err(e) => format!("Export failed: {e}"),
            };
        }
    }

    /// Drive the active tool from pointer state each frame. Handles instantaneous
    /// actions (bone pick), drag selections (marquee/lasso) and continuous strokes
    /// (brush-select, weight-paint). Call when the pointer isn't over egui and the
    /// camera isn't in mouse-look.
    pub fn handle_pointer(
        &mut self,
        camera: &FlyCamera,
        cursor: Vec2,
        width: u32,
        height: u32,
        down: bool,
        just_pressed: bool,
        just_released: bool,
    ) {
        if !self.enabled || !self.ctrl.has_mesh() {
            return;
        }
        let vp = Self::pick_view_proj(camera, width, height);
        let viewport = Vec2::new(width as f32, height as f32);
        let cam_pos = camera.pos;

        match self.ctrl.state.tool {
            Tool::BoneSelect => {
                if just_pressed {
                    self.ctrl.click(&vp, viewport, cursor, cam_pos);
                }
            }
            Tool::BrushSelect => {
                // Continuous: select under the brush while dragging.
                if down {
                    self.ctrl.click(&vp, viewport, cursor, cam_pos);
                }
            }
            Tool::WeightPaint => {
                if just_pressed {
                    self.ctrl.begin_stroke("paint stroke");
                    self.stroking = true;
                }
                if down && self.stroking {
                    self.ctrl.paint(&vp, viewport, cursor, cam_pos);
                }
                if just_released {
                    self.stroking = false;
                }
            }
            Tool::Marquee => {
                if just_pressed {
                    self.drag_start = Some(cursor);
                }
                if just_released {
                    if let Some(start) = self.drag_start.take() {
                        self.ctrl.marquee(&vp, viewport, start, cursor, cam_pos);
                    }
                }
            }
            Tool::Lasso => {
                if just_pressed {
                    self.lasso_points.clear();
                    self.lasso_points.push(cursor);
                }
                if down {
                    // Throttle points so the polygon stays light.
                    if self.lasso_points.last().map_or(true, |p| (*p - cursor).length() > 4.0) {
                        self.lasso_points.push(cursor);
                    }
                }
                if just_released {
                    if self.lasso_points.len() >= 3 {
                        let poly = std::mem::take(&mut self.lasso_points);
                        self.ctrl.lasso(&vp, viewport, &poly, cam_pos);
                    }
                    self.lasso_points.clear();
                }
            }
        }
    }

    /// True if the controller's render data changed and the GPU meshes should be
    /// rebuilt. Clears the flag.
    pub fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.ctrl.render_dirty)
    }

    /// Build the renderer vertex/index data for this frame: the weight-coloured
    /// wireframe and the bone line segments. Both are line lists.
    pub fn take_render_data(&self) -> (Vec<Vertex>, Vec<u32>, Vec<Vertex>, Vec<u32>) {
        let (wv, wi) = self.ctrl.wireframe();
        let (bv, bi) = self.ctrl.bone_lines();
        (to_vertices(&wv), wi, to_vertices(&bv), bi)
    }
}

fn to_vertices(src: &[RenderVertex]) -> Vec<Vertex> {
    src.iter()
        .map(|v| Vertex { position: v.position, normal: v.normal, uv: v.uv, color: v.color })
        .collect()
}

fn short_path(p: &str) -> String {
    std::path::Path::new(p)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.to_string())
}
