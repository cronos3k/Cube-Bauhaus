//! In-game editor — Cube2/Sauerbraten-style octree editing with visual feedback.
//!
//! Replicates the rendereditcursor() / editface() / universaldelta workflow
//! from Sauerbraten's octaedit.cpp as closely as possible.
//!
//! How it works (matching C++ exactly):
//!   - A crosshair is always drawn at screen center.
//!   - In edit mode, a raycast from the camera through screen center hits a cube face.
//!   - That cube face becomes the "cursor" (gray wireframe box + white face highlight).
//!   - Without clicking, scroll already edits the cursor cube (fill/empty mode 1).
//!   - LMB click starts a drag selection; release finalizes it.
//!   - With a selection, scroll/face-push/corner-push operate on the selection.
//!
//! Keybindings (matching stdedit.cfg):
//!   E         = toggle edit mode
//!   LMB       = start selection drag
//!   Space     = cancel selection
//!   Scroll    = fill/empty (editface mode 1) — default modifier
//!   F+Scroll  = push/pull face edges (editface mode 0)
//!   Q+Scroll  = push corner (editface mode 2)
//!   G+Scroll  = adjust grid size
//!   1+Scroll  = cycle texture slot (Ctrl=all faces)
//!   2+Scroll  = rotate texture
//!   3+Scroll  = scale texture
//!   4+Scroll  = offset texture (Shift=second axis)
//!   5+Scroll  = cycle material
//!   R+Scroll  = rotate selection
//!   X         = flip selection
//!   Delete    = delete selected cubes
//!   C         = copy
//!   V         = paste
//!   Z / U     = undo
//!   I         = redo

use bbc_renderer::{FlyCamera, Vertex};
use cube_world::{
    EditWorld, Selection, CubeBlock, RayHit,
    FACE_DIM, FACE_SIDE, R, C,
    O_LEFT, O_RIGHT, O_BACK, O_FRONT, O_BOTTOM, O_TOP,
    TextureRegistry, VSlot,
    VSLOT_ROTATION, VSLOT_SCALE, VSLOT_OFFSET, VSLOT_COLOR,
};
use cube_world::octree::OctreeWorld;
use winit::keyboard::KeyCode;

use crate::input::InputState;

// ── Orient conversion: renderer Y-up ↔ Cube2 Z-up ───────────────────────────
//
// Renderer axes: X=right, Y=up, Z=back  (Y-up)
// Cube2 axes:    X=right, Y=back, Z=up  (Z-up)
// Mapping: renderer_Y = cube2_Z, renderer_Z = cube2_Y
//
// Orient values: O_LEFT=0(-X), O_RIGHT=1(+X), O_BACK=2(-Y), O_FRONT=3(+Y),
//                O_BOTTOM=4(-Z), O_TOP=5(+Z)
//
// Renderer orient → Cube2 orient:
//   0 (render -X)  → 0 (cube2 -X)    LEFT → LEFT
//   1 (render +X)  → 1 (cube2 +X)    RIGHT → RIGHT
//   2 (render -Y)  → 4 (cube2 -Z)    render "BACK" → cube2 BOTTOM
//   3 (render +Y)  → 5 (cube2 +Z)    render "FRONT" → cube2 TOP
//   4 (render -Z)  → 2 (cube2 -Y)    render "BOTTOM" → cube2 BACK
//   5 (render +Z)  → 3 (cube2 +Y)    render "TOP" → cube2 FRONT
const ORIENT_RENDERER_TO_CUBE2: [usize; 6] = [0, 1, 4, 5, 2, 3];

// ── Editor State ──────────────────────────────────────────────────────────────

pub struct EditorState {
    pub edit_world: EditWorld,
    pub edit_mode: bool,

    // Selection (Cube2 "sel" struct equivalent)
    pub selection: Selection,
    pub have_sel: bool,
    pub dragging: bool,
    pub select_corners: bool, // true = RMB drag (corner-precise), false = LMB drag (face)
    drag_start: [i32; 3],
    drag_start_orient: usize,
    drag_start_cor: [i32; 2], // doubled-grid coords at drag start [R, C]

    // Grid cursor (the cube the crosshair points at) — in Cube2 Z-up coords
    pub hover: Option<RayHit>,
    pub grid_power: u32,
    pub grid_size: i32,

    // Clipboard
    pub clipboard: Option<CubeBlock>,

    // Frame counter for undo timestamps
    pub timestamp: u64,

    // Dirty flag — set when geometry needs rebuild
    pub mesh_dirty: bool,

    // Texture system
    pub tex_registry: TextureRegistry,

    // Entity editing
    pub selected_entity: Option<usize>,  // index into world.entities

    // File operation requests (handled by main.rs)
    pub request_save: bool,
    pub request_load: bool,
    pub request_newmap: bool,
    pub request_export_glb: bool,
    pub request_export_fbx: bool,
    pub request_package_map: bool,
    pub current_map_path: Option<String>,
    pub packages_dir: Option<String>,
}

impl EditorState {
    pub fn new(world: OctreeWorld) -> Self {
        let grid_power = 3u32; // default like Cube2
        Self {
            edit_world: EditWorld::new(world),
            edit_mode: false,
            selection: Selection::default(),
            have_sel: false,
            dragging: false,
            select_corners: false,
            drag_start: [0; 3],
            drag_start_orient: 0,
            drag_start_cor: [0; 2],
            hover: None,
            grid_power,
            grid_size: 1 << grid_power,
            clipboard: None,
            timestamp: 0,
            mesh_dirty: false,
            tex_registry: Self::make_default_registry(),
            selected_entity: None,
            request_save: false,
            request_load: false,
            request_newmap: false,
            request_export_glb: false,
            request_export_fbx: false,
            request_package_map: false,
            current_map_path: None,
            packages_dir: None,
        }
    }

    /// Create a TextureRegistry with a set of default procedural slots
    /// so that texture cycling (Y+Scroll) has visible variety.
    pub fn make_default_registry_static() -> TextureRegistry {
        Self::make_default_registry()
    }

    fn make_default_registry() -> TextureRegistry {
        use cube_world::texture::TexType;
        let mut reg = TextureRegistry::new(); // slots 0=sky, 1=default

        // Add 14 more slots (total 16) for testing texture operations
        let names = [
            "brick", "stone", "wood", "metal", "grass", "sand",
            "tile", "concrete", "marble", "plaster", "rust", "dirt",
            "gravel", "stucco",
        ];
        for name in &names {
            reg.add_slot_with_tex("stdworld", TexType::Diffuse, name);
        }
        reg
    }

    /// Process one frame of input. Returns true if the world mesh needs rebuilding.
    pub fn update(&mut self, inp: &InputState, camera: &FlyCamera) -> bool {
        self.timestamp += 1;
        self.mesh_dirty = false;

        // ── File operations (work in any mode) ──────────────────────────
        // Ctrl+S = save, Ctrl+O = load, Ctrl+N = new map, Ctrl+E = export GLB
        if inp.ctrl && inp.just_pressed(KeyCode::KeyS) {
            self.request_save = true;
            return false;
        }
        if inp.ctrl && inp.just_pressed(KeyCode::KeyO) {
            self.request_load = true;
            return false;
        }
        if inp.ctrl && inp.just_pressed(KeyCode::KeyN) {
            self.request_newmap = true;
            return false;
        }
        if inp.ctrl && inp.shift && inp.just_pressed(KeyCode::KeyE) {
            self.request_export_fbx = true;
            return false;
        }
        if inp.ctrl && inp.just_pressed(KeyCode::KeyP) {
            self.request_package_map = true;
            return false;
        }
        if inp.ctrl && inp.just_pressed(KeyCode::KeyE) {
            self.request_export_glb = true;
            return false;
        }

        // Toggle edit mode
        if inp.just_pressed(KeyCode::KeyE) && !inp.ctrl {
            self.edit_mode = !self.edit_mode;
            if !self.edit_mode {
                self.cancel_sel();
            }
            println!("Edit mode: {}", if self.edit_mode { "ON" } else { "OFF" });
        }

        if !self.edit_mode {
            self.hover = None;
            return false;
        }

        // Update hover (raycast from camera center)
        // raycast() returns renderer Y-up coords and renderer orient;
        // convert both to Cube2 Z-up space for all editor internals.
        self.hover = self.edit_world.world.raycast(
            camera.pos.to_array(),
            camera.forward().to_array(),
        ).map(|mut h| {
            // Position: renderer (x, y_render, z_render) → Cube2 (x, z_render, y_render)
            h.origin = (h.origin.0, h.origin.2, h.origin.1);
            // Orient: already in Cube2 space — no conversion needed.

            // Normalize to grid size (like C++'s normalizelookupcube).
            // When gridsize < leaf size: snap origin to grid boundary within the leaf.
            // When gridsize > leaf size: snap origin to grid boundary (mask lower bits).
            let gs = self.grid_size;
            let lu_size = h.size;
            if lu_size > gs {
                // Snap the hit position to grid boundaries within the large cube.
                // C++: lu.x += (o.x - lu.x) / gridsize * gridsize
                // where o = actual world hit point, lu = cube origin
                // We approximate using the hit ray parameter to get the actual hit point.
                let hp = [
                    camera.pos.x + h.t * camera.forward().x,
                    camera.pos.z + h.t * camera.forward().z,
                    camera.pos.y + h.t * camera.forward().y,
                ];
                let ox = h.origin.0 as f32;
                let oy = h.origin.1 as f32;
                let oz = h.origin.2 as f32;
                let gf = gs as f32;
                h.origin.0 += (((hp[0] - ox) / gf) as i32) * gs;
                h.origin.1 += (((hp[1] - oy) / gf) as i32) * gs;
                h.origin.2 += (((hp[2] - oz) / gf) as i32) * gs;
            } else if gs > lu_size {
                // Mask to grid boundary
                let mask = !(gs - 1);
                h.origin.0 &= mask;
                h.origin.1 &= mask;
                h.origin.2 &= mask;
            }
            h.size = gs; // Always report grid_size as the cursor size
            h
        });

        // Handle selection input
        self.handle_selection(inp, camera);

        // Handle editing input — works on hover OR selection
        self.handle_editing(inp, camera);

        // Check if editing ops made the world dirty
        if self.edit_world.dirty {
            self.edit_world.dirty = false;
            self.mesh_dirty = true;
        }

        self.mesh_dirty
    }

    fn cancel_sel(&mut self) {
        self.have_sel = false;
        self.dragging = false;
    }

    // ── Selection handling ────────────────────────────────────────────────────
    // Matches rendereditcursor() from octaedit.cpp

    fn handle_selection(&mut self, inp: &InputState, camera: &FlyCamera) {
        // Space = cancel selection
        if inp.just_pressed(KeyCode::Space) {
            self.cancel_sel();
            return;
        }

        // LMB = start face drag, RMB/MMB = start corner drag
        // C++: MOUSE1 = face selection (selectcorners=0)
        //      MOUSE3 = corner selection (selectcorners=1)
        let corner_click = inp.rmb_just_pressed || inp.mmb_just_pressed;
        let start_drag = inp.lmb_just_pressed || corner_click;
        if start_drag {
            if let Some(ref h) = self.hover {
                self.dragging = true;
                self.select_corners = corner_click;

                let g = self.grid_size;
                self.drag_start = [
                    (h.origin.0 / g) * g,
                    (h.origin.1 / g) * g,
                    (h.origin.2 / g) * g,
                ];
                self.drag_start_orient = h.orient;
                self.drag_start_cor = self.compute_cor(h, camera);

                // Initialize 1x1x1 selection
                self.selection.origin = self.drag_start;
                self.selection.size = [1, 1, 1];
                self.selection.grid = g;
                self.selection.orient = h.orient;
                self.selection.corner = self.compute_corner(h, camera);
                // Full face for LMB, will be refined during drag for RMB
                self.selection.cx = 0;
                self.selection.cy = 0;
                self.selection.cxs = 2;
                self.selection.cys = 2;
                self.have_sel = true;
            }
        }

        // LMB/RMB held = drag to extend selection
        // Matches C++ rendereditcursor's dragging block (lines 452-475)
        if self.dragging && (inp.lmb || inp.rmb || inp.mmb) {
            if let Some(ref h) = self.hover {
                let g = self.grid_size;
                let end = [
                    (h.origin.0 / g) * g,
                    (h.origin.1 / g) * g,
                    (h.origin.2 / g) * g,
                ];

                // Update 3D selection box (updateselection)
                for i in 0..3 {
                    let a = self.drag_start[i].min(end[i]);
                    let b = self.drag_start[i].max(end[i]);
                    self.selection.origin[i] = a;
                    self.selection.size[i] = ((b - a) / g) + 1;
                }
                self.selection.grid = g;
                self.selection.orient = self.drag_start_orient;
                self.selection.corner = self.compute_corner(h, camera);

                // Compute sub-face selection (cx/cy/cxs/cys) from doubled-grid coords
                // C++ lines 452-475 of octaedit.cpp
                let cor = self.compute_cor(h, camera);
                let lastcor = self.drag_start_cor;
                let d = FACE_DIM[self.drag_start_orient];

                let mut cx = cor[0].min(lastcor[0]);
                let mut cy = cor[1].min(lastcor[1]);
                let mut cxs = cor[0].max(lastcor[0]);
                let mut cys = cor[1].max(lastcor[1]);

                if !self.select_corners {
                    // LMB: round to even (edge-aligned, full face per cube)
                    cx &= !1;
                    cy &= !1;
                    cxs &= !1;
                    cys &= !1;
                    cxs -= cx - 2;
                    cys -= cy - 2;
                } else {
                    // RMB: keep odd values (corner-precise)
                    cxs -= cx - 1;
                    cys -= cy - 1;
                }

                self.selection.cx = cx & 1;
                self.selection.cy = cy & 1;
                self.selection.cxs = cxs;
                self.selection.cys = cys;
            }
        }

        // Mouse released = finalize drag
        if (inp.lmb_just_released || inp.rmb_just_released || inp.mmb_just_released) && self.dragging {
            self.dragging = false;
        }

        // If no confirmed selection, create a hover-based default selection
        // (this is what C++ does in rendereditcursor when !havesel && !dragging)
        if !self.have_sel && !self.dragging {
            if let Some(ref h) = self.hover {
                let g = self.grid_size;
                self.selection.origin = [
                    (h.origin.0 / g) * g,
                    (h.origin.1 / g) * g,
                    (h.origin.2 / g) * g,
                ];
                self.selection.size = [1, 1, 1];
                self.selection.grid = g;
                self.selection.orient = h.orient;
                // Full face selection by default
                self.selection.cx = 0;
                self.selection.cy = 0;
                self.selection.cxs = 2;
                self.selection.cys = 2;
                // Compute corner from cursor position within the face
                self.selection.corner = self.compute_corner(h, camera);
            }
        }
    }

    /// Compute the doubled-grid coordinates (cor) for the hit point.
    /// Returns [cor_R, cor_C] in the face's 2D coordinate system.
    /// C++: cor = ivec(vec(w).mul(2).div(gridsize))
    fn compute_cor(&self, hit: &RayHit, camera: &FlyCamera) -> [i32; 2] {
        let hit_point = [
            camera.pos.x + hit.t * camera.forward().x,
            camera.pos.z + hit.t * camera.forward().z, // renderer z → cube2 y
            camera.pos.y + hit.t * camera.forward().y, // renderer y → cube2 z
        ];

        let g = self.grid_size as f32;
        let d = FACE_DIM[hit.orient];

        let cor_r = ((hit_point[R[d]] * 2.0) / g).floor() as i32;
        let cor_c = ((hit_point[C[d]] * 2.0) / g).floor() as i32;

        [cor_r, cor_c]
    }

    /// Compute which corner (0-3) of the face grid the cursor points at.
    fn compute_corner(&self, hit: &RayHit, camera: &FlyCamera) -> usize {
        let cor = self.compute_cor(hit, camera);
        let g = self.grid_size as f32;
        let d = FACE_DIM[hit.orient];
        let lu = [hit.origin.0 as f32, hit.origin.1 as f32, hit.origin.2 as f32];
        let base_r = ((lu[R[d]] * 2.0) / g) as i32;
        let base_c = ((lu[C[d]] * 2.0) / g) as i32;

        let cr = (cor[0] - base_r).clamp(0, 1) as usize;
        let cc = (cor[1] - base_c).clamp(0, 1) as usize;

        cr + cc * 2
    }

    // ── Editing operations ────────────────────────────────────────────────────
    // These match the stdedit.cfg universaldelta / editfacewentpush flow.

    fn handle_editing(&mut self, inp: &InputState, camera: &FlyCamera) {
        // The selection is always valid when we have a hover (even without clicking)
        if !self.selection.is_valid() { return; }

        // Grid size adjustment: G + Scroll
        if inp.held(KeyCode::KeyG) && inp.scroll_y != 0.0 {
            if inp.scroll_y > 0.0 && self.grid_power < 12 {
                self.grid_power += 1;
            } else if inp.scroll_y < 0.0 && self.grid_power > 0 {
                self.grid_power -= 1;
            }
            self.grid_size = 1 << self.grid_power;
            self.cancel_sel();
            println!("Grid: {} (2^{})", self.grid_size, self.grid_power);
            return;
        }

        // Scroll with no modifier = fill/empty (editface mode 1)
        // BUT: if select_corners is true (RMB was used), use edge push (mode 0)
        // so the sub-face cx/cy/cxs/cys selection controls which vertices move.
        //
        // This is C++'s delta_edit_0 → editfacewentpush $arg1 1
        //
        // Matching C++ mpeditface exactly:
        //   seldir = dc ? -dir : dir
        //   PRE:  if(dir<0) sel.o[d] += sel.grid * seldir   (before fill)
        //   POST: if(dir>0) sel.o[d] += sel.grid * seldir   (after fill)
        //
        // The editface() function handles the pre-advance internally (it
        // clones sel, so we don't see it). We must do the post-advance here
        // so that consecutive scrolls keep extending in the same direction.
        if inp.scroll_y != 0.0
            && !inp.held(KeyCode::KeyG)
            && !inp.held(KeyCode::KeyF)
            && !inp.held(KeyCode::KeyR)
            && !inp.held(KeyCode::Period)
            && !inp.held(KeyCode::Digit1)
            && !inp.held(KeyCode::Digit2)
            && !inp.held(KeyCode::Digit3)
            && !inp.held(KeyCode::Digit4)
            && !inp.held(KeyCode::Digit5)
        {
            let dir = if inp.scroll_y > 0.0 { 1 } else { -1 };
            let mut sel = self.active_sel();
            self.edit_world.make_undo(&sel, self.timestamp);

            // If RMB was used for selection (select_corners=true), the sub-face
            // cx/cy/cxs/cys encode which vertices are selected. Mode 1 (fill)
            // auto-downgrades to mode 0 (edge push) inside editface when it
            // detects sub-face selection. This makes scroll directly push the
            // selected vertices without needing F or Period key.
            self.edit_world.editface(&sel, dir, 1);

            // Only advance origin for true fill mode (full-face selection).
            // Edge push (sub-face) doesn't advance — it deforms in place.
            let is_subface = sel.cx != 0 || sel.cy != 0 || (sel.cxs & 1) != 0 || (sel.cys & 1) != 0;
            if !is_subface {
                // Advance selection origin for the NEXT scroll step
                let d = FACE_DIM[sel.orient];
                let dc = FACE_SIDE[sel.orient];
                let seldir = if dc != 0 { -dir } else { dir };
                sel.origin[d] += sel.grid * seldir;
            }

            self.selection = sel;
            self.have_sel = true;
            return;
        }

        // Face push/pull: F + Scroll (editface mode 0)
        // C++'s delta_edit_2 → editfacewentpush $arg1 0
        if inp.held(KeyCode::KeyF) && inp.scroll_y != 0.0 {
            let dir = if inp.scroll_y > 0.0 { 1 } else { -1 };
            let sel = self.active_sel();
            self.edit_world.make_undo(&sel, self.timestamp);
            self.edit_world.editface(&sel, dir, 0);
            self.have_sel = true; // Lock selection
            return;
        }

        // Corner push: Period + Scroll (editface mode 2)
        // C++ uses Q+Scroll (modifier 3), but Q is WASD down key.
        // Period (.) is the alternative. MMB is now used for vertex selection (= RMB).
        if inp.held(KeyCode::Period)
            && inp.scroll_y != 0.0
        {
            let dir = if inp.scroll_y > 0.0 { 1 } else { -1 };
            let sel = self.active_sel();
            self.edit_world.make_undo(&sel, self.timestamp);
            self.edit_world.editface(&sel, dir, 2);
            self.have_sel = true; // Lock selection
            return;
        }

        // Rotate: R + Scroll
        if inp.held(KeyCode::KeyR) && inp.scroll_y != 0.0 {
            let cw = if inp.scroll_y > 0.0 { 1 } else { -1 };
            let mut sel = self.active_sel();
            // C++ squares the selection BEFORE making undo, so undo captures
            // the full expanded area. We must do the same.
            let d = FACE_DIM[sel.orient];
            let ss = sel.size[C[d]].max(sel.size[R[d]]);
            sel.size[C[d]] = ss;
            sel.size[R[d]] = ss;
            self.edit_world.make_undo(&sel, self.timestamp);
            self.edit_world.rotate(cw, &mut sel);
            self.selection = sel;
            self.have_sel = true;
            return;
        }

        // ── Texture & Material keys: 1–5 + Scroll ─────────────────────────
        //   1 = cycle texture slot
        //   2 = rotate texture
        //   3 = scale texture
        //   4 = offset texture (axis 1), Shift+4 = offset (axis 2)
        //   5 = cycle material

        // 1 + Scroll: cycle texture slot
        if inp.held(KeyCode::Digit1) && inp.scroll_y != 0.0 {
            let delta = if inp.scroll_y > 0.0 { 1i16 } else { -1 };
            let sel = self.active_sel();
            let orient = sel.orient;
            let current_tex = {
                let (cube, _, _) = self.edit_world.world.lookup(
                    sel.origin[0], sel.origin[1], sel.origin[2],
                );
                cube.texture[orient]
            };
            let new_tex = (current_tex as i32 + delta as i32).max(0) as u16;
            self.edit_world.make_undo(&sel, self.timestamp);
            let all_faces = inp.ctrl;
            self.edit_world.edit_texture(&sel, new_tex, all_faces);
            self.have_sel = true;
            let num_slots = self.tex_registry.num_slots();
            let faces_str = if all_faces { " (all faces)" } else { "" };
            println!("Texture slot: {} (of {}){}", new_tex, num_slots, faces_str);
            return;
        }

        // 2 + Scroll: rotate texture
        if inp.held(KeyCode::Digit2) && inp.scroll_y != 0.0 {
            let dir = if inp.scroll_y > 0.0 { 1 } else { -1 };
            let sel = self.active_sel();
            let mut delta = VSlot::default();
            delta.rotation = dir;
            delta.changed = VSLOT_ROTATION;
            self.edit_world.make_undo(&sel, self.timestamp);
            self.tex_registry.edit_vslot_selection(
                &mut self.edit_world.world, &sel, &delta, false,
            );
            self.edit_world.dirty = true;
            self.have_sel = true;
            println!("Texture rotate: {} dir", dir);
            return;
        }

        // 3 + Scroll: scale texture (×2 or ÷2 per step)
        if inp.held(KeyCode::Digit3) && inp.scroll_y != 0.0 {
            let factor = if inp.scroll_y > 0.0 { 2.0 } else { 0.5 };
            let sel = self.active_sel();
            let mut delta = VSlot::default();
            delta.scale = factor;
            delta.changed = VSLOT_SCALE;
            self.edit_world.make_undo(&sel, self.timestamp);
            self.tex_registry.edit_vslot_selection(
                &mut self.edit_world.world, &sel, &delta, false,
            );
            self.edit_world.dirty = true;
            self.have_sel = true;
            println!("Texture scale: x{}", factor);
            return;
        }

        // 4 + Scroll: offset texture axis 1 (S), Shift+4 = axis 2 (T)
        if inp.held(KeyCode::Digit4) && inp.scroll_y != 0.0 {
            let offset_delta = if inp.scroll_y > 0.0 { 16 } else { -16 };
            let sel = self.active_sel();
            let mut delta = VSlot::default();
            if inp.shift {
                delta.offset = [0, offset_delta]; // axis 2 (T)
            } else {
                delta.offset = [offset_delta, 0]; // axis 1 (S)
            }
            delta.changed = VSLOT_OFFSET;
            self.edit_world.make_undo(&sel, self.timestamp);
            self.tex_registry.edit_vslot_selection(
                &mut self.edit_world.world, &sel, &delta, false,
            );
            self.edit_world.dirty = true;
            self.have_sel = true;
            let axis = if inp.shift { "T" } else { "S" };
            println!("Texture offset {}: {} texels", axis, offset_delta);
            return;
        }

        // 5 + Scroll: cycle material
        // Cycle through: air(0) → water(1) → lava(2) → clip(3) → glass(4) → air...
        if inp.held(KeyCode::Digit5) && inp.scroll_y != 0.0 {
            use cube_world::{MAT_AIR, MAT_WATER, MAT_LAVA, MAT_CLIP, MAT_GLASS};
            let sel = self.active_sel();
            let current_mat = {
                let (cube, _, _) = self.edit_world.world.lookup(
                    sel.origin[0], sel.origin[1], sel.origin[2],
                );
                cube.material
            };
            let mats = [MAT_AIR, MAT_WATER, MAT_LAVA, MAT_CLIP, MAT_GLASS];
            let mat_names = ["air", "water", "lava", "clip", "glass"];
            let cur_idx = mats.iter().position(|&m| m == current_mat).unwrap_or(0);
            let dir = if inp.scroll_y > 0.0 { 1i32 } else { -1 };
            let new_idx = (cur_idx as i32 + dir).rem_euclid(mats.len() as i32) as usize;
            self.edit_world.make_undo(&sel, self.timestamp);
            self.edit_world.set_material(&sel, mats[new_idx]);
            self.have_sel = true;
            println!("Material: {} ({})", mat_names[new_idx], mats[new_idx]);
            return;
        }

        // Flip: X
        if inp.just_pressed(KeyCode::KeyX) && !inp.ctrl {
            let sel = self.active_sel();
            self.edit_world.make_undo(&sel, self.timestamp);
            self.edit_world.flip(&sel);
            return;
        }

        // Delete: Delete key
        if inp.just_pressed(KeyCode::Delete) {
            let sel = self.active_sel();
            self.edit_world.make_undo(&sel, self.timestamp);
            self.edit_world.delete(&sel);
            return;
        }

        // Copy: C
        if inp.just_pressed(KeyCode::KeyC) && !inp.ctrl {
            let sel = self.active_sel();
            println!("COPY: sel valid={} origin={:?} size={:?} grid={} orient={}",
                sel.is_valid(), sel.origin, sel.size, sel.grid, sel.orient);
            self.clipboard = Some(self.edit_world.copy(&sel));
            println!("Copied {} cubes", self.clipboard.as_ref().unwrap().cubes.len());
            return;
        }

        // Paste: V
        if inp.just_pressed(KeyCode::KeyV) && !inp.ctrl {
            if let Some(ref clip) = self.clipboard {
                let sel = self.active_sel();
                println!("PASTE: {} cubes → sel origin={:?} size={:?} grid={} orient={}",
                    clip.cubes.len(), sel.origin, sel.size, sel.grid, sel.orient);
                self.edit_world.make_undo(&sel, self.timestamp);
                self.edit_world.paste(clip, &sel);
            } else {
                println!("PASTE: no clipboard!");
            }
            return;
        }

        // Undo: Z or U
        if inp.just_pressed(KeyCode::KeyZ) || inp.just_pressed(KeyCode::KeyU) {
            self.edit_world.undo(self.timestamp);
            return;
        }

        // Redo: I
        if inp.just_pressed(KeyCode::KeyI) && !inp.ctrl {
            self.edit_world.redo(self.timestamp);
            return;
        }

        // ── Entity editing ───────────────────────────────────────────────
        // P = place or move playerstart at camera position
        // N = select nearest entity (cycle through nearby)
        // Shift+P = delete selected entity

        if inp.just_pressed(KeyCode::KeyP) && !inp.ctrl && !inp.shift {
            // Camera pos is renderer Y-up; entities store Cube2 Z-up coords
            let cam_pos = [camera.pos.x, camera.pos.z, camera.pos.y]; // renderer→cube2
            let yaw_attr = (camera.yaw.to_degrees() as i16 + 360) % 360;

            if let Some(idx) = self.selected_entity {
                // Move selected entity to camera position
                if idx < self.edit_world.world.entities.len() {
                    self.edit_world.world.entities[idx].pos = cam_pos;
                    self.edit_world.world.entities[idx].attr[0] = yaw_attr as i16;
                    println!("Moved entity {} to ({:.0}, {:.0}, {:.0})",
                        idx, cam_pos[0], cam_pos[1], cam_pos[2]);
                }
            } else {
                // Place new playerstart (etype=1)
                use cube_world::octree::MapEntity;
                let ent = MapEntity {
                    pos: cam_pos,
                    etype: 1, // ET_PLAYERSTART
                    attr: [yaw_attr as i16, 0, 0, 0, 0],
                };
                self.edit_world.world.entities.push(ent);
                let idx = self.edit_world.world.entities.len() - 1;
                self.selected_entity = Some(idx);
                println!("Placed playerstart #{} at ({:.0}, {:.0}, {:.0}) yaw={}",
                    idx, cam_pos[0], cam_pos[1], cam_pos[2], yaw_attr);
            }
            return;
        }

        // Shift+P = delete selected entity
        if inp.just_pressed(KeyCode::KeyP) && inp.shift && !inp.ctrl {
            if let Some(idx) = self.selected_entity {
                if idx < self.edit_world.world.entities.len() {
                    let e = self.edit_world.world.entities.remove(idx);
                    println!("Deleted entity #{} (type={}) at ({:.0}, {:.0}, {:.0})",
                        idx, e.etype, e.pos[0], e.pos[1], e.pos[2]);
                    self.selected_entity = None;
                }
            }
            return;
        }

        // N = select nearest entity to camera
        if inp.just_pressed(KeyCode::KeyN) && !inp.ctrl {
            let cam = [camera.pos.x, camera.pos.z, camera.pos.y]; // renderer→cube2
            let ents = &self.edit_world.world.entities;
            if ents.is_empty() {
                println!("No entities in map");
                return;
            }
            // Find nearest, skipping current selection to cycle
            let skip = self.selected_entity;
            let mut best = None;
            let mut best_dist = f32::MAX;
            let mut second_best = None;
            let mut second_dist = f32::MAX;
            for (i, e) in ents.iter().enumerate() {
                let dx = e.pos[0] - cam[0];
                let dy = e.pos[1] - cam[1];
                let dz = e.pos[2] - cam[2];
                let d = dx*dx + dy*dy + dz*dz;
                if d < best_dist {
                    second_best = best;
                    second_dist = best_dist;
                    best = Some(i);
                    best_dist = d;
                } else if d < second_dist {
                    second_best = Some(i);
                    second_dist = d;
                }
            }
            let pick = if skip == best { second_best.or(best) } else { best };
            if let Some(idx) = pick {
                self.selected_entity = Some(idx);
                let e = &ents[idx];
                let type_name = match e.etype {
                    1 => "playerstart",
                    2 => "light",
                    3 => "mapmodel",
                    7 => "envmap",
                    _ => "unknown",
                };
                println!("Selected entity #{}: {} at ({:.0}, {:.0}, {:.0})",
                    idx, type_name, e.pos[0], e.pos[1], e.pos[2]);
            }
            return;
        }
    }

    /// Get the active selection for editing.
    fn active_sel(&self) -> Selection {
        self.selection.clone()
    }

    // ── Overlay geometry ──────────────────────────────────────────────────────

    /// Build line geometry for the editor overlay (crosshair, cursor box, selection).
    /// Returns (vertices, indices) for LINE_LIST topology.
    ///
    /// All positions are in renderer Y-up coordinates (swapped from Cube2 Z-up).
    pub fn build_overlay(&self) -> (Vec<Vertex>, Vec<u32>) {
        let mut verts = Vec::new();
        let mut idxs = Vec::new();

        if !self.edit_mode { return (verts, idxs); }

        let bias = 0.05f32; // small inset to prevent z-fighting

        // 1. Grid cursor box (gray) — the cube the crosshair points at
        //    Matches C++: gle::colorub(120,120,120); boxs(orient, vec(lu), vec(lusize));
        if let Some(ref h) = self.hover {
            let color = [0.47, 0.47, 0.47, 0.9]; // gray
            let s = h.size as f32;
            self.push_box_3d(
                &mut verts, &mut idxs,
                h.origin.0 as f32 - bias, h.origin.1 as f32 - bias, h.origin.2 as f32 - bias,
                s + bias * 2.0, s + bias * 2.0, s + bias * 2.0,
                color,
            );

            // Highlight the hit face in brighter white
            // Matches C++: the face-specific box drawing via boxs(orient, ...)
            let face_color = [0.9, 0.9, 0.9, 1.0];
            self.push_face_outline(
                &mut verts, &mut idxs,
                h.origin.0 as f32, h.origin.1 as f32, h.origin.2 as f32,
                h.size as f32,
                h.orient,
                face_color,
            );
        }

        // 2. Selection box (blue 3D wireframe + grid lines + origin marker)
        //    Matches C++ rendereditcursor's "if(havesel || moving)" block
        if self.have_sel || self.dragging {
            let sel = &self.selection;
            let g = sel.grid as f32;
            let sx = sel.size[0] as f32 * g;
            let sy = sel.size[1] as f32 * g;
            let sz = sel.size[2] as f32 * g;
            let ox = sel.origin[0] as f32;
            let oy = sel.origin[1] as f32;
            let oz = sel.origin[2] as f32;

            // Grid lines inside selection (C++: gle::colorub(50,50,50); boxsgrid(...))
            let grid_color = [0.2, 0.2, 0.2, 0.5];
            self.push_grid_lines(&mut verts, &mut idxs, sel, grid_color);

            // Red origin marker (C++: gle::colorub(200,0,0); boxs3D(sel.o - 0.5*sz, sz))
            let red = [0.78, 0.0, 0.0, 0.9];
            let marker_sz = (g * 0.25).min(2.0);
            self.push_box_3d(
                &mut verts, &mut idxs,
                ox - marker_sz * 0.5, oy - marker_sz * 0.5, oz - marker_sz * 0.5,
                marker_sz, marker_sz, marker_sz,
                red,
            );

            // White selection face outline (C++: gle::colorub(200,200,200); boxs(orient, co, cs))
            let white = [0.78, 0.78, 0.78, 0.9];
            self.push_selection_face(&mut verts, &mut idxs, sel, white);

            // Blue 3D bounding box (C++: gle::colorub(0,0,120); boxs3D(sel.o, sel.s, sel.grid))
            let blue = [0.0, 0.0, 0.47, 0.9];
            self.push_box_3d(&mut verts, &mut idxs,
                ox - bias, oy - bias, oz - bias,
                sx + bias * 2.0, sy + bias * 2.0, sz + bias * 2.0,
                blue);
        }

        // 3. Entity markers — show all entities as colored wireframe boxes
        //    Playerstart = green, light = yellow, mapmodel = cyan, other = magenta
        //    Selected entity gets a white highlight.
        for (i, ent) in self.edit_world.world.entities.iter().enumerate() {
            let color = match ent.etype {
                1 => [0.0, 1.0, 0.2, 0.9],   // playerstart: green
                2 => [1.0, 1.0, 0.0, 0.7],   // light: yellow
                3 => [0.0, 0.8, 0.8, 0.7],   // mapmodel: cyan
                _ => [0.7, 0.0, 0.7, 0.5],   // other: magenta
            };
            let sz = if ent.etype == 1 { 8.0 } else { 4.0 }; // playerstart bigger
            // Entity pos is Cube2 Z-up (x, y, z) → push_box_3d expects Cube2 coords
            self.push_box_3d(
                &mut verts, &mut idxs,
                ent.pos[0] - sz * 0.5, ent.pos[1] - sz * 0.5, ent.pos[2] - sz * 0.5,
                sz, sz, sz,
                color,
            );

            // Draw direction arrow for playerstart (yaw from attr[0])
            if ent.etype == 1 {
                let yaw_deg = ent.attr[0] as f32;
                let yaw_rad = yaw_deg.to_radians();
                let arrow_len = 12.0;
                // Cube2 coords: X right, Y forward, Z up
                let dx = yaw_rad.sin() * arrow_len;
                let dy = yaw_rad.cos() * arrow_len;
                let base = verts.len() as u32;
                // Swap Y↔Z for renderer (push_box_3d does this, but for lines we do it manually)
                verts.push(Vertex::new(
                    [ent.pos[0], ent.pos[2], ent.pos[1]], [0.0; 3], [0.0; 2], color,
                ));
                verts.push(Vertex::new(
                    [ent.pos[0] + dx, ent.pos[2], ent.pos[1] + dy], [0.0; 3], [0.0; 2], color,
                ));
                idxs.push(base);
                idxs.push(base + 1);
            }

            // Highlight selected entity with white outline
            if self.selected_entity == Some(i) {
                let white = [1.0, 1.0, 1.0, 1.0];
                let hs = sz + 2.0;
                self.push_box_3d(
                    &mut verts, &mut idxs,
                    ent.pos[0] - hs * 0.5, ent.pos[1] - hs * 0.5, ent.pos[2] - hs * 0.5,
                    hs, hs, hs,
                    white,
                );
            }
        }

        (verts, idxs)
    }

    // ── Line geometry helpers ─────────────────────────────────────────────────
    // All inputs in Cube2 Z-up coords, converted to renderer Y-up on output.

    /// Push a 3D wireframe box as LINE_LIST segments.
    fn push_box_3d(
        &self,
        verts: &mut Vec<Vertex>, idxs: &mut Vec<u32>,
        ox: f32, oy: f32, oz: f32,
        sx: f32, sy: f32, sz: f32,
        color: [f32; 4],
    ) {
        // 8 corners in Cube2 Z-up, converted to renderer Y-up (swap Y↔Z)
        let corners = [
            [ox,      oz,      oy     ],  // 0: (0,0,0)
            [ox + sx, oz,      oy     ],  // 1: (1,0,0)
            [ox + sx, oz,      oy + sy],  // 2: (1,1,0)
            [ox,      oz,      oy + sy],  // 3: (0,1,0)
            [ox,      oz + sz, oy     ],  // 4: (0,0,1)
            [ox + sx, oz + sz, oy     ],  // 5: (1,0,1)
            [ox + sx, oz + sz, oy + sy],  // 6: (1,1,1)
            [ox,      oz + sz, oy + sy],  // 7: (0,1,1)
        ];

        let base = verts.len() as u32;
        for c in &corners {
            verts.push(Vertex::new(*c, [0.0; 3], [0.0; 2], color));
        }

        let edges: [(u32, u32); 12] = [
            (0,1), (1,2), (2,3), (3,0), // bottom
            (4,5), (5,6), (6,7), (7,4), // top
            (0,4), (1,5), (2,6), (3,7), // verticals
        ];
        for (a, b) in &edges {
            idxs.push(base + a);
            idxs.push(base + b);
        }
    }

    /// Push a single face outline (4 edges) for the given orient.
    fn push_face_outline(
        &self,
        verts: &mut Vec<Vertex>, idxs: &mut Vec<u32>,
        ox: f32, oy: f32, oz: f32,
        size: f32,
        orient: usize,
        color: [f32; 4],
    ) {
        let corners = face_corners_c2(ox, oy, oz, size, size, size, orient);
        let base = verts.len() as u32;
        for c in &corners {
            let pos = [c[0], c[2], c[1]]; // Y↔Z swap
            verts.push(Vertex::new(pos, [0.0; 3], [0.0; 2], color));
        }
        idxs.push(base); idxs.push(base + 1);
        idxs.push(base + 1); idxs.push(base + 2);
        idxs.push(base + 2); idxs.push(base + 3);
        idxs.push(base + 3); idxs.push(base);
    }

    /// Push the active face outline of the selection, respecting cx/cy/cxs/cys.
    /// Matches C++ lines 530-536 of octaedit.cpp:
    ///   co[R[d]] += 0.5*(sel.cx*gridsize);
    ///   co[C[d]] += 0.5*(sel.cy*gridsize);
    ///   cs[R[d]]  = 0.5*(sel.cxs*gridsize);
    ///   cs[C[d]]  = 0.5*(sel.cys*gridsize);
    ///   cs[D[d]] *= gridsize;
    fn push_selection_face(
        &self,
        verts: &mut Vec<Vertex>, idxs: &mut Vec<u32>,
        sel: &Selection,
        color: [f32; 4],
    ) {
        let g = sel.grid as f32;
        let d = FACE_DIM[sel.orient];

        // Start with full selection box
        let mut co = [sel.origin[0] as f32, sel.origin[1] as f32, sel.origin[2] as f32];
        let mut cs = [sel.size[0] as f32 * g, sel.size[1] as f32 * g, sel.size[2] as f32 * g];

        // Apply sub-face selection offset and size
        co[R[d]] += 0.5 * (sel.cx as f32 * g);
        co[C[d]] += 0.5 * (sel.cy as f32 * g);
        cs[R[d]]  = 0.5 * (sel.cxs as f32 * g);
        cs[C[d]]  = 0.5 * (sel.cys as f32 * g);

        let corners = face_corners_c2(co[0], co[1], co[2], cs[0], cs[1], cs[2], sel.orient);
        let base = verts.len() as u32;
        for c in &corners {
            let pos = [c[0], c[2], c[1]]; // Y↔Z swap
            verts.push(Vertex::new(pos, [0.0; 3], [0.0; 2], color));
        }
        idxs.push(base); idxs.push(base + 1);
        idxs.push(base + 1); idxs.push(base + 2);
        idxs.push(base + 2); idxs.push(base + 3);
        idxs.push(base + 3); idxs.push(base);
    }

    /// Push grid lines inside the selection on the active face.
    fn push_grid_lines(
        &self,
        verts: &mut Vec<Vertex>, idxs: &mut Vec<u32>,
        sel: &Selection,
        color: [f32; 4],
    ) {
        let g = sel.grid as f32;
        let ox = sel.origin[0] as f32;
        let oy = sel.origin[1] as f32;
        let oz = sel.origin[2] as f32;
        let sx = sel.size[0] as f32 * g;
        let sy = sel.size[1] as f32 * g;
        let sz = sel.size[2] as f32 * g;

        let d = FACE_DIM[sel.orient];
        let dc = FACE_SIDE[sel.orient];

        let face_pos = if dc != 0 {
            [ox, oy, oz][d] + [sx, sy, sz][d]
        } else {
            [ox, oy, oz][d]
        };

        let r_axis = R[d];
        let c_axis = C[d];
        let r_size = sel.size[r_axis];
        let c_size = sel.size[c_axis];
        let r_origin = [ox, oy, oz][r_axis];
        let c_origin = [ox, oy, oz][c_axis];

        // Lines along R axis (varying C)
        for i in 0..=c_size {
            let c_pos = c_origin + i as f32 * g;
            let mut p0 = [0.0f32; 3];
            let mut p1 = [0.0f32; 3];
            p0[d] = face_pos;     p1[d] = face_pos;
            p0[r_axis] = r_origin; p1[r_axis] = r_origin + r_size as f32 * g;
            p0[c_axis] = c_pos;    p1[c_axis] = c_pos;

            let v0 = [p0[0], p0[2], p0[1]]; // Y↔Z swap
            let v1 = [p1[0], p1[2], p1[1]];
            let base = verts.len() as u32;
            verts.push(Vertex::new(v0, [0.0; 3], [0.0; 2], color));
            verts.push(Vertex::new(v1, [0.0; 3], [0.0; 2], color));
            idxs.push(base);
            idxs.push(base + 1);
        }

        // Lines along C axis (varying R)
        for i in 0..=r_size {
            let r_pos = r_origin + i as f32 * g;
            let mut p0 = [0.0f32; 3];
            let mut p1 = [0.0f32; 3];
            p0[d] = face_pos;     p1[d] = face_pos;
            p0[r_axis] = r_pos;    p1[r_axis] = r_pos;
            p0[c_axis] = c_origin; p1[c_axis] = c_origin + c_size as f32 * g;

            let v0 = [p0[0], p0[2], p0[1]];
            let v1 = [p1[0], p1[2], p1[1]];
            let base = verts.len() as u32;
            verts.push(Vertex::new(v0, [0.0; 3], [0.0; 2], color));
            verts.push(Vertex::new(v1, [0.0; 3], [0.0; 2], color));
            idxs.push(base);
            idxs.push(base + 1);
        }
    }
}

// ── Face geometry helpers ─────────────────────────────────────────────────────

/// Return the 4 corners of a box face in Cube2 Z-up coordinates.
fn face_corners_c2(
    ox: f32, oy: f32, oz: f32,
    sx: f32, sy: f32, sz: f32,
    orient: usize,
) -> [[f32; 3]; 4] {
    match orient {
        O_LEFT => [  // -X face
            [ox, oy,      oz],
            [ox, oy + sy, oz],
            [ox, oy + sy, oz + sz],
            [ox, oy,      oz + sz],
        ],
        O_RIGHT => [ // +X face
            [ox + sx, oy,      oz],
            [ox + sx, oy,      oz + sz],
            [ox + sx, oy + sy, oz + sz],
            [ox + sx, oy + sy, oz],
        ],
        O_BACK => [  // -Y face
            [ox,      oy, oz],
            [ox,      oy, oz + sz],
            [ox + sx, oy, oz + sz],
            [ox + sx, oy, oz],
        ],
        O_FRONT => [ // +Y face
            [ox,      oy + sy, oz],
            [ox + sx, oy + sy, oz],
            [ox + sx, oy + sy, oz + sz],
            [ox,      oy + sy, oz + sz],
        ],
        O_BOTTOM => [ // -Z face
            [ox,      oy,      oz],
            [ox + sx, oy,      oz],
            [ox + sx, oy + sy, oz],
            [ox,      oy + sy, oz],
        ],
        O_TOP => [   // +Z face
            [ox,      oy,      oz + sz],
            [ox,      oy + sy, oz + sz],
            [ox + sx, oy + sy, oz + sz],
            [ox + sx, oy,      oz + sz],
        ],
        _ => [[0.0; 3]; 4],
    }
}
