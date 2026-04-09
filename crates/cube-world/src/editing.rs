//! In-world selection and editing operations.
//!
//! Porting the Cube2/Sauerbraten editing system to Rust, layer by layer.
//! See octaedit.cpp for the original C++ implementation.

use crate::octree::{
    Cube, OctreeWorld, MAT_AIR, F_EMPTY,
    FACE_DIM, FACE_SIDE, D, C, R,
    edge_get, edge_set, edge_idx,
    copycube, pastecube,
};

// ── Selection ──────────────────────────────────────────────────────────────────

/// A box selection in world coordinates, aligned to a grid.
#[derive(Debug, Clone)]
pub struct Selection {
    /// Grid-aligned origin of the selection.
    pub origin: [i32; 3],
    /// Size in grid cells (number of cubes in each axis).
    pub size:   [i32; 3],
    /// Grid cell size (must be a power of 2).
    pub grid:   i32,
    /// Which face of the selection is "active" for face ops.
    pub orient: usize,
    /// Corner selection within the face (for corner push/pull).
    pub corner: usize,
    /// Sub-face selection in doubled-grid coordinates (0 or 1).
    /// cx/cy: start offset (0=left/top edge, 1=right/bottom edge)
    /// cxs/cys: size (1=half-face, 2=full-face)
    pub cx: i32,
    pub cy: i32,
    pub cxs: i32,
    pub cys: i32,
}

impl Default for Selection {
    fn default() -> Self {
        Self {
            origin: [0; 3], size: [0; 3], grid: 0, orient: 0, corner: 0,
            cx: 0, cy: 0, cxs: 2, cys: 2, // full face by default
        }
    }
}

impl Selection {
    pub fn is_valid(&self) -> bool {
        self.grid > 0
            && self.size[0] > 0
            && self.size[1] > 0
            && self.size[2] > 0
    }
}

// ── Block operations ──────────────────────────────────────────────────────────

/// Compute world coordinates from selection-oriented (x, y, z) indices.
/// Port of Cube2 `blockcube()` from octaedit.cpp:267-274.
///
/// `x`/`y` iterate across the face (R/C axes), `z` iterates along the
/// depth (D axis).  The `orient` determines which face and direction.
fn blockcube_coords(sel: &Selection, x: i32, y: i32, z: i32) -> (i32, i32, i32) {
    let dim = FACE_DIM[sel.orient];
    let dc = FACE_SIDE[sel.orient];
    let mut s = [0i32; 3];
    s[R[dim]] = x * sel.grid;
    s[C[dim]] = y * sel.grid;
    s[D[dim]] = dc as i32 * (sel.size[dim] - 1) * sel.grid;
    s[0] += sel.origin[0];
    s[1] += sel.origin[1];
    s[2] += sel.origin[2];
    if dc != 0 {
        s[dim] -= z * sel.grid;
    } else {
        s[dim] += z * sel.grid;
    }
    (s[0], s[1], s[2])
}

/// A block of cubes copied from the world, with the selection they came from.
/// Rust equivalent of Cube2 `block3` + inline cube array.
///
/// Cubes are stored in the same order as `loopxyz`: z (depth) outermost,
/// then y (C axis), then x (R axis) innermost.
#[derive(Clone)]
pub struct CubeBlock {
    pub sel: Selection,
    pub cubes: Vec<Cube>,
}

impl CubeBlock {
    pub fn size(&self) -> usize {
        (self.sel.size[0] * self.sel.size[1] * self.sel.size[2]) as usize
    }
}

// ── Undo / Redo ────────────────────────────────────────────────────────────────

/// A snapshot of a selection's cube data for undo/redo.
/// Stores the block of cubes plus a gridmap recording the actual leaf size
/// at each cell (needed because `blockcopy` with negative rgrid doesn't subdivide).
///
/// Port of Cube2 `undoblock` from octa.h:232-244.
#[derive(Clone)]
pub struct UndoBlock {
    pub block: CubeBlock,
    /// Log2 of the actual leaf size at each cell position.
    /// Length == block.size().
    pub gridmap: Vec<u8>,
    pub timestamp: u64,
}

pub struct UndoStack {
    pub(crate) undo: Vec<UndoBlock>,
    pub(crate) redo: Vec<UndoBlock>,
    /// Maximum total memory (in estimated cube count) for undo history.
    max_cubes: usize,
    pub(crate) total_cubes: usize,
}

impl UndoStack {
    pub fn new() -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
            max_cubes: 1 << 18, // ~256K cubes
            total_cubes: 0,
        }
    }

    pub fn can_undo(&self) -> bool { !self.undo.is_empty() }
    pub fn can_redo(&self) -> bool { !self.redo.is_empty() }

    pub fn add_undo(&mut self, block: UndoBlock) {
        let size = count_block_cubes(&block);
        self.total_cubes += size;
        self.undo.push(block);
        self.prune();
    }

    fn prune(&mut self) {
        // Clear all redo when a new undo is pushed
        for r in self.redo.drain(..) {
            self.total_cubes = self.total_cubes.saturating_sub(count_block_cubes(&r));
        }
        // Prune oldest undos if over budget
        while self.total_cubes > self.max_cubes && !self.undo.is_empty() {
            let u = self.undo.remove(0);
            self.total_cubes = self.total_cubes.saturating_sub(count_block_cubes(&u));
        }
    }

    pub fn pop_undo(&mut self) -> Option<UndoBlock> {
        let u = self.undo.pop()?;
        self.total_cubes = self.total_cubes.saturating_sub(count_block_cubes(&u));
        Some(u)
    }

    pub fn pop_redo(&mut self) -> Option<UndoBlock> {
        let u = self.redo.pop()?;
        self.total_cubes = self.total_cubes.saturating_sub(count_block_cubes(&u));
        Some(u)
    }

    pub fn push_redo(&mut self, block: UndoBlock) {
        let size = count_block_cubes(&block);
        self.total_cubes += size;
        self.redo.push(block);
    }
}

impl Default for UndoStack {
    fn default() -> Self { Self::new() }
}

/// Log2 of a value (position of lowest set bit). Equivalent to C++ `bitscan`.
fn bitscan(v: u32) -> u8 {
    if v == 0 { return 0; }
    v.trailing_zeros() as u8
}

/// Count leaf cubes in a block (recursively counting children).
fn count_block_cubes(ub: &UndoBlock) -> usize {
    fn family_size(c: &Cube) -> usize {
        match &c.children {
            None => 1,
            Some(ch) => ch.iter().map(family_size).sum::<usize>(),
        }
    }
    ub.block.cubes.iter().map(|c| family_size(c)).sum()
}

// ── Face-flip helpers ─────────────────────────────────────────────────────────
// Port of Cube2 dflip/cflip/rflip/mflip from octaedit.cpp:2571-2574.
// These operate on a face as u32 (4 edge bytes packed).

/// Flip a face along its depth axis: swap low/high nibbles and complement.
fn dflip(face: u32) -> u32 {
    if face == F_EMPTY { return face; }
    0x88888888u32.wrapping_sub(((face & 0xF0F0F0F0) >> 4) | ((face & 0x0F0F0F0F) << 4))
}

/// Flip a face along the C axis: swap bytes within u16 halves.
fn cflip(face: u32) -> u32 {
    ((face & 0xFF00FF00) >> 8) | ((face & 0x00FF00FF) << 8)
}

/// Flip a face along the R axis: swap the two u16 halves.
fn rflip(face: u32) -> u32 {
    ((face & 0xFFFF0000) >> 16) | ((face & 0x0000FFFF) << 16)
}

/// Mirror a face: swap the two middle bytes.
fn mflip(face: u32) -> u32 {
    (face & 0xFF0000FF) | ((face & 0x00FF0000) >> 8) | ((face & 0x0000FF00) << 8)
}

/// Flip a cube along dimension `d`. Recursively flips children.
/// Port of Cube2 `flipcube(c, d)` from octaedit.cpp:2576-2587.
pub fn flipcube(c: &mut Cube, d: usize) {
    // Swap textures on the two faces of dimension d
    c.texture.swap(d * 2, d * 2 + 1);
    // Flip the three face u32s
    let fd = c.face(D[d]); c.set_face(D[d], dflip(fd));
    let fc = c.face(C[d]); c.set_face(C[d], cflip(fc));
    let fr = c.face(R[d]); c.set_face(R[d], rflip(fr));
    if let Some(children) = &mut c.children {
        let dim_bit = 1 << d; // octadim(d)
        // Swap children across the dimension
        for i in 0..8 {
            if i & dim_bit != 0 {
                children.swap(i, i ^ dim_bit);
            }
        }
        // Recursively flip all children
        for i in 0..8 {
            flipcube(&mut children[i], d);
        }
    }
}

/// Rotate a cube clockwise around dimension `d`.
/// Port of Cube2 `rotatecube(c, d)` from octaedit.cpp:2594-2618.
pub fn rotatecube(c: &mut Cube, d: usize) {
    // Rotate the three face u32s
    let fd = c.face(D[d]); c.set_face(D[d], cflip(mflip(fd)));
    let fc = c.face(C[d]); c.set_face(C[d], dflip(mflip(fc)));
    let fr = c.face(R[d]); c.set_face(R[d], rflip(mflip(fr)));
    // Swap R and C faces
    let r_face = c.face(R[d]);
    let c_face = c.face(C[d]);
    c.set_face(R[d], c_face);
    c.set_face(C[d], r_face);

    // Rotate textures: 4-way cycle
    let t = c.texture[2 * R[d]];
    c.texture[2 * R[d]] = c.texture[2 * C[d] + 1];
    c.texture[2 * C[d] + 1] = c.texture[2 * R[d] + 1];
    c.texture[2 * R[d] + 1] = c.texture[2 * C[d]];
    c.texture[2 * C[d]] = t;

    if let Some(children) = &mut c.children {
        let row = 1usize << R[d]; // octadim(R[d])
        let col = 1usize << C[d]; // octadim(C[d])
        let dep = 1usize << D[d]; // octadim(D[d])
        // Rotate child quads at each depth layer
        for i in (0..=dep).step_by(dep.max(1)) {
            // rotatequad: a←b, b←c, c←d, d←a
            let (ia, ib, ic, id) = (i + row, i, i + col, i + col + row);
            let t = std::mem::replace(&mut children[ia], Cube::empty());
            children[ia] = std::mem::replace(&mut children[ib], Cube::empty());
            children[ib] = std::mem::replace(&mut children[ic], Cube::empty());
            children[ic] = std::mem::replace(&mut children[id], Cube::empty());
            children[id] = t;
        }
        // Recursively rotate all children
        for i in 0..8 {
            rotatecube(&mut children[i], d);
        }
    }
}

// ── Texture helpers ───────────────────────────────────────────────────────────

/// Set texture on a cube (recursively on children).
/// Port of Cube2 `edittexcube(c, tex, orient, findrep)`.
/// `orient < 0` means all faces.
fn edittexcube(c: &mut Cube, tex: u16, orient: i32) {
    if orient < 0 {
        c.texture = [tex; 6];
    } else {
        let i = c.visible_orient(orient as usize);
        c.texture[i] = tex;
    }
    if let Some(children) = &mut c.children {
        for i in 0..8 {
            edittexcube(&mut children[i], tex, orient);
        }
    }
}

/// Recursively resolve the material of a cube.
/// If the cube has children, returns the common material if all children agree,
/// otherwise MAT_AIR. Port of C++ `getmaterial(cube &c)` from octaedit.cpp:1946.
fn get_material(c: &Cube) -> u8 {
    if let Some(ref children) = c.children {
        let mat = get_material(&children[7]);
        for i in 0..7 {
            if mat != get_material(&children[i]) {
                return MAT_AIR;
            }
        }
        mat
    } else {
        c.material
    }
}

/// Replace a texture on a cube (recursively on children).
/// Port of Cube2 `replacetexcube(c, oldtex, newtex)`.
fn replacetexcube(c: &mut Cube, old_tex: u16, new_tex: u16) {
    for i in 0..6 {
        if c.texture[i] == old_tex {
            c.texture[i] = new_tex;
        }
    }
    if let Some(children) = &mut c.children {
        for i in 0..8 {
            replacetexcube(&mut children[i], old_tex, new_tex);
        }
    }
}

// ── Material helpers ──────────────────────────────────────────────────────────

/// Geometry filter constants for setmat.
pub const EDITMATF_EMPTY: i32 = 1;
pub const EDITMATF_NOTEMPTY: i32 = 2;
pub const EDITMATF_SOLID: i32 = 3;
pub const EDITMATF_NOTSOLID: i32 = 4;

/// Set material on a cube with filtering. Recurses into children.
/// Port of Cube2 `setmat(c, mat, matmask, filtermat, filtermask, filtergeom)`.
fn setmat_cube(
    c: &mut Cube,
    mat: u16, mat_mask: u16,
    filter_mat: u16, filter_mask: u16,
    filter_geom: i32,
) {
    if let Some(children) = &mut c.children {
        for i in 0..8 {
            setmat_cube(&mut children[i], mat, mat_mask, filter_mat, filter_mask, filter_geom);
        }
    } else if (c.material as u16 & filter_mask) == filter_mat {
        let pass = match filter_geom {
            EDITMATF_EMPTY => c.is_empty(),
            EDITMATF_NOTEMPTY => !c.is_empty(),
            EDITMATF_SOLID => c.is_solid(),
            EDITMATF_NOTSOLID => !c.is_solid(),
            _ => true,
        };
        if !pass { return; }
        if mat != MAT_AIR as u16 {
            c.material = ((c.material as u16 & mat_mask) | mat) as u8;
        } else {
            c.material = MAT_AIR;
        }
    }
}

// ── EditWorld ─────────────────────────────────────────────────────────────────

/// Wraps OctreeWorld with edit operations and undo/redo.
/// All edit ops set `dirty = true`; caller must rebuild mesh.
pub struct EditWorld {
    pub world: OctreeWorld,
    pub undo:  UndoStack,
    pub dirty: bool,
}

impl EditWorld {
    pub fn new(world: OctreeWorld) -> Self {
        Self { world, undo: UndoStack::new(), dirty: false }
    }

    // ── Basic edit ops ───────────────────────────────────────────────────────

    /// Fill the selected cubes with solid geometry.
    /// Port of Cube2 mode=1,dir=1 in `mpeditface`.
    pub fn fill(&mut self, sel: &Selection) {
        self.apply(sel, |cube| {
            cube.solidfaces();
            cube.children = None;
        });
    }

    /// Delete (empty) the selected cubes.
    /// Port of Cube2 `mpdelcube`.
    pub fn delete(&mut self, sel: &Selection) {
        self.apply(sel, |cube| {
            cube.emptyfaces();
            cube.children = None;
        });
    }

    // ── Material ops — port of Cube2 setmat / mpeditmat ─────────────────────

    /// Set material on selected cubes (simple version).
    pub fn set_material(&mut self, sel: &Selection, mat: u8) {
        self.apply(sel, |cube| cube.material = mat);
    }

    /// Set material with filtering.
    /// Port of Cube2 `setmat(c, mat, matmask, filtermat, filtermask, filtergeom)`.
    ///
    /// `filter_geom`: 0=any, 1=empty, 2=notempty, 3=solid, 4=notsolid.
    pub fn set_material_filtered(
        &mut self, sel: &Selection,
        mat: u16, mat_mask: u16,
        filter_mat: u16, filter_mask: u16,
        filter_geom: i32,
    ) {
        self.apply(sel, |cube| {
            setmat_cube(cube, mat, mat_mask, filter_mat, filter_mask, filter_geom);
        });
    }

    // ── Texture ops — port of Cube2 edittexcube / mpedittex ───────────────

    /// Set texture on selected cubes (all faces or just active face).
    /// Simple version — sets texture directly.
    pub fn set_texture(&mut self, sel: &Selection, tex: u16, all_faces: bool) {
        let orient = sel.orient;
        self.apply(sel, |cube| {
            if all_faces {
                cube.texture = [tex; 6];
            } else {
                cube.texture[orient] = tex;
            }
        });
    }

    /// Set texture with visibleorient redirect.
    /// Port of Cube2 `edittexcube(c, tex, orient, findrep)`.
    pub fn edit_texture(&mut self, sel: &Selection, tex: u16, all_faces: bool) {
        let orient = if all_faces { -1i32 } else { sel.orient as i32 };
        self.apply(sel, |cube| {
            edittexcube(cube, tex, orient);
        });
    }

    /// Replace one texture with another in the selection.
    /// Port of Cube2 `replacetexcube`.
    pub fn replace_texture(&mut self, sel: &Selection, old_tex: u16, new_tex: u16) {
        self.apply(sel, |cube| {
            replacetexcube(cube, old_tex, new_tex);
        });
    }

    /// Replace one texture with another in the entire world.
    pub fn replace_texture_world(&mut self, old_tex: u16, new_tex: u16) {
        for i in 0..8 {
            replacetexcube(&mut self.world.root[i], old_tex, new_tex);
        }
        self.dirty = true;
    }

    // ── Core geometry editing — ports of Cube2 mpeditface ──────────────────

    /// Push/pull an edge endpoint, clamping to [0,8] and enforcing start<=end.
    /// Port of Cube2 `pushedge(uchar &edge, int dir, int dc)`.
    fn pushedge(edge: &mut u8, dir: i32, dc: usize) {
        let ne = (edge_get(*edge, dc) as i32 + dir).clamp(0, 8) as u8;
        edge_set(edge, dc, ne);
        let oe = edge_get(*edge, 1 - dc);
        if (dir < 0 && dc != 0 && oe > ne) || (dir > 0 && dc == 0 && oe < ne) {
            edge_set(edge, 1 - dc, ne);
        }
    }

    /// Push all edges that share the same vertex position as (d, x, y, dc).
    /// Port of Cube2 `linkedpush(cube &c, int d, int x, int y, int dc, int dir)`.
    fn linkedpush(cube: &mut Cube, d: usize, x: usize, y: usize, dc: usize, dir: i32) {
        let v = cube.get_cube_vector(d, x, y, dc);
        for i in 0..2 {
            for j in 0..2 {
                let p = cube.get_cube_vector(d, i, j, dc);
                if v == p {
                    Self::pushedge(&mut cube.edges[edge_idx(d, i, j)], dir, dc);
                }
            }
        }
    }

    /// Main face editing operation. Faithful port of Cube2 `mpeditface()`.
    ///
    /// - `dir`: +1 or -1 (push direction)
    /// - `mode`: 0 = push/pull edges, 1 = fill/empty, 2 = corner push
    pub fn editface(&mut self, sel: &Selection, dir: i32, mode: i32) {
        if !sel.is_valid() { return; }

        // C++ line 1961: if sub-face selection active, downgrade fill to edge push
        let mut mode = mode;
        if mode == 1 && (sel.cx != 0 || sel.cy != 0 || (sel.cxs & 1) != 0 || (sel.cys & 1) != 0) {
            mode = 0;
        }

        let d = FACE_DIM[sel.orient];
        let dc = FACE_SIDE[sel.orient];
        let seldir = if dc != 0 { -dir } else { dir };

        // For mode 1 (fill/empty), adjust selection origin
        let mut sel = sel.clone();
        if mode == 1 {
            let h = sel.origin[d] + (dc as i32) * sel.grid;
            let ws = self.world.world_size();
            if (dir > 0) == (dc != 0) && h <= 0 { return; }
            if (dir < 0) == (dc != 0) && h >= ws { return; }
            if dir < 0 { sel.origin[d] += sel.grid * seldir; }
        }

        // Flatten selection to 1 cube thick along the face dimension
        if dc != 0 {
            sel.origin[d] += sel.size[d] * sel.grid - sel.grid;
        }
        sel.size[d] = 1;

        // We need sel's sub-face fields for edge filtering in mode 0
        let sel_cx = sel.cx;
        let sel_cy = sel.cy;
        let sel_cxs = sel.cxs;
        let sel_cys = sel.cys;
        let sel_corner = sel.corner;
        let sel_sr = sel.size[R[d]];
        let sel_sc = sel.size[C[d]];

        if mode == 1 {
            // Fill/empty mode — needs adjacent cube lookup for texture copy.
            // C++ mpeditface lines 1984-1994: when filling (dir<0), copies all 6
            // textures from the adjacent cube at depth=1 (the cube "behind" the fill).
            let dim = FACE_DIM[sel.orient];
            let sr = sel.size[R[dim]];
            let sc = sel.size[C[dim]];

            // First, collect adjacent cube textures if filling
            let mut adj_textures: Vec<[u16; 6]> = Vec::new();
            if dir < 0 {
                for y in 0..sc {
                    for x in 0..sr {
                        let (wx, wy, wz) = blockcube_coords(&sel, x, y, 1);
                        let (adj, _, _) = self.world.lookup(wx, wy, wz);
                        let tex = if adj.children.is_some() {
                            [crate::octree::DEFAULT_GEOM; 6]
                        } else {
                            adj.texture
                        };
                        adj_textures.push(tex);
                    }
                }
            }

            // Now apply fill/empty
            let mut idx = 0usize;
            self.apply_with_pos(&sel, |cube, _pos_r, _pos_c| {
                let mat = get_material(cube);
                if cube.children.is_some() { cube.solidfaces(); }
                crate::octree::discard_children(cube);
                cube.material = mat;
                if dir < 0 {
                    cube.solidfaces();
                    // Copy textures from adjacent cube (C++ lines 1988-1990)
                    if idx < adj_textures.len() {
                        cube.texture = adj_textures[idx];
                    }
                } else {
                    cube.emptyfaces();
                }
                idx += 1;
            });
        } else {
            // Edge push (mode 0) or corner push (mode 2) — needs position info
            self.apply_with_pos(&sel, |cube, pos_r, pos_c| {
                let mat = get_material(cube);
                if cube.children.is_some() { cube.solidfaces(); }
                crate::octree::discard_children(cube);
                cube.material = mat;

                let bak = cube.face(d);

                if mode == 2 {
                    // Corner push mode — push the exact corner pointed at
                    let cx = sel_corner & 1;
                    let cy = sel_corner >> 1;
                    Self::linkedpush(cube, d, cx, cy, dc, seldir);
                } else {
                    // Edge push/pull mode (mode 0)
                    // C++ loopselxyz gives x = position along R[d], y = position along C[d].
                    // Edge filtering skips OUTER edges at selection boundaries when
                    // cx/cy indicate a sub-face offset, leaving only interior vertices.
                    //
                    // C++ conditions (octaedit.cpp mpeditface, mode==0):
                    //   if(x==0 && mx==0 && sel.cx) continue;
                    //   if(y==0 && my==0 && sel.cy) continue;
                    //   if(x==sel.s[R[d]]-1 && mx==1 && (sel.cx+sel.cxs)&1) continue;
                    //   if(y==sel.s[C[d]]-1 && my==1 && (sel.cy+sel.cys)&1) continue;
                    for mx in 0..2usize {
                        for my in 0..2usize {
                            // Skip outer edges at selection boundaries per sub-face selection:
                            // - First cube (pos==0), low edge (m==0): skip if cx/cy offset
                            // - Last cube (pos==max), high edge (m==1): skip if sub-face ends mid-cube
                            if mx == 0 && pos_r == 0 && sel_cx != 0 { continue; }
                            if my == 0 && pos_c == 0 && sel_cy != 0 { continue; }
                            if mx == 1 && pos_r == sel_sr - 1 && (sel_cx + sel_cxs) & 1 != 0 { continue; }
                            if my == 1 && pos_c == sel_sc - 1 && (sel_cy + sel_cys) & 1 != 0 { continue; }

                            // Only push edges that haven't been modified yet
                            let edge_byte = cube.edges[edge_idx(d, mx, my)];
                            let bak_byte = bak.to_le_bytes()[mx + my * 2];
                            if edge_byte != bak_byte { continue; }
                            Self::linkedpush(cube, d, mx, my, dc, seldir);
                        }
                    }
                }

                // Optimize: if face collapsed, make cube empty
                cube.optiface(d);

                // Validate: if cube became invalid, revert
                if !crate::octree::is_valid_cube(cube) {
                    let new_face = cube.face(d);
                    let old_bytes = bak.to_le_bytes();
                    let new_bytes = new_face.to_le_bytes();

                    // Try partial edits — accept each changed edge individually
                    for k in 0..4 {
                        if new_bytes[k] != old_bytes[k] {
                            cube.set_face(d, bak);
                            cube.edges[d * 4 + k] = new_bytes[k];
                            if !crate::octree::is_valid_cube(cube) {
                                cube.edges[d * 4 + k] = old_bytes[k];
                            }
                        }
                    }
                    // If still invalid after partial, full revert
                    if !crate::octree::is_valid_cube(cube) {
                        cube.set_face(d, bak);
                    }
                }
            });
        }
    }

    // ── Block operations ────────────────────────────────────────────────────

    /// Copy cubes from the world into a `CubeBlock`.
    /// Port of Cube2 `blockcopy(s, rgrid, b)` from octaedit.cpp:639-644.
    ///
    /// Uses `lookup_at(sel.grid)` to stop at the grid level, preserving any
    /// children (finer detail) within each grid cell. This matches C++'s
    /// `lookupcube(pos, sel.grid)` which returns the cube AT the grid level
    /// with its full subtree intact.
    pub fn blockcopy(&self, sel: &Selection) -> CubeBlock {
        let dim = FACE_DIM[sel.orient];
        let dz = sel.size[D[dim]];
        let dy = sel.size[C[dim]];
        let dx = sel.size[R[dim]];

        let mut cubes = Vec::with_capacity((dx * dy * dz) as usize);
        for z in 0..dz {
            for y in 0..dy {
                for x in 0..dx {
                    let (wx, wy, wz) = blockcube_coords(sel, x, y, z);
                    // Stop at sel.grid level — preserves children subtrees
                    let (cube, _origin, _size) = self.world.lookup_at(wx, wy, wz, sel.grid);
                    cubes.push(copycube(cube));
                }
            }
        }

        CubeBlock { sel: sel.clone(), cubes }
    }

    /// Paste cubes from a `CubeBlock` back into the world.
    /// Port of Cube2 `pasteblock(b, sel)` from octaedit.cpp:1247-1253.
    ///
    /// Only pastes non-empty cubes (transparent paste — empty block cells
    /// don't overwrite the world).
    pub fn pasteblock(&mut self, block: &CubeBlock, sel: &Selection) {
        let dim = FACE_DIM[sel.orient];
        let dz = sel.size[D[dim]];
        let dy = sel.size[C[dim]];
        let dx = sel.size[R[dim]];

        let mut idx = 0usize;
        for z in 0..dz {
            for y in 0..dy {
                for x in 0..dx {
                    if idx < block.cubes.len() {
                        let src = &block.cubes[idx];
                        if !src.is_empty() || src.children.is_some() || src.material != MAT_AIR {
                            let (wx, wy, wz) = blockcube_coords(sel, x, y, z);
                            self.world.subdivide_to(wx, wy, wz, sel.grid);
                            let (dst, _origin, _size) = self.world.lookup_mut(wx, wy, wz, sel.grid);
                            pastecube(src, dst);
                        }
                    }
                    idx += 1;
                }
            }
        }
        self.dirty = true;
    }

    /// Generate a gridmap recording the log2 of the actual leaf size at each
    /// cell of the selection.
    /// Port of Cube2 `selgridmap(sel, g)` from octaedit.cpp:662-665.
    // ── Copy / Paste ─────────────────────────────────────────────────────────

    /// Copy the current selection to a clipboard block.
    /// Port of Cube2 `mpcopy(e, sel, local)` from octaedit.cpp:1500-1508.
    pub fn copy(&self, sel: &Selection) -> CubeBlock {
        self.blockcopy(sel)
    }

    /// Paste a clipboard block into the world at the given selection.
    /// Port of Cube2 `mppaste(e, sel, local)` from octaedit.cpp:1510-1515.
    pub fn paste(&mut self, clipboard: &CubeBlock, sel: &Selection) {
        self.pasteblock(clipboard, sel);
    }

    // ── Grid helpers ────────────────────────────────────────────────────────

    pub fn selgridmap(&self, sel: &Selection) -> Vec<u8> {
        let dim = FACE_DIM[sel.orient];
        let dz = sel.size[D[dim]];
        let dy = sel.size[C[dim]];
        let dx = sel.size[R[dim]];

        let mut gridmap = Vec::with_capacity((dx * dy * dz) as usize);
        for z in 0..dz {
            for y in 0..dy {
                for x in 0..dx {
                    let (wx, wy, wz) = blockcube_coords(sel, x, y, z);
                    let (_cube, _origin, size) = self.world.lookup(wx, wy, wz);
                    // bitscan = log2 of leaf size
                    gridmap.push(bitscan(size as u32));
                }
            }
        }
        gridmap
    }

    // ── Undo / Redo ──────────────────────────────────────────────────────────

    /// Create an undo record for the current state of the given selection.
    /// Port of Cube2 `newundocube(sel)` from octaedit.cpp:762-776.
    pub fn make_undo_block(&self, sel: &Selection, timestamp: u64) -> UndoBlock {
        let gridmap = self.selgridmap(sel);
        let block = self.blockcopy(sel);
        UndoBlock { block, gridmap, timestamp }
    }

    /// Save undo state for a selection before editing.
    /// Port of Cube2 `makeundo(sel)` from octaedit.cpp:789-793.
    pub fn make_undo(&mut self, sel: &Selection, timestamp: u64) {
        let ub = self.make_undo_block(sel, timestamp);
        self.undo.add_undo(ub);
    }

    /// Paste an undo block back into the world, respecting the gridmap
    /// (each cube is pasted at its original leaf size).
    /// Port of Cube2 `pasteundoblock(b, g)` from octaedit.cpp:673-677.
    fn paste_undo_block(&mut self, ub: &UndoBlock) {
        let sel = &ub.block.sel;
        let dim = FACE_DIM[sel.orient];
        let dz = sel.size[D[dim]];
        let dy = sel.size[C[dim]];
        let dx = sel.size[R[dim]];
        let ws = self.world.world_scale;

        let mut idx = 0usize;
        for z in 0..dz {
            for y in 0..dy {
                for x in 0..dx {
                    if idx < ub.block.cubes.len() && idx < ub.gridmap.len() {
                        let grid_log2 = (ub.gridmap[idx] as u32).min(ws - 1);
                        let grid = 1i32 << grid_log2;
                        let (wx, wy, wz) = blockcube_coords(sel, x, y, z);
                        self.world.subdivide_to(wx, wy, wz, grid);
                        let (dst, _, _) = self.world.lookup_mut(wx, wy, wz, grid);
                        pastecube(&ub.block.cubes[idx], dst);
                    }
                    idx += 1;
                }
            }
        }
        self.dirty = true;
    }

    /// Swap undo/redo: pop from `from_stack`, snapshot current state into
    /// `to_stack`, then paste the old state back.
    /// Port of Cube2 `swapundo(a, b, op)` from octaedit.cpp:811-864.
    pub fn undo(&mut self, timestamp: u64) {
        if let Some(old) = self.undo.pop_undo() {
            // Snapshot current state of that region → push to redo
            let current = self.make_undo_block(&old.block.sel, timestamp);
            self.undo.push_redo(current);
            // Restore old state
            self.paste_undo_block(&old);
        }
    }

    pub fn redo(&mut self, timestamp: u64) {
        if let Some(old) = self.undo.pop_redo() {
            // Snapshot current state → push to undo (without clearing redo)
            let current = self.make_undo_block(&old.block.sel, timestamp);
            let size = count_block_cubes(&current);
            self.undo.total_cubes += size;
            self.undo.undo.push(current);
            // Restore old state
            self.paste_undo_block(&old);
        }
    }

    // ── Transforms ───────────────────────────────────────────────────────────

    /// Flip the selection along its orient axis.
    /// Port of Cube2 `mpflip(sel, local)` from octaedit.cpp:2620-2638.
    pub fn flip(&mut self, sel: &Selection) {
        if !sel.is_valid() { return; }
        let d = FACE_DIM[sel.orient];
        let zs = sel.size[D[d]];
        let ys = sel.size[C[d]];
        let xs = sel.size[R[d]];

        // Ensure world is subdivided to sel.grid
        self.subdivide_sel(sel);

        // Flip each cube, then swap along depth
        for y in 0..ys {
            for x in 0..xs {
                for z in 0..zs {
                    let (wx, wy, wz) = blockcube_coords(sel, x, y, z);
                    let (cube, _, _) = self.world.lookup_mut(wx, wy, wz, sel.grid);
                    flipcube(cube, d);
                }
                // Swap cubes along depth axis (mirror)
                for z in 0..zs / 2 {
                    let (wa, wya, wza) = blockcube_coords(sel, x, y, z);
                    let (wb, wyb, wzb) = blockcube_coords(sel, x, y, zs - z - 1);
                    // Copy both, then write back swapped
                    let a = copycube(&self.world.lookup(wa, wya, wza).0);
                    let b = copycube(&self.world.lookup(wb, wyb, wzb).0);
                    let (dst_a, _, _) = self.world.lookup_mut(wa, wya, wza, sel.grid);
                    *dst_a = b;
                    let (dst_b, _, _) = self.world.lookup_mut(wb, wyb, wzb, sel.grid);
                    *dst_b = a;
                }
            }
        }
        self.dirty = true;
    }

    /// Rotate the selection clockwise (cw > 0) or counter-clockwise (cw < 0).
    /// Port of Cube2 `mprotate(cw, sel, local)` from octaedit.cpp:2647-2667.
    pub fn rotate(&mut self, cw: i32, sel: &mut Selection) {
        if !sel.is_valid() { return; }
        let d = FACE_DIM[sel.orient];
        let dc = FACE_SIDE[sel.orient];
        let cw_adj = if dc == 0 { -cw } else { cw };
        // Make selection square (largest of R/C)
        let ss = sel.size[C[d]].max(sel.size[R[d]]);
        sel.size[C[d]] = ss;
        sel.size[R[d]] = ss;

        self.subdivide_sel(sel);

        let iterations = if cw_adj > 0 { 1 } else { 3 };
        let dz = sel.size[D[d]];

        for z in 0..dz {
            for _ in 0..iterations {
                // Rotate each cube in-place
                for y in 0..ss {
                    for x in 0..ss {
                        let (wx, wy, wz) = blockcube_coords(sel, x, y, z);
                        let (cube, _, _) = self.world.lookup_mut(wx, wy, wz, sel.grid);
                        rotatecube(cube, d);
                    }
                }
                // Rotate the grid of cubes (shell rotation)
                for y in 0..ss / 2 {
                    for x in 0..(ss - 1 - y * 2) {
                        // Four corners of the rotation quad:
                        // (ss-1-y, x+y) → (x+y, y) → (y, ss-1-x-y) → (ss-1-x-y, ss-1-y)
                        let coords = [
                            blockcube_coords(sel, ss - 1 - y, x + y, z),
                            blockcube_coords(sel, x + y, y, z),
                            blockcube_coords(sel, y, ss - 1 - x - y, z),
                            blockcube_coords(sel, ss - 1 - x - y, ss - 1 - y, z),
                        ];
                        // rotatequad: a←b, b←c, c←d, d←a
                        let t = copycube(self.world.lookup(coords[0].0, coords[0].1, coords[0].2).0);
                        let b = copycube(self.world.lookup(coords[1].0, coords[1].1, coords[1].2).0);
                        let c = copycube(self.world.lookup(coords[2].0, coords[2].1, coords[2].2).0);
                        let dd = copycube(self.world.lookup(coords[3].0, coords[3].1, coords[3].2).0);

                        let (dst, _, _) = self.world.lookup_mut(coords[0].0, coords[0].1, coords[0].2, sel.grid);
                        *dst = b;
                        let (dst, _, _) = self.world.lookup_mut(coords[1].0, coords[1].1, coords[1].2, sel.grid);
                        *dst = c;
                        let (dst, _, _) = self.world.lookup_mut(coords[2].0, coords[2].1, coords[2].2, sel.grid);
                        *dst = dd;
                        let (dst, _, _) = self.world.lookup_mut(coords[3].0, coords[3].1, coords[3].2, sel.grid);
                        *dst = t;
                    }
                }
            }
        }
        self.dirty = true;
    }

    // ── Private ──────────────────────────────────────────────────────────────

    /// Ensure all cells in the selection are subdivided to sel.grid.
    fn subdivide_sel(&mut self, sel: &Selection) {
        if !sel.is_valid() { return; }
        let x0 = sel.origin[0];
        let y0 = sel.origin[1];
        let z0 = sel.origin[2];
        let x1 = x0 + sel.size[0] * sel.grid;
        let y1 = y0 + sel.size[1] * sel.grid;
        let z1 = z0 + sel.size[2] * sel.grid;

        if sel.grid < self.world.world_size() {
            let mut cz = z0;
            while cz < z1 {
                let mut cy = y0;
                while cy < y1 {
                    let mut cx = x0;
                    while cx < x1 {
                        self.world.subdivide_to(cx, cy, cz, sel.grid);
                        cx += sel.grid;
                    }
                    cy += sel.grid;
                }
                cz += sel.grid;
            }
        }
    }

    /// Walk the octree for all leaves that overlap the selection's bounding box
    /// and apply `op` to each.  Sets `dirty = true`.
    ///
    /// If the selection's `grid` is finer than existing leaf nodes, those nodes
    /// are subdivided first so the op is applied at the correct resolution.
    fn apply<F>(&mut self, sel: &Selection, mut op: F)
    where F: FnMut(&mut Cube) {
        if !sel.is_valid() { return; }

        let x0 = sel.origin[0];
        let y0 = sel.origin[1];
        let z0 = sel.origin[2];
        let x1 = x0 + sel.size[0] * sel.grid;
        let y1 = y0 + sel.size[1] * sel.grid;
        let z1 = z0 + sel.size[2] * sel.grid;

        // If the selection grid is finer than some leaves, subdivide those
        // leaves down to sel.grid first, cell by cell.
        if sel.grid < self.world.world_size() {
            let mut cz = z0;
            while cz < z1 {
                let mut cy = y0;
                while cy < y1 {
                    let mut cx = x0;
                    while cx < x1 {
                        self.world.subdivide_to(cx, cy, cz, sel.grid);
                        cx += sel.grid;
                    }
                    cy += sel.grid;
                }
                cz += sel.grid;
            }
        }

        self.world.for_each_leaf_mut_in_aabb(x0, y0, z0, x1, y1, z1, |cube, _, _| {
            op(cube);
        });

        self.dirty = true;
    }

    /// Like apply() but passes the cube's (ox, oy) position within the
    /// selection's R/C face axes (0-based indices). Used by editface mode 0
    /// for position-dependent edge filtering (C++ loopselxyz gives this).
    fn apply_with_pos<F>(&mut self, sel: &Selection, mut op: F)
    where F: FnMut(&mut Cube, i32, i32) {
        if !sel.is_valid() { return; }

        let d = FACE_DIM[sel.orient];
        let x0 = sel.origin[0];
        let y0 = sel.origin[1];
        let z0 = sel.origin[2];
        let g = sel.grid;

        // Subdivide all cells to grid size first
        let x1 = x0 + sel.size[0] * g;
        let y1 = y0 + sel.size[1] * g;
        let z1 = z0 + sel.size[2] * g;
        if g < self.world.world_size() {
            let mut cz = z0;
            while cz < z1 {
                let mut cy = y0;
                while cy < y1 {
                    let mut cx = x0;
                    while cx < x1 {
                        self.world.subdivide_to(cx, cy, cz, g);
                        cx += g;
                    }
                    cy += g;
                }
                cz += g;
            }
        }

        // Iterate and compute position within selection for each cube
        self.world.for_each_leaf_mut_in_aabb(x0, y0, z0, x1, y1, z1, |cube, origin, _size| {
            let ox = (origin.0 - x0) / g; // position along X axis
            let oy = (origin.1 - y0) / g; // position along Y axis
            let oz = (origin.2 - z0) / g; // position along Z axis
            // Map world-axis positions to face R/C positions
            let pos_r = [ox, oy, oz][R[d]];
            let pos_c = [ox, oy, oz][C[d]];
            op(cube, pos_r, pos_c);
        });

        self.dirty = true;
    }

    // ── New map ──────────────────────────────────────────────────────────────

    /// Create a fresh map at the given scale (world_size = 1 << scale).
    /// Bottom half is solid, top half is empty (same as the default test world).
    /// Clears undo/redo stacks.
    pub fn newmap(&mut self, scale: u32) {
        let mut root: [Cube; 8] = Default::default();
        for i in 0..8usize {
            let z_high = (i >> 2) & 1 == 1;
            root[i] = if z_high { Cube::empty() } else { Cube::solid() };
        }
        self.world = OctreeWorld {
            root: Box::new(root),
            world_scale: scale,
            entities: Vec::new(),
            ogz_version: 33,
        };
        self.undo = UndoStack::new();
        self.dirty = true;
    }
}

// ── Heightmap editing ─────────────────────────────────────────────────────────
// Port of Cube2 `namespace hmap` from octaedit.cpp:1624-1914.

pub const MAXBRUSH: usize = 64;
pub const MAXBRUSH2: usize = 32;
const MAXBRUSHC: usize = 63;

const HMAP_PAINTED: u8 = 1;
const HMAP_NOTHMAP: u8 = 2;
const HMAP_MAPPED:  u8 = 16;

/// Brush pattern for heightmap painting.
/// Center is at (MAXBRUSH2, MAXBRUSH2).
#[derive(Clone)]
pub struct Brush {
    pub data: [[i32; MAXBRUSH]; MAXBRUSH],
    pub min_x: usize,
    pub max_x: usize,
    pub min_y: usize,
    pub max_y: usize,
    pub paint: bool,
}

impl Default for Brush {
    fn default() -> Self {
        Self {
            data: [[0; MAXBRUSH]; MAXBRUSH],
            min_x: MAXBRUSH,
            max_x: 0,
            min_y: MAXBRUSH,
            max_y: 0,
            paint: false,
        }
    }
}

impl Brush {
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Add a brush vertex. Port of Cube2 `brushvert(x, y, v)`.
    pub fn set_vert(&mut self, x: i32, y: i32, v: i32) {
        let bx = (x + MAXBRUSH2 as i32) as usize;
        let by = (y + MAXBRUSH2 as i32) as usize;
        if bx >= MAXBRUSH || by >= MAXBRUSH { return; }
        self.data[bx][by] = v.clamp(0, 8);
        self.paint = self.paint || self.data[bx][by] > 0;
        self.max_x = (MAXBRUSH - 1).min(self.max_x.max(bx + 1));
        self.max_y = (MAXBRUSH - 1).min(self.max_y.max(by + 1));
        self.min_x = self.min_x.min(bx);
        self.min_y = self.min_y.min(by);
    }
}

/// Parameters for a heightmap edit operation.
pub struct HeightmapParams {
    /// Cursor position in world space.
    pub cursor: [i32; 3],
    /// Grid size (power of 2).
    pub grid_size: i32,
    /// Log2 of grid size.
    pub grid_power: u32,
    /// Selection orientation (O_LEFT..O_TOP).
    pub orient: usize,
    /// Corner index (0-3).
    pub corner: usize,
    /// Whether a selection box is active.
    pub have_sel: bool,
    /// Selection bounds (only used if have_sel).
    pub sel_origin: [i32; 3],
    pub sel_size: [i32; 3],
    /// Textures to treat as heightmap (empty = all).
    pub htextures: Vec<u16>,
}

/// Heightmap editor state. Created per-operation, not persistent.
/// Port of Cube2 `namespace hmap` from octaedit.cpp.
pub struct HeightmapEditor {
    flags: [[u8; MAXBRUSH]; MAXBRUSH],
    map: [[i32; MAXBRUSH]; MAXBRUSH],
    mapz: [[i32; MAXBRUSHC]; MAXBRUSHC],
    d: usize,
    dc: usize,
    dcr: i32,
    dr: i32,
    biasup: bool,
    hws: i32,
    fg: i32,
    fs: u32,
    gx: i32, gy: i32, gz: i32,
    mx: i32, my: i32,
    nx: i32, ny: i32,
    _mz: i32, _nz: i32,
    bmx: i32, bmy: i32,
    bnx: i32, bny: i32,
}

impl HeightmapEditor {
    /// Run a heightmap edit operation.
    /// Port of Cube2 `hmap::run(dir, mode)` from octaedit.cpp:1851-1913.
    ///
    /// `dir`: +1 or -1 (raise/lower)
    /// `mode`: >= 0 = paint with brush, < 0 = smooth selection
    pub fn run(
        world: &mut EditWorld,
        params: &HeightmapParams,
        brush: &Brush,
        dir: i32,
        mode: i32,
    ) {
        let d = FACE_DIM[params.orient];
        let dc = FACE_SIDE[params.orient];
        let dcr = if dc != 0 { 1i32 } else { -1 };
        let dr = if dir > 0 { 1 } else { -1 };
        let biasup = dir < 0;
        let paint_mode = brush.paint && mode >= 0;

        let cx = if params.corner & 1 != 0 { 0 } else { -1 };
        let cy = if params.corner & 2 != 0 { 0 } else { -1 };
        let world_size = world.world.world_size();
        let hws_cells = world_size >> params.grid_power;
        let gx = (params.cursor[R[d]] >> params.grid_power) + cx - MAXBRUSH2 as i32;
        let gy = (params.cursor[C[d]] >> params.grid_power) + cy - MAXBRUSH2 as i32;
        let gz = params.cursor[D[d]] >> params.grid_power;
        let fs = if dc != 0 { 4u32 } else { 0 };
        let fg = if dc != 0 { params.grid_size } else { -params.grid_size };

        let mut mx = 0i32.max(-gx);
        let mut my = 0i32.max(-gy);
        let mut nx = (MAXBRUSH as i32 - 1).min(hws_cells - gx) - 1;
        let mut ny = (MAXBRUSH as i32 - 1).min(hws_cells - gy) - 1;

        let bmx;
        let bmy;
        let bnx;
        let bny;

        if params.have_sel {
            bmx = mx.max((params.sel_origin[R[d]] >> params.grid_power) - gx);
            bmy = my.max((params.sel_origin[C[d]] >> params.grid_power) - gy);
            bnx = nx.min((params.sel_size[R[d]] + (params.sel_origin[R[d]] >> params.grid_power)) - gx - 1);
            bny = ny.min((params.sel_size[C[d]] + (params.sel_origin[C[d]] >> params.grid_power)) - gy - 1);
            mx = bmx; my = bmy; nx = bnx; ny = bny;
        } else {
            bmx = mx.max(brush.min_x as i32);
            bmy = my.max(brush.min_y as i32);
            bnx = nx.min(brush.max_x as i32 - 1);
            bny = ny.min(brush.max_y as i32 - 1);
        }

        let nz = world_size - params.grid_size;
        let mz = 0i32;

        let mut ed = HeightmapEditor {
            flags: [[0u8; MAXBRUSH]; MAXBRUSH],
            map: [[0i32; MAXBRUSH]; MAXBRUSH],
            mapz: [[0i32; MAXBRUSHC]; MAXBRUSHC],
            d, dc, dcr, dr, biasup, hws: hws_cells, fg,
            fs, gx, gy, gz, mx, my, nx, ny, _mz: mz, _nz: nz,
            bmx, bmy, bnx, bny,
        };

        // Select phase: find heightmap surfaces and build the map
        ed.select_recursive(world, params, bmx, bmy, bnx, bny,
            if dc != 0 { gz } else { hws_cells - gz },
        );

        // Paint or smooth
        if paint_mode {
            ed.paint(brush);
        } else {
            ed.smooth();
        }

        // Ripple and set: propagate changes and write back to cubes
        ed.ripple_and_set(world, params, world_size);

        world.dirty = true;
    }

    fn select_recursive(
        &mut self, world: &mut EditWorld, params: &HeightmapParams,
        bmx: i32, bmy: i32, bnx: i32, bny: i32, z: i32,
    ) {
        // Select the center point, which recursively selects adjacent cubes
        let cx = (MAXBRUSH2 as i32).clamp(bmx, bnx);
        let cy = (MAXBRUSH2 as i32).clamp(bmy, bny);
        self.select_point(world, params, cx, cy, z, true);
    }

    fn select_point(
        &mut self, world: &mut EditWorld, params: &HeightmapParams,
        x: i32, y: i32, z: i32, selecting: bool,
    ) {
        if x < 0 || y < 0 || x >= MAXBRUSH as i32 || y >= MAXBRUSH as i32 { return; }
        let xu = x as usize;
        let yu = y as usize;
        if (self.flags[xu][yu] & HMAP_NOTHMAP) != 0 || (self.flags[xu][yu] & HMAP_PAINTED) != 0 {
            return;
        }

        let mut z = z;
        let mut t = [0i32; 3];
        t[R[self.d]] = (x + self.gx) << params.grid_power;
        t[C[self.d]] = (y + self.gy) << params.grid_power;
        t[D[self.d]] = if self.dc != 0 { z } else { self.hws - z };
        t[D[self.d]] <<= params.grid_power;

        // Look at the cube at this position
        let (c1, _, _) = world.world.lookup(t[0], t[1], t[2]);
        let c1_empty = c1.is_empty();
        let c1_tex = c1.texture[params.orient];

        if !c1_empty {
            // Try going up
            let mut tu = t;
            tu[self.d] += self.dcr * params.grid_size;
            let (cup, _, _) = world.world.lookup(tu[0], tu[1], tu[2]);
            if !cup.is_empty() {
                self.flags[xu][yu] |= HMAP_NOTHMAP;
                return;
            }
            z += 1;
        } else {
            // Drop down
            z -= 1;
            t[self.d] -= self.fg;
            let (c1d, _, _) = world.world.lookup(t[0], t[1], t[2]);
            if c1d.is_empty() {
                self.flags[xu][yu] |= HMAP_NOTHMAP;
                return;
            }
        }

        // Check heightmap texture filter
        if !params.htextures.is_empty() && !params.htextures.contains(&c1_tex) && !params.have_sel {
            self.flags[xu][yu] |= HMAP_NOTHMAP;
            return;
        }

        self.flags[xu][yu] |= HMAP_PAINTED;
        self.mapz[xu][yu] = z;

        // Get face value and add points
        let (c, _, _) = world.world.lookup(t[0], t[1], t[2]);
        let face = self.getface_val(c);
        let f = face.to_le_bytes();
        self.addpoint(x, y, z, f[0] as i32);
        self.addpoint(x + 1, y, z, f[1] as i32);
        self.addpoint(x, y + 1, z, f[2] as i32);
        self.addpoint(x + 1, y + 1, z, f[3] as i32);

        if selecting {
            if x > self.bmx { self.select_point(world, params, x - 1, y, z, true); }
            if x < self.bnx { self.select_point(world, params, x + 1, y, z, true); }
            if y > self.bmy { self.select_point(world, params, x, y - 1, z, true); }
            if y < self.bny { self.select_point(world, params, x, y + 1, z, true); }
        }
    }

    fn getface_val(&self, c: &Cube) -> u32 {
        let face = c.face(self.d);
        let adjusted = if self.dc != 0 { face } else { 0x88888888u32.wrapping_sub(face) };
        0x0F0F0F0F & (adjusted >> self.fs)
    }

    fn addpoint(&mut self, x: i32, y: i32, z: i32, v: i32) {
        if x < 0 || y < 0 || x >= MAXBRUSH as i32 || y >= MAXBRUSH as i32 { return; }
        let xu = x as usize;
        let yu = y as usize;
        if (self.flags[xu][yu] & HMAP_MAPPED) == 0 {
            self.map[xu][yu] = v + z * 8;
        }
        self.flags[xu][yu] |= HMAP_MAPPED;
    }

    fn paint(&mut self, brush: &Brush) {
        for x in self.bmx..=self.bnx + 1 {
            for y in self.bmy..=self.bny + 1 {
                if x >= 0 && y >= 0 && (x as usize) < MAXBRUSH && (y as usize) < MAXBRUSH {
                    self.map[x as usize][y as usize] -=
                        self.dr * brush.data[x as usize][y as usize];
                }
            }
        }
    }

    fn smooth(&mut self) {
        for x in self.bmx..=self.bnx - 1 {
            for y in self.bmy..=self.bny - 1 {
                if x < 0 || y < 0 { continue; }
                let xu = x as usize;
                let yu = y as usize;
                let mut sum = 0i32;
                let mut div = 9i32;
                for i in 0..3 {
                    for j in 0..3 {
                        let mx = xu + i;
                        let my = yu + j;
                        if mx < MAXBRUSH && my < MAXBRUSH && (self.flags[mx][my] & HMAP_MAPPED) != 0 {
                            sum += self.map[mx][my];
                        } else {
                            div -= 1;
                        }
                    }
                }
                if div > 0 && xu + 1 < MAXBRUSH && yu + 1 < MAXBRUSH {
                    self.map[xu + 1][yu + 1] = sum / div;
                }
            }
        }
    }

    fn ripple_and_set(&mut self, world: &mut EditWorld, params: &HeightmapParams, world_size: i32) {
        for x in self.bmx..=self.bnx {
            for y in self.bmy..=self.bny {
                self.ripple(world, params, x, y, self.gz, false, world_size);
            }
        }
    }

    fn ripple(
        &mut self, world: &mut EditWorld, params: &HeightmapParams,
        x: i32, y: i32, z: i32, force: bool, world_size: i32,
    ) {
        if x < 0 || y < 0 || x >= MAXBRUSH as i32 || y >= MAXBRUSH as i32 { return; }
        let xu = x as usize;
        let yu = y as usize;

        if force {
            self.select_point(world, params, x, y, z, false);
        }
        if (self.flags[xu][yu] & HMAP_NOTHMAP) != 0 || (self.flags[xu][yu] & HMAP_PAINTED) == 0 {
            return;
        }

        // Pull/push heightmap values for smoothing
        let mut changed = false;
        {
            let o = [
                &mut self.map[xu][yu] as *mut i32,
                &mut self.map[xu + 1][yu] as *mut i32,
                &mut self.map[xu][yu + 1] as *mut i32,
                &mut self.map[xu + 1][yu + 1] as *mut i32,
            ];
            unsafe {
                if self.biasup {
                    changed = Self::pullhmap_up(&o);
                } else {
                    changed = Self::pullhmap_down(&o, world_size);
                }
            }
        }

        // Write back to cubes
        let mz = self.mapz[xu][yu];
        for k in 0..4i32 {
            let cube_z = mz + 3 - k;
            let mut t = [0i32; 3];
            t[R[self.d]] = (x + self.gx) << params.grid_power;
            t[C[self.d]] = (y + self.gy) << params.grid_power;
            t[D[self.d]] = if self.dc != 0 { cube_z } else { self.hws - cube_z };
            t[D[self.d]] <<= params.grid_power;

            if t[0] < 0 || t[1] < 0 || t[2] < 0
                || t[0] >= world_size || t[1] >= world_size || t[2] >= world_size
            {
                continue;
            }

            world.world.subdivide_to(t[0], t[1], t[2], params.grid_size);
            let (c, _, _) = world.world.lookup_mut(t[0], t[1], t[2], params.grid_size);

            let mut notempty = 0i32;
            let mut e = [[0i32; 2]; 2];
            for i in 0..2 {
                for j in 0..2 {
                    let mx = (x + i as i32) as usize;
                    let my = (y + j as i32) as usize;
                    if mx < MAXBRUSH && my < MAXBRUSH {
                        e[i][j] = 8i32.min(self.map[mx][my] - (mz + 3 - k) * 8);
                    }
                    notempty |= if e[i][j] > 0 { 1 } else { 0 };
                }
            }

            if notempty != 0 {
                c.solidfaces();
                for i in 0..2usize {
                    for j in 0..2usize {
                        let f = e[i][j];
                        if f < 0 || (f == 0 && e[1 - i][j] == 0 && e[i][1 - j] == 0) {
                            // pushside: flip R component
                            let v0 = c.get_cube_vector(self.d, i, j, 0);
                            let mut v0m = v0;
                            v0m[R[self.d]] = 8 - v0m[R[self.d]];
                            c.set_cube_vector(self.d, i, j, 0, v0m);

                            let v1 = c.get_cube_vector(self.d, i, j, 1);
                            let mut v1m = v1;
                            v1m[R[self.d]] = 8 - v1m[R[self.d]];
                            c.set_cube_vector(self.d, i, j, 1, v1m);
                        }
                        let fval = if self.dc != 0 { f } else { 8 - f };
                        let fval = fval.clamp(0, 8) as u8;
                        edge_set(&mut c.edges[edge_idx(self.d, i, j)], self.dc, fval);
                    }
                }
            } else {
                c.emptyfaces();
            }
        }

        if !changed { return; }
        // Ripple to neighbors
        if x > self.mx { self.ripple(world, params, x - 1, y, mz, true, world_size); }
        if x < self.nx { self.ripple(world, params, x + 1, y, mz, true, world_size); }
        if y > self.my { self.ripple(world, params, x, y - 1, mz, true, world_size); }
        if y < self.ny { self.ripple(world, params, x, y + 1, mz, true, world_size); }
    }

    unsafe fn pullhmap_up(o: &[*mut i32; 4]) -> bool {
        let mut changed = false;
        let mut best = 0i32;
        for i in 0..4 {
            if *o[i] > best { best = *o[i] - 1; }
        }
        let par = (best & !7) + 0;
        // Single layer
        for j in 0..4 {
            if *o[j] > par { *o[j] = par; changed = true; }
        }
        changed
    }

    unsafe fn pullhmap_down(o: &[*mut i32; 4], world_size: i32) -> bool {
        let mut changed = false;
        let mut best = world_size * 8;
        for i in 0..4 {
            if *o[i] < best { best = *o[i]; }
        }
        let par = (best & !7) + 8;
        for j in 0..4 {
            if *o[j] < par { *o[j] = par; changed = true; }
        }
        changed
    }
}
