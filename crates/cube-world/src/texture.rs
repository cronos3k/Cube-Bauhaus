//! Texture Slot and VSlot system — port of Cube2/Sauerbraten's texture management.
//!
//! Architecture (matching C++ exactly):
//!   cube.texture[face] = u16 VSlot index
//!   VSlot → Slot (back-reference)
//!   Slot has a linked list of VSlot variants (same textures, different transforms)
//!
//! VSlot properties: rotation (0-7), scale, offset, scroll, color, alpha, layer.
//! UVs are computed at mesh-build time via world-space planar projection.

/// Texture rotation table — 8 entries matching C++'s texrotations[8].
/// Each entry defines flipx, flipy, swapxy for UV transform.
#[derive(Debug, Clone, Copy, Default)]
pub struct TexRotation {
    pub flip_x: bool,
    pub flip_y: bool,
    pub swap_xy: bool,
}

pub const TEX_ROTATIONS: [TexRotation; 8] = [
    TexRotation { flip_x: false, flip_y: false, swap_xy: false }, // 0: identity
    TexRotation { flip_x: false, flip_y: true,  swap_xy: true  }, // 1: 90° CW
    TexRotation { flip_x: true,  flip_y: true,  swap_xy: false }, // 2: 180°
    TexRotation { flip_x: true,  flip_y: false, swap_xy: true  }, // 3: 270° CW
    TexRotation { flip_x: true,  flip_y: false, swap_xy: false }, // 4: flip X
    TexRotation { flip_x: false, flip_y: true,  swap_xy: false }, // 5: flip Y
    TexRotation { flip_x: false, flip_y: false, swap_xy: true  }, // 6: transpose
    TexRotation { flip_x: true,  flip_y: true,  swap_xy: true  }, // 7: flipped transpose
];

/// C++'s TEX_SCALE = 8.0 — at scale=1, one texture repeat covers 8 world units
/// per texture-pixel-dimension.
pub const TEX_SCALE: f32 = 8.0;

/// Sub-texture types within a Slot (C++'s TEX_DIFFUSE etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TexType {
    Diffuse = 0,
    Unknown = 1,
    Decal = 2,
    Normal = 3,
    Glow = 4,
    Spec = 5,
    Depth = 6,
    Alpha = 7,
    Envmap = 8,
}

/// A sub-texture entry within a Slot (path + type).
#[derive(Debug, Clone)]
pub struct SlotTex {
    pub tex_type: TexType,
    pub path: String,
    /// GPU texture array layer index (set by the renderer after loading)
    pub layer: u32,
    /// Texture dimensions (set after loading)
    pub width: u32,
    pub height: u32,
}

/// A shader parameter override.
#[derive(Debug, Clone)]
pub struct SlotShaderParam {
    pub name: String,
    pub val: [f32; 4],
}

/// VSlot changed bitmask flags.
pub const VSLOT_SHPARAM:  u32 = 1 << 0;
pub const VSLOT_SCALE:    u32 = 1 << 1;
pub const VSLOT_ROTATION: u32 = 1 << 2;
pub const VSLOT_OFFSET:   u32 = 1 << 3;
pub const VSLOT_SCROLL:   u32 = 1 << 4;
pub const VSLOT_LAYER:    u32 = 1 << 5;
pub const VSLOT_ALPHA:    u32 = 1 << 6;
pub const VSLOT_COLOR:    u32 = 1 << 7;

/// A virtual texture slot — per-face texture parameter overrides.
/// Multiple VSlots can share the same parent Slot (same texture images,
/// different rotation/scale/offset/etc.)
///
/// Port of C++'s `VSlot` struct from texture.h.
#[derive(Debug, Clone)]
pub struct VSlot {
    /// Index of the parent Slot in the global slots array.
    pub slot: usize,
    /// Index of this VSlot in the global vslots array.
    pub index: usize,
    /// Bitmask of which fields were changed from the base slot defaults.
    pub changed: u32,

    // ── Transform parameters ──
    /// Texture scale multiplier (default 1.0, range [1/8, 8]).
    pub scale: f32,
    /// Rotation index 0-7 (see TEX_ROTATIONS).
    pub rotation: i32,
    /// Texel offset (integer pixels).
    pub offset: [i32; 2],
    /// Scroll speed (texels/millisecond after /1000).
    pub scroll: [f32; 2],

    // ── Appearance parameters ──
    /// Blend layer VSlot index (0 = none).
    pub layer: i32,
    /// Front-face alpha (default 0.5).
    pub alpha_front: f32,
    /// Back-face alpha (default 0.0).
    pub alpha_back: f32,
    /// RGB color multiplier (default [1,1,1]).
    pub color_scale: [f32; 3],
    /// Glow color multiplier (default [1,1,1]).
    pub glow_color: [f32; 3],

    /// Shader parameter overrides.
    pub params: Vec<SlotShaderParam>,
}

impl Default for VSlot {
    fn default() -> Self {
        Self {
            slot: 0,
            index: 0,
            changed: 0,
            scale: 1.0,
            rotation: 0,
            offset: [0, 0],
            scroll: [0.0, 0.0],
            layer: 0,
            alpha_front: 0.5,
            alpha_back: 0.0,
            color_scale: [1.0, 1.0, 1.0],
            glow_color: [1.0, 1.0, 1.0],
            params: Vec::new(),
        }
    }
}

impl VSlot {
    /// Reset all parameters to defaults.
    pub fn reset(&mut self) {
        self.changed = 0;
        self.scale = 1.0;
        self.rotation = 0;
        self.offset = [0, 0];
        self.scroll = [0.0, 0.0];
        self.layer = 0;
        self.alpha_front = 0.5;
        self.alpha_back = 0.0;
        self.color_scale = [1.0, 1.0, 1.0];
        self.glow_color = [1.0, 1.0, 1.0];
        self.params.clear();
    }
}

/// A texture slot — one set of texture images (diffuse + optional normal/spec/glow).
/// Port of C++'s `Slot` struct from texture.h.
#[derive(Debug, Clone)]
pub struct Slot {
    /// Index in the global slots array.
    pub index: usize,
    /// Sub-textures (diffuse, normal, spec, glow, etc.)
    pub textures: Vec<SlotTex>,
    /// Shader name (e.g., "stdworld", "bumpspecmapworld").
    pub shader: String,
    /// Slot-level shader parameter defaults.
    pub params: Vec<SlotShaderParam>,
    /// Indices of VSlot variants that use this slot (first = base variant).
    pub variants: Vec<usize>,
    /// Whether GPU textures have been loaded.
    pub loaded: bool,
}

impl Default for Slot {
    fn default() -> Self {
        Self {
            index: 0,
            textures: Vec::new(),
            shader: "stdworld".to_string(),
            params: Vec::new(),
            variants: Vec::new(),
            loaded: false,
        }
    }
}

/// The global texture manager — holds all Slots and VSlots.
/// Port of C++'s global `slots` and `vslots` vectors.
#[derive(Debug)]
pub struct TextureRegistry {
    pub slots: Vec<Slot>,
    pub vslots: Vec<VSlot>,
}

impl TextureRegistry {
    pub fn new() -> Self {
        let mut reg = Self {
            slots: Vec::new(),
            vslots: Vec::new(),
        };
        // Slot 0 = sky (DEFAULT_SKY), Slot 1 = default geometry (DEFAULT_GEOM)
        reg.add_slot_with_tex("sky", TexType::Diffuse, "");
        reg.add_slot_with_tex("default", TexType::Diffuse, "");
        reg
    }

    /// Add a new Slot with a single diffuse texture, creating a base VSlot.
    /// Returns the slot index.
    pub fn add_slot_with_tex(&mut self, shader: &str, tex_type: TexType, path: &str) -> usize {
        let slot_idx = self.slots.len();
        let vslot_idx = self.vslots.len();

        let mut slot = Slot::default();
        slot.index = slot_idx;
        slot.shader = shader.to_string();
        slot.textures.push(SlotTex {
            tex_type,
            path: path.to_string(),
            layer: 0,
            width: 256,   // default until loaded
            height: 256,
        });
        slot.variants.push(vslot_idx);

        let vslot = VSlot {
            slot: slot_idx,
            index: vslot_idx,
            ..VSlot::default()
        };

        self.slots.push(slot);
        self.vslots.push(vslot);
        slot_idx
    }

    /// Add a sub-texture (normal, spec, glow) to the last-added slot.
    pub fn add_subtex_to_last(&mut self, tex_type: TexType, path: &str) {
        if let Some(slot) = self.slots.last_mut() {
            slot.textures.push(SlotTex {
                tex_type,
                path: path.to_string(),
                layer: 0,
                width: 256,
                height: 256,
            });
        }
    }

    /// Look up a VSlot by index, with fallback to DEFAULT_GEOM's base variant.
    pub fn lookup_vslot(&self, index: u16) -> &VSlot {
        let idx = index as usize;
        if idx < self.vslots.len() {
            &self.vslots[idx]
        } else if self.vslots.len() > 1 {
            &self.vslots[1] // DEFAULT_GEOM's base vslot
        } else {
            &self.vslots[0]
        }
    }

    /// Look up a Slot by index.
    pub fn lookup_slot(&self, index: usize) -> &Slot {
        if index < self.slots.len() {
            &self.slots[index]
        } else if self.slots.len() > 1 {
            &self.slots[1] // DEFAULT_GEOM
        } else {
            &self.slots[0]
        }
    }

    /// Get the Slot for a VSlot.
    pub fn slot_for_vslot(&self, vslot: &VSlot) -> &Slot {
        self.lookup_slot(vslot.slot)
    }

    // ── VSlot editing ────────────────────────────────────────────────────────

    /// Find or create a VSlot that matches `src` with `delta` applied.
    /// This is the copy-on-write mechanism: if an identical VSlot already
    /// exists in the Slot's variant list, reuse it. Otherwise clone a new one.
    ///
    /// Port of C++'s `editvslot(src, delta)` + `findvslot`.
    pub fn edit_vslot(&mut self, src_index: usize, delta: &VSlot) -> usize {
        if src_index >= self.vslots.len() { return src_index; }

        // Build the merged result
        let merged = self.merge_vslot(src_index, delta);

        // Check if an identical VSlot already exists for this Slot
        let slot_idx = self.vslots[src_index].slot;
        if slot_idx < self.slots.len() {
            for &vi in &self.slots[slot_idx].variants {
                if vi < self.vslots.len() && vslots_match(&self.vslots[vi], &merged) {
                    return vi;
                }
            }
        }

        // Clone a new VSlot
        let new_idx = self.vslots.len();
        let mut new_vs = merged;
        new_vs.index = new_idx;
        self.vslots.push(new_vs);

        // Add to slot's variant list
        if slot_idx < self.slots.len() {
            self.slots[slot_idx].variants.push(new_idx);
        }

        new_idx
    }

    /// Merge delta VSlot properties into src VSlot, producing a new VSlot.
    /// Port of C++'s `mergevslot(ms, vs, ds)` — delta mode.
    fn merge_vslot(&self, src_index: usize, delta: &VSlot) -> VSlot {
        let src = &self.vslots[src_index];
        let mut dst = src.clone();
        dst.changed |= delta.changed;

        if delta.changed & VSLOT_SCALE != 0 {
            dst.scale = (dst.scale * delta.scale).clamp(0.125, 8.0);
        }
        if delta.changed & VSLOT_ROTATION != 0 {
            dst.rotation = (dst.rotation + delta.rotation).rem_euclid(8);
        }
        if delta.changed & VSLOT_OFFSET != 0 {
            dst.offset[0] += delta.offset[0];
            dst.offset[1] += delta.offset[1];
        }
        if delta.changed & VSLOT_SCROLL != 0 {
            dst.scroll[0] += delta.scroll[0];
            dst.scroll[1] += delta.scroll[1];
        }
        if delta.changed & VSLOT_LAYER != 0 {
            dst.layer = delta.layer;
        }
        if delta.changed & VSLOT_ALPHA != 0 {
            dst.alpha_front = delta.alpha_front;
            dst.alpha_back = delta.alpha_back;
        }
        if delta.changed & VSLOT_COLOR != 0 {
            dst.color_scale[0] *= delta.color_scale[0];
            dst.color_scale[1] *= delta.color_scale[1];
            dst.color_scale[2] *= delta.color_scale[2];
        }
        if delta.changed & VSLOT_SHPARAM != 0 {
            for dp in &delta.params {
                if let Some(ep) = dst.params.iter_mut().find(|p| p.name == dp.name) {
                    ep.val = dp.val;
                } else {
                    dst.params.push(dp.clone());
                }
            }
        }
        dst
    }

    /// Apply a VSlot delta to all faces in a selection.
    /// Port of C++'s `mpeditvslot(usevdelta, ds, allfaces, sel, local)`.
    ///
    /// `all_faces`: if true, apply to all 6 faces; if false, only to sel.orient.
    pub fn edit_vslot_selection(
        &mut self,
        world: &mut crate::octree::OctreeWorld,
        sel: &crate::editing::Selection,
        delta: &VSlot,
        all_faces: bool,
    ) {
        if !sel.is_valid() { return; }

        let x0 = sel.origin[0];
        let y0 = sel.origin[1];
        let z0 = sel.origin[2];
        let x1 = x0 + sel.size[0] * sel.grid;
        let y1 = y0 + sel.size[1] * sel.grid;
        let z1 = z0 + sel.size[2] * sel.grid;

        // Build a remap cache to avoid creating duplicate VSlots
        let mut remap: std::collections::HashMap<u16, u16> = std::collections::HashMap::new();

        world.for_each_leaf_mut_in_aabb(x0, y0, z0, x1, y1, z1, |cube, _, _| {
            for face in 0..6 {
                if !all_faces && face != sel.orient { continue; }
                let old_idx = cube.texture[face];
                let new_idx = if let Some(&cached) = remap.get(&old_idx) {
                    cached
                } else {
                    let ni = self.edit_vslot(old_idx as usize, delta) as u16;
                    remap.insert(old_idx, ni);
                    ni
                };
                cube.texture[face] = new_idx;
            }
        });
    }

    /// Compact VSlots: remove unreferenced VSlots and renumber.
    /// Port of C++'s `compactvslots()`.
    pub fn compact_vslots(&mut self, world: &mut crate::octree::OctreeWorld) {
        let num = self.vslots.len();
        if num == 0 { return; }

        // Mark which vslots are referenced
        let mut used = vec![false; num];
        // Always keep slot 0 and 1 base variants
        if num > 0 { used[0] = true; }
        if num > 1 { used[1] = true; }

        // Walk the octree and mark referenced vslots
        world.for_each_leaf(|cube, _, _| {
            for face in 0..6 {
                let idx = cube.texture[face] as usize;
                if idx < num { used[idx] = true; }
            }
        });

        // Build remap table
        let mut remap = vec![0u16; num];
        let mut new_idx = 0u16;
        for i in 0..num {
            if used[i] {
                remap[i] = new_idx;
                new_idx += 1;
            }
        }

        // Remap cube texture indices
        world.for_each_leaf_mut(|cube, _, _| {
            for face in 0..6 {
                let idx = cube.texture[face] as usize;
                if idx < num {
                    cube.texture[face] = remap[idx];
                }
            }
        });

        // Compact the vslots array
        let mut new_vslots = Vec::with_capacity(new_idx as usize);
        for (i, vs) in self.vslots.drain(..).enumerate() {
            if i < num && used[i] {
                let mut vs = vs;
                vs.index = new_vslots.len();
                new_vslots.push(vs);
            }
        }
        self.vslots = new_vslots;

        // Rebuild slot variant lists
        for slot in &mut self.slots {
            slot.variants.clear();
        }
        for (i, vs) in self.vslots.iter().enumerate() {
            if vs.slot < self.slots.len() {
                self.slots[vs.slot].variants.push(i);
            }
        }
    }

    /// Get the number of texture slots (not vslots).
    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }

    /// Get the number of vslots.
    pub fn num_vslots(&self) -> usize {
        self.vslots.len()
    }
}

/// Check if two VSlots have identical parameters.
fn vslots_match(a: &VSlot, b: &VSlot) -> bool {
    a.slot == b.slot
        && a.scale == b.scale
        && a.rotation == b.rotation
        && a.offset == b.offset
        && a.scroll == b.scroll
        && a.layer == b.layer
        && a.alpha_front == b.alpha_front
        && a.alpha_back == b.alpha_back
        && a.color_scale == b.color_scale
        && a.glow_color == b.glow_color
        && a.params.len() == b.params.len()
        && a.params.iter().zip(b.params.iter()).all(|(pa, pb)| {
            pa.name == pb.name && pa.val == pb.val
        })
}

// ── UV generation ────────────────────────────────────────────────────────────
// Port of C++'s calctexgen() from octarender.cpp.

/// Compute UV texgen planes for a face, given VSlot transforms and texture dimensions.
///
/// Returns (s_gen, t_gen) where UV = (dot(pos, s_gen.xyz) + s_gen.w, dot(pos, t_gen.xyz) + t_gen.w)
///
/// Port of C++'s `calctexgen(VSlot &vslot, int dim, vec4 &sgen, vec4 &tgen)`.
pub fn calc_texgen(vslot: &VSlot, dim: usize, tex_w: u32, tex_h: u32) -> ([f32; 4], [f32; 4]) {
    let r = &TEX_ROTATIONS[vslot.rotation.rem_euclid(8) as usize];
    let k = TEX_SCALE / vslot.scale;

    let xs = if r.flip_x { -(tex_w as f32) } else { tex_w as f32 };
    let ys = if r.flip_y { -(tex_h as f32) } else { tex_h as f32 };

    let sk = k / xs;
    let tk = k / ys;

    let soff = -(if r.swap_xy { vslot.offset[1] } else { vslot.offset[0] }) as f32 / xs;
    let toff = -(if r.swap_xy { vslot.offset[0] } else { vslot.offset[1] }) as f32 / ys;

    // C++ axis mapping:
    // dim 0 (X-facing): s=Y(1), t=Z(2)
    // dim 1 (Y-facing): s=X(0), t=Z(2)
    // dim 2 (Z-facing): s=X(0), t=Y(1)
    const SI: [usize; 3] = [1, 0, 0];
    const TI: [usize; 3] = [2, 2, 1];
    let sdim = SI[dim];
    let tdim = TI[dim];

    let mut sgen = [0.0f32; 4];
    let mut tgen = [0.0f32; 4];
    sgen[3] = soff;
    tgen[3] = toff;

    if r.swap_xy {
        sgen[tdim] = if dim <= 1 { -sk } else { sk };
        tgen[sdim] = tk;
    } else {
        sgen[sdim] = sk;
        tgen[tdim] = if dim <= 1 { -tk } else { tk };
    }

    (sgen, tgen)
}

/// Compute UV for a single vertex position given texgen planes.
pub fn apply_texgen(pos: [f32; 3], sgen: &[f32; 4], tgen: &[f32; 4]) -> [f32; 2] {
    let u = pos[0] * sgen[0] + pos[1] * sgen[1] + pos[2] * sgen[2] + sgen[3];
    let v = pos[0] * tgen[0] + pos[1] * tgen[1] + pos[2] * tgen[2] + tgen[3];
    [u, v]
}

// ── Texture config file parsing ──────────────────────────────────────────────

/// Load texture slot definitions from a simple text config file.
///
/// Format (simplified from Cube2's .cfg):
/// ```text
/// shader bumpspecmapworld
/// texture 0 "path/to/diffuse.png"
/// texture n "path/to/normal.png"
/// texture s "path/to/spec.png"
/// texrotate 1
/// texscale 0.5
///
/// shader stdworld
/// texture 0 "path/to/another.png"
/// ```
///
/// Lines starting with `texture 0` or `texture c` create a new Slot.
/// Other `texture` lines add sub-textures to the current Slot.
/// `shader`, `texrotate`, `texscale`, `texoffset`, `texscroll`, `texcolor`,
/// `texalpha` set properties on the current Slot's base VSlot.
pub fn load_texture_config(registry: &mut TextureRegistry, text: &str) {
    let mut current_shader = "stdworld".to_string();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") { continue; }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() { continue; }

        match parts[0] {
            "shader" | "setshader" => {
                if parts.len() >= 2 {
                    current_shader = parts[1].trim_matches('"').to_string();
                }
            }
            "setshaderparam" => {
                if parts.len() >= 3 {
                    let name = parts[1].trim_matches('"').to_string();
                    let x: f32 = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    let y: f32 = parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    let z: f32 = parts.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    let w: f32 = parts.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                    // Apply to last slot's base vslot
                    if let Some(slot) = registry.slots.last() {
                        if let Some(&vi) = slot.variants.first() {
                            if vi < registry.vslots.len() {
                                registry.vslots[vi].params.push(SlotShaderParam {
                                    name,
                                    val: [x, y, z, w],
                                });
                                registry.vslots[vi].changed |= VSLOT_SHPARAM;
                            }
                        }
                    }
                }
            }
            "texture" => {
                if parts.len() >= 3 {
                    let tex_type_str = parts[1].trim_matches('"');
                    let mut path = parts[2].trim_matches('"').to_string();
                    // Strip Cube2 <dds> / <premul> / <luma> prefixes
                    while path.starts_with('<') {
                        if let Some(end) = path.find('>') {
                            path = path[end + 1..].to_string();
                        } else {
                            break;
                        }
                    }

                    let tex_type = match tex_type_str {
                        "0" | "c" => TexType::Diffuse,
                        "1" | "u" => TexType::Unknown,
                        "d" => TexType::Decal,
                        "n" => TexType::Normal,
                        "g" => TexType::Glow,
                        "s" => TexType::Spec,
                        "z" => TexType::Depth,
                        "a" => TexType::Alpha,
                        "e" => TexType::Envmap,
                        _ => TexType::Unknown,
                    };

                    if tex_type == TexType::Diffuse {
                        // New slot
                        registry.add_slot_with_tex(&current_shader, tex_type, &path);
                    } else {
                        // Sub-texture for current slot
                        registry.add_subtex_to_last(tex_type, &path);
                    }
                }
            }
            "texrotate" => {
                if parts.len() >= 2 {
                    if let Ok(rot) = parts[1].parse::<i32>() {
                        if let Some(slot) = registry.slots.last() {
                            if let Some(&vi) = slot.variants.first() {
                                if vi < registry.vslots.len() {
                                    registry.vslots[vi].rotation = rot.rem_euclid(8);
                                    registry.vslots[vi].changed |= VSLOT_ROTATION;
                                }
                            }
                        }
                    }
                }
            }
            "texscale" => {
                if parts.len() >= 2 {
                    if let Ok(s) = parts[1].parse::<f32>() {
                        if let Some(slot) = registry.slots.last() {
                            if let Some(&vi) = slot.variants.first() {
                                if vi < registry.vslots.len() {
                                    registry.vslots[vi].scale = s.clamp(0.125, 8.0);
                                    registry.vslots[vi].changed |= VSLOT_SCALE;
                                }
                            }
                        }
                    }
                }
            }
            "texoffset" => {
                if parts.len() >= 3 {
                    let x: i32 = parts[1].parse().unwrap_or(0);
                    let y: i32 = parts[2].parse().unwrap_or(0);
                    if let Some(slot) = registry.slots.last() {
                        if let Some(&vi) = slot.variants.first() {
                            if vi < registry.vslots.len() {
                                registry.vslots[vi].offset = [x, y];
                                registry.vslots[vi].changed |= VSLOT_OFFSET;
                            }
                        }
                    }
                }
            }
            "texscroll" => {
                if parts.len() >= 3 {
                    let s: f32 = parts[1].parse().unwrap_or(0.0);
                    let t: f32 = parts[2].parse().unwrap_or(0.0);
                    if let Some(slot) = registry.slots.last() {
                        if let Some(&vi) = slot.variants.first() {
                            if vi < registry.vslots.len() {
                                registry.vslots[vi].scroll = [s / 1000.0, t / 1000.0];
                                registry.vslots[vi].changed |= VSLOT_SCROLL;
                            }
                        }
                    }
                }
            }
            "texcolor" => {
                if parts.len() >= 4 {
                    let r: f32 = parts[1].parse().unwrap_or(1.0);
                    let g: f32 = parts[2].parse().unwrap_or(1.0);
                    let b: f32 = parts[3].parse().unwrap_or(1.0);
                    if let Some(slot) = registry.slots.last() {
                        if let Some(&vi) = slot.variants.first() {
                            if vi < registry.vslots.len() {
                                registry.vslots[vi].color_scale = [r, g, b];
                                registry.vslots[vi].changed |= VSLOT_COLOR;
                            }
                        }
                    }
                }
            }
            "texalpha" => {
                if parts.len() >= 3 {
                    let f: f32 = parts[1].parse().unwrap_or(0.5);
                    let b: f32 = parts[2].parse().unwrap_or(0.0);
                    if let Some(slot) = registry.slots.last() {
                        if let Some(&vi) = slot.variants.first() {
                            if vi < registry.vslots.len() {
                                registry.vslots[vi].alpha_front = f;
                                registry.vslots[vi].alpha_back = b;
                                registry.vslots[vi].changed |= VSLOT_ALPHA;
                            }
                        }
                    }
                }
            }
            _ => {} // ignore unknown commands
        }
    }
}
