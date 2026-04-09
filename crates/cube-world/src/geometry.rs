//! Mesh generation from the octree.
//!
//! Faithfully ports Sauerbraten's rendering pipeline:
//!   visibletris()  → per-face visibility bitmask
//!   gencubeverts() → vertex emission with order flip for concave quads
//!
//! The Y↔Z coordinate swap (Cube2 Z-up → renderer Y-up) reverses triangle
//! winding, so we emit indices in reversed order to keep front faces correct.

use glam::Vec3;
use crate::octree::{OctreeWorld, Cube, FACE_DIM, FACEEDGESIDX, F_SOLID};
use crate::texture::{TextureRegistry, calc_texgen, apply_texgen};

/// A single vertex produced by the mesh builder.
/// Matches the layout of `bbc_renderer::Vertex` (48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal:   [f32; 3],
    pub uv:       [f32; 2],
    pub color:    [f32; 4],
}

// ── Cube2-faithful face visibility helpers ────────────────────────────────────
// These delegate to methods on Cube (defined in octree.rs).
// Kept as free functions for readability in the visibility pipeline.

/// From Cube2 `collapsedface()` (octa.cpp:964): check if all 4 face vertices
/// are coplanar (face has zero area). Uses two cross products.
fn collapsedface(cube: &Cube, orient: usize) -> bool {
    let e0 = cube.edges[FACEEDGESIDX[orient][0]] as i32;
    let e1 = cube.edges[FACEEDGESIDX[orient][1]] as i32;
    let e2 = cube.edges[FACEEDGESIDX[orient][2]] as i32;
    let e3 = cube.edges[FACEEDGESIDX[orient][3]] as i32;

    let dim = orient >> 1;
    let face_base = dim * 4;
    let mut f0 = cube.edges[face_base] as i32;
    let mut f1 = cube.edges[face_base + 1] as i32;
    let mut f2 = cube.edges[face_base + 2] as i32;
    let mut f3 = cube.edges[face_base + 3] as i32;

    if (orient & 1) != 0 {
        f0 >>= 4; f1 >>= 4; f2 >>= 4; f3 >>= 4;
    } else {
        f0 &= 0xF; f1 &= 0xF; f2 &= 0xF; f3 &= 0xF;
    }

    let v0 = [e0 & 0xF, e2 & 0xF, f0];
    let v1 = [e0 >> 4,  e3 & 0xF, f1];
    let v2 = [e1 >> 4,  e3 >> 4,  f3];
    let v3 = [e1 & 0xF, e2 >> 4,  f2];

    let d1 = sub_i32(v1, v0);
    let d2 = sub_i32(v2, v0);
    let d3 = sub_i32(v3, v0);

    let c1 = cross_i32(d1, d2);
    let c2 = cross_i32(d2, d3);
    is_zero(c1) && is_zero(c2)
}

// ── visibletris — faithful port from Cube2 octa.cpp ──────────────────────────

/// Generate the 4 face vertices in Cube2 local coordinates (0–8).
/// Matches Cube2's `genfaceverts()` / fv[6][4] exactly.
fn genfaceverts(cube: &Cube, orient: usize) -> [[i32; 3]; 4] {
    let c = cube.face_corners(orient);
    [
        [c[0][0] as i32, c[0][1] as i32, c[0][2] as i32],
        [c[1][0] as i32, c[1][1] as i32, c[1][2] as i32],
        [c[2][0] as i32, c[2][1] as i32, c[2][2] as i32],
        [c[3][0] as i32, c[3][1] as i32, c[3][2] as i32],
    ]
}

/// Cross product of two i32 vectors.
#[inline]
fn cross_i32(a: [i32; 3], b: [i32; 3]) -> [i32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Dot product of two i32 vectors.
#[inline]
fn dot_i32(a: [i32; 3], b: [i32; 3]) -> i32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn sub_i32(a: [i32; 3], b: [i32; 3]) -> [i32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn is_zero(v: [i32; 3]) -> bool {
    v[0] == 0 && v[1] == 0 && v[2] == 0
}

/// Cube2 notouchmasks[order][touching] — determines which triangles don't need
/// neighbor checking based on which vertices touch the cube boundary.
const NOTOUCHMASKS: [[u8; 16]; 2] = [
    // order 0: flat or convex
    [3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 1, 3, 0],
    // order 1: concave
    [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 2, 0],
];

/// Port of Cube2 `visibletris()` — returns visibility bitmask:
///   bit 0 = triangle 1 visible
///   bit 1 = triangle 2 visible
///   bit 2 = order flip (use order=1)
///
/// Does NOT do full insideface/occludesface neighbor occlusion — uses simplified
/// neighbor check (conservative: shows face when unsure).
fn visibletris(
    cube: &Cube,
    orient: usize,
    world: &OctreeWorld,
    ox: i32, oy: i32, oz: i32,
    size: i32,
) -> u8 {
    let v = genfaceverts(cube, orient);

    // Compute face normal: n = (v[1]-v[0]) × (v[2]-v[0])
    let e1 = sub_i32(v[1], v[0]);
    let e2 = sub_i32(v[2], v[0]);
    let n = cross_i32(e1, e2);

    // Convexity: (v[0]-v[3]) · n
    let e3 = sub_i32(v[0], v[3]);
    let convex = dot_i32(e3, n);

    let mut vis: u8 = 3; // both triangles visible
    let mut touching: u8 = 0xF; // all 4 vertices assumed touching

    if convex == 0 {
        // Flat or degenerate quad
        if is_zero(cross_i32(e3, e2)) || v[1] == v[3] {
            // Triangle 2 is degenerate (v3 collapsed onto edge v0-v2 or v1==v3)
            if is_zero(n) { return 0; } // entire face is degenerate
            vis = 1; // only triangle 1
            touching &= !(1u8 << 3);
        } else if is_zero(n) {
            // Triangle 1 is degenerate but triangle 2 is not
            vis = 2; // only triangle 2
            touching &= !(1u8 << 1);
        }
    }

    // Check which vertices actually touch the cube boundary
    let dim = orient >> 1; // dimension(orient)
    let coord = orient & 1; // dimcoord(orient)
    let target = (coord as i32) * 8;
    if v[0][dim] != target { touching &= !(1u8 << 0); }
    if v[1][dim] != target { touching &= !(1u8 << 1); }
    if v[2][dim] != target { touching &= !(1u8 << 2); }
    if v[3][dim] != target { touching &= !(1u8 << 3); }

    let order: usize = if convex < 0 { 1 } else { 0 };
    let notouch = NOTOUCHMASKS[order][touching as usize];

    // If all visible triangles are "not touching", they're interior faces — always visible.
    // C++ line 1120: returns vis WITHOUT bit 2.
    if (vis & notouch) == vis {
        return vis;
    }

    // ── Neighbor occlusion check ──
    let ws = world.world_size();

    const FACE_STEP: [[i32; 3]; 6] = [
        [-1, 0, 0], [1, 0, 0],
        [0, -1, 0], [0, 1, 0],
        [0, 0, -1], [0, 0, 1],
    ];

    let nd = FACE_STEP[orient];
    let nx = ox + nd[0] * size;
    let ny = oy + nd[1] * size;
    let nz = oz + nd[2] * size;

    // Outside world → not visible
    if nx < 0 || nx >= ws || ny < 0 || ny >= ws || nz < 0 || nz >= ws {
        return 0;
    }

    let (neighbor, _norigin, nsize) = world.lookup(nx, ny, nz);
    let opp = orient ^ 1;

    if nsize >= size && neighbor.children.is_none() {
        if neighbor.is_empty() || neighbor.not_touching_face(opp) {
            // C++ line 1135: returns vis WITHOUT bit 2
            return vis;
        }
        if neighbor.is_solid() || (neighbor.touching_face(opp) && neighbor.face_edges(opp) == F_SOLID) {
            // C++ line 1137: returns vis&notouch WITHOUT bit 2
            return vis & notouch;
        }
        // C++ line 1142: returns vis WITHOUT bit 2
        return vis;
    }

    // Smaller neighbors — conservative: show the face.
    // Bit 2 (order flip) is ONLY set in C++'s retry loop (line 1173),
    // which we don't implement. So never set it here.
    vis
}

/// Build a renderable triangle mesh from the entire world.
///
/// Faithfully ports Cube2's gencubeverts() vertex emission logic:
/// - Uses visibletris() for per-triangle visibility
/// - Handles order flip for concave quads
/// - Reverses triangle winding to compensate for Y↔Z coordinate swap
pub fn build_mesh(world: &OctreeWorld) -> (Vec<MeshVertex>, Vec<u32>) {
    build_mesh_with_textures(world, None)
}

/// Build mesh with optional texture registry for proper UV computation.
/// When `registry` is Some, UVs are computed via Cube2-style planar projection
/// using VSlot rotation/scale/offset. When None, uses simple fallback UVs.
pub fn build_mesh_with_textures(
    world: &OctreeWorld,
    registry: Option<&TextureRegistry>,
) -> (Vec<MeshVertex>, Vec<u32>) {
    let mut verts:   Vec<MeshVertex> = Vec::new();
    let mut indices: Vec<u32>        = Vec::new();

    world.for_each_leaf(|cube, (ox, oy, oz), size| {
        if cube.is_empty() { return; }

        let scale = size as f32;

        for orient in 0..6usize {
            // Cube2 pre-filter: skip collapsed (zero-area) faces
            if collapsedface(cube, orient) { continue; }

            let vis = visibletris(cube, orient, world, ox, oy, oz, size);
            if vis == 0 { continue; }

            // ── Get face vertices (Cube2 local coords 0–8) ──
            let v = genfaceverts(cube, orient);

            // Order: bit 2 of vis determines diagonal split direction.
            let order: usize = if (vis & 4) != 0 { 1 } else { 0 };

            // ── Emit vertices per Cube2's gencubeverts logic ──
            let mut positions: Vec<Vec3> = Vec::with_capacity(4);
            let mut local_corners: Vec<[i32; 3]> = Vec::with_capacity(4);

            local_corners.push(v[order]);
            if vis & 1 != 0 { local_corners.push(v[order + 1]); }
            local_corners.push(v[order + 2]);
            if vis & 2 != 0 { local_corners.push(v[(order + 3) & 3]); }

            // Convert to world space with Y↔Z swap (Cube2 Z-up → renderer Y-up)
            // Also compute Cube2-space world positions (no swap) for UV generation
            let mut cube2_positions: Vec<[f32; 3]> = Vec::with_capacity(4);
            for c in &local_corners {
                let c2x = ox as f32 + c[0] as f32 * scale / 8.0;
                let c2y = oy as f32 + c[1] as f32 * scale / 8.0;
                let c2z = oz as f32 + c[2] as f32 * scale / 8.0;
                positions.push(Vec3::new(c2x, c2z, c2y)); // Y↔Z swap for renderer
                cube2_positions.push([c2x, c2y, c2z]);    // Cube2 space for UVs
            }

            let numverts = positions.len();
            if numverts < 3 { continue; }

            // Compute face normal from first triangle
            let d0 = positions[1] - positions[0];
            let d1 = positions[2] - positions[0];
            let cross = d1.cross(d0);
            let normal = if cross.length_squared() > 1e-10 {
                cross.normalize()
            } else {
                face_axis_normal(orient)
            };

            // Compute UVs: use VSlot texgen if registry available, else fallback
            let dim = FACE_DIM[orient];
            let tex_idx = cube.texture[orient];

            let (sgen, tgen) = if let Some(reg) = registry {
                let vs = reg.lookup_vslot(tex_idx);
                let slot = reg.slot_for_vslot(vs);
                let (tw, th) = if let Some(dtex) = slot.textures.first() {
                    (dtex.width, dtex.height)
                } else {
                    (256, 256)
                };
                calc_texgen(vs, dim, tw, th)
            } else {
                // Fallback: simple planar projection
                let mut sgen = [0.0f32; 4];
                let mut tgen = [0.0f32; 4];
                let si = [1usize, 0, 0];
                let ti = [2usize, 2, 1];
                sgen[si[dim]] = 1.0 / 8.0;
                tgen[ti[dim]] = 1.0 / 8.0;
                (sgen, tgen)
            };

            // Build vertex color: rgb = tint, a = texture layer index (or special material code)
            // Material codes (negative alpha): -2=glass, -3=water, -4=lava, -5=clip
            use crate::octree::{MAT_WATER, MAT_LAVA, MAT_GLASS, MAT_CLIP};
            let color = match cube.material {
                MAT_GLASS => [0.6, 0.8, 1.0, -2.0],  // light blue tint
                MAT_WATER => [0.2, 0.4, 0.9, -3.0],  // blue tint
                MAT_LAVA  => [1.0, 0.4, 0.1, -4.0],  // orange tint
                MAT_CLIP  => [1.0, 0.2, 0.2, -5.0],  // red tint
                _ => {
                    // Normal solid geometry
                    if let Some(reg) = registry {
                        let vs = reg.lookup_vslot(tex_idx);
                        let slot = reg.slot_for_vslot(vs);
                        if slot.loaded {
                            let layer = slot.textures.first().map_or(0, |t| t.layer) as f32;
                            [vs.color_scale[0], vs.color_scale[1], vs.color_scale[2], layer]
                        } else {
                            face_debug_color(orient, tex_idx)
                        }
                    } else {
                        face_debug_color(orient, tex_idx)
                    }
                }
            };

            // Emit vertices
            let base = verts.len() as u32;
            for i in 0..numverts {
                let uv = apply_texgen(cube2_positions[i], &sgen, &tgen);
                verts.push(MeshVertex {
                    position: positions[i].to_array(),
                    normal:   normal.to_array(),
                    uv,
                    color,
                });
            }

            // Triangle fan indices — REVERSED winding to compensate for Y↔Z swap.
            for i in 0..(numverts as u32 - 2) {
                indices.push(base);
                indices.push(base + i + 2);
                indices.push(base + i + 1);
            }
        }
    });

    (verts, indices)
}

/// Axis-aligned fallback normal for degenerate faces.
/// Cube2 axes: X=right, Y=forward, Z=up
/// Renderer:   X=right, Y=up,      Z=forward  (Y↔Z swap)
fn face_axis_normal(orient: usize) -> Vec3 {
    match orient {
        0 => Vec3::NEG_X,  // LEFT  (-X → -X)
        1 => Vec3::X,      // RIGHT (+X → +X)
        2 => Vec3::NEG_Z,  // BACK  (-Y → -Z)
        3 => Vec3::Z,      // FRONT (+Y → +Z)
        4 => Vec3::NEG_Y,  // BOTTOM(-Z → -Y)
        5 => Vec3::Y,      // TOP   (+Z → +Y)
        _ => Vec3::Y,
    }
}

/// Debug color per texture slot — generates visually distinct colors using
/// a golden-ratio hue distribution. Face orientation provides subtle brightness variation.
fn face_debug_color(orient: usize, tex_slot: u16) -> [f32; 4] {
    // Golden ratio hue distribution for maximum slot differentiation
    let slot = tex_slot as f32;
    let hue = (slot * 0.618033988749895) % 1.0;
    let sat = 0.4 + (slot * 0.137) % 0.3; // 0.4-0.7 saturation
    let val = 0.6 + ((orient as f32) * 0.04); // orient varies brightness slightly

    // HSV to RGB
    let h = hue * 6.0;
    let i = h.floor() as i32;
    let f = h - i as f32;
    let p = val * (1.0 - sat);
    let q = val * (1.0 - sat * f);
    let t = val * (1.0 - sat * (1.0 - f));
    let (r, g, b) = match i % 6 {
        0 => (val, t, p),
        1 => (q, val, p),
        2 => (p, val, t),
        3 => (p, q, val),
        4 => (t, p, val),
        _ => (val, p, q),
    };
    [r, g, b, -1.0] // -1.0 alpha = no texture loaded (shader uses checkerboard fallback)
}

/// Build a wireframe overlay mesh (edges of all cubes) for the selection grid.
pub fn build_wireframe(world: &OctreeWorld) -> (Vec<MeshVertex>, Vec<u32>) {
    let mut verts:   Vec<MeshVertex> = Vec::new();
    let mut indices: Vec<u32>        = Vec::new();

    let wire_color = [0.9f32, 0.9, 0.9, 0.5];

    world.for_each_leaf(|cube, (ox, oy, oz), size| {
        if cube.is_empty() { return; }
        let s = size as f32;

        // Generate 8 world-space corners for a box outline (Y↔Z swap)
        let corners: [Vec3; 8] = std::array::from_fn(|i| {
            let c = cube.corner(i);
            Vec3::new(
                ox as f32 + c[0] as f32 * s / 8.0,
                oz as f32 + c[2] as f32 * s / 8.0,
                oy as f32 + c[1] as f32 * s / 8.0,
            )
        });

        // 12 edges of a cube, as pairs of corner indices
        const EDGES: [(usize,usize); 12] = [
            (0,1),(2,3),(4,5),(6,7), // X edges
            (0,2),(1,3),(4,6),(5,7), // Y edges
            (0,4),(1,5),(2,6),(3,7), // Z edges
        ];

        let base = verts.len() as u32;
        for &p in corners.iter() {
            verts.push(MeshVertex {
                position: p.to_array(),
                normal:   [0.0, 1.0, 0.0],
                uv:       [0.0, 0.0],
                color:    wire_color,
            });
        }
        for (a, b) in EDGES {
            indices.push(base + a as u32);
            indices.push(base + b as u32);
        }
    });

    (verts, indices)
}
