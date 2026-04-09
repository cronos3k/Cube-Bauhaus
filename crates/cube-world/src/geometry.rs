//! Mesh generation from the octree.
//!
//! Faithfully ports Sauerbraten's rendering pipeline:
//!   visibletris()  → per-face visibility bitmask
//!   gencubeverts() → vertex emission with order flip for concave quads
//!
//! The Y↔Z coordinate swap (Cube2 Z-up → renderer Y-up) reverses triangle
//! winding, so we emit indices in reversed order to keep front faces correct.

use glam::Vec3;
use crate::octree::{OctreeWorld, Cube, FACE_DIM, FACEEDGESIDX, F_SOLID, R, C};
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

// ── 2D face-plane polygon types and occlusion helpers ────────────────────────
// Port of Cube2's facevec / genfacevecs / insideface / occludesface from octa.cpp.

/// 2D integer vector in the face plane (matches Cube2 `facevec`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct FaceVec {
    x: i32,
    y: i32,
}

/// Project the face vertices of `cube` at orientation `orient` into the 2D face
/// plane, producing up to 4 `FaceVec`s.  Returns the number of valid vertices.
///
/// Port of Cube2 `genfacevecs(cube, orient, pos, size, solid, fvecs, v)`.
/// The 2D axes are C[dim] → x, R[dim] → y (matching the C++ convention).
/// Coordinates are in "world×8" units: `vertex_component * size + pos_component << 3`.
fn genfacevecs_2d(
    cube: &Cube,
    orient: usize,
    pos: [i32; 3],
    size: i32,
    solid: bool,
    fvecs: &mut [FaceVec; 4],
) -> usize {
    let dim = orient >> 1;
    let coord = orient & 1;
    let ca = C[dim]; // column axis index
    let ra = R[dim]; // row axis index

    if solid {
        // Solid cube: vertices are the full face rectangle.
        // Match C++ `GENFACEVERTS(pos.x, pos.x+size, ...)` solid branch.
        let c0 = pos[ca] << 3;
        let c1 = (pos[ca] + size) << 3;
        let r0 = pos[ra] << 3;
        let r1 = (pos[ra] + size) << 3;
        // Vertex order depends on orient (matches C++ GENFACEORIENT dimcoord branch).
        // For dimcoord=1: v0,v1,v2,v3 in one order; dimcoord=0: reversed.
        if coord != 0 {
            // dimcoord=1: (c1,r1), (c0,r1), (c0,r0), (c1,r0)
            fvecs[0] = FaceVec { x: c1, y: r1 };
            fvecs[1] = FaceVec { x: c0, y: r1 };
            fvecs[2] = FaceVec { x: c0, y: r0 };
            fvecs[3] = FaceVec { x: c1, y: r0 };
        } else {
            // dimcoord=0: (c1,r0), (c0,r0), (c0,r1), (c1,r1)
            fvecs[0] = FaceVec { x: c1, y: r0 };
            fvecs[1] = FaceVec { x: c0, y: r0 };
            fvecs[2] = FaceVec { x: c0, y: r1 };
            fvecs[3] = FaceVec { x: c1, y: r1 };
        }
        return 4;
    }

    // Non-solid: project each of the 4 face corners.
    // Use genfaceverts to get 3D corners (0–8), then project to 2D.
    let v3d = genfaceverts(cube, orient);

    let mut count = 0usize;
    let sentinel = FaceVec { x: i32::MAX, y: i32::MAX };
    let mut prev = sentinel;

    for vi in 0..4 {
        let e = v3d[vi];
        // The depth component must touch the face boundary
        if e[dim as usize] != (coord as i32) * 8 {
            continue;
        }
        let f = FaceVec {
            x: e[ca] as i32 * size + (pos[ca] << 3),
            y: e[ra] as i32 * size + (pos[ra] << 3),
        };
        if f != prev {
            fvecs[count] = f;
            prev = f;
            count += 1;
        }
    }
    // Deduplicate wrap-around (first == last)
    if count > 0 && fvecs[0] == prev {
        count -= 1;
    }
    count
}

/// Port of Cube2 `insideface()` — test if polygon `p` (nump vertices) is entirely
/// inside polygon `o` (numo vertices) using 2D half-plane tests.
///
/// Each edge of `o` defines a half-plane; every vertex of `p` must be on the
/// inside (≤ offset) of ALL edges.  Returns true only if `o` has ≥ 3 edges
/// and all tests pass.
fn insideface(p: &[FaceVec], nump: usize, o: &[FaceVec], numo: usize) -> bool {
    let mut bounds = 0;
    let mut prev = o[numo - 1];
    for i in 0..numo {
        let cur = o[i];
        let dx = cur.x - prev.x;
        let dy = cur.y - prev.y;
        let offset = (dx as i64) * (prev.y as i64) - (dy as i64) * (prev.x as i64);
        for j in 0..nump {
            if (dx as i64) * (p[j].y as i64) - (dy as i64) * (p[j].x as i64) > offset {
                return false;
            }
        }
        bounds += 1;
        prev = cur;
    }
    bounds >= 3
}

/// Port of Cube2 `clipfacevecy()`.
fn clipfacevecy(o: &FaceVec, dir: &FaceVec, cx: i32, cy: i32, size: i32, r: &mut FaceVec) -> usize {
    if dir.x >= 0 {
        if cx <= o.x || cx >= o.x + dir.x { return 0; }
    } else if cx <= o.x + dir.x || cx >= o.x {
        return 0;
    }
    let t = (o.y - cy) + (cx - o.x) * dir.y / dir.x;
    if t <= 0 || t >= size { return 0; }
    r.x = cx;
    r.y = cy + t;
    1
}

/// Port of Cube2 `clipfacevecx()`.
fn clipfacevecx(o: &FaceVec, dir: &FaceVec, cx: i32, cy: i32, size: i32, r: &mut FaceVec) -> usize {
    if dir.y >= 0 {
        if cy <= o.y || cy >= o.y + dir.y { return 0; }
    } else if cy <= o.y + dir.y || cy >= o.y {
        return 0;
    }
    let t = (o.x - cx) + (cy - o.y) * dir.x / dir.y;
    if t <= 0 || t >= size { return 0; }
    r.x = cx + t;
    r.y = cy;
    1
}

/// Port of Cube2 `clipfacevec()`.
fn clipfacevec(o: &FaceVec, dir: &FaceVec, cx: i32, cy: i32, size: i32, rvecs: &mut [FaceVec]) -> usize {
    let mut r = 0usize;
    if o.x >= cx && o.x <= cx + size
        && o.y >= cy && o.y <= cy + size
        && ((o.x != cx && o.x != cx + size) || (o.y != cy && o.y != cy + size))
    {
        rvecs[0] = *o;
        r += 1;
    }
    r += clipfacevecx(o, dir, cx, cy, size, &mut rvecs[r]);
    r += clipfacevecx(o, dir, cx, cy + size, size, &mut rvecs[r]);
    r += clipfacevecy(o, dir, cx, cy, size, &mut rvecs[r]);
    r += clipfacevecy(o, dir, cx + size, cy, size, &mut rvecs[r]);
    debug_assert!(r <= 2);
    r
}

/// Port of Cube2 `clipfacevecs()` — clip polygon `o` against the axis-aligned
/// rectangle at (cx, cy) with given size (all in cube-local units, scaled <<3).
fn clipfacevecs(o: &[FaceVec], numo: usize, cx: i32, cy: i32, size: i32, rvecs: &mut [FaceVec; 8]) -> usize {
    let cx = cx << 3;
    let cy = cy << 3;
    let size = size << 3;

    let mut r = 0usize;
    let mut prev = o[numo - 1];
    for i in 0..numo {
        let cur = o[i];
        let dir = FaceVec { x: cur.x - prev.x, y: cur.y - prev.y };
        r += clipfacevec(&prev, &dir, cx, cy, size, &mut rvecs[r..]);
        prev = cur;
    }
    // Check if rectangle corners are inside the polygon
    let corners = [
        FaceVec { x: cx, y: cy },
        FaceVec { x: cx + size, y: cy },
        FaceVec { x: cx + size, y: cy + size },
        FaceVec { x: cx, y: cy + size },
    ];
    for corner in &corners {
        if insideface(std::slice::from_ref(corner), 1, o, numo) {
            rvecs[r] = *corner;
            r += 1;
        }
    }
    debug_assert!(r <= 8);
    r
}

/// Port of Cube2 `occludesface()` — recursively check if cube `c` (neighbor)
/// fully occludes the face polygon `vf` (numv vertices in 2D face-plane coords).
///
/// `orient` is the orientation from the NEIGHBOR's perspective (opposite of the
/// original face).  `o` is the neighbor origin, `size` is the neighbor size.
fn occludesface(
    c: &Cube,
    orient: usize,
    o: [i32; 3],
    size: i32,
    vf: &[FaceVec],
    numv: usize,
) -> bool {
    let dim = orient >> 1;
    let coord = orient & 1;

    if c.children.is_none() {
        // Leaf node
        if c.is_solid() {
            return true;
        }
        if c.touching_face(orient) && c.face_edges(orient) == F_SOLID {
            return true;
        }
        // Clip the source polygon against this leaf's rectangle in the face plane.
        let mut cf = [FaceVec { x: 0, y: 0 }; 8];
        let numc = clipfacevecs(vf, numv, o[C[dim]], o[R[dim]], size, &mut cf);
        if numc < 3 {
            return true; // degenerate clipped polygon → fully covered
        }
        if c.is_empty() || c.not_touching_face(orient) {
            return false;
        }
        // Generate the neighbor's face polygon and test containment
        let mut of = [FaceVec { x: 0, y: 0 }; 4];
        let numo = genfacevecs_2d(c, orient, o, size, false, &mut of);
        return numo >= 3 && insideface(&cf[..numc], numc, &of[..numo], numo);
    }

    // Branch node: recurse into the 4 children on the shared face.
    let half = size >> 1;
    let children = c.children.as_ref().unwrap();
    for i in 0..8u8 {
        // octacoord(dim, i) = (i >> dim) & 1
        if ((i >> dim) & 1) == coord as u8 {
            // Compute child origin: ivec(i, o.x, o.y, o.z, size)
            // In C++: ivec(i, ox, oy, oz, size) = (ox + (i&1)*size, oy + ((i>>1)&1)*size, oz + ((i>>2)&1)*size)
            let co = [
                o[0] + ((i & 1) as i32) * half,
                o[1] + (((i >> 1) & 1) as i32) * half,
                o[2] + (((i >> 2) & 1) as i32) * half,
            ];
            if !occludesface(&children[i as usize], orient, co, half, vf, numv) {
                return false;
            }
        }
    }
    true
}

/// Port of Cube2 `visibletris()` — returns visibility bitmask:
///   bit 0 = triangle 1 visible
///   bit 1 = triangle 2 visible
///   bit 2 = order flip (use order=1)
///
/// Full port including insideface/occludesface neighbor occlusion and the
/// per-triangle retry loop (C++ lines 1061–1149).
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

    let mut order: usize = if convex < 0 { 1 } else { 0 };
    let notouch = NOTOUCHMASKS[order][touching as usize];

    // If all visible triangles are "not touching", they're interior faces — always visible.
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

    // Outside world → face is at world boundary, not visible
    if nx < 0 || nx >= ws || ny < 0 || ny >= ws || nz < 0 || nz >= ws {
        return 0;
    }

    let (neighbor, norigin, nsize) = world.lookup(nx, ny, nz);
    let opp = orient ^ 1;

    // Mask origins to 12 bits (matching C++ vo.mask(0xFFF) / no.mask(0xFFF))
    let vo = [ox & 0xFFF, oy & 0xFFF, oz & 0xFFF];
    let no = [norigin.0 & 0xFFF, norigin.1 & 0xFFF, norigin.2 & 0xFFF];

    // Generate the source face polygon in 2D face-plane coordinates
    let mut cf = [FaceVec { x: 0, y: 0 }; 4];
    let mut of = [FaceVec { x: 0, y: 0 }; 4];
    let mut numo: usize = 0;
    let numc: usize;

    if nsize > size || (nsize == size && neighbor.children.is_none()) {
        // Same-size or larger neighbor leaf
        if neighbor.is_empty() || neighbor.not_touching_face(opp) {
            return vis;
        }
        if neighbor.is_solid() || (neighbor.touching_face(opp) && neighbor.face_edges(opp) == F_SOLID) {
            return vis & notouch;
        }

        numc = genfacevecs_2d(cube, orient, vo, size, false, &mut cf);
        numo = genfacevecs_2d(neighbor, opp, no, nsize, false, &mut of);
        if numo < 3 { return vis; }
        if insideface(&cf[..numc], numc, &of[..numo], numo) {
            return vis & notouch;
        }
    } else {
        // Smaller neighbor (has children) — use occludesface
        numc = genfacevecs_2d(cube, orient, vo, size, false, &mut cf);
        if occludesface(neighbor, opp, no, nsize, &cf[..numc], numc) {
            return vis & notouch;
        }
    }

    // ── Per-triangle retry loop (C++ lines 1121–1148) ──
    // If the whole face isn't occluded, check individual triangles.
    if vis != 3 || notouch != 0 { return vis; }

    // C++ triverts[order][coord][tri_index][vert] — which of the 4 cf[] vertices
    // form each triangle, indexed by [order][coord][triangle].
    const TRIVERTS: [[[[usize; 3]; 2]; 2]; 2] = [
        // order 0
        [
            // coord 0
            [ [1, 2, 3], [0, 1, 3] ],
            // coord 1
            [ [0, 1, 2], [0, 2, 3] ],
        ],
        // order 1
        [
            // coord 0
            [ [0, 1, 2], [3, 0, 2] ],
            // coord 1
            [ [1, 2, 3], [1, 3, 0] ],
        ],
    ];

    loop {
        for i in 0..2usize {
            let verts = &TRIVERTS[order][coord][i];
            let tf = [cf[verts[0]], cf[verts[1]], cf[verts[2]]];
            let occluded = if numo > 0 {
                insideface(&tf, 3, &of[..numo], numo)
            } else {
                occludesface(neighbor, opp, no, nsize, &tf, 3)
            };
            if !occluded { continue; }
            return vis & !(1u8 << i);
        }
        vis |= 4;
        order += 1;
        if order > 1 { break; }
    }

    3
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

            // Build vertex color: rgb = tint, a = texture layer index (or -1 if no textures)
            let color = if let Some(reg) = registry {
                let vs = reg.lookup_vslot(tex_idx);
                let slot = reg.slot_for_vslot(vs);
                if slot.loaded {
                    // Real texture: white tint * VSlot color_scale, layer = diffuse layer
                    let layer = slot.textures.first().map_or(0, |t| t.layer) as f32;
                    [vs.color_scale[0], vs.color_scale[1], vs.color_scale[2], layer]
                } else {
                    // Slot exists but not loaded — debug color, no texture sampling
                    face_debug_color(orient, tex_idx)
                }
            } else {
                face_debug_color(orient, tex_idx)
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
