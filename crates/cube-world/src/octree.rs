//! Core octree data structures for Cube2/Sauerbraten geometry.
//!
//! Key insight: NOT voxels.  Each leaf cube has 12 edge bytes, each storing
//! two 4-bit endpoint values (0–8).  This allows per-corner deformation while
//! staying compact and hierarchical.

// ── Material constants ─────────────────────────────────────────────────────────
pub const MAT_AIR:   u8 = 0x00;
pub const MAT_WATER: u8 = 0x04;
pub const MAT_LAVA:  u8 = 0x08;
pub const MAT_GLASS: u8 = 0x10;
pub const MAT_CLIP:  u8 = 0x20;
pub const MAT_ALPHA: u8 = 0x80;

// ── Edge encoding helpers ──────────────────────────────────────────────────────

/// Extract a 4-bit endpoint from an edge byte.
/// `coord=0` → low nibble (start), `coord=1` → high nibble (end).
#[inline]
pub fn edge_get(edge: u8, coord: usize) -> u8 {
    if coord != 0 { edge >> 4 } else { edge & 0xF }
}

/// Store a 4-bit endpoint back into an edge byte.
#[inline]
pub fn edge_set(edge: &mut u8, coord: usize, val: u8) {
    if coord != 0 {
        *edge = (*edge & 0x0F) | ((val & 0xF) << 4);
    } else {
        *edge = (*edge & 0xF0) | (val & 0xF);
    }
}

/// `cubeedge(c, dim, x, y)` → index into `cube.edges[dim<<2 | y<<1 | x]`
#[inline]
pub const fn edge_idx(dim: usize, x: usize, y: usize) -> usize {
    (dim << 2) | (y << 1) | x
}

// ── Cube2 coordinate mapping tables ───────────────────────────────────────────
// From geom.h: R[d]=row axis, C[d]=col axis, D[d]=depth axis for dimension d.
// Used by getcubevector/setcubevector, subdividecube, octaindex, etc.

/// Row axis for each dimension: R[0]=Y, R[1]=Z, R[2]=X
pub const R: [usize; 3] = [1, 2, 0];
/// Column axis for each dimension: C[0]=Z, C[1]=X, C[2]=Y
pub const C: [usize; 3] = [2, 0, 1];
/// Depth axis for each dimension: D[0]=X, D[1]=Y, D[2]=Z
pub const D: [usize; 3] = [0, 1, 2];

/// `octaindex(d, x, y, z)` — compute child index from face-oriented coordinates.
/// Maps (row, col, depth) in dimension `d` to a 3-bit child index.
#[inline]
pub const fn octaindex(d: usize, x: usize, y: usize, z: usize) -> usize {
    (z << D[d]) | (y << C[d]) | (x << R[d])
}

/// `octastep(x, y, z, scale)` — child index at a given octree scale.
#[inline]
pub fn octastep(x: i32, y: i32, z: i32, scale: u32) -> usize {
    ((((z >> scale) & 1) << 2) | (((y >> scale) & 1) << 1) | ((x >> scale) & 1)) as usize
}

/// `oppositeocta(d, i)` — the child index on the opposite side of dimension d.
#[inline]
pub const fn oppositeocta(d: usize, i: usize) -> usize {
    i ^ (1 << D[d])
}

// ── Face/corner constants ──────────────────────────────────────────────────────

/// Face value for a completely EMPTY face (all edge endpoints at 0).
pub const F_EMPTY: u32 = 0x00000000;
/// Face value for a completely SOLID face (all edge endpoints at 0→8).
pub const F_SOLID: u32 = 0x80808080;

/// Edge bytes for a completely EMPTY cube (all faces have 0 extent).
pub const EDGES_EMPTY: [u8; 12] = [0x00; 12];
/// Edge bytes for a completely SOLID cube (all edges span 0→8).
pub const EDGES_SOLID: [u8; 12] = [0x80; 12];

/// Face orientations (same numbering as Cube2 source).
pub const O_LEFT:   usize = 0; // -X
pub const O_RIGHT:  usize = 1; // +X
pub const O_BACK:   usize = 2; // -Y
pub const O_FRONT:  usize = 3; // +Y
pub const O_BOTTOM: usize = 4; // -Z
pub const O_TOP:    usize = 5; // +Z

/// Which dimension (0=X,1=Y,2=Z) each orientation belongs to.
pub const FACE_DIM: [usize; 6] = [0, 0, 1, 1, 2, 2];
/// Which side of the dimension (0=low, 1=high) each orientation is.
pub const FACE_SIDE: [usize; 6] = [0, 1, 0, 1, 0, 1];

/// Cube2 fv[6][4] — vertex indices per face, in GENCUBEVERTS numbering.
pub const FV: [[usize; 4]; 6] = [
    [2, 1, 6, 5], // LEFT   (-X)
    [3, 4, 7, 0], // RIGHT  (+X)
    [4, 5, 6, 7], // BACK   (-Y)
    [1, 2, 3, 0], // FRONT  (+Y)
    [6, 1, 0, 7], // BOTTOM (-Z)
    [5, 4, 3, 2], // TOP    (+Z)
];

/// Edge indices surrounding each face orientation.
/// [0..1] = row edges, [2..3] = column edges.
/// From Cube2 `faceedgesidx[6][4]`.
pub const FACEEDGESIDX: [[usize; 4]; 6] = [
    [4, 5, 8, 10],   // LEFT
    [6, 7, 9, 11],   // RIGHT
    [8, 9, 0, 2],    // BACK
    [10, 11, 1, 3],  // FRONT
    [0, 1, 4, 6],    // BOTTOM
    [2, 3, 5, 7],    // TOP
];

/// Default texture slot index (matches Cube2 DEFAULT_GEOM = 1).
pub const DEFAULT_GEOM: u16 = 1;

// ── The core Cube type ─────────────────────────────────────────────────────────

/// A node in the octree.  Either a branch (has children) or a leaf.
///
/// Leaves store geometry via `edges` (12 bytes) and per-face textures.
/// Children are stored in a flat `Box<[Cube; 8]>` to avoid fat pointers.
#[derive(Debug, Clone)]
pub struct Cube {
    /// 8 children, or `None` for a leaf node.
    pub children: Option<Box<[Cube; 8]>>,

    /// 12 edge bytes — the deformation data.
    /// edges[dim<<2 | y<<1 | x] = byte with two 4-bit endpoints.
    /// See `edge_idx`, `edge_get`, `edge_set`.
    /// Meaningful only for leaf nodes.
    pub edges: [u8; 12],

    /// Texture slot index per face (6 faces).
    pub texture: [u16; 6],

    /// Material (MAT_AIR, MAT_WATER, …)
    pub material: u8,

    /// Merged-face bitmask (rendering optimisation, read from OGZ v27+).
    pub merged: u8,
}

impl Default for Cube {
    fn default() -> Self {
        Self {
            children: None,
            edges: EDGES_SOLID,
            texture: [0u16; 6],
            material: MAT_AIR,
            merged: 0,
        }
    }
}

impl Cube {
    pub fn empty() -> Self { Self { edges: EDGES_EMPTY, ..Default::default() } }
    pub fn solid() -> Self { Self::default() }

    /// Match Cube2's `isempty(c)` — checks only faces[0] (first 4 edge bytes).
    /// A cube is empty if its X-dimension edges are all zero.
    pub fn is_empty(&self) -> bool {
        self.edges[0] == 0 && self.edges[1] == 0 && self.edges[2] == 0 && self.edges[3] == 0
    }

    /// Match Cube2's `isentirelysolid(c)` — checks all 3 face u32s == F_SOLID.
    pub fn is_solid(&self) -> bool { self.edges == EDGES_SOLID }
    pub fn is_leaf(&self)  -> bool { self.children.is_none() }

    /// Get one of the 12 edge bytes.
    pub fn edge(&self, dim: usize, x: usize, y: usize) -> u8 {
        self.edges[edge_idx(dim, x, y)]
    }

    // ── getcubevector / setcubevector (by corner index) ──────────────────────
    // Port of Cube2 `getcubevector(cube &c, int i, ivec &p)` and
    // `setcubevector(cube &c, int i, const ivec &p)` from octa.cpp:141-153.
    // `i` is a bit-encoded corner index: x=bit0, y=bit1, z=bit2.

    /// Get the deformed position of corner `i` (0–7) in cube-local units (0–8).
    /// `i` encodes (x,y,z) as bit 0, 1, 2 respectively.
    /// Equivalent to Cube2 `getcubevector(c, i, p)`.
    pub fn corner(&self, i: usize) -> [u8; 3] {
        let x = i & 1;
        let y = (i >> 1) & 1;
        let z = (i >> 2) & 1;
        [
            edge_get(self.edges[edge_idx(0, y, z)], x),
            edge_get(self.edges[edge_idx(1, z, x)], y),
            edge_get(self.edges[edge_idx(2, x, y)], z),
        ]
    }

    /// Set the deformed position of corner `i` (0–7) in cube-local units (0–8).
    /// Equivalent to Cube2 `setcubevector(c, i, p)`.
    pub fn set_corner(&mut self, i: usize, p: [u8; 3]) {
        let x = i & 1;
        let y = (i >> 1) & 1;
        let z = (i >> 2) & 1;
        edge_set(&mut self.edges[edge_idx(0, y, z)], x, p[0]);
        edge_set(&mut self.edges[edge_idx(1, z, x)], y, p[1]);
        edge_set(&mut self.edges[edge_idx(2, x, y)], z, p[2]);
    }

    // ── getcubevector / setcubevector (by face-oriented coordinates) ──────
    // Port of Cube2 `getcubevector(cube &c, int d, int x, int y, int z, ivec &p)`
    // and `setcubevector(cube &c, int d, int x, int y, int z, const ivec &p)`.
    // `d` = dimension, `x`/`y` = row/col (0 or 1), `z` = depth (0 or 1).

    /// Get vertex position using face-oriented coordinates.
    /// `d` = dimension (0=X, 1=Y, 2=Z), `x`/`y` = row/col, `z` = depth.
    pub fn get_cube_vector(&self, d: usize, x: usize, y: usize, z: usize) -> [u8; 3] {
        // Build the 3D coordinate vector: v[R[d]]=x, v[C[d]]=y, v[D[d]]=z
        let mut v = [0usize; 3];
        v[R[d]] = x; v[C[d]] = y; v[D[d]] = z;
        [
            edge_get(self.edges[edge_idx(0, v[R[0]], v[C[0]])], v[D[0]]),
            edge_get(self.edges[edge_idx(1, v[R[1]], v[C[1]])], v[D[1]]),
            edge_get(self.edges[edge_idx(2, v[R[2]], v[C[2]])], v[D[2]]),
        ]
    }

    /// Set vertex position using face-oriented coordinates.
    pub fn set_cube_vector(&mut self, d: usize, x: usize, y: usize, z: usize, p: [u8; 3]) {
        let mut v = [0usize; 3];
        v[R[d]] = x; v[C[d]] = y; v[D[d]] = z;
        edge_set(&mut self.edges[edge_idx(0, v[R[0]], v[C[0]])], v[D[0]], p[0]);
        edge_set(&mut self.edges[edge_idx(1, v[R[1]], v[C[1]])], v[D[1]], p[1]);
        edge_set(&mut self.edges[edge_idx(2, v[R[2]], v[C[2]])], v[D[2]], p[2]);
    }

    // ── faces[] access (reinterpret 4 edge bytes as u32) ──────────────────

    /// Read `faces[dim]` — 4 consecutive edge bytes as a little-endian u32.
    /// Matches C++ `c.faces[d]` union access.
    #[inline]
    pub fn face(&self, dim: usize) -> u32 {
        u32::from_le_bytes([
            self.edges[dim * 4],
            self.edges[dim * 4 + 1],
            self.edges[dim * 4 + 2],
            self.edges[dim * 4 + 3],
        ])
    }

    /// Write `faces[dim]` — set 4 consecutive edge bytes from a u32.
    #[inline]
    pub fn set_face(&mut self, dim: usize, val: u32) {
        let b = val.to_le_bytes();
        self.edges[dim * 4]     = b[0];
        self.edges[dim * 4 + 1] = b[1];
        self.edges[dim * 4 + 2] = b[2];
        self.edges[dim * 4 + 3] = b[3];
    }

    // ── solidfaces / emptyfaces ───────────────────────────────────────────

    /// `setfaces(c, face)` — set all 3 face u32s to the same value.
    pub fn set_faces(&mut self, face: u32) {
        self.set_face(0, face);
        self.set_face(1, face);
        self.set_face(2, face);
    }

    /// `solidfaces(c)` — make this cube entirely solid.
    pub fn solidfaces(&mut self) { self.set_faces(F_SOLID); }

    /// `emptyfaces(c)` — make this cube entirely empty.
    pub fn emptyfaces(&mut self) { self.set_faces(F_EMPTY); }

    // ── optiface ──────────────────────────────────────────────────────────
    // Port of Cube2 `optiface(uchar *p, cube &c)` from octa.cpp:155-159.
    // If all edge pairs on a face match (low nibble == high nibble), the
    // face has zero thickness → the cube is effectively empty.

    /// Optimize a single face: if all edge start==end, cube is empty.
    /// `dim` is the face dimension (0=X, 1=Y, 2=Z).
    pub fn optiface(&mut self, dim: usize) {
        let f = self.face(dim);
        if ((f >> 4) & 0x0F0F0F0F) == (f & 0x0F0F0F0F) {
            self.emptyfaces();
        }
    }

    // ── visibleorient ─────────────────────────────────────────────────────
    // Port of Cube2 `visibleorient(cube &c, int orient)` from octa.cpp:443-457.
    // If a face is crushed and touching, redirect to the adjacent orient.

    /// Check if a face's edges are "crushed" (both endpoints at same extreme).
    #[inline]
    fn crushed_edge(e: u8, dc: usize) -> bool {
        if dc != 0 { e == 0 } else { e == 0x88 }
    }

    /// Determine the actual visible orientation for a face that might be crushed.
    pub fn visible_orient(&self, orient: usize) -> usize {
        for i in 0..2 {
            let a = FACEEDGESIDX[orient][i * 2];
            let b = FACEEDGESIDX[orient][i * 2 + 1];
            for j in 0..2 {
                if Self::crushed_edge(self.edges[a], j)
                    && Self::crushed_edge(self.edges[b], j)
                    && self.touching_face(orient)
                {
                    return ((a >> 2) << 1) + j;
                }
            }
        }
        orient
    }

    /// Port of Cube2 `touchingface()` — does this face reach the cube boundary?
    #[inline]
    pub fn touching_face(&self, orient: usize) -> bool {
        let face = self.face(orient >> 1);
        if (orient & 1) != 0 {
            (face & 0xF0F0F0F0) == 0x80808080
        } else {
            (face & 0x0F0F0F0F) == 0
        }
    }

    /// Port of Cube2 `notouchingface()`.
    #[inline]
    pub fn not_touching_face(&self, orient: usize) -> bool {
        let face = self.face(orient >> 1);
        if (orient & 1) != 0 {
            (face & 0x80808080) == 0
        } else {
            (0x88888888u32.wrapping_sub(face) & 0x08080808) == 0
        }
    }

    /// Port of Cube2 `flataxisface()` — is this face flat and axis-aligned?
    #[inline]
    pub fn flat_axis_face(&self, orient: usize) -> bool {
        let mut face = self.face(orient >> 1);
        if (orient & 1) != 0 { face >>= 4; }
        (face & 0x0F0F0F0F) == 0x01010101u32.wrapping_mul(face & 0x0F)
    }

    /// Port of Cube2 `faceedges()` — pack the 4 edges around a face orient.
    #[inline]
    pub fn face_edges(&self, orient: usize) -> u32 {
        let idx = &FACEEDGESIDX[orient];
        u32::from_le_bytes([
            self.edges[idx[0]], self.edges[idx[1]],
            self.edges[idx[2]], self.edges[idx[3]],
        ])
    }

    /// Get 4 corner positions (cube-local 0–8 units) for the given face orientation.
    ///
    /// Vertex order matches Cube2's `fv[6][4]` table (from `genfaceverts()`),
    /// translated to our bit-encoded corner indices (bit = z<<2 | y<<1 | x).
    ///
    /// Cube2 GENCUBEVERTS vertex numbering:
    ///   0=(1,1,0) 1=(0,1,0) 2=(0,1,1) 3=(1,1,1) 4=(1,0,1) 5=(0,0,1) 6=(0,0,0) 7=(1,0,0)
    /// Cube2 fv[6][4]:
    ///   {2,1,6,5}, {3,4,7,0}, {4,5,6,7}, {1,2,3,0}, {6,1,0,7}, {5,4,3,2}
    /// Mapped to bit-encoding (z<<2|y<<1|x):
    ///   Cube2→bit: [3, 2, 6, 7, 5, 4, 0, 1]
    pub fn face_corners(&self, orient: usize) -> [[u8; 3]; 4] {
        const FACE_CORNERS: [[usize; 4]; 6] = [
            [6, 2, 0, 4], // LEFT   (-X): Cube2 fv {2,1,6,5}
            [7, 5, 1, 3], // RIGHT  (+X): Cube2 fv {3,4,7,0}
            [5, 4, 0, 1], // BACK   (-Y): Cube2 fv {4,5,6,7}
            [2, 6, 7, 3], // FRONT  (+Y): Cube2 fv {1,2,3,0}
            [0, 2, 3, 1], // BOTTOM (-Z): Cube2 fv {6,1,0,7}
            [4, 5, 7, 6], // TOP    (+Z): Cube2 fv {5,4,3,2}
        ];
        let ci = FACE_CORNERS[orient];
        [self.corner(ci[0]), self.corner(ci[1]), self.corner(ci[2]), self.corner(ci[3])]
    }
}

// ── Octree structural helpers ──────────────────────────────────────────────────

/// Create 8 new cubes initialized to `face` (F_EMPTY or F_SOLID) and material.
/// Port of Cube2 `newcubes(uint face, int mat)`.
pub fn newcubes(face: u32, mat: u8) -> Box<[Cube; 8]> {
    let b = face.to_le_bytes();
    let edges = [b[0], b[1], b[2], b[3], b[0], b[1], b[2], b[3], b[0], b[1], b[2], b[3]];
    let proto = Cube {
        children: None,
        edges,
        texture: [DEFAULT_GEOM; 6],
        material: mat,
        merged: 0,
    };
    Box::new([
        proto.clone(), proto.clone(), proto.clone(), proto.clone(),
        proto.clone(), proto.clone(), proto.clone(), proto.clone(),
    ])
}

/// Discard children of a cube, resetting it to a leaf.
/// Port of Cube2 `discardchildren()` (simplified — no VA/ext/lightmap).
pub fn discard_children(c: &mut Cube) {
    c.material = MAT_AIR;
    c.merged = 0;
    c.children = None;
}

/// Deep-copy a cube, clearing rendering metadata (merged).
/// Port of Cube2 `copycube(const cube &src, cube &dst)`.
pub fn copycube(src: &Cube) -> Cube {
    let mut dst = Cube {
        children: None,
        edges: src.edges,
        texture: src.texture,
        material: src.material,
        merged: 0,
    };
    if let Some(children) = &src.children {
        let mut new_children: Box<[Cube; 8]> = Box::new(std::array::from_fn(|_| Cube::empty()));
        for i in 0..8 {
            new_children[i] = copycube(&children[i]);
        }
        dst.children = Some(new_children);
    }
    dst
}

/// Discard dst's children and deep-copy src into dst.
/// Port of Cube2 `pastecube(const cube &src, cube &dst)`.
pub fn pastecube(src: &Cube, dst: &mut Cube) {
    discard_children(dst);
    *dst = copycube(src);
}

/// Simple subdivide: split a leaf into 8 identical children.
/// Used by serialization and basic editing. No-op if already has children.
pub fn subdivide(cube: &mut Cube) {
    if cube.children.is_some() { return; }
    let face = if cube.is_empty() { F_EMPTY } else if cube.is_solid() { F_SOLID } else {
        // Non-trivial geometry: use full subdividecube
        subdividecube(cube);
        return;
    };
    let mut children = newcubes(face, cube.material);
    for i in 0..8 {
        children[i].texture = cube.texture;
    }
    cube.children = Some(children);
}

// ── Full subdividecube ────────────────────────────────────────────────────────
// Port of Cube2 `subdividecube(cube &c, bool fullcheck, bool brighten)`
// from octa.cpp:364-439. Handles deformed geometry with midedge interpolation.

/// Compute the midpoint of an edge in one axis, interpolating to the center (value 8).
fn midedge(a: &[i32; 3], b: &[i32; 3], xd: usize, yd: usize) -> i32 {
    let (ax, ay, bx, by) = (a[xd], a[yd], b[xd], b[yd]);
    if ay == by { return ay; }
    if ax == bx { return ay; }
    let crossx = (ax < 8 && bx > 8) || (ax > 8 && bx < 8);
    let crossy = (ay < 8 && by > 8) || (ay > 8 && by < 8);
    if crossy && !crossx { return 8; }
    if ax <= 8 && bx <= 8 { return if ax > bx { ay } else { by }; }
    if ax >= 8 && bx >= 8 { return if ax < bx { ay } else { by }; }
    let risex = (by - ay) * (8 - ax) * 256;
    let s = risex / (bx - ax);
    let y = s / 256 + ay;
    if crossy { 8 } else { y.clamp(0, 16) }
}

/// Full subdivision of a deformed cube into 8 children.
/// Port of Cube2 `subdividecube()` from octa.cpp:364-439.
pub fn subdividecube(c: &mut Cube) {
    if c.children.is_some() { return; }
    if c.is_empty() || c.is_solid() {
        let face = if c.is_empty() { F_EMPTY } else { F_SOLID };
        let mut ch = newcubes(face, c.material);
        for i in 0..8 { ch[i].texture = c.texture; }
        c.children = Some(ch);
        return;
    }

    let mut ch = newcubes(F_SOLID, c.material);

    // Get all 8 corner positions, scaled up by 2 for subdivision math
    let mut v = [[0i32; 3]; 8];
    for i in 0..8 {
        let corner = c.corner(i);
        v[i] = [corner[0] as i32 * 2, corner[1] as i32 * 2, corner[2] as i32 * 2];
    }

    for j in 0..6 {
        let d = j >> 1; // dimension
        let z = j & 1;  // dimcoord

        let v00 = v[octaindex(d, 0, 0, z)];
        let v10 = v[octaindex(d, 1, 0, z)];
        let v01 = v[octaindex(d, 0, 1, z)];
        let v11 = v[octaindex(d, 1, 1, z)];

        let mut e = [[0i32; 3]; 3];
        // Corners
        e[0][0] = v00[d]; e[0][2] = v01[d];
        e[2][0] = v10[d]; e[2][2] = v11[d];
        // Edge midpoints
        e[0][1] = midedge(&v00, &v01, C[d], d);
        e[1][0] = midedge(&v00, &v10, R[d], d);
        e[1][2] = midedge(&v11, &v01, R[d], d);
        e[2][1] = midedge(&v11, &v10, C[d], d);
        // Center — pick the diagonal that preserves more detail
        let c1 = midedge(&v00, &v11, R[d], d);
        let c2 = midedge(&v01, &v10, R[d], d);
        e[1][1] = if (z != 0 && c1 > c2) || (z == 0 && c1 < c2) { c1 } else { c2 };

        for i in 0..8 {
            ch[i].texture[j] = c.texture[j];
            let rd = (i >> R[d]) & 1;
            let cd = (i >> C[d]) & 1;
            let dd = (i >> D[d]) & 1;
            let val = |r: usize, c_: usize| -> u8 {
                (e[r][c_] - (dd as i32) * 8).clamp(0, 8) as u8
            };
            edge_set(&mut ch[i].edges[edge_idx(d, 0, 0)], z, val(rd, cd));
            edge_set(&mut ch[i].edges[edge_idx(d, 1, 0)], z, val(1 + rd, cd));
            edge_set(&mut ch[i].edges[edge_idx(d, 0, 1)], z, val(rd, 1 + cd));
            edge_set(&mut ch[i].edges[edge_idx(d, 1, 1)], z, val(1 + rd, 1 + cd));
        }
    }

    // Validate children — emptyfaces any that have invalid edge data
    for i in 0..8 {
        for j in 0..3 {
            let f = ch[i].face(j);
            let e0 = f & 0x0F0F0F0F;
            let e1 = (f >> 4) & 0x0F0F0F0F;
            if e0 == e1 || ((e1.wrapping_add(0x07070707)) | (e1.wrapping_sub(e0))) & 0xF0F0F0F0 != 0 {
                ch[i].emptyfaces();
                break;
            }
        }
    }

    c.children = Some(ch);
}

// ── validatec ─────────────────────────────────────────────────────────────────
// Port of Cube2 `validatec(cube *c, int size)` from octa.cpp:184-215.
// Ensures all cubes in the tree have valid geometry.

/// Validate an array of 8 cubes at a given `size` (half of parent size).
/// Fixes invalid edge data (start > end, etc.) by emptying bad cubes.
/// Subdivides cubes that are too large (> 0x1000 units).
pub fn validatec(c: &mut [Cube; 8], size: i32) {
    for i in 0..8 {
        if c[i].children.is_some() {
            if size <= 1 {
                c[i].solidfaces();
                discard_children(&mut c[i]);
            } else if let Some(ref mut children) = c[i].children {
                validatec(children, size >> 1);
            }
        } else if size > 0x1000 {
            subdividecube(&mut c[i]);
            if let Some(ref mut children) = c[i].children {
                validatec(children, size >> 1);
            }
        } else {
            // Check edge validity: for each dimension, low must be <= high
            for j in 0..3 {
                let f = c[i].face(j);
                let e0 = f & 0x0F0F0F0F;
                let e1 = (f >> 4) & 0x0F0F0F0F;
                if e0 == e1 || ((e1.wrapping_add(0x07070707)) | (e1.wrapping_sub(e0))) & 0xF0F0F0F0 != 0 {
                    c[i].emptyfaces();
                    break;
                }
            }
        }
    }
}

// ── isvalidcube ───────────────────────────────────────────────────────────────
// Port of Cube2 `isvalidcube(const cube &c)` from octa.cpp:172-182.
// Checks if a cube is convex by generating clip planes and testing all 8 verts.
// Simplified version that checks edge constraints without full clip planes.

/// Check if a cube has valid (non-degenerate, convex) geometry.
/// Returns false if any edge has start > end.
pub fn is_valid_cube(c: &Cube) -> bool {
    for j in 0..3 {
        let f = c.face(j);
        let e0 = f & 0x0F0F0F0F;
        let e1 = (f >> 4) & 0x0F0F0F0F;
        // Invalid if start==end (zero thickness) or start > end
        if e0 == e1 { return false; }
        if ((e1.wrapping_add(0x07070707)) | (e1.wrapping_sub(e0))) & 0xF0F0F0F0 != 0 {
            return false;
        }
    }
    true
}

// ── getmippedtexture ──────────────────────────────────────────────────────────
// Port of Cube2 `getmippedtexture()` from octa.cpp:294-313.
// When collapsing children, pick the most common/best texture for each face.

/// Get the best texture for a face when mipping (collapsing children).
pub fn get_mipped_texture(parent: &Cube, orient: usize) -> u16 {
    let children = match &parent.children {
        Some(c) => c,
        None => return parent.texture[orient],
    };
    let d = orient >> 1;
    let dc = orient & 1;
    let mut texs = [0u16; 4];
    let mut numtexs = 0usize;
    for x in 0..2 {
        for y in 0..2 {
            let mut n = octaindex(d, x, y, dc);
            if children[n].is_empty() {
                n = oppositeocta(d, n);
                if children[n].is_empty() { continue; }
            }
            let tex = children[n].texture[orient];
            // Check for duplicates — return the first duplicate found
            if tex > DEFAULT_GEOM {
                for k in 0..numtexs {
                    if texs[k] == tex { return tex; }
                }
            }
            texs[numtexs] = tex;
            numtexs += 1;
        }
    }
    // Return last texture > DEFAULT_GEOM, or fall back to DEFAULT_GEOM
    for k in (0..numtexs).rev() {
        if k == 0 || texs[k] > DEFAULT_GEOM { return texs[k]; }
    }
    DEFAULT_GEOM
}

// ── forcemip ──────────────────────────────────────────────────────────────────
// Port of Cube2 `forcemip(cube &c, bool fixtex)` from octa.cpp:315-335.
// Collapse children into parent by sampling nearest corner vertices.

/// Force-collapse children into parent cube geometry.
/// Each parent corner takes the value from the nearest non-empty child.
pub fn forcemip(c: &mut Cube, fixtex: bool) {
    let ch: Box<[Cube; 8]> = match c.children.take() {
        Some(ch) => ch,
        None => return,
    };
    c.emptyfaces();

    for i in 0..8usize {
        // Breadth-first search for nearest non-empty child at this corner
        for j in 0..8usize {
            let n = i ^ (if j == 3 { 4 } else if j == 4 { 3 } else { j });
            if !ch[n].is_empty() {
                let v = ch[n].corner(i);
                // Adjust from child coords (0-8) to parent coords (0-8),
                // accounting for child position within parent: (n, v, 8).shr(1)
                let px = ((n & 1) * 8 + v[0] as usize) >> 1;
                let py = (((n >> 1) & 1) * 8 + v[1] as usize) >> 1;
                let pz = (((n >> 2) & 1) * 8 + v[2] as usize) >> 1;
                c.set_corner(i, [px as u8, py as u8, pz as u8]);
                break;
            }
        }
    }

    if fixtex {
        // Temporarily put children back to read textures
        c.children = Some(ch);
        for j in 0..6 {
            c.texture[j] = get_mipped_texture(c, j);
        }
        c.children = None;
    }
}

// ── Octree child indexing ──────────────────────────────────────────────────────

/// Child index from world coordinates at a given scale.
/// Bit 0=X, bit 1=Y, bit 2=Z (same as Cube2).
#[inline]
pub fn child_idx(x: i32, y: i32, z: i32, scale: u32) -> usize {
    ((((z >> scale) & 1) << 2) | (((y >> scale) & 1) << 1) | ((x >> scale) & 1)) as usize
}

/// Origin of child `i` given parent origin at the given scale.
#[inline]
pub fn child_origin(px: i32, py: i32, pz: i32, child: usize, scale: u32) -> (i32, i32, i32) {
    let s = 1 << scale;
    (
        px + ((child & 1) as i32) * s,
        py + (((child >> 1) & 1) as i32) * s,
        pz + (((child >> 2) & 1) as i32) * s,
    )
}

// ── OctreeWorld ────────────────────────────────────────────────────────────────

/// The full loaded world: 8 root cubes + metadata.
pub struct OctreeWorld {
    /// The 8 root cubes of the octree.
    pub root: Box<[Cube; 8]>,

    /// World scale: size = 1 << world_scale  (e.g., scale=10 → 1024 units).
    pub world_scale: u32,

    /// Entities loaded from the map (position + type + attrs).
    pub entities: Vec<MapEntity>,

    /// OGZ format version this was loaded from.
    pub ogz_version: i32,
}

impl OctreeWorld {
    pub fn world_size(&self) -> i32 { 1 << self.world_scale }

    /// Recursively look up the leaf cube at world position (x,y,z), returning
    /// `(cube, origin, size)`.  Clamps out-of-bounds coordinates.
    pub fn lookup(&self, mut x: i32, mut y: i32, mut z: i32)
        -> (&Cube, (i32, i32, i32), i32)
    {
        let ws = self.world_size();
        x = x.clamp(0, ws - 1);
        y = y.clamp(0, ws - 1);
        z = z.clamp(0, ws - 1);

        let mut scale = self.world_scale - 1;
        let mut cube = &self.root[child_idx(x, y, z, scale)];
        let mut ox = ((x >> scale) & !1) << scale;
        let mut oy = ((y >> scale) & !1) << scale;
        let mut oz = ((z >> scale) & !1) << scale;

        while let Some(children) = &cube.children {
            if scale == 0 { break; }
            scale -= 1;
            let ci = child_idx(x, y, z, scale);
            cube = &children[ci];
            ox += ((ci & 1) as i32) << scale;
            oy += (((ci >> 1) & 1) as i32) << scale;
            oz += (((ci >> 2) & 1) as i32) << scale;
        }

        (cube, (ox, oy, oz), 1 << scale)
    }

    /// Lookup stopping at a target size. Returns the cube at `target_size` level,
    /// which may still have children (finer detail). This matches C++'s
    /// `lookupcube(to, tsize)` with negative tsize (no auto-subdivide).
    /// Used by blockcopy to copy subtrees at grid resolution.
    pub fn lookup_at(&self, mut x: i32, mut y: i32, mut z: i32, target_size: i32)
        -> (&Cube, (i32, i32, i32), i32)
    {
        let ws = self.world_size();
        x = x.clamp(0, ws - 1);
        y = y.clamp(0, ws - 1);
        z = z.clamp(0, ws - 1);

        let target_scale = (target_size as u32).trailing_zeros();

        let mut scale = self.world_scale - 1;
        let mut cube = &self.root[child_idx(x, y, z, scale)];
        let mut ox = ((x >> scale) & !1) << scale;
        let mut oy = ((y >> scale) & !1) << scale;
        let mut oz = ((z >> scale) & !1) << scale;

        while let Some(children) = &cube.children {
            if scale <= target_scale { break; } // Stop at target size
            if scale == 0 { break; }
            scale -= 1;
            let ci = child_idx(x, y, z, scale);
            cube = &children[ci];
            ox += ((ci & 1) as i32) << scale;
            oy += (((ci >> 1) & 1) as i32) << scale;
            oz += (((ci >> 2) & 1) as i32) << scale;
        }

        (cube, (ox, oy, oz), 1 << scale)
    }

    /// Mutable lookup: find and return a mutable reference to the cube at (x,y,z).
    /// If `target_size > 0`, auto-subdivides down to that grid size.
    /// Port of Cube2 `lookupcube(to, tsize, ro, rsize)` from octa.cpp:219-244.
    pub fn lookup_mut(&mut self, x: i32, y: i32, z: i32, target_size: i32)
        -> (&mut Cube, (i32, i32, i32), i32)
    {
        let ws = self.world_size();
        let tx = x.clamp(0, ws - 1);
        let ty = y.clamp(0, ws - 1);
        let tz = z.clamp(0, ws - 1);

        let mut scale = self.world_scale - 1;
        let csize = target_size.unsigned_abs() as u32;

        let ci = child_idx(tx, ty, tz, scale);
        let mut cube: *mut Cube = &mut self.root[ci];

        // Navigate down until we reach target scale or a leaf
        if csize >> scale == 0 {
            loop {
                let c = unsafe { &mut *cube };
                if c.children.is_none() {
                    if target_size > 0 {
                        // Auto-subdivide
                        loop {
                            subdividecube(unsafe { &mut *cube });
                            scale -= 1;
                            let ci2 = child_idx(tx, ty, tz, scale);
                            cube = &mut unsafe { &mut *cube }.children.as_mut().unwrap()[ci2];
                            if csize >> scale != 0 { break; }
                        }
                    }
                    break;
                }
                scale -= 1;
                let ci2 = child_idx(tx, ty, tz, scale);
                cube = &mut c.children.as_mut().unwrap()[ci2];
                if csize >> scale != 0 { break; }
            }
        }

        let mask = !((1i32 << scale) - 1);
        let origin = (tx & mask, ty & mask, tz & mask);
        (unsafe { &mut *cube }, origin, 1 << scale)
    }

    /// Iterate all leaf cubes, calling `f(cube, origin, size)`.
    pub fn for_each_leaf<F>(&self, mut f: F)
    where F: FnMut(&Cube, (i32, i32, i32), i32) {
        let half = self.world_size() / 2;
        for i in 0..8 {
            let (ox, oy, oz) = child_origin(0, 0, 0, i, self.world_scale - 1);
            Self::recurse_leaves(&self.root[i], ox, oy, oz, half, &mut f);
        }
    }

    fn recurse_leaves<F>(
        cube: &Cube, ox: i32, oy: i32, oz: i32, size: i32, f: &mut F,
    ) where F: FnMut(&Cube, (i32, i32, i32), i32) {
        if let Some(children) = &cube.children {
            let half = size >> 1;
            for i in 0..8 {
                let cx = ox + ((i & 1) as i32) * half;
                let cy = oy + (((i >> 1) & 1) as i32) * half;
                let cz = oz + (((i >> 2) & 1) as i32) * half;
                Self::recurse_leaves(&children[i], cx, cy, cz, half, f);
            }
        } else {
            f(cube, (ox, oy, oz), size);
        }
    }

    /// Mutably iterate ALL leaf cubes in the entire world.
    pub fn for_each_leaf_mut<F>(&mut self, mut f: F)
    where F: FnMut(&mut Cube, (i32, i32, i32), i32) {
        let half = self.world_size() / 2;
        for i in 0..8 {
            let (ox, oy, oz) = child_origin(0, 0, 0, i, self.world_scale - 1);
            Self::recurse_leaves_mut(&mut self.root[i], ox, oy, oz, half, &mut f);
        }
    }

    fn recurse_leaves_mut<F>(
        cube: &mut Cube, ox: i32, oy: i32, oz: i32, size: i32, f: &mut F,
    ) where F: FnMut(&mut Cube, (i32, i32, i32), i32) {
        if let Some(children) = &mut cube.children {
            let half = size >> 1;
            for i in 0..8 {
                let cx = ox + ((i & 1) as i32) * half;
                let cy = oy + (((i >> 1) & 1) as i32) * half;
                let cz = oz + (((i >> 2) & 1) as i32) * half;
                Self::recurse_leaves_mut(&mut children[i], cx, cy, cz, half, f);
            }
        } else {
            f(cube, (ox, oy, oz), size);
        }
    }

    /// Mutably iterate all leaf cubes that overlap the axis-aligned box
    /// `[x0, x1) × [y0, y1) × [z0, z1)` in world space.
    ///
    /// Only visits existing leaf nodes — no automatic subdivision.
    /// If you need sub-leaf editing, call `subdivide_to` first for each cell.
    pub fn for_each_leaf_mut_in_aabb<F>(
        &mut self,
        x0: i32, y0: i32, z0: i32,
        x1: i32, y1: i32, z1: i32,
        mut f: F,
    ) where F: FnMut(&mut Cube, (i32, i32, i32), i32) {
        let half = self.world_size() / 2;
        for i in 0..8usize {
            let ox = ((i & 1) as i32) * half;
            let oy = (((i >> 1) & 1) as i32) * half;
            let oz = (((i >> 2) & 1) as i32) * half;
            Self::recurse_leaves_mut_aabb(
                &mut self.root[i], ox, oy, oz, half,
                x0, y0, z0, x1, y1, z1, &mut f,
            );
        }
    }

    fn recurse_leaves_mut_aabb<F>(
        cube: &mut Cube,
        ox: i32, oy: i32, oz: i32, size: i32,
        x0: i32, y0: i32, z0: i32,
        x1: i32, y1: i32, z1: i32,
        f: &mut F,
    ) where F: FnMut(&mut Cube, (i32, i32, i32), i32) {
        // Reject if this node's AABB doesn't overlap the query AABB
        if ox + size <= x0 || ox >= x1 { return; }
        if oy + size <= y0 || oy >= y1 { return; }
        if oz + size <= z0 || oz >= z1 { return; }

        if cube.children.is_some() {
            let half = size >> 1;
            for i in 0..8usize {
                let cx = ox + ((i & 1) as i32) * half;
                let cy = oy + (((i >> 1) & 1) as i32) * half;
                let cz = oz + (((i >> 2) & 1) as i32) * half;
                let child = &mut cube.children.as_mut().unwrap()[i];
                Self::recurse_leaves_mut_aabb(
                    child, cx, cy, cz, half,
                    x0, y0, z0, x1, y1, z1, f,
                );
            }
        } else {
            f(cube, (ox, oy, oz), size);
        }
    }

    /// Ensure there exists a leaf node at exactly `(tx, ty, tz)` with cell size
    /// `target_size` (must be a power of 2 ≤ half world size).  Any oversized
    /// leaf covering that point is subdivided down to `target_size`.
    pub fn subdivide_to(&mut self, tx: i32, ty: i32, tz: i32, target_size: i32) {
        let half = self.world_size() / 2;
        for i in 0..8usize {
            let ox = ((i & 1) as i32) * half;
            let oy = (((i >> 1) & 1) as i32) * half;
            let oz = (((i >> 2) & 1) as i32) * half;
            if tx >= ox && tx < ox + half
                && ty >= oy && ty < oy + half
                && tz >= oz && tz < oz + half
            {
                Self::recurse_subdivide(&mut self.root[i], ox, oy, oz, half, tx, ty, tz, target_size);
                return;
            }
        }
    }

    fn recurse_subdivide(
        cube: &mut Cube,
        ox: i32, oy: i32, oz: i32, size: i32,
        tx: i32, ty: i32, tz: i32, target_size: i32,
    ) {
        if size <= target_size { return; } // already at or below target
        subdivide(cube);
        let half = size >> 1;
        for i in 0..8usize {
            let cx = ox + ((i & 1) as i32) * half;
            let cy = oy + (((i >> 1) & 1) as i32) * half;
            let cz = oz + (((i >> 2) & 1) as i32) * half;
            if tx >= cx && tx < cx + half
                && ty >= cy && ty < cy + half
                && tz >= cz && tz < cz + half
            {
                Self::recurse_subdivide(
                    &mut cube.children.as_mut().unwrap()[i],
                    cx, cy, cz, half, tx, ty, tz, target_size,
                );
                return;
            }
        }
    }
}

// ── Raycast ────────────────────────────────────────────────────────────────────

/// Result of a successful ray–octree intersection.
#[derive(Debug, Clone)]
pub struct RayHit {
    /// World-space origin of the hit cube (its corner at minimum XYZ).
    pub origin: (i32, i32, i32),
    /// Side length of the hit cube in world units.
    pub size: i32,
    /// Which face was entered (O_LEFT … O_TOP).
    pub orient: usize,
    /// Ray parameter `t` at the entry face (ray_origin + t * ray_dir = hit point).
    pub t: f32,
}

impl OctreeWorld {
    /// Cast a ray through the octree and return the nearest non-empty leaf hit.
    ///
    /// `ray_origin` and `ray_dir` are in renderer space (Y-up).
    /// Internally we convert to Cube2 space (Z-up) for the traversal,
    /// then convert the hit origin back.
    pub fn raycast(&self, ray_origin: [f32; 3], ray_dir: [f32; 3]) -> Option<RayHit> {
        // Renderer Y-up → Cube2 Z-up: swap Y↔Z
        let ro = [ray_origin[0], ray_origin[2], ray_origin[1]];
        let rd = [ray_dir[0],    ray_dir[2],    ray_dir[1]];

        let half = (self.world_size() / 2) as f32;
        let mut best: Option<RayHit> = None;

        for i in 0..8usize {
            let ox = ((i & 1) as f32) * half;
            let oy = (((i >> 1) & 1) as f32) * half;
            let oz = (((i >> 2) & 1) as f32) * half;
            Self::raycast_node(
                &self.root[i],
                ox, oy, oz, half,
                ro, rd,
                &mut best,
            );
        }

        // Convert hit origin back: Cube2 (x,y,z) → Renderer (x,z,y)
        best.map(|mut h| {
            let (cx, cy, cz) = h.origin;
            h.origin = (cx, cz, cy);
            h
        })
    }

    fn raycast_node(
        cube: &Cube,
        ox: f32, oy: f32, oz: f32, size: f32,
        ro: [f32; 3], rd: [f32; 3],
        best: &mut Option<RayHit>,
    ) {
        // Slab test against this node's AABB [ox, ox+size]³
        let (t_enter, orient) = match aabb_slab(ro, rd, ox, oy, oz, size) {
            Some(r) => r,
            None => return,
        };

        // Prune: only recurse if this node is closer than current best
        if let Some(ref b) = best {
            if t_enter >= b.t { return; }
        }

        if let Some(children) = &cube.children {
            let half = size * 0.5;
            // Collect child hits, sort near-first, recurse in order
            let mut child_hits: [(f32, usize); 8] = [(f32::MAX, 0); 8];
            let mut count = 0usize;
            for i in 0..8usize {
                let cx = ox + ((i & 1) as f32) * half;
                let cy = oy + (((i >> 1) & 1) as f32) * half;
                let cz = oz + (((i >> 2) & 1) as f32) * half;
                if let Some((t, _)) = aabb_slab(ro, rd, cx, cy, cz, half) {
                    child_hits[count] = (t, i);
                    count += 1;
                }
            }
            child_hits[..count].sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            for k in 0..count {
                let i = child_hits[k].1;
                let cx = ox + ((i & 1) as f32) * half;
                let cy = oy + (((i >> 1) & 1) as f32) * half;
                let cz = oz + (((i >> 2) & 1) as f32) * half;
                Self::raycast_node(&children[i], cx, cy, cz, half, ro, rd, best);
            }
        } else if !cube.is_empty() {
            // Leaf hit — record if closest so far
            let is_better = best.as_ref().map_or(true, |b| t_enter < b.t);
            if is_better {
                *best = Some(RayHit {
                    origin: (ox as i32, oy as i32, oz as i32),
                    size: size as i32,
                    orient,
                    t: t_enter,
                });
            }
        }
    }
}

/// Ray–AABB slab intersection. Returns `Some((t_enter, orient))` or `None`.
/// `orient` is the face index (O_LEFT…O_TOP) of the entry face.
fn aabb_slab(
    ro: [f32; 3], rd: [f32; 3],
    ox: f32, oy: f32, oz: f32, size: f32,
) -> Option<(f32, usize)> {
    let mins = [ox, oy, oz];
    let maxs = [ox + size, oy + size, oz + size];

    let mut t_min = f32::NEG_INFINITY;
    let mut t_max = f32::INFINITY;
    let mut entry_axis = 0usize;
    let mut entry_side = 0usize; // 0 = low face, 1 = high face

    for axis in 0..3 {
        let inv = 1.0 / rd[axis];
        let mut t0 = (mins[axis] - ro[axis]) * inv;
        let mut t1 = (maxs[axis] - ro[axis]) * inv;
        let flipped = inv < 0.0;
        if flipped { std::mem::swap(&mut t0, &mut t1); }
        if t0 > t_min {
            t_min = t0;
            entry_axis = axis;
            entry_side = if flipped { 1 } else { 0 };
        }
        t_max = t_max.min(t1);
    }

    if t_min > t_max || t_max < 0.0 { return None; }
    let t = t_min.max(0.0);
    // orient: axis*2 + side  →  LEFT=0,RIGHT=1,BACK=2,FRONT=3,BOTTOM=4,TOP=5
    let orient = entry_axis * 2 + entry_side;
    Some((t, orient))
}

// ── Remip / Optimize ──────────────────────────────────────────────────────────

impl OctreeWorld {
    /// Walk the entire tree bottom-up and merge children cubes that can be
    /// consolidated.  If all 8 children are empty leaves -> replace with empty.
    /// If all 8 children are solid leaves with the same material -> replace with solid.
    pub fn remip(&mut self) {
        for cube in self.root.iter_mut() {
            remip_cube(cube);
        }
    }
}

fn remip_cube(cube: &mut Cube) {
    let children = match cube.children.as_mut() {
        Some(c) => c,
        None => return, // leaf, nothing to do
    };

    // Recurse first (bottom-up)
    for child in children.iter_mut() {
        remip_cube(child);
    }

    // Check if all 8 children are leaves (no grandchildren)
    let all_leaves = children.iter().all(|c| c.children.is_none());
    if !all_leaves {
        return;
    }

    // Check if all empty
    let all_empty = children.iter().all(|c| c.edges == EDGES_EMPTY);
    if all_empty {
        cube.children = None;
        cube.edges = EDGES_EMPTY;
        return;
    }

    // Check if all solid with same material
    let all_solid = children.iter().all(|c| c.edges == EDGES_SOLID);
    if all_solid {
        let mat = children[0].material;
        let same_mat = children.iter().all(|c| c.material == mat);
        if same_mat {
            cube.children = None;
            cube.edges = EDGES_SOLID;
            cube.material = mat;
            return;
        }
    }
}

// ── Entities ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MapEntity {
    pub pos:   [f32; 3],
    pub etype: u8,
    pub attr:  [i16; 5],
}
