//! Vertex-selection tools — the "prep" stage before assigning weights.
//!
//! Selection is a boolean mask over mesh vertices. Screen-space tools (marquee,
//! lasso) take a view-projection matrix and a viewport so they work against
//! whatever the editor camera currently shows; topological tools (grow, shrink,
//! linked) use the mesh adjacency graph; semantic tools (by-bone, unweighted)
//! read the influence data. All are pure functions of their inputs so they can
//! be unit-tested headless.

use glam::{Mat4, Vec2, Vec3, Vec4};

use crate::mesh::SkinnedMesh;

/// How a tool combines with the existing selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectMode {
    /// Discard the old selection and use only the new hits.
    Replace,
    /// Union the new hits into the selection (Shift).
    Add,
    /// Remove the new hits from the selection (Ctrl).
    Subtract,
}

/// A boolean selection mask over a mesh's vertices.
#[derive(Debug, Clone, Default)]
pub struct VertexSelection {
    mask: Vec<bool>,
}

impl VertexSelection {
    pub fn new(vertex_count: usize) -> Self {
        Self { mask: vec![false; vertex_count] }
    }

    pub fn len(&self) -> usize {
        self.mask.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mask.is_empty()
    }

    pub fn count(&self) -> usize {
        self.mask.iter().filter(|&&b| b).count()
    }

    pub fn contains(&self, v: usize) -> bool {
        self.mask.get(v).copied().unwrap_or(false)
    }

    pub fn set(&mut self, v: usize, on: bool) {
        if let Some(slot) = self.mask.get_mut(v) {
            *slot = on;
        }
    }

    pub fn clear(&mut self) {
        self.mask.iter_mut().for_each(|b| *b = false);
    }

    pub fn select_all(&mut self) {
        self.mask.iter_mut().for_each(|b| *b = true);
    }

    pub fn invert(&mut self) {
        self.mask.iter_mut().for_each(|b| *b = !*b);
    }

    /// Indices of all selected vertices.
    pub fn indices(&self) -> Vec<u32> {
        self.mask
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| i as u32)
            .collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.mask
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(|(i, _)| i as u32)
    }

    /// Apply a set of newly hit vertices under the given combine mode.
    pub fn apply(&mut self, hits: &[u32], mode: SelectMode) {
        match mode {
            SelectMode::Replace => {
                self.clear();
                for &v in hits {
                    self.set(v as usize, true);
                }
            }
            SelectMode::Add => {
                for &v in hits {
                    self.set(v as usize, true);
                }
            }
            SelectMode::Subtract => {
                for &v in hits {
                    self.set(v as usize, false);
                }
            }
        }
    }

    /// Resize the mask to match a mesh, preserving existing bits where possible.
    pub fn resize(&mut self, vertex_count: usize) {
        self.mask.resize(vertex_count, false);
    }
}

/// Project a world position into pixel coordinates (origin top-left, y down).
/// Returns `None` if the point is behind the camera.
pub fn project(view_proj: &Mat4, world: Vec3, viewport: Vec2) -> Option<Vec2> {
    let clip: Vec4 = *view_proj * world.extend(1.0);
    if clip.w <= 1e-6 {
        return None; // behind / on the camera plane
    }
    let ndc = clip.truncate() / clip.w;
    let x = (ndc.x * 0.5 + 0.5) * viewport.x;
    let y = (1.0 - (ndc.y * 0.5 + 0.5)) * viewport.y;
    Some(Vec2::new(x, y))
}

/// True if a vertex faces the camera (its normal points toward `camera_pos`).
fn is_front_facing(mesh: &SkinnedMesh, v: usize, camera_pos: Vec3) -> bool {
    let view_dir = camera_pos - mesh.vertices[v].pos();
    mesh.vertices[v].nrm().dot(view_dir) > 0.0
}

/// Optional filter applied to screen-space selection tools.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScreenFilter {
    /// Only select vertices whose normal faces the camera. Requires `camera_pos`.
    pub front_facing_only: bool,
    pub camera_pos: Vec3,
}

/// Select vertices whose screen projection falls inside an axis-aligned
/// rectangle (a marquee / box drag). `rect_min`/`rect_max` are pixel corners.
pub fn marquee(
    mesh: &SkinnedMesh,
    view_proj: &Mat4,
    viewport: Vec2,
    rect_min: Vec2,
    rect_max: Vec2,
    filter: ScreenFilter,
) -> Vec<u32> {
    let lo = rect_min.min(rect_max);
    let hi = rect_min.max(rect_max);
    let mut hits = Vec::new();
    for (i, vert) in mesh.vertices.iter().enumerate() {
        if filter.front_facing_only && !is_front_facing(mesh, i, filter.camera_pos) {
            continue;
        }
        if let Some(p) = project(view_proj, vert.pos(), viewport) {
            if p.x >= lo.x && p.x <= hi.x && p.y >= lo.y && p.y <= hi.y {
                hits.push(i as u32);
            }
        }
    }
    hits
}

/// Select vertices whose screen projection falls inside a closed polygon (a
/// free-hand lasso). `polygon` is a list of pixel points; the edge from the
/// last point back to the first is implied.
pub fn lasso(
    mesh: &SkinnedMesh,
    view_proj: &Mat4,
    viewport: Vec2,
    polygon: &[Vec2],
    filter: ScreenFilter,
) -> Vec<u32> {
    if polygon.len() < 3 {
        return Vec::new();
    }
    let mut hits = Vec::new();
    for (i, vert) in mesh.vertices.iter().enumerate() {
        if filter.front_facing_only && !is_front_facing(mesh, i, filter.camera_pos) {
            continue;
        }
        if let Some(p) = project(view_proj, vert.pos(), viewport) {
            if point_in_polygon(p, polygon) {
                hits.push(i as u32);
            }
        }
    }
    hits
}

/// Standard ray-crossing (even-odd) point-in-polygon test in 2D.
pub fn point_in_polygon(p: Vec2, poly: &[Vec2]) -> bool {
    let mut inside = false;
    let n = poly.len();
    let mut j = n - 1;
    for i in 0..n {
        let a = poly[i];
        let b = poly[j];
        let crosses = (a.y > p.y) != (b.y > p.y);
        if crosses {
            let x_at = a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x);
            if p.x < x_at {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Select all vertices within `radius` pixels of a screen point (paint-select
/// brush dab).
pub fn brush(
    mesh: &SkinnedMesh,
    view_proj: &Mat4,
    viewport: Vec2,
    center: Vec2,
    radius: f32,
    filter: ScreenFilter,
) -> Vec<u32> {
    let r2 = radius * radius;
    let mut hits = Vec::new();
    for (i, vert) in mesh.vertices.iter().enumerate() {
        if filter.front_facing_only && !is_front_facing(mesh, i, filter.camera_pos) {
            continue;
        }
        if let Some(p) = project(view_proj, vert.pos(), viewport) {
            if (p - center).length_squared() <= r2 {
                hits.push(i as u32);
            }
        }
    }
    hits
}

/// Grow the selection by one adjacency ring: any vertex neighbouring a selected
/// vertex becomes selected. Returns the newly added vertices.
pub fn grow(mesh: &mut SkinnedMesh, sel: &mut VertexSelection) -> Vec<u32> {
    let adj = mesh.adjacency().to_vec();
    let mut added = Vec::new();
    let seeds: Vec<u32> = sel.indices();
    for v in seeds {
        for &n in &adj[v as usize] {
            if !sel.contains(n as usize) {
                sel.set(n as usize, true);
                added.push(n);
            }
        }
    }
    added
}

/// Shrink the selection by one ring: any selected vertex that has an
/// unselected neighbour is removed. Returns the removed vertices.
pub fn shrink(mesh: &mut SkinnedMesh, sel: &mut VertexSelection) -> Vec<u32> {
    let adj = mesh.adjacency().to_vec();
    let mut remove = Vec::new();
    for v in sel.indices() {
        if adj[v as usize].iter().any(|&n| !sel.contains(n as usize)) {
            remove.push(v);
        }
    }
    for &v in &remove {
        sel.set(v as usize, false);
    }
    remove
}

/// Flood-fill the connected mesh component(s) touching the current selection
/// ("select linked"). Returns the full connected selection's vertex indices.
pub fn select_linked(mesh: &mut SkinnedMesh, sel: &mut VertexSelection) -> Vec<u32> {
    let adj = mesh.adjacency().to_vec();
    let mut stack: Vec<u32> = sel.indices();
    while let Some(v) = stack.pop() {
        for &n in &adj[v as usize] {
            if !sel.contains(n as usize) {
                sel.set(n as usize, true);
                stack.push(n);
            }
        }
    }
    sel.indices()
}

/// Select every vertex whose dominant bone is `bone`.
pub fn by_dominant_bone(mesh: &SkinnedMesh, bone: u16) -> Vec<u32> {
    (0..mesh.vertex_count())
        .filter(|&v| mesh.dominant_bone(v) == Some(bone))
        .map(|v| v as u32)
        .collect()
}

/// Select every vertex with *any* influence from `bone` above `min_weight`.
pub fn by_any_influence(mesh: &SkinnedMesh, bone: u16, min_weight: f32) -> Vec<u32> {
    (0..mesh.vertex_count())
        .filter(|&v| {
            mesh.influences[v]
                .iter()
                .any(|i| i.bone == bone && i.weight > min_weight)
        })
        .map(|v| v as u32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::RigVertex;
    use glam::Vec3;

    fn grid_mesh() -> SkinnedMesh {
        // 3×3 grid of verts in the z=0 plane, 0..1 in x and y
        let mut verts = Vec::new();
        for gy in 0..3 {
            for gx in 0..3 {
                verts.push(RigVertex {
                    position: [gx as f32 * 0.5, gy as f32 * 0.5, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    uv: [0.0, 0.0],
                });
            }
        }
        // triangulate the 2×2 cell grid
        let mut idx = Vec::new();
        let at = |x: u32, y: u32| y * 3 + x;
        for cy in 0..2 {
            for cx in 0..2 {
                let (a, b, c, d) = (at(cx, cy), at(cx + 1, cy), at(cx, cy + 1), at(cx + 1, cy + 1));
                idx.extend_from_slice(&[a, b, c, b, d, c]);
            }
        }
        SkinnedMesh::new(verts, idx)
    }

    #[test]
    fn project_behind_camera_is_none() {
        // looking down -Z from origin; a point behind the camera (+Z) is culled
        let view = Mat4::look_at_rh(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), Vec3::Y);
        let proj = Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0);
        let vp = proj * view;
        assert!(project(&vp, Vec3::new(0.0, 0.0, 5.0), Vec2::new(100.0, 100.0)).is_none());
    }

    #[test]
    fn point_in_polygon_square() {
        let sq = [Vec2::new(0.0, 0.0), Vec2::new(2.0, 0.0), Vec2::new(2.0, 2.0), Vec2::new(0.0, 2.0)];
        assert!(point_in_polygon(Vec2::new(1.0, 1.0), &sq));
        assert!(!point_in_polygon(Vec2::new(3.0, 1.0), &sq));
    }

    #[test]
    fn grow_then_shrink_roundtrips_interior() {
        let mut m = grid_mesh();
        let mut sel = VertexSelection::new(m.vertex_count());
        sel.set(4, true); // center vertex
        let added = grow(&mut m, &mut sel);
        assert!(!added.is_empty());
        assert!(sel.count() > 1);
        shrink(&mut m, &mut sel);
        // center is fully interior to the grown ring, so it survives the shrink
        assert!(sel.contains(4));
    }

    #[test]
    fn linked_selects_whole_component() {
        let mut m = grid_mesh();
        let mut sel = VertexSelection::new(m.vertex_count());
        sel.set(0, true);
        let all = select_linked(&mut m, &mut sel);
        assert_eq!(all.len(), m.vertex_count());
    }

    #[test]
    fn by_bone_filters_dominant() {
        let mut m = grid_mesh();
        m.set_rigid(0, 5);
        m.set_rigid(1, 5);
        m.set_rigid(2, 9);
        let mut got = by_dominant_bone(&m, 5);
        got.sort();
        assert_eq!(got, vec![0, 1]);
    }

    #[test]
    fn apply_modes() {
        let mut sel = VertexSelection::new(5);
        sel.apply(&[0, 1, 2], SelectMode::Replace);
        assert_eq!(sel.count(), 3);
        sel.apply(&[2, 3], SelectMode::Add);
        assert_eq!(sel.count(), 4);
        sel.apply(&[0, 1], SelectMode::Subtract);
        assert_eq!(sel.indices(), vec![2, 3]);
    }
}
