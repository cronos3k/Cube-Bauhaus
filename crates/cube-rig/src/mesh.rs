//! Skinnable mesh: stable-ID vertices, triangle indices, and per-vertex bone
//! influences. Unlike the octree mesh (which is regenerated on every edit and
//! has no persistent vertex identity), these vertices keep a fixed index for
//! the lifetime of the rig, so weights can be attached to them.

use glam::Vec3;

/// Maximum bone influences per vertex. Matches the glTF `JOINTS_0`/`WEIGHTS_0`
/// vec4 convention and the common 4-bone GPU skinning limit.
pub const MAX_INFLUENCES: usize = 4;

/// One vertex of a skinnable mesh.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RigVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

impl RigVertex {
    pub fn pos(&self) -> Vec3 {
        Vec3::from_array(self.position)
    }
    pub fn nrm(&self) -> Vec3 {
        Vec3::from_array(self.normal)
    }
}

/// A single (bone, weight) pair. `weight == 0` marks an unused slot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Influence {
    pub bone: u16,
    pub weight: f32,
}

impl Influence {
    pub const NONE: Influence = Influence { bone: 0, weight: 0.0 };
}

/// The fixed-size influence set for one vertex.
pub type VertexInfluences = [Influence; MAX_INFLUENCES];

const EMPTY_INFLUENCES: VertexInfluences = [Influence::NONE; MAX_INFLUENCES];

/// A mesh that can be bound to a [`crate::Skeleton`].
#[derive(Clone, Debug, Default)]
pub struct SkinnedMesh {
    pub vertices: Vec<RigVertex>,
    pub indices: Vec<u32>,
    /// One influence set per vertex (parallel to `vertices`).
    pub influences: Vec<VertexInfluences>,
    /// Cached vertex→neighbour adjacency, built lazily for grow/shrink/smooth.
    adjacency: Option<Vec<Vec<u32>>>,
}

impl SkinnedMesh {
    pub fn new(vertices: Vec<RigVertex>, indices: Vec<u32>) -> Self {
        let influences = vec![EMPTY_INFLUENCES; vertices.len()];
        Self { vertices, indices, influences, adjacency: None }
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// True if every vertex has at least one non-zero influence.
    pub fn is_fully_skinned(&self) -> bool {
        self.influences
            .iter()
            .all(|inf| inf.iter().any(|i| i.weight > 0.0))
    }

    /// Vertices with no influence at all (would collapse to the origin when
    /// skinned). Useful for a "select unweighted" prep tool.
    pub fn unweighted_vertices(&self) -> Vec<u32> {
        self.influences
            .iter()
            .enumerate()
            .filter(|(_, inf)| inf.iter().all(|i| i.weight <= 0.0))
            .map(|(i, _)| i as u32)
            .collect()
    }

    /// Set a vertex to be rigidly bound (weight 1.0) to a single bone.
    pub fn set_rigid(&mut self, vertex: usize, bone: u16) {
        let mut inf = EMPTY_INFLUENCES;
        inf[0] = Influence { bone, weight: 1.0 };
        self.influences[vertex] = inf;
    }

    /// Paint `delta` weight for `bone` at `vertex` (negative erases), using
    /// normalized-paint semantics: the target bone moves toward `weight + delta`
    /// (clamped to 0..1) and the remaining weight is redistributed across the
    /// other bones in proportion to their current weights, so the set always
    /// sums to 1. Only the strongest [`MAX_INFLUENCES`] bones are kept.
    pub fn add_weight(&mut self, vertex: usize, bone: u16, delta: f32) {
        if delta == 0.0 {
            return;
        }
        let inf = &self.influences[vertex];
        let cur = inf
            .iter()
            .find(|i| i.weight > 0.0 && i.bone == bone)
            .map(|i| i.weight)
            .unwrap_or(0.0);
        let target = (cur + delta).clamp(0.0, 1.0);

        // Other bones get the leftover (1 - target), scaled to preserve ratios.
        let mut others: Vec<(u16, f32)> = inf
            .iter()
            .filter(|i| i.weight > 0.0 && i.bone != bone)
            .map(|i| (i.bone, i.weight))
            .collect();
        let others_sum: f32 = others.iter().map(|(_, w)| *w).sum();
        let remaining = (1.0 - target).max(0.0);
        if others_sum > 0.0 {
            let k = remaining / others_sum;
            for o in others.iter_mut() {
                o.1 *= k;
            }
        }

        let mut all = others;
        if target > 0.0 {
            all.push((bone, target));
        }
        // Keep the strongest MAX_INFLUENCES, then renormalise.
        all.sort_by(|a, b| b.1.total_cmp(&a.1));
        all.truncate(MAX_INFLUENCES);
        let mut new = EMPTY_INFLUENCES;
        for (slot, (b, w)) in new.iter_mut().zip(all) {
            *slot = Influence { bone: b, weight: w };
        }
        normalize_one(&mut new);
        self.influences[vertex] = new;
    }

    /// The dominant bone for a vertex (highest weight), if any.
    pub fn dominant_bone(&self, vertex: usize) -> Option<u16> {
        self.influences[vertex]
            .iter()
            .filter(|i| i.weight > 0.0)
            .max_by(|a, b| a.weight.total_cmp(&b.weight))
            .map(|i| i.bone)
    }

    /// Renormalise every vertex so its weights sum to 1 (vertices with no
    /// weight are left untouched).
    pub fn normalize_all(&mut self) {
        for inf in &mut self.influences {
            normalize_one(inf);
        }
    }

    /// Build (and cache) the undirected vertex adjacency graph from triangle
    /// edges. Vertices sharing an edge are neighbours.
    pub fn adjacency(&mut self) -> &[Vec<u32>] {
        if self.adjacency.is_none() {
            self.adjacency = Some(build_adjacency(self.vertices.len(), &self.indices));
        }
        self.adjacency.as_ref().unwrap()
    }

    /// Invalidate the cached adjacency (call after editing indices).
    pub fn invalidate_adjacency(&mut self) {
        self.adjacency = None;
    }

    /// Axis-aligned bounding box of all vertex positions.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for v in &self.vertices {
            min = min.min(v.pos());
            max = max.max(v.pos());
        }
        (min, max)
    }
}

/// Normalise a single influence set in place: clamp negatives, drop zeros, and
/// scale so the surviving weights sum to 1. A set with no positive weight is
/// left as all-zero (an explicitly unweighted vertex).
pub fn normalize_one(inf: &mut VertexInfluences) {
    let mut sum = 0.0;
    for i in inf.iter_mut() {
        if i.weight < 0.0 {
            i.weight = 0.0;
        }
        sum += i.weight;
    }
    if sum > 0.0 {
        for i in inf.iter_mut() {
            i.weight /= sum;
        }
    }
}

fn build_adjacency(vertex_count: usize, indices: &[u32]) -> Vec<Vec<u32>> {
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); vertex_count];
    let mut connect = |a: u32, b: u32| {
        let list = &mut adj[a as usize];
        if !list.contains(&b) {
            list.push(b);
        }
    };
    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        connect(a, b);
        connect(b, a);
        connect(b, c);
        connect(c, b);
        connect(c, a);
        connect(a, c);
    }
    adj
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad() -> SkinnedMesh {
        // two triangles sharing edge 1-2, four corner verts
        let v = |x: f32, y: f32| RigVertex { position: [x, y, 0.0], normal: [0.0, 0.0, 1.0], uv: [0.0, 0.0] };
        let verts = vec![v(0.0, 0.0), v(1.0, 0.0), v(0.0, 1.0), v(1.0, 1.0)];
        let idx = vec![0, 1, 2, 1, 3, 2];
        SkinnedMesh::new(verts, idx)
    }

    #[test]
    fn rigid_bind_sets_full_weight() {
        let mut m = quad();
        m.set_rigid(0, 7);
        assert_eq!(m.influences[0][0], Influence { bone: 7, weight: 1.0 });
        assert_eq!(m.dominant_bone(0), Some(7));
        assert!(!m.is_fully_skinned());
    }

    #[test]
    fn add_weight_normalizes() {
        let mut m = quad();
        m.add_weight(0, 1, 0.5);
        m.add_weight(0, 2, 0.5);
        let sum: f32 = m.influences[0].iter().map(|i| i.weight).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn add_weight_caps_influences_and_normalizes() {
        let mut m = quad();
        // paint six different bones with increasing strength
        for b in 0..6u16 {
            m.add_weight(0, b, (b as f32 + 1.0) * 0.1);
        }
        let bones: Vec<u16> = m.influences[0].iter().filter(|i| i.weight > 0.0).map(|i| i.bone).collect();
        // never more than MAX_INFLUENCES survive
        assert!(bones.len() <= MAX_INFLUENCES);
        // weights stay normalised
        let sum: f32 = m.influences[0].iter().map(|i| i.weight).sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // the strongest, most-recent bone dominates
        assert_eq!(m.dominant_bone(0), Some(5));
    }

    #[test]
    fn add_weight_erases_with_negative_delta() {
        let mut m = quad();
        m.add_weight(0, 1, 0.6);
        m.add_weight(0, 2, 0.4);
        // erase all of bone 1; bone 2 absorbs the remainder
        m.add_weight(0, 1, -1.0);
        assert_eq!(m.dominant_bone(0), Some(2));
        let sum: f32 = m.influences[0].iter().map(|i| i.weight).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn adjacency_from_shared_edge() {
        let mut m = quad();
        let adj = m.adjacency().to_vec();
        // vertex 1 touches 0, 2 (tri 0) and 3, 2 (tri 1)
        let mut n1 = adj[1].clone();
        n1.sort();
        assert_eq!(n1, vec![0, 2, 3]);
    }

    #[test]
    fn unweighted_listing() {
        let mut m = quad();
        m.set_rigid(0, 0);
        assert_eq!(m.unweighted_vertices(), vec![1, 2, 3]);
    }
}
