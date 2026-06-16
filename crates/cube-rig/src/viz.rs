//! Per-vertex colour generation for previewing skin weights in the viewport.
//!
//! The renderer already carries a per-vertex colour; these helpers fill it so
//! the artist can *see* the rig while working:
//!
//!   * [`dominant_bone_colors`] — paints each vertex with a stable, distinct
//!     colour per dominant bone (a "weight islands" view).
//!   * [`weight_gradient_colors`] — a blue→green→red heatmap of one bone's
//!     influence (the classic weight-paint gradient).
//!   * [`blend_selection`] — tints selected vertices so the prep selection is
//!     visible on top of either mode.
//!
//! All functions return `[f32; 4]` RGBA per vertex, parallel to the mesh's
//! vertices, ready to drop into the vertex buffer's colour channel.

use crate::mesh::SkinnedMesh;
use crate::select::VertexSelection;

/// What the vertex colours should encode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightView {
    /// Distinct colour per dominant bone.
    DominantBone,
    /// Heatmap of the active bone's weight (set via the `bone` argument).
    Gradient,
    /// No weight tint (white) — show only geometry/selection.
    Off,
}

/// A stable, visually distinct colour for a bone index, using the golden-ratio
/// hue sequence so adjacent indices look different and the palette never
/// repeats for reasonable bone counts.
pub fn bone_color(bone: u16) -> [f32; 3] {
    const GOLDEN: f32 = 0.618_034;
    let hue = (bone as f32 * GOLDEN).fract();
    // vary saturation/value slightly so same-hue collisions still differ
    let sat = 0.55 + 0.25 * ((bone / 12) as f32 * GOLDEN).fract();
    hsv_to_rgb(hue, sat.min(0.85), 0.95)
}

/// Colour every vertex by its dominant bone. Unweighted vertices are grey.
pub fn dominant_bone_colors(mesh: &SkinnedMesh) -> Vec<[f32; 4]> {
    (0..mesh.vertex_count())
        .map(|v| match mesh.dominant_bone(v) {
            Some(b) => {
                let [r, g, bl] = bone_color(b);
                [r, g, bl, 1.0]
            }
            None => [0.35, 0.35, 0.35, 1.0],
        })
        .collect()
}

/// Heatmap of `bone`'s weight per vertex: 0 → blue, 0.5 → green, 1 → red.
/// Vertices with no influence from `bone` are dark grey.
pub fn weight_gradient_colors(mesh: &SkinnedMesh, bone: u16) -> Vec<[f32; 4]> {
    (0..mesh.vertex_count())
        .map(|v| {
            let w = mesh.influences[v]
                .iter()
                .find(|i| i.bone == bone && i.weight > 0.0)
                .map(|i| i.weight);
            match w {
                Some(w) => {
                    let [r, g, b] = heat(w);
                    [r, g, b, 1.0]
                }
                None => [0.15, 0.15, 0.18, 1.0],
            }
        })
        .collect()
}

/// Produce vertex colours for the chosen [`WeightView`]. `active_bone` is only
/// used by [`WeightView::Gradient`].
pub fn weight_colors(mesh: &SkinnedMesh, view: WeightView, active_bone: Option<u16>) -> Vec<[f32; 4]> {
    match view {
        WeightView::DominantBone => dominant_bone_colors(mesh),
        WeightView::Gradient => match active_bone {
            Some(b) => weight_gradient_colors(mesh, b),
            None => vec![[1.0, 1.0, 1.0, 1.0]; mesh.vertex_count()],
        },
        WeightView::Off => vec![[1.0, 1.0, 1.0, 1.0]; mesh.vertex_count()],
    }
}

/// Tint selected vertices toward a highlight colour, in place. `amount` (0..1)
/// is how strongly the highlight overrides the base colour.
pub fn blend_selection(colors: &mut [[f32; 4]], sel: &VertexSelection, highlight: [f32; 3], amount: f32) {
    let a = amount.clamp(0.0, 1.0);
    for v in sel.iter() {
        if let Some(c) = colors.get_mut(v as usize) {
            for k in 0..3 {
                c[k] = c[k] * (1.0 - a) + highlight[k] * a;
            }
        }
    }
}

// ── colour helpers ────────────────────────────────────────────────────────────

/// Blue → cyan → green → yellow → red ramp over `t` in 0..1.
fn heat(t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    // piecewise across four segments
    let (r, g, b);
    if t < 0.25 {
        let k = t / 0.25;
        r = 0.0;
        g = k;
        b = 1.0;
    } else if t < 0.5 {
        let k = (t - 0.25) / 0.25;
        r = 0.0;
        g = 1.0;
        b = 1.0 - k;
    } else if t < 0.75 {
        let k = (t - 0.5) / 0.25;
        r = k;
        g = 1.0;
        b = 0.0;
    } else {
        let k = (t - 0.75) / 0.25;
        r = 1.0;
        g = 1.0 - k;
        b = 0.0;
    }
    [r, g, b]
}

/// HSV (all in 0..1) → linear RGB.
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h6 = (h.fract() * 6.0).clamp(0.0, 6.0);
    let i = h6.floor() as i32 % 6;
    let f = h6 - h6.floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::RigVertex;

    fn mesh3() -> SkinnedMesh {
        let v = |x: f32| RigVertex { position: [x, 0.0, 0.0], normal: [0.0, 0.0, 1.0], uv: [0.0, 0.0] };
        let mut m = SkinnedMesh::new(vec![v(0.0), v(1.0), v(2.0)], vec![0, 1, 2]);
        m.set_rigid(0, 0);
        m.set_rigid(1, 1);
        // vertex 2 left unweighted
        m
    }

    #[test]
    fn dominant_colors_distinguish_bones_and_unweighted() {
        let m = mesh3();
        let c = dominant_bone_colors(&m);
        assert_eq!(c.len(), 3);
        assert_ne!(c[0], c[1]); // different bones → different colours
        assert_eq!(c[2], [0.35, 0.35, 0.35, 1.0]); // unweighted grey
    }

    #[test]
    fn gradient_full_weight_is_hot() {
        let m = mesh3();
        let c = weight_gradient_colors(&m, 0);
        // vertex 0 has weight 1.0 for bone 0 → red end (r high, b zero)
        assert!(c[0][0] > 0.9 && c[0][2] < 0.1);
        // vertex 1 has no bone-0 influence → dark grey
        assert_eq!(c[1], [0.15, 0.15, 0.18, 1.0]);
    }

    #[test]
    fn selection_blend_moves_toward_highlight() {
        let m = mesh3();
        let mut c = dominant_bone_colors(&m);
        let before = c[0];
        let mut sel = VertexSelection::new(3);
        sel.set(0, true);
        blend_selection(&mut c, &sel, [1.0, 1.0, 1.0], 0.5);
        assert_ne!(c[0], before);
        assert!(c[1] == dominant_bone_colors(&m)[1]); // unselected unchanged
    }

    #[test]
    fn hsv_primary_colors() {
        assert_eq!(hsv_to_rgb(0.0, 1.0, 1.0), [1.0, 0.0, 0.0]); // red
        let g = hsv_to_rgb(1.0 / 3.0, 1.0, 1.0);
        assert!(g[1] > 0.9 && g[0] < 0.1); // green
    }
}
