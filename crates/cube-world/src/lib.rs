//! cube-world — Cube2/Sauerbraten octree geometry, editing, and serialization.
//! Zero renderer dependencies.  Produces plain vertex/index arrays for any renderer.

pub mod octree;
pub mod geometry;
pub mod editing;
pub mod serialize;
pub mod texture;
pub mod export;

pub use octree::{
    Cube, OctreeWorld, RayHit,
    subdivide, subdividecube, newcubes, discard_children, copycube, pastecube,
    MAT_AIR, MAT_WATER, MAT_LAVA, MAT_CLIP, MAT_GLASS,
    F_EMPTY, F_SOLID, EDGES_EMPTY, EDGES_SOLID,
    R, C, D, FV, FACEEDGESIDX, DEFAULT_GEOM,
    FACE_DIM, FACE_SIDE, O_LEFT, O_RIGHT, O_BACK, O_FRONT, O_BOTTOM, O_TOP,
    edge_get, edge_set, edge_idx, octaindex, octastep, oppositeocta,
};
pub use geometry::{build_mesh, build_mesh_with_textures, build_wireframe, MeshVertex};
pub use editing::{
    Selection, EditWorld, UndoStack, UndoBlock, CubeBlock,
    flipcube, rotatecube,
    Brush, HeightmapEditor, HeightmapParams,
    EDITMATF_EMPTY, EDITMATF_NOTEMPTY, EDITMATF_SOLID, EDITMATF_NOTSOLID,
};
pub use serialize::{load_ogz, save_ogz, OgzError};
pub use texture::{
    TextureRegistry, Slot, VSlot, SlotTex, SlotShaderParam, TexType, TexRotation,
    TEX_ROTATIONS, TEX_SCALE,
    VSLOT_SHPARAM, VSLOT_SCALE, VSLOT_ROTATION, VSLOT_OFFSET,
    VSLOT_SCROLL, VSLOT_LAYER, VSLOT_ALPHA, VSLOT_COLOR,
    calc_texgen, apply_texgen, load_texture_config,
};
pub use export::{export_glb, export_fbx, optimize_mesh};
