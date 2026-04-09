# Cube2 Octree Geometry System — Rust Port Documentation

## Overview

This document describes the octree-based geometry system ported from **Cube 2: Sauerbraten** (C++) to Rust, as implemented in the `cube-world` crate and the BBC editor frontend. It covers the data structures, editing operations, rendering pipeline, and provides guidance for integrating this system into other Rust-based game engines as a modular volume editing tool.

---

## 1. The Octree Data Model

### 1.1 World Structure

The world is a single octree rooted at a power-of-two world size (e.g., 1024 units = 2^10). The root contains 8 child cubes. Each child can either be a **leaf** (storing geometry) or a **branch** (subdivided into 8 smaller children), recursively down to size 1.

```
OctreeWorld
├── root: [Cube; 8]          // 8 root octants
├── world_scale: u32         // log2(world_size), e.g., 10 for 1024
└── world_size(): i32        // 1 << world_scale
```

### 1.2 The Cube

Each `Cube` stores:

```rust
pub struct Cube {
    pub children: Option<Box<[Cube; 8]>>,  // None = leaf
    pub edges: [u8; 12],                    // 12 edge deformation bytes
    pub texture: [u16; 6],                  // texture index per face
    pub material: u8,                       // material (air, water, etc.)
    pub merged: u8,                         // face merge flags for rendering
}
```

**Edge encoding:** Each of the 12 edges stores two 4-bit values (endpoints), ranging 0–8. These define how far each edge endpoint is pushed inward from the cube's outer boundary. An edge byte of `0x08` means: start at 0/8 (flush left), end at 8/8 (flush right) — a full edge. `0x44` means both endpoints at 4/8 — a zero-width edge at the midpoint.

**Face convention:**
- 3 axes × 2 sides = 6 faces (orients)
- `O_LEFT=0` (-X), `O_RIGHT=1` (+X), `O_BACK=2` (-Y), `O_FRONT=3` (+Y), `O_BOTTOM=4` (-Z), `O_TOP=5` (+Z)
- Each face is defined by 4 edges (2×2 grid on that face)

**Special states:**
- `F_SOLID` (all edges 0→8): fully solid cube
- `F_EMPTY` (all edges 0→0): fully empty (air)
- Anything in between: deformed geometry

### 1.3 Axis Helpers

```rust
pub const FACE_DIM: [usize; 6]  = [0, 0, 1, 1, 2, 2];   // which axis this orient is on
pub const FACE_SIDE: [usize; 6] = [0, 1, 0, 1, 0, 1];    // 0=negative, 1=positive side
pub const D: [usize; 3] = [0, 1, 2];  // depth axis (same as dim)
pub const R: [usize; 3] = [1, 2, 0];  // row axis (perpendicular)
pub const C: [usize; 3] = [2, 0, 1];  // column axis (perpendicular)
```

For a face on axis `d`:
- `D[d]` = the axis perpendicular to the face (depth/extrusion axis)
- `R[d]` = first tangent axis (row)
- `C[d]` = second tangent axis (column)

### 1.4 Edge Indexing

```rust
pub fn edge_idx(dim: usize, x: usize, y: usize) -> usize {
    dim * 4 + (y * 2 + x)  // 0..12
}
pub fn edge_get(edge: u8, which: usize) -> u8;  // get endpoint 0 or 1 (4-bit)
pub fn edge_set(edge: &mut u8, which: usize, val: u8);  // set endpoint
```

Each face has a 2×2 grid of edges (`x=0..2, y=0..2`), giving 4 edges per face × 3 faces = 12 edges total.

---

## 2. Octree Traversal & Lookup

### 2.1 Read-Only Lookup

`lookup(x, y, z)` descends to the **leaf** at position (x,y,z), returning the cube, its origin, and its size.

`lookup_at(x, y, z, target_size)` descends only to `target_size` level, returning a cube that may still have children. This is essential for **copy operations** that need to preserve sub-grid detail.

### 2.2 Mutable Lookup with Auto-Subdivision

`lookup_mut(x, y, z, target_size)` descends and **auto-subdivides** any leaf larger than `target_size`, ensuring a cube exists at exactly that resolution. This is the workhorse for all editing operations.

### 2.3 Subdivision

When a solid or empty leaf must be split into 8 children:
- **Simple case** (solid/empty): 8 children inherit the parent's edges and textures
- **Deformed case** (`subdividecube`): Edge values are interpolated to the midpoint (value 8 maps to center), creating 8 children whose combined geometry matches the parent's shape

---

## 3. Geometry Editing Operations

All editing is implemented in `EditWorld` (wraps `OctreeWorld` + undo stack + dirty flag).

### 3.1 Face Editing — `editface(sel, dir, mode)`

The core operation, ported from C++'s `mpeditface`:

| Mode | Name | Behavior |
|------|------|----------|
| 0 | Edge push | Push/pull individual edges on the selected face. Controlled by sub-face selection (`cx/cy/cxs/cys`). |
| 1 | Fill/empty | Add or remove whole cubes along the face normal. Copies textures from adjacent cubes when filling. Auto-downgrades to mode 0 if sub-face selection detected. |
| 2 | Corner push | Push/pull a single corner vertex via `linkedpush`, which moves all edges sharing the same vertex position. |

**`linkedpush`**: When pushing corner (x,y) on face `d`, finds all edges on that face whose vertex position matches, and pushes them together. This prevents cracks.

**`pushedge`**: Adjusts one endpoint of an edge byte, clamping to [0,8] and maintaining start ≤ end invariant.

**Direction logic:**
```
seldir = dc ? -dir : dir    // dc = face side (0 or 1)
```
- `dir > 0` with fill mode: empty cubes (remove material)
- `dir < 0` with fill mode: solid cubes (add material)
- For edge/corner: `seldir` determines push direction

### 3.2 Sub-Face Selection (`cx/cy/cxs/cys`)

Doubled-grid coordinates that specify a sub-region within the face:
- `cx, cy`: start offset (0 = edge, 1 = half-cell inset)
- `cxs, cys`: size (1 = half-face, 2 = full-face)
- **LMB selection**: rounds to even coords → full face per cube
- **RMB/MMB selection**: preserves odd coords → corner-precise selection

When `cx != 0 || cy != 0 || cxs&1 || cys&1`, fill mode auto-downgrades to edge push, and the edge filter skips outer boundary edges based on position within the selection.

### 3.3 Rotation — `rotate(cw, sel)`

Ported from `mprotate`:
1. Selection is **squared** (max of R/C dimensions)
2. Each cube's internal geometry is rotated via `rotatecube` (edge/texture permutation)
3. The grid of cubes is rotated via shell rotation (`rotatequad`)
4. Counter-clockwise = 3 clockwise iterations

### 3.4 Flip — `flip(sel)`

Ported from `mpflip`:
1. Each cube's internal geometry is flipped via `flipcube` (edge/texture mirror along depth axis)
2. Cubes are swapped in pairs along the depth axis (mirror the grid)

### 3.5 Copy/Paste

**Copy** (`blockcopy`):
- Uses `lookup_at(pos, sel.grid)` to read cubes at grid resolution, preserving sub-grid children
- Deep-copies via `copycube` (recursive on children)
- Stores cubes + selection metadata in a `CubeBlock`

**Paste** (`pasteblock`):
- Iterates over the destination selection
- `lookup_mut(pos, sel.grid)` auto-subdivides destination to grid size
- `pastecube` deep-copies source into destination, discarding old children

### 3.6 Undo/Redo

- Before each edit, `make_undo(sel)` snapshots the selection region
- Stores a `CubeBlock` + `gridmap` (log2 of actual leaf size at each cell)
- `paste_undo_block` restores cubes at their original resolution using the gridmap
- Undo/redo swap: pop from undo stack, snapshot current state to redo stack, paste old state

### 3.7 Delete, Texture Edit, Material Edit

- **Delete**: empties all cubes in selection
- **Texture**: sets texture index on selected faces
- **Material**: sets material byte on selected cubes

---

## 4. The Selection System

### 4.1 Selection Struct

```rust
pub struct Selection {
    pub origin: [i32; 3],   // world-space origin (grid-aligned)
    pub size: [i32; 3],     // number of grid cells per axis
    pub grid: i32,          // grid cell size (power of 2)
    pub orient: usize,      // active face (0-5)
    pub corner: usize,      // which corner (0-3) of the face
    pub cx, cy: i32,        // sub-face start offset (doubled-grid coords)
    pub cxs, cys: i32,      // sub-face size (doubled-grid coords)
}
```

### 4.2 Selection Flow

1. **Hover**: raycast from camera through screen center hits a cube face → creates default 1×1×1 selection at hover position
2. **LMB drag**: start + drag creates a multi-cube face selection (selectcorners=false, full-face mode)
3. **RMB/MMB drag**: start + drag creates a sub-face vertex selection (selectcorners=true, corner-precise mode)
4. **After editing**: selection is "locked" (`have_sel=true`), preventing hover from overwriting it
5. **Space**: cancels selection, returns to hover mode
6. **Scroll**: applies editing operation and optionally advances selection origin for multi-step extrusion

### 4.3 blockcube_coords

Maps selection-relative (x, y, z) indices to world coordinates, accounting for orient direction:
```rust
fn blockcube_coords(sel, x, y, z) -> (i32, i32, i32)
// x iterates along R[dim], y along C[dim], z along D[dim]
```

---

## 5. Rendering

### 5.1 Mesh Generation

The octree is traversed to generate triangle meshes. For each leaf cube that isn't empty:

1. **`visibleface(orient)`**: determines if a face is visible by checking if the adjacent cube on that side is solid
2. **`genfaceverts(orient, edges)`**: reads 4 edge endpoints to compute the face's 4 corner positions
3. **Triangle generation**: each visible face produces 2 triangles (or 1 if degenerate)
4. **Normal computation**: cross product of face edges
5. **Texture coordinates**: derived from vertex positions on the face plane

### 5.2 Coordinate Systems

- **Cube2 internal**: Z-up (X=right, Y=forward, Z=up)
- **Renderer (Vulkan)**: Y-up (X=right, Y=up, Z=back)
- **Conversion**: swap Y↔Z when outputting vertex positions

### 5.3 Wireframe and Overlay

- Solid mesh: standard filled triangles with per-face textures
- Wire mesh: LINE_LIST edges for all visible faces
- Editor overlay: cursor box (gray), face highlight (white), selection box (blue), grid lines, origin marker (red), sub-face selection bracket

### 5.4 GPU Pipeline

Built on Vulkan via `ash`:
- Forward rendering pipeline with fill + wireframe + line modes
- Swapchain with mailbox present mode
- 2 frames in flight with frame-delayed deletion queue (3 slots) for safe mesh lifecycle
- Screenshot capture via staging buffer + image copy

---

## 6. Provenance and Borrowed Components

### 6.1 From Cube2/Sauerbraten (C++)

The following were directly ported from `octaedit.cpp`, `octa.cpp`, and related Cube2 source:

- **Octree data structure**: `Cube`, 12-edge encoding, face conventions
- **Subdivision**: `subdividecube` with midpoint edge interpolation
- **Editing operations**: `editface` (modes 0/1/2), `linkedpush`, `pushedge`
- **Rotation/flip**: `rotatecube`, `flipcube`, `rotatequad`, shell rotation
- **Copy/paste**: `blockcopy`, `pasteblock`, `copycube`, `pastecube`
- **Undo/redo**: `undoblock` with gridmap, `swapundo` pattern
- **OGZ serialization**: loading `.ogz` map files (gzip-compressed octree + entities)
- **Raycast**: axis-aligned slab intersection against the octree
- **Selection system**: `rendereditcursor`, `selectcorners`, doubled-grid coords
- **Face visibility**: `visibleface`, `genfaceverts`, `visibletris`

### 6.2 From the BBC Renderer (Original Rust)

- **Vulkan renderer**: instance, device, swapchain, pipeline, memory allocation (via `gpu-allocator`)
- **FlyCamera**: position, yaw/pitch, WASD movement, mouse look
- **Shader pipeline**: vertex/fragment GLSL shaders compiled to SPIR-V at build time
- **Input manager**: centralized `InputState` with keyboard, mouse, scroll tracking
- **Screenshot**: staging buffer capture + bitmap font text overlay

### 6.3 Integration Bridges

- **Coordinate conversion**: Y↔Z swap between Cube2 Z-up and Vulkan Y-up at mesh generation and raycast boundaries
- **Orient mapping**: raycast returns Cube2-space orients directly (no conversion needed since raycast does internal coord swap)
- **Mesh rebuild**: full octree re-meshing on each edit (simple but effective for current scale)

---

## 7. Integration Guide — Using as an Editing Module in Another Engine

### 7.1 Architecture

The system is designed with clean separation:

```
┌─────────────────────┐
│  Your Game Engine    │
│  (Bevy, wgpu, etc.) │
│                      │
│  ┌────────────────┐  │
│  │ cube-world     │  │  ← Pure Rust, no rendering dependency
│  │ (editing crate) │  │
│  └────────────────┘  │
│         ↕            │
│  ┌────────────────┐  │
│  │ Your Renderer   │  │  ← Generates meshes from octree
│  └────────────────┘  │
│         ↕            │
│  ┌────────────────┐  │
│  │ Your Input      │  │  ← Maps engine input to edit commands
│  └────────────────┘  │
└─────────────────────┘
```

The `cube-world` crate has **zero rendering dependencies**. It only deals with octree data and editing operations. Your engine provides:
1. A raycast (or you use the built-in octree raycast)
2. Input translation (scroll, mouse buttons → edit commands)
3. Mesh generation from the octree (the crate provides traversal helpers)

### 7.2 Embedding as Editable Volumes

For arbitrary volumes that can be transformed, rotated, and moved within a larger world:

```rust
/// A placeable octree volume in your game world.
struct OctreeVolume {
    /// The octree geometry (from cube-world crate)
    edit_world: EditWorld,

    /// Transform in your engine's world space
    position: Vec3,
    rotation: Quat,
    scale: f32,

    /// Cached GPU mesh (regenerate when edit_world.dirty)
    mesh: Option<YourMeshHandle>,

    /// Selection state for in-world editing
    selection: Option<Selection>,
    grid_power: u32,
}
```

**Key integration points:**

1. **Raycast**: Transform the ray from engine world space into the volume's local space before calling `edit_world.world.raycast()`. Apply inverse transform: `local_ray = inverse(volume_transform) * world_ray`.

2. **Mesh generation**: Traverse the octree, generate vertices in local space, upload as a mesh. Apply the volume's transform as a model matrix in your shader.

3. **Editing**: All editing operations work in the octree's local coordinate system. The volume's world-space transform is irrelevant to the editing logic.

4. **Multiple volumes**: Each `OctreeVolume` is independent. You can have many in a scene, each with its own octree, selection state, and undo history.

### 7.3 Coordinate System Adaptation

The octree internally uses Cube2's Z-up convention. When integrating with a Y-up engine:

```rust
// When generating mesh vertices:
fn octree_to_engine(cube2_pos: [f32; 3]) -> Vec3 {
    Vec3::new(cube2_pos[0], cube2_pos[2], cube2_pos[1])  // swap Y↔Z
}

// When converting raycasts:
fn engine_to_octree(engine_pos: Vec3) -> [f32; 3] {
    [engine_pos.x, engine_pos.z, engine_pos.y]  // swap Y↔Z
}
```

### 7.4 Bevy Integration Example (Conceptual)

```rust
#[derive(Component)]
struct OctreeVolume {
    edit_world: EditWorld,
    grid_power: u32,
}

fn handle_editing(
    mut volumes: Query<(&mut OctreeVolume, &Transform)>,
    input: Res<Input<MouseButton>>,
    camera: Query<&Camera>,
) {
    let ray = camera.screen_to_world_ray(cursor_pos);

    for (mut vol, transform) in volumes.iter_mut() {
        // Transform ray into volume's local space
        let local_ray = transform.inverse() * ray;
        let hit = vol.edit_world.world.raycast(local_ray.origin, local_ray.dir);

        if let Some(hit) = hit {
            // Apply edit operations in local space
            let sel = build_selection_from_hit(&hit, vol.grid_power);
            vol.edit_world.editface(&sel, dir, mode);
        }
    }
}
```

### 7.5 Performance Considerations

- **Mesh rebuild**: Currently does full octree traversal on each edit. For large worlds, consider incremental meshing (track which octree branches changed).
- **LOD**: The octree naturally supports level-of-detail — render coarser cubes for distant volumes.
- **Serialization**: Use the existing OGZ format, or implement a simpler binary format for your volumes.
- **Memory**: Each leaf cube is ~30 bytes. A fully subdivided 1024³ world is impractical; rely on the octree's spatial compression.

---

## 8. File Map

```
crates/cube-world/
├── src/
│   ├── octree.rs      — Cube struct, octree traversal, lookup, subdivision,
│   │                     edge helpers, raycast, mesh generation (genfaceverts)
│   ├── editing.rs     — Selection, EditWorld, editface, rotate, flip,
│   │                     copy/paste, undo/redo, heightmap (partial)
│   ├── serialize.rs   — OGZ format loading (Sauerbraten map files)
│   └── lib.rs         — Public API re-exports

src/
├── main.rs            — Event loop, Vulkan rendering, mesh rebuild, screenshot
├── editor.rs          — EditorState, input→editing dispatch, overlay geometry,
│                        selection handling, coordinate conversion
├── input.rs           — InputState (keyboard/mouse/scroll tracking)
└── screenshot.rs      — Bitmap font renderer for debug text overlay

crates/bbc-renderer/
└── src/
    ├── renderer.rs    — Vulkan renderer (swapchain, pipeline, mesh upload)
    ├── instance.rs    — Vulkan instance + GPU selection
    ├── device.rs      — Logical device + queue
    ├── swapchain.rs   — Swapchain management
    ├── pipeline.rs    — Graphics pipeline (fill + wire + line)
    ├── memory.rs      — GPU memory allocator wrapper
    ├── commands.rs    — Command buffer pool
    ├── sync.rs        — Frame synchronization (fences, semaphores)
    └── camera.rs      — FlyCamera (position, orientation, movement)
```

---

## 9. Current Status and Known Gaps

### Working:
- ✅ Octree loading from OGZ format
- ✅ Full mesh generation with normals and texcoords
- ✅ Face push/pull (editface mode 0) with sub-face selection
- ✅ Fill/empty (editface mode 1) with texture copy from adjacent
- ✅ Corner push (editface mode 2) via linkedpush
- ✅ RMB/MMB vertex selection → scroll directly pushes selected vertices
- ✅ Multi-step extrusion (selection advance)
- ✅ Grid size adjustment (G+Scroll)
- ✅ Copy/paste with subtree preservation
- ✅ Rotation (R+Scroll) with correct undo area
- ✅ Flip (X)
- ✅ Undo/redo with gridmap
- ✅ Visual overlay (cursor, selection, grid lines, sub-face bracket)
- ✅ Screenshot with debug text

### Not Yet Ported:
- ❌ VSlot editing (texture rotation/scale/offset/color/alpha)
- ❌ Heightmap editing (struct exists, not wired to input)
- ❌ Material editing (function exists, no keybinding)
- ❌ Selection moving (drag selection to new position)
- ❌ `pushsel` / `reorient` / `selextend` commands
- ❌ Blend map painting
- ❌ OGZ saving (only loading is implemented)
- ❌ Incremental mesh rebuild (currently full rebuild on each edit)
- ❌ Entity system (lights, spawns, etc. — loaded but not rendered)
