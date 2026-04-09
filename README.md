# Cube Bauhaus

**Standalone octree CSG geometry editor — build levels for any engine.**

Cube Bauhaus is a real-time octree geometry editor reimplemented in Rust/Vulkan, based on the legendary Cube2/Sauerbraten editing system. It runs as a standalone application and exports optimized meshes with textures to standard formats (GLB, FBX) for use in Unreal Engine, Unity, Godot, or any other game engine.

The Cube2 octree CSG system is one of the most elegant real-time geometry editing systems ever created. This project frees that technology from its original engine and makes it available to modern game development.

## Features

- **Real-time octree CSG editing** — fill, push, corner drag, copy/paste, undo/redo
- **Texture system** — slot cycling, rotate, scale, offset per face
- **Material support** — air, water, lava, clip, glass
- **Sauerbraten map loading** — reads OGZ v33 files with full texture loading from .cfg
- **OGZ save/load** — native format roundtrip with file dialogs
- **GLB export** — binary glTF 2.0 with embedded textures and PBR materials
- **FBX export** — ASCII FBX 7.4 with per-slot materials and texture references
- **Mesh optimization** — vertex deduplication, degenerate triangle removal, full interior face occlusion culling
- **Vulkan renderer** — GPU texture arrays, anisotropic filtering, dynamic rendering
- **egui overlay** — menu bar, status bar, mouse-driven UI via AltGr toggle

## Controls

| Key | Action |
|-----|--------|
| E | Toggle edit mode |
| WASD | Move camera |
| Mouse | Look (click to capture, Esc to release) |
| Scroll | Fill/push geometry |
| LMB/RMB | Select face / Select vertices |
| 1+Scroll | Cycle texture slot |
| 2+Scroll | Rotate texture |
| 3+Scroll | Scale texture |
| 4+Scroll | Offset texture (Shift = 2nd axis) |
| 5+Scroll | Cycle material |
| C / V | Copy / Paste |
| Z / I | Undo / Redo |
| X | Flip selection |
| Del | Delete selection |
| G+Scroll | Grid size |
| F | Toggle wireframe |
| Ctrl+S | Save OGZ |
| Ctrl+O | Open OGZ |
| Ctrl+N | New map |
| Ctrl+E | Export GLB |
| Ctrl+Shift+E | Export FBX |
| AltGr | Toggle UI mouse mode |

## Building

Requires Rust 1.75+ and the Vulkan SDK.

```bash
cargo build --release
```

## Usage

```bash
# New empty map
cube-bauhaus

# Load a Sauerbraten map
cube-bauhaus path/to/map.ogz
```

## Architecture

```
cube-bauhaus/
  src/              # Application: editor state, input, main loop
  crates/
    cube-world/     # Pure geometry: octree, editing, serialize, export (no GPU deps)
    bbc-renderer/   # Vulkan renderer: pipelines, shaders, egui integration
```

The `cube-world` crate has zero renderer dependencies — it produces plain vertex/index arrays that any renderer can consume. This makes it reusable as a library for engine plugins.

## Acknowledgments

Algorithms and file formats reimplemented from [Cube2/Sauerbraten](http://sauerbraten.org) by Wouter van Oortmerssen (zlib license). This is an independent Rust reimplementation — no original C++ source code is included.

## License

MIT — see [LICENSE](LICENSE) for details.

The editor and export pipeline are free and open source. A commercial Unreal Engine plugin for real-time in-editor CSG editing is in development separately.
