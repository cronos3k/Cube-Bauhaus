//! bbc-renderer — minimal Vulkan forward renderer for BBC.
//! Borrowed from woi2/vk-renderer, stripped to the essentials.

pub mod camera;
pub mod commands;
pub mod device;
pub mod egui_integration;
pub mod instance;
pub mod memory;
pub mod mesh;
pub mod pipeline;
pub mod renderer;
pub mod swapchain;
pub mod sync;

// Convenience re-exports
pub use camera::{CameraUniform, FlyCamera};
pub use egui_integration::EguiRenderer;
pub use mesh::{GpuMesh, Vertex};
pub use renderer::{Renderer, one_time_submit};
pub use sync::MAX_FRAMES_IN_FLIGHT;
