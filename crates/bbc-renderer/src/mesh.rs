//! Vertex format and GPU mesh upload for the BBC renderer.

use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::MemoryLocation;
use crate::memory::{AllocatedBuffer, GpuMemory};

/// 48-byte vertex — position, normal, UV, and a face color for selection highlights.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],  // 12
    pub normal:   [f32; 3],  // 12
    pub uv:       [f32; 2],  //  8
    pub color:    [f32; 4],  // 16  (RGBA; used for face tint + selection highlight)
}

impl Vertex {
    pub fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2], color: [f32; 4]) -> Self {
        Self { position, normal, uv, color }
    }
}

/// Vertex input attribute descriptions for the pipeline.
pub fn vertex_attributes() -> [vk::VertexInputAttributeDescription; 4] {
    [
        vk::VertexInputAttributeDescription {
            location: 0, binding: 0,
            format: vk::Format::R32G32B32_SFLOAT, offset: 0,
        },
        vk::VertexInputAttributeDescription {
            location: 1, binding: 0,
            format: vk::Format::R32G32B32_SFLOAT, offset: 12,
        },
        vk::VertexInputAttributeDescription {
            location: 2, binding: 0,
            format: vk::Format::R32G32_SFLOAT, offset: 24,
        },
        vk::VertexInputAttributeDescription {
            location: 3, binding: 0,
            format: vk::Format::R32G32B32A32_SFLOAT, offset: 32,
        },
    ]
}

pub fn vertex_binding() -> vk::VertexInputBindingDescription {
    vk::VertexInputBindingDescription {
        binding: 0,
        stride: std::mem::size_of::<Vertex>() as u32,
        input_rate: vk::VertexInputRate::VERTEX,
    }
}

/// A mesh uploaded to GPU memory (CpuToGpu, no staging buffer — fine for
/// geometry that is rebuilt on edits).
pub struct GpuMesh {
    pub vertex_buffer: AllocatedBuffer,
    pub index_buffer:  AllocatedBuffer,
    pub index_count:   u32,
}

impl GpuMesh {
    pub fn upload(
        device:   &ash::Device,
        memory:   &GpuMemory,
        vertices: &[Vertex],
        indices:  &[u32],
    ) -> Self {
        let vsize = (vertices.len() * std::mem::size_of::<Vertex>()) as u64;
        let isize = (indices.len()  * std::mem::size_of::<u32>())    as u64;

        let vertex_buffer = memory.create_buffer(
            device, vsize,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            MemoryLocation::CpuToGpu,
            "cube_verts",
        );
        let index_buffer = memory.create_buffer(
            device, isize,
            vk::BufferUsageFlags::INDEX_BUFFER,
            MemoryLocation::CpuToGpu,
            "cube_indices",
        );

        // Write vertex data
        let vdata = bytemuck::cast_slice::<Vertex, u8>(vertices);
        unsafe {
            let dst = vertex_buffer.allocation.as_ref().unwrap()
                .mapped_ptr().unwrap().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(vdata.as_ptr(), dst, vdata.len());
        }

        // Write index data
        let idata = bytemuck::cast_slice::<u32, u8>(indices);
        unsafe {
            let dst = index_buffer.allocation.as_ref().unwrap()
                .mapped_ptr().unwrap().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(idata.as_ptr(), dst, idata.len());
        }

        GpuMesh { vertex_buffer, index_buffer, index_count: indices.len() as u32 }
    }

    pub fn destroy(&mut self, device: &ash::Device, memory: &GpuMemory) {
        memory.destroy_buffer(device, &mut self.vertex_buffer);
        memory.destroy_buffer(device, &mut self.index_buffer);
    }
}
