//! egui rendering backend for the BBC Vulkan renderer.
//!
//! Renders egui draw lists using Vulkan 1.3 dynamic rendering.
//! Supports a single font/atlas texture (TextureId::Managed(0)).

use ash::vk;
use gpu_allocator::MemoryLocation;
use tracing::info;

use crate::{
    memory::{AllocatedBuffer, AllocatedImage, GpuMemory},
    pipeline::create_shader_module,
    renderer::one_time_submit,
    sync::MAX_FRAMES_IN_FLIGHT,
};

/// Vulkan renderer for egui primitives.
pub struct EguiRenderer {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    desc_set_layout: vk::DescriptorSetLayout,
    desc_pool: vk::DescriptorPool,
    desc_set: vk::DescriptorSet,
    font_image: Option<AllocatedImage>,
    font_sampler: vk::Sampler,
    vertex_buffers: [Option<AllocatedBuffer>; MAX_FRAMES_IN_FLIGHT],
    index_buffers: [Option<AllocatedBuffer>; MAX_FRAMES_IN_FLIGHT],
}

impl EguiRenderer {
    /// Create the egui rendering pipeline.
    ///
    /// `color_format` must match the swapchain format used with dynamic rendering.
    /// The pipeline has no depth attachment — depth test and write are disabled.
    pub fn new(device: &ash::Device, color_format: vk::Format) -> Self {
        // ── Descriptor set layout: binding 0 = combined image sampler (font tex) ──
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let desc_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(std::slice::from_ref(&binding)),
                None,
            )
        }
        .unwrap();

        // ── Push constant: vec2 screen_size (8 bytes, vertex stage) ──
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(8); // 2 * f32

        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(std::slice::from_ref(&desc_set_layout))
                    .push_constant_ranges(std::slice::from_ref(&push_range)),
                None,
            )
        }
        .unwrap();

        // ── Shaders ──
        let vert_spirv = include_bytes!(concat!(env!("OUT_DIR"), "/egui.vert.spv"));
        let frag_spirv = include_bytes!(concat!(env!("OUT_DIR"), "/egui.frag.spv"));
        let vert_mod = create_shader_module(device, vert_spirv);
        let frag_mod = create_shader_module(device, frag_spirv);

        let pipeline = build_egui_pipeline(
            device,
            pipeline_layout,
            color_format,
            vert_mod,
            frag_mod,
        );

        unsafe {
            device.destroy_shader_module(vert_mod, None);
            device.destroy_shader_module(frag_mod, None);
        }

        // ── Sampler ──
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .min_lod(0.0)
            .max_lod(0.0);
        let font_sampler = unsafe { device.create_sampler(&sampler_info, None) }.unwrap();

        // ── Descriptor pool + set ──
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1);

        let desc_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(std::slice::from_ref(&pool_size))
                    .max_sets(1),
                None,
            )
        }
        .unwrap();

        let desc_set = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(desc_pool)
                    .set_layouts(std::slice::from_ref(&desc_set_layout)),
            )
        }
        .unwrap()[0];

        info!("EguiRenderer pipeline created");

        Self {
            pipeline,
            pipeline_layout,
            desc_set_layout,
            desc_pool,
            desc_set,
            font_image: None,
            font_sampler,
            vertex_buffers: [None, None],
            index_buffers: [None, None],
        }
    }

    /// Handle egui's `TexturesDelta` — upload/update the font atlas texture.
    ///
    /// Call this once per frame BEFORE `render()`.  The `cmd_pool` and `queue`
    /// are used for one-shot staging uploads.
    pub fn update_texture(
        &mut self,
        device: &ash::Device,
        memory: &GpuMemory,
        cmd_pool: vk::CommandPool,
        queue: vk::Queue,
        textures_delta: &egui::TexturesDelta,
    ) {
        for (tex_id, image_delta) in &textures_delta.set {
            // We only support the font atlas (Managed(0)) for now.
            if *tex_id != egui::TextureId::Managed(0) {
                continue;
            }

            // Full replacement only (partial sub-rect updates could be added later)
            if image_delta.pos.is_some() {
                // Sub-region update — skip for now, full texture will be re-set eventually
                continue;
            }

            let pixels = &image_delta.image;
            let (width, height, rgba_data) = match pixels {
                egui::ImageData::Color(color_image) => {
                    let w = color_image.width() as u32;
                    let h = color_image.height() as u32;
                    let mut data = Vec::with_capacity((w * h * 4) as usize);
                    for pixel in &color_image.pixels {
                        data.push(pixel.r());
                        data.push(pixel.g());
                        data.push(pixel.b());
                        data.push(pixel.a());
                    }
                    (w, h, data)
                }
                egui::ImageData::Font(font_image) => {
                    let w = font_image.width() as u32;
                    let h = font_image.height() as u32;
                    // Alpha-only: expand to RGBA (white + alpha)
                    let mut data = Vec::with_capacity((w * h * 4) as usize);
                    for &coverage in &font_image.pixels {
                        let alpha = (coverage * 255.0 + 0.5) as u8;
                        data.push(255);
                        data.push(255);
                        data.push(255);
                        data.push(alpha);
                    }
                    (w, h, data)
                }
            };

            // Destroy previous font image if any
            if let Some(mut old) = self.font_image.take() {
                memory.destroy_image(device, &mut old);
            }

            // Create new image
            let font_image = memory.create_image(
                device,
                width,
                height,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                vk::ImageAspectFlags::COLOR,
                "egui_font_atlas",
            );

            // Upload via staging buffer
            let byte_count = rgba_data.len() as u64;
            let staging = memory.create_buffer(
                device,
                byte_count,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
                "egui_font_staging",
            );

            unsafe {
                let dst = staging
                    .allocation
                    .as_ref()
                    .unwrap()
                    .mapped_ptr()
                    .unwrap()
                    .as_ptr() as *mut u8;
                std::ptr::copy_nonoverlapping(rgba_data.as_ptr(), dst, rgba_data.len());
            }

            one_time_submit(device, cmd_pool, queue, |cmd| {
                // UNDEFINED -> TRANSFER_DST
                let barrier = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(font_image.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                let dep = vk::DependencyInfo::default()
                    .image_memory_barriers(std::slice::from_ref(&barrier));
                unsafe { device.cmd_pipeline_barrier2(cmd, &dep) };

                let region = vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    });
                unsafe {
                    device.cmd_copy_buffer_to_image(
                        cmd,
                        staging.buffer,
                        font_image.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[region],
                    );
                }

                // TRANSFER_DST -> SHADER_READ_ONLY
                let barrier2 = vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(font_image.image)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                let dep2 = vk::DependencyInfo::default()
                    .image_memory_barriers(std::slice::from_ref(&barrier2));
                unsafe { device.cmd_pipeline_barrier2(cmd, &dep2) };
            });

            // Clean up staging
            let mut staging = staging;
            memory.destroy_buffer(device, &mut staging);

            // Update descriptor set
            let image_info = vk::DescriptorImageInfo::default()
                .sampler(self.font_sampler)
                .image_view(font_image.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

            let write = vk::WriteDescriptorSet::default()
                .dst_set(self.desc_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&image_info));

            unsafe { device.update_descriptor_sets(&[write], &[]) };

            self.font_image = Some(font_image);
        }

        // We don't need to handle textures_delta.free for the font atlas —
        // it's never freed during the application lifetime.
    }

    /// Record egui draw commands into the given command buffer.
    ///
    /// Must be called inside an active dynamic rendering pass (between
    /// `cmd_begin_rendering` and `cmd_end_rendering`).
    ///
    /// `frame_index` selects which double-buffered vertex/index buffers to use
    /// (0 or 1, matching the frame-in-flight index).
    ///
    /// `screen_size` is `[width_pixels, height_pixels]` of the render target.
    pub fn render(
        &mut self,
        device: &ash::Device,
        memory: &GpuMemory,
        cmd: vk::CommandBuffer,
        frame_index: usize,
        clipped_primitives: &[egui::ClippedPrimitive],
        screen_size: [f32; 2],
    ) {
        // Bail out if no font texture has been uploaded yet.
        if self.font_image.is_none() {
            return;
        }

        // ── Gather all vertices and indices ──
        let mut all_vertices: Vec<u8> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();
        let mut draw_calls: Vec<EguiDrawCall> = Vec::new();

        for clipped in clipped_primitives {
            match &clipped.primitive {
                egui::epaint::Primitive::Mesh(mesh) => {
                    if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                        continue;
                    }

                    let vertex_offset = all_vertices.len() / EGUI_VERTEX_SIZE;
                    let index_offset = all_indices.len() as u32;

                    // Append raw vertex bytes (egui::epaint::Vertex is 20 bytes)
                    let vert_bytes = unsafe {
                        std::slice::from_raw_parts(
                            mesh.vertices.as_ptr() as *const u8,
                            mesh.vertices.len() * EGUI_VERTEX_SIZE,
                        )
                    };
                    all_vertices.extend_from_slice(vert_bytes);
                    all_indices.extend_from_slice(&mesh.indices);

                    // Clamp scissor rect to viewport
                    let clip = clipped.clip_rect;
                    let x = clip.min.x.max(0.0) as i32;
                    let y = clip.min.y.max(0.0) as i32;
                    let w = (clip.max.x.min(screen_size[0]) - clip.min.x.max(0.0)).max(0.0) as u32;
                    let h = (clip.max.y.min(screen_size[1]) - clip.min.y.max(0.0)).max(0.0) as u32;

                    if w == 0 || h == 0 {
                        continue;
                    }

                    draw_calls.push(EguiDrawCall {
                        index_count: mesh.indices.len() as u32,
                        first_index: index_offset,
                        vertex_offset: vertex_offset as i32,
                        scissor: vk::Rect2D {
                            offset: vk::Offset2D { x, y },
                            extent: vk::Extent2D { width: w, height: h },
                        },
                    });
                }
                egui::epaint::Primitive::Callback(_) => {
                    // Custom render callbacks not supported
                }
            }
        }

        if draw_calls.is_empty() {
            return;
        }

        // ── Upload vertex buffer ──
        let vb_size = all_vertices.len() as u64;
        let ib_size = (all_indices.len() * std::mem::size_of::<u32>()) as u64;

        // Recreate if too small
        if self.vertex_buffers[frame_index]
            .as_ref()
            .map_or(true, |b| b.size < vb_size)
        {
            if let Some(mut old) = self.vertex_buffers[frame_index].take() {
                memory.destroy_buffer(device, &mut old);
            }
            // Allocate with some headroom to avoid frequent re-allocations
            let alloc_size = vb_size.max(64 * 1024).next_power_of_two();
            self.vertex_buffers[frame_index] = Some(memory.create_buffer(
                device,
                alloc_size,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                MemoryLocation::CpuToGpu,
                "egui_verts",
            ));
        }

        if self.index_buffers[frame_index]
            .as_ref()
            .map_or(true, |b| b.size < ib_size)
        {
            if let Some(mut old) = self.index_buffers[frame_index].take() {
                memory.destroy_buffer(device, &mut old);
            }
            let alloc_size = ib_size.max(64 * 1024).next_power_of_two();
            self.index_buffers[frame_index] = Some(memory.create_buffer(
                device,
                alloc_size,
                vk::BufferUsageFlags::INDEX_BUFFER,
                MemoryLocation::CpuToGpu,
                "egui_indices",
            ));
        }

        // Copy data into mapped buffers
        let vb = self.vertex_buffers[frame_index].as_ref().unwrap();
        let ib = self.index_buffers[frame_index].as_ref().unwrap();

        unsafe {
            let dst = vb
                .allocation
                .as_ref()
                .unwrap()
                .mapped_ptr()
                .unwrap()
                .as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(all_vertices.as_ptr(), dst, all_vertices.len());

            let dst = ib
                .allocation
                .as_ref()
                .unwrap()
                .mapped_ptr()
                .unwrap()
                .as_ptr() as *mut u8;
            let idx_bytes = bytemuck::cast_slice::<u32, u8>(&all_indices);
            std::ptr::copy_nonoverlapping(idx_bytes.as_ptr(), dst, idx_bytes.len());
        }

        // ── Record draw commands ──
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.desc_set],
                &[],
            );

            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(&screen_size),
            );

            device.cmd_bind_vertex_buffers(cmd, 0, &[vb.buffer], &[0]);
            device.cmd_bind_index_buffer(cmd, ib.buffer, 0, vk::IndexType::UINT32);

            for dc in &draw_calls {
                device.cmd_set_scissor(cmd, 0, &[dc.scissor]);
                device.cmd_draw_indexed(cmd, dc.index_count, 1, dc.first_index, dc.vertex_offset, 0);
            }
        }
    }

    /// Destroy all Vulkan resources owned by this renderer.
    pub fn destroy(&mut self, device: &ash::Device, memory: &GpuMemory) {
        for vb in &mut self.vertex_buffers {
            if let Some(buf) = vb.take() {
                let mut buf = buf;
                memory.destroy_buffer(device, &mut buf);
            }
        }
        for ib in &mut self.index_buffers {
            if let Some(buf) = ib.take() {
                let mut buf = buf;
                memory.destroy_buffer(device, &mut buf);
            }
        }
        if let Some(mut img) = self.font_image.take() {
            memory.destroy_image(device, &mut img);
        }
        unsafe {
            device.destroy_sampler(self.font_sampler, None);
            device.destroy_descriptor_pool(self.desc_pool, None);
            device.destroy_descriptor_set_layout(self.desc_set_layout, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

// ── Private helpers ──────────────────────────────────────────────────────────

/// Size of `egui::epaint::Vertex` in bytes: pos(8) + uv(8) + color(4) = 20.
const EGUI_VERTEX_SIZE: usize = 20;

struct EguiDrawCall {
    index_count: u32,
    first_index: u32,
    vertex_offset: i32,
    scissor: vk::Rect2D,
}

fn build_egui_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    color_format: vk::Format,
    vert_mod: vk::ShaderModule,
    frag_mod: vk::ShaderModule,
) -> vk::Pipeline {
    let entry = c"main";
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_mod)
            .name(entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_mod)
            .name(entry),
    ];

    // egui vertex: pos(R32G32), uv(R32G32), color(R8G8B8A8_UNORM)
    let attrs = [
        vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: 0,
        },
        vk::VertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: 8,
        },
        vk::VertexInputAttributeDescription {
            location: 2,
            binding: 0,
            format: vk::Format::R8G8B8A8_UNORM,
            offset: 16,
        },
    ];

    let bindings = [vk::VertexInputBindingDescription {
        binding: 0,
        stride: EGUI_VERTEX_SIZE as u32,
        input_rate: vk::VertexInputRate::VERTEX,
    }];

    let vert_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_attribute_descriptions(&attrs)
        .vertex_binding_descriptions(&bindings);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let msaa = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    // No depth test, no depth write
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false);

    // Pre-multiplied alpha blending: src_alpha, one_minus_src_alpha
    let blend_attach = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD);

    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(std::slice::from_ref(&blend_attach));

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let color_formats = [color_format];
    let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&color_formats);
    // No depth attachment format — egui doesn't use depth

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vert_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&msaa)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .push_next(&mut rendering_info);

    unsafe {
        device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .expect("Failed to create egui pipeline")[0]
    }
}
