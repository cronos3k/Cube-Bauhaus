//! High-level Renderer — owns all Vulkan state, exposes a simple per-frame API.

use ash::vk;
use bytemuck::bytes_of;
use gpu_allocator::MemoryLocation;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use tracing::info;
use winit::window::Window;

use crate::{
    camera::CameraUniform,
    commands::CommandManager,
    device::{create_device, find_queue_families, QueueFamilies},
    egui_integration::EguiRenderer,
    instance::{create_instance, select_physical_device},
    memory::{AllocatedBuffer, AllocatedImage, GpuMemory},
    mesh::{GpuMesh, Vertex},
    pipeline::{DescriptorSets, ForwardPipeline},
    swapchain::Swapchain,
    sync::{FrameSync, MAX_FRAMES_IN_FLIGHT},
};

pub struct Renderer {
    // Vulkan core
    pub entry:           ash::Entry,
    pub instance:        ash::Instance,
    pub surface:         vk::SurfaceKHR,
    pub surface_loader:  ash::khr::surface::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub device:          ash::Device,
    pub graphics_queue:  vk::Queue,
    pub queue_families:  QueueFamilies,

    // Frame infrastructure
    pub swapchain: Swapchain,
    pub sync:      FrameSync,
    pub commands:  CommandManager,
    pub memory:    GpuMemory,

    // Depth buffer (single, reused across frames)
    pub depth_image: AllocatedImage,

    // Pipeline + descriptors
    pub pipeline:  ForwardPipeline,
    pub desc_sets: DescriptorSets,
    pub cam_ubos:  [AllocatedBuffer; MAX_FRAMES_IN_FLIGHT],

    // Texture array (optional — present when map textures loaded)
    pub texture_array: Option<AllocatedImage>,
    pub texture_sampler: Option<vk::Sampler>,

    // State
    pub wireframe: bool,
}

impl Renderer {
    pub fn new(window: &Window) -> Self {
        let entry = unsafe { ash::Entry::load() }.expect("Failed to load Vulkan");

        // Instance + surface
        let instance = create_instance(&entry, window);
        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
        let surface = unsafe {
            ash_window::create_surface(
                &entry, &instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
        }
        .expect("Failed to create surface");

        // Device
        let physical_device = select_physical_device(&instance);
        let queue_families =
            find_queue_families(&instance, physical_device, &surface_loader, surface);
        let (device, graphics_queue) =
            create_device(&instance, physical_device, &queue_families);

        // Swapchain
        let size = window.inner_size();
        let swapchain = Swapchain::new(
            &instance, &device, physical_device, surface, &surface_loader,
            queue_families.graphics, size.width, size.height,
        );

        // Memory / sync / commands
        let memory   = GpuMemory::new(&instance, &device, physical_device);
        let sync     = FrameSync::new(&device);
        let commands = CommandManager::new(&device, queue_families.graphics);

        // Depth buffer
        let depth_image = Self::create_depth_image(
            &device, &memory, &commands, graphics_queue,
            swapchain.extent.width, swapchain.extent.height,
        );

        // Camera UBOs (one per frame in flight)
        let ubo_size = std::mem::size_of::<CameraUniform>() as u64;
        let cam_ubos = [
            memory.create_buffer(&device, ubo_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER, MemoryLocation::CpuToGpu, "cam_ubo_0"),
            memory.create_buffer(&device, ubo_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER, MemoryLocation::CpuToGpu, "cam_ubo_1"),
        ];

        // Pipeline + descriptors
        let pipeline = ForwardPipeline::new(
            &device, swapchain.format.format, vk::Format::D32_SFLOAT,
        );
        let desc_sets = DescriptorSets::new(&device, pipeline.desc_set_layout, &cam_ubos);

        // Create a 1-layer dummy texture array so the descriptor binding is always valid
        let dummy_pixels: Vec<u8> = vec![128, 128, 128, 255]; // 1x1 gray
        let dummy_image = memory.create_image_array(
            &device, 1, 1, 1,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            "dummy_tex",
        );
        // Upload the dummy pixel
        let dummy_staging = memory.create_buffer(
            &device, 4,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu, "dummy_staging",
        );
        unsafe {
            let dst = dummy_staging.allocation.as_ref().unwrap()
                .mapped_ptr().unwrap().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(dummy_pixels.as_ptr(), dst, 4);
        }
        one_time_submit(&device, commands.pool, graphics_queue, |cmd| {
            let barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(dummy_image.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0, level_count: 1,
                    base_array_layer: 0, layer_count: 1,
                });
            let dep = vk::DependencyInfo::default()
                .image_memory_barriers(std::slice::from_ref(&barrier));
            unsafe { device.cmd_pipeline_barrier2(cmd, &dep) };

            let region = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0, base_array_layer: 0, layer_count: 1,
                })
                .image_extent(vk::Extent3D { width: 1, height: 1, depth: 1 });
            unsafe {
                device.cmd_copy_buffer_to_image(
                    cmd, dummy_staging.buffer, dummy_image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[region],
                );
            }

            let barrier2 = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(dummy_image.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0, level_count: 1,
                    base_array_layer: 0, layer_count: 1,
                });
            let dep2 = vk::DependencyInfo::default()
                .image_memory_barriers(std::slice::from_ref(&barrier2));
            unsafe { device.cmd_pipeline_barrier2(cmd, &dep2) };
        });
        let mut dummy_staging = dummy_staging;
        memory.destroy_buffer(&device, &mut dummy_staging);

        let dummy_sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT);
        let dummy_sampler = unsafe { device.create_sampler(&dummy_sampler_info, None) }.unwrap();

        // Bind dummy texture to descriptor set binding 1
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let image_info = vk::DescriptorImageInfo::default()
                .sampler(dummy_sampler)
                .image_view(dummy_image.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(desc_sets.sets[i])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&image_info));
            unsafe { device.update_descriptor_sets(&[write], &[]) };
        }

        info!("Renderer ready");
        Self {
            entry, instance, surface, surface_loader,
            physical_device, device, graphics_queue, queue_families,
            swapchain, sync, commands, memory, depth_image,
            pipeline, desc_sets, cam_ubos,
            texture_array: Some(dummy_image),
            texture_sampler: Some(dummy_sampler),
            wireframe: false,
        }
    }

    // ── Per-frame API ─────────────────────────────────────────────────────────

    /// Wait for the current frame's fence, acquire a swapchain image.
    /// Returns `None` if the swapchain needs recreation (caller should resize).
    pub fn begin_frame(&self) -> Option<(vk::CommandBuffer, usize, u32)> {
        let frame = self.sync.current_frame;
        self.sync.wait_and_reset(&self.device);

        let (img_avail, _, _) = self.sync.current();
        let (image_index, suboptimal) = match self.swapchain.acquire_next_image(img_avail) {
            Ok(r) => r,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return None,
            Err(e) => panic!("Acquire failed: {e}"),
        };
        if suboptimal { return None; }

        let cmd = self.commands.begin(&self.device, frame);
        Some((cmd, frame, image_index))
    }

    /// Record the standard render pass: clear, geometry, end.
    /// Call between begin_frame and end_frame.
    pub fn begin_rendering(&self, cmd: vk::CommandBuffer, image_index: u32) {
        let sw_image = self.swapchain.images[image_index as usize];
        let sw_view  = self.swapchain.views[image_index as usize];
        let extent   = self.swapchain.extent;

        // Transition swapchain image: UNDEFINED → COLOR_ATTACHMENT
        self.transition_image(cmd, sw_image,
            vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageAspectFlags::COLOR);

        let color_attach = vk::RenderingAttachmentInfo::default()
            .image_view(sw_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.08, 0.09, 0.12, 1.0] },
            });

        let depth_attach = vk::RenderingAttachmentInfo::default()
            .image_view(self.depth_image.view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
            });

        let render_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent,
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attach))
            .depth_attachment(&depth_attach);

        unsafe { self.device.cmd_begin_rendering(cmd, &render_info) };

        // Viewport + scissor
        let viewport = vk::Viewport {
            x: 0.0, y: 0.0,
            width: extent.width as f32, height: extent.height as f32,
            min_depth: 0.0, max_depth: 1.0,
        };
        let scissor = vk::Rect2D { offset: vk::Offset2D::default(), extent };

        unsafe {
            self.device.cmd_set_viewport(cmd, 0, &[viewport]);
            self.device.cmd_set_scissor(cmd, 0, &[scissor]);
        }
    }

    pub fn end_rendering(&self, cmd: vk::CommandBuffer, image_index: u32) {
        unsafe { self.device.cmd_end_rendering(cmd) };

        // Transition swapchain image: COLOR_ATTACHMENT → PRESENT_SRC
        self.transition_image(
            cmd,
            self.swapchain.images[image_index as usize],
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::ImageAspectFlags::COLOR,
        );
    }

    /// Bind the solid/wireframe pipeline and descriptor set for the current frame.
    pub fn bind_pipeline(&self, cmd: vk::CommandBuffer, frame: usize) {
        let pipe = if self.wireframe {
            self.pipeline.wire_handle
        } else {
            self.pipeline.fill_handle
        };
        unsafe {
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipe);
            self.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.layout, 0,
                &[self.desc_sets.sets[frame]], &[],
            );
        }
    }

    /// Bind the LINE_LIST overlay pipeline (LEQUAL depth, alpha blend).
    /// Descriptor set is already bound; call draw_mesh after this.
    pub fn bind_line_pipeline(&self, cmd: vk::CommandBuffer, frame: usize) {
        unsafe {
            self.device.cmd_bind_pipeline(
                cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline.line_handle,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd, vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.layout, 0,
                &[self.desc_sets.sets[frame]], &[],
            );
        }
    }

    /// Draw a mesh with a given model matrix (push constant).
    pub fn draw_mesh(&self, cmd: vk::CommandBuffer, mesh: &GpuMesh, model: &glam::Mat4) {
        unsafe {
            self.device.cmd_push_constants(
                cmd, self.pipeline.layout,
                vk::ShaderStageFlags::VERTEX, 0,
                bytes_of(model),
            );
            self.device.cmd_bind_vertex_buffers(cmd, 0, &[mesh.vertex_buffer.buffer], &[0]);
            self.device.cmd_bind_index_buffer(
                cmd, mesh.index_buffer.buffer, 0, vk::IndexType::UINT32,
            );
            self.device.cmd_draw_indexed(cmd, mesh.index_count, 1, 0, 0, 0);
        }
    }

    /// Upload camera uniform for the given frame.
    pub fn upload_camera(&self, frame: usize, cam: &CameraUniform) {
        let data = bytes_of(cam);
        unsafe {
            let dst = self.cam_ubos[frame].allocation.as_ref().unwrap()
                .mapped_ptr().unwrap().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
    }

    /// Record egui draw commands into the current command buffer.
    ///
    /// Call this between `begin_rendering` and `end_rendering`, AFTER all 3D
    /// drawing.  The `EguiRenderer` is owned by the caller (typically the
    /// main crate) since it also needs access during texture upload.
    pub fn render_egui(
        &self,
        cmd: vk::CommandBuffer,
        frame: usize,
        egui_renderer: &mut EguiRenderer,
        clipped_primitives: &[egui::ClippedPrimitive],
        screen_size: [f32; 2],
    ) {
        // Set the viewport to match the full render area (egui needs it after
        // the 3D pass may have changed it).
        let extent = self.swapchain.extent;
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        unsafe {
            self.device.cmd_set_viewport(cmd, 0, &[viewport]);
        }

        egui_renderer.render(
            &self.device,
            &self.memory,
            cmd,
            frame,
            clipped_primitives,
            screen_size,
        );
    }

    /// Submit and present. Returns false if swapchain is out of date.
    pub fn end_frame(&mut self, frame: usize, image_index: u32) -> bool {
        let _cmd = self.commands.end(&self.device, frame);
        let (img_avail, rend_fin, fence) = self.sync.current();
        self.commands.submit(
            &self.device, self.graphics_queue, frame, img_avail, rend_fin, fence,
        );
        let ok = match self.swapchain.present(self.graphics_queue, image_index, rend_fin) {
            Ok(suboptimal) => !suboptimal,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => false,
            Err(e) => panic!("Present failed: {e}"),
        };
        self.sync.advance();
        ok
    }

    /// Recreate swapchain + depth buffer after a resize.
    pub fn resize(&mut self, width: u32, height: u32) {
        unsafe { self.device.device_wait_idle().unwrap() };

        self.swapchain.destroy(&self.device);
        self.swapchain = Swapchain::new(
            &self.instance, &self.device, self.physical_device,
            self.surface, &self.surface_loader,
            self.queue_families.graphics, width, height,
        );

        self.memory.destroy_image(&self.device, &mut self.depth_image);
        self.depth_image = Self::create_depth_image(
            &self.device, &self.memory, &self.commands, self.graphics_queue, width, height,
        );

        info!("Swapchain resized to {}x{}", width, height);
    }

    /// Upload a mesh to GPU and return a handle.
    pub fn upload_mesh(&self, vertices: &[Vertex], indices: &[u32]) -> GpuMesh {
        GpuMesh::upload(&self.device, &self.memory, vertices, indices)
    }

    /// Upload a texture array from RGBA pixel data.
    /// `layers`: Vec of (width, height, rgba_pixels) — all will be stored at `tex_size×tex_size`.
    /// Caller should pre-resize to uniform dimensions.
    pub fn upload_texture_array(&mut self, tex_size: u32, layers: &[Vec<u8>]) {
        let num_layers = layers.len() as u32;
        if num_layers == 0 { return; }

        let format = vk::Format::R8G8B8A8_SRGB;
        let bytes_per_layer = (tex_size * tex_size * 4) as u64;

        // Create array image
        let tex_image = self.memory.create_image_array(
            &self.device, tex_size, tex_size, num_layers,
            format,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            "texture_array",
        );

        // Create staging buffer for all layers
        let total_bytes = bytes_per_layer * num_layers as u64;
        let staging = self.memory.create_buffer(
            &self.device, total_bytes,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu, "tex_staging",
        );

        // Copy all layer data into staging buffer
        unsafe {
            let dst = staging.allocation.as_ref().unwrap()
                .mapped_ptr().unwrap().as_ptr() as *mut u8;
            for (i, layer_data) in layers.iter().enumerate() {
                let offset = i as u64 * bytes_per_layer;
                let copy_len = layer_data.len().min(bytes_per_layer as usize);
                std::ptr::copy_nonoverlapping(
                    layer_data.as_ptr(),
                    dst.add(offset as usize),
                    copy_len,
                );
                // Zero-fill if layer data is shorter
                if (copy_len as u64) < bytes_per_layer {
                    std::ptr::write_bytes(
                        dst.add(offset as usize + copy_len),
                        128, // gray for missing data
                        (bytes_per_layer as usize) - copy_len,
                    );
                }
            }
        }

        // Transfer: transition → copy → transition
        one_time_submit(&self.device, self.commands.pool, self.graphics_queue, |cmd| {
            // UNDEFINED → TRANSFER_DST
            self.transition_image(cmd, tex_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageAspectFlags::COLOR);

            // Copy each layer from staging buffer
            let mut regions = Vec::with_capacity(num_layers as usize);
            for i in 0..num_layers {
                regions.push(vk::BufferImageCopy::default()
                    .buffer_offset(i as u64 * bytes_per_layer)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: i,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(vk::Extent3D { width: tex_size, height: tex_size, depth: 1 }));
            }

            unsafe {
                self.device.cmd_copy_buffer_to_image(
                    cmd, staging.buffer, tex_image.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL, &regions,
                );
            }

            // TRANSFER_DST → SHADER_READ_ONLY
            self.transition_image(cmd, tex_image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageAspectFlags::COLOR);
        });

        // Clean up staging
        let mut staging = staging;
        self.memory.destroy_buffer(&self.device, &mut staging);

        // Create sampler
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(16.0)
            .min_lod(0.0)
            .max_lod(0.0);
        let sampler = unsafe { self.device.create_sampler(&sampler_info, None) }.unwrap();

        // Update descriptor sets — binding 1 = combined image sampler
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let image_info = vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(tex_image.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

            let write = vk::WriteDescriptorSet::default()
                .dst_set(self.desc_sets.sets[i])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(std::slice::from_ref(&image_info));

            unsafe { self.device.update_descriptor_sets(&[write], &[]) };
        }

        info!("Texture array uploaded: {}x{}, {} layers", tex_size, tex_size, num_layers);
        self.texture_array = Some(tex_image);
        self.texture_sampler = Some(sampler);
    }

    /// Read back the swapchain image pixels as RGBA u8 data.
    /// Call AFTER `end_frame()`.  Blocks until copy completes.
    /// Returns (width, height, rgba_pixels).
    pub fn capture_screenshot(&self, image_index: u32) -> (u32, u32, Vec<u8>) {
        let width  = self.swapchain.extent.width;
        let height = self.swapchain.extent.height;
        let pixel_bytes = 4u64; // BGRA8 → we'll swizzle to RGBA
        let buf_size = width as u64 * height as u64 * pixel_bytes;

        unsafe { self.device.device_wait_idle().unwrap() };

        // Create a host-visible staging buffer
        let staging = self.memory.create_buffer(
            &self.device, buf_size,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
            "screenshot-staging",
        );

        let sw_image = self.swapchain.images[image_index as usize];

        // One-shot: transition, copy, transition back
        one_time_submit(&self.device, self.commands.pool, self.graphics_queue, |cmd| {
            // PRESENT_SRC → TRANSFER_SRC
            self.transition_image(cmd, sw_image,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageAspectFlags::COLOR);

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
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D { width, height, depth: 1 });

            unsafe {
                self.device.cmd_copy_image_to_buffer(
                    cmd, sw_image, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    staging.buffer, &[region],
                );
            }

            // TRANSFER_SRC → PRESENT_SRC (restore)
            self.transition_image(cmd, sw_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::ImageAspectFlags::COLOR);
        });

        // Read pixels from staging buffer
        let ptr = staging.allocation.as_ref().unwrap()
            .mapped_ptr().unwrap().as_ptr() as *const u8;
        let raw = unsafe { std::slice::from_raw_parts(ptr, buf_size as usize) };

        // Convert BGRA → RGBA
        let mut rgba = vec![0u8; buf_size as usize];
        for i in 0..(width * height) as usize {
            let off = i * 4;
            rgba[off]     = raw[off + 2]; // R ← B
            rgba[off + 1] = raw[off + 1]; // G
            rgba[off + 2] = raw[off];     // B ← R
            rgba[off + 3] = 255;          // A
        }

        // Cleanup staging buffer
        let mut staging = staging;
        self.memory.destroy_buffer(&self.device, &mut staging);

        (width, height, rgba)
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn create_depth_image(
        device:   &ash::Device,
        memory:   &GpuMemory,
        commands: &CommandManager,
        queue:    vk::Queue,
        width:    u32,
        height:   u32,
    ) -> AllocatedImage {
        let image = memory.create_image(
            device, width, height,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH,
            "depth",
        );

        // Transition depth image from UNDEFINED → DEPTH_ATTACHMENT_OPTIMAL once.
        one_time_submit(device, commands.pool, queue, |cmd| {
            let barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS)
                .dst_access_mask(
                    vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                )
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0, level_count: 1,
                    base_array_layer: 0, layer_count: 1,
                });

            let dep = vk::DependencyInfo::default()
                .image_memory_barriers(std::slice::from_ref(&barrier));
            unsafe { device.cmd_pipeline_barrier2(cmd, &dep) };
        });

        image
    }

    /// Synchronization2 image layout transition (generic helper).
    fn transition_image(
        &self,
        cmd:        vk::CommandBuffer,
        image:      vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        aspect:     vk::ImageAspectFlags,
    ) {
        let (src_stage, src_access, dst_stage, dst_access) = match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            ),
            (vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR) => (
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                vk::AccessFlags2::NONE,
            ),
            _ => (
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            ),
        };

        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0, level_count: 1,
                base_array_layer: 0, layer_count: 1,
            });

        let dep = vk::DependencyInfo::default()
            .image_memory_barriers(std::slice::from_ref(&barrier));
        unsafe { self.device.cmd_pipeline_barrier2(cmd, &dep) };
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe { self.device.device_wait_idle().unwrap() };

        if let Some(sampler) = self.texture_sampler.take() {
            unsafe { self.device.destroy_sampler(sampler, None) };
        }
        if let Some(mut tex) = self.texture_array.take() {
            self.memory.destroy_image(&self.device, &mut tex);
        }
        self.desc_sets.destroy(&self.device);
        self.pipeline.destroy(&self.device);
        for ubo in &mut self.cam_ubos {
            self.memory.destroy_buffer(&self.device, ubo);
        }
        self.memory.destroy_image(&self.device, &mut self.depth_image);
        self.commands.destroy(&self.device);
        self.sync.destroy(&self.device);
        self.swapchain.destroy(&self.device);
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

// ── One-shot command execution ────────────────────────────────────────────────

pub fn one_time_submit<F>(
    device: &ash::Device,
    pool:   vk::CommandPool,
    queue:  vk::Queue,
    f:      F,
) where F: FnOnce(vk::CommandBuffer) {
    let alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let cmd = unsafe { device.allocate_command_buffers(&alloc) }.unwrap()[0];
    unsafe {
        device.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        ).unwrap();
    }
    f(cmd);
    unsafe {
        device.end_command_buffer(cmd).unwrap();
        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
        device.queue_submit(queue, &[submit], vk::Fence::null()).unwrap();
        device.queue_wait_idle(queue).unwrap();
        device.free_command_buffers(pool, &[cmd]);
    }
}
