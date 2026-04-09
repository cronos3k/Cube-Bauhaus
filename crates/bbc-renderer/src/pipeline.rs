//! Forward rendering pipeline — simple Lambert + directional light.
//! No MRT, no GBuffer, no compose pass.
//! Uses Vulkan 1.3 dynamic rendering (no render pass object).
//!
//! Three pipelines share the same layout:
//!   fill_handle  — TRIANGLE_LIST + FILL (solid geometry)
//!   wire_handle  — TRIANGLE_LIST + LINE (triangle-edge wireframe view)
//!   line_handle  — LINE_LIST + FILL     (explicit edge wireframe overlay,
//!                                        depth LEQUAL so lines sit on top)

use ash::vk;
use crate::{camera::CameraUniform, mesh::{vertex_attributes, vertex_binding}, sync::MAX_FRAMES_IN_FLIGHT};
use tracing::info;

pub struct ForwardPipeline {
    pub fill_handle:  vk::Pipeline,
    pub wire_handle:  vk::Pipeline,
    pub line_handle:  vk::Pipeline,   // LINE_LIST, depth LEQUAL, no depth write
    pub layout:          vk::PipelineLayout,
    pub desc_set_layout: vk::DescriptorSetLayout,
}

pub fn create_shader_module(device: &ash::Device, spirv: &[u8]) -> vk::ShaderModule {
    assert!(spirv.len() % 4 == 0, "SPIR-V not 4-byte aligned");
    let code: Vec<u32> = spirv
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let info = vk::ShaderModuleCreateInfo::default().code(&code);
    unsafe { device.create_shader_module(&info, None) }.expect("Failed to create shader module")
}

impl ForwardPipeline {
    pub fn new(
        device:       &ash::Device,
        color_format: vk::Format,
        depth_format: vk::Format,
    ) -> Self {
        // ── Descriptor set layout: binding 0 = camera UBO, binding 1 = texture array ──
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];

        let desc_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&bindings),
                None,
            )
        }
        .unwrap();

        // ── Pipeline layout: set 0 + push constant = mat4 (64 bytes) ──────
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(64);

        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(std::slice::from_ref(&desc_set_layout))
                    .push_constant_ranges(std::slice::from_ref(&push_range)),
                None,
            )
        }
        .unwrap();

        // ── Shaders ────────────────────────────────────────────────────────
        let vert_spirv  = include_bytes!(concat!(env!("OUT_DIR"), "/cube.vert.spv"));
        let frag_spirv  = include_bytes!(concat!(env!("OUT_DIR"), "/cube.frag.spv"));
        let wire_vert   = include_bytes!(concat!(env!("OUT_DIR"), "/wire.vert.spv"));
        let wire_frag   = include_bytes!(concat!(env!("OUT_DIR"), "/wire.frag.spv"));

        let vert_mod  = create_shader_module(device, vert_spirv);
        let frag_mod  = create_shader_module(device, frag_spirv);
        let wvert_mod = create_shader_module(device, wire_vert);
        let wfrag_mod = create_shader_module(device, wire_frag);

        let fill_handle = build_pipeline(device, layout, color_format, depth_format,
            vert_mod, frag_mod,
            vk::PrimitiveTopology::TRIANGLE_LIST, vk::PolygonMode::FILL,
            vk::CullModeFlags::BACK, DepthMode::WriteAndTest);

        let wire_handle = build_pipeline(device, layout, color_format, depth_format,
            vert_mod, frag_mod,
            vk::PrimitiveTopology::TRIANGLE_LIST, vk::PolygonMode::LINE,
            vk::CullModeFlags::NONE, DepthMode::WriteAndTest);

        // Line overlay: LINE_LIST, LEQUAL depth (no write) so lines sit on surface
        let line_handle = build_pipeline(device, layout, color_format, depth_format,
            wvert_mod, wfrag_mod,
            vk::PrimitiveTopology::LINE_LIST, vk::PolygonMode::FILL,
            vk::CullModeFlags::NONE, DepthMode::TestNoWrite);

        unsafe {
            device.destroy_shader_module(vert_mod, None);
            device.destroy_shader_module(frag_mod, None);
            device.destroy_shader_module(wvert_mod, None);
            device.destroy_shader_module(wfrag_mod, None);
        }

        info!("Forward pipeline created (fill + wire + line)");
        Self { fill_handle, wire_handle, line_handle, layout, desc_set_layout }
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.fill_handle, None);
            device.destroy_pipeline(self.wire_handle, None);
            device.destroy_pipeline(self.line_handle, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.desc_set_layout, None);
        }
    }
}

// ── Pipeline builder ──────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum DepthMode {
    WriteAndTest,  // normal depth test + write (solid geometry)
    TestNoWrite,   // depth test but no write (overlay lines, LEQUAL)
}

fn build_pipeline(
    device:       &ash::Device,
    layout:       vk::PipelineLayout,
    color_format: vk::Format,
    depth_format: vk::Format,
    vert_mod:     vk::ShaderModule,
    frag_mod:     vk::ShaderModule,
    topology:     vk::PrimitiveTopology,
    poly_mode:    vk::PolygonMode,
    cull_mode:    vk::CullModeFlags,
    depth_mode:   DepthMode,
) -> vk::Pipeline {
    let entry = c"main";
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX).module(vert_mod).name(entry),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT).module(frag_mod).name(entry),
    ];

    let attrs    = vertex_attributes();
    let bindings = [vertex_binding()];
    let vert_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_attribute_descriptions(&attrs)
        .vertex_binding_descriptions(&bindings);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(topology);

    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(poly_mode)
        .cull_mode(cull_mode)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let msaa = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let depth_stencil = match depth_mode {
        DepthMode::WriteAndTest => vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS),
        DepthMode::TestNoWrite => vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL),
    };

    // Alpha blending for the line overlay (wire_frag outputs translucent lines)
    let blend_attach = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(matches!(depth_mode, DepthMode::TestNoWrite))
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD);

    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(std::slice::from_ref(&blend_attach));

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default()
        .dynamic_states(&dynamic_states);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let color_formats = [color_format];
    let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&color_formats)
        .depth_attachment_format(depth_format);

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
            .expect("Pipeline creation failed")[0]
    }
}

// ── Descriptor pool + sets ────────────────────────────────────────────────────

pub struct DescriptorSets {
    pub pool: vk::DescriptorPool,
    pub sets: [vk::DescriptorSet; MAX_FRAMES_IN_FLIGHT],
}

impl DescriptorSets {
    pub fn new(
        device:      &ash::Device,
        layout:      vk::DescriptorSetLayout,
        ubo_buffers: &[crate::memory::AllocatedBuffer; MAX_FRAMES_IN_FLIGHT],
    ) -> Self {
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32),
        ];

        let pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(MAX_FRAMES_IN_FLIGHT as u32),
                None,
            )
        }
        .unwrap();

        let layouts = [layout; MAX_FRAMES_IN_FLIGHT];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);

        let allocated = unsafe { device.allocate_descriptor_sets(&alloc_info) }.unwrap();
        let sets = [allocated[0], allocated[1]];

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let buf_info = vk::DescriptorBufferInfo::default()
                .buffer(ubo_buffers[i].buffer)
                .offset(0)
                .range(std::mem::size_of::<CameraUniform>() as u64);

            let write = vk::WriteDescriptorSet::default()
                .dst_set(sets[i])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_info));

            unsafe { device.update_descriptor_sets(&[write], &[]) };
        }

        Self { pool, sets }
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe { device.destroy_descriptor_pool(self.pool, None) };
    }
}
