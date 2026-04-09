//! Command pool and per-frame command buffer management.
//! Copied verbatim from woi2/vk-renderer.

use ash::vk;
use crate::sync::MAX_FRAMES_IN_FLIGHT;
use tracing::info;

pub struct CommandManager {
    pub pool: vk::CommandPool,
    pub buffers: [vk::CommandBuffer; MAX_FRAMES_IN_FLIGHT],
}

impl CommandManager {
    pub fn new(device: &ash::Device, graphics_family: u32) -> Self {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(graphics_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let pool = unsafe { device.create_command_pool(&pool_info, None) }.unwrap();

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32);

        let buffers_vec = unsafe { device.allocate_command_buffers(&alloc_info) }.unwrap();
        let mut buffers = [vk::CommandBuffer::null(); MAX_FRAMES_IN_FLIGHT];
        for (i, &buf) in buffers_vec.iter().enumerate() {
            buffers[i] = buf;
        }

        info!("Command pool: {} buffers (family {})", MAX_FRAMES_IN_FLIGHT, graphics_family);
        Self { pool, buffers }
    }

    pub fn begin(&self, device: &ash::Device, frame: usize) -> vk::CommandBuffer {
        let cmd = self.buffers[frame];
        unsafe {
            device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
        }
        cmd
    }

    pub fn end(&self, device: &ash::Device, frame: usize) -> vk::CommandBuffer {
        let cmd = self.buffers[frame];
        unsafe { device.end_command_buffer(cmd).unwrap() };
        cmd
    }

    pub fn submit(
        &self,
        device: &ash::Device,
        queue: vk::Queue,
        frame: usize,
        wait_semaphore: vk::Semaphore,
        signal_semaphore: vk::Semaphore,
        fence: vk::Fence,
    ) {
        let cmd = self.buffers[frame];
        let wait_semaphores = [wait_semaphore];
        let signal_semaphores = [signal_semaphore];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [cmd];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        unsafe { device.queue_submit(queue, &[submit_info], fence).unwrap() };
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe { device.destroy_command_pool(self.pool, None) };
    }
}
