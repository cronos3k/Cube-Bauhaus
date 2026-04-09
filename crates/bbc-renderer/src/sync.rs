//! Frame synchronization — semaphores, fences, frames-in-flight.
//! Copied verbatim from woi2/vk-renderer.

use ash::vk;
use tracing::info;

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct FrameSync {
    pub image_available: [vk::Semaphore; MAX_FRAMES_IN_FLIGHT],
    pub render_finished: [vk::Semaphore; MAX_FRAMES_IN_FLIGHT],
    pub in_flight: [vk::Fence; MAX_FRAMES_IN_FLIGHT],
    pub current_frame: usize,
}

impl FrameSync {
    pub fn new(device: &ash::Device) -> Self {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let mut image_available = [vk::Semaphore::null(); MAX_FRAMES_IN_FLIGHT];
        let mut render_finished = [vk::Semaphore::null(); MAX_FRAMES_IN_FLIGHT];
        let mut in_flight = [vk::Fence::null(); MAX_FRAMES_IN_FLIGHT];

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            image_available[i] =
                unsafe { device.create_semaphore(&semaphore_info, None) }.unwrap();
            render_finished[i] =
                unsafe { device.create_semaphore(&semaphore_info, None) }.unwrap();
            in_flight[i] = unsafe { device.create_fence(&fence_info, None) }.unwrap();
        }

        info!("Frame sync: {} frames in flight", MAX_FRAMES_IN_FLIGHT);
        Self { image_available, render_finished, in_flight, current_frame: 0 }
    }

    pub fn wait_and_reset(&self, device: &ash::Device) {
        let fence = self.in_flight[self.current_frame];
        unsafe {
            device.wait_for_fences(&[fence], true, u64::MAX).unwrap();
            device.reset_fences(&[fence]).unwrap();
        }
    }

    pub fn current(&self) -> (vk::Semaphore, vk::Semaphore, vk::Fence) {
        (
            self.image_available[self.current_frame],
            self.render_finished[self.current_frame],
            self.in_flight[self.current_frame],
        )
    }

    pub fn advance(&mut self) {
        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    pub fn destroy(&self, device: &ash::Device) {
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe {
                device.destroy_semaphore(self.image_available[i], None);
                device.destroy_semaphore(self.render_finished[i], None);
                device.destroy_fence(self.in_flight[i], None);
            }
        }
    }
}
