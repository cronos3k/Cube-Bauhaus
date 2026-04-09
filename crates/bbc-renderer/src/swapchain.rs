//! Swapchain creation and management. Copied verbatim from woi2/vk-renderer.

use ash::{khr, vk};
use tracing::info;

pub struct Swapchain {
    pub handle: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    pub format: vk::SurfaceFormatKHR,
    pub extent: vk::Extent2D,
    pub loader: khr::swapchain::Device,
}

impl Swapchain {
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        surface_loader: &khr::surface::Instance,
        graphics_family: u32,
        width: u32,
        height: u32,
    ) -> Self {
        let capabilities = unsafe {
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
        }
        .expect("Failed to get surface capabilities");

        let formats = unsafe {
            surface_loader.get_physical_device_surface_formats(physical_device, surface)
        }
        .expect("Failed to get surface formats");

        let present_modes = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(physical_device, surface)
        }
        .expect("Failed to get present modes");

        let format = formats
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&formats[0])
            .clone();

        let present_mode = if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        };

        let extent = vk::Extent2D {
            width: width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        };

        let image_count = (capabilities.min_image_count + 1).min(if capabilities.max_image_count > 0 {
            capabilities.max_image_count
        } else {
            u32::MAX
        });

        let loader = khr::swapchain::Device::new(instance, device);
        let queue_families = [graphics_family];

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_families)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let handle = unsafe { loader.create_swapchain(&create_info, None) }
            .expect("Failed to create swapchain");

        let images =
            unsafe { loader.get_swapchain_images(handle) }.expect("Failed to get swapchain images");

        let views: Vec<vk::ImageView> = images
            .iter()
            .map(|&image| {
                let view_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format.format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                unsafe { device.create_image_view(&view_info, None) }
                    .expect("Failed to create swapchain image view")
            })
            .collect();

        info!(
            "Swapchain: {}x{} {:?} {:?} ({} images)",
            extent.width,
            extent.height,
            format.format,
            present_mode,
            images.len()
        );

        Self { handle, images, views, format, extent, loader }
    }

    pub fn acquire_next_image(
        &self,
        signal_semaphore: vk::Semaphore,
    ) -> Result<(u32, bool), vk::Result> {
        unsafe {
            self.loader
                .acquire_next_image(self.handle, u64::MAX, signal_semaphore, vk::Fence::null())
        }
    }

    pub fn present(
        &self,
        queue: vk::Queue,
        image_index: u32,
        wait_semaphore: vk::Semaphore,
    ) -> Result<bool, vk::Result> {
        let swapchains = [self.handle];
        let image_indices = [image_index];
        let wait_semaphores = [wait_semaphore];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        unsafe { self.loader.queue_present(queue, &present_info) }
    }

    pub fn destroy(&self, device: &ash::Device) {
        for &view in &self.views {
            unsafe { device.destroy_image_view(view, None) };
        }
        unsafe { self.loader.destroy_swapchain(self.handle, None) };
    }
}
