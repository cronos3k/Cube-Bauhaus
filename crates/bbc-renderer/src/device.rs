//! Logical device creation and queue family discovery.
//! Copied from woi2/vk-renderer — RT / barycentric extensions removed.
//! Only VK_KHR_swapchain required; dynamic rendering via Vulkan 1.3 core.

use ash::vk;
use tracing::info;

#[derive(Debug, Clone, Copy)]
pub struct QueueFamilies {
    pub graphics: u32,
}

pub fn find_queue_families(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> QueueFamilies {
    let families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

    let mut graphics = None;

    for (i, family) in families.iter().enumerate() {
        let i = i as u32;
        let supports_graphics = family.queue_flags.contains(vk::QueueFlags::GRAPHICS);
        let supports_present = unsafe {
            surface_loader
                .get_physical_device_surface_support(physical_device, i, surface)
                .unwrap_or(false)
        };
        if supports_graphics && supports_present && graphics.is_none() {
            graphics = Some(i);
        }
    }

    let graphics = graphics.expect("No graphics+present queue family found");
    info!("Queue family: graphics+present={}", graphics);
    QueueFamilies { graphics }
}

pub fn create_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_families: &QueueFamilies,
) -> (ash::Device, vk::Queue) {
    let queue_priorities = [1.0f32];
    let queue_create_infos = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_families.graphics)
        .queue_priorities(&queue_priorities)];

    // Only what we need: swapchain.  No RT, no barycentrics.
    let device_extensions = [c"VK_KHR_swapchain".as_ptr()];

    // Vulkan 1.3 core: dynamic rendering + synchronization2
    let mut features_13 = vk::PhysicalDeviceVulkan13Features::default()
        .dynamic_rendering(true)
        .synchronization2(true);

    // Vulkan 1.2: none required for simple forward rendering.
    // Keep struct in chain so driver sees it (all false = no extra requirements).
    let mut features_12 = vk::PhysicalDeviceVulkan12Features::default();

    let features = vk::PhysicalDeviceFeatures::default()
        .sampler_anisotropy(true)
        .fill_mode_non_solid(true); // wireframe toggle

    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&device_extensions)
        .enabled_features(&features)
        .push_next(&mut features_13)
        .push_next(&mut features_12);

    let device = unsafe { instance.create_device(physical_device, &device_create_info, None) }
        .expect("Failed to create logical device");

    let graphics_queue = unsafe { device.get_device_queue(queue_families.graphics, 0) };

    info!("Logical device created (graphics queue={})", queue_families.graphics);
    (device, graphics_queue)
}
