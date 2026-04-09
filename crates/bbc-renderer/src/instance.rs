//! Vulkan instance creation and physical device selection.
//! Copied from woi2/vk-renderer, app name changed.

use ash::vk;
use std::ffi::CStr;
use tracing::info;

pub fn create_instance(
    entry: &ash::Entry,
    window: &dyn raw_window_handle::HasDisplayHandle,
) -> ash::Instance {
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"BBC — Cube2 Geometry Editor")
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(c"bbc-renderer")
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_3);

    let mut extensions = ash_window::enumerate_required_extensions(
        window.display_handle().unwrap().as_raw(),
    )
    .expect("Failed to get required surface extensions")
    .to_vec();

    let has_debug = unsafe { entry.enumerate_instance_extension_properties(None) }
        .unwrap_or_default()
        .iter()
        .any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == c"VK_EXT_debug_utils");

    if has_debug {
        extensions.push(c"VK_EXT_debug_utils".as_ptr());
    }

    let available_layers = unsafe { entry.enumerate_instance_layer_properties() }
        .unwrap_or_default();
    let validation_name = c"VK_LAYER_KHRONOS_validation";
    let has_validation = available_layers
        .iter()
        .any(|l| unsafe { CStr::from_ptr(l.layer_name.as_ptr()) } == validation_name);

    let mut layer_names: Vec<*const i8> = Vec::new();
    let want_validation = std::env::var("VK_VALIDATE").map(|v| v == "1").unwrap_or(false);
    if has_validation && want_validation {
        layer_names.push(c"VK_LAYER_KHRONOS_validation".as_ptr());
        info!("Vulkan validation layer ENABLED (VK_VALIDATE=1)");
    } else {
        info!("Vulkan validation layer DISABLED (set VK_VALIDATE=1 to enable)");
    }

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&extensions)
        .enabled_layer_names(&layer_names);

    let instance = unsafe { entry.create_instance(&create_info, None) }
        .expect("Failed to create Vulkan instance");

    info!("Vulkan instance created (API 1.3)");
    instance
}

pub fn select_physical_device(instance: &ash::Instance) -> vk::PhysicalDevice {
    let devices = unsafe { instance.enumerate_physical_devices() }
        .expect("Failed to enumerate physical devices");

    if devices.is_empty() {
        panic!("No Vulkan-capable GPU found");
    }

    let mut best = devices[0];
    let mut best_score = 0u32;

    for &device in &devices {
        let props = unsafe { instance.get_physical_device_properties(device) };
        let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
        let score = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU   => 1000,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 100,
            vk::PhysicalDeviceType::VIRTUAL_GPU    => 10,
            _                                      => 1,
        };
        info!("  GPU: {} (type={:?}, score={})", name.to_string_lossy(), props.device_type, score);
        if score > best_score {
            best = device;
            best_score = score;
        }
    }

    let props = unsafe { instance.get_physical_device_properties(best) };
    let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
    info!("Selected GPU: {}", name.to_string_lossy());
    best
}
