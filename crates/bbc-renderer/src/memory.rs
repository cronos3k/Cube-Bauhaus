//! GPU memory allocation via gpu-allocator.
//! Copied from woi2/vk-renderer — 3D image, image arrays, and buffer device
//! address helpers removed (not needed for simple forward rendering).

use ash::vk;
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc},
    MemoryLocation,
};
use std::sync::Mutex;
use tracing::info;

pub struct GpuMemory {
    allocator: Mutex<Allocator>,
}

pub struct AllocatedBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub size: u64,
}

pub struct AllocatedImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub allocation: Option<Allocation>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

impl GpuMemory {
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
    ) -> Self {
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: false, // not needed without RT
            allocation_sizes: Default::default(),
        })
        .expect("Failed to create GPU allocator");

        info!("GPU memory allocator created");
        Self { allocator: Mutex::new(allocator) }
    }

    pub fn create_buffer(
        &self,
        device: &ash::Device,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        label: &str,
    ) -> AllocatedBuffer {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&buffer_info, None) }.unwrap();
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation = self
            .allocator
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name: label,
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .expect("Failed to allocate buffer memory");

        unsafe {
            device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .unwrap();
        }

        AllocatedBuffer { buffer, allocation: Some(allocation), size }
    }

    pub fn create_image(
        &self,
        device: &ash::Device,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspect: vk::ImageAspectFlags,
        label: &str,
    ) -> AllocatedImage {
        let extent = vk::Extent2D { width, height };
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { device.create_image(&image_info, None) }.unwrap();
        let requirements = unsafe { device.get_image_memory_requirements(image) };

        let allocation = self
            .allocator
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name: label,
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .expect("Failed to allocate image memory");

        unsafe {
            device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .unwrap();
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        let view = unsafe { device.create_image_view(&view_info, None) }.unwrap();

        AllocatedImage { image, view, allocation: Some(allocation), format, extent }
    }

    /// Create a 2D array image (for texture arrays).
    pub fn create_image_array(
        &self,
        device: &ash::Device,
        width: u32,
        height: u32,
        layers: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        label: &str,
    ) -> AllocatedImage {
        let extent = vk::Extent2D { width, height };
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(1)
            .array_layers(layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image = unsafe { device.create_image(&image_info, None) }.unwrap();
        let requirements = unsafe { device.get_image_memory_requirements(image) };

        let allocation = self
            .allocator
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name: label,
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .expect("Failed to allocate image array memory");

        unsafe {
            device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .unwrap();
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: layers,
            });
        let view = unsafe { device.create_image_view(&view_info, None) }.unwrap();

        AllocatedImage { image, view, allocation: Some(allocation), format, extent }
    }

    pub fn destroy_buffer(&self, device: &ash::Device, buffer: &mut AllocatedBuffer) {
        unsafe { device.destroy_buffer(buffer.buffer, None) };
        if let Some(alloc) = buffer.allocation.take() {
            self.allocator.lock().unwrap().free(alloc).ok();
        }
    }

    pub fn destroy_image(&self, device: &ash::Device, image: &mut AllocatedImage) {
        unsafe {
            device.destroy_image_view(image.view, None);
            device.destroy_image(image.image, None);
        }
        if let Some(alloc) = image.allocation.take() {
            self.allocator.lock().unwrap().free(alloc).ok();
        }
    }
}
