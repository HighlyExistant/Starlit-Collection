#![allow(unused)]
pub use starlit_alloc as alloc;
pub mod algorithms;
pub mod error;
mod collection;
mod object;
pub use collection::*;
pub use object::*;
#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};
    use alloc::{FreeListAllocator, HostDeviceConversions};
    use ash::vk;
    use nightfall_core::{buffers::{BufferUsageFlags, MemoryPropertyFlags}, commands::{CommandBufferBeginInfo, CommandBufferLevel, CommandPool, CommandPoolCreateFlags}, descriptors::{DescriptorPoolBuilder, DescriptorType}, device::{LogicalDevice, LogicalDeviceBuilder}, instance::{Instance, InstanceBuilder}, queue::{DeviceQueueCreateFlags, Queue, QueueFlags, Submission}, Version};
    use starlit_alloc::{GeneralAllocator, GpuAllocators, StandardAllocator};

    use super::*;
    fn barebones() -> (Arc<LogicalDevice>, impl ExactSizeIterator<Item = Arc<Queue>>) {
        let instance = InstanceBuilder::new()
        .set_version(Version::new(1, 0, 0))
        .build().unwrap();
        let physical_device = instance.enumerate_physical_devices().unwrap().next().unwrap();
        
        let queue_family_index = physical_device.enumerate_queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_queue_family_index, queue_family_properties)|{
            queue_family_properties.queue_flags.contains(QueueFlags::COMPUTE | QueueFlags::GRAPHICS)
        }).unwrap();
        
        LogicalDeviceBuilder::new()
            .add_queue(DeviceQueueCreateFlags::empty(), queue_family_index as u32, 1, 0, &1.0)
            .build(physical_device.clone()).unwrap()
    }
    #[test]
    fn barebones_vec_test() {
        let (device, mut queues) = barebones();
        let queue = queues.next().unwrap();
        let gpu_state = StandardAllocator::new(device.clone()).unwrap();
        let host_freelist = gpu_state.freelist(BufferUsageFlags::STORAGE_BUFFER, MemoryPropertyFlags::HOST_VISIBLE|MemoryPropertyFlags::HOST_COHERENT).unwrap();
        let device_freelist = gpu_state.freelist(BufferUsageFlags::STORAGE_BUFFER, MemoryPropertyFlags::DEVICE_LOCAL).unwrap();
        let mut hostvec = SlVec::<i32, dyn GeneralAllocator>::new(host_freelist.clone());
        let mut host2vec = SlVec::<i32, dyn GeneralAllocator>::new(host_freelist.clone());
        let devvec = SlVec::<i32, dyn GeneralAllocator>::new_zeroed(500, device_freelist.clone()).unwrap();
        for i in 0..500 {
            hostvec.push(i);
            host2vec.push(i);
        }
    }
    // #[test]
    fn buffer_addressing_vec_test() {
        let instance = InstanceBuilder::new()
        .set_version(Version::new(1, 2, 0))
        .validation_layers()
        .get_physical_device_properties2()
        .device_group_creation_extension()
        .build().unwrap();
        let physical_device = instance.enumerate_physical_devices().unwrap().next().unwrap();
        let queue_family_index = physical_device.enumerate_queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_queue_family_index, queue_family_properties)|{
            queue_family_properties.queue_flags.contains(QueueFlags::COMPUTE | QueueFlags::GRAPHICS)
        }).unwrap();
        let (device, mut queues) = LogicalDeviceBuilder::new()
            .add_queue(DeviceQueueCreateFlags::empty(), queue_family_index as u32, 1, 0, &1.0)
            .enable_buffer_addressing()
            .device_group()
            .build(physical_device.clone()).unwrap();
        let queue = queues.next().unwrap();
        let gpu_state = StandardAllocator::new(device.clone()).unwrap();
        let host_freelist = gpu_state.freelist(BufferUsageFlags::STORAGE_BUFFER, MemoryPropertyFlags::HOST_VISIBLE|MemoryPropertyFlags::HOST_COHERENT).unwrap();
        let device_freelist = gpu_state.freelist(BufferUsageFlags::STORAGE_BUFFER|BufferUsageFlags::SHADER_DEVICE_ADDRESS, MemoryPropertyFlags::DEVICE_LOCAL).unwrap();
        let devvec = SlVec::<i32, dyn GeneralAllocator>::with_capacity(50, device_freelist.clone()).unwrap();
        let addr = device_freelist.as_device_ptr(devvec.get_allocation()).unwrap();
    }
}
