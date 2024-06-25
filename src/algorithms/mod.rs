use std::{marker::PhantomData, sync::{Arc}};

use ash::vk;
use nightfall_core::{barriers::Barriers, buffers::{BufferCreateFlags, BufferCreateInfo, BufferOffset, BufferUsageFlags, MemoryPropertyFlags}, commands::CommandPoolAllocation, descriptors::{DescriptorLayout, DescriptorPool, DescriptorSetAllocation}, device::LogicalDevice, memory::DevicePointer, queue::Queue};
use crate::{alloc::{FreeListAllocator, GeneralAllocator, StackAllocator}, error::StarlitError};

/// standard morton code calculator module 
pub mod morton;
/// standard prefix sum module 
pub mod prefix_sum;
/// onesweep module that must support thread forwarding 
pub mod onesweep;
/// standard device sort module that doesn't need thread forwarding
pub mod device_sort;
/// standard device sort module that doesn't need thread forwarding for sorting in pairs
pub mod device_sort_pairs;
/// simple module for filling in iterator with a range.
pub mod range;
/// TODO simple module for doing a Chained Scan Decoupled Lookback Fallback Exclusive **NOT IMPLEMENTED**
mod csdlfe;


#[derive(Clone)]
pub enum Strategy {
    UsesDeviceAddress,
    UsesDescriptorSets {
        pool: Arc<DescriptorPool>
    },
    HybridDeviceAddressDescriptorSets {
        pool: Arc<DescriptorPool>
    },
}
#[derive(Clone)]
pub struct StarlitStrategy {
    pub strategy: Strategy,
    pub use_subgroups: bool,
}
impl StarlitStrategy {
    pub fn new(strategy: Strategy, use_subgroups: bool) -> Self {
        Self { strategy, use_subgroups }
    }
}
/// Holds the internal strategy of all [`StarlitShaderKernel`]'s.
pub enum StarlitStrategyInternal {
    UsesDeviceAddress,
    UsesDescriptorSets {
        sets: Vec<DescriptorSetAllocation>
    },
    HybridDeviceAddressDescriptorSets {
        pool: Arc<DescriptorPool>
    },
}
/// Tracks the internal state of all [`StarlitShaderKernel`]'s.
pub struct StarlitStrategyState<T> {
    pub strategy: StarlitStrategyInternal,
    pub input: Option<T>,
}

/// To simplify operations done on Pipelines and generalize them to not have to worry about the extensions we use
/// [`StarlitShaderKernel`]. You start by requesting a strategy for the task that depends on the features the device
/// might support using [`StarlitStrategy`]. After creating the structure using that strategy, to use it you must register
/// your input using ```register```. The data has not yet been completely registered however and you should call
/// ```input``` to actually input the registered data. After this is done you can ```dispatch```, don't forget to ```bind``` 
/// beforehand.
pub trait StarlitShaderKernel: Sized {
    type Input;
    fn strategy(device: Arc<LogicalDevice>, strategy: StarlitStrategy) -> Result<Self, StarlitError>;
    fn bind(&self, command_buffer: vk::CommandBuffer);
    fn register(&mut self, input: &Self::Input) -> Result<(), StarlitError>;
    fn input(&self, command_buffer: vk::CommandBuffer) -> Result<(), StarlitError>;
    fn dispatch(&self, command_buffer: vk::CommandBuffer, dispatch_x: u32, dispatch_y: u32, dispatch_z: u32);
    fn dispatch_indirect(&self, command_buffer: vk::CommandBuffer, indirect: BufferOffset);
    fn dispatch_with_barrier(&self, command_buffer: vk::CommandBuffer, dispatch_x: u32, dispatch_y: u32, dispatch_z: u32) -> Option<Barriers> {
        self.dispatch(command_buffer, dispatch_x, dispatch_y, dispatch_z);
        None
    }
    fn dispatch_indirect_with_barrier(&self, command_buffer: vk::CommandBuffer, indirect: BufferOffset) -> Option<Barriers> {
        self.dispatch_indirect(command_buffer, indirect);
        None
    }
    fn get_state(&self) -> &StarlitStrategyState<Self::Input>;
}
/// Extension to [`StarlitShaderKernel`] so that it can support creation using previously allocated descriptor sets from another descriptor layout.
pub trait StarlitShaderKernelFromDescriptorSets: StarlitShaderKernel {
    fn from_descriptor_sets(device: Arc<LogicalDevice>, sets: &[DescriptorSetAllocation]) -> Result<Self, StarlitError>;
}
/// Extension to [`StarlitShaderKernel`] so that it can support registering only the constants.
pub trait StarlitShaderKernelConstants: StarlitShaderKernel {
    type Constant;
    fn register_constants(&mut self, constant: &Self::Constant);
}
/// Functioning as a collection of Kernels working together. Usually a single kernel is not supposed to work alone, and instead you
/// should use [`StarlitShaderAlgorithm`] which organizes the kernels to do what the algorithm requires.
pub trait StarlitShaderAlgorithm<A: GeneralAllocator + ?Sized>: Sized {
    type Input;
    fn strategy(device: Arc<LogicalDevice>, freelist: Arc<A>, strategy: StarlitStrategy) -> Result<Self, StarlitError>;
    fn register(&mut self, input: &Self::Input) -> Result<(), StarlitError>;
    fn fill_command_listing(&mut self, command_buffer: &CommandPoolAllocation) -> Result<(), StarlitError>;
}
/// used to quickly register, input and dispatch shaders when your too lazy to type all that out at the time.
pub trait StarlitShaderExecute {
    type Input;
    fn execute(&mut self, command_buffer: &CommandPoolAllocation, input: &Self::Input) -> Result<(), StarlitError>;
    fn execute_with_barrier(&mut self, command_buffer: &nightfall_core::commands::CommandPoolAllocation, input: &Self::Input) -> Result<Option<Barriers>, StarlitError> {
        self.execute(command_buffer, input)?;
        Ok(None)
    }
}