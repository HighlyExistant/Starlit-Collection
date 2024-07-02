use std::sync::Arc;

use ash::vk;
use nightfall_core::{barriers::Barriers, descriptors::{DescriptorLayoutBuilder, DescriptorType}, image::PipelineStageFlags, memory::AccessFlags, pipeline::{compute::ComputePipeline, layout::{PipelineLayout, PipelineLayoutBuilder}, shader::{ShaderCreateInfo, ShaderStageFlags}}, NfPtr};

use crate::algorithms::{StarlitShaderExecute, StarlitShaderKernel, StarlitStrategyInternal, StarlitStrategyState, Strategy};

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ScatterMemoryInputDSPC {
    pub index_count: u32,
    // in terms of u32
    pub payload_size: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ScatterMemoryInput {
    pub index_count: u32,
    pub payload_size: u32,
    pub indices: NfPtr,
    pub payload_src: NfPtr,
    pub payload_dst: NfPtr,
}
pub struct ScatterMemory {
    pipeline: Arc<ComputePipeline>,
    layout: Arc<PipelineLayout>,
    state: StarlitStrategyState<ScatterMemoryInput>,
}

const SCATTER_MEMORY_DS: &[u8] = include_bytes!("build/scatter_ds.comp.spv");
impl StarlitShaderKernel for ScatterMemory {
    type Input = ScatterMemoryInput;
    fn strategy(device: Arc<nightfall_core::device::LogicalDevice>, strategy: crate::algorithms::StarlitStrategy) -> Result<Self, crate::error::StarlitError> {
        let (layout, strategy, shader) = match strategy.strategy {
            Strategy::UsesDescriptorSets { pool } => {
                let shader = nightfall_core::pipeline::shader::Shader::new(device.clone(), ShaderCreateInfo {
                    entry: "main\0",
                    // spirv-opt works better with descriptor sets. This is from personal tests though.
                    data: &SCATTER_MEMORY_DS,
                    stage: ShaderStageFlags::COMPUTE,
                })?;
                let desc_layout = DescriptorLayoutBuilder::new()
                    // Indices : Input
                    .add_binding(0, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                    // Payload Source : Input
                    .add_binding(1, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                    // Payload Destination : Output
                    .add_binding(2, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                    .build(device.clone());
                let layout = PipelineLayoutBuilder::new()
                    .add_descriptor_layout(desc_layout.layout())
                    .add_push_constant::<ScatterMemoryInputDSPC>(ShaderStageFlags::COMPUTE)
                    .build(device.clone());
                let sets = pool.allocate(&[desc_layout])?.collect::<_>();
                (layout, StarlitStrategyInternal::UsesDescriptorSets { sets }, shader)
            }
            _ => panic!("Not Implemented"),
        };
        let pipeline = Arc::new(ComputePipeline::new(device.clone(), layout.clone(), shader.clone())?);
        Ok(Self { pipeline, layout, state: StarlitStrategyState { strategy, input: None } })
    }
    fn bind(&self, command_buffer: ash::vk::CommandBuffer) {
        self.pipeline.bind(command_buffer)
    }
    fn register(&mut self, input: &Self::Input) -> Result<(), crate::error::StarlitError> {
        match &self.state.strategy {
            StarlitStrategyInternal::UsesDescriptorSets { sets } => {
                let writer = nightfall_core::descriptors::DescriptorWriter::new()
                    .add_storage_buffer(sets[0].set(), 1, 0, 0, &input.indices.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0].set(), 1, 1, 0, &input.payload_src.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0].set(), 1, 2, 0, &input.payload_dst.as_descriptor_buffer_info());
                writer.write(self.pipeline.device());
            }
            StarlitStrategyInternal::UsesDeviceAddress => {
                return Err(crate::error::StarlitError::Internal("Not Implemented".into()));
            }
            StarlitStrategyInternal::HybridDeviceAddressDescriptorSets { pool } => {
                return Err(crate::error::StarlitError::Internal("Not Implemented".into()));
            }
        }
        self.state.input = Some(*input);
        Ok(())
    }
    
    fn input(&self, command_buffer: vk::CommandBuffer) -> Result<(), crate::error::StarlitError> {
        if let Some(input) = &self.state.input {
            match &self.state.strategy {
                StarlitStrategyInternal::UsesDescriptorSets { sets } => {
                    let dspc = ScatterMemoryInputDSPC {
                        index_count: input.index_count,
                        payload_size: input.payload_size,
                    };
                    self.pipeline.device().push_constants(
                        command_buffer, 
                        self.pipeline.layout().get_layout(), 
                        ShaderStageFlags::COMPUTE, 
                        0, 
                        &dspc
                    );

                    unsafe { 
                        self.pipeline.device().device().cmd_bind_descriptor_sets(
                            command_buffer, 
                            vk::PipelineBindPoint::COMPUTE, 
                            self.pipeline.layout().get_layout(), 
                            0, 
                            &[sets[0].set()], 
                            &[]
                        ) 
                    }
                }
                StarlitStrategyInternal::UsesDeviceAddress => {
                    return Err(crate::error::StarlitError::Internal("Not Implemented".into()));
                }
                StarlitStrategyInternal::HybridDeviceAddressDescriptorSets { pool } => {
                    return Err(crate::error::StarlitError::Internal("Not Implemented".into()));
                }
            }
        }
        Ok(())
    }
    fn dispatch(&self, command_buffer: vk::CommandBuffer, dispatch_x: u32, dispatch_y: u32, dispatch_z: u32) {
        self.pipeline.dispatch(command_buffer, dispatch_x, 1, 1);
    }
    fn dispatch_indirect(&self, command_buffer: vk::CommandBuffer, indirect: nightfall_core::buffers::BufferOffset) {
        self.pipeline.dispatch_indirect(command_buffer, indirect);
    }
    fn dispatch_with_barrier(&self, command_buffer: vk::CommandBuffer, dispatch_x: u32, dispatch_y: u32, dispatch_z: u32) -> Option<Barriers> {
        self.dispatch(command_buffer, dispatch_x, dispatch_y, dispatch_z);
        let input = self.state.input.as_ref().expect("no input has been registered");
        let mut barrier = input.payload_dst.as_buffer_memory_barrier(AccessFlags::SHADER_WRITE, AccessFlags::SHADER_READ|AccessFlags::SHADER_WRITE);
        Some(Barriers::new(PipelineStageFlags::COMPUTE_SHADER, PipelineStageFlags::COMPUTE_SHADER, vec![], vec![], vec![barrier]))
    }
    fn dispatch_indirect_with_barrier(&self, command_buffer: vk::CommandBuffer, indirect: nightfall_core::buffers::BufferOffset) -> Option<Barriers> {
        self.dispatch_indirect(command_buffer, indirect);
        let input = self.state.input.as_ref().expect("no input has been registered");
        let mut barrier = input.payload_dst.as_buffer_memory_barrier(AccessFlags::SHADER_WRITE, AccessFlags::SHADER_READ|AccessFlags::SHADER_WRITE);
        Some(Barriers::new(PipelineStageFlags::COMPUTE_SHADER, PipelineStageFlags::COMPUTE_SHADER, vec![], vec![], vec![barrier]))
    }
    fn get_state(&self) -> &StarlitStrategyState<Self::Input> {
        &self.state
    }
}


impl StarlitShaderExecute for ScatterMemory {
    type Input = <Self as StarlitShaderKernel>::Input;
    fn execute(&mut self, command_buffer: &nightfall_core::commands::CommandPoolAllocation, input: &Self::Input) -> Result<(), crate::error::StarlitError> {
        match &self.state.strategy {
            StarlitStrategyInternal::UsesDescriptorSets { sets } => {
                let writer = nightfall_core::descriptors::DescriptorWriter::new()
                    .add_storage_buffer(sets[0].set(), 1, 0, 0, &input.indices.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0].set(), 1, 1, 0, &input.payload_src.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0].set(), 1, 2, 0, &input.payload_dst.as_descriptor_buffer_info());
                writer.write(self.pipeline.device());

                let dspc = ScatterMemoryInputDSPC {
                    index_count: input.index_count,
                    payload_size: input.payload_size,
                };
                self.pipeline.device().push_constants(
                    command_buffer.get_command_buffer(), 
                    self.pipeline.layout().get_layout(), 
                    ShaderStageFlags::COMPUTE, 
                    0, 
                    &dspc
                );

                unsafe { 
                    self.pipeline.device().device().cmd_bind_descriptor_sets(
                        command_buffer.get_command_buffer(), 
                        vk::PipelineBindPoint::COMPUTE, 
                        self.pipeline.layout().get_layout(), 
                        0, 
                        &[sets[0].set()], 
                        &[]
                    ) 
                }
                self.bind(command_buffer.get_command_buffer());
                self.dispatch(command_buffer.get_command_buffer(), input.index_count.div_ceil(1024), 1, 1);
            }
            StarlitStrategyInternal::UsesDeviceAddress => {
                panic!("Not Implemented");
            }
            StarlitStrategyInternal::HybridDeviceAddressDescriptorSets { pool } => {
                panic!("Not Implemented");
            }
        }
        self.state.input = Some(*input);
        Ok(())
    }
    fn execute_with_barrier(&mut self, command_buffer: &nightfall_core::commands::CommandPoolAllocation, input: &Self::Input) -> Result<Option<Barriers>, crate::error::StarlitError> {
        self.execute(command_buffer, input)?;
        let barrier = input.payload_dst.as_buffer_memory_barrier(AccessFlags::SHADER_WRITE, AccessFlags::SHADER_READ);
        let barriers = Barriers::new(PipelineStageFlags::COMPUTE_SHADER, PipelineStageFlags::COMPUTE_SHADER, vec![], vec![], vec![
            barrier
        ]);
        Ok(Some(barriers))
    }
}


#[cfg(test)]
mod test {
    use std::sync::Arc;

    use ash::vk::SubmitInfo;
    use nightfall_core::{buffers::{BufferUsageFlags, MemoryPropertyFlags}, commands::{CommandBufferBeginInfo, CommandBufferLevel, CommandPool, CommandPoolCreateFlags}, descriptors::{DescriptorPoolBuilder, DescriptorPoolCreateFlags, DescriptorType}, device::{LogicalDevice, LogicalDeviceBuilder}, instance::InstanceBuilder, queue::{DeviceQueueCreateFlags, Queue, QueueFlags}, sync::Fence, AsNfptr, Version};
    use starlit_alloc::{GeneralAllocator, GpuAllocators, StandardAllocator};

    use crate::{algorithms::{StarlitShaderExecute, StarlitShaderKernel, StarlitStrategy}, SlVec};

    use super::{ScatterMemory, ScatterMemoryInput};

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
    fn scatter_test() {
        let (device, mut queues) = barebones();
        let compute_queue = queues.next().unwrap();
        let descriptor_pool = DescriptorPoolBuilder::new()
            .add_max_sets(48)
            .set_flag(DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .add_pool_size(DescriptorType::STORAGE_BUFFER, 32)
            .add_pool_size(DescriptorType::UNIFORM_BUFFER, 8)
            .add_pool_size(DescriptorType::COMBINED_IMAGE_SAMPLER, 8)
            .build(compute_queue.device());
        let mut scatter = ScatterMemory::strategy(device.clone(), StarlitStrategy::new(crate::algorithms::Strategy::UsesDescriptorSets { pool: descriptor_pool.clone() }, false)).unwrap();
        let command_pool = CommandPool::new(device.clone(), CommandPoolCreateFlags::empty(), compute_queue.family_index()).unwrap();
        let cmd = unsafe { command_pool.allocate_command_buffers(CommandBufferLevel::PRIMARY, 1).unwrap().next().unwrap() };
        let allocators = StandardAllocator::new(device.clone()).unwrap();
        let host_freelist = allocators.freelist(BufferUsageFlags::STORAGE_BUFFER, MemoryPropertyFlags::HOST_VISIBLE_COHERENT).unwrap();
        let payload_src = SlVec::from_iter((0..512).into_iter(), host_freelist.clone()).unwrap();
        let payload_dst = SlVec::<i32, dyn GeneralAllocator>::new_zeroed(512, host_freelist.clone()).unwrap();
        let indices = SlVec::from_iter((0..512).into_iter().rev(), host_freelist.clone()).unwrap();

        cmd.begin(CommandBufferBeginInfo::default()).unwrap();
        scatter.execute(&cmd, &ScatterMemoryInput {
            index_count: indices.len() as u32,
            indices: unsafe { indices.as_nfptr() },
            payload_src: unsafe { payload_src.as_nfptr() },
            payload_dst: unsafe { payload_dst.as_nfptr() },
            payload_size: 1,
        });
        cmd.end().unwrap();
        let fence = Fence::new(device.clone(), false);
        compute_queue.submit_raw(&[
            SubmitInfo::builder()
                .command_buffers(&[cmd.get_command_buffer()])
                .build()
        ], &fence).unwrap();
        fence.wait_max().unwrap();
        for (host_i, i) in payload_dst.iter().zip((0..512).rev()) {
            assert!(*host_i == i);
        }
    }
    #[derive(Debug)]
    struct LargeValue {
        x: u32,
        y: u32,
        z: u32,
        w: u32,
    }
    #[test]
    fn scatter_sized_test() {
        let (device, mut queues) = barebones();
        let compute_queue = queues.next().unwrap();
        let descriptor_pool = DescriptorPoolBuilder::new()
            .add_max_sets(48)
            .set_flag(DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .add_pool_size(DescriptorType::STORAGE_BUFFER, 32)
            .add_pool_size(DescriptorType::UNIFORM_BUFFER, 8)
            .add_pool_size(DescriptorType::COMBINED_IMAGE_SAMPLER, 8)
            .build(compute_queue.device());
        let mut scatter = ScatterMemory::strategy(device.clone(), StarlitStrategy::new(crate::algorithms::Strategy::UsesDescriptorSets { pool: descriptor_pool.clone() }, false)).unwrap();
        let command_pool = CommandPool::new(device.clone(), CommandPoolCreateFlags::empty(), compute_queue.family_index()).unwrap();
        let cmd = unsafe { command_pool.allocate_command_buffers(CommandBufferLevel::PRIMARY, 1).unwrap().next().unwrap() };
        let allocators = StandardAllocator::new(device.clone()).unwrap();
        let host_freelist = allocators.freelist(BufferUsageFlags::STORAGE_BUFFER, MemoryPropertyFlags::HOST_VISIBLE_COHERENT).unwrap();
        let payload_src = SlVec::from_iter((0..512).into_iter().map(|i|{
            LargeValue {
                x: i,
                y: i,
                z: i,
                w: i,
            }
        }), host_freelist.clone()).unwrap();
        let payload_dst = SlVec::<LargeValue, dyn GeneralAllocator>::new_zeroed(512, host_freelist.clone()).unwrap();
        let indices = SlVec::from_iter((0..512).into_iter().rev(), host_freelist.clone()).unwrap();

        cmd.begin(CommandBufferBeginInfo::default()).unwrap();
        scatter.execute(&cmd, &ScatterMemoryInput {
            index_count: indices.len() as u32,
            indices: unsafe { indices.as_nfptr() },
            payload_src: unsafe { payload_src.as_nfptr() },
            payload_dst: unsafe { payload_dst.as_nfptr() },
            payload_size: 4 as u32,
        });
        cmd.end().unwrap();
        let fence = Fence::new(device.clone(), false);
        compute_queue.submit_raw(&[
            SubmitInfo::builder()
                .command_buffers(&[cmd.get_command_buffer()])
                .build()
        ], &fence).unwrap();
        fence.wait_max().unwrap();
        for (host_i, i) in payload_dst.iter().zip((0..512).rev()) {
            assert!(host_i.x == i);
            assert!(host_i.y == i);
            assert!(host_i.z == i);
            assert!(host_i.w == i);
        }
    }
}