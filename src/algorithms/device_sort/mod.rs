use std::sync::Arc;

use ash::vk;
use nightfall_core::{commands::CommandPoolAllocation, descriptors::{DescriptorLayout, DescriptorLayoutBuilder, DescriptorSetAllocation, DescriptorSetLayoutCreateFlags, DescriptorType, DescriptorWriter}, device::LogicalDevice, image::PipelineStageFlags, memory::{AccessFlags, DependencyFlags}, pipeline::{compute::ComputePipeline, layout::{self, PipelineLayout, PipelineLayoutBuilder}, shader::{ShaderCreateInfo, ShaderStageFlags}}, NfPtr};
use starlit_alloc::GeneralAllocator;

use crate::{error::StarlitError, SlVec};

use super::{onesweep::tuning::TuningParameters, StarlitShaderAlgorithm, StarlitStrategyInternal, Strategy};

// DescriptorLayout
// GlobalHistogram : Binding = 0
// HistogramSort   : Binding = 1
// HistogramPass   : Binding = 2
// PushConstant
// uint thread_blocks
// uint radix_shift
// uint num_keys

pub struct DeviceRadixSortState<A: GeneralAllocator> {
    pub global_histogram: SlVec<u32, A>,
    pub sort: NfPtr,
    pub alt: SlVec<u32, A>,
    pub pass: SlVec<u32, A>,
    pub num_keys: usize,
    pub thread_blocks: usize,
}
#[repr(C)]
pub struct DeviceRadixSortPC {
    num_keys: u32,
    threadblocks: u32,
    radix_shift: u32,
}
pub struct DeviceRadixSortInput {
    pub sort: NfPtr,
    pub num_keys: usize,
}
pub struct DeviceRadixSort<A: GeneralAllocator> {
    device: Arc<LogicalDevice>,
    descriptor_layout: Arc<DescriptorLayout>,
    layout: Arc<PipelineLayout>,
    reset: ComputePipeline,
    upsweep: ComputePipeline,
    scan: ComputePipeline,
    downsweep: ComputePipeline,
    tuning: TuningParameters,
    strategy: StarlitStrategyInternal,
    freelist: Arc<A>,
    pub input: Option<DeviceRadixSortState<A>>,
}
impl<A: GeneralAllocator> DeviceRadixSort<A> {
    const PART_SIZE: usize = 7680; // 3840
    fn initialize_static_buffers(freelist: Arc<A>, input: &mut DeviceRadixSortState<A>, thread_blocks: usize) -> Result<(), StarlitError> {
        input.global_histogram = SlVec::<u32, A>::new_zeroed(1024, freelist.clone())?;
        Ok(())
        // input.global_partition_tiles = VkVec::<u32, A>::new_zeroed(4 as usize, freelist.clone()).unwrap();
    }
    fn initialize_buffers(freelist: Arc<A>, input: &mut DeviceRadixSortState<A>, thread_blocks: usize, num_keys: usize) -> Result<(), StarlitError> {
        input.alt = SlVec::<u32, A>::new_zeroed(num_keys, freelist.clone())?;
        input.pass = SlVec::<u32, A>::new_zeroed(thread_blocks*256, freelist.clone())?;
        Ok(())
    }
    fn radix_pass(&self, command_buffer: &CommandPoolAllocation, radix_shift: u32, input: &DeviceRadixSortState<A>) {
        self.device.push_constants(
            command_buffer.get_command_buffer(), 
            self.layout.get_layout(), 
            ShaderStageFlags::COMPUTE, 
            0, 
            &DeviceRadixSortPC {
                num_keys: input.num_keys as u32,
                threadblocks: input.thread_blocks as u32,
                radix_shift,
        });
        self.upsweep.bind(command_buffer.get_command_buffer());
        self.upsweep.dispatch(command_buffer.get_command_buffer(), input.thread_blocks as u32, 1, 1);
        command_buffer.pipeline_barrier(
            PipelineStageFlags::COMPUTE_SHADER, 
            PipelineStageFlags::COMPUTE_SHADER,
            DependencyFlags::empty(), 
            &[], 
            &[
                input.pass.get_allocation().as_buffer_memory_barrier(AccessFlags::SHADER_WRITE|AccessFlags::SHADER_READ, AccessFlags::SHADER_WRITE|AccessFlags::SHADER_READ),
                ], 
            &[]
        );
        self.scan.bind(command_buffer.get_command_buffer());
        self.scan.dispatch(command_buffer.get_command_buffer(), 256, 1, 1);
        command_buffer.pipeline_barrier(
            PipelineStageFlags::COMPUTE_SHADER, 
            PipelineStageFlags::COMPUTE_SHADER,
            DependencyFlags::empty(), 
            &[], 
            &[
                input.global_histogram.get_allocation().as_buffer_memory_barrier(AccessFlags::SHADER_WRITE, AccessFlags::SHADER_WRITE),
                input.pass.get_allocation().as_buffer_memory_barrier(AccessFlags::SHADER_WRITE, AccessFlags::SHADER_WRITE),
                ], 
            &[]
        );
        self.downsweep.bind(command_buffer.get_command_buffer());
        self.downsweep.dispatch(command_buffer.get_command_buffer(), input.thread_blocks as u32, 1, 1);
        command_buffer.pipeline_barrier(
            PipelineStageFlags::COMPUTE_SHADER, 
            PipelineStageFlags::COMPUTE_SHADER,
            DependencyFlags::empty(), 
            &[], 
            &[
                input.sort.as_buffer_memory_barrier(AccessFlags::SHADER_WRITE, AccessFlags::SHADER_WRITE),
                input.alt.get_allocation().as_buffer_memory_barrier(AccessFlags::SHADER_WRITE, AccessFlags::SHADER_WRITE),
                ], 
            &[]
        );
    }
}
const RESET_DS_SG: &[u8] = include_bytes!("build/reset.comp.spv");
const UPSWEEP_DS_SG: &[u8] = include_bytes!("build/upsweep.comp.spv");
const SCAN_DS_SG: &[u8] = include_bytes!("build/scan.comp.spv");
const DOWNSWEEP_DS_SG: &[u8] = include_bytes!("build/downsweep.comp.spv");

impl<A: GeneralAllocator> StarlitShaderAlgorithm<A> for DeviceRadixSort<A> {
    type Input = DeviceRadixSortInput;
    fn strategy(device: Arc<nightfall_core::device::LogicalDevice>, freelist: Arc<A>, strategy: super::StarlitStrategy) -> Result<Self, crate::error::StarlitError> {
        let (reset, upsweep, scan, downsweep, desc_layout, layout, strategy) = match strategy.strategy {
            Strategy::UsesDescriptorSets { pool } => {
                if strategy.use_subgroups {
                    let reset = nightfall_core::pipeline::shader::Shader::new(device.clone(), ShaderCreateInfo {
                        entry: "main\0",
                        data: &RESET_DS_SG,
                        stage: ShaderStageFlags::COMPUTE,
                    })?;
                    let upsweep = nightfall_core::pipeline::shader::Shader::new(device.clone(), ShaderCreateInfo {
                        entry: "main\0",
                        data: &UPSWEEP_DS_SG,
                        stage: ShaderStageFlags::COMPUTE,
                    })?;
                    let scan = nightfall_core::pipeline::shader::Shader::new(device.clone(), ShaderCreateInfo {
                        entry: "main\0",
                        data: &SCAN_DS_SG,
                        stage: ShaderStageFlags::COMPUTE,
                    })?;
                    let downsweep = nightfall_core::pipeline::shader::Shader::new(device.clone(), ShaderCreateInfo {
                        entry: "main\0",
                        data: &DOWNSWEEP_DS_SG,
                        stage: ShaderStageFlags::COMPUTE,
                    })?;
                    let desc_layout = DescriptorLayoutBuilder::new()
                        .uses_binding_flags()
                        .set_flag(DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                        // GlobalHistogram  : Inout
                        .add_binding(0, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                        // HistogramPass    : Inout
                        .add_binding(1, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                        // HistogramSort    : Inout
                        .add_binding(2, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                        // AlternateBuffer  : Inout
                        .add_binding(3, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                        .build(device.clone());
                    let allocation_layouts = [desc_layout.clone(), desc_layout.clone()];
                    let sets = pool.allocate(&allocation_layouts)?.collect::<Vec<_>>();
                    let layout = PipelineLayoutBuilder::new()
                        .add_descriptor_layout(desc_layout.layout())
                        .add_push_constant::<DeviceRadixSortPC>(ShaderStageFlags::COMPUTE)
                        .build(device.clone());
                    (reset, upsweep, scan, downsweep, desc_layout, layout, StarlitStrategyInternal::UsesDescriptorSets { sets })
                } else {
                    return Err(StarlitError::Internal("Not Implemented".into()));
                }
            }
            _ => {
                return Err(StarlitError::Internal("Not Implemented".into()));
            }
        };
        let shaders = [reset.clone(), upsweep.clone(), scan.clone(), downsweep.clone()];
        let layouts = [layout.clone(), layout.clone(), layout.clone(), layout.clone()];
        let mut pipelines = ComputePipeline::new_group(device.clone(), &layouts, &shaders)?;
        let reset = pipelines.next().unwrap();
        let upsweep = pipelines.next().unwrap();
        let scan = pipelines.next().unwrap();
        let downsweep = pipelines.next().unwrap();
        Ok(Self { 
            device: device.clone(), 
            reset, 
            upsweep, 
            scan, 
            downsweep, 
            tuning: TuningParameters::new(device.clone()), 
            freelist: freelist.clone(), 
            input: None, strategy, 
            descriptor_layout: desc_layout, 
            layout 
        })
    }
    fn register(&mut self, input: &Self::Input) -> Result<(), crate::error::StarlitError> {
        if let Some(prev_input) = &mut self.input {
            let thread_blocks = input.num_keys.div_ceil(self.tuning.partition_size as usize) as u32; // OneSweepU32::<A>::PART_SIZE
            if prev_input.num_keys < input.num_keys {
                prev_input.alt = SlVec::<u32, A>::new_zeroed(input.num_keys, self.freelist.clone())?;
            }
            if prev_input.thread_blocks < thread_blocks as usize {
                prev_input.pass = SlVec::<u32, A>::new_zeroed(256*thread_blocks as usize, self.freelist.clone())?;
            }
            prev_input.num_keys = input.num_keys;
            prev_input.thread_blocks = thread_blocks as usize;
            prev_input.sort = input.sort;
        } else {
            let thread_blocks = input.num_keys.div_ceil(self.tuning.partition_size as usize) as u32; // OneSweepU32::<A>::PART_SIZE
            let mut state = DeviceRadixSortState {
                global_histogram: SlVec::new(self.freelist.clone()),
                alt: SlVec::new(self.freelist.clone()),
                pass: SlVec::new(self.freelist.clone()),
                num_keys: input.num_keys,
                thread_blocks: thread_blocks as usize,
                sort: input.sort,
            };
            Self::initialize_buffers(self.freelist.clone(), &mut state, thread_blocks as usize, input.num_keys as usize);
            Self::initialize_static_buffers(self.freelist.clone(), &mut state, thread_blocks as usize);
            self.input = Some(state);
        }
        let get_input = self.input.as_ref().unwrap();
        match &self.strategy {
            StarlitStrategyInternal::UsesDescriptorSets { sets } => {
                let writer = DescriptorWriter::new()
                    .add_storage_buffer(sets[0].set(), 1, 0, 0, &get_input.global_histogram.get_allocation().as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0].set(), 1, 1, 0, &get_input.pass.get_allocation().as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0].set(), 1, 2, 0, &get_input.sort.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0].set(), 1, 3, 0, &get_input.alt.get_allocation().as_descriptor_buffer_info());
                writer.write(self.device.clone());
                let writer = DescriptorWriter::new()
                    .add_storage_buffer(sets[1].set(), 1, 0, 0, &get_input.global_histogram.get_allocation().as_descriptor_buffer_info())
                    .add_storage_buffer(sets[1].set(), 1, 1, 0, &get_input.pass.get_allocation().as_descriptor_buffer_info())
                    .add_storage_buffer(sets[1].set(), 1, 2, 0, &get_input.alt.get_allocation().as_descriptor_buffer_info())
                    .add_storage_buffer(sets[1].set(), 1, 3, 0, &get_input.sort.as_descriptor_buffer_info());
                writer.write(self.device.clone());
            }
            _ => {
                return Err(StarlitError::Internal("Not Implemented".into()));
            }
        }
        Ok(())
    }
    fn fill_command_listing(&mut self, command_buffer: &nightfall_core::commands::CommandPoolAllocation) -> Result<(), StarlitError> {
        let input = self.input.as_ref().ok_or(StarlitError::NoInputRegistered("DeviceRadixSort does not have any input registered. Consider using `register'".into()))?;
        match &self.strategy {
            StarlitStrategyInternal::UsesDescriptorSets { sets } => {
                unsafe {
                    self.device.device().cmd_bind_descriptor_sets(
                        command_buffer.get_command_buffer(), 
                        vk::PipelineBindPoint::COMPUTE, 
                        self.layout.get_layout(), 
                        0, 
                        &[sets[0].set()], 
                        &[]
                    );
                }
                self.reset.bind(command_buffer.get_command_buffer());
                self.reset.dispatch(command_buffer.get_command_buffer(), input.thread_blocks as u32, 1, 1);
                command_buffer.pipeline_barrier(
                    PipelineStageFlags::COMPUTE_SHADER, 
                    PipelineStageFlags::COMPUTE_SHADER,
                    DependencyFlags::empty(), 
                    &[], 
                    &[
                        input.global_histogram.get_allocation().as_buffer_memory_barrier(AccessFlags::SHADER_WRITE, AccessFlags::SHADER_WRITE),
                        ], 
                    &[]
                );
                // self.radix_pass(&command_buffer, 0, input);
                // let mut swap = 1;
                let mut swap = 0;
                for i in 0..4 {
                    unsafe {
                        self.device.device().cmd_bind_descriptor_sets(
                            command_buffer.get_command_buffer(), 
                            vk::PipelineBindPoint::COMPUTE, 
                            self.layout.get_layout(), 
                            0, 
                            &[sets[swap].set()], 
                            &[]
                        );
                    }
                    self.radix_pass(&command_buffer, i*8, input);
                    swap = swap+1&1;
                }
            }
            _ => {
                return Err(StarlitError::Internal("Not Implemented".into()));
            }
        }
        Ok(())
    }
}

