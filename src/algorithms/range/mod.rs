use std::sync::Arc;

use ash::vk;
use nightfall_core::{barriers::Barriers, descriptors::{DescriptorLayout, DescriptorLayoutBuilder, DescriptorSetLayoutCreateFlags, DescriptorType, DescriptorWriter}, device::LogicalDevice, image::PipelineStageFlags, memory::AccessFlags, pipeline::{compute::ComputePipeline, layout::{PipelineLayout, PipelineLayoutBuilder}, shader::{Shader, ShaderCreateInfo, ShaderStageFlags}}, NfPtr};


use crate::error::StarlitError;

use super::{StarlitShaderAlgorithm, StarlitShaderExecute, StarlitShaderKernel, StarlitStrategyInternal, StarlitStrategyState, Strategy};

pub struct DeviceRangeInputDSPC {
    start: u32,
    size: u32,
}
#[derive(Clone)]
pub struct DeviceRangeInput {
    /// start of the iteration
    pub start: u32,
    /// end of the iteration
    pub end: u32,
    /// allocation to place the iteration
    pub value: NfPtr,
}
pub struct DeviceRange {
    device: Arc<LogicalDevice>,
    range: ComputePipeline,
    desc_layout: Arc<DescriptorLayout>,
    layout: Arc<PipelineLayout>,
    state: StarlitStrategyState<DeviceRangeInput>,
}
const RANGE_DS: &[u8] = include_bytes!("build/range.comp.spv");

impl StarlitShaderKernel for DeviceRange {
    type Input = DeviceRangeInput;
    fn strategy(device: std::sync::Arc<nightfall_core::device::LogicalDevice>, strategy: super::StarlitStrategy) -> Result<Self, crate::error::StarlitError> {
        let (shader, desc_layout, layout, strategy) = match strategy.strategy {
            Strategy::UsesDescriptorSets { pool } => {
                let shader = Shader::new(device.clone(), ShaderCreateInfo {
                    entry: "main\0",
                    data: &RANGE_DS,
                    stage: ShaderStageFlags::COMPUTE,
                })?;
                let desc_layout = DescriptorLayoutBuilder::new()
                        .uses_binding_flags()
                        .set_flag(DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                        // GlobalHistogram  : Inout
                        .add_binding(0, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                        .build(device.clone());
                let pipeline_layout = PipelineLayoutBuilder::new()
                    .add_push_constant::<DeviceRangeInputDSPC>(ShaderStageFlags::COMPUTE)
                    .add_descriptor_layout(desc_layout.layout())
                    .build(device.clone());
                let sets = pool.allocate(&[desc_layout.clone()])?.collect::<Vec<_>>();
                (shader, desc_layout, pipeline_layout, StarlitStrategyInternal::UsesDescriptorSets { sets })
            },
            _ => {
                return Err(StarlitError::Internal("Not Implemented".into()));
            }
        };
        let pipeline = ComputePipeline::new(device.clone(), layout.clone(), shader.clone())?;
        
        Ok(Self { device, range: pipeline, desc_layout, layout, state: StarlitStrategyState { strategy, input: None } })
    }
    fn register(&mut self, input: &Self::Input) -> Result<(), StarlitError> {
        // let state = DeviceRangeState {
        //     start: input.start,
        //     size: input.end,
        //     value: input.value,
        // };
        match &self.state.strategy {
            StarlitStrategyInternal::UsesDescriptorSets { sets } => {
                let writer = DescriptorWriter::new()
                    .add_storage_buffer(sets[0].set(), 1, 0, 0, &input.value.as_descriptor_buffer_info());
                writer.write(self.device.clone()); 
            }
            _ => {
                return Err(StarlitError::Internal("Not Implemented".into()));
            }
        }
        self.state.input = Some(input.clone());
        Ok(())
    }
    fn input(&self, command_buffer: vk::CommandBuffer) -> Result<(), StarlitError> {
        match &self.state.strategy {
            StarlitStrategyInternal::UsesDescriptorSets { sets } => {
                let input = self.state.input.as_ref().ok_or(StarlitError::NoInputRegistered("DeviceRange does not have any input registered. Consider using `register'".into()))?;
                let dspc = DeviceRangeInputDSPC {
                    start: input.start,
                    size: input.end,
                };
                self.device.push_constants(command_buffer, self.layout.get_layout(), ShaderStageFlags::COMPUTE, 0, &dspc);
                unsafe { 
                    self.range.device().device().cmd_bind_descriptor_sets(
                        command_buffer, 
                        vk::PipelineBindPoint::COMPUTE, 
                        self.range.layout().get_layout(), 
                        0, 
                        &[sets[0].set()], 
                        &[]
                    ) 
                }
            }
            _ => {
                return Err(StarlitError::Internal("Not Implemented".into()));
            }
        }
        Ok(())
    }
    fn bind(&self, command_buffer: vk::CommandBuffer) {
        self.range.bind(command_buffer);
    }
    fn dispatch(&self, command_buffer: vk::CommandBuffer, dispatch_x: u32, dispatch_y: u32, dispatch_z: u32) {
        self.range.dispatch(command_buffer, dispatch_x, 1, 1);
    }
    fn dispatch_indirect(&self, command_buffer: vk::CommandBuffer, indirect: nightfall_core::buffers::BufferOffset) {
        self.range.dispatch_indirect(command_buffer, indirect);
    }
    fn dispatch_with_barrier(&self, command_buffer: vk::CommandBuffer, dispatch_x: u32, dispatch_y: u32, dispatch_z: u32) -> Option<nightfall_core::barriers::Barriers> {
        self.dispatch(command_buffer, dispatch_x, dispatch_y, dispatch_z);
        let input = self.state.input.as_ref().expect("no input has been registered");
        let mut offset: nightfall_core::barriers::BufferMemoryBarrier = input.value.as_buffer_memory_barrier(AccessFlags::SHADER_WRITE, AccessFlags::SHADER_READ|AccessFlags::SHADER_WRITE);
        Some(Barriers::new(PipelineStageFlags::COMPUTE_SHADER, PipelineStageFlags::COMPUTE_SHADER, vec![], vec![], vec![offset]))
    }
    fn dispatch_indirect_with_barrier(&self, command_buffer: vk::CommandBuffer, indirect: nightfall_core::buffers::BufferOffset) -> Option<Barriers> {
        self.dispatch_indirect(command_buffer, indirect);
        let input = self.state.input.as_ref().expect("no input has been registered");
        let mut offset = input.value.as_buffer_memory_barrier(AccessFlags::SHADER_WRITE, AccessFlags::SHADER_READ|AccessFlags::SHADER_WRITE);
        Some(Barriers::new(PipelineStageFlags::COMPUTE_SHADER, PipelineStageFlags::COMPUTE_SHADER, vec![], vec![], vec![offset]))
    }
    fn get_state(&self) -> &super::StarlitStrategyState<Self::Input> {
        &self.state
    }
}

impl StarlitShaderExecute for DeviceRange {
    type Input = <Self as StarlitShaderKernel>::Input;
    fn execute(&mut self, command_buffer: &nightfall_core::commands::CommandPoolAllocation, input: &Self::Input) -> Result<(), StarlitError> {
        match &self.state.strategy {
            StarlitStrategyInternal::UsesDescriptorSets { sets } => {
                let writer = DescriptorWriter::new()
                    .add_storage_buffer(sets[0].set(), 1, 0, 0, &input.value.as_descriptor_buffer_info());
                writer.write(self.device.clone());
                let dspc = DeviceRangeInputDSPC {
                    start: input.start,
                    size: input.end,
                };
                self.state.input = Some(input.clone());
                self.device.push_constants(command_buffer.get_command_buffer(), self.layout.get_layout(), ShaderStageFlags::COMPUTE, 0, &dspc);
                unsafe { 
                    self.range.device().device().cmd_bind_descriptor_sets(
                        command_buffer.get_command_buffer(), 
                        vk::PipelineBindPoint::COMPUTE, 
                        self.range.layout().get_layout(), 
                        0, 
                        &[sets[0].set()], 
                        &[]
                    ) 
                }
                self.range.bind(command_buffer.get_command_buffer());
                let dispatch_x = (input.end-input.start).div_ceil(1024);
                self.range.dispatch(command_buffer.get_command_buffer(), dispatch_x, 1, 1);
            }
            _ => {
                return Err(StarlitError::Internal("Not Implemented".into()));
            }
        }
        Ok(())
    }
    fn execute_with_barrier(&mut self, command_buffer: &nightfall_core::commands::CommandPoolAllocation, input: &Self::Input) -> Result<Option<Barriers>, StarlitError> {
        self.execute(command_buffer, input);
        let barrier = self.state.input.as_ref().unwrap().value.as_buffer_memory_barrier(AccessFlags::SHADER_WRITE, AccessFlags::SHADER_READ|AccessFlags::SHADER_WRITE);
        Ok(Some(Barriers::new(PipelineStageFlags::COMPUTE_SHADER, PipelineStageFlags::COMPUTE_SHADER, vec![], vec![], vec![barrier])))
    }
}