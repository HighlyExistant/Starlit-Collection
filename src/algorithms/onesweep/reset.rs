use std::sync::Arc;

use ash::vk;
use nightfall_core::{buffers::BufferOffset, descriptors::{DescriptorLayoutBuilder, DescriptorSetLayoutCreateFlags, DescriptorType}, pipeline::{compute::ComputePipeline, layout::PipelineLayoutBuilder, shader::{ShaderCreateInfo, ShaderStageFlags}}, NfPtr};

use crate::{algorithms::{StarlitShaderKernel, StarlitStrategyInternal, StarlitStrategyState, Strategy}, error::StarlitError};

#[derive(Clone, Copy)]
pub struct Radix256ResetInputDSPC {
    thread_blocks: u32,
}
#[derive(Clone, Copy)]
pub struct Radix256ResetInput {
    pub thread_blocks: u32,
    pub histogram_out: NfPtr,
    pub histogram_pass_out: NfPtr,
    pub partition_tile_indices_out: NfPtr,
}
pub struct Radix256Reset {
    pipeline: Arc<ComputePipeline>,
    state: StarlitStrategyState<<Self as StarlitShaderKernel>::Input>,
}
const ONESWEEP_RESET_DS: &[u8] = include_bytes!("build/onesweep_reset.comp.spv");
impl StarlitShaderKernel for Radix256Reset {
    type Input = Radix256ResetInput;
    fn strategy(device: Arc<nightfall_core::device::LogicalDevice>, strategy: crate::algorithms::StarlitStrategy) -> Result<Self, StarlitError> {
        let (layout, strategy, shader) = match strategy.strategy {
            Strategy::UsesDescriptorSets { pool } => {
                let shader = nightfall_core::pipeline::shader::Shader::new(device.clone(), ShaderCreateInfo {
                    entry: "main\0",
                    data: &ONESWEEP_RESET_DS,
                    stage: ShaderStageFlags::COMPUTE,
                })?;
                let desc_layout = DescriptorLayoutBuilder::new()
                    .uses_binding_flags()
                    .set_flag(DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                    // Global Histogram : Out
                    .add_binding(0, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                    // Histogram Pass : Out
                    .add_binding(1, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                    // Partition Tiles : Out
                    .add_binding(2, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                    .build(device.clone());
                    // .set_binding_flag(DescriptorBindingFlags::UPDATE_AFTER_BIND | DescriptorBindingFlags::PARTIALLY_BOUND)
            
                let layout = PipelineLayoutBuilder::new()
                    .add_descriptor_layout(desc_layout.layout())
                    .add_push_constant::<Radix256ResetInputDSPC>(ShaderStageFlags::COMPUTE)
                    .build(device.clone());
                let sets = pool.allocate(&[desc_layout.clone(), desc_layout.clone()])?.collect::<Vec<_>>();
                (layout, StarlitStrategyInternal::UsesDescriptorSets { sets }, shader)
            }
            _ => {
                return Err(StarlitError::Internal("Not Implemented".into()));
            }
        };
        let pipeline = Arc::new(ComputePipeline::new(device.clone(), layout.clone(), shader.clone())?);
        Ok(Self { pipeline, state: StarlitStrategyState { strategy, input: None } })
    }
    fn register(&mut self, input: &Self::Input) -> Result<(), StarlitError> {
        match &self.state.strategy {
            StarlitStrategyInternal::UsesDescriptorSets { sets } => {
                let writer = nightfall_core::descriptors::DescriptorWriter::new()
                    .add_storage_buffer(sets[0].set(), 1, 0, 0, &input.histogram_out.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0].set(), 1, 1, 0, &input.histogram_pass_out.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0].set(), 1, 2, 0, &input.partition_tile_indices_out.as_descriptor_buffer_info());
                writer.write(self.pipeline.device());
            }
            _ => {
                return Err(StarlitError::Internal("Not Implemented".into()));
            }
        }
        self.state.input = Some(*input);
        Ok(())
    }
    fn input(&self, command_buffer: ash::vk::CommandBuffer) -> Result<(), StarlitError> {
        if let Some(input) = self.state.input {
            match &self.state.strategy {
                StarlitStrategyInternal::UsesDescriptorSets { sets } => {
                    let dspc = Radix256ResetInputDSPC {
                        thread_blocks: input.thread_blocks,
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
                    Ok(())
                }
                _ => {
                    return Err(StarlitError::Internal("Not Implemented".into()));
                }
            }
        } else {
            Err(StarlitError::NoInputRegistered("Radix256Reset does not have any input registered. Consider using `register'".into()))
        }
    }
    fn bind(&self, command_buffer: vk::CommandBuffer) {
        self.pipeline.bind(command_buffer)
    }
    fn dispatch(&self, command_buffer: vk::CommandBuffer, dispatch_x: u32, dispatch_y: u32, dispatch_z: u32) {
        self.pipeline.dispatch(command_buffer, dispatch_x, 1, 1);
    }
    fn dispatch_indirect(&self, command_buffer: vk::CommandBuffer, indirect: BufferOffset) {
        self.pipeline.dispatch_indirect(command_buffer, indirect)
    }
    fn get_state(&self) -> &StarlitStrategyState<Self::Input> {
        &self.state
    }
}