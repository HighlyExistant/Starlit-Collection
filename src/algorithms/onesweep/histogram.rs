use std::sync::Arc;

use ash::vk;
use nightfall_core::{buffers::BufferOffset, descriptors::{DescriptorLayoutBuilder, DescriptorType}, pipeline::{compute::ComputePipeline, layout::PipelineLayoutBuilder, shader::{ShaderCreateInfo, ShaderStageFlags}}, NfPtr};

use crate::{algorithms::{StarlitShaderKernel, StarlitStrategy, StarlitStrategyInternal, StarlitStrategyState, Strategy}, error::StarlitError};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Radix256HistogramInputDSPC {
    pub num_keys: u32,
    pub thread_blocks: u32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Radix256HistogramInput {
    pub num_keys: u32,
    pub thread_blocks: u32,
    pub u32_in: NfPtr,
    pub histogram_out: NfPtr,
}
pub struct Radix256Histogram {
    pipeline: Arc<ComputePipeline>,
    state: StarlitStrategyState<<Self as StarlitShaderKernel>::Input>,
}
const RADIX256_HISTOGRAM_DS: &[u8] = include_bytes!("build/histogram_radix256.comp.spv");
const RADIX256_HISTOGRAM_KEYS7_DS: &[u8] = include_bytes!("build/keys7/histogram_radix256.comp.spv");
impl Radix256Histogram {
    pub fn push_constant(&self, command_buffer: vk::CommandBuffer, input: &Radix256HistogramInputDSPC) {
        self.pipeline.device().push_constants(
            command_buffer, 
            self.pipeline.layout().get_layout(), 
            ShaderStageFlags::COMPUTE, 
            0, 
            input
        );
    }
}
impl StarlitShaderKernel for Radix256Histogram {
    type Input = Radix256HistogramInput;
    fn strategy(device: Arc<nightfall_core::device::LogicalDevice>, strategy: StarlitStrategy) -> Result<Self, StarlitError> {
        let (layout, strategy, shader) = match strategy.strategy {
            Strategy::UsesDescriptorSets { pool } => {
                let shader = nightfall_core::pipeline::shader::Shader::new(device.clone(), ShaderCreateInfo {
                    entry: "main\0",
                    // spirv-opt works better with descriptor sets. This is from personal tests though.
                    data: &RADIX256_HISTOGRAM_DS,
                    stage: ShaderStageFlags::COMPUTE,
                })?;
                let desc_layout = DescriptorLayoutBuilder::new()
                    // Monoid : Input
                    .add_binding(0, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                    // Monoid : Output
                    .add_binding(1, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                    // State
                    .add_binding(2, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                    // Debug
                    .add_binding(3, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                    .build(device.clone());
                let layout = PipelineLayoutBuilder::new()
                    .add_descriptor_layout(desc_layout.layout())
                    .add_push_constant::<Radix256HistogramInputDSPC>(ShaderStageFlags::COMPUTE)
                    .build(device.clone());
                let sets = pool.allocate(&[desc_layout])?.collect::<_>();
                (layout, StarlitStrategyInternal::UsesDescriptorSets { sets }, shader)
            }
            Strategy::UsesDeviceAddress => {
                return Err(StarlitError::NotDeviceAddressable)
            }
            Strategy::HybridDeviceAddressDescriptorSets { pool } => {
                return Err(StarlitError::Internal("Not Implemented".into()))
            }
        };
        let pipeline = Arc::new(ComputePipeline::new(device.clone(), layout.clone(), shader.clone())?);
        Ok(Self { pipeline, state: StarlitStrategyState { strategy, input: None } })
    }
    fn bind(&self, command_buffer: ash::vk::CommandBuffer) {
        self.pipeline.bind(command_buffer)
    }
    fn register(&mut self, input: &Self::Input) -> Result<(), StarlitError> {
        match &self.state.strategy {
            StarlitStrategyInternal::UsesDescriptorSets { sets } => {
                let writer = nightfall_core::descriptors::DescriptorWriter::new()
                        .add_storage_buffer(sets[0].set(), 1, 0, 0, &input.u32_in.as_descriptor_buffer_info())
                        .add_storage_buffer(sets[0].set(), 1, 1, 0, &input.histogram_out.as_descriptor_buffer_info());
                writer.write(self.pipeline.device());
            }
            _ => {
                return Err(StarlitError::Internal("Not Implemented".into()));
            }
        }
        self.state.input = Some(*input);
        Ok(())
    }
    fn input(&self, command_buffer: vk::CommandBuffer) -> Result<(), StarlitError> {
        if let Some(input) = self.state.input {
            match &self.state.strategy {
                StarlitStrategyInternal::UsesDescriptorSets { sets } => {
                    let dspc = Radix256HistogramInputDSPC {
                        num_keys: input.num_keys,
                        thread_blocks: input.thread_blocks,
                    };
                    self.push_constant(command_buffer, &dspc);
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
                StarlitStrategyInternal::UsesDeviceAddress => {
                    return Err(StarlitError::NotDeviceAddressable)
                }
                StarlitStrategyInternal::HybridDeviceAddressDescriptorSets { pool } => {
                    return Err(StarlitError::Internal("Not Implemented".into()))
                }
            }
        } else {
            Err(StarlitError::NoInputRegistered("Radix256Histogram does not have any input registered. Consider using `register'".into()))

        }
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