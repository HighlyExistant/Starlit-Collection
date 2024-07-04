use std::sync::Arc;

use ash::vk;
use nightfall_core::{descriptors::{DescriptorLayoutBuilder, DescriptorType}, device::LogicalDevice, error::NightfallError, pipeline::{compute::ComputePipeline, layout::PipelineLayoutBuilder, shader::{ShaderCreateInfo, ShaderStageFlags}}, NfPtr};
use crate::{algorithms::{StarlitShaderKernel, StarlitStrategy, StarlitStrategyInternal, StarlitStrategyState, Strategy}, error::StarlitError};
#[derive(Clone, Copy)]
pub struct Radix256ScanInputPC {
    pub thread_blocks: u32,
}
#[derive(Clone)]
pub struct Radix256ScanInput {
    pub thread_blocks: u32,
    pub global_histogram_in: NfPtr,
    pub histogram_pass_out: NfPtr,
}

pub struct Radix256Scan {
    pipeline: Arc<ComputePipeline>,
    state: StarlitStrategyState<<Self as StarlitShaderKernel>::Input>,
}
const RADIX256_SCAN_DS_SG: &[u8] = include_bytes!("build/scan_radix256_sg.comp.spv");

impl Radix256Scan {
    pub fn push_constant(&self, command_buffer: vk::CommandBuffer, input: &Radix256ScanInputPC) {
        self.pipeline.device().push_constants(
            command_buffer, 
            self.pipeline.layout().get_layout(), 
            ShaderStageFlags::COMPUTE, 
            0, 
            input
        );
    }

}

impl StarlitShaderKernel for Radix256Scan {
    type Input = Radix256ScanInput;
    fn strategy(device: Arc<LogicalDevice>, strategy: StarlitStrategy) -> Result<Self, StarlitError> {
        let (layout, strategy, shader) = match strategy.strategy {
            Strategy::UsesDescriptorSets { pool } => {
                if strategy.use_subgroups {
                    let shader = nightfall_core::pipeline::shader::Shader::new(device.clone(), ShaderCreateInfo {
                        entry: "main\0",
                        // spirv-opt works better with descriptor sets. This is from personal tests though.
                        data: &RADIX256_SCAN_DS_SG,
                        stage: ShaderStageFlags::COMPUTE,
                    }).map_err(StarlitError::from)?;
                    let desc_layout = DescriptorLayoutBuilder::new()
                        // Global Histogram : Input
                        .add_binding(0, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                        // Histogram Pass : Output
                        .add_binding(1, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                        .build(device.clone());
                    let layout = PipelineLayoutBuilder::new()
                        .add_descriptor_layout(desc_layout.layout())
                        .add_push_constant::<Radix256ScanInputPC>(ShaderStageFlags::COMPUTE)
                        .build(device.clone());
                    let sets = pool.allocate(&[desc_layout])?.collect::<_>();
                    (layout, StarlitStrategyInternal::UsesDescriptorSets { sets }, shader)
                } else {
                        return Err(StarlitError::Internal("Not Implemented".into()));
                    }
                }
            Strategy::UsesDeviceAddress => {
                return Err(StarlitError::NotDeviceAddressable);
            }
            Strategy::HybridDeviceAddressDescriptorSets { pool } => {
                return Err(StarlitError::Internal("Not Implemented".into()));
            }
        };
        let pipeline = Arc::new(ComputePipeline::new(device.clone(), layout.clone(), shader.clone())?);
        Ok(Self { pipeline, state: StarlitStrategyState { strategy, input: None } })
    }
    fn register(&mut self, input: &Self::Input) -> Result<(), StarlitError> {
        match &self.state.strategy {
            StarlitStrategyInternal::UsesDescriptorSets { sets } => {
                let sets = sets.iter().map(|set|{set.set()}).collect::<Vec<_>>();
                let writer = nightfall_core::descriptors::DescriptorWriter::new()
                    .add_storage_buffer(sets[0], 1, 0, 0, &input.global_histogram_in.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0], 1, 1, 0, &input.histogram_pass_out.as_descriptor_buffer_info());
                writer.write(self.pipeline.device());
            }
            StarlitStrategyInternal::UsesDeviceAddress => {
                return Err(StarlitError::NotDeviceAddressable);
            }
            StarlitStrategyInternal::HybridDeviceAddressDescriptorSets { pool } => {
                return Err(StarlitError::Internal("Not Implemented".into()));
            }
        }
        self.state.input = Some(input.clone());
        Ok(())
    }
    fn input(&self, command_buffer: vk::CommandBuffer) -> Result<(), StarlitError> {
        if let Some(input) = &self.state.input {
            match &self.state.strategy {
                StarlitStrategyInternal::UsesDescriptorSets { sets } => {
                    let dspc = Radix256ScanInputPC {
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
                    };
                    Ok(())
                }
                _ => Err(StarlitError::Internal("Not Implemented".into()))
            }
        } else {
            Err(StarlitError::NoInputRegistered("Radix256Scan does not have any input registered. Consider using `register'".into()))
        }
    }
    fn dispatch(&self, command_buffer: vk::CommandBuffer, dispatch_x: u32, dispatch_y: u32, dispatch_z: u32) {
        self.pipeline.dispatch(command_buffer, dispatch_x, dispatch_y, dispatch_z)
    }
    fn dispatch_indirect(&self, command_buffer: vk::CommandBuffer, indirect: nightfall_core::buffers::BufferOffset) {
        self.pipeline.dispatch_indirect(command_buffer, indirect);
    }
    fn bind(&self, command_buffer: ash::vk::CommandBuffer) {
        self.pipeline.bind(command_buffer);
    }
    fn get_state(&self) -> &StarlitStrategyState<Self::Input> {
        &self.state
    }
}