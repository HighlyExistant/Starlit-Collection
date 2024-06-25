use std::sync::Arc;

use ash::vk;
use nightfall_core::{buffers::BufferOffset, descriptors::{DescriptorLayoutBuilder, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateFlags, DescriptorType}, device::LogicalDevice, error::NightfallError, pipeline::{compute::ComputePipeline, layout::PipelineLayoutBuilder, shader::{ShaderCreateInfo, ShaderStageFlags}}, NfPtr};
use super::{StarlitStrategy, StarlitStrategyInternal, Strategy};

#[repr(C)]
#[derive(Clone, Copy)]
pub struct InclusivePrefixSumCalculatorInput {
    pub monoid_in: NfPtr,
    pub monoid_out: NfPtr,
    /// should point to PrefixSumCalculatorInputStateBuffer
    pub state: NfPtr,
    pub debug: NfPtr,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct InclusivePrefixSumCalculatorInputState {
    pub flag: u32,
    pub aggregate: u32,
    pub prefix: u32,
}
#[repr(C)]
pub struct InclusivePrefixSumCalculatorInputStateBuffer {
    pub part_counter: u32,
    pub size: u32,
    pub state: [InclusivePrefixSumCalculatorInputState; 0],
}
pub struct InclusivePrefixSumCalculator {
    pipeline: Arc<ComputePipeline>,
    strategy: StarlitStrategyInternal,
}
const PREFIX_SUM_SHADER_DS: &[u8] = include_bytes!("shaders/prefix_sum.comp.spv");

impl InclusivePrefixSumCalculator {
    pub fn new(device: Arc<LogicalDevice>, strategy: StarlitStrategy) -> Result<Self, NightfallError> {
        let (layout, strategy, shader) = match strategy.strategy {
            Strategy::UsesDescriptorSets { pool } => {
                let shader = nightfall_core::pipeline::shader::Shader::new(device.clone(), ShaderCreateInfo {
                    entry: "main\0",
                    // spirv-opt works better with descriptor sets. This is from personal tests though.
                    data: &PREFIX_SUM_SHADER_DS,
                    stage: ShaderStageFlags::COMPUTE,
                })?;
                let desc_layout = DescriptorLayoutBuilder::fill(device.clone(), &[
                    // Monoid : Input
                    DescriptorSetLayoutBinding { binding: 0, descriptor_count: 1, descriptor_type: DescriptorType::STORAGE_BUFFER, immutable_samplers: None, stage_flags: ShaderStageFlags::COMPUTE },
                    // Monoid : Output
                    DescriptorSetLayoutBinding { binding: 1, descriptor_count: 1, descriptor_type: DescriptorType::STORAGE_BUFFER, immutable_samplers: None, stage_flags: ShaderStageFlags::COMPUTE },
                    // State
                    DescriptorSetLayoutBinding { binding: 2, descriptor_count: 1, descriptor_type: DescriptorType::STORAGE_BUFFER, immutable_samplers: None, stage_flags: ShaderStageFlags::COMPUTE },
                    // Debug
                    DescriptorSetLayoutBinding { binding: 3, descriptor_count: 1, descriptor_type: DescriptorType::STORAGE_BUFFER, immutable_samplers: None, stage_flags: ShaderStageFlags::COMPUTE },
                ], &[], DescriptorSetLayoutCreateFlags::empty());
                // let desc_layout = DescriptorLayoutBuilder::new()
                //     // Monoid : Input
                //     .add_binding(0, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                //     // Monoid : Output
                //     .add_binding(1, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                //     // State
                //     .add_binding(2, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                //     // Debug
                //     .add_binding(3, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                //     .build(device.clone());
                let layout = PipelineLayoutBuilder::new()
                    .add_descriptor_layout(desc_layout.layout())
                    .build(device.clone());
                let sets = pool.allocate(&[desc_layout])?.collect::<_>();
                (layout, StarlitStrategyInternal::UsesDescriptorSets { sets }, shader)
            }
            Strategy::UsesDeviceAddress => {
                panic!("Not Implemented")
            }
            Strategy::HybridDeviceAddressDescriptorSets { pool } => {
                panic!("Not Implemented")
            }
        };
        let pipeline = Arc::new(ComputePipeline::new(device.clone(), layout.clone(), shader.clone())?);
        Ok(Self { pipeline, strategy })
    }
    pub fn bind(&self, command_buffer: vk::CommandBuffer) {
        self.pipeline.bind(command_buffer)
    }
    pub fn inputs(&self, command_buffer: vk::CommandBuffer, constant: &InclusivePrefixSumCalculatorInput) {
        match &self.strategy {
            StarlitStrategyInternal::UsesDescriptorSets { sets } => {
                let sets = sets.iter().map(|set|{set.set()}).collect::<Vec<_>>();
                let writer = nightfall_core::descriptors::DescriptorWriter::new()
                    .add_storage_buffer(sets[0], 1, 0, 0, &constant.monoid_in.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0], 1, 1, 0, &constant.monoid_out.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0], 1, 2, 0, &constant.state.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0], 1, 3, 0, &constant.debug.as_descriptor_buffer_info());
                writer.write(self.pipeline.device());
                unsafe { 
                    self.pipeline.device().device().cmd_bind_descriptor_sets(
                        command_buffer, 
                        vk::PipelineBindPoint::COMPUTE, 
                        self.pipeline.layout().get_layout(), 
                        0, 
                        &sets, 
                        &[]
                    ) 
                }
            }
            StarlitStrategyInternal::UsesDeviceAddress => {
                panic!("Not Implemented")
            }
            StarlitStrategyInternal::HybridDeviceAddressDescriptorSets { pool } => {
                panic!("Not Implemented")
            }
        }
    }
    pub fn dispatch(&self, command_buffer: vk::CommandBuffer, dispatch_x: u32, dispatch_y: u32, dispatch_z: u32) {
        self.pipeline.dispatch(command_buffer, dispatch_x, 1, 1);
    }
    pub fn dispatch_indirect(&self, command_buffer: vk::CommandBuffer, indirect: BufferOffset) {
        self.pipeline.dispatch_indirect(command_buffer, indirect)
    }
}
