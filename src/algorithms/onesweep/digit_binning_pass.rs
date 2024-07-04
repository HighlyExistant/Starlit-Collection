use std::{cell::Cell, sync::Arc};

use ash::vk;
use nightfall_core::{buffers::BufferOffset, descriptors::{DescriptorBindingFlags, DescriptorLayoutBuilder, DescriptorSetLayoutCreateFlags, DescriptorType}, device::LogicalDevice, pipeline::{compute::ComputePipeline, layout::PipelineLayoutBuilder, shader::{ShaderCreateInfo, ShaderStageFlags}}, NfPtr};

use crate::{algorithms::{StarlitShaderKernel, StarlitShaderKernelConstants, StarlitStrategy, StarlitStrategyInternal, StarlitStrategyState, Strategy}, error::StarlitError};

use super::histogram::Radix256HistogramInputDSPC;
/// Push Constant for Radix256DigitBinningPass for Descriptor Sets
#[derive(Clone)]
struct Radix256DigitBinningPassInputDSPC {
    thread_blocks: u32,
    radix_shift: u32,
    num_keys: u32,
    uses_debug: u32,
}
#[derive(Clone)]
pub struct Radix256DigitBinningPassPushConstant {
    pub thread_blocks: u32,
    pub radix_shift: u32,
    pub num_keys: u32,
}
#[derive(Clone)]
pub struct Radix256DigitBinningPassInput {
    pub thread_blocks: u32,
    pub radix_shift: u32,
    pub num_keys: u32,
    pub partition_tile_indices: NfPtr,
    pub histogram_pass_inout: NfPtr,
    pub sorting_buffer: NfPtr,
    pub alternate_buffer: NfPtr,
    pub debug: Option<NfPtr>,
}
pub struct Radix256DigitBinningPass {
    pipeline: Arc<ComputePipeline>,
    state: StarlitStrategyState<<Self as StarlitShaderKernel>::Input>,
    // contains index to descriptor set that decides which buffer will be written to on this pass
    sort: Cell<usize>,
}
const DIGIT_BINNING_PASS_DS_SG: &[u8] = include_bytes!("build/digit_binning_pass_sg.comp.spv");
const DIGIT_BINNING_PASS_KEYS7_DS_SG: &[u8] = include_bytes!("build/keys7/digit_binning_pass_sg.comp.spv");

impl StarlitShaderKernel for Radix256DigitBinningPass {
    type Input = Radix256DigitBinningPassInput;
    fn strategy(device: Arc<LogicalDevice>, strategy: StarlitStrategy) -> Result<Self, StarlitError> {
        let (layout, strategy, shader) = match strategy.strategy {
            Strategy::UsesDescriptorSets { pool } => {
                if strategy.use_subgroups {
                    let shader = nightfall_core::pipeline::shader::Shader::new(device.clone(), ShaderCreateInfo {
                        entry: "main\0",
                        data: &DIGIT_BINNING_PASS_DS_SG,
                        stage: ShaderStageFlags::COMPUTE,
                    })?;
                    let desc_layout = DescriptorLayoutBuilder::new()
                        .uses_binding_flags()
                        .set_flag(DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                        // Partition Tiles : Inout
                        .add_binding(0, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                        // Histogram Pass : Inout
                        .add_binding(1, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                        // Sorting Buffer : Inout
                        .add_binding(2, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                        // Alternate Buffer : Inout
                        .add_binding(3, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                        // Alternate Debug : Inout
                        .add_binding(4, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                        .build(device.clone());
                        // .set_binding_flag(DescriptorBindingFlags::UPDATE_AFTER_BIND | DescriptorBindingFlags::PARTIALLY_BOUND)
                
                    let layout = PipelineLayoutBuilder::new()
                        .add_descriptor_layout(desc_layout.layout())
                        .add_push_constant::<Radix256DigitBinningPassInputDSPC>(ShaderStageFlags::COMPUTE)
                        .build(device.clone());
                    let sets = pool.allocate(&[desc_layout.clone(), desc_layout.clone()])?.collect::<Vec<_>>();
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
        Ok(Self { pipeline, state: StarlitStrategyState { strategy, input: None }, sort: Cell::new(0) })
    }
    fn register(&mut self, input: &Self::Input) -> Result<(), StarlitError> {
        match &self.state.strategy {
            StarlitStrategyInternal::UsesDescriptorSets { sets } => {
                let mut writer = nightfall_core::descriptors::DescriptorWriter::new()
                    .add_storage_buffer(sets[0].set(), 1, 0, 0, &input.partition_tile_indices.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0].set(), 1, 1, 0, &input.histogram_pass_inout.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0].set(), 1, 2, 0, &input.sorting_buffer.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0].set(), 1, 3, 0, &input.alternate_buffer.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[1].set(), 1, 0, 0, &input.partition_tile_indices.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[1].set(), 1, 1, 0, &input.histogram_pass_inout.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[1].set(), 1, 2, 0, &input.alternate_buffer.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[1].set(), 1, 3, 0, &input.sorting_buffer.as_descriptor_buffer_info());
                if let Some(debug) = &input.debug {
                    writer.push_storage_buffer(sets[0].set(), 1, 4, 0, &debug.as_descriptor_buffer_info());
                    writer.push_storage_buffer(sets[1].set(), 1, 4, 0, &debug.as_descriptor_buffer_info());
                }
                writer.write(self.pipeline.device());
            }
            _ => {
                return Err(StarlitError::Internal("Not Implemented".into()));
            }
        }
        self.state.input = Some(input.clone());
        Ok(())
    }
    fn input(&self, command_buffer: ash::vk::CommandBuffer) -> Result<(), StarlitError> {
        if let Some(input) = &self.state.input {
            match &self.state.strategy {
                StarlitStrategyInternal::UsesDescriptorSets { sets } => {
                    let dspc = Radix256DigitBinningPassInputDSPC {
                        thread_blocks: input.thread_blocks,
                        radix_shift: input.radix_shift,
                        num_keys: input.num_keys,
                        uses_debug: if input.debug.is_some() { 1 } else { 0 }
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
                            &[sets[self.sort.get()].set()], 
                            &[]
                        ) 
                    }
                    Ok(())
                }
                StarlitStrategyInternal::UsesDeviceAddress => {
                    return Err(StarlitError::NotDeviceAddressable);
                }
                StarlitStrategyInternal::HybridDeviceAddressDescriptorSets { pool } => {
                    return Err(StarlitError::Internal("Not Implemented".into()));
                }
            }
        } else {
            Err(StarlitError::NoInputRegistered("Radix256DigitBinningPass does not have any input registered. Consider using `register'".into()))
        }
    }
    fn bind(&self, command_buffer: vk::CommandBuffer) {
        self.pipeline.bind(command_buffer)
    }
    fn dispatch(&self, command_buffer: vk::CommandBuffer, dispatch_x: u32, dispatch_y: u32, dispatch_z: u32) {
        self.sort.set((self.sort.get()+1)&1);
        self.pipeline.dispatch(command_buffer, dispatch_x, 1, 1);
    }
    fn dispatch_indirect(&self, command_buffer: vk::CommandBuffer, indirect: BufferOffset) {
        self.sort.set((self.sort.get()+1)&1);
        self.pipeline.dispatch_indirect(command_buffer, indirect)
    }
    fn get_state(&self) -> &StarlitStrategyState<Self::Input> {
        &self.state
    }
}

impl StarlitShaderKernelConstants for Radix256DigitBinningPass {
    type Constant = Radix256DigitBinningPassPushConstant;
    fn register_constants(&mut self, constant: &Self::Constant) {
        if let Some(input) = &mut self.state.input {
            input.num_keys = constant.num_keys;
            input.radix_shift = constant.radix_shift;
            input.thread_blocks = constant.thread_blocks;
        } else {
            panic!("Input has not previously been registered. To register only constants, you must have registered all before");
        }
    }
}