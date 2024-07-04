use std::sync::Arc;

use affogato::geometry::Cube;
use ash::vk::{self, DescriptorSet};
use nightfall_core::{barriers::{Barriers, BufferMemoryBarrier}, buffers::BufferOffset, descriptors::{DescriptorLayout, DescriptorLayoutBuilder, DescriptorPool, DescriptorType}, device::LogicalDevice, error::NightfallError, image::PipelineStageFlags, memory::{AccessFlags, DevicePointer}, pipeline::{compute::ComputePipeline, layout::{PipelineLayout, PipelineLayoutBuilder}, shader::{ShaderCreateInfo, ShaderStageFlags}}, queue::Queue, NfPtr};

use super::{StarlitShaderExecute, StarlitShaderKernel, StarlitStrategy, StarlitStrategyInternal, StarlitStrategyState, Strategy};
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct AABBMortonCodeCalculatorInputDSPC {
    pub total: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct AABBMortonCodeCalculatorInputDAPC {
    pub whole: DevicePointer,
    /// count of all aabb's
    pub aabb: DevicePointer,
    pub aabb_indices: DevicePointer,
    pub morton_indices: DevicePointer,
    pub total: u32,
}
#[repr(C)]
#[derive(Clone)]
pub struct AABBMortonCodeCalculatorInput {
    /// count of all aabb's
    pub total: u32,
    pub whole: NfPtr,
    pub aabb: NfPtr,
    pub aabb_indices: NfPtr,
    pub morton_indices: NfPtr,
}
/// this algorithm uses [this NVIDIA article](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/)\ to do the morton code calculations.
pub struct AABBMortonCodeCalculator {
    pipeline: Arc<ComputePipeline>,
    layout: Arc<PipelineLayout>,
    state: StarlitStrategyState<AABBMortonCodeCalculatorInput>,
}
// const MORTON_CODE_SHADER_DS: &[u8] = include_bytes!("shaders/morton_ds.comp.spv");
const MORTON_CODE_SHADER_DA: &[u8] = include_bytes!("build/morton_da.comp.spv");
const MORTON_CODE_SHADER_DS_OPT: &[u8] = include_bytes!("build/morton_ds_opt.comp.spv");
const MORTON_CODE_SHADER_DS: &[u8] = include_bytes!("build/morton_ds.comp.spv");
// const MORTON_CODE_SHADER_DA_OPT: &[u8] = include_bytes!("shaders/morton_da_opt.comp.spv");
impl StarlitShaderKernel for AABBMortonCodeCalculator {
    type Input = AABBMortonCodeCalculatorInput;
    fn strategy(device: Arc<LogicalDevice>, strategy: StarlitStrategy) -> Result<Self, crate::error::StarlitError> {
        let (layout, strategy, shader) = match strategy.strategy {
            Strategy::UsesDescriptorSets { pool } => {
                let shader = nightfall_core::pipeline::shader::Shader::new(device.clone(), ShaderCreateInfo {
                    entry: "main\0",
                    // spirv-opt works better with descriptor sets. This is from personal tests though.
                    data: &MORTON_CODE_SHADER_DS,
                    stage: ShaderStageFlags::COMPUTE,
                })?;
                let desc_layout = DescriptorLayoutBuilder::new()
                    // AABB : Input
                    .add_binding(0, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                    // AABB Indices : Input
                    .add_binding(1, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                    // Morton Indices : Output
                    .add_binding(2, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                    // AABB Whole : Input
                    .add_binding(3, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                    .build(device.clone());
                let layout = PipelineLayoutBuilder::new()
                    .add_descriptor_layout(desc_layout.layout())
                    .add_push_constant::<AABBMortonCodeCalculatorInputDSPC>(ShaderStageFlags::COMPUTE)
                    .build(device.clone());
                let sets = pool.allocate(&[desc_layout])?.collect::<_>();
                (layout, StarlitStrategyInternal::UsesDescriptorSets { sets }, shader)
            }
            Strategy::UsesDeviceAddress => {
                let shader = nightfall_core::pipeline::shader::Shader::new(device.clone(), ShaderCreateInfo {
                    entry: "main\0",
                    // spirv-opt does not work well with device addresses apparently. This is from personal tests though.
                    data: &MORTON_CODE_SHADER_DA,
                    stage: ShaderStageFlags::COMPUTE,
                })?;
                let layout = PipelineLayoutBuilder::new()
                .add_push_constant::<AABBMortonCodeCalculatorInputDAPC>(ShaderStageFlags::COMPUTE)
                .build(device.clone());
                (layout, StarlitStrategyInternal::UsesDeviceAddress, shader)
            }
            Strategy::HybridDeviceAddressDescriptorSets { pool } => {
                panic!("Not Implemented")
            }
        };
        let pipeline = Arc::new(ComputePipeline::new(device.clone(), layout.clone(), shader.clone())?);
        Ok(Self { pipeline, layout, state: StarlitStrategyState { strategy, input: None } })
    }
    fn bind(&self, command_buffer: vk::CommandBuffer) {
        self.pipeline.bind(command_buffer)
    }
    fn register(&mut self, input: &Self::Input) -> Result<(), crate::error::StarlitError> {
        match &self.state.strategy {
            StarlitStrategyInternal::UsesDescriptorSets { sets } => {
                let writer = nightfall_core::descriptors::DescriptorWriter::new()
                    .add_storage_buffer(sets[0].set(), 1, 0, 0, &input.aabb.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0].set(), 1, 1, 0, &input.aabb_indices.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0].set(), 1, 2, 0, &input.morton_indices.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0].set(), 1, 3, 0, &input.aabb.as_descriptor_buffer_info());
                writer.write(self.pipeline.device());
            }
            StarlitStrategyInternal::UsesDeviceAddress => {
                
            }
            StarlitStrategyInternal::HybridDeviceAddressDescriptorSets { pool } => {
                return Err(crate::error::StarlitError::Internal("Not Implemented".into()));
            }
        }
        self.state.input = Some(input.clone());
        Ok(())
    }
    fn input(&self, command_buffer: vk::CommandBuffer) -> Result<(), crate::error::StarlitError> {
        if let Some(input) = &self.state.input {
            match &self.state.strategy {
                StarlitStrategyInternal::UsesDescriptorSets { sets } => {
                    let dspc = AABBMortonCodeCalculatorInputDSPC {
                        total: input.total,
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
                    let dapc = AABBMortonCodeCalculatorInputDAPC {
                        whole: input.whole.device_address().unwrap(),
                        total: input.total,
                        aabb: input.aabb.device_address().unwrap(),
                        aabb_indices: input.aabb_indices.device_address().unwrap(),
                        morton_indices: input.morton_indices.device_address().unwrap(),
                    };
                    self.pipeline.device().push_constants(
                        command_buffer, 
                        self.pipeline.layout().get_layout(), 
                        ShaderStageFlags::COMPUTE, 
                        0, 
                        &dapc
                    );
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
        let mut barrier = input.morton_indices.as_buffer_memory_barrier(AccessFlags::SHADER_WRITE, AccessFlags::SHADER_READ|AccessFlags::SHADER_WRITE);
        Some(Barriers::new(PipelineStageFlags::COMPUTE_SHADER, PipelineStageFlags::COMPUTE_SHADER, vec![], vec![], vec![barrier]))
    }
    fn dispatch_indirect_with_barrier(&self, command_buffer: vk::CommandBuffer, indirect: nightfall_core::buffers::BufferOffset) -> Option<Barriers> {
        self.dispatch_indirect(command_buffer, indirect);
        let input = self.state.input.as_ref().expect("no input has been registered");
        let mut barrier = input.morton_indices.as_buffer_memory_barrier(AccessFlags::SHADER_WRITE, AccessFlags::SHADER_READ|AccessFlags::SHADER_WRITE);
        Some(Barriers::new(PipelineStageFlags::COMPUTE_SHADER, PipelineStageFlags::COMPUTE_SHADER, vec![], vec![], vec![barrier]))
    }
    fn get_state(&self) -> &StarlitStrategyState<Self::Input> {
        &self.state
    }
}

impl StarlitShaderExecute for AABBMortonCodeCalculator {
    type Input = <Self as StarlitShaderKernel>::Input;
    fn execute(&mut self, command_buffer: &nightfall_core::commands::CommandPoolAllocation, input: &Self::Input) -> Result<(), crate::error::StarlitError> {
        self.state.input = Some(input.clone());
        match &self.state.strategy {
            StarlitStrategyInternal::UsesDescriptorSets { sets } => {
                let writer = nightfall_core::descriptors::DescriptorWriter::new()
                    .add_storage_buffer(sets[0].set(), 1, 0, 0, &input.aabb.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0].set(), 1, 1, 0, &input.aabb_indices.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0].set(), 1, 2, 0, &input.morton_indices.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0].set(), 1, 3, 0, &input.whole.as_descriptor_buffer_info());
                writer.write(self.pipeline.device());
                let dspc = AABBMortonCodeCalculatorInputDSPC {
                    total: input.total,
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
                self.pipeline.bind(command_buffer.get_command_buffer());
                self.pipeline.dispatch(command_buffer.get_command_buffer(), input.total, 1, 1);
                
            }
            StarlitStrategyInternal::UsesDeviceAddress => {
                let dapc = AABBMortonCodeCalculatorInputDAPC {
                    whole: input.whole.device_address().unwrap(),
                    total: input.total,
                    aabb: input.aabb.device_address().unwrap(),
                    aabb_indices: input.aabb_indices.device_address().unwrap(),
                    morton_indices: input.morton_indices.device_address().unwrap(),
                };
                self.pipeline.device().push_constants(
                    command_buffer.get_command_buffer(), 
                    self.pipeline.layout().get_layout(), 
                    ShaderStageFlags::COMPUTE, 
                    0, 
                    &dapc
                );
                self.pipeline.bind(command_buffer.get_command_buffer());
                self.pipeline.dispatch(command_buffer.get_command_buffer(), input.total/32, 1, 1);
            }
            StarlitStrategyInternal::HybridDeviceAddressDescriptorSets { pool } => {
                return Err(crate::error::StarlitError::Internal("Not Implemented".into()));
            }
        }
        Ok(())
    }
    fn execute_with_barrier(&mut self, command_buffer: &nightfall_core::commands::CommandPoolAllocation, input: &Self::Input) -> Result<Option<Barriers>, crate::error::StarlitError> {
        self.execute(command_buffer, input)?;
        let input = self.state.input.as_ref().expect("no input has been registered");
        let mut barrier = input.morton_indices.as_buffer_memory_barrier(AccessFlags::SHADER_WRITE, AccessFlags::SHADER_READ|AccessFlags::SHADER_WRITE);
        Ok(Some(Barriers::new(PipelineStageFlags::COMPUTE_SHADER, PipelineStageFlags::COMPUTE_SHADER, vec![], vec![], vec![barrier])))
    }
}
pub struct Morton32ToMorton64InputDSPC {
    pub total: u32,
}
pub struct Morton32ToMorton64InputDAPC {
    pub total: u32,
    pub m32: DevicePointer,
    pub idx: DevicePointer,
    pub m64: DevicePointer,
}
#[derive(Clone)]
pub struct Morton32ToMorton64Input {
    pub total: u32,
    pub m32: NfPtr,
    pub idx: NfPtr,
    pub m64: NfPtr,
}
pub struct Morton32ToMorton64 {
    pipeline: Arc<ComputePipeline>,
    layout: Arc<PipelineLayout>,
    state: StarlitStrategyState<Morton32ToMorton64Input>,
}
const M32_TO_M64_DS: &[u8] = include_bytes!("build/m32tom64_ds.comp.spv");
const M32_TO_M64_DA: &[u8] = include_bytes!("build/m32tom64_da.comp.spv");

impl StarlitShaderKernel for Morton32ToMorton64 {
    type Input = Morton32ToMorton64Input;
    fn strategy(device: Arc<LogicalDevice>, strategy: StarlitStrategy) -> Result<Self, crate::error::StarlitError> {
        let (layout, strategy, shader) = match strategy.strategy {
            Strategy::UsesDescriptorSets { pool } => {
                let shader = nightfall_core::pipeline::shader::Shader::new(device.clone(), ShaderCreateInfo {
                    entry: "main\0",
                    // spirv-opt works better with descriptor sets. This is from personal tests though.
                    data: &M32_TO_M64_DS,
                    stage: ShaderStageFlags::COMPUTE,
                })?;
                let desc_layout = DescriptorLayoutBuilder::new()
                    // Morton32 : Input
                    .add_binding(0, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                    // Morton Indices : Input
                    .add_binding(1, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                    // Morton64 : Output
                    .add_binding(2, DescriptorType::STORAGE_BUFFER, 1, ShaderStageFlags::COMPUTE)
                    .build(device.clone());
                let layout = PipelineLayoutBuilder::new()
                    .add_descriptor_layout(desc_layout.layout())
                    .add_push_constant::<Morton32ToMorton64InputDSPC>(ShaderStageFlags::COMPUTE)
                    .build(device.clone());
                let sets = pool.allocate(&[desc_layout])?.collect::<_>();
                (layout, StarlitStrategyInternal::UsesDescriptorSets { sets }, shader)
            }
            Strategy::UsesDeviceAddress => {
                let shader = nightfall_core::pipeline::shader::Shader::new(device.clone(), ShaderCreateInfo {
                    entry: "main\0",
                    // spirv-opt does not work well with device addresses apparently. This is from personal tests though.
                    data: &M32_TO_M64_DA,
                    stage: ShaderStageFlags::COMPUTE,
                })?;
                let layout = PipelineLayoutBuilder::new()
                .add_push_constant::<AABBMortonCodeCalculatorInputDAPC>(ShaderStageFlags::COMPUTE)
                .build(device.clone());
                (layout, StarlitStrategyInternal::UsesDeviceAddress, shader)
            }
            Strategy::HybridDeviceAddressDescriptorSets { pool } => {
                panic!("Not Implemented")
            }
        };
        let pipeline = Arc::new(ComputePipeline::new(device.clone(), layout.clone(), shader.clone())?);
        Ok(Self { pipeline, layout, state: StarlitStrategyState { strategy, input: None } })
    }
    fn bind(&self, command_buffer: vk::CommandBuffer) {
        self.pipeline.bind(command_buffer)
    }
    fn register(&mut self, input: &Self::Input) -> Result<(), crate::error::StarlitError> {
        match &self.state.strategy {
            StarlitStrategyInternal::UsesDescriptorSets { sets } => {
                let writer = nightfall_core::descriptors::DescriptorWriter::new()
                    .add_storage_buffer(sets[0].set(), 1, 0, 0, &input.m32.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0].set(), 1, 1, 0, &input.idx.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0].set(), 1, 2, 0, &input.m64.as_descriptor_buffer_info());
                writer.write(self.pipeline.device());
            }
            StarlitStrategyInternal::UsesDeviceAddress => {
                
            }
            StarlitStrategyInternal::HybridDeviceAddressDescriptorSets { pool } => {

            }
        }
        self.state.input = Some(input.clone());
        Ok(())
    }
    fn input(&self, command_buffer: vk::CommandBuffer) -> Result<(), crate::error::StarlitError> {
        if let Some(input) = &self.state.input {
            match &self.state.strategy {
                StarlitStrategyInternal::UsesDescriptorSets { sets } => {
                    let dspc = Morton32ToMorton64InputDSPC {
                        total: input.total,
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
                    let dapc = Morton32ToMorton64InputDAPC {
                        total: input.total,
                        m32: input.m32.device_address().unwrap(),
                        idx: input.idx.device_address().unwrap(),
                        m64: input.m64.device_address().unwrap(),
                    };
                    self.pipeline.device().push_constants(
                        command_buffer, 
                        self.pipeline.layout().get_layout(), 
                        ShaderStageFlags::COMPUTE, 
                        0, 
                        &dapc
                    );
                }
                StarlitStrategyInternal::HybridDeviceAddressDescriptorSets { pool } => {
                    
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
        let mut barrier = input.m64.as_buffer_memory_barrier(AccessFlags::SHADER_WRITE, AccessFlags::SHADER_READ|AccessFlags::SHADER_WRITE);
        Some(Barriers::new(PipelineStageFlags::COMPUTE_SHADER, PipelineStageFlags::COMPUTE_SHADER, vec![], vec![], vec![barrier]))
    }
    fn dispatch_indirect_with_barrier(&self, command_buffer: vk::CommandBuffer, indirect: nightfall_core::buffers::BufferOffset) -> Option<Barriers> {
        self.dispatch_indirect(command_buffer, indirect);
        let input = self.state.input.as_ref().expect("no input has been registered");
        let mut barrier = input.m64.as_buffer_memory_barrier(AccessFlags::SHADER_WRITE, AccessFlags::SHADER_READ|AccessFlags::SHADER_WRITE);
        Some(Barriers::new(PipelineStageFlags::COMPUTE_SHADER, PipelineStageFlags::COMPUTE_SHADER, vec![], vec![], vec![barrier]))
    }
    fn get_state(&self) -> &super::StarlitStrategyState<Self::Input> {
        &self.state
    }
}

impl StarlitShaderExecute for Morton32ToMorton64 {
    type Input = <Self as StarlitShaderKernel>::Input;
    fn execute(&mut self, command_buffer: &nightfall_core::commands::CommandPoolAllocation, input: &Self::Input) -> Result<(), crate::error::StarlitError> {
        match &self.state.strategy {
            StarlitStrategyInternal::UsesDescriptorSets { sets } => {
                let writer = nightfall_core::descriptors::DescriptorWriter::new()
                    .add_storage_buffer(sets[0].set(), 1, 0, 0, &input.m32.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0].set(), 1, 1, 0, &input.idx.as_descriptor_buffer_info())
                    .add_storage_buffer(sets[0].set(), 1, 2, 0, &input.m64.as_descriptor_buffer_info());
                writer.write(self.pipeline.device());

                let dspc = AABBMortonCodeCalculatorInputDSPC {
                    total: input.total,
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
                self.dispatch(command_buffer.get_command_buffer(), input.total.div_ceil(1024), 1, 1);
            }
            StarlitStrategyInternal::UsesDeviceAddress => {
                panic!("Not Implemented");
            }
            StarlitStrategyInternal::HybridDeviceAddressDescriptorSets { pool } => {
                panic!("Not Implemented");
            }
        }
        self.state.input = Some(input.clone());
        Ok(())
    }
    fn execute_with_barrier(&mut self, command_buffer: &nightfall_core::commands::CommandPoolAllocation, input: &Self::Input) -> Result<Option<Barriers>, crate::error::StarlitError> {
        self.execute(command_buffer, input)?;
        let m64 = input.m64.as_buffer_memory_barrier(AccessFlags::SHADER_WRITE, AccessFlags::SHADER_READ);
        let barriers = Barriers::new(PipelineStageFlags::COMPUTE_SHADER, PipelineStageFlags::COMPUTE_SHADER, vec![], vec![], vec![
            m64
        ]);
        Ok(Some(barriers))
    }
}