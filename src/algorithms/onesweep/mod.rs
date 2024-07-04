use std::{rc::Rc, sync::Arc};

use nightfall_core::{commands::CommandPoolAllocation, device::LogicalDevice, image::PipelineStageFlags, memory::{AccessFlags, DependencyFlags}, NfPtr};

use crate::{algorithms::onesweep::digit_binning_pass::Radix256DigitBinningPassPushConstant, alloc::{FreeListAllocator, GeneralAllocator}, error::StarlitError, SlVec};

use self::{digit_binning_pass::{Radix256DigitBinningPass, Radix256DigitBinningPassInput}, histogram::{Radix256Histogram, Radix256HistogramInput}, reset::{Radix256Reset, Radix256ResetInput}, scan::{Radix256Scan, Radix256ScanInput}, tuning::TuningParameters};

use super::{StarlitShaderAlgorithm, StarlitShaderKernel, StarlitShaderKernelConstants, StarlitStrategy, StarlitStrategyInternal, StarlitStrategyState, Strategy};

pub mod reset;
pub mod histogram;
pub mod scan;
pub mod tuning;
pub mod digit_binning_pass;
#[derive(Clone)]
pub struct OneSweepU32Input {
    pub sort: NfPtr,
    pub num_keys: usize,
}
pub struct OneSweepState<A: GeneralAllocator> {
    // histogram_input: <Radix256Histogram as StarlitShaderKernel>::Input,
    // scan_input: <Radix256Scan as StarlitShaderKernel>::Input,
    // digit_pass_input: <Radix256DigitBinningPass as StarlitShaderKernel>::Input,
    histogram: SlVec<u32, A>,
    pub histogram_pass: SlVec<u32, A>,
    pub sort_alt_buffer: SlVec<u32, A>,
    global_partition_tiles: SlVec<u32, A>,
    /// used when theres not enough shared memory for an operation
    global_shared_memory: SlVec<u32, A>,
    pub debug: Option<SlVec<u32, A>>,
    sort_buffer: NfPtr,
    num_keys: usize,
    thread_blocks: usize,
    global_histogram_partitions: usize,
}
pub struct OneSweepU32<A: GeneralAllocator> {
    device: Arc<LogicalDevice>,
    freelist: Arc<A>,
    reset: Radix256Reset,
    histogram: Radix256Histogram,
    scan: Radix256Scan,
    digit_pass: Radix256DigitBinningPass,
    tuning: TuningParameters,
    /// if this is Some, the vector will not be deallocated after
    /// use, and will be reused on every use, only being reallocated when
    /// the array is larger. The value inside is the current num_keys at 0 and threadblocks at 1
    pub input: Option<OneSweepState<A>>,
}

impl<A: GeneralAllocator> OneSweepU32<A> {
    const PART_SIZE: usize = 7680; // 3840
    fn initialize_static_buffers(freelist: Arc<A>, input: &mut OneSweepState<A>, thread_blocks: usize) {
        input.histogram = SlVec::<u32, A>::new_zeroed(1024, freelist.clone()).unwrap();
        input.global_partition_tiles = SlVec::<u32, A>::new_zeroed(4 as usize, freelist.clone()).unwrap();
    }
    fn initialize_buffers(freelist: Arc<A>, input: &mut OneSweepState<A>, thread_blocks: usize, num_keys: usize) {
        input.sort_alt_buffer = SlVec::<u32, A>::new_zeroed(num_keys, freelist.clone()).unwrap();
        input.histogram_pass = SlVec::<u32, A>::new_zeroed(1024*thread_blocks, freelist.clone()).unwrap();
    }
}

impl<A: GeneralAllocator> StarlitShaderAlgorithm<A> for OneSweepU32<A> {
    type Input = OneSweepU32Input;
    fn strategy(device: Arc<LogicalDevice>, freelist: Arc<A>, strategy: StarlitStrategy) -> Result<Self, StarlitError> {
        let (reset, histogram, scan, digit_pass, device) = match &strategy.strategy {
            Strategy::UsesDescriptorSets { pool } => {
                if strategy.use_subgroups {
                    let reset = Radix256Reset::strategy(
                        device.clone(), 
                        strategy.clone()
                    )?;
                    let histogram = Radix256Histogram::strategy(
                        device.clone(), 
                        strategy.clone()
                    )?;
                    let scan = Radix256Scan::strategy(
                        device.clone(), 
                        strategy.clone()
                    )?;
                    let digit_pass = Radix256DigitBinningPass::strategy(
                        device.clone(), 
                        strategy.clone()
                    )?;

                    (reset, histogram, scan, digit_pass, device)
                } else {
                    return Err(StarlitError::Internal("Not Implemented".into()));
                }
            }
            _ => {
                return Err(StarlitError::Internal("Not Implemented".into()));
            }
        };
        let tuning = TuningParameters::new(device.clone());
        println!("{:#?}", tuning);
        Ok(Self { device, freelist, reset, histogram, scan, digit_pass, input: None, tuning })
    }
    fn register(&mut self, input: &Self::Input) -> Result<(), StarlitError> {
        // Check to see if the cache has num keys
        if let Some(prev_input) = &mut self.input {
            let thread_blocks = input.num_keys.div_ceil(self.tuning.partition_size as usize) as u32; // OneSweepU32::<A>::PART_SIZE
            prev_input.global_histogram_partitions = input.num_keys.div_ceil(32768); // 32768
            if prev_input.num_keys < input.num_keys {
                prev_input.sort_alt_buffer = SlVec::<u32, A>::new_zeroed(input.num_keys, self.freelist.clone()).unwrap();
            }
            if prev_input.thread_blocks < thread_blocks as usize {
                prev_input.histogram_pass = SlVec::<u32, A>::new_zeroed(1024*thread_blocks as usize, self.freelist.clone()).unwrap();
            }
            prev_input.num_keys = input.num_keys;
            prev_input.thread_blocks = thread_blocks as usize;
            prev_input.sort_buffer = input.sort.clone();
        } else { // if the cache has not been used it's the first time and all must be allocated
            let thread_blocks = input.num_keys.div_ceil(self.tuning.partition_size as usize) as u32; // OneSweepU32::<A>::PART_SIZE
            let global_histogram_partitions = input.num_keys.div_ceil(32768);
            let debug = SlVec::new_zeroed(1024 as usize, self.freelist.clone()).unwrap();
            let mut state = OneSweepState {
                histogram: SlVec::new(self.freelist.clone()),
                sort_alt_buffer: SlVec::new(self.freelist.clone()),
                histogram_pass: SlVec::new(self.freelist.clone()),
                global_partition_tiles: SlVec::new(self.freelist.clone()),
                global_shared_memory: SlVec::new(self.freelist.clone()),
                num_keys: input.num_keys,
                thread_blocks: thread_blocks as usize,
                sort_buffer: input.sort.clone(),
                global_histogram_partitions,
                debug: Some(debug),
            };
            Self::initialize_buffers(self.freelist.clone(), &mut state, thread_blocks as usize, input.num_keys as usize);
            Self::initialize_static_buffers(self.freelist.clone(), &mut state, thread_blocks as usize);
            self.input = Some(state);
        }
        Ok(())
    }
    fn fill_command_listing(&mut self, command_buffer: &CommandPoolAllocation) -> Result<(), StarlitError> {
        if let Some(input) = &self.input {
            println!("input. {:#?}", input.thread_blocks);
            self.reset.bind(command_buffer.get_command_buffer());
            self.reset.register(&Radix256ResetInput {
                thread_blocks: input.thread_blocks as u32,
                histogram_out: input.histogram.get_allocation(),
                histogram_pass_out: input.histogram_pass.get_allocation(),
                partition_tile_indices_out: input.global_partition_tiles.get_allocation(),
            }).unwrap();
            self.reset.input(command_buffer.get_command_buffer()).unwrap();
            self.reset.dispatch(command_buffer.get_command_buffer(), 256, 1, 1);

            self.histogram.bind(command_buffer.get_command_buffer());
            self.histogram.register(&Radix256HistogramInput {
                num_keys: input.num_keys as u32,
                thread_blocks: input.thread_blocks as u32,
                u32_in: input.sort_buffer.clone(),
                histogram_out: input.histogram.get_allocation(),
            }).unwrap();
            self.histogram.input(command_buffer.get_command_buffer()).unwrap();
            command_buffer.pipeline_barrier(
                PipelineStageFlags::COMPUTE_SHADER, 
                PipelineStageFlags::COMPUTE_SHADER,
                DependencyFlags::empty(), 
                &[], 
                &[input.histogram.get_allocation().as_buffer_memory_barrier(AccessFlags::SHADER_WRITE, AccessFlags::SHADER_READ)], 
                &[]
            );
            self.histogram.dispatch(command_buffer.get_command_buffer(), input.global_histogram_partitions as u32, 1, 1);

            self.scan.bind(command_buffer.get_command_buffer());
            self.scan.register(&Radix256ScanInput {
                thread_blocks: input.thread_blocks as u32,
                global_histogram_in: input.histogram.get_allocation(),
                histogram_pass_out: input.histogram_pass.get_allocation(),
            }).unwrap();
            self.scan.input(command_buffer.get_command_buffer()).unwrap();
            command_buffer.pipeline_barrier(
                PipelineStageFlags::COMPUTE_SHADER, 
                PipelineStageFlags::COMPUTE_SHADER,
                DependencyFlags::empty(), 
                &[], 
                &[input.histogram.get_allocation().as_buffer_memory_barrier(AccessFlags::SHADER_WRITE, AccessFlags::SHADER_READ)], 
                &[]
            );
            self.scan.dispatch(command_buffer.get_command_buffer(), 4, 1, 1);
            // return;
            command_buffer.pipeline_barrier(
                PipelineStageFlags::COMPUTE_SHADER, 
                PipelineStageFlags::COMPUTE_SHADER,
                DependencyFlags::empty(), 
                &[], 
                &[input.histogram_pass.get_allocation().as_buffer_memory_barrier(AccessFlags::SHADER_WRITE, AccessFlags::SHADER_WRITE)], 
                &[]
            );
            let debug = if let Some(debug) = &input.debug {
                Some(debug.get_allocation())
            } else {
                None
            };
            self.digit_pass.register(&Radix256DigitBinningPassInput {
                radix_shift: 0,
                thread_blocks: input.thread_blocks as u32,
                num_keys: input.num_keys as u32,
                sorting_buffer: input.sort_buffer.clone(),
                histogram_pass_inout: input.histogram_pass.get_allocation(),
                partition_tile_indices: input.global_partition_tiles.get_allocation(),
                alternate_buffer: input.sort_alt_buffer.get_allocation(),
                debug
            }).unwrap();
            self.digit_pass.bind(command_buffer.get_command_buffer());
            for i in 0..4 {
                self.digit_pass.register_constants(&Radix256DigitBinningPassPushConstant {
                    num_keys: input.num_keys as u32,
                    thread_blocks: input.thread_blocks as u32,
                    radix_shift: i * 8,
                });
                self.digit_pass.input(command_buffer.get_command_buffer()).unwrap();
                self.digit_pass.dispatch(command_buffer.get_command_buffer(), input.thread_blocks as u32, 1, 1);
                command_buffer.pipeline_barrier(
                    PipelineStageFlags::COMPUTE_SHADER, 
                    PipelineStageFlags::COMPUTE_SHADER,
                    DependencyFlags::empty(), 
                    &[], 
                    &[
                        input.histogram_pass.get_allocation().as_buffer_memory_barrier(AccessFlags::SHADER_WRITE, AccessFlags::SHADER_WRITE),
                        input.sort_alt_buffer.get_allocation().as_buffer_memory_barrier(AccessFlags::SHADER_WRITE, AccessFlags::SHADER_WRITE),
                        input.sort_buffer.as_buffer_memory_barrier(AccessFlags::SHADER_WRITE, AccessFlags::SHADER_WRITE),
                        ], 
                    &[]
                );
            }
        }
        Ok(())
    }
}

