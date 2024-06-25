use std::sync::Arc;

use nightfall_core::device::LogicalDevice;
#[derive(Clone, Copy, Debug)]
pub struct TuningParameters {
    pub keys_per_thread: u32,
    pub threads_per_block: u32,
    pub partition_size: u32,
    pub total_shared_memory: u32,
}

impl TuningParameters {
    pub fn new(device: Arc<LogicalDevice>) -> Self {
        return Self {
            keys_per_thread: 15,
            partition_size: 7680,
            threads_per_block: 512,
            total_shared_memory: 7936
        };
        let subgroup_properties = device.physical_device().query_subgroup_properties();
        let (keys_per_thread, threads_per_block) = match device.physical_device().properties().limits.max_compute_shared_memory_size {
            16384 => (7, 512),
            49152 => (7, 512),
            65536 => (7, 256),
            131072 => (15, 256),
            _ => return Self::default(),
        };
        let partition_size = keys_per_thread * threads_per_block;
		let histogram_shared_memory = threads_per_block / subgroup_properties.subgroup_size * 256;
        let combined_part_size = partition_size + 256;
        let total_shared_memory = if histogram_shared_memory > combined_part_size {
            histogram_shared_memory
        } else {
            combined_part_size
        };
        Self { 
            keys_per_thread, 
            threads_per_block, 
            partition_size, 
            total_shared_memory 
        }
    }
}

impl Default for TuningParameters {
    fn default() -> Self {
        Self { 
            keys_per_thread: 15, 
            threads_per_block: 256, 
            partition_size: 15*256, 
            total_shared_memory: 15*256+256 
        }
    }
}