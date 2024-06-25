mod boxed;
mod vec;
pub use boxed::*;
pub use vec::*;
use nightfall_core::descriptors::DescriptorBufferInfo;

pub trait GetBufferInfo {
    fn buffer_info(&self, offset: usize, range: usize) -> DescriptorBufferInfo;
    fn full_buffer_info(&self, offset: usize) -> DescriptorBufferInfo;
}