use crate::alloc::StarlitAllocError;
use nightfall_core::error::VulkanError;
use thiserror::Error;

#[derive(Error, Debug, PartialEq, PartialOrd)]
pub enum StarlitError {
    #[error("{0}")]
    Internal(String),
    #[error("{0}")]
    AllocError(StarlitAllocError),
    #[error("{0}")]
    VulkanCoreError(VulkanError),
    #[error("Attempted to cast allocation to `DevicePointer' but it is not marked as device addressable")]
    NotDeviceAddressable,
    #[error("Attempted to use memory on the host but it is not marked as host visible.")]
    NotHostMappable,
    #[error("The amount of descriptor sets necessary for this operation is {1} but {0} were provided")]
    NotEnoughDescriptorSets(usize, usize),
    #[error("{0}")]
    NoInputRegistered(String),
}

impl From<StarlitAllocError> for StarlitError {
    fn from(value: StarlitAllocError) -> Self {
        StarlitError::AllocError(value)
    }
}
impl From<VulkanError> for StarlitError {
    fn from(value: VulkanError) -> Self {
        StarlitError::VulkanCoreError(value)
    }
}