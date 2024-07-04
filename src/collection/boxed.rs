use std::{alloc::Layout, marker::PhantomData, os::raw::c_void, ptr::NonNull, sync::Arc};

use nightfall_core::{device::LogicalDevice, memory::DevicePointer, AsNfptr, NfPtr};
use starlit_alloc::{GeneralAllocator, HostDeviceConversions};
pub struct SlBox<T: ?Sized, A: GeneralAllocator + ?Sized>(NfPtr, NonNull<T>, Arc<A>);

impl<T, A: GeneralAllocator + ?Sized> SlBox<T, A> {
    // if the allocator does not support host allocations this function will return None
    pub fn new(x: T, alloc: Arc<A>) -> Option<Self> {
        if !alloc.is_host_mappable() { return None };

        let allocation = alloc.allocate(Layout::new::<T>()).ok()?;
        let host_ptr = alloc.as_host_mut_ptr(allocation.clone()).unwrap().cast::<T>();
        unsafe { host_ptr.copy_from_nonoverlapping(&x, std::mem::size_of::<T>()) }
        let nfptr = NfPtr::new(allocation.id(), allocation.offset(), allocation.device_address(), allocation.size());
        let hptr = NonNull::new(host_ptr)?;
        
        Some(SlBox(nfptr,hptr,alloc))
    }
    pub fn new_zeroed(alloc: Arc<A>) -> Option<Self> {
        if alloc.is_host_mappable() {
            let allocation = alloc.allocate(Layout::new::<T>()).ok()?;
            let host_ptr = alloc.as_host_mut_ptr(allocation.clone()).unwrap().cast::<T>();
            
            Some(SlBox(NfPtr::new(allocation.id(), allocation.offset(), allocation.device_address(), allocation.size()), NonNull::new(host_ptr)?, alloc.clone()))
        } else {
            let allocation = alloc.allocate(Layout::new::<T>()).ok()?;
            Some(SlBox(NfPtr::new(allocation.id(), allocation.offset(), allocation.device_address(), allocation.size()), NonNull::dangling(), alloc.clone()))
        }
        // Some(SlBox(x.0,x.1,alloc))
    }
    pub fn get_allocation(&self) -> NfPtr {
        self.0.clone()
    }
    pub fn as_host_ptr(&self) -> Option<*const c_void> {
        self.2.as_host_ptr(self.0.clone())
    }
    pub fn as_host_mut_ptr(&self) -> Option<*mut c_void> {
        self.2.as_host_mut_ptr(self.0.clone())
    }
    pub fn as_device_ptr(&self) -> Option<DevicePointer> {
        self.2.as_device_ptr(self.0.clone())
    }
    #[inline]
    pub fn get_allocator(&self) -> Arc<A> {
        self.2.clone()
    }
    pub fn device(&self) -> Arc<LogicalDevice> {
        self.2.get_device()
    }
}
impl<T: ?Sized, A: GeneralAllocator + ?Sized> std::ops::Deref for SlBox<T, A>  {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        unsafe { self.1.as_ref() }
    }
}
impl<T: ?Sized, A: GeneralAllocator + ?Sized> std::ops::DerefMut for SlBox<T, A>  {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.1.as_mut() }
    }
}

impl<T: ?Sized, A: GeneralAllocator + ?Sized> Drop for SlBox<T, A> {
    fn drop(&mut self) {
        self.2.deallocate(self.0.clone(), Layout::from_size_align(self.0.size(), 8).unwrap()).unwrap();
    }
}
unsafe impl<T, A: GeneralAllocator + ?Sized> AsNfptr for SlBox<T, A> {
    unsafe fn as_nfptr(&self) -> NfPtr {
        self.0.clone()
    }
}