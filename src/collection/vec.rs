use std::{alloc::Layout, marker::PhantomData, ops::IndexMut, os::raw::c_void, ptr::NonNull, sync::{Arc, Mutex}};

use nightfall_core::{barriers::Barriers, buffers::Buffer, commands::{BufferCopy, CommandBufferBeginInfo, CommandPoolAllocation}, descriptors::DescriptorBufferInfo, device::LogicalDevice, image::PipelineStageFlags, memory::{AccessFlags, DevicePointer}, queue::{Queue, Submission}, sync::Fence, NfPtr};
use crate::{alloc::{GeneralAllocator, HostDeviceConversions}, error::StarlitError, GetBufferInfo};

#[derive(Debug)]
struct VkRawVec<T, A: GeneralAllocator + ?Sized> {
    // Only 
    pointer: Option<NonNull<T>>,
    allocation: NfPtr,
    capacity: usize,
    alloc: Arc<A>,
}

impl<T, A: GeneralAllocator + ?Sized> VkRawVec<T, A> {
    pub(crate) const MIN_NON_ZERO_CAP: usize = if std::mem::size_of::<T>() == 1 {
        8
    } else if std::mem::size_of::<T>() <= 1024 {
        4
    } else {
        1
    };
    fn needs_to_grow(&self, len: usize, additional: usize) -> bool {
        additional > self.capacity().wrapping_sub(len)
    }
    pub const fn new(alloc: Arc<A>) -> Self {
        Self { pointer: None, capacity: 0, alloc, allocation: NfPtr::new(0, 0, None, 0) }
    }
    pub fn with_capacity(capacity: usize, alloc: Arc<A>) -> Result<Self, StarlitError> {
        let layout = match Layout::array::<T>(capacity) {
            Ok(layout) => layout,
            Err(_) => panic!("capacity overflow"),
        };
        // let pointer = alloc.allocate allocate(layout).unwrap().cast::<T>();
        let allocation = alloc.allocate(layout)?;
        let pointer = alloc.as_host_ptr(allocation).map(|val|{NonNull::new(val as *mut _).unwrap()});
        Ok(Self { pointer, capacity, alloc, allocation,  })
    }
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn reserve(&mut self, len: usize, additional: usize) -> Result<(), StarlitError> {
        #[cold]
        fn do_reserve_and_handle<T, A: GeneralAllocator + ?Sized>(
            slf: &mut VkRawVec<T, A>,
            len: usize,
            additional: usize,
        ) -> Result<(), StarlitError> {
            slf.grow_amortized(len, additional)
        }

        if self.needs_to_grow(len, additional) {
            do_reserve_and_handle(self, len, additional)?;
        }
        Ok(())
    }
    fn grow_amortized(&mut self, len: usize, additional: usize) -> Result<(), StarlitError>{
        let required_cap = len.checked_add(additional).unwrap();
        let cap = std::cmp::max(self.capacity * 2, required_cap);
        let cap = std::cmp::max(Self::MIN_NON_ZERO_CAP, cap);
        let new_layout = Layout::array::<T>(cap).unwrap();
        let current_memory = self.current_memory();
        let memory = if let Some((ptr, old_layout)) = current_memory {
            self.alloc.reallocate(ptr, Layout::from_size_align(self.allocation.size(), std::mem::align_of::<T>()).unwrap(), new_layout)
        } else {
            self.alloc.allocate(new_layout)
        }?;
        self.capacity = cap;
        self.allocation = memory;
        self.pointer = self.alloc.as_host_mut_ptr(self.allocation).map(|val|{ NonNull::new(val).unwrap().cast() });
        Ok(())
    }
    pub fn reserve_for_push(&mut self, len: usize) {
        self.grow_amortized(len, 1).unwrap();
    }
    pub fn get_allocation(&self) -> NfPtr {
        self.allocation
    }
    #[inline]
    pub fn ptr(&self) -> Option<*mut T> {
        Some(self.pointer?.as_ptr())
    }
    fn current_memory(&self) -> Option<(NfPtr, Layout)> {
        if self.capacity() == 0 {
            None
        } else {
            unsafe {
                let align = std::mem::align_of::<T>();
                let size = std::mem::size_of::<T>().wrapping_mul(self.capacity());
                let layout = Layout::from_size_align_unchecked(size, align);
                Some((self.allocation, layout))
            }
        }
    }
}
#[derive(Debug)]
pub struct VkVec<T, A: GeneralAllocator + ?Sized> {
    buf: VkRawVec<T, A>,
    len: usize
}

impl<T, A: GeneralAllocator + ?Sized> VkVec<T, A> {
    pub fn new(alloc: Arc<A>) -> Self {
        VkVec { buf: VkRawVec::new(alloc.clone()), len: 0 }
    }
    pub fn new_zeroed(len: usize, alloc: Arc<A>) -> Result<Self, StarlitError> {
        let mut this = Self::with_capacity(len, alloc)?;
        unsafe { this.set_len(len) };
        Ok(this)
    }
    pub fn with_capacity(capacity: usize, alloc: Arc<A>) -> Result<Self, StarlitError> {
        Ok(VkVec { buf: VkRawVec::with_capacity(capacity, alloc)?, len: 0 })
    }
    pub fn from_elem(elem: T, n: usize, alloc: Arc<A>) -> Result<Self, StarlitError> 
        where T: Clone {
        let mut v = VkVec::with_capacity(n, alloc)?;
        v.extend_with(n, elem);
        Ok(v)
    }
    pub fn from_iter<TT: ExactSizeIterator<Item = T>>(iter: TT, alloc: Arc<A>) -> Result<Self, StarlitError> {
        if !alloc.is_host_mappable() {
            return Err(StarlitError::NotHostMappable);
        }

        let mut this = Self::new_zeroed(iter.len(), alloc)?;
        for (i, val) in iter.enumerate() {
            this[i] = val;
        }
        Ok(this)
    }
    #[inline]
    pub fn as_mut_ptr(&mut self) -> Option<*mut T> {
        self.buf.ptr()
    }
    #[inline]
    pub fn as_ptr(&self) -> Option<*const T> {
        self.buf.ptr().map(|val|{val as *const _})
    }
    #[inline]
    pub fn device_ptr(&self) -> Result<DevicePointer, StarlitError> {
        self.buf.alloc.as_device_ptr(self.buf.allocation).ok_or(StarlitError::NotDeviceAddressable)
    }
    pub fn push(&mut self, value: T) {
        if self.len == self.buf.capacity() {
            self.buf.reserve_for_push(self.len);
        }
        unsafe {
            let end = self.as_mut_ptr().unwrap().add(self.len);
            std::ptr::copy_nonoverlapping(&value as *const T, end.cast(), 1);
            // std::ptr::write(end, value);
            self.len += 1;
        }
    }
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            unsafe {
                self.len -= 1;
                Some(std::ptr::read(self.as_ptr().unwrap().add(self.len())))
            }
        }
    }
    #[inline(always)]
    pub fn len(&self) -> usize {
       self.len
    }
    #[inline(always)]
    pub fn capacity(&self) -> usize {
       self.buf.capacity()
    }
    pub fn iter(&self) -> VkBufferIterator<T> {
        VkBufferIterator {
            start: self.buf.ptr().unwrap(),
            end: unsafe { self.buf.ptr().unwrap().add(self.len()) },
            phantom_: PhantomData::default()
        }
    }
    pub fn iter_mut(&mut self) -> VkBufferIteratorMut<T> {
        VkBufferIteratorMut {
            start: self.buf.ptr().unwrap(),
            end: unsafe { self.buf.ptr().unwrap().add(self.len()) },
            phantom_: PhantomData::default(),
        }
    }
    pub fn reserve(&mut self, additional: usize) -> Result<(), StarlitError> {
        self.buf.reserve(self.len, additional)
    }
    fn extend_with(&mut self, n: usize, value: T) 
        where T: Clone {
        self.reserve(n).unwrap();

        unsafe {
            let mut ptr = self.as_mut_ptr().unwrap().add(self.len());
            let mut local_len = self.len;
            // Write all elements except the last one
            for _ in 1..n {
                std::ptr::write(ptr, value.clone());
                ptr = ptr.add(1);
                // Increment the length in every step in case clone() panics
                local_len += 1;
            }

            if n > 0 {
                // We can write the last element directly without cloning needlessly
                std::ptr::write(ptr, value);
                local_len += 1;
            }
            self.len = local_len;
        }
    }
    pub unsafe fn set_len(&mut self, len: usize) {
        self.len = len;
    }
    pub fn get_allocation(&self) -> NfPtr {
        self.buf.get_allocation()
    }
    pub fn as_device_ptr(&self) -> Option<DevicePointer> {
        self.buf.get_allocation().device_address()
    }
    pub fn copy_with_fence<O: GeneralAllocator>(&mut self, other: &VkVec<T, O>, length: usize, queue: Arc<Queue>, command_buffer: &CommandPoolAllocation) -> Result<Arc<Fence>, StarlitError> {
        self.copy_allocation_with_fence(other.get_allocation(), length*std::mem::size_of::<T>(), queue.clone(), command_buffer)
    }
    pub fn record_copy_allocation_with_barrier(&mut self, device: Arc<LogicalDevice>, allocation: NfPtr, size: usize, dst: PipelineStageFlags, command_buffer: &CommandPoolAllocation) -> Result<Barriers, StarlitError> {
        // command_buffer.begin(CommandBufferBeginInfo::SINGLE_SUBMIT).unwrap();
        command_buffer.copy_buffer(allocation.buffer(), self.buf.allocation.buffer(), &[BufferCopy {
            src_offset: allocation.offset() as u64,
            dst_offset: self.buf.allocation.offset() as u64,
            size: size as u64,
        }]);
        Ok(Barriers::new(PipelineStageFlags::TRANSFER, dst, vec![], vec![], vec![allocation.as_buffer_memory_barrier(AccessFlags::TRANSFER_WRITE, AccessFlags::empty())]))
        // command_buffer.end().unwrap();
        // let mut submission = Submission::new();
        // submission.add_command_buffer(command_buffer.get_command_buffer());
        // queue.submit_with_fence(&[&submission]).map_err(StarlitError::from)
    }
    pub fn copy_allocation_with_fence(&mut self, allocation: NfPtr, size: usize, queue: Arc<Queue>, command_buffer: &CommandPoolAllocation) -> Result<Arc<Fence>, StarlitError> {
        command_buffer.begin(CommandBufferBeginInfo::SINGLE_SUBMIT).unwrap();
        command_buffer.copy_buffer(self.buf.allocation.buffer(), allocation.buffer(), &[BufferCopy {
            src_offset: self.buf.allocation.offset() as u64,
            dst_offset: allocation.offset() as u64,
            size: size as u64,
        }]);
        command_buffer.end().unwrap();
        let mut submission = Submission::new();
        submission.add_command_buffer(command_buffer.get_command_buffer());
        queue.submit_with_fence(&[&submission]).map_err(StarlitError::from)
    }
}
impl<T, A: GeneralAllocator + ?Sized> std::ops::Index<usize> for VkVec<T, A> {
    fn index(&self, index: usize) -> &Self::Output {
        
        unsafe { self.buf.ptr().unwrap().add(index).as_ref().unwrap() }
    }
    type Output = T;
}
impl<T, A: GeneralAllocator + ?Sized> IndexMut<usize> for VkVec<T, A> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        unsafe { self.buf.ptr().unwrap().add(index).as_mut().unwrap() }
    }
}
impl<T, A: GeneralAllocator + ?Sized> Drop for VkVec<T, A> {
    fn drop(&mut self) {
        println!("Dropped Vector");
        if self.capacity() != 0 {
            self.buf.alloc.deallocate(self.buf.allocation, Layout::from_size_align(self.len*std::mem::size_of::<T>(), std::mem::align_of::<T>()).unwrap()).unwrap();
        }
    }
}
pub struct VkBufferIterator<'a, T> {
    start: *const T,
    end: *const T,
    phantom_: PhantomData<&'a T>
}
impl<'a, T: Copy + Sized> Iterator for VkBufferIterator<'a, T>  {
    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            if self.start == self.end as *const T {
                None
            } else {
                let ret = self.start.as_ref().unwrap();
                self.start = self.start.add(1);
                Some(ret)
            }
        }
    }
    type Item = &'a T;
}
pub struct VkBufferIteratorMut<'a, T> {
    start: *mut T,
    end: *const T,
    phantom_: PhantomData<&'a T>
}
impl<'a, T: Copy + Sized> Iterator for VkBufferIteratorMut<'a, T>  {
    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            if self.start == self.end as *mut T {
                None
            } else {
                let ret = self.start.as_mut().unwrap();
                self.start = self.start.add(1);
                Some(ret)
            }
        }
    }
    type Item = &'a mut T;
}
