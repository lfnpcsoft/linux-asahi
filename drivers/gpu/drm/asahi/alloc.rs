// SPDX-License-Identifier: GPL-2.0-only OR MIT
#![allow(missing_docs)]
#![allow(unused_imports)]
#![allow(dead_code)]

//! Asahi kernel object allocator

use kernel::{
    drm::{device, gem, gem::shmem, mm},
    error::Result,
    prelude::*,
};

use crate::object::{GPUArray, GPUObject, GPUStruct};
use crate::driver::AsahiDevice;

use alloc::fmt;
use core::fmt::{Debug, Formatter};
use core::mem;
use core::mem::MaybeUninit;

pub(crate) trait Allocation<T>: Debug {
    fn ptr(&self) -> *mut T;
    fn gpu_ptr(&self) -> u64;
    fn size(&self) -> usize;

    fn device(&self) -> &AsahiDevice;
}

pub(crate) trait Allocator {
    type Allocation<T>: Allocation<T>;

    fn new<T: GPUStruct>(
        &mut self,
        inner: T,
        callback: impl for<'a> FnOnce(&'a T) -> T::Raw<'a>,
    ) -> Result<GPUObject<T, Self::Allocation<T>>>;

    fn new_boxed<T: GPUStruct>(
        &mut self,
        inner: Box<T>,
        callback: impl for<'a> FnOnce(&'a T, *mut MaybeUninit<T::Raw<'a>>) -> Result<&'a mut T::Raw<'a>>,
    ) -> Result<GPUObject<T, Self::Allocation<T>>>;

    fn new_inplace<T: GPUStruct>(
        &mut self,
        inner: T,
        callback: impl for<'a> FnOnce(&'a T, *mut MaybeUninit<T::Raw<'a>>) -> Result<&'a mut T::Raw<'a>>,
    ) -> Result<GPUObject<T, Self::Allocation<T>>>;

    fn prealloc<T: GPUStruct>(&mut self) -> Result<Self::Allocation<T>>;

    fn new_prealloc<T: GPUStruct>(
        &mut self,
        inner_cb: impl FnOnce(GPUWeakPointer<T>) -> Result<Box<T>>,
        raw_cb: impl for<'a> FnOnce(&'a T, *mut MaybeUninit<T::Raw<'a>>) -> Result<&'a mut T::Raw<'a>>,
    ) -> Result<GPUObject<T, Self::Allocation<T>>>;

    fn new_default<T: GPUStruct + Default>(&mut self) -> Result<GPUObject<T, Self::Allocation<T>>>
    where
        for<'a> <T as GPUStruct>::Raw<'a>: Default;

    fn array_empty<T: Sized + Default>(
        &mut self,
        count: usize,
    ) -> Result<GPUArray<T, Self::Allocation<T>>>;

    fn device(&self) -> &AsahiDevice;
}

pub(crate) struct SimpleAllocation<T> {
    dev: AsahiDevice,
    ptr: *mut T,
    gpu_ptr: u64,
    size: usize,
    vm: crate::mmu::VM,
    obj: crate::gem::ObjectRef,
}

impl<T> Debug for SimpleAllocation<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct(core::any::type_name::<SimpleAllocation<T>>())
            .field("ptr", &format_args!("{:p}", self.ptr))
            .field("gpu_ptr", &format_args!("{:#X?}", self.gpu_ptr))
            .field("size", &format_args!("{:#X?}", self.size))
            .finish()
    }
}
impl<T> Allocation<T> for SimpleAllocation<T> {
    fn ptr(&self) -> *mut T {
        self.ptr
    }
    fn gpu_ptr(&self) -> u64 {
        self.gpu_ptr
    }
    fn size(&self) -> usize {
        self.size
    }

    fn device(&self) -> &AsahiDevice {
        &self.dev
    }
}

pub(crate) struct SimpleAllocator {
    dev: AsahiDevice,
    vm: crate::mmu::VM,
    min_align: usize,
}

impl SimpleAllocator {
    pub(crate) fn new(
        dev: &AsahiDevice,
        vm: &crate::mmu::VM,
        min_align: usize,
    ) -> SimpleAllocator {
        SimpleAllocator {
            dev: dev.clone(),
            vm: vm.clone(),
            min_align,
        }
    }

    #[inline(never)]
    fn alloc_object<T: GPUStruct>(&mut self) -> Result<SimpleAllocation<T>> {
        let size = mem::size_of::<T::Raw<'static>>();
        let size_aligned = (size + crate::mmu::UAT_PGSZ - 1) & !crate::mmu::UAT_PGMSK;
        let align = self.min_align.max(mem::align_of::<T::Raw<'static>>());
        let offset = (size_aligned - size) & !(align - 1);

        dev_info!(
            &self.dev,
            "Allocator::new: size={:#x} size_al={:#x} al={:#x} off={:#x}",
            size,
            size_aligned,
            align,
            offset
        );

        let mut obj = crate::gem::new_object(&self.dev, size_aligned)?;
        let p = obj.vmap()?.as_mut_ptr() as *mut u8;
        let map = obj.map_into(&self.vm)?;

        let ptr = unsafe { p.add(offset) } as *mut T;
        let gpu_ptr = (map.iova() + offset) as u64;

        dev_info!(
            &self.dev,
            "Allocator::new -> {:#?} / {:#?} | {:#x} / {:#x}",
            p,
            ptr,
            map.iova(),
            gpu_ptr
        );

        Ok(SimpleAllocation {
            dev: self.dev.clone(),
            ptr,
            gpu_ptr,
            size,
            vm: self.vm.clone(),
            obj,
        })
    }
}

impl Allocator for SimpleAllocator {
    type Allocation<T> = SimpleAllocation<T>;

    #[inline(never)]
    fn new<T: GPUStruct>(
        &mut self,
        inner: T,
        callback: impl for<'a> FnOnce(&'a T) -> T::Raw<'a>,
    ) -> Result<GPUObject<T, Self::Allocation<T>>> {
        GPUObject::<T, Self::Allocation<T>>::new(self.alloc_object()?, inner, callback)
    }

    #[inline(never)]
    fn new_boxed<T: GPUStruct>(
        &mut self,
        inner: Box<T>,
        callback: impl for<'a> FnOnce(&'a T, *mut MaybeUninit<T::Raw<'a>>) -> Result<&'a mut T::Raw<'a>>,
    ) -> Result<GPUObject<T, Self::Allocation<T>>> {
        GPUObject::<T, Self::Allocation<T>>::new_boxed(self.alloc_object()?, inner, callback)
    }

    #[inline(never)]
    fn new_inplace<T: GPUStruct>(
        &mut self,
        inner: T,
        callback: impl for<'a> FnOnce(&'a T, *mut MaybeUninit<T::Raw<'a>>) -> Result<&'a mut T::Raw<'a>>,
    ) -> Result<GPUObject<T, Self::Allocation<T>>> {
        GPUObject::<T, Self::Allocation<T>>::new_inplace(self.alloc_object()?, inner, callback)
    }

    #[inline(never)]
    fn new_default<T: GPUStruct + Default>(&mut self) -> Result<GPUObject<T, Self::Allocation<T>>>
    where
        for<'a> <T as GPUStruct>::Raw<'a>: Default,
    {
        GPUObject::<T, Self::Allocation<T>>::new_default(self.alloc_object()?)
    }

    #[inline(never)]
    fn prealloc<T: GPUStruct>(&mut self) -> Result<Self::Allocation<T>> {
        self.alloc_object()
    }

    #[inline(never)]
    fn new_prealloc<T: GPUStruct>(
        &mut self,
        inner_cb: impl FnOnce(GPUWeakPointer<T>) -> Result<Box<T>>,
        raw_cb: impl for<'a> FnOnce(&'a T, *mut MaybeUninit<T::Raw<'a>>) -> Result<&'a mut T::Raw<'a>>,
    ) -> Result<GPUObject<T, Self::Allocation<T>>> {
        GPUObject::<T, Self::Allocation<T>>::new_prealloc(self.alloc_object()?, inner_cb, raw_cb)
    }

    fn array_empty<T: Sized + Default>(
        &mut self,
        count: usize,
    ) -> Result<GPUArray<T, Self::Allocation<T>>> {
        let size = mem::size_of::<T>() * count;
        let size_aligned = (size + crate::mmu::UAT_PGSZ - 1) & !crate::mmu::UAT_PGMSK;
        let align = self.min_align.max(mem::align_of::<T>());
        let offset = (size_aligned - size) & !(align - 1);

        dev_info!(
            &self.dev,
            "Allocator::array_empty: size={:#x} size_al={:#x} al={:#x} off={:#x} ({:#x} * {:#x})",
            size,
            size_aligned,
            align,
            offset,
            mem::size_of::<T>(),
            count
        );

        let mut obj = crate::gem::new_object(&self.dev, size_aligned)?;
        let p = obj.vmap()?.as_mut_ptr() as *mut u8;
        let ptr = unsafe { p.add(offset) } as *mut T;
        let map = obj.map_into(&self.vm)?;
        let gpu_ptr = (map.iova() + offset) as u64;

        dev_info!(
            &self.dev,
            "Allocator::array_empty -> {:#?} / {:#?} | {:#x} / {:#x}",
            p,
            ptr,
            map.iova(),
            gpu_ptr
        );

        let alloc = SimpleAllocation {
            dev: self.dev.clone(),
            ptr,
            gpu_ptr,
            size,
            vm: self.vm.clone(),
            obj,
        };

        GPUArray::<T, Self::Allocation<T>>::empty(alloc, count)
    }

    fn device(&self) -> &AsahiDevice {
        &self.dev
    }
}
