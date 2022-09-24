// SPDX-License-Identifier: GPL-2.0-only OR MIT
#![allow(missing_docs)]
#![allow(unused_imports)]
#![allow(dead_code)]

//! Asahi GPU object model

use kernel::macros::versions;

use kernel::{error::code::*, prelude::*};

use alloc::{boxed::Box, fmt};
use core::fmt::Debug;
use core::fmt::Error;
use core::fmt::Formatter;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::num::NonZeroU64;
use core::ops::{Deref, DerefMut, Index, IndexMut};
use core::sync::atomic::{AtomicU32, Ordering};
use core::{mem, ptr, slice};

use crate::alloc::Allocation;

#[repr(C, packed(4))]
pub(crate) struct GPUPointer<'a, T: ?Sized>(NonZeroU64, PhantomData<&'a T>);

impl<'a, T: ?Sized> GPUPointer<'a, T> {
    pub(crate) fn or(&self, other: u64) -> GPUPointer<'a, T> {
        GPUPointer(self.0 | other, PhantomData)
    }

    // The third argument is a type inference hack
    pub(crate) unsafe fn offset<U>(&self, off: usize, _: *const U) -> GPUPointer<'a, U> {
        GPUPointer::<'a, U>(
            NonZeroU64::new(self.0.get() + (off as u64)).unwrap(),
            PhantomData,
        )
    }
}

impl<'a, T: ?Sized> fmt::Debug for GPUPointer<'a, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let val = self.0;
        f.write_fmt(format_args!("{:#x} ({})", val, core::any::type_name::<T>()))
    }
}

#[repr(C, packed(4))]
pub(crate) struct GPUWeakPointer<T: ?Sized>(NonZeroU64, PhantomData<*const T>);

impl<T: ?Sized> Copy for GPUWeakPointer<T> {}

impl<T: ?Sized> Clone for GPUWeakPointer<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ?Sized> GPUWeakPointer<T> {
    pub(crate) fn or(&self, other: u64) -> GPUWeakPointer<T> {
        GPUWeakPointer(self.0 | other, PhantomData)
    }

    // The third argument is a type inference hack
    pub(crate) unsafe fn offset<U>(&self, off: usize, _: *const U) -> GPUWeakPointer<U> {
        GPUWeakPointer::<U>(
            NonZeroU64::new(self.0.get() + (off as u64)).unwrap(),
            PhantomData,
        )
    }
}

impl<T: ?Sized> fmt::Debug for GPUWeakPointer<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let val = self.0;
        f.write_fmt(format_args!("{:#x} ({})", val, core::any::type_name::<T>()))
    }
}

#[repr(transparent)]
pub(crate) struct GPURawPointer(NonZeroU64);

#[macro_export]
macro_rules! inner_ptr {
    ($gpuva:expr, $($f:tt)*) => ({
        fn uninit_from<'a, T: GPUStruct>(_: GPUPointer<'a, T>) -> core::mem::MaybeUninit<T::Raw<'static>> {
            core::mem::MaybeUninit::uninit()
        }
        let tmp = uninit_from($gpuva);
        let outer = tmp.as_ptr();
        let p: *const _ = unsafe { core::ptr::addr_of!((*outer).$($f)*) };
        let inner = p as *const u8;
        let off = unsafe { inner.offset_from(outer as *const u8) };
        unsafe { $gpuva.offset(off.try_into().unwrap(), p) }
    })
}

#[macro_export]
macro_rules! inner_weak_ptr {
    ($gpuva:expr, $($f:tt)*) => ({
        fn uninit_from<T: GPUStruct>(_: GPUWeakPointer<T>) -> core::mem::MaybeUninit<T::Raw<'static>> {
            core::mem::MaybeUninit::uninit()
        }
        let tmp = uninit_from($gpuva);
        let outer = tmp.as_ptr();
        let p: *const _ = unsafe { core::ptr::addr_of!((*outer).$($f)*) };
        let inner = p as *const u8;
        let off = unsafe { inner.offset_from(outer as *const u8) };
        unsafe { $gpuva.offset(off.try_into().unwrap(), p) }
    })
}

pub(crate) trait GPUStruct: 'static {
    type Raw<'a>: Sized;
}

pub(crate) struct GPUObject<T: GPUStruct, U: Allocation<T>> {
    raw: *mut T::Raw<'static>,
    alloc: U,
    gpu_ptr: GPUWeakPointer<T>,
    inner: Box<T>,
}

impl<T: GPUStruct, U: Allocation<T>> GPUObject<T, U> {
    pub(crate) fn new(
        alloc: U,
        inner: T,
        callback: impl for<'a> FnOnce(&'a T) -> T::Raw<'a>,
    ) -> Result<Self> {
        let size = mem::size_of::<T::Raw<'static>>();
        if size > 0x1000 {
            dev_crit!(
                alloc.device(),
                "Allocating {} of size {:#x}, with new, please use new_boxed!",
                core::any::type_name::<T>(),
                size
            );
        }
        if alloc.size() < size {
            return Err(ENOMEM);
        }
        let gpu_ptr =
            GPUWeakPointer::<T>(NonZeroU64::new(alloc.gpu_ptr()).ok_or(EINVAL)?, PhantomData);
        dev_info!(
            alloc.device(),
            "Allocating {} @ {:#x}",
            core::any::type_name::<T>(),
            alloc.gpu_ptr()
        );
        let p = alloc.ptr() as *mut T::Raw<'static>;
        let mut raw = callback(&inner);
        unsafe {
            p.copy_from(&mut raw as *mut _ as *mut u8 as *mut _, 1);
        }
        mem::forget(raw);
        Ok(Self {
            raw: p,
            gpu_ptr,
            alloc,
            inner: Box::try_new(inner)?,
        })
    }

    pub(crate) fn new_boxed(
        alloc: U,
        inner: Box<T>,
        callback: impl for<'a> FnOnce(&'a T, *mut MaybeUninit<T::Raw<'a>>) -> Result<&'a mut T::Raw<'a>>,
    ) -> Result<Self> {
        if alloc.size() < mem::size_of::<T::Raw<'static>>() {
            return Err(ENOMEM);
        }
        let gpu_ptr =
            GPUWeakPointer::<T>(NonZeroU64::new(alloc.gpu_ptr()).ok_or(EINVAL)?, PhantomData);
        dev_info!(
            alloc.device(),
            "Allocating {} @ {:#x}",
            core::any::type_name::<T>(),
            alloc.gpu_ptr()
        );
        let p = alloc.ptr() as *mut MaybeUninit<T::Raw<'_>>;
        let raw = callback(&inner, p)? as *mut _ as *mut MaybeUninit<T::Raw<'_>>;
        if p != raw {
            dev_err!(
                alloc.device(),
                "Allocation callback returned a mismatched reference ({})",
                core::any::type_name::<T>(),
            );
            return Err(EINVAL);
        }
        Ok(Self {
            raw: p as *mut u8 as *mut T::Raw<'static>,
            gpu_ptr,
            alloc,
            inner,
        })
    }

    pub(crate) fn new_inplace(
        alloc: U,
        inner: T,
        callback: impl for<'a> FnOnce(&'a T, *mut MaybeUninit<T::Raw<'a>>) -> Result<&'a mut T::Raw<'a>>,
    ) -> Result<Self> {
        GPUObject::<T, U>::new_boxed(alloc, Box::try_new(inner)?, callback)
    }

    pub(crate) fn new_prealloc(
        alloc: U,
        inner_cb: impl FnOnce(GPUWeakPointer<T>) -> Result<Box<T>>,
        raw_cb: impl for<'a> FnOnce(&'a T, *mut MaybeUninit<T::Raw<'a>>) -> Result<&'a mut T::Raw<'a>>,
    ) -> Result<Self> {
        if alloc.size() < mem::size_of::<T::Raw<'static>>() {
            return Err(ENOMEM);
        }
        let gpu_ptr =
            GPUWeakPointer::<T>(NonZeroU64::new(alloc.gpu_ptr()).ok_or(EINVAL)?, PhantomData);
        dev_info!(
            alloc.device(),
            "Allocating {} @ {:#x}",
            core::any::type_name::<T>(),
            alloc.gpu_ptr()
        );
        let inner = inner_cb(gpu_ptr)?;
        let p = alloc.ptr() as *mut MaybeUninit<T::Raw<'_>>;
        let raw = raw_cb(&*inner, p)? as *mut _ as *mut MaybeUninit<T::Raw<'_>>;
        if p != raw {
            dev_err!(
                alloc.device(),
                "Allocation callback returned a mismatched reference ({})",
                core::any::type_name::<T>(),
            );
            return Err(EINVAL);
        }
        Ok(Self {
            raw: p as *mut u8 as *mut T::Raw<'static>,
            gpu_ptr,
            alloc,
            inner,
        })
    }

    pub(crate) fn gpu_va(&self) -> NonZeroU64 {
        self.gpu_ptr.0
    }

    pub(crate) fn gpu_pointer(&self) -> GPUPointer<'_, T> {
        GPUPointer(self.gpu_ptr.0, PhantomData)
    }

    pub(crate) fn weak_pointer(&self) -> GPUWeakPointer<T> {
        GPUWeakPointer(self.gpu_ptr.0, PhantomData)
    }

    /* FIXME: unsound
    pub(crate) fn raw_ref(&self) -> &<T as GPUStruct>::Raw<'_> {
        unsafe { &*(self.raw as *mut u8 as *mut <T as GPUStruct>::Raw<'_>) }
    }

    pub(crate) fn raw_mut(&mut self) -> &mut <T as GPUStruct>::Raw<'_> {
        unsafe { &mut *(self.raw as *mut u8 as *mut <T as GPUStruct>::Raw<'_>) }
    }
    */

    pub(crate) fn with_mut<RetVal>(
        &mut self,
        callback: impl for<'a> FnOnce(&'a mut <T as GPUStruct>::Raw<'a>, &'a mut T) -> RetVal,
    ) -> RetVal {
        unsafe { callback(&mut *self.raw, &mut *(&mut *self.inner as *mut _)) }
    }

    pub(crate) fn with<RetVal>(
        &self,
        callback: impl for<'a> FnOnce(&'a <T as GPUStruct>::Raw<'a>, &'a T) -> RetVal,
    ) -> RetVal {
        unsafe { callback(&*self.raw, &*(&*self.inner as *const _)) }
    }
}

pub(crate) trait OpaqueGPUObject {}
impl<'a, T: GPUStruct, U: Allocation<T>> OpaqueGPUObject for GPUObject<T, U> {}

impl<'a, T: GPUStruct, U: Allocation<T>> Deref for GPUObject<T, U> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.inner
    }
}

impl<'a, T: GPUStruct, U: Allocation<T>> DerefMut for GPUObject<T, U> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.inner
    }
}

impl<'a, T: GPUStruct + Debug, U: Allocation<T>> Debug for GPUObject<T, U>
where
    <T as GPUStruct>::Raw<'static>: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct(core::any::type_name::<T>())
            .field("raw", &format_args!("{:#X?}", unsafe { &*self.raw }))
            .field("inner", &format_args!("{:#X?}", &self.inner))
            .field("alloc", &format_args!("{:?}", &self.alloc))
            .finish()
    }
}

impl<T: GPUStruct + Default, U: Allocation<T>> GPUObject<T, U>
where
    for<'a> <T as GPUStruct>::Raw<'a>: Default,
{
    pub(crate) fn new_default(alloc: U) -> Result<Self> {
        GPUObject::<T, U>::new_inplace(alloc, Default::default(), |_inner, raw| {
            Ok(unsafe {
                ptr::write_bytes(raw, 0, 1);
                (*raw).assume_init_mut()
            })
        })
    }
}

pub(crate) struct GPUArray<T: Sized, U: Allocation<T>> {
    raw: *mut T,
    len: usize,
    alloc: U,
    gpu_ptr: NonZeroU64,
}

impl<T: Sized + Copy, U: Allocation<T>> GPUArray<T, U> {
    pub(crate) fn new(alloc: U, data: &[T]) -> Result<GPUArray<T, U>> {
        let bytes = data.len() * mem::size_of::<T>();
        let gpu_ptr = NonZeroU64::new(alloc.gpu_ptr()).ok_or(EINVAL)?;
        if alloc.size() < bytes {
            return Err(ENOMEM);
        }
        let p = alloc.ptr() as *mut T;
        unsafe {
            ptr::copy(data.as_ptr(), p, bytes);
        }
        Ok(Self {
            raw: p,
            len: data.len(),
            alloc,
            gpu_ptr,
        })
    }
}

impl<T: Sized + Default, U: Allocation<T>> GPUArray<T, U> {
    pub(crate) fn empty(alloc: U, count: usize) -> Result<GPUArray<T, U>> {
        let bytes = count * mem::size_of::<T>();
        let gpu_ptr = NonZeroU64::new(alloc.gpu_ptr()).ok_or(EINVAL)?;
        dev_info!(
            alloc.device(),
            "Allocating {} * {:#x} @ {:#x}",
            core::any::type_name::<T>(),
            bytes,
            alloc.gpu_ptr(),
        );
        if alloc.size() < bytes {
            return Err(ENOMEM);
        }
        let p = alloc.ptr() as *mut T;
        let mut pi = p;
        for _i in 0..count {
            unsafe {
                pi.write(Default::default());
            }
            pi = unsafe { pi.add(1) };
        }
        Ok(Self {
            raw: p,
            len: count,
            alloc,
            gpu_ptr,
        })
    }
}

impl<T: Sized, U: Allocation<T>> GPUArray<T, U> {
    pub(crate) fn gpu_va(&self) -> NonZeroU64 {
        self.gpu_ptr
    }

    pub(crate) fn gpu_pointer(&self) -> GPUPointer<'_, &'_ [T]> {
        GPUPointer(self.gpu_ptr, PhantomData)
    }

    pub(crate) fn weak_pointer(&self) -> GPUWeakPointer<[T]> {
        GPUWeakPointer(self.gpu_ptr, PhantomData)
    }

    pub(crate) fn gpu_item_pointer(&self, index: usize) -> GPUPointer<'_, &'_ T> {
        if index > self.len {
            panic!("Index {} out of bounds (len: {})", index, self.len);
        }
        GPUPointer(
            NonZeroU64::new(self.gpu_ptr.get() + (index * mem::size_of::<T>()) as u64).unwrap(),
            PhantomData,
        )
    }

    pub(crate) fn weak_item_pointer(&self, index: usize) -> GPUWeakPointer<T> {
        if index > self.len {
            panic!("Index {} out of bounds (len: {})", index, self.len);
        }
        GPUWeakPointer(
            NonZeroU64::new(self.gpu_ptr.get() + (index * mem::size_of::<T>()) as u64).unwrap(),
            PhantomData,
        )
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.raw, self.len) }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.raw, self.len) }
    }
}

impl<T: Sized, U: Allocation<T>> Index<usize> for GPUArray<T, U> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        if index >= self.len {
            panic!("Index {} out of bounds (len: {})", index, self.len);
        }
        unsafe { &*(self.raw.add(index)) }
    }
}

impl<T: Sized, U: Allocation<T>> IndexMut<usize> for GPUArray<T, U> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        if index >= self.len {
            panic!("Index {} out of bounds (len: {})", index, self.len);
        }
        unsafe { &mut *(self.raw.add(index)) }
    }
}

unsafe impl<T: GPUStruct + Send, U: Allocation<T>> Send for GPUObject<T, U> {}
unsafe impl<T: Sized + Send, U: Allocation<T>> Send for GPUArray<T, U> {}

impl<T: GPUStruct, U: Allocation<T>> Drop for GPUObject<T, U> {
    fn drop(&mut self) {
        dev_info!(
            self.alloc.device(),
            "Dropping {} @ {:?}",
            core::any::type_name::<T>(),
            self.gpu_pointer()
        );
    }
}

impl<T: Sized, U: Allocation<T>> Drop for GPUArray<T, U> {
    fn drop(&mut self) {
        dev_info!(
            self.alloc.device(),
            "Dropping {} @ {:?}",
            core::any::type_name::<T>(),
            self.gpu_pointer()
        );
    }
}

impl<'a, T: Sized + fmt::Debug, U: Allocation<T>> fmt::Debug for GPUArray<T, U> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct(core::any::type_name::<T>())
            .field("array", &format_args!("{:#X?}", self.as_slice()))
            .finish()
    }
}
