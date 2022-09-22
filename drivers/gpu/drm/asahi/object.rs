// SPDX-License-Identifier: GPL-2.0-only OR MIT
#![allow(missing_docs)]
#![allow(unused_imports)]
#![allow(dead_code)]

//! Asahi GPU object model

use kernel::macros::versions;

use kernel::{error::code::*, prelude::*};

use alloc::{fmt, boxed::Box};
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

#[repr(transparent)]
pub(crate) struct GPUPointer<'a, T>(NonZeroU64, PhantomData<&'a T>);

impl<'a, T> fmt::Debug for GPUPointer<'a, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!(
            "{:#x} ({})",
            self.0,
            core::any::type_name::<T>()
        ))
    }
}

#[repr(transparent)]
pub(crate) struct GPUWeakPointer<T>(NonZeroU64, PhantomData<T>);

impl<T> GPUWeakPointer<T> {
    // The third argument is a type inference hack
    pub(crate) unsafe fn offset<U>(&self, off: usize, _: *const U) -> GPUWeakPointer<U> {
        GPUWeakPointer::<U>(
            NonZeroU64::new(self.0.get() + (off as u64)).unwrap(),
            PhantomData,
        )
    }
}

#[repr(transparent)]
pub(crate) struct GPURawPointer(NonZeroU64);

#[macro_export]
macro_rules! inner_ptr {
    ($gpuva:expr, $($f:tt)*) => ({
        fn uninit_from<T: GPUStruct>(_: &GPUWeakPointer<T>) -> MaybeUninit<T::Raw<'static>> {
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
    alloc: Box<U>,
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
            alloc: Box::try_new(alloc)?,
            inner: Box::try_new(inner)?,
        })
    }

    pub(crate) fn new_boxed(
        alloc: U,
        inner: Box<T>,
        callback: impl for<'a> FnOnce(&'a T) -> Result<Box<T::Raw<'a>>>,
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
        let p = alloc.ptr() as *mut T::Raw<'static>;
        let raw = Box::into_raw(callback(&*inner)?);
        unsafe {
            p.copy_from(raw as *mut u8 as *mut _, 1);
            alloc::alloc::dealloc(raw as *mut u8, core::alloc::Layout::new::<T::Raw<'static>>());
        }
        mem::forget(raw);
        Ok(Self {
            raw: p,
            gpu_ptr,
            alloc: Box::try_new(alloc)?,
            inner,
        })
    }

    pub(crate) fn new_prealloc(
        alloc: U,
        inner_cb: impl FnOnce(&GPUWeakPointer<T>) -> Box<T>,
        raw_cb: impl for<'a> FnOnce(&'a T) -> T::Raw<'a>,
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
        let inner = inner_cb(&gpu_ptr);
        let p = alloc.ptr() as *mut T::Raw<'static>;
        let mut raw = raw_cb(&*inner);
        unsafe {
            p.copy_from(&mut raw as *mut _ as *mut u8 as *mut _, 1);
        }
        mem::forget(raw);
        Ok(Self {
            raw: p,
            gpu_ptr,
            alloc: Box::try_new(alloc)?,
            inner,
        })
    }

    pub(crate) fn gpu_pointer(&self) -> GPUPointer<'_, T> {
        GPUPointer(self.gpu_ptr.0, PhantomData)
    }

    pub(crate) fn weak_pointer(&self) -> GPUWeakPointer<T> {
        GPUWeakPointer(self.gpu_ptr.0, PhantomData)
    }

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

impl<'a, T: GPUStruct + Debug, U: Allocation<T>> Debug for GPUObject<T, U> where <T as GPUStruct>::Raw<'static>: Debug {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct(core::any::type_name::<T>())
            .field("raw", &format_args!("{:#X?}", unsafe { &*self.raw }))
            .field("inner", &format_args!("{:#X?}", &self.inner))
            .finish()
    }
}

pub(crate) struct GPUArray<T: Sized, U: Allocation<T>> {
    raw: *mut T,
    len: usize,
    alloc: Box<U>,
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
            alloc: Box::try_new(alloc)?,
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
            alloc: Box::try_new(alloc)?,
            gpu_ptr,
        })
    }
}

impl<T: Sized, U: Allocation<T>> GPUArray<T, U> {
    pub(crate) fn gpu_pointer(&self) -> GPUPointer<'_, &'_ [T]> {
        GPUPointer(self.gpu_ptr, PhantomData)
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
