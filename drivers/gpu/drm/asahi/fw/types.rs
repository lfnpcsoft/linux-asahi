// SPDX-License-Identifier: GPL-2.0-only OR MIT
#![allow(missing_docs)]
#![allow(unused_imports)]

//! Common types for firmware structure definitions

pub(crate) use kernel::macros::versions;
use crate::{object, alloc};

use core::sync::atomic;
use core::fmt;
use core::ops::{Index, IndexMut, Deref, DerefMut};
pub(crate) use ::alloc::boxed::Box;
pub(crate) use core::sync::atomic::{AtomicU64, AtomicU32, AtomicU16, AtomicU8};
pub(crate) use crate::object::{GPUStruct, GPUPointer};
pub(crate) type GPUObject<T> = object::GPUObject<T, alloc::SimpleAllocation<T>>;
pub(crate) type GPUArray<T> = object::GPUArray<T, alloc::SimpleAllocation<T>>;
pub(crate) use crate::alloc::Allocator as _Allocator;
pub(crate) type Allocator = alloc::SimpleAllocator;
pub(crate) use core::marker::PhantomData;
pub(crate) use core::fmt::Debug;

#[derive(Default, Debug, Copy, Clone)]
pub(crate) struct F32(u32);

impl F32 {
    pub(crate) const fn new(v: f32) -> F32 {
        F32(v.to_bits())
    }
}

#[macro_export]
macro_rules! const_f32 {
    ($val:expr) => {
        {
            const _K: F32 = F32::new($val);
            _K
        }
    };
}

#[derive(Copy, Clone)]
#[repr(C, packed)]
pub(crate) struct Pad<const N: usize>([u8; N]);

impl<const N: usize> Default for Pad<N> {
    fn default() -> Self {
        Pad([0; N])
    }
}

impl<const N: usize> fmt::Debug for Pad<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_fmt(format_args!("<pad>"))
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) struct Array<const N: usize, T: Sized>([T; N]);

impl<const N: usize, T: Sized> Array<N, T> {
    pub(crate) fn new(data: [T; N]) -> Self {
        Self(data)
    }
}

impl<const N: usize, T: Sized + Default + Copy> Default for Array<N, T> {
    fn default() -> Self {
        Array([Default::default(); N])
    }
}

impl<const N: usize, T: Sized> Index<usize> for Array<N, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const N: usize, T: Sized> IndexMut<usize> for Array<N, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<const N: usize, T: Sized> Deref for Array<N, T> {
    type Target = [T; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const N: usize, T: Sized> DerefMut for Array<N, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<const N: usize, T: Sized + fmt::Debug> fmt::Debug for Array<N, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/*
#[derive(Debug, Default)]
#[repr(transparent)]
struct Atomic<T>(T);

impl<T: Sized> Deref for Array<N, T> {
    type Target = [T; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<const N: usize, T: Sized> DerefMut for Array<N, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
*/
