// SPDX-License-Identifier: GPL-2.0 OR MIT
#![allow(missing_docs)]

//! DRM GEM API
//!
//! C header: [`include/linux/drm/drm_gem.h`](../../../../include/linux/drm/drm_gem.h)

pub mod shmem;

use alloc::boxed::Box;

use crate::{
    bindings,
    drm::{device, drv, private},
    to_result, Result,
};
use core::{mem, mem::ManuallyDrop, ops::Deref};

/// GEM object functions
pub trait BaseDriverObject<T: BaseObject>: Sync + Send + Sized {
    fn init(_obj: &mut T) -> Result<()> {
        Ok(())
    }
    fn uninit(_obj: &mut T) {}
}

pub trait IntoGEMObject: Sized + private::Sealed {
    fn gem_obj(&self) -> &bindings::drm_gem_object;
}

pub trait BaseObject: IntoGEMObject {
    fn size(&self) -> usize;
    fn reference(&self) -> ObjectRef<Self>;
}

#[repr(C)]
pub struct Object<T: DriverObject> {
    obj: bindings::drm_gem_object,
    dev: ManuallyDrop<device::Device<T::Driver>>,
    pub p: T,
}

pub trait DriverObject: BaseDriverObject<Object<Self>> {
    type Driver: drv::Driver;
}

pub struct ObjectRef<T: IntoGEMObject> {
    // Invariant: the pointer is valid and initialized, and this ObjectRef owns a reference to it
    ptr: *const T,
}

unsafe extern "C" fn free_callback<T: DriverObject>(obj: *mut bindings::drm_gem_object) {
    // SAFETY: All of our objects are Object<T>.
    let this = crate::container_of!(obj, Object<T>, obj) as *mut _;

    // SAFETY: The pointer must be valid since we used container_of!()
    T::uninit(unsafe { &mut *this });

    // SAFETY: The pointer we got has to be valid
    unsafe { bindings::drm_gem_object_release(obj) };

    // SAFETY: All of our objects are allocated via Box<>, and we're in the
    // free callback which guarantees this object has zero remaining references,
    // so we can drop it
    unsafe { Box::from_raw(this) };
}

impl<T: DriverObject> IntoGEMObject for Object<T> {
    fn gem_obj(&self) -> &bindings::drm_gem_object {
        &self.obj
    }
}

impl<T: IntoGEMObject + Sized> BaseObject for T {
    fn size(&self) -> usize {
        self.gem_obj().size
    }

    fn reference(&self) -> ObjectRef<Self> {
        // SAFETY: Having a reference to an Object implies holding a GEM reference
        unsafe {
            bindings::drm_gem_object_get(self.gem_obj() as *const _ as *mut _);
        }
        ObjectRef {
            ptr: self as *const _,
        }
    }
}

impl<T: DriverObject> private::Sealed for Object<T> {}

impl<T: DriverObject> drv::AllocImpl for Object<T> {
    const ALLOC_OPS: drv::AllocOps = drv::AllocOps {
        gem_create_object: None,
        prime_handle_to_fd: Some(bindings::drm_gem_prime_handle_to_fd),
        prime_fd_to_handle: Some(bindings::drm_gem_prime_fd_to_handle),
        gem_prime_import: None,
        gem_prime_import_sg_table: None,
        gem_prime_mmap: Some(bindings::drm_gem_prime_mmap),
        dumb_create: None,
        dumb_map_offset: None,
        dumb_destroy: None,
    };
}

impl<T: DriverObject> Object<T> {
    const SIZE: usize = mem::size_of::<Self>();

    const OBJECT_FUNCS: bindings::drm_gem_object_funcs = bindings::drm_gem_object_funcs {
        free: Some(free_callback::<T>),
        open: None,
        close: None,
        print_info: None,
        export: None,
        pin: None,
        unpin: None,
        get_sg_table: None,
        vmap: None,
        vunmap: None,
        mmap: None,
        vm_ops: core::ptr::null_mut(),
    };

    pub fn new(
        dev: &device::Device<T::Driver>,
        private: T,
        size: usize,
    ) -> Result<ObjectRef<Self>> {
        let mut obj: Box<Self> = Box::try_new(Self {
            // SAFETY: This struct is expected to be zero-initialized
            obj: unsafe { mem::zeroed() },
            // SAFETY: The drm subsystem guarantees that the drm_device will live as long as
            // the GEM object lives, so we can conjure a reference out of thin air.
            dev: ManuallyDrop::new(unsafe { device::Device::from_raw(dev.ptr) }),
            p: private,
        })?;

        obj.obj.funcs = &Self::OBJECT_FUNCS;
        to_result(unsafe {
            bindings::drm_gem_object_init(dev.raw() as *mut _, &mut obj.obj, size)
        })?;

        let obj_ref = ObjectRef {
            ptr: Box::leak(obj),
        };

        // SAFETY: We have the only ref so far, so it's safe to deref as mutable
        T::init(unsafe { &mut *(obj_ref.ptr as *mut _) })?;

        Ok(obj_ref)
    }

    pub fn dev(&self) -> &device::Device<T::Driver> {
        &self.dev
    }
}

impl<T: IntoGEMObject> Clone for ObjectRef<T> {
    fn clone(&self) -> Self {
        self.reference()
    }
}

impl<T: IntoGEMObject> Drop for ObjectRef<T> {
    fn drop(&mut self) {
        // SAFETY: Having an ObjectRef implies holding a GEM reference
        // The free callback will take care of deallocation
        unsafe {
            bindings::drm_gem_object_put((*self.ptr).gem_obj() as *const _ as *mut _);
        }
    }
}

impl<T: IntoGEMObject> Deref for ObjectRef<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: The pointer is valid per the invariant
        unsafe { &*self.ptr }
    }
}
