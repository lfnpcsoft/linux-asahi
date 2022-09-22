// SPDX-License-Identifier: GPL-2.0-only OR MIT
#![allow(missing_docs)]
#![allow(unused_imports)]
#![allow(dead_code)]

//! Asahi GEM object implementation

use kernel::{
    bindings,
    c_str,
    drm,
    drm::{drv, device, gem, gem::shmem},
    error::{Result, to_result},
    io_mem::IoMem,
    module_platform_driver, of, platform,
    prelude::*,
    soc::apple::rtkit,
    sync::{Ref, RefBorrow},
    sync::smutex::Mutex,
};

use kernel::drm::gem::BaseObject;

pub(crate) struct DriverObject {

}

pub(crate) type Object = shmem::Object<DriverObject>;

pub(crate) struct ObjectRef {
    pub(crate) gem: gem::ObjectRef<shmem::Object<DriverObject>>,
    pub(crate) mapping: Option<crate::mmu::Mapping>,
    pub(crate) vmap: Option<shmem::VMap<DriverObject>>,
}

impl ObjectRef {
    pub(crate) fn vmap(&mut self) -> Result<&mut shmem::VMap<DriverObject>> {
        if self.vmap.is_none() {
            self.vmap = Some(self.gem.vmap()?);
        }
        Ok(self.vmap.as_mut().unwrap())
    }

    pub(crate) fn map_into(&mut self, context: &crate::mmu::Context) -> Result<&crate::mmu::Mapping> {
        if self.mapping.is_some() {
            Err(EBUSY)
        } else {

            let sgt = self.gem.sg_table()?;
            let mapping = context.map(self.gem.size(), &mut sgt.iter())?;

            self.mapping = Some(mapping);
            Ok(self.mapping.as_ref().unwrap())
        }
    }
}

pub(crate) fn new_object(dev: &device::Device, size: usize) -> Result<ObjectRef> {
    let private = DriverObject {
    };
    Ok(ObjectRef {
        gem:shmem::Object::new(dev, private, size)?,
        mapping: None,
        vmap: None,
    })
}

impl gem::BaseDriverObject<Object> for DriverObject {
    fn init(obj: &mut Object) -> Result<()> {
        dev_info!(obj.dev(), "DriverObject::init\n");
        Ok(())
    }
    fn uninit(obj: &mut Object) {
        dev_info!(obj.dev(), "DriverObject::uninit\n");
    }
}

impl shmem::DriverObject for DriverObject {
    type Driver = crate::driver::AsahiDevice;
}

impl rtkit::Buffer for ObjectRef {
    fn iova(&self) -> Option<usize> {
        Some(self.mapping.as_ref()?.iova())
    }
    fn buf(&mut self) -> Option<&mut [u8]> {
        let vmap = self.vmap.as_mut()?;
        Some(vmap.as_mut_slice())
    }
}
