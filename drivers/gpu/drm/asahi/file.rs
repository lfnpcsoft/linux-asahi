// SPDX-License-Identifier: GPL-2.0-only OR MIT
#![allow(missing_docs)]
#![allow(unused_imports)]
#![allow(dead_code)]

//! Asahi File state

use crate::driver::AsahiDevice;
use crate::{driver, gpu};
use kernel::drm;
use kernel::prelude::*;
use kernel::sync::Ref;

pub(crate) struct File {}

impl drm::file::DriverFile for File {
    type Driver = driver::AsahiDriver;

    fn open(device: &AsahiDevice) -> Result<Box<Self>> {
        dev_info!(device, "DRM device opened");

        Ok(Box::try_new(Self {})?)
    }
}
