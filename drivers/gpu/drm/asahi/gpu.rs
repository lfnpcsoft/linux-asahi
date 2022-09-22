// SPDX-License-Identifier: GPL-2.0-only OR MIT
#![allow(missing_docs)]
#![allow(dead_code)]

use core::mem;

use kernel::{
    drm,
    macros::versions,
    prelude::*,
    soc::apple::rtkit,
    sync::{smutex::Mutex, Ref, UniqueRef},
    PointerWrapper,
};

use crate::{alloc, channel, fw, gem, hw, initdata, mmu};

const EP_FIRMWARE: u8 = 0x20;
const EP_DOORBELL: u8 = 0x21;

const MSG_INIT: u64 = 0x81 << 48;
const INIT_DATA_MASK: u64 = (1 << 44) - 1;

const MSG_TX_DOORBELL: u64 = 0x83 << 48;
const MSG_FWCTL: u64 = 0x84 << 48;
const MSG_HALT: u64 = 0x85 << 48;

const MSG_RX_DOORBELL: u64 = 0x42 << 48;

pub(crate) struct KernelAllocators {
    pub(crate) private: alloc::SimpleAllocator,
    pub(crate) shared: alloc::SimpleAllocator,
    pub(crate) gpu: alloc::SimpleAllocator,
}

struct RXChannels {
    event: channel::EventChannel,
    fw_log: channel::FWLogChannel,
    ktrace: channel::KTraceChannel,
    stats: channel::StatsChannel,
}

#[versions(AGX)]
pub(crate) struct GPUManager {
    dev: drm::device::Device,
    initialized: bool,
    initdata: fw::types::GPUObject<fw::initdata::InitData::ver>,
    uat: mmu::UAT,
    alloc: KernelAllocators,
    io_mappings: Vec<mmu::Mapping>,
    rtkit: Mutex<Option<rtkit::RTKit<GPUManager::ver>>>,
    rx_channels: Mutex<RXChannels>,
}

pub(crate) trait GPUManager: Send + Sync {
    fn init(&self) -> Result;
    fn alloc(&mut self) -> &mut KernelAllocators;
}

#[versions(AGX)]
#[vtable]
impl rtkit::Operations for GPUManager::ver {
    type Data = Ref<GPUManager::ver>;
    type Buffer = gem::ObjectRef;

    fn recv_message(data: <Self::Data as PointerWrapper>::Borrowed<'_>, ep: u8, msg: u64) {
        let dev = &data.dev;
        dev_info!(dev, "RTKit message: {:#x}:{:#x}\n", ep, msg);

        if ep != EP_FIRMWARE || msg != MSG_RX_DOORBELL {
            dev_err!(dev, "Unknown message: {:#x}:{:#x}\n", ep, msg);
            return;
        }

        let mut ch = data.rx_channels.lock();

        ch.fw_log.poll();
        ch.ktrace.poll();
        ch.stats.poll();
        ch.event.poll();
    }

    fn shmem_alloc(
        data: <Self::Data as PointerWrapper>::Borrowed<'_>,
        size: usize,
    ) -> Result<Self::Buffer> {
        let dev = &data.dev;
        dev_info!(dev, "shmem_alloc() {:#x} bytes\n", size);

        let mut obj = gem::new_object(dev, size)?;
        obj.vmap()?;
        let map = obj.map_into(data.uat.kernel_context())?;
        dev_info!(dev, "shmem_alloc() -> VA {:#x}\n", map.iova());
        Ok(obj)
    }
}

#[versions(AGX)]
impl GPUManager::ver {
    pub(crate) fn new(
        dev: &drm::device::Device,
        cfg: &hw::HWConfig,
    ) -> Result<Ref<GPUManager::ver>> {
        let uat = mmu::UAT::new(dev)?;
        let mut alloc = KernelAllocators {
            private: alloc::SimpleAllocator::new(dev, uat.kernel_context(), 0x20),
            shared: alloc::SimpleAllocator::new(dev, uat.kernel_context(), 0x20),
            gpu: alloc::SimpleAllocator::new(dev, uat.kernel_context(), 0x20),
        };

        let dyncfg = hw::HWDynConfig {
            uat_context_table_base: uat.context_table_base(),
        };

        let mut builder = initdata::InitDataBuilder::ver::new(&mut alloc, &cfg, &dyncfg);
        let initdata = builder.build()?;

        mem::drop(builder);

        let mut mgr = UniqueRef::try_new(GPUManager::ver {
            dev: dev.clone(),
            initialized: false,
            initdata,
            uat,
            io_mappings: Vec::new(),
            rtkit: Mutex::new(None),
            rx_channels: Mutex::new(RXChannels {
                event: channel::EventChannel::new(&mut alloc)?,
                fw_log: channel::FWLogChannel::new(&mut alloc)?,
                ktrace: channel::KTraceChannel::new(&mut alloc)?,
                stats: channel::StatsChannel::new(&mut alloc)?,
            }),
            alloc,
        })?;

        {
            let rxc = mgr.rx_channels.lock();
            let p_event = rxc.event.to_raw();
            let p_fw_log = rxc.fw_log.to_raw();
            let p_ktrace = rxc.ktrace.to_raw();
            let p_stats = rxc.stats.to_raw();
            mem::drop(rxc);

            mgr.initdata.runtime_pointers.with_mut(|raw, _inner| {
                raw.event = p_event;
                raw.fw_log = p_fw_log;
                raw.ktrace = p_ktrace;
                raw.stats = p_stats;
            });
        }

        for (i, map) in cfg.io_mappings.iter().enumerate() {
            if let Some(map) = map.as_ref() {
                mgr.iomap(i, map)?;
            }
        }

        let mgr = Ref::from(mgr);

        let rtkit = unsafe { rtkit::RTKit::<GPUManager::ver>::new(dev, None, 0, mgr.clone()) }?;

        *mgr.rtkit.lock() = Some(rtkit);

        return Ok(mgr);
    }

    fn iomap(&mut self, index: usize, map: &hw::IOMapping) -> Result {
        let off = map.base & mmu::UAT_PGMSK;
        let base = map.base - off;
        let end = (map.base + map.size + mmu::UAT_PGMSK) & !mmu::UAT_PGMSK;
        let mapping = self.uat.kernel_context().map_io(base, end - base)?;

        self.initdata.runtime_pointers.hwdata_b.with_mut(|raw, _| {
            raw.io_mappings[index] = fw::initdata::raw::IOMapping {
                phys_addr: map.base as u64,
                virt_addr: (mapping.iova() + off) as u64,
                size: mapping.size() as u32,
                range_size: map.range_size as u32,
                readwrite: map.writable as u64,
            };
        });

        self.io_mappings.try_push(mapping)?;
        Ok(())
    }
}

#[versions(AGX)]
impl GPUManager for GPUManager::ver {
    fn init(&self) -> Result {
        let initdata = self.initdata.gpu_va().get();
        let mut guard = self.rtkit.lock();
        let rtk = guard.as_mut().unwrap();

        rtk.boot()?;
        rtk.start_endpoint(EP_FIRMWARE)?;
        rtk.start_endpoint(EP_DOORBELL)?;
        rtk.send_message(EP_FIRMWARE, MSG_INIT | (initdata & INIT_DATA_MASK))?;

        Ok(())
    }

    fn alloc(&mut self) -> &mut KernelAllocators {
        &mut self.alloc
    }
}

#[versions(AGX)]
unsafe impl Sync for GPUManager::ver {}

#[versions(AGX)]
unsafe impl Send for GPUManager::ver {}
