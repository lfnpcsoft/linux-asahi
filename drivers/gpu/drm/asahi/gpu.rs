// SPDX-License-Identifier: GPL-2.0-only OR MIT
#![allow(missing_docs)]
#![allow(dead_code)]

use core::any::Any;
use core::mem;

use kernel::{
    macros::versions,
    prelude::*,
    soc::apple::rtkit,
    sync::{smutex::Mutex, Guard, Ref, UniqueRef},
    PointerWrapper,
};

use crate::driver::AsahiDevice;
use crate::fw::channels::DeviceControlMsg;
use crate::fw::channels::PipeType;
use crate::{alloc, buffer, channel, event, fw, gem, hw, initdata, mmu, workqueue};

const EP_FIRMWARE: u8 = 0x20;
const EP_DOORBELL: u8 = 0x21;

const MSG_INIT: u64 = 0x81 << 48;
const INIT_DATA_MASK: u64 = (1 << 44) - 1;

const MSG_TX_DOORBELL: u64 = 0x83 << 48;
const MSG_FWCTL: u64 = 0x84 << 48;
const MSG_HALT: u64 = 0x85 << 48;

const MSG_RX_DOORBELL: u64 = 0x42 << 48;

const DOORBELL_KICKFW: u64 = 0x10;
const DOORBELL_DEVCTRL: u64 = 0x11;

pub(crate) struct KernelAllocators {
    pub(crate) private: alloc::SimpleAllocator,
    pub(crate) shared: alloc::SimpleAllocator,
    pub(crate) gpu: alloc::SimpleAllocator,
}

#[versions(AGX)]
struct RXChannels {
    event: channel::EventChannel,
    fw_log: channel::FWLogChannel,
    ktrace: channel::KTraceChannel,
    stats: channel::StatsChannel::ver,
}

struct PipeChannels {
    pub(crate) vtx: Vec<Mutex<channel::PipeChannel>>,
    pub(crate) frag: Vec<Mutex<channel::PipeChannel>>,
    pub(crate) comp: Vec<Mutex<channel::PipeChannel>>,
}

struct TXChannels {
    pub(crate) device_control: channel::DeviceControlChannel,
}

const NUM_PIPES: usize = 4;

#[versions(AGX)]
pub(crate) struct GPUManager {
    dev: AsahiDevice,
    initialized: bool,
    pub(crate) initdata: fw::types::GPUObject<fw::initdata::InitData::ver>,
    uat: mmu::UAT,
    alloc: Mutex<KernelAllocators>,
    io_mappings: Vec<mmu::Mapping>,
    rtkit: Mutex<Option<rtkit::RTKit<GPUManager::ver>>>,
    rx_channels: Mutex<RXChannels::ver>,
    tx_channels: Mutex<TXChannels>,
    pipes: PipeChannels,
    event_manager: Ref<event::EventManager>,
    buffer_mgr: buffer::BufferManager,
}

pub(crate) trait GPUManager: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn init(&self) -> Result;
    fn test(&self) -> Result;
    fn alloc(&self) -> Guard<'_, Mutex<KernelAllocators>>;
    fn new_vm(&self) -> Result<mmu::VM>;
    fn bind_vm(&self, vm: &mmu::VM) -> Result<mmu::VMBind>;
    fn submit_batch(&self, batch: workqueue::WorkQueueBatch<'_>) -> Result;
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

        let mut obj = gem::new_kernel_object(dev, size)?;
        obj.vmap()?;
        let iova = obj.map_into(data.uat.kernel_vm())?;
        dev_info!(dev, "shmem_alloc() -> VA {:#x}\n", iova);
        Ok(obj)
    }
}

#[versions(AGX)]
impl GPUManager::ver {
    pub(crate) fn new(dev: &AsahiDevice, cfg: &hw::HWConfig) -> Result<Ref<GPUManager::ver>> {
        let uat = mmu::UAT::new(dev)?;
        let mut alloc = KernelAllocators {
//             private: alloc::SimpleAllocator::new(dev, uat.kernel_vm(), 0x20, mmu::PROT_FW_PRIV_RW),
            private: alloc::SimpleAllocator::new(dev, uat.kernel_vm(), 0x20, mmu::PROT_FW_SHARED_RW),
            shared: alloc::SimpleAllocator::new(dev, uat.kernel_vm(), 0x20, mmu::PROT_FW_SHARED_RW),
            gpu: alloc::SimpleAllocator::new(dev, uat.kernel_vm(), 0x20, mmu::PROT_GPU_FW_SHARED_RW),
        };

        let dyncfg = hw::HWDynConfig {
            uat_context_table_base: uat.context_table_base(),
        };

        let mut builder = initdata::InitDataBuilder::ver::new(&mut alloc, &cfg, &dyncfg);
        let initdata = builder.build()?;

        mem::drop(builder);

        let mut pipes = PipeChannels {
            vtx: Vec::new(),
            frag: Vec::new(),
            comp: Vec::new(),
        };

        for _i in 0..=NUM_PIPES - 1 {
            pipes
                .vtx
                .try_push(Mutex::new(channel::PipeChannel::new(&mut alloc)?))?;
            pipes
                .frag
                .try_push(Mutex::new(channel::PipeChannel::new(&mut alloc)?))?;
            pipes
                .comp
                .try_push(Mutex::new(channel::PipeChannel::new(&mut alloc)?))?;
        }

        let event_manager = Ref::try_new(event::EventManager::new(&mut alloc)?)?;

        let mut mgr = UniqueRef::try_new(GPUManager::ver {
            dev: dev.clone(),
            initialized: false,
            initdata,
            uat,
            io_mappings: Vec::new(),
            rtkit: Mutex::new(None),
            rx_channels: Mutex::new(RXChannels::ver {
                event: channel::EventChannel::new(&mut alloc, event_manager.clone())?,
                fw_log: channel::FWLogChannel::new(&mut alloc)?,
                ktrace: channel::KTraceChannel::new(&mut alloc)?,
                stats: channel::StatsChannel::ver::new(&mut alloc)?,
            }),
            tx_channels: Mutex::new(TXChannels {
                device_control: channel::DeviceControlChannel::new(&mut alloc)?,
            }),
            pipes,
            event_manager,
            buffer_mgr: buffer::BufferManager::new()?,
            alloc: Mutex::new(alloc),
        })?;

        {
            let txc = mgr.tx_channels.lock();
            let p_device_control = txc.device_control.to_raw();
            mem::drop(txc);

            let rxc = mgr.rx_channels.lock();
            let p_event = rxc.event.to_raw();
            let p_fw_log = rxc.fw_log.to_raw();
            let p_ktrace = rxc.ktrace.to_raw();
            let p_stats = rxc.stats.to_raw();
            mem::drop(rxc);

            mgr.initdata.runtime_pointers.with_mut(|raw, _inner| {
                raw.device_control = p_device_control;
                raw.event = p_event;
                raw.fw_log = p_fw_log;
                raw.ktrace = p_ktrace;
                raw.stats = p_stats;
            });
        }

        let mut p_pipes: Vec<fw::initdata::raw::PipeChannels> = Vec::new();

        for ((v, f), c) in mgr
            .pipes
            .vtx
            .iter()
            .zip(&mgr.pipes.frag)
            .zip(&mgr.pipes.comp)
        {
            p_pipes.try_push(fw::initdata::raw::PipeChannels {
                vtx: v.lock().to_raw(),
                frag: f.lock().to_raw(),
                comp: c.lock().to_raw(),
            })?;
        }

        mgr.initdata.runtime_pointers.with_mut(|raw, _inner| {
            for (i, p) in p_pipes.into_iter().enumerate() {
                raw.pipes[i].vtx = p.vtx;
                raw.pipes[i].frag = p.frag;
                raw.pipes[i].comp = p.comp;
            }
        });

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
        let mapping = self.uat.kernel_vm().map_io(base, end - base)?;

        self.initdata.runtime_pointers.hwdata_b.with_mut(|raw, _| {
            raw.io_mappings[index] = fw::initdata::raw::IOMapping {
                phys_addr: map.base as u64,
                virt_addr: (mapping.iova() + off) as u64,
                size: map.size as u32,
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
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn init(&self) -> Result {
        let initdata = self.initdata.gpu_va().get();
        let mut guard = self.rtkit.lock();
        let rtk = guard.as_mut().unwrap();

        rtk.boot()?;
        rtk.start_endpoint(EP_FIRMWARE)?;
        rtk.start_endpoint(EP_DOORBELL)?;
        rtk.send_message(EP_FIRMWARE, MSG_INIT | (initdata & INIT_DATA_MASK))?;

        self.tx_channels
            .lock()
            .device_control
            .send(&DeviceControlMsg::Initialize);

        rtk.send_message(EP_DOORBELL, MSG_TX_DOORBELL | DOORBELL_DEVCTRL)?;
        Ok(())
    }

    fn test(&self) -> Result {
        Ok(())
    }

    fn alloc(&self) -> Guard<'_, Mutex<KernelAllocators>> {
        self.alloc.lock()
    }

    fn new_vm(&self) -> Result<mmu::VM> {
        self.uat.new_vm()
    }

    fn bind_vm(&self, vm: &mmu::VM) -> Result<mmu::VMBind> {
        self.uat.bind(vm)
    }

    fn submit_batch(&self, batch: workqueue::WorkQueueBatch<'_>) -> Result {
        let pipe_type = batch.pipe_type();
        let pipes = match pipe_type {
            PipeType::Vertex => &self.pipes.vtx,
            PipeType::Fragment => &self.pipes.frag,
            PipeType::Compute => &self.pipes.comp,
        };

        /* TODO: need try_lock()
        let pipe = 'outer: loop {
            for p in pipes.iter() {
                if let Ok(guard) = p.try_lock() {
                    break 'outer guard;
                }
            }
            break pipes[0].lock();
        };
        */

        let index: usize = 0;
        let mut pipe = pipes[index].lock();

        batch.submit(&mut pipe)?;

        let mut guard = self.rtkit.lock();
        let rtk = guard.as_mut().unwrap();
        rtk.send_message(
            EP_DOORBELL,
            MSG_TX_DOORBELL | pipe_type as u64 | ((index as u64) << 2),
        )?;

        Ok(())
    }
}

#[versions(AGX)]
unsafe impl Sync for GPUManager::ver {}

#[versions(AGX)]
unsafe impl Send for GPUManager::ver {}
