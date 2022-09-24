// SPDX-License-Identifier: GPL-2.0-only OR MIT
#![allow(missing_docs)]
#![allow(unused_imports)]

//! Asahi ring buffer channels

use crate::fw::channels::*;
use crate::fw::initdata::{raw, ChannelRing};
use crate::fw::types::*;
use crate::gpu;
use core::time::Duration;
use kernel::{dbg, delay::coarse_sleep, prelude::*};

pub(crate) struct RXChannel<T: RXChannelState, U: Copy + Default>
where
    for<'a> <T as GPUStruct>::Raw<'a>: Debug + Default,
{
    ring: ChannelRing<T, U>,
    // FIXME: needs feature(generic_const_exprs)
    //rptr: [u32; T::SUB_CHANNELS],
    rptr: [u32; 6],
    count: u32,
}

impl<T: RXChannelState, U: Copy + Default> RXChannel<T, U>
where
    for<'a> <T as GPUStruct>::Raw<'a>: Debug + Default,
{
    pub(crate) fn new(alloc: &mut gpu::KernelAllocators, count: usize) -> Result<RXChannel<T, U>> {
        Ok(RXChannel {
            ring: ChannelRing {
                state: alloc.shared.new_default()?,
                ring: alloc.shared.array_empty(count)?,
            },
            rptr: Default::default(),
            count: count as u32,
        })
    }

    pub(crate) fn get(&mut self, index: usize) -> Option<U> {
        self.ring.state.with(|raw, _inner| {
            let wptr = T::wptr(raw, index);
            let rptr = &mut self.rptr[index];
            if wptr == *rptr {
                None
            } else {
                let msg = self.ring.ring[*rptr as usize];
                *rptr = (*rptr + 1) % self.count;
                T::set_rptr(raw, index, *rptr);
                Some(msg)
            }
        })
    }
}

pub(crate) struct TXChannel<T: TXChannelState, U: Copy + Default>
where
    for<'a> <T as GPUStruct>::Raw<'a>: Debug + Default,
{
    ring: ChannelRing<T, U>,
    wptr: u32,
    count: u32,
}

impl<T: TXChannelState, U: Copy + Default> TXChannel<T, U>
where
    for<'a> <T as GPUStruct>::Raw<'a>: Debug + Default,
{
    pub(crate) fn new(alloc: &mut gpu::KernelAllocators, count: usize) -> Result<TXChannel<T, U>> {
        Ok(TXChannel {
            ring: ChannelRing {
                state: alloc.shared.new_default()?,
                ring: alloc.private.array_empty(count)?,
            },
            wptr: 0,
            count: count as u32,
        })
    }

    pub(crate) fn put(&mut self, msg: &U) {
        self.ring.state.with(|raw, _inner| {
            let next_wptr = (self.wptr + 1) % self.count;
            let mut rptr = T::rptr(raw);
            if next_wptr == rptr {
                pr_err!(
                    "TX ring buffer is full! Waiting... ({}, {})",
                    next_wptr,
                    rptr
                );
                // TODO: block properly on incoming messages?
                while next_wptr == rptr {
                    coarse_sleep(Duration::from_millis(8));
                    rptr = T::rptr(raw);
                }
            }
            self.ring.ring[self.wptr as usize] = *msg;
            T::set_wptr(raw, next_wptr);
            self.wptr = next_wptr;
        })
    }
}

pub(crate) struct DeviceControlChannel {
    ch: TXChannel<ChannelState, DeviceControlMsg>,
}

impl DeviceControlChannel {
    pub(crate) fn new(alloc: &mut gpu::KernelAllocators) -> Result<DeviceControlChannel> {
        Ok(DeviceControlChannel {
            ch: TXChannel::<ChannelState, DeviceControlMsg>::new(alloc, 0x100)?,
        })
    }

    pub(crate) fn to_raw(&self) -> raw::ChannelRing<ChannelState, DeviceControlMsg> {
        self.ch.ring.to_raw()
    }

    pub(crate) fn send(&mut self, msg: &DeviceControlMsg) {
        self.ch.put(msg);
    }
}

pub(crate) struct PipeChannel {
    ch: TXChannel<ChannelState, PipeMsg>,
}

impl PipeChannel {
    pub(crate) fn new(alloc: &mut gpu::KernelAllocators) -> Result<PipeChannel> {
        Ok(PipeChannel {
            ch: TXChannel::<ChannelState, PipeMsg>::new(alloc, 0x100)?,
        })
    }

    pub(crate) fn to_raw(&self) -> raw::ChannelRing<ChannelState, PipeMsg> {
        self.ch.ring.to_raw()
    }

    pub(crate) fn send(&mut self, msg: &PipeMsg) {
        self.ch.put(msg);
    }
}

pub(crate) struct EventChannel {
    ch: RXChannel<ChannelState, RawEventMsg>,
}

impl EventChannel {
    pub(crate) fn new(alloc: &mut gpu::KernelAllocators) -> Result<EventChannel> {
        Ok(EventChannel {
            ch: RXChannel::<ChannelState, RawEventMsg>::new(alloc, 0x100)?,
        })
    }

    pub(crate) fn to_raw(&self) -> raw::ChannelRing<ChannelState, RawEventMsg> {
        self.ch.ring.to_raw()
    }

    pub(crate) fn poll(&mut self) {
        while let Some(msg) = self.ch.get(0) {
            pr_info!("Event: {:?}", msg);
        }
    }
}

pub(crate) struct FWLogChannel {
    ch: RXChannel<FWLogChannelState, RawFWLogMsg>,
}

impl FWLogChannel {
    pub(crate) fn new(alloc: &mut gpu::KernelAllocators) -> Result<FWLogChannel> {
        Ok(FWLogChannel {
            ch: RXChannel::<FWLogChannelState, RawFWLogMsg>::new(alloc, 0x100)?,
        })
    }

    pub(crate) fn to_raw(&self) -> raw::ChannelRing<FWLogChannelState, RawFWLogMsg> {
        self.ch.ring.to_raw()
    }

    pub(crate) fn poll(&mut self) {
        for i in 0..=FWLogChannelState::SUB_CHANNELS - 1 {
            while let Some(msg) = self.ch.get(i) {
                pr_info!("FWLog{}: {:?}", i, msg);
            }
        }
    }
}

pub(crate) struct KTraceChannel {
    ch: RXChannel<ChannelState, RawKTraceMsg>,
}

impl KTraceChannel {
    pub(crate) fn new(alloc: &mut gpu::KernelAllocators) -> Result<KTraceChannel> {
        Ok(KTraceChannel {
            ch: RXChannel::<ChannelState, RawKTraceMsg>::new(alloc, 0x200)?,
        })
    }

    pub(crate) fn to_raw(&self) -> raw::ChannelRing<ChannelState, RawKTraceMsg> {
        self.ch.ring.to_raw()
    }

    pub(crate) fn poll(&mut self) {
        while let Some(msg) = self.ch.get(0) {
            pr_info!("KTrace: {:?}", msg);
        }
    }
}

#[versions(AGX)]
pub(crate) struct StatsChannel {
    ch: RXChannel<ChannelState, RawStatsMsg::ver>,
}

#[versions(AGX)]
impl StatsChannel::ver {
    pub(crate) fn new(alloc: &mut gpu::KernelAllocators) -> Result<StatsChannel::ver> {
        Ok(StatsChannel::ver {
            ch: RXChannel::<ChannelState, RawStatsMsg::ver>::new(alloc, 0x100)?,
        })
    }

    pub(crate) fn to_raw(&self) -> raw::ChannelRing<ChannelState, RawStatsMsg::ver> {
        self.ch.ring.to_raw()
    }

    pub(crate) fn poll(&mut self) {
        while let Some(msg) = self.ch.get(0) {
            let tag = unsafe { msg.raw.0 };
            match tag {
                0..=STATS_MAX::ver => {
                    let msg = unsafe { msg.msg };
                    pr_info!("Stats: {:?}", msg);
                }
                _ => {
                    pr_warn!("Unknown stats message: {:?}", unsafe { msg.raw });
                }
            }
        }
    }
}
