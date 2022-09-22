// SPDX-License-Identifier: GPL-2.0-only OR MIT
#![allow(missing_docs)]
#![allow(unused_imports)]

//! Asahi ring buffer channels

use crate::fw::channels::*;
use crate::fw::initdata::{raw, ChannelRing};
use crate::fw::types::*;
use crate::gpu;
use crate::object::GPUStruct;
use kernel::{dbg, prelude::*};

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
                let msg = self.ring.ring.as_slice()[*rptr as usize];
                *rptr = (*rptr + 1) % self.count;
                T::set_rptr(raw, index, *rptr);
                Some(msg)
            }
        })
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

pub(crate) struct StatsChannel {
    ch: RXChannel<ChannelState, RawStatsMsg>,
}

impl StatsChannel {
    pub(crate) fn new(alloc: &mut gpu::KernelAllocators) -> Result<StatsChannel> {
        Ok(StatsChannel {
            ch: RXChannel::<ChannelState, RawStatsMsg>::new(alloc, 0x100)?,
        })
    }

    pub(crate) fn to_raw(&self) -> raw::ChannelRing<ChannelState, RawStatsMsg> {
        self.ch.ring.to_raw()
    }

    pub(crate) fn poll(&mut self) {
        while let Some(msg) = self.ch.get(0) {
            let tag = unsafe { msg.raw.0 };
            match tag {
                0..=STATS_MAX => {
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
