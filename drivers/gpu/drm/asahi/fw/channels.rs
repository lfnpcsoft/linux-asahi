// SPDX-License-Identifier: GPL-2.0-only OR MIT
#![allow(missing_docs)]

//! GPU communication channels (ring buffers)

use super::types::*;

pub(crate) mod raw {
    use super::*;

    #[derive(Debug, Default)]
    #[repr(C)]
    pub(crate) struct ChannelState<'a> {
        pub(crate) read_ptr: AtomicU32,
        __pad0: Pad<0x1c>,
        pub(crate) write_ptr: AtomicU32,
        __pad1: Pad<0xc>,
        _p: PhantomData<&'a ()>,
    }

    #[derive(Debug, Default)]
    #[repr(C)]
    pub(crate) struct FWCtlChannelState<'a> {
        pub(crate) read_ptr: AtomicU32,
        __pad0: Pad<0xc>,
        pub(crate) write_ptr: AtomicU32,
        __pad1: Pad<0xc>,
        _p: PhantomData<&'a ()>,
    }
}

#[derive(Debug, Default)]
pub(crate) struct ChannelState {}

impl GPUStruct for ChannelState {
    type Raw<'a> = raw::ChannelState<'a>;
}

#[derive(Debug, Default)]
pub(crate) struct FWLogChannelState {}

impl FWLogChannelState {
    const SUB_CHANNELS: usize = 6;
}

impl GPUStruct for FWLogChannelState {
    type Raw<'a> = Array<{ Self::SUB_CHANNELS }, raw::ChannelState<'a>>;
}

#[derive(Debug, Default)]
pub(crate) struct FWCtlChannelState {}

impl GPUStruct for FWCtlChannelState {
    type Raw<'a> = raw::FWCtlChannelState<'a>;
}

pub(crate) type RunCmdQueueMsg = Array<0x30, u8>;
pub(crate) type DeviceControlMsg = Array<0x30, u8>;
pub(crate) type EventMsg = Array<0x38, u8>;
pub(crate) type FWLogMsg = Array<0xd8, u8>;
pub(crate) type KTraceMsg = Array<0x38, u8>;
pub(crate) type StatsMsg = Array<0x60, u8>;
pub(crate) type FWCtlMsg = Array<0x14, u8>;
