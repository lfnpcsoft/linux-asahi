// SPDX-License-Identifier: GPL-2.0-only OR MIT
#![allow(missing_docs)]
#![allow(dead_code)]

//! GPU work queues

use super::event;
use super::types::*;
use crate::trivial_gpustruct;

#[derive(Debug)]
#[repr(u32)]
pub(crate) enum CommandType {
    RunVertex = 0,
    RunFragment = 1,
    RunBlitter = 2,
    RunCompute = 3,
    Barrier = 4,
    InitBufferManager = 6,
}

pub(crate) trait Command : GPUStruct {}

pub(crate) mod raw {
    use super::*;

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct Barrier {
        pub(crate) tag: CommandType,
        pub(crate) stamp: GPUWeakPointer<u32>,
        pub(crate) wait_value: u32,
        pub(crate) event: u32,
        pub(crate) stamp_self: u32,
        pub(crate) uuid: u32,
        pub(crate) unk: u32,
    }

    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct GPUContextData {
        unk_0: u16,
        unk_2: Array<0x3, u8>,
        unk_5: u8,
        unk_6: Array<0x18, u8>,
        unk_1e: u8,
        unk_1f: u8,
        unk_20: Array<0x3, u8>,
        unk_23: u8,
        unk_24: Array<0x1c, u8>,
    }

    impl Default for GPUContextData {
        fn default() -> Self {
            Self {
                unk_0: 0xffff,
                unk_2: Default::default(),
                unk_5: 1,
                unk_6: Default::default(),
                unk_1e: 0xff,
                unk_1f: 0,
                unk_20: Default::default(),
                unk_23: 2,
                unk_24: Default::default(),
            }
        }
    }

    #[derive(Debug, Default)]
    #[repr(C)]
    pub(crate) struct RingState {
        pub(crate) gpu_doneptr: AtomicU32,
        __pad0: Pad<0xc>,
        pub(crate) unk_10: AtomicU32,
        __pad1: Pad<0xc>,
        pub(crate) unk_20: AtomicU32,
        __pad2: Pad<0xc>,
        pub(crate) gpu_rptr: AtomicU32,
        __pad3: Pad<0xc>,
        pub(crate) cpu_wptr: AtomicU32,
        __pad4: Pad<0xc>,
        pub(crate) rb_size: u32,
        __pad5: Pad<0xc>,
    }

    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct Priority(u32, u32, U64, u32, u32, u32);

    pub(crate) const PRIORITY: [Priority; 4] = [
        Priority(0, 0, U64(0xffff_ffff_ffff_0000), 1, 0, 1),
        Priority(1, 1, U64(0xffff_ffff_0000_0000), 0, 0, 0),
        Priority(2, 2, U64(0xffff_0000_0000_0000), 0, 0, 2),
        Priority(3, 3, U64(0x0000_0000_0000_0000), 0, 0, 3),
    ];

    impl Default for Priority {
        fn default() -> Priority {
            PRIORITY[2]
        }
    }

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct QueueInfo<'a> {
        pub(crate) state: GPUPointer<'a, super::RingState>,
        pub(crate) ring: GPUPointer<'a, &'a [u64]>,
        pub(crate) notifier_list: GPUPointer<'a, event::NotifierList>,
        pub(crate) gpu_buf: GPUPointer<'a, &'a [u8]>,
        pub(crate) gpu_rptr1: AtomicU32,
        pub(crate) gpu_rptr2: AtomicU32,
        pub(crate) gpu_rptr3: AtomicU32,
        pub(crate) event_id: AtomicI32,
        pub(crate) priority: Priority,
        pub(crate) unk_4c: i32,
        pub(crate) uuid: u32,
        pub(crate) unk_54: i32,
        pub(crate) unk_58: U64,
        pub(crate) busy: AtomicU32,
        pub(crate) __pad: Pad<0x20>,
        pub(crate) unk_84_state: AtomicU32,
        pub(crate) unk_88: u32,
        pub(crate) unk_8c: u32,
        pub(crate) unk_90: u32,
        pub(crate) unk_94: u32,
        pub(crate) pending: AtomicU32,
        pub(crate) unk_9c: u32,
        pub(crate) gpu_context: GPUPointer<'a, super::GPUContextData>,
        pub(crate) unk_a8: U64,
    }
}

trivial_gpustruct!(Barrier);
trivial_gpustruct!(GPUContextData);
trivial_gpustruct!(RingState);

impl Command for Barrier {}

#[derive(Debug)]
pub(crate) struct QueueInfo {
    pub(crate) state: GPUObject<RingState>,
    pub(crate) ring: GPUArray<u64>,
    pub(crate) notifier_list: GPUObject<event::NotifierList>,
    pub(crate) gpu_buf: GPUArray<u8>,
    pub(crate) gpu_context: GPUObject<GPUContextData>,
}

impl GPUStruct for QueueInfo {
    type Raw<'a> = raw::QueueInfo<'a>;
}
