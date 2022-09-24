// SPDX-License-Identifier: GPL-2.0-only OR MIT
#![allow(missing_docs)]
#![allow(dead_code)]

//! GPU tiler buffer control structures

use super::types::*;
use super::workqueue;
use crate::{trivial_gpustruct, no_debug};
use kernel::sync::{smutex::Mutex, Ref};

pub(crate) mod raw {
    use super::*;

    #[derive(Debug, Default)]
    #[repr(C)]
    pub(crate) struct BlockControl {
        pub(crate) total: AtomicU32,
        pub(crate) wptr: AtomicU32,
        pub(crate) unk: AtomicU32,
        pub(crate) pad: Pad<0x34>,
    }

    #[derive(Debug, Default)]
    #[repr(C)]
    pub(crate) struct Counter {
        pub(crate) count: AtomicU32,
        __pad: Pad<0x3c>,
    }

    #[derive(Debug, Default)]
    #[repr(C)]
    pub(crate) struct Stats {
        pub(crate) gpu_0: AtomicU32,
        pub(crate) gpu_4: AtomicU32,
        pub(crate) gpu_8: AtomicU32,
        pub(crate) gpu_c: AtomicU32,
        pub(crate) __pad0: Pad<0x10>,
        pub(crate) cpu_flag: AtomicU32,
        pub(crate) __pad1: Pad<0x1c>,
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct Info<'a> {
        pub(crate) gpu_counter: u32,
        pub(crate) unk_4: u32,
        pub(crate) last_id: i32,
        pub(crate) cur_id: i32,
        pub(crate) unk_10: u32,
        pub(crate) gpu_counter2: u32,
        pub(crate) unk_18: u32,

        #[ver(V < V13_0B4)]
        pub(crate) unk_1c: u32,

        pub(crate) page_list: GPUPointer<'a, &'a [u32]>,
        pub(crate) page_list_size: u32,
        pub(crate) page_count: AtomicU32,
        pub(crate) unk_30: u32,
        pub(crate) block_count: AtomicU32,
        pub(crate) unk_38: u32,
        pub(crate) block_list: GPUPointer<'a, &'a [u32]>,
        pub(crate) block_ctl: GPUPointer<'a, super::BlockControl>,
        pub(crate) last_page: AtomicU32,
        pub(crate) gpu_page_ptr1: u32,
        pub(crate) gpu_page_ptr2: u32,
        pub(crate) unk_58: u32,
        pub(crate) block_size: u32,
        pub(crate) unk_60: U64,
        pub(crate) counter: GPUPointer<'a, super::Counter>,
        pub(crate) unk_70: u32,
        pub(crate) unk_74: u32,
        pub(crate) unk_78: u32,
        pub(crate) unk_7c: u32,
        pub(crate) unk_80: u32,
        pub(crate) unk_84: u32,
        pub(crate) unk_88: u32,
        pub(crate) unk_8c: u32,
        pub(crate) unk_90: Array<0x30, u8>,
    }

    #[derive(Debug, Default)]
    #[repr(C)]
    pub(crate) struct PreemptBuffer {
        pub(crate) part_3: Array<0x20, u8>,
        pub(crate) part_2: Array<0x280, u8>,
        pub(crate) part_1: Array<0x540, u8>,
    }

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct Scene<'a> {
        pub(crate) unk_0: U64,
        pub(crate) unk_8: U64,
        pub(crate) unk_10: U64,
        pub(crate) user_buffer: GPUPointer<'a, &'a [u8]>,
        pub(crate) unk_20: u32,
        pub(crate) stats: GPUPointer<'a, super::Stats>,
        pub(crate) unk_2c: u32,
        pub(crate) unk_30: U64,
        pub(crate) unk_38: U64,
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct InitBuffer {
        pub(crate) tag: workqueue::CommandType,
        pub(crate) context_id: u32,
        pub(crate) buffer_mgr_slot: u32,
        pub(crate) unk_c: u32,
        pub(crate) unk_10: u32,
        pub(crate) buffer_mgr: GPUWeakPointer<super::Info::ver>,
        pub(crate) stamp_value: u32,
    }
}

trivial_gpustruct!(BlockControl);
trivial_gpustruct!(Counter);
trivial_gpustruct!(Stats);
trivial_gpustruct!(PreemptBuffer);

#[versions(AGX)]
#[derive(Debug)]
pub(crate) struct Info {
    pub(crate) block_ctl: GPUObject<BlockControl>,
    pub(crate) counter: GPUObject<Counter>,
    pub(crate) page_list: GPUArray<u32>,
    pub(crate) block_list: GPUArray<u32>,
}

#[versions(AGX)]
impl GPUStruct for Info::ver {
    type Raw<'a> = raw::Info::ver<'a>;
}

#[versions(AGX)]
pub(crate) struct Scene {
    pub(crate) user_buffer: GPUArray<u8>,
    pub(crate) stats: GPUObject<Stats>,
    pub(crate) buffer: Ref<Mutex<crate::buffer::BufferInner::ver>>,
    pub(crate) tvb_heapmeta: GPUArray<u8>,
    pub(crate) tvb_tilemap: GPUArray<u8>,
    pub(crate) preempt_buf: GPUObject<PreemptBuffer>,
    pub(crate) seq_buf: GPUArray<u64>,
}

#[versions(AGX)]
no_debug!(Scene::ver);

#[versions(AGX)]
impl GPUStruct for Scene::ver {
    type Raw<'a> = raw::Scene<'a>;
}

#[versions(AGX)]
pub(crate) struct InitBuffer {
    pub(crate) buffer: Ref<Mutex<crate::buffer::BufferInner::ver>>,
}

#[versions(AGX)]
no_debug!(InitBuffer::ver);

#[versions(AGX)]
impl workqueue::Command for InitBuffer::ver {}

#[versions(AGX)]
impl GPUStruct for InitBuffer::ver {
    type Raw<'a> = raw::InitBuffer::ver;
}
