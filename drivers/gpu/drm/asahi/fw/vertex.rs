// SPDX-License-Identifier: GPL-2.0-only OR MIT
#![allow(missing_docs)]
#![allow(dead_code)]

//! GPU communication channels (ring buffers)

use super::types::*;
use super::{event, job, workqueue};
use crate::{buffer, fw, microseq, mmu};
use kernel::sync::Ref;

pub(crate) mod raw {
    use super::*;

    #[derive(Debug, Default, Copy, Clone)]
    #[repr(C)]
    pub(crate) struct TilingParameters {
        pub(crate) size1: u32,
        pub(crate) unk_4: u32,
        pub(crate) unk_8: u32,
        pub(crate) x_max: u16,
        pub(crate) y_max: u16,
        pub(crate) tile_count: u32,
        pub(crate) x_blocks: u32,
        pub(crate) y_blocks: u32,
        pub(crate) size2: u32,
        pub(crate) size3: u32,
        pub(crate) unk_24: u32,
        pub(crate) unk_28: u32,
    }

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct JobParameters1<'a> {
        pub(crate) unk_0: U64,
        pub(crate) unk_8: u32,
        pub(crate) unk_c: u32,
        pub(crate) tvb_tilemap: GPUPointer<'a, &'a [u8]>,
        pub(crate) unkptr_18: U64,
        pub(crate) tvb_something: GPUWeakPointer<[u8]>,
        pub(crate) tvb_heapmeta: GPUPointer<'a, &'a [u8]>,
        pub(crate) iogpu_unk_54: u32,
        pub(crate) iogpu_unk_55: u32,
        pub(crate) iogpu_unk_56: U64,
        pub(crate) unk_40: U64,
        pub(crate) unk_48: U64,
        pub(crate) unk_50: U64,
        pub(crate) tvb_heapmeta_2: GPUPointer<'a, &'a [u8]>,
        pub(crate) unk_60: U64,
        pub(crate) unk_68: U64,
        pub(crate) preempt_buf1: GPUPointer<'a, Array<0x540, u8>>,
        pub(crate) preempt_buf2: GPUPointer<'a, Array<0x280, u8>>,
        pub(crate) unk_80: U64,
        pub(crate) preempt_buf3: GPUPointer<'a, Array<0x20, u8>>,
        pub(crate) encoder_addr: U64,
        pub(crate) unk_98: Array<2, U64>,
        pub(crate) unk_a8: U64,
        pub(crate) unk_b0: Array<6, U64>,
        pub(crate) pipeline_base: U64,
        pub(crate) unk_e8: U64,
        pub(crate) unk_f0: U64,
        pub(crate) unk_f8: U64,
        pub(crate) unk_100: Array<3, U64>,
        pub(crate) unk_118: u32,
    }

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct JobParameters2<'a> {
        pub(crate) unk_480: Array<4, u32>,
        pub(crate) unk_498: U64,
        pub(crate) unk_4a0: u32,
        pub(crate) preempt_buf1: GPUPointer<'a, Array<0x540, u8>>,
        pub(crate) unk_4ac: u32,
        pub(crate) unk_4b0: U64,
        pub(crate) unk_4b8: u32,
        pub(crate) unk_4bc: U64,
        pub(crate) unk_4c4_padding: Array<0x48, u8>,
        pub(crate) unk_50c: u32,
        pub(crate) unk_510: U64,
        pub(crate) unk_518: U64,
        pub(crate) unk_520: U64,
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct RunVertex<'a> {
        pub(crate) tag: workqueue::CommandType,

        #[ver(V >= V13_0B4)]
        pub(crate) counter: U64,

        pub(crate) vm_slot: u32,
        pub(crate) unk_8: u32,
        pub(crate) notifier: GPUPointer<'a, event::Notifier>,
        pub(crate) buffer_slot: u32,
        pub(crate) unk_1c: u32,
        pub(crate) buffer: GPUWeakPointer<fw::buffer::Info::ver>,
        pub(crate) scene: GPUPointer<'a, fw::buffer::Scene::ver>,
        pub(crate) unk_buffer_buf: GPUWeakPointer<[u8]>,
        pub(crate) unk_34: u32,
        pub(crate) job_params1: JobParameters1<'a>,
        pub(crate) unk_154: Array<0x268, u8>,
        pub(crate) tiling_params: TilingParameters,
        pub(crate) unk_3e8: Array<0x74, u8>,
        pub(crate) tvb_something: GPUWeakPointer<[u8]>,
        pub(crate) tvb_size: U64,
        pub(crate) microsequence: GPUPointer<'a, &'a [u8]>,
        pub(crate) microsequence_size: u32,
        pub(crate) fragment_stamp_slot: u32,
        pub(crate) stamp_value: EventValue,
        pub(crate) unk_pointee: u32,
        pub(crate) unk_pad: u32,
        pub(crate) job_params2: JobParameters2<'a>,
        pub(crate) encoder_params: job::EncoderParams<'a>,
        pub(crate) unk_568: u32,
        pub(crate) unk_56c: u32,
        pub(crate) meta: job::JobMeta,
        pub(crate) unk_5c4: u32,
        pub(crate) unk_5c8: u32,
        pub(crate) unk_5cc: u32,
        pub(crate) unk_5d0: u32,
        pub(crate) unk_5d4: u8,
        pub(crate) pad_5d5: Array<3, u8>,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_5e0: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_5e4: u8,

        #[ver(V >= V13_0B4)]
        pub(crate) ts_flag: u8,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_5e6: u16,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_5e8: [u8; 0x18],

        pub(crate) pad_5d8: Pad<0x8>,
        // Alignment - handled by allocator
        //#[ver(V >= V13_0B4)]
        //pad_5e0: Pad<0x18>,
    }
}

#[versions(AGX)]
#[derive(Debug)]
pub(crate) struct RunVertex {
    pub(crate) notifier: Ref<GPUObject<event::Notifier>>,
    pub(crate) scene: Ref<buffer::Scene::ver>,
    pub(crate) micro_seq: microseq::MicroSequence,
    pub(crate) vm_bind: mmu::VMBind,
}

#[versions(AGX)]
impl GPUStruct for RunVertex::ver {
    type Raw<'a> = raw::RunVertex::ver<'a>;
}

#[versions(AGX)]
impl workqueue::Command for RunVertex::ver {}
