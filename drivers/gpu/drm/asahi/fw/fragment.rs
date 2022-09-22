// SPDX-License-Identifier: GPL-2.0-only OR MIT
#![allow(missing_docs)]
#![allow(dead_code)]

//! GPU fragment jobs

use super::types::*;
use super::{event, job, workqueue};
use crate::{buffer, fw, microseq};
use kernel::sync::Ref;

pub(crate) mod raw {
    use super::*;

    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct ClearPipelineBinding {
        pub(crate) pipeline_bind: U64,
        pub(crate) address: U64,
    }

    #[derive(Debug, Clone, Copy, Default)]
    #[repr(C)]
    pub(crate) struct StorePipelineBinding {
        pub(crate) unk_0: U64,
        pub(crate) unk_8: u32,
        pub(crate) pipeline_bind: u32,
        pub(crate) unk_10: u32,
        pub(crate) address: u32,
        pub(crate) unk_18: u32,
        pub(crate) unk_1c_padding: u32,
    }

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct ArrayAddr {
        pub(crate) ptr: U64,
        pub(crate) unk_padding: U64,
    }

    #[versions(AGX)]
    #[derive(Debug, Clone, Copy)]
    #[repr(C)]
    pub(crate) struct AuxFBInfo {
        pub(crate) unk1: u32,
        pub(crate) unk2: u32,
        pub(crate) width: u32,
        pub(crate) height: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk3: U64,
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct JobParameters1 {
        pub(crate) unk_0: U64,
        pub(crate) clear_pipeline: ClearPipelineBinding,
        pub(crate) unk_18: U64,
        pub(crate) scissor_array: U64,
        pub(crate) depth_bias_array: U64,
        pub(crate) aux_fb: AuxFBInfo::ver,
        pub(crate) depth_dimensions: U64,
        pub(crate) unk_48: U64,
        pub(crate) depth_flags: U64,
        pub(crate) depth_buffer_ptr1: U64,
        pub(crate) depth_buffer_ptr2: U64,
        pub(crate) stencil_buffer_ptr1: U64,
        pub(crate) stencil_buffer_ptr2: U64,
        pub(crate) unk_68: Array<0xc, U64>,
        pub(crate) tvb_tilemap: U64,
        pub(crate) tvb_heapmeta_addr: U64,
        pub(crate) unk_e8: U64,
        pub(crate) tvb_heapmeta_addr2: U64,
        pub(crate) unk_f8: U64,
        pub(crate) aux_fb_ptr: U64,
        pub(crate) unk_108: Array<0x6, U64>,
        pub(crate) pipeline_base: U64,
        pub(crate) unk_140: U64,
        pub(crate) unk_148: U64,
        pub(crate) unk_150: U64,
        pub(crate) unk_158: U64,
        pub(crate) unk_160_padding: Array<0x1e0, u8>,

        #[ver(V < V13_0B4)]
        pub(crate) __pad0: Pad<0x8>,
    }

    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct JobParameters2 {
        pub(crate) store_pipeline_bind: u32,
        pub(crate) store_pipeline_addr: u32,
        pub(crate) unk_8: u32,
        pub(crate) unk_c: u32,
        pub(crate) uuid1: u32,
        pub(crate) uuid2: u32,
        pub(crate) unk_18: U64,
        pub(crate) tile_blocks_y: u16,
        pub(crate) tile_blocks_x: u16,
        pub(crate) unk_24: u32,
        pub(crate) tile_counts: u32,
        pub(crate) unk_2c: u32,
        pub(crate) depth_clear_val1: F32,
        pub(crate) stencil_clear_val1: u8,
        pub(crate) unk_35: u8,
        pub(crate) unk_36: u16,
        pub(crate) unk_38: u32,
        pub(crate) unk_3c: u32,
        pub(crate) unk_40: u32,
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct JobParameters3 {
        pub(crate) unk_40_padding: Array<0xac, u8>,
        pub(crate) depth_bias_array: ArrayAddr,
        pub(crate) scissor_array: ArrayAddr,
        pub(crate) unk_110: U64,
        pub(crate) unk_118: U64,
        pub(crate) unk_120: [U64; 0x25],
        pub(crate) unk_reload_pipeline: ClearPipelineBinding,
        pub(crate) unk_258: U64,
        pub(crate) unk_260: U64,
        pub(crate) unk_268: U64,
        pub(crate) unk_270: U64,
        pub(crate) reload_pipeline: ClearPipelineBinding,
        pub(crate) depth_flags: U64,
        pub(crate) unk_290: U64,
        pub(crate) depth_buffer_ptr1: U64,
        pub(crate) unk_2a0: U64,
        pub(crate) unk_2a8: U64,
        pub(crate) depth_buffer_ptr2: U64,
        pub(crate) depth_buffer_ptr3: U64,
        pub(crate) unk_2c0: U64,
        pub(crate) stencil_buffer_ptr1: U64,
        pub(crate) unk_2d0: U64,
        pub(crate) unk_2d8: U64,
        pub(crate) stencil_buffer_ptr2: U64,
        pub(crate) stencil_buffer_ptr3: U64,
        pub(crate) unk_2f0: Array<3, U64>,
        pub(crate) aux_fb_unk0: u32,
        pub(crate) unk_30c: u32,
        pub(crate) aux_fb: AuxFBInfo::ver,
        pub(crate) unk_320_padding: Array<0x10, u8>,
        pub(crate) unk_partial_store_pipeline: StorePipelineBinding,
        pub(crate) partial_store_pipeline: StorePipelineBinding,
        pub(crate) depth_clear_val2: F32,
        pub(crate) stencil_clear_val2: u8,
        pub(crate) unk_375: u8,
        pub(crate) unk_376: u16,
        pub(crate) unk_378: u32,
        pub(crate) unk_37c: u32,
        pub(crate) unk_380: U64,
        pub(crate) unk_388: U64,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_390_0: U64,

        pub(crate) depth_dimensions: U64,
    }

    #[versions(AGX)]
    #[derive(Debug)]
    #[repr(C)]
    pub(crate) struct RunFragment<'a> {
        pub(crate) tag: workqueue::CommandType,

        #[ver(V >= V13_0B4)]
        pub(crate) counter: U64,

        pub(crate) vm_slot: u32,
        pub(crate) unk_8: u32,
        pub(crate) microsequence: GPUPointer<'a, &'a [u8]>,
        pub(crate) microsequence_size: u32,
        pub(crate) notifier: GPUPointer<'a, event::Notifier>,
        pub(crate) buffer: GPUPointer<'a, fw::buffer::Info::ver>,
        pub(crate) scene: GPUPointer<'a, fw::buffer::Scene::ver>,
        pub(crate) unk_emptybuf_addr: U64,
        pub(crate) tvb_tilemap: GPUPointer<'a, &'a [u8]>,
        pub(crate) unk_40: U64,
        pub(crate) unk_48: u32,
        pub(crate) tile_blocks_y: u16,
        pub(crate) tile_blocks_x: u16,
        pub(crate) unk_50: U64,
        pub(crate) unk_58: U64,
        pub(crate) uuid1: u32,
        pub(crate) uuid2: u32,
        pub(crate) unk_68: U64,
        pub(crate) tile_count: U64,
        pub(crate) params_1: JobParameters1::ver,
        pub(crate) params_2: JobParameters2,
        pub(crate) params_3: JobParameters3::ver,
        pub(crate) unk_758_flag: u32,
        pub(crate) unk_75c_flag: u32,
        pub(crate) unk_buf: Array<0x110, u8>,
        pub(crate) busy_flag: u32,
        pub(crate) tvb_overflow_count: u32,
        pub(crate) unk_878: u32,
        pub(crate) encoder_params: job::EncoderParams,
        pub(crate) unk_pointee: u32,
        pub(crate) meta: job::JobMeta,
        pub(crate) unk_914: u32,
        pub(crate) unk_918: U64,
        pub(crate) unk_920: u32,
        pub(crate) unk_924: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_928_0: u32,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_928_4: u8,

        #[ver(V >= V13_0B4)]
        pub(crate) ts_flag: u8,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_5e6: u16,

        #[ver(V >= V13_0B4)]
        pub(crate) unk_5e8: Array<0x20, u8>,
        // Alignment - handled by allocator
        //pad_928: [u8; 0x18],
    }
}

#[versions(AGX)]
#[derive(Debug)]
pub(crate) struct RunFragment {
    pub(crate) notifier: Ref<GPUObject<event::Notifier>>,
    pub(crate) scene: Ref<buffer::Scene::ver>,
    pub(crate) micro_seq: microseq::MicroSequence,
}

#[versions(AGX)]
impl GPUStruct for RunFragment::ver {
    type Raw<'a> = raw::RunFragment::ver<'a>;
}
