// SPDX-License-Identifier: GPL-2.0-only OR MIT
#![allow(missing_docs)]
#![allow(unused_imports)]
#![allow(dead_code)]

//! Asahi ring buffer channels

use crate::fw::event::*;
use crate::fw::initdata::raw;
use crate::fw::types::*;
use crate::{gpu, slotalloc, workqueue};
use core::cmp;
use core::sync::atomic::Ordering;
use kernel::sync::Ref;
use kernel::{dbg, prelude::*};

const NUM_EVENTS: u32 = 128;

pub(crate) struct EventInner {
    stamp: *const AtomicU32,
    gpu_stamp: GPUWeakPointer<Stamp>,
    gpu_fw_stamp: GPUWeakPointer<FWStamp>,
}

pub(crate) type Token = slotalloc::SlotToken;
pub(crate) type Event = slotalloc::Guard<EventInner>;

#[derive(Eq, PartialEq, Copy, Clone, Debug)]
pub(crate) struct EventValue(u32);

impl EventValue {
    pub(crate) fn stamp(&self) -> u32 {
        self.0
    }

    pub(crate) fn counter(&self) -> u32 {
        self.0 >> 8
    }

    pub(crate) fn next(&self) -> EventValue {
        EventValue(self.0.wrapping_add(0x100))
    }

    pub(crate) fn increment(&mut self) {
        self.0 = self.0.wrapping_add(0x100);
    }

    pub(crate) fn delta(&self, other: &EventValue) -> i32 {
        self.0.wrapping_sub(other.0) as i32
    }
}

impl PartialOrd for EventValue {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EventValue {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.delta(other).cmp(&0)
    }
}

impl EventInner {
    pub(crate) fn stamp_pointer(&self) -> GPUWeakPointer<Stamp> {
        self.gpu_stamp
    }

    pub(crate) fn fw_stamp_pointer(&self) -> GPUWeakPointer<FWStamp> {
        self.gpu_fw_stamp
    }

    pub(crate) fn current(&self) -> EventValue {
        // SAFETY: The pointer is always valid as constructed in
        // EventManager below, and outside users cannot construct
        // new EventInners, nor move or copy them, and Guards as
        // returned by the SlotAllocator hold a reference to the
        // SlotAllocator containing the EventManagerInner, which
        // keeps the GPUObject the stamp is contained within alive.
        EventValue(unsafe { &*self.stamp }.load(Ordering::Acquire))
    }
}

impl slotalloc::SlotItem for EventInner {
    type Owner = EventManagerInner;

    fn release(&mut self, owner: &mut Self::Owner, slot: u32) {
        pr_info!("EventManager: Released slot {}", slot);
        owner.owners[slot as usize] = None;
    }
}

pub(crate) struct EventManagerInner {
    stamps: GPUArray<Stamp>,
    fw_stamps: GPUArray<FWStamp>,
    owners: Vec<Option<Ref<workqueue::WorkQueue>>>,
}

pub(crate) struct EventManager {
    alloc: slotalloc::SlotAllocator<EventInner>,
}

impl EventManager {
    pub(crate) fn new(alloc: &mut gpu::KernelAllocators) -> Result<EventManager> {
        let mut owners = Vec::new();
        for _i in 0..(NUM_EVENTS as usize) {
            owners.try_push(None)?;
        }
        let inner = EventManagerInner {
            stamps: alloc.shared.array_empty(NUM_EVENTS as usize)?,
            fw_stamps: alloc.private.array_empty(NUM_EVENTS as usize)?,
            owners,
        };

        Ok(EventManager {
            alloc: slotalloc::SlotAllocator::new(
                NUM_EVENTS,
                inner,
                |inner: &mut EventManagerInner, slot| EventInner {
                    stamp: &inner.stamps[slot as usize].0,
                    gpu_stamp: inner.stamps.weak_item_pointer(slot as usize),
                    gpu_fw_stamp: inner.fw_stamps.weak_item_pointer(slot as usize),
                },
            )?,
        })
    }

    pub(crate) fn get(
        &self,
        token: Option<Token>,
        owner: Ref<workqueue::WorkQueue>,
    ) -> Result<Event> {
        pr_info!("EventManager::get");
        let ev = self.alloc.get_inner(token, |inner, ev| {
            pr_info!(
                "EventManager: Registered owner {:p} on slot {}",
                &*owner,
                ev.slot()
            );
            inner.owners[ev.slot() as usize] = Some(owner);
            Ok(())
        })?;
        Ok(ev)
    }

    pub(crate) fn signal(&self, slot: u32) {
        match self
            .alloc
            .with_inner(|inner| match inner.owners[slot as usize].as_ref() {
                Some(owner) => Some(owner.clone()),
                None => None,
            }) {
            Some(owner) => {
                owner.signal();
            }
            None => {
                pr_err!("Received event for empty slot {}", slot);
            }
        }
    }
}

unsafe impl Send for EventManager {}
unsafe impl Sync for EventManager {}
