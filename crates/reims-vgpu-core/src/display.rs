//! Backend-independent display and cursor state.

/// Complete cursor position and visibility snapshot for a host adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CursorPosition {
    pub x: u16,
    pub y: u16,
    pub visible: bool,
}

/// Borrowed complete cursor glyph for a host adapter.
#[derive(Clone, Copy, Debug)]
pub struct CursorGlyph<'a> {
    pub width: u16,
    pub height: u16,
    pub hot_x: u16,
    pub hot_y: u16,
    pub pixels: &'a [u32],
}

/// Hardware cursor state with atomic glyph publication.
#[derive(Clone, Debug, Default)]
pub struct CursorState {
    show: bool,
    x: u16,
    y: u16,
    width: u16,
    height: u16,
    hot_x: u16,
    hot_y: u16,
    /// Host cursor pixels as `0xAARRGGBB`.
    pixels: Vec<u32>,
    glyph_ready: bool,
}

impl CursorState {
    pub fn initially_visible() -> Self {
        Self {
            show: true,
            ..Self::default()
        }
    }

    pub fn position(&self) -> CursorPosition {
        CursorPosition {
            x: self.x,
            y: self.y,
            visible: self.show,
        }
    }

    pub fn set_position(&mut self, x: u16, y: u16) {
        self.x = x;
        self.y = y;
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.show = visible;
    }

    pub fn publish_glyph(
        &mut self,
        width: u16,
        height: u16,
        hot_x: u16,
        hot_y: u16,
        pixels: Vec<u32>,
    ) {
        self.width = width;
        self.height = height;
        self.hot_x = hot_x;
        self.hot_y = hot_y;
        self.pixels = pixels;
        self.glyph_ready = true;
    }

    pub fn glyph(&self) -> Option<CursorGlyph<'_>> {
        (self.glyph_ready && !self.pixels.is_empty()).then_some(CursorGlyph {
            width: self.width,
            height: self.height,
            hot_x: self.hot_x,
            hot_y: self.hot_y,
            pixels: &self.pixels,
        })
    }
}

/// Guest display shared-state handshake and online-redrive progress.
#[derive(Clone, Debug, Default)]
pub struct DisplayHandshake {
    shared_gpa: u64,
    display_index: u32,
    online_acked: bool,
    online_tries: u32,
    poll_ctr: u32,
    /// Timestamp of the refresh grid slot this generation last consumed, or
    /// `None` before the first one. See [`DisplayRefreshCadence`].
    last_refresh_us: Option<u64>,
}

/// Published display shared page and the interrupt bit that belongs to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DisplaySharedPage {
    pub gpa: u64,
    pub display_index: u32,
}

pub const DISPLAY_SHARED_PENDING_OFFSET: u64 = 0x100;
pub const DISPLAY_SHARED_ENABLE_MASK_OFFSET: u64 = 0x104;
pub const DISPLAY_PRESENT_EVENT_MASK: u32 = 1 << 1;
pub const DISPLAY_ONLINE_EVENT_MASK: u32 = 1 << 2;
/// The guest's `signalVBLInterrupt` class, armed by its `enableVBLInterrupt`.
///
/// This is what paces the guest's compositor: its display driver holds every
/// swap until a vertical blank arrives, so a device that never raises this
/// class never receives a Present, however much the guest renders.
pub const DISPLAY_VBL_EVENT_MASK: u32 = 1 << 0;

/// The interval between refresh pulses, derived from the refresh rate the
/// device advertises in its timing elements.
///
/// The two must agree. macOS paces CoreAnimation to the advertised rate, and
/// the guest paces to what is *delivered*, so a hand-written interval that
/// disagrees with the advertised rate is a rate the guest latches and the
/// device never meant: an 8 ms interval against an advertised 120 Hz delivers
/// 125. Deriving it here is what stops the two spellings existing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DisplayRefreshCadence {
    interval_us: u64,
}

impl DisplayRefreshCadence {
    /// # Panics
    ///
    /// If `advertised_hz` is zero, which no timing element may advertise.
    pub const fn from_advertised_hz(advertised_hz: u32) -> Self {
        assert!(advertised_hz != 0, "a display advertises a nonzero refresh");
        Self {
            interval_us: 1_000_000 / advertised_hz as u64,
        }
    }

    pub const fn interval_us(self) -> u64 {
        self.interval_us
    }

    /// The grid slot a pulse at `now_us` would consume, or `None` while the
    /// previous slot still stands.
    ///
    /// The claimed timestamp advances on a **fixed interval grid**
    /// (`last + interval`) rather than to `now_us`. Resetting to `now` lets
    /// poll jitter shift the cadence phase permanently: a poll landing slightly
    /// late pushes the next deadline out another whole interval, so the
    /// delivered rate aliases down — and when the poll spacing sits just under
    /// the interval it takes two polls per delivery and halves. Advancing by
    /// exactly one interval keeps delivery phase-locked and lets a late poll
    /// catch up. A stall of two intervals or more resyncs to `now_us` so the
    /// catch-up cannot become a burst.
    pub const fn claim(self, last_us: Option<u64>, now_us: u64) -> Option<u64> {
        let Some(last) = last_us else {
            return Some(now_us);
        };
        let gap = now_us.saturating_sub(last);
        if gap < self.interval_us {
            return None;
        }
        Some(if gap >= 2 * self.interval_us {
            now_us
        } else {
            last + self.interval_us
        })
    }
}

/// One planned display refresh pulse.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DisplayRefreshNotification {
    /// No shared page is published, or the guest has not acknowledged the
    /// online offer yet. Refresh belongs to a display the guest is driving.
    NotOnline,
    /// The guest has armed neither refresh class.
    ///
    /// Reported **without consuming a grid slot**: a tick that finds nothing
    /// armed must not push the phase out, or the first pulse after the guest
    /// arms arrives a whole interval late for no reason.
    Disabled(DisplaySharedPage),
    /// The previous slot still stands.
    TooSoon(DisplaySharedPage),
    Deliver {
        page: DisplaySharedPage,
        pending: u32,
        interrupt_bit: u32,
        /// The grid slot this pulse consumes, to be handed back to
        /// [`DisplayHandshake::record_refresh_pulse`] once it is delivered.
        claimed_us: u64,
        vbl: bool,
        present: bool,
        stale_online_removed: bool,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DisplayRefreshNotificationError {
    EnableMaskUnreadable(DisplaySharedPage),
    PendingUnreadable(DisplaySharedPage),
    DisplayIndexOutOfRange(DisplaySharedPage),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DisplayPresentNotification {
    Disabled(DisplaySharedPage),
    Deliver {
        page: DisplaySharedPage,
        pending: u32,
        interrupt_bit: u32,
        stale_online_removed: bool,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DisplayPresentNotificationError {
    SharedPageUnavailable,
    EnableMaskUnreadable(DisplaySharedPage),
    PendingUnreadable(DisplaySharedPage),
    DisplayIndexOutOfRange(DisplaySharedPage),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DisplayOnlineNotification {
    Idle,
    WaitingForEnable(DisplaySharedPage),
    Deliver {
        page: DisplaySharedPage,
        pending: u32,
        interrupt_bit: u32,
        first_pulse: bool,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DisplayOnlineNotificationError {
    EnableMaskUnreadable(DisplaySharedPage),
    PendingUnreadable(DisplaySharedPage),
    DisplayIndexOutOfRange(DisplaySharedPage),
}

/// One online-handshake poll after lifecycle admission and cadence decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayOnlinePoll {
    /// No shared page is published, or this generation was already acknowledged.
    Idle,
    /// This generation exhausted its permitted online pulses.
    Exhausted(DisplaySharedPage),
    /// Read the enable mask. `pulse_if_enabled` carries the retry cadence decision.
    Inspect {
        page: DisplaySharedPage,
        poll_count: u32,
        pulse_if_enabled: bool,
        first_pulse: bool,
    },
}

impl DisplayHandshake {
    /// Publish a new shared-state page and restart the online handshake.
    /// Returns whether the previous handshake had reached online.
    pub fn reinitialize(&mut self, display_index: u32, shared_gpa: u64) -> bool {
        let was_online = self.online_acked;
        self.display_index = display_index;
        self.shared_gpa = shared_gpa;
        self.online_acked = false;
        self.online_tries = 0;
        self.poll_ctr = 0;
        self.last_refresh_us = None;
        was_online
    }

    /// Return the currently published shared page, if any.
    pub fn shared_page(&self) -> Option<DisplaySharedPage> {
        (self.shared_gpa != 0).then_some(DisplaySharedPage {
            gpa: self.shared_gpa,
            display_index: self.display_index,
        })
    }

    pub fn is_online(&self) -> bool {
        self.online_acked
    }

    /// Complete the online handshake for the current shared-page generation.
    pub fn acknowledge_online(&mut self) {
        self.online_acked = true;
    }

    /// Plan the guest-visible completion event for one successfully presented
    /// frame. Memory values are supplied by the host adapter; this owner alone
    /// decides whether the class is enabled and whether an acknowledged stale
    /// ONLINE bit may be removed.
    pub fn plan_present_notification(
        &self,
        enable_mask: Option<u32>,
        pending: Option<u32>,
    ) -> Result<DisplayPresentNotification, DisplayPresentNotificationError> {
        let page = self
            .shared_page()
            .ok_or(DisplayPresentNotificationError::SharedPageUnavailable)?;
        let enable_mask =
            enable_mask.ok_or(DisplayPresentNotificationError::EnableMaskUnreadable(page))?;
        if enable_mask & DISPLAY_PRESENT_EVENT_MASK == 0 {
            return Ok(DisplayPresentNotification::Disabled(page));
        }
        let pending = pending.ok_or(DisplayPresentNotificationError::PendingUnreadable(page))?;
        let interrupt_bit = 1u32.checked_shl(page.display_index).ok_or(
            DisplayPresentNotificationError::DisplayIndexOutOfRange(page),
        )?;
        let stale_online_removed = self.online_acked && pending & DISPLAY_ONLINE_EVENT_MASK != 0;
        let pending = if stale_online_removed {
            pending & !DISPLAY_ONLINE_EVENT_MASK
        } else {
            pending
        } | DISPLAY_PRESENT_EVENT_MASK;
        Ok(DisplayPresentNotification::Deliver {
            page,
            pending,
            interrupt_bit,
            stale_online_removed,
        })
    }

    /// Plan one refresh pulse for a display the guest is driving.
    ///
    /// Memory values are supplied by the host adapter; this owner alone decides
    /// which classes are armed, whether the cadence permits a pulse, and whether
    /// an acknowledged stale ONLINE bit may be removed. A stale ONLINE bit that
    /// is re-delivered makes the guest re-run its connection-change path and
    /// rebuild its overlays, so it is dropped here rather than carried forward.
    pub fn plan_refresh_notification(
        &self,
        cadence: DisplayRefreshCadence,
        now_us: u64,
        enable_mask: Option<u32>,
        pending: Option<u32>,
    ) -> Result<DisplayRefreshNotification, DisplayRefreshNotificationError> {
        let Some(page) = self.shared_page().filter(|_| self.online_acked) else {
            return Ok(DisplayRefreshNotification::NotOnline);
        };
        let enable_mask =
            enable_mask.ok_or(DisplayRefreshNotificationError::EnableMaskUnreadable(page))?;
        let vbl = enable_mask & DISPLAY_VBL_EVENT_MASK != 0;
        let present = enable_mask & DISPLAY_PRESENT_EVENT_MASK != 0;
        if !vbl && !present {
            return Ok(DisplayRefreshNotification::Disabled(page));
        }
        let Some(claimed_us) = cadence.claim(self.last_refresh_us, now_us) else {
            return Ok(DisplayRefreshNotification::TooSoon(page));
        };
        let pending = pending.ok_or(DisplayRefreshNotificationError::PendingUnreadable(page))?;
        let interrupt_bit = 1u32.checked_shl(page.display_index).ok_or(
            DisplayRefreshNotificationError::DisplayIndexOutOfRange(page),
        )?;
        let stale_online_removed = pending & DISPLAY_ONLINE_EVENT_MASK != 0;
        let mut next = if stale_online_removed {
            pending & !DISPLAY_ONLINE_EVENT_MASK
        } else {
            pending
        };
        if vbl {
            next |= DISPLAY_VBL_EVENT_MASK;
        }
        if present {
            next |= DISPLAY_PRESENT_EVENT_MASK;
        }
        Ok(DisplayRefreshNotification::Deliver {
            page,
            pending: next,
            interrupt_bit,
            claimed_us,
            vbl,
            present,
            stale_online_removed,
        })
    }

    /// Consume the grid slot a delivered pulse claimed.
    pub fn record_refresh_pulse(&mut self, claimed_us: u64) {
        self.last_refresh_us = Some(claimed_us);
    }

    /// Plan one idempotent ONLINE offer. A generation remains eligible on
    /// every host poll until the guest acknowledges it; the API declares no
    /// retry limit or cadence that would authorize abandoning the handshake.
    pub fn plan_online_notification(
        &self,
        enable_mask: Option<u32>,
        pending: Option<u32>,
    ) -> Result<DisplayOnlineNotification, DisplayOnlineNotificationError> {
        let Some(page) = self.shared_page() else {
            return Ok(DisplayOnlineNotification::Idle);
        };
        if self.online_acked {
            return Ok(DisplayOnlineNotification::Idle);
        }
        let enable_mask =
            enable_mask.ok_or(DisplayOnlineNotificationError::EnableMaskUnreadable(page))?;
        if enable_mask & DISPLAY_ONLINE_EVENT_MASK == 0 {
            return Ok(DisplayOnlineNotification::WaitingForEnable(page));
        }
        let pending = pending.ok_or(DisplayOnlineNotificationError::PendingUnreadable(page))?
            | DISPLAY_ONLINE_EVENT_MASK;
        let interrupt_bit = 1u32
            .checked_shl(page.display_index)
            .ok_or(DisplayOnlineNotificationError::DisplayIndexOutOfRange(page))?;
        Ok(DisplayOnlineNotification::Deliver {
            page,
            pending,
            interrupt_bit,
            first_pulse: self.online_tries == 0,
        })
    }

    /// Advance one online poll and decide whether the caller may inspect/pulse.
    pub fn begin_online_poll(&mut self, max_tries: u32, retry_divisor: u32) -> DisplayOnlinePoll {
        let Some(page) = self.shared_page() else {
            return DisplayOnlinePoll::Idle;
        };
        if self.online_acked {
            return DisplayOnlinePoll::Idle;
        }
        if self.online_tries >= max_tries {
            return DisplayOnlinePoll::Exhausted(page);
        }

        self.poll_ctr = self.poll_ctr.wrapping_add(1);
        DisplayOnlinePoll::Inspect {
            page,
            poll_count: self.poll_ctr,
            pulse_if_enabled: self.online_tries == 0 || self.poll_ctr.is_multiple_of(retry_divisor),
            first_pulse: self.online_tries == 0,
        }
    }

    /// Record a pulse admitted by [`Self::begin_online_poll`].
    pub fn record_online_pulse(&mut self) {
        self.online_tries = self.online_tries.saturating_add(1);
    }

    #[cfg(any(test, feature = "test-fixtures"))]
    pub fn online_progress(&self) -> (u32, u32) {
        (self.online_tries, self.poll_ctr)
    }

    #[cfg(any(test, feature = "test-fixtures"))]
    pub fn online_tries(&self) -> u32 {
        self.online_tries
    }

    #[cfg(any(test, feature = "test-fixtures"))]
    pub fn poll_count(&self) -> u32 {
        self.poll_ctr
    }

    #[cfg(any(test, feature = "test-fixtures"))]
    pub fn set_online_progress(&mut self, online_tries: u32, poll_ctr: u32) {
        self.online_tries = online_tries;
        self.poll_ctr = poll_ctr;
    }

    #[cfg(any(test, feature = "test-fixtures"))]
    pub fn set_online_tries(&mut self, online_tries: u32) {
        self.online_tries = online_tries;
    }

    #[cfg(any(test, feature = "test-fixtures"))]
    pub fn set_poll_count(&mut self, poll_ctr: u32) {
        self.poll_ctr = poll_ctr;
    }

    #[cfg(any(test, feature = "test-fixtures"))]
    pub fn withdraw_online_ack(&mut self) {
        self.online_acked = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HZ: u32 = 120;

    fn online() -> DisplayHandshake {
        let mut display = DisplayHandshake::default();
        display.reinitialize(0, 0x1000);
        display.acknowledge_online();
        display
    }

    /// The advertised rate and the delivered interval are one value. A device
    /// that advertises 120 Hz and delivers 125 gets a guest paced at 125.
    #[test]
    fn the_refresh_interval_is_the_advertised_rate_and_not_a_second_number() {
        assert_eq!(
            DisplayRefreshCadence::from_advertised_hz(120).interval_us(),
            8_333
        );
        assert_eq!(
            DisplayRefreshCadence::from_advertised_hz(60).interval_us(),
            16_666
        );
    }

    /// A late poll must not push the phase out, or the delivered rate aliases
    /// down toward half the grid and the guest latches that instead.
    #[test]
    fn a_late_poll_catches_the_grid_up_instead_of_shifting_its_phase() {
        let cadence = DisplayRefreshCadence::from_advertised_hz(HZ);
        let interval = cadence.interval_us();
        assert_eq!(cadence.claim(None, 1_000), Some(1_000));
        assert_eq!(cadence.claim(Some(1_000), 1_000 + interval - 1), None);
        // One interval late: the next slot is the grid's, not the poll's.
        assert_eq!(
            cadence.claim(Some(1_000), 1_000 + interval + 10),
            Some(1_000 + interval)
        );
        // Two intervals or more is a stall, and resyncs rather than bursting.
        assert_eq!(
            cadence.claim(Some(1_000), 1_000 + 2 * interval),
            Some(1_000 + 2 * interval)
        );
    }

    /// A tick that finds nothing armed must leave the grid alone, or the first
    /// pulse after the guest arms arrives a whole interval late.
    #[test]
    fn a_disarmed_refresh_class_consumes_no_grid_slot() {
        let cadence = DisplayRefreshCadence::from_advertised_hz(HZ);
        let display = online();
        assert!(matches!(
            display.plan_refresh_notification(cadence, 5_000, Some(0), Some(0)),
            Ok(DisplayRefreshNotification::Disabled(_))
        ));
        assert!(matches!(
            display.plan_refresh_notification(
                cadence,
                5_000,
                Some(DISPLAY_VBL_EVENT_MASK),
                Some(0)
            ),
            Ok(DisplayRefreshNotification::Deliver {
                claimed_us: 5_000,
                ..
            })
        ));
    }

    #[test]
    fn a_refresh_pulse_carries_exactly_the_classes_the_guest_armed() {
        let cadence = DisplayRefreshCadence::from_advertised_hz(HZ);
        let display = online();
        let Ok(DisplayRefreshNotification::Deliver {
            pending,
            vbl,
            present,
            interrupt_bit,
            ..
        }) = display.plan_refresh_notification(cadence, 1, Some(DISPLAY_VBL_EVENT_MASK), Some(0))
        else {
            panic!("an armed VBL class delivers");
        };
        assert!(vbl && !present);
        assert_eq!(pending, DISPLAY_VBL_EVENT_MASK);
        assert_eq!(interrupt_bit, 1);

        let Ok(DisplayRefreshNotification::Deliver { pending, .. }) = display
            .plan_refresh_notification(
                cadence,
                1,
                Some(DISPLAY_VBL_EVENT_MASK | DISPLAY_PRESENT_EVENT_MASK),
                Some(0),
            )
        else {
            panic!("both armed classes deliver");
        };
        assert_eq!(pending, DISPLAY_VBL_EVENT_MASK | DISPLAY_PRESENT_EVENT_MASK);
    }

    /// Re-delivering an acknowledged ONLINE bit makes the guest re-run its
    /// connection-change path and rebuild its overlays.
    #[test]
    fn a_refresh_pulse_drops_the_acknowledged_online_bit_it_finds_pending() {
        let cadence = DisplayRefreshCadence::from_advertised_hz(HZ);
        let Ok(DisplayRefreshNotification::Deliver {
            pending,
            stale_online_removed,
            ..
        }) = online().plan_refresh_notification(
            cadence,
            1,
            Some(DISPLAY_VBL_EVENT_MASK),
            Some(DISPLAY_ONLINE_EVENT_MASK),
        )
        else {
            panic!("an armed VBL class delivers");
        };
        assert!(stale_online_removed);
        assert_eq!(pending & DISPLAY_ONLINE_EVENT_MASK, 0);
        assert_eq!(pending, DISPLAY_VBL_EVENT_MASK);
    }

    /// Refresh belongs to a display the guest is driving. Pulsing before the
    /// online handshake completes interrupts a guest that has not asked.
    #[test]
    fn refresh_waits_for_the_online_acknowledgement_and_for_a_shared_page() {
        let cadence = DisplayRefreshCadence::from_advertised_hz(HZ);
        assert_eq!(
            DisplayHandshake::default().plan_refresh_notification(
                cadence,
                1,
                Some(DISPLAY_VBL_EVENT_MASK),
                Some(0)
            ),
            Ok(DisplayRefreshNotification::NotOnline)
        );
        let mut unacked = DisplayHandshake::default();
        unacked.reinitialize(0, 0x1000);
        assert_eq!(
            unacked.plan_refresh_notification(cadence, 1, Some(DISPLAY_VBL_EVENT_MASK), Some(0)),
            Ok(DisplayRefreshNotification::NotOnline)
        );
    }

    /// A republished shared page is a new display generation, and the old
    /// generation's grid slot must not hold the new one's first pulse back.
    #[test]
    fn republishing_the_shared_page_restarts_the_refresh_grid() {
        let cadence = DisplayRefreshCadence::from_advertised_hz(HZ);
        let mut display = online();
        display.record_refresh_pulse(1_000_000);
        assert!(matches!(
            display.plan_refresh_notification(
                cadence,
                1_000_001,
                Some(DISPLAY_VBL_EVENT_MASK),
                Some(0)
            ),
            Ok(DisplayRefreshNotification::TooSoon(_))
        ));
        display.reinitialize(0, 0x2000);
        display.acknowledge_online();
        assert!(matches!(
            display.plan_refresh_notification(
                cadence,
                1_000_001,
                Some(DISPLAY_VBL_EVENT_MASK),
                Some(0)
            ),
            Ok(DisplayRefreshNotification::Deliver {
                claimed_us: 1_000_001,
                ..
            })
        ));
    }

    #[test]
    fn glyph_publication_exposes_one_complete_snapshot() {
        let mut cursor = CursorState::initially_visible();
        assert!(cursor.position().visible);
        assert!(cursor.glyph().is_none());

        cursor.publish_glyph(2, 3, 1, 2, vec![0xff00_0000; 6]);
        let glyph = cursor.glyph().unwrap();
        assert_eq!((glyph.width, glyph.height), (2, 3));
        assert_eq!((glyph.hot_x, glyph.hot_y), (1, 2));
        assert_eq!(glyph.pixels.len(), 6);
    }

    #[test]
    fn shared_state_reinitialization_withdraws_every_prior_online_witness() {
        let mut display = DisplayHandshake::default();
        display.reinitialize(1, 0x1000);
        display.acknowledge_online();
        display.set_online_progress(3, 9);
        assert!(display.reinitialize(2, 0x4000));
        assert_eq!(
            display.shared_page(),
            Some(DisplaySharedPage {
                display_index: 2,
                gpa: 0x4000
            })
        );
        assert!(!display.is_online());
        assert_eq!(display.online_progress(), (0, 0));
    }

    #[test]
    fn online_poll_owns_admission_cadence_and_retry_progress() {
        let mut display = DisplayHandshake::default();
        assert_eq!(display.begin_online_poll(3, 4), DisplayOnlinePoll::Idle);

        display.reinitialize(2, 0x4000);
        assert_eq!(
            display.begin_online_poll(3, 4),
            DisplayOnlinePoll::Inspect {
                page: DisplaySharedPage {
                    gpa: 0x4000,
                    display_index: 2
                },
                poll_count: 1,
                pulse_if_enabled: true,
                first_pulse: true,
            }
        );
        display.record_online_pulse();
        assert!(matches!(
            display.begin_online_poll(3, 4),
            DisplayOnlinePoll::Inspect {
                pulse_if_enabled: false,
                first_pulse: false,
                ..
            }
        ));
        display.set_online_progress(3, 8);
        assert!(matches!(
            display.begin_online_poll(3, 4),
            DisplayOnlinePoll::Exhausted(_)
        ));
    }

    #[test]
    fn replacement_online_plan_never_abandons_an_unacknowledged_generation() {
        let mut display = DisplayHandshake::default();
        display.reinitialize(3, 0x4000);
        for pulse in 0..1_000 {
            let planned = display
                .plan_online_notification(
                    Some(DISPLAY_ONLINE_EVENT_MASK),
                    Some(DISPLAY_PRESENT_EVENT_MASK),
                )
                .unwrap();
            assert_eq!(
                planned,
                DisplayOnlineNotification::Deliver {
                    page: DisplaySharedPage {
                        gpa: 0x4000,
                        display_index: 3,
                    },
                    pending: DISPLAY_ONLINE_EVENT_MASK | DISPLAY_PRESENT_EVENT_MASK,
                    interrupt_bit: 1 << 3,
                    first_pulse: pulse == 0,
                }
            );
            display.record_online_pulse();
        }
        display.acknowledge_online();
        assert_eq!(
            display
                .plan_online_notification(Some(DISPLAY_ONLINE_EVENT_MASK), Some(0))
                .unwrap(),
            DisplayOnlineNotification::Idle
        );
    }

    #[test]
    fn present_notification_preserves_only_a_live_online_event() {
        let mut display = DisplayHandshake::default();
        display.reinitialize(3, 0x4000);
        assert_eq!(
            display
                .plan_present_notification(
                    Some(DISPLAY_PRESENT_EVENT_MASK),
                    Some(DISPLAY_ONLINE_EVENT_MASK),
                )
                .unwrap(),
            DisplayPresentNotification::Deliver {
                page: DisplaySharedPage {
                    gpa: 0x4000,
                    display_index: 3,
                },
                pending: DISPLAY_ONLINE_EVENT_MASK | DISPLAY_PRESENT_EVENT_MASK,
                interrupt_bit: 1 << 3,
                stale_online_removed: false,
            }
        );

        display.acknowledge_online();
        assert_eq!(
            display
                .plan_present_notification(
                    Some(DISPLAY_PRESENT_EVENT_MASK),
                    Some(DISPLAY_ONLINE_EVENT_MASK),
                )
                .unwrap(),
            DisplayPresentNotification::Deliver {
                page: DisplaySharedPage {
                    gpa: 0x4000,
                    display_index: 3,
                },
                pending: DISPLAY_PRESENT_EVENT_MASK,
                interrupt_bit: 1 << 3,
                stale_online_removed: true,
            }
        );
    }

    #[test]
    fn present_notification_requires_the_guest_enable_and_exact_pending_word() {
        let mut display = DisplayHandshake::default();
        display.reinitialize(u32::BITS, 0x4000);
        let page = display.shared_page().unwrap();
        assert_eq!(
            display.plan_present_notification(Some(0), None),
            Ok(DisplayPresentNotification::Disabled(page))
        );
        assert_eq!(
            display.plan_present_notification(Some(DISPLAY_PRESENT_EVENT_MASK), None),
            Err(DisplayPresentNotificationError::PendingUnreadable(page))
        );
        assert_eq!(
            display.plan_present_notification(Some(DISPLAY_PRESENT_EVENT_MASK), Some(0)),
            Err(DisplayPresentNotificationError::DisplayIndexOutOfRange(
                page
            ))
        );
    }
}
