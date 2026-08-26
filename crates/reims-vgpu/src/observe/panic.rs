//! What the device says when it panics.
//!
//! Every C ABI entry point wraps its body in [`crate::unwind_safe`], so a panic
//! anywhere in this crate unwinds to the boundary and becomes a return value
//! instead of tearing down QEMU. That is the right behaviour — a guest that
//! finds one bad path should not lose the whole VM — but it makes a panic the
//! single largest thing this device can drop: not one refused record, but the
//! entire call, with whatever guest work was in flight behind it.
//!
//! Such a drop was invisible. `unwind_safe` discarded the payload and returned
//! `REIMS_VGPU_QEMU_ERR_PANIC`, and **no shim compares against that code** — the
//! constant is declared in `reims_vgpu_qemu_abi.h` and read by nobody. So the
//! largest possible loss of guest work reached neither the fail log, nor a
//! counter, nor the caller. Only the default hook's stderr line survived, mixed
//! into QEMU's own output and absent from every artifact a session collects.
//!
//! This module is the missing half. [`arm`] installs a hook that records where
//! the panic was raised; [`report`] turns the caught payload into an always-on
//! fail line naming the entry point, the source location and the message.
//!
//! # Why the location needs a hook
//!
//! `catch_unwind` hands back the payload and nothing else — the `&str` or
//! `String` passed to `panic!`, with no file, no line, and no indication of
//! which of a hundred `unwrap`s produced it. `Location` exists only inside the
//! hook, which runs at the raise site before unwinding starts. Capturing it
//! there into a thread-local is the only way to have it at the catch.
//!
//! The previous hook is chained rather than replaced, so the default stderr
//! report is still printed and `RUST_BACKTRACE` still works. No backtrace is
//! folded into the log line: the sink splits records on whitespace, a backtrace
//! is multi-line by nature, and `file:line:col` plus the entry point already
//! names one statement.
//!
//! # Why the hook is armed lazily
//!
//! From `unwind_safe` itself, behind a `Once`, so arming cannot be missed by an
//! entry point that runs before `reims_vgpu_qemu_device_create` — several do,
//! and `reims_vgpu_qemu_abi_version` is called before any device exists. The
//! steady-state cost is one acquire load on a completed `Once`, against entry
//! points that run at MMIO and drain granularity rather than per draw.

use crate::observe::{decline_display, Decline, Emit};
use std::any::Any;
use std::cell::RefCell;
use std::sync::Once;

thread_local! {
    /// Where the most recent panic on this thread was raised, as recorded by
    /// the hook [`arm`] installs.
    ///
    /// Thread-local because a panic unwinds on the thread that raised it, so
    /// the hook and the matching [`report`] are always the same thread and no
    /// cross-thread panic can overwrite another's site between the two. A
    /// global would be a race in exactly the situation this exists to describe.
    ///
    /// Both accesses go through `try_with`, never `with`. A thread-local that
    /// is already being destroyed makes `with` *panic*, and a panic raised
    /// inside a panic hook aborts the process — so the one place `with` would
    /// be wrong is a thread unwinding at teardown, which is precisely a moment
    /// this code exists to survive. Losing the location there costs an
    /// `at=unknown`; taking it costs the VM.
    static SITE: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Guards the one-time hook installation.
static HOOK: Once = Once::new();

/// Install the location-capturing panic hook, once per process.
///
/// Idempotent and cheap to re-call; see the module doc for why it is armed from
/// the catch site rather than from device creation.
pub(crate) fn arm() {
    HOOK.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let at = info
                .location()
                .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()));
            let _ = SITE.try_with(|s| *s.borrow_mut() = at);
            previous(info);
        }));
    });
}

/// A panic that unwound out of a C ABI entry point and was turned into a status
/// code.
///
/// One slug, with the entry point as a field rather than as part of the slug:
/// the *check* here is the same one everywhere — "the body did not return" — and
/// splitting it per entry would give twenty-two slugs that all mean the same
/// thing while [`Emit::fail_once`]'s latch already separates instances by
/// discriminant.
pub(crate) struct AbiPanic {
    /// The C symbol whose body unwound.
    pub(crate) entry: &'static str,
    /// `file:line:col` of the raise, or `unknown` when the hook saw no location
    /// (a panic raised before [`arm`] ran, or one carrying no `Location`).
    pub(crate) at: String,
    /// The panic payload rendered as text, whitespace-flattened.
    pub(crate) msg: String,
}

impl Decline for AbiPanic {
    fn slug(&self) -> &'static str {
        "abi_entry_panicked"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("entry", self.entry.to_string()),
            ("at", self.at.clone()),
            ("msg", self.msg.clone()),
        ]
    }
}

decline_display!(AbiPanic);

/// Replacement for any whitespace run inside a panic message.
///
/// The sink writes one record per line and readers split records on spaces, so
/// a raw message would break both: `assert_eq!` renders multi-line, and every
/// standard slice message contains spaces. Flattening keeps the message legible
/// and the record parseable.
///
/// There is deliberately no length cap. A truncated message is the one that
/// cuts off the value that explains the panic, and no `panic!` in this crate
/// formats an unbounded string — the sink's own flood detector, not a cut here,
/// is what bounds a runaway emitter.
const WHITESPACE_STANDIN: char = '_';

/// Render a caught panic payload as a single whitespace-free token.
///
/// `catch_unwind` yields `Box<dyn Any>`; `panic!("literal")` puts a `&'static
/// str` inside and `panic!("{fmt}")` a `String`. Anything else — a payload from
/// `panic_any` — has no textual form, and saying so beats an empty field.
fn payload_text(payload: &(dyn Any + Send)) -> String {
    let raw = payload
        .downcast_ref::<&'static str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "<non-string panic payload>".to_string());
    flatten(&raw)
}

/// Collapse every whitespace run in `text` to a single [`WHITESPACE_STANDIN`].
fn flatten(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_space = false;
    for c in text.chars() {
        if c.is_whitespace() {
            if !in_space {
                out.push(WHITESPACE_STANDIN);
                in_space = true;
            }
        } else {
            out.push(c);
            in_space = false;
        }
    }
    out
}

/// Emit the always-on record for a panic that unwound out of `entry`, and count
/// it.
///
/// Latched per distinct `(entry, raise site)` so a panic on a per-frame path
/// cannot bury the log, and counted on every occurrence so the magnitude the
/// latch hides is still readable: `abi_entry_panicked` in `store_routes` is the
/// rate, the fail line is the identity. Reading either as the other is the
/// mistake `AGENTS.md` warns about.
pub(crate) fn report(entry: &'static str, payload: &(dyn Any + Send)) {
    let at = SITE
        .try_with(|s| s.borrow_mut().take())
        .ok()
        .flatten()
        .unwrap_or_else(|| "unknown".to_string());
    let discriminant = reims_vgpu_core::fnv::fold_bytes(
        reims_vgpu_core::fnv::fold_bytes(reims_vgpu_core::fnv::FNV_OFFSET_BASIS, entry.as_bytes()),
        at.as_bytes(),
    );
    let decline = AbiPanic {
        entry,
        at,
        msg: payload_text(payload),
    };
    Emit::decline("abi_entry_panic", &decline).fail_once(discriminant);
}
