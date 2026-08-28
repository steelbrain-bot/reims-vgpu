//! Every environment variable this device reads, and the one way they parse.
//!
//! # Why they all live here
//!
//! An override is a rule the operator states from outside the process, so it has
//! the same problem the ABI header has: nothing in the toolchain finds the second
//! copy. A variable read at its point of use is invisible to everyone who does not
//! already know it exists, two sites spelling one variable's "off" differently is
//! a divergence no test can see, and a name that gets renamed in one place keeps
//! working in the other. Naming them here makes the set greppable and makes the
//! parse shared.
//!
//! # What an override may do
//!
//! **An override may only narrow what this device does. It may never widen it.**
//!
//! A switch can turn a rail *off* that the host was capable of running, because
//! that is a statement about policy and is always satisfiable. A switch may not
//! turn a rail *on* that the host reported it cannot run: capability is measured
//! from the device, and a variable that could override the measurement would turn
//! "this host has no such extension" into a crash or, worse, undefined behavior
//! inside a driver. Every gate stays where it is; a switch can only add a reason
//! to refuse.
//!
//! That rule is why [`Switch::On`] exists but is nowhere sufficient on its own.
//! Reading it is how a caller notices an operator asked for something the host
//! cannot give and says so, rather than ignoring the request in silence.

/// Guest RAM reaches the GPU as a host-pointer import over whole RAMBlocks.
/// Setting this off makes the device take the copying rails on a host that
/// could have imported — see
/// `reims-vgpu-vulkan::host_pointer`.
///
/// This is the switch that matters for verification. Where the import works
/// every guest window takes it and the copying rails run zero times, so a green
/// boot says nothing about them — and they are the only rails on a host without
/// the extension, and the rails a discrete GPU takes regardless.
///
/// # What the copying rails cost, driven macos-13, two boots each arm
///
/// One binary, interleaved, same probe. The gate held: `disabled_by_env` once
/// per boot, `guest_ram_map_no_backend_import` ~1 000 times, and
/// `sampled_guest_imports`/`compute_buffer_guest_imports` **zero in every one of
/// 77 and 75 windows** — against non-zero on the import-on arm, which is what
/// says the counters would have caught a bind running past a closed gate. No
/// panics; the desktop renders correctly.
///
/// ```text
///                    import on    import off
/// present_hz            14.80     6.80 / 6.85
/// duty                   0.81     0.91 / 0.92
/// draw_us per draw      41.4 us   126.3 / 127.8 us
/// exec_phase finish    639.6 ms/s  810.9 / 821.1 ms/s
/// ```
///
/// **Less than half the frame rate and 3x the per-draw cost**, with the whole
/// difference landing in `ExecPhase::Finish` — the writeback copies. That is the
/// rail working, not a regression: the copy is the point on a host that cannot
/// import, and the guest observes the same pixels. It stays a *performance*
/// difference, which is what the support matrix requires of it.
///
/// **This measures the no-import column, not an iGPU.** A unified-memory host
/// *with* the extension binds a `GuestSlice` directly and is the fastest cell of
/// that matrix, not this one. Nothing in this reading was taken on Intel or AMD
/// hardware.
///
/// # What the GPU-side clock says, and it inverts the reading above
///
/// Everything above is CPU wall clock. `gpu_span` times the submission on the
/// GPU's own clock, and on that clock the copying rails are the **cheaper** arm.
/// One regime-matched pair, driven macos-13, same pin:
///
/// ```text
///                        import off    import on    import on
/// draws per frame           249.6        252.4        253.5
/// window_publish fresh       59.5         59.0         59.0
/// GPU us per draw            6.04        15.26        16.03
/// draw us per submission    78.42       226.66       230.92
/// drain duty                 0.66         0.37         0.36
/// gather regions/draw         0.0         15.4         13.4
/// ```
///
/// **Same frames, 61 % less GPU work per draw, and nearly twice the drain duty.**
/// The gather is what moves: with the import on, every scattered guest window is
/// assembled by the GPU out of guest RAM, which on a discrete host is a PCIe copy
/// — 4.46 GB/s of it, running at ~18 GB/s effective, and about 55 % of all the
/// GPU time this device spends. With the import off there is no gather at all;
/// the CPU packs the same bytes into staging and the drain worker pays for it.
///
/// So the two arms are not "fast" and "slow", they are **which engine does the
/// copy**. On this host, with the GPU at 45 % occupancy and the worker at 0.37
/// duty, moving it to the GPU is free and the import wins on the unmatched
/// regimes. On a host where the GPU is the constraint and the CPU is not — which
/// is the iGPU column of the support matrix — the trade plausibly inverts, and
/// `=off` is already the switch that takes it. That is a hypothesis this host
/// cannot test and it is written here so an operator on an iGPU knows there is a
/// second arm worth two boots.
///
/// One matched pair for the frame rate; the per-draw GPU cost has since
/// replicated across **four boots an arm and three compositing regimes**, and the
/// arms do not come close to touching:
///
/// ```text
/// import off   6.89  6.04  6.61  6.60
/// import on   15.37 13.72 13.84 16.03  (and 14.90, 15.26, 15.85, 13.94)
/// ```
///
/// Drain duty runs 0.66-0.82 on the copying arm against 0.36-0.64 on the import
/// arm, which is the same trade seen from the other side: the worker pays what
/// the GPU does not.
///
/// The frame-rate half stays a single matched pair, because `fresh` is not
/// comparable across compositing regimes and only one pair matched.
pub const GUEST_IMPORT: &str = "REIMS_VGPU_GUEST_IMPORT";

/// `off` keeps a `Shared` allocation on the imported-transfer route -- a
/// separate working buffer with the guest's pages as its transfer endpoint --
/// even on a unified host where the guest's pages could be the execution
/// object outright.
///
/// This is a narrowing-only A/B control: it cannot make an alias out of a host
/// that has no import or no unified memory, both of which are measured from the
/// device. It exists because the two arms differ in *what the guest can see* --
/// on the alias a GPU write is immediately the guest's bytes, on the transfer
/// route it is not until something plans a copy -- so a defect that only
/// appears when the GPU writes guest RAM directly has a single-branch arm to be
/// bisected against.
pub const GUEST_ALIAS: &str = "REIMS_VGPU_GUEST_ALIAS";

/// `off` keeps descriptor state on the allocated Vulkan 1.2 set path even when
/// the device advertises `VK_KHR_push_descriptor` and the layout fits its
/// reported limit.
///
/// This is a narrowing-only A/B control: it cannot enable an extension the
/// device lacks, and it cannot make an over-limit layout use push descriptors.
/// The two arms encode the same descriptor writes; only their Vulkan lifetime
/// differs (command-buffer state versus an allocated set).
pub const PUSH_DESCRIPTORS: &str = "REIMS_VGPU_PUSH_DESCRIPTORS";

/// Verbose per-draw logging on top of the always-on fail sink.
pub const DRAW_LOG: &str = "REIMS_VGPU_DRAW_LOG";

/// `off` stops narrowing a guest buffer bind to the extent the shader's
/// reflection proved it can read, so the bind walks the rest of the allocation
/// exactly as it did before that rail existed.
///
/// This is the A/B instrument for the rail, and it is why the rail can be
/// measured at all: the two arms differ by one branch in one process, so a
/// driven boot of each on one build and one rail attributes a change in gathered
/// bytes to the narrowing rather than to a rebuild. Without it the comparison is
/// a boot of `HEAD` against a boot of `HEAD~1`, which also moves every other
/// difference between the two binaries into the result.
///
/// It only ever *widens the window this device reads*, never what the guest may
/// see, so it obeys the rule the module doc states: it turns a rail off, and
/// there is no spelling of it that turns one on. `on` and unset are the same
/// arm — the default — because a capability that is not measured is not a
/// capability this switch may grant.
pub const BUFFER_EXTENT: &str = "REIMS_VGPU_BUFFER_EXTENT";

/// **Default off.** `on` asks the window system to give the host presentation
/// window the whole monitor it opens on, with no decorations — on Linux (X11 and
/// Wayland alike) that is a borderless full-screen window, which is what winit's
/// `Fullscreen::Borderless` maps to there.
///
/// It changes nothing the guest observes and it grants this device no capability:
/// the window geometry is a request to the *host's* window system, the presenter
/// aspect-fits the guest frame into whatever geometry it ends up with, and the
/// pointer maps through that same viewport. A compositor that refuses the
/// request leaves an ordinary sized window and nothing else changes.
///
/// The one behavioral term it carries is the guest-driven native resize. A
/// full-screen window cannot honour one, so the window stops asking: a guest
/// mode change would otherwise sit out the full resize hold and then log
/// `native_resize_not_applied` about a refusal the operator asked for. The guest
/// still gets its mode — letterboxed into the monitor — which is the same
/// outcome a tiling compositor already produces. `host_window::present`'s
/// `WindowMode` owns both halves.
pub const FULLSCREEN: &str = "REIMS_VGPU_FULLSCREEN";

/// What one variable says, including the two ways it says nothing usable.
///
/// Four states rather than a `bool` because "unset", "explicitly on" and
/// "spelled wrong" are three different operator intents and a `bool` collapses
/// them into the default. The last one matters most: a typo that silently reads
/// as the default is how an operator concludes a switch does not work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Switch {
    /// Not in the environment, or exported empty — which is how a shell says
    /// "not set" when a variable is assigned from an unset variable.
    Unset,
    /// An affirmative spelling. Never sufficient by itself; see the module doc.
    On,
    /// A negative spelling. This is the state that may change behavior.
    Off,
    /// Present, non-empty, and not one of the spellings below. Carries nothing:
    /// the value is handed back by [`read`] for the caller to name in its own
    /// refusal, because only the caller knows which variable this was.
    Unrecognized,
}

/// The spellings accepted for each state, ASCII-case-insensitively.
///
/// The conventional shell set rather than a chosen one, so an operator does not
/// have to look up which of `0`/`false`/`no` this particular program wanted. The
/// two lists are disjoint and every entry is lowercase, which
/// `the_spellings_are_disjoint_and_lowercase` pins.
const ON_SPELLINGS: [&str; 4] = ["1", "on", "true", "yes"];
const OFF_SPELLINGS: [&str; 4] = ["0", "off", "false", "no"];

/// Classify `name`'s value, and hand back the raw value for a caller that needs
/// to quote it.
///
/// Pure: it reads the environment and parses, and emits nothing. Deliberately —
/// `reims-vgpu-observe` itself reads a variable through here, so an emit on this
/// path would recurse through the sink that is asking whether it is enabled.
/// The caller emits, and it is better placed to: it knows which rail the answer
/// gates and what the consequence of refusing is.
pub fn read(name: &str) -> (Switch, Option<String>) {
    let Some(raw) = std::env::var_os(name) else {
        return (Switch::Unset, None);
    };
    let value = raw.to_string_lossy().into_owned();
    let folded = value.trim().to_ascii_lowercase();
    if folded.is_empty() {
        return (Switch::Unset, None);
    }
    let state = if ON_SPELLINGS.contains(&folded.as_str()) {
        Switch::On
    } else if OFF_SPELLINGS.contains(&folded.as_str()) {
        Switch::Off
    } else {
        Switch::Unrecognized
    };
    (state, Some(value))
}

/// [`read`] for a caller that has nothing to say about the value.
pub fn switch(name: &str) -> Switch {
    read(name).0
}

/// Every variable this device reads.
///
/// The one place the set is enumerable. A boot line built from this reports what
/// an operator actually set, which is the difference between a bug report that
/// says "it is slow" and one that says "it is slow with a rail switched off" —
/// and an operator who mistyped a value learns it from the same line, because
/// [`Switch::Unrecognized`] has its own spelling here.
///
/// Nothing enforces that a new `pub const` above is added to this list; the rule
/// is stated and honestly unenforced. What keeps it small is that the list is
/// next to the constants, and [`report_line`] is the only consumer.
pub const ALL: [&str; 6] = [
    GUEST_IMPORT,
    GUEST_ALIAS,
    PUSH_DESCRIPTORS,
    DRAW_LOG,
    BUFFER_EXTENT,
    FULLSCREEN,
];

/// The state of every variable in [`ALL`], for the one-shot boot line.
///
/// Unset variables are on the line too, and deliberately: the reading a report
/// needs is "these five are the whole set and four of them are default", not a
/// line that goes empty and leaves a reader unsure whether it ran.
pub fn report_line() -> String {
    let mut out = String::from("vgpu_env");
    for name in ALL {
        let (state, value) = read(name);
        let short = name.strip_prefix("REIMS_VGPU_").unwrap_or(name);
        let state = match state {
            Switch::Unset => "unset".to_owned(),
            Switch::On => "on".to_owned(),
            Switch::Off => "off".to_owned(),
            // The raw value, because an operator who typed `REIMS_VGPU_GPU_STAMP=disabled`
            // needs to see what the parse rejected, not just that it did.
            Switch::Unrecognized => format!("unrecognized({})", value.unwrap_or_default()),
        };
        out.push_str(&format!(" {}={state}", short.to_ascii_lowercase()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One process-wide lock for every test that mutates the environment.
    /// `set_var` is process-global and unsynchronized; two tests setting
    /// different variables concurrently is fine, but two setting the *same* one
    /// is not, and these all touch the same probe name.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Set `PROBE` to `value` (or unset it), run `body`, and restore.
    fn with_probe<R>(value: Option<&str>, body: impl FnOnce() -> R) -> R {
        const PROBE: &str = "REIMS_VGPU_TEST_PROBE";
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the lock above serializes every mutation of this variable in
        // this process, and nothing outside these tests reads it.
        unsafe {
            match value {
                Some(v) => std::env::set_var(PROBE, v),
                None => std::env::remove_var(PROBE),
            }
        }
        let out = body();
        unsafe { std::env::remove_var(PROBE) };
        out
    }

    fn probe(value: Option<&str>) -> Switch {
        with_probe(value, || switch("REIMS_VGPU_TEST_PROBE"))
    }

    /// Both directions, in every spelling the module claims to accept. A
    /// spelling that silently reads as `Unrecognized` is a switch an operator
    /// sets and watches do nothing.
    #[test]
    fn every_documented_spelling_parses() {
        for on in ON_SPELLINGS {
            assert_eq!(probe(Some(on)), Switch::On, "{on}");
            assert_eq!(probe(Some(&on.to_ascii_uppercase())), Switch::On, "{on}");
        }
        for off in OFF_SPELLINGS {
            assert_eq!(probe(Some(off)), Switch::Off, "{off}");
            assert_eq!(probe(Some(&off.to_ascii_uppercase())), Switch::Off, "{off}");
        }
    }

    /// An unset variable and one exported empty are the same answer. `FOO=$BAR`
    /// with `BAR` unset produces the second, and reading it as a value would
    /// make an unrelated typo elsewhere in a boot script silently flip a rail.
    #[test]
    fn unset_and_empty_are_the_same_answer() {
        assert_eq!(probe(None), Switch::Unset);
        assert_eq!(probe(Some("")), Switch::Unset);
        assert_eq!(probe(Some("   ")), Switch::Unset);
    }

    /// A typo is its own answer and keeps its value, so the caller's refusal can
    /// quote what was actually written. Collapsing this into `Unset` is how a
    /// misspelled switch reads as working.
    #[test]
    fn a_value_that_is_neither_keeps_itself_for_the_message() {
        let (state, value) = with_probe(Some("mabye"), || read("REIMS_VGPU_TEST_PROBE"));
        assert_eq!(state, Switch::Unrecognized);
        assert_eq!(value.as_deref(), Some("mabye"));
    }

    /// Surrounding whitespace is not a value. A trailing space picked up from a
    /// heredoc or a `docker run -e` line would otherwise read as a typo.
    #[test]
    fn surrounding_whitespace_is_trimmed() {
        assert_eq!(probe(Some(" off ")), Switch::Off);
        assert_eq!(probe(Some("\t1\n")), Switch::On);
    }

    /// The two lists cannot overlap and are compared lowercased, so an entry
    /// with a capital in it would never match anything.
    #[test]
    fn the_spellings_are_disjoint_and_lowercase() {
        for on in ON_SPELLINGS {
            assert!(!OFF_SPELLINGS.contains(&on), "{on} is in both lists");
            assert_eq!(on, on.to_ascii_lowercase(), "{on} would never match");
        }
        for off in OFF_SPELLINGS {
            assert_eq!(off, off.to_ascii_lowercase(), "{off} would never match");
        }
    }

    /// Every variable the crate honors is named here, spelled consistently. A
    /// name that does not carry the crate prefix is one an operator cannot find
    /// by grepping their own environment.
    #[test]
    fn every_name_carries_the_crate_prefix() {
        // The declared lists rather than a third one written here: a list
        // written twice is the thing this module exists to stop, and the boot
        // line reads the same two. The uniqueness check below spans both, so a
        // name appearing in each — read once as a switch and once as a count —
        // fails here rather than reaching the line twice with two answers.
        let names: Vec<&str> = ALL.to_vec();
        for name in &names {
            assert!(name.starts_with("REIMS_VGPU_"), "{name}");
            assert!(
                name.bytes()
                    .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_'),
                "{name}"
            );
        }
        for (i, a) in names.iter().enumerate() {
            for b in &names[i + 1..] {
                assert_ne!(a, b, "two variables share a name");
            }
        }
    }

    /// The boot line names every variable, including the ones nobody set.
    ///
    /// A line that only reported what was set would go empty on a default boot,
    /// and an empty line cannot be told from an absent one — so a report from a
    /// machine with a rail switched off would look exactly like a report from a
    /// machine with a build that never emitted it.
    #[test]
    fn the_boot_line_names_every_variable_set_or_not() {
        let line = report_line();
        assert!(line.starts_with("vgpu_env "), "{line}");
        for name in ALL.iter() {
            let short = name
                .strip_prefix("REIMS_VGPU_")
                .expect("the prefix is asserted above")
                .to_ascii_lowercase();
            assert!(line.contains(&format!(" {short}=")), "{short} in {line}");
        }
    }

    /// A value the parse rejects reaches the line verbatim. An operator who
    /// wrote `disabled` instead of `off` otherwise reads `unset` and concludes
    /// the switch does not work.
    #[test]
    fn an_unrecognized_value_reaches_the_boot_line() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the lock serializes every mutation of this variable in this
        // process; `report_line` below is the only reader.
        unsafe { std::env::set_var(GUEST_IMPORT, "disabled") };
        let line = report_line();
        unsafe { std::env::remove_var(GUEST_IMPORT) };
        assert!(
            line.contains("guest_import=unrecognized(disabled)"),
            "{line}"
        );
    }
}
