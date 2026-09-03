//! `dead/` is source to read, and this is what makes that claim checkable.
//!
//! # Why a test and not a convention
//!
//! The project owner's 2026-09-02 amendment replaced the plan's *delete the
//! legacy counterpart* with *disconnect it*: the commit that wires a group
//! moves that group's files to `crates/<crate>/src/dead/` and removes every
//! `mod` declaration that reaches them. The strength being bought is that the
//! prohibition on two semantic models holds **by construction** — a removed
//! `mod` is not a flag anyone can flip, and a file no module tree reaches is
//! not compiled, not feature-gated and not linkable.
//!
//! That strength lasts exactly as long as nobody adds the `mod` back. Nothing
//! in the build would complain: `mod dead;` compiles, and the second semantic
//! model it re-links would show up as a behaviour difference on a live boot
//! weeks later, attributed to whatever else changed that week. A convention
//! with no gate is the thing the amendment was written against.
//!
//! # What it checks, and why each one
//!
//! * **No `mod` declaration names `dead`.** The direct form.
//! * **No `#[path]` attribute points into a `dead/` directory.** The indirect
//!   form, which a reader scanning for `mod dead;` would not find — a module
//!   declared under any name at all can be given a `dead/` file to compile.
//! * **Nothing outside `dead/` names it as a path.** A `crate::dead::…` or
//!   `use …dead::…` cannot compile without one of the two above, so a hit here
//!   is a leftover from a reversal in progress and is worth naming separately
//!   from the mechanism that would carry it.
//! * **No `include!` names a `dead/` file.** A third way to compile one: the
//!   macro splices the file into whatever module the call sits in, with no
//!   `mod` and no `#[path]` for either of the checks above to see.
//! * **No `Cargo.toml` target points into `dead/`.** The module tree is not the
//!   only door. A `[[test]]` or `[[bin]]` with `path = "src/dead/…"` compiles
//!   and links the file as its own crate root, which is a second semantic model
//!   that the whole `src/` walk would report as clean.
//! * **Every `dead/` has a `README.md`.** The register is where a move records
//!   what it replaced and which owner-level tests replaced the legacy tests
//!   that moved with it. Those tests stop running the moment they move, so a
//!   `dead/` with no register is a group that has silently lost its coverage.
//!
//! The walk is over sources on disk rather than over anything the compiler
//! knows, because the whole point is to catch a file the compiler *has* been
//! told about.

use std::path::{Path, PathBuf};

/// The workspace root, from this crate's manifest directory.
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crates/<crate> sits two levels under the workspace root")
        .to_path_buf()
}

/// Every `.rs` file under `crates/*/src`, with the ones inside a `dead/`
/// directory left out — `dead/` is not compiled, so what it says about itself
/// is not evidence about the live tree.
fn live_rust_sources() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let crates = workspace_root().join("crates");
    let entries = std::fs::read_dir(&crates).expect("crates/ is readable");
    for entry in entries.flatten() {
        walk(&entry.path().join("src"), &mut out);
    }
    assert!(
        out.len() > 100,
        "the walk found {} sources, which is too few to have visited the tree — \
         a test that silently inspects nothing passes for the wrong reason",
        out.len()
    );
    out
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|n| n == "dead") {
                continue;
            }
            walk(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// A line with its `//` and `//!` comment tail removed, so the prose in this
/// file — and in `dead/README.md`'s neighbours — cannot fail the test it
/// describes.
///
/// Deliberately not a parser. A `mod dead;` inside a string literal or a block
/// comment would be flagged; that is the safe direction for a check whose
/// failure mode is a re-linked semantic model, and no such literal exists.
fn code_of(line: &str) -> &str {
    match line.find("//") {
        Some(i) => &line[..i],
        None => line,
    }
}

#[test]
fn no_module_declaration_reaches_dead() {
    let mut offenders = Vec::new();
    for path in live_rust_sources() {
        let text = std::fs::read_to_string(&path).expect("a source file is readable");
        for (n, line) in text.lines().enumerate() {
            let code = code_of(line);
            let declares_dead = code
                .split_whitespace()
                .collect::<Vec<_>>()
                .windows(2)
                .any(|w| w[0] == "mod" && w[1].trim_end_matches(';') == "dead");
            if declares_dead {
                offenders.push(format!("{}:{}", path.display(), n + 1));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "a `mod dead` declaration re-links the disconnected legacy source, which \
         is the second semantic model the plan forbids wearing a different hat. \
         `dead/` is read and the fix lands in the new owner; nothing is \
         resurrected from it. Found at: {offenders:?}"
    );
}

#[test]
fn no_path_attribute_points_into_dead() {
    let mut offenders = Vec::new();
    for path in live_rust_sources() {
        let text = std::fs::read_to_string(&path).expect("a source file is readable");
        for (n, line) in text.lines().enumerate() {
            let code = code_of(line);
            if code.contains("path") && code.contains("dead/") && code.contains('#') {
                offenders.push(format!("{}:{}", path.display(), n + 1));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "a `#[path = \"…dead/…\"]` compiles a disconnected file under some other \
         module's name, which a reader scanning for `mod dead;` would not find. \
         Found at: {offenders:?}"
    );
}

#[test]
fn nothing_outside_dead_names_it_as_a_path() {
    let mut offenders = Vec::new();
    for path in live_rust_sources() {
        let text = std::fs::read_to_string(&path).expect("a source file is readable");
        for (n, line) in text.lines().enumerate() {
            if code_of(line).contains("dead::") {
                offenders.push(format!("{}:{}", path.display(), n + 1));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "a path into `dead` cannot compile without a module declaration this \
         suite already refuses, so a hit here is a reversal in progress rather \
         than a finished one. Found at: {offenders:?}"
    );
}

#[test]
fn every_dead_directory_carries_its_register() {
    let crates = workspace_root().join("crates");
    let mut seen = 0usize;
    for entry in std::fs::read_dir(&crates)
        .expect("crates/ is readable")
        .flatten()
    {
        let dead = entry.path().join("src").join("dead");
        if !dead.is_dir() {
            continue;
        }
        seen += 1;
        assert!(
            dead.join("README.md").is_file(),
            "{} has no register. Every move appends a row naming what moved, \
             which commit replaced it, and which owner-level tests replaced the \
             legacy tests that moved with it — those tests stop running the \
             moment they move, and the row is where that is caught",
            dead.display()
        );
    }
    assert!(
        seen > 0,
        "no `dead/` directory found at all. If the last one has been deleted \
         wholesale — which happens once, after every group has moved and the \
         gates are green — this suite has done its job and goes with it"
    );
}

#[test]
fn no_include_macro_splices_a_dead_file() {
    let mut offenders = Vec::new();
    for path in live_rust_sources() {
        let text = std::fs::read_to_string(&path).expect("a source file is readable");
        for (n, line) in text.lines().enumerate() {
            let code = code_of(line);
            if code.contains("dead/")
                && (code.contains("include!") || code.contains("include_str!"))
            {
                offenders.push(format!("{}:{}", path.display(), n + 1));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "an `include!` splices a disconnected file into whatever module the call \
         sits in, with no `mod` and no `#[path]` for the other checks to see. \
         Found at: {offenders:?}"
    );
}

#[test]
fn no_cargo_target_points_into_dead() {
    let crates = workspace_root().join("crates");
    let mut offenders = Vec::new();
    let mut seen = 0usize;
    for entry in std::fs::read_dir(&crates)
        .expect("crates/ is readable")
        .flatten()
    {
        let manifest = entry.path().join("Cargo.toml");
        let Ok(text) = std::fs::read_to_string(&manifest) else {
            continue;
        };
        seen += 1;
        for (n, line) in text.lines().enumerate() {
            let code = match line.find('#') {
                Some(i) => &line[..i],
                None => line,
            };
            if code.contains("dead/") {
                offenders.push(format!("{}:{}", manifest.display(), n + 1));
            }
        }
    }
    assert!(
        seen > 1,
        "only {seen} manifests were read; the walk missed the tree"
    );
    assert!(
        offenders.is_empty(),
        "a Cargo target with `path = \"src/dead/…\"` compiles and links the file \
         as its own crate root — a second semantic model the `src/` walk would \
         report as clean, because the module tree is not the only door. \
         Found at: {offenders:?}"
    );
}
