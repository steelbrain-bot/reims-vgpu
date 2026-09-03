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
//! * **Every function the register names still exists.** A row's last column
//!   names the owner-level tests that replaced the legacy tests moving with
//!   the group, and those legacy tests stopped running the moment they moved.
//!   A row naming a test that has since been renamed or deleted is the silent
//!   coverage loss the rule exists to catch, and prose cannot notice it.
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

/// Every function the register's replacement-coverage column names is still a
/// function.
///
/// # The row is the only record that the coverage survived
///
/// A group's legacy tests move to `dead/` with its source and stop running
/// there. What replaced them is written in the register and nowhere else — no
/// build step relates the two — so a row naming
/// `runtime::exec::tests::a_bounded_event_wait_is_refused_by_contract…` is the
/// whole evidence that the case is still covered. Rename or delete that test
/// and the row keeps reading as if it were, which is the exact failure the
/// rule was written against, arriving in the record that was supposed to
/// prevent it.
///
/// # What counts as a name, and why the rule is that shape
///
/// Two forms, because one of them was not enough.
///
/// The first is a backticked path with at least one `::` whose last segment is
/// snake_case with an underscore in it. Type and variant names end in CamelCase
/// and are skipped.
///
/// The second exists because almost no coverage claim is written that way. A
/// test named in a row is named the way a reader would say it — bare, as
/// `a_declined_publish_vouches_for_nothing` — and a bare backticked word cannot
/// be told from a census route, a counter or an observation slug, of which the
/// rows are full. So this check read nine coverage claims and verified two of
/// them, while its own doc and the register's rules both said it verified all
/// nine. A gate that reports on what it did not inspect is worse than no gate:
/// the rule it was standing in for was being trusted.
///
/// The fix is to make the claim say it is one. A coverage name is written
/// ``fn `name` `` after the `Replacement coverage` marker, and that `fn` is what
/// separates the claim from the prose around it — which is how a row can go on
/// naming `pools/`, `winpub_*` or a slug that no longer exists in the same
/// sentence without either being mistaken for a promise.
///
/// The marker is required because the register's other columns name functions
/// while describing what *moved*, and a moved function is allowed to be gone.
///
/// Existence, not location: a name in the register is prose written by a
/// human, and pinning the module as well would make the check fail on a
/// correct row whose module was reorganised — a gate that fails on correct
/// work is a gate that gets weakened.
#[test]
fn every_function_the_register_names_still_exists() {
    let register = workspace_root()
        .join("crates")
        .join("reims-vgpu")
        .join("src")
        .join("dead")
        .join("README.md");
    let text = std::fs::read_to_string(&register).expect("the register is readable");

    let mut named: Vec<String> = Vec::new();
    let mut rows = 0usize;
    let mut claims = 0usize;
    for line in text.lines() {
        if !line.starts_with('|') || line.starts_with("|---") {
            continue;
        }
        // The register proper has five columns and its fifth is the
        // replacement-coverage column. The live-validation table below it has
        // three, and its rows carry coverage claims in prose — seven of the
        // nine in this file, which is why reading only the five-column table
        // was reading the smaller half.
        if line.matches('|').count() >= 6 {
            if let Some(column) = line.split('|').nth(5) {
                rows += 1;
                named.extend(backticked_function_paths(column));
            }
        }
        let Some(marker) = line.find("eplacement coverage") else {
            continue;
        };
        let claimed = declared_coverage_names(&line[marker..]);
        claims += claimed.len();
        named.extend(claimed);
    }
    assert!(
        rows > 5,
        "only {rows} register rows were parsed, which is too few to have read \
         the table — a check that inspects nothing passes for the wrong reason"
    );
    assert!(
        claims > 10,
        "only {claims} `fn `name`` coverage claims were found. Either the rows \
         stopped declaring them in the form this gate reads, or this parser \
         stopped reading it — and both leave the register's coverage column \
         unchecked while it still reads as checked"
    );
    assert!(
        named.len() > 10,
        "only {} names were extracted from {rows} rows",
        named.len()
    );

    let defined = defined_function_names();
    let missing: Vec<&String> = named.iter().filter(|n| !defined.contains(*n)).collect();
    assert!(
        missing.is_empty(),
        "the register names {missing:?}, and no `fn` by that name exists. A row \
         claiming coverage that has been renamed or deleted is the silent loss \
         the rule exists to catch: either restore the function, or amend the row \
         to name what covers the case now"
    );
}

/// The names a row *declares* as its replacement coverage, which is the subset
/// of its backticks written as ``fn `name` ``.
///
/// Everything else in the sentence is prose and stays prose: `pools/` is a
/// directory, `winpub_*` is a route family, and a row is allowed to name a slug
/// that no longer exists in order to say that it no longer exists. Only what
/// carries the `fn` is a promise about a function.
fn declared_coverage_names(tail: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = tail;
    while let Some(at) = rest.find("fn `") {
        rest = &rest[at + 4..];
        let Some(end) = rest.find('`') else { break };
        let token = &rest[..end];
        rest = &rest[end + 1..];
        if token.is_empty()
            || token
                .chars()
                .any(|c| !(c.is_ascii_alphanumeric() || c == '_' || c == ':'))
        {
            continue;
        }
        let last = token.rsplit("::").next().unwrap_or_default();
        if !last.is_empty() {
            out.push(last.to_string());
        }
    }
    out
}

/// The snake_case tails of backticked `a::b::c` paths in one register cell.
fn backticked_function_paths(cell: &str) -> Vec<String> {
    let mut out = Vec::new();
    for token in cell.split('`').skip(1).step_by(2) {
        if !token.contains("::") {
            continue;
        }
        if token
            .chars()
            .any(|c| !(c.is_ascii_alphanumeric() || c == '_' || c == ':'))
        {
            // A type parameter, a field with its type, a lifetime — not a plain
            // path, and guessing at one is how this check starts failing on
            // prose.
            continue;
        }
        let last = token.rsplit("::").next().unwrap_or_default();
        if last.contains('_') && last.chars().all(|c| !c.is_ascii_uppercase()) {
            out.push(last.to_string());
        }
    }
    out
}

/// Every `fn` name defined anywhere in the workspace's crates, `dead/`
/// included — a row may legitimately name a function that moved, as long as it
/// still exists somewhere to be read.
fn defined_function_names() -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    let mut files = Vec::new();
    let crates = workspace_root().join("crates");
    for entry in std::fs::read_dir(&crates)
        .expect("crates/ is readable")
        .flatten()
    {
        walk_all(&entry.path(), &mut files);
    }
    for path in files {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        for line in text.lines() {
            let Some(at) = line.find("fn ") else { continue };
            let rest = &line[at + 3..];
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                out.insert(name);
            }
        }
    }
    out
}

fn walk_all(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_all(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}
