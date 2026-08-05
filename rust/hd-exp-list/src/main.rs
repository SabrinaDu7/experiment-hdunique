//! `hd-exp-list` — the `hd-exp list` sub-command, without the Python import cost.
//!
//! Listing questions needs three facts per module: its docstring, its `EXPERIMENTS`, and whether
//! it defines `collect`. Importing the modules to read them pulls in sklearn, scipy, pandas and
//! pynapple — about 2.6 s. Reading them out of the source text instead is ~5 ms.
//!
//! The registry in `src/experiments/__init__.py` stays the single source of truth: this binary
//! reads `QUESTION_IDS` and `PLANNED` from it rather than keeping its own copy. Output is
//! byte-for-byte identical to the Python `cmd_list`, which `tests/test_exp_list.py` pins.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// What `list` needs to know about one question module.
struct Module {
    doc: String,
    experiments: Vec<String>,
    has_collect: bool,
}

fn main() -> ExitCode {
    let root = match resolve_root(std::env::args().nth(1)) {
        Ok(root) => root,
        Err(err) => {
            eprintln!("hd-exp-list: {err}");
            return ExitCode::FAILURE;
        }
    };
    match render(&root) {
        Ok(text) => {
            print!("{text}");
            ExitCode::SUCCESS
        }
        Err(err) => {
            eprintln!("hd-exp-list: {err}");
            ExitCode::FAILURE
        }
    }
}

/// The `experiments` package directory, from an explicit argument or by walking up from the cwd.
fn resolve_root(arg: Option<String>) -> Result<PathBuf, String> {
    if let Some(arg) = arg {
        let path = PathBuf::from(arg);
        let dir = if path.ends_with("experiments") { path } else { path.join("src/experiments") };
        if dir.join("__init__.py").is_file() {
            return Ok(dir);
        }
        return Err(format!("{} is not an experiments package", dir.display()));
    }
    let mut here = std::env::current_dir().map_err(|e| e.to_string())?;
    loop {
        let dir = here.join("src/experiments");
        if dir.join("__init__.py").is_file() {
            return Ok(dir);
        }
        if !here.pop() {
            return Err("could not find src/experiments above the working directory".into());
        }
    }
}

/// The full listing, in `QUESTION_IDS` order.
fn render(root: &Path) -> Result<String, String> {
    let registry = read(&root.join("__init__.py"))?;
    let question_ids = string_collection(&registry, "QUESTION_IDS")
        .ok_or("QUESTION_IDS not found in experiments/__init__.py")?;
    let planned = string_collection(&registry, "PLANNED").unwrap_or_default();

    let mut out = String::new();
    for qid in &question_ids {
        if planned.contains(qid) {
            out.push_str(&format!("  {qid:12} [{:15}] not implemented yet\n", "planned"));
            continue;
        }
        let module = parse_module(&read(&root.join(format!("{qid}.py")))?);
        let tag = if module.has_collect { "collect+analyse" } else { "analyse only" };
        out.push_str(&format!("  {qid:12} [{tag:15}] {}\n", module.doc));
        for name in &module.experiments {
            out.push_str(&format!("      {name}\n"));
        }
    }
    Ok(out)
}

fn read(path: &Path) -> Result<String, String> {
    fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
}

/// Pull a question module's docstring, `EXPERIMENTS` and top-level `collect` out of its source.
fn parse_module(src: &str) -> Module {
    Module {
        doc: docstring(src)
            .map(|d| d.trim().lines().next().unwrap_or("").to_string())
            .unwrap_or_default(),
        experiments: string_collection(src, "EXPERIMENTS").unwrap_or_default(),
        has_collect: has_top_level_def(src, "collect"),
    }
}

/// The module docstring's raw text, skipping any leading comments and blank lines.
fn docstring(src: &str) -> Option<String> {
    let bytes = src.as_bytes();
    let mut i = 0;
    loop {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b'#' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        break;
    }
    for quote in ["\"\"\"", "'''", "\"", "'"] {
        if src[i..].starts_with(quote) {
            let start = i + quote.len();
            return src[start..].find(quote).map(|end| src[start..start + end].to_string());
        }
    }
    None
}

/// Whether the module defines `name` at top level, mirroring `hasattr(module, name)` for a def.
fn has_top_level_def(src: &str, name: &str) -> bool {
    let needle = format!("def {name}(");
    src.starts_with(&needle) || src.contains(&format!("\n{needle}"))
}

/// The string literals of a module-level assignment such as `EXPERIMENTS = ("a", "b")`.
///
/// Handles the annotated and wrapped forms the registry uses — `QUESTION_IDS: tuple[str, ...] =
/// (...)` and `PLANNED: frozenset[str] = frozenset({...})` — by taking the balanced bracket region
/// after the `=` and reading every string literal inside it.
fn string_collection(src: &str, name: &str) -> Option<Vec<String>> {
    let bytes = src.as_bytes();
    let mut search = 0;
    while let Some(found) = src[search..].find(name) {
        let at = search + found;
        search = at + name.len();
        // Module-level only: the name must start a line and be a whole token.
        if at != 0 && bytes[at - 1] != b'\n' {
            continue;
        }
        let rest = &src[at + name.len()..];
        if rest.starts_with(|c: char| c.is_alphanumeric() || c == '_') {
            continue;
        }
        let Some(eq) = rest.find('=') else { continue };
        // Only an annotation may sit between the name and the `=`.
        if rest[..eq].contains('\n') {
            continue;
        }
        let tail = &rest[eq + 1..];
        // Skip a constructor call like `frozenset(` to reach the literal's brackets.
        let open = tail.find(|c: char| !c.is_whitespace() && !c.is_alphanumeric() && c != '_')?;
        if !matches!(tail.as_bytes()[open], b'(' | b'[' | b'{') {
            return Some(Vec::new());
        }
        return Some(string_literals(balanced(&tail[open..])?));
    }
    None
}

/// The bracketed region at the start of `s`, including its delimiters, respecting strings.
fn balanced(s: &str) -> Option<&str> {
    let bytes = s.as_bytes();
    let mut depth = 0usize;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[..=i]);
                }
            }
            b'#' => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
                continue;
            }
            q @ (b'"' | b'\'') => {
                i += 1;
                while i < bytes.len() && bytes[i] != q {
                    i += if bytes[i] == b'\\' { 2 } else { 1 };
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Every string literal in a bracketed region, in order, with backslash escapes resolved.
fn string_literals(s: &str) -> Vec<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'#' => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            q @ (b'"' | b'\'') => {
                i += 1;
                let mut lit = String::new();
                while i < bytes.len() && bytes[i] != q {
                    if bytes[i] == b'\\' && i + 1 < bytes.len() {
                        i += 1;
                    }
                    let ch = s[i..].chars().next().unwrap();
                    lit.push(ch);
                    i += ch.len_utf8();
                }
                out.push(lit);
                i += 1;
            }
            _ => i += 1,
        }
    }
    out
}
