//! JSON-lines front end for akhtabooti-core's text rules.
//!
//! stdin:  {"id": "...", "text": "..."} per line
//! stdout: {"id": "...", "detections": {"<category>": ["<evidence>", ...]}, "keyword_categories": [...]}
//!
//! One output line is written and flushed per input line, so a caller can keep
//! the process open and send texts one at a time.
//!
//! Options:
//!   --version               print the compiled-in core commit, API and rule file as JSON
//!   --definitions <path>    load this rule file instead of the vendored one (API v2 only)
use std::collections::{BTreeMap, HashSet};
use std::io::{self, BufRead, Write};
use std::process::exit;

use serde::{Deserialize, Serialize};

#[allow(dead_code, unused_imports, unused_variables, clippy::all)]
mod core_text {
    include!(concat!(env!("OUT_DIR"), "/text_utils.rs"));
}

#[derive(Deserialize)]
struct Input {
    id: String,
    text: String,
}

#[derive(Serialize)]
struct Output<'a> {
    id: &'a str,
    detections: BTreeMap<String, Vec<String>>,
    keyword_categories: Vec<String>,
}

fn sorted(set: HashSet<String>) -> Vec<String> {
    let mut v: Vec<String> = set.into_iter().collect();
    v.sort();
    v
}

#[cfg(core_api_v2)]
mod engine {
    use super::*;
    use std::collections::HashMap;
    pub const ACCEPTS_DEFINITIONS: bool = true;
    pub struct Engine {
        defs: core_text::Definitions,
    }
    impl Engine {
        pub fn load(path: &str) -> Result<Engine, String> {
            core_text::load_definitions(path)
                .map(|defs| Engine { defs })
                .map_err(|e| format!("cannot load rule file {path}: {e}"))
        }
        pub fn run(&self, text: &str) -> (BTreeMap<String, Vec<String>>, Vec<String>) {
            let found: HashMap<String, HashSet<String>> = core_text::detect(&self.defs, text);
            (
                found.into_iter().map(|(k, v)| (k, sorted(v))).collect(),
                Vec::new(),
            )
        }
    }
}

#[cfg(core_api_v1)]
mod engine {
    use super::*;
    // The v1 revision under evaluation embeds definitions.json at compile time.
    pub const ACCEPTS_DEFINITIONS: bool = false;
    pub struct Engine {
        rules: serde_json::Value,
    }
    impl Engine {
        // Every v1 revision keeps the `get_regexes(path)` signature, including the
        // one that embeds definitions.json at compile time and ignores the path.
        pub fn load(path: &str) -> Result<Engine, String> {
            core_text::get_regexes(path)
                .map(|rules| Engine { rules })
                .map_err(|e| format!("cannot load rule file {path}: {e}"))
        }
        pub fn run(&self, text: &str) -> (BTreeMap<String, Vec<String>>, Vec<String>) {
            let mut det = BTreeMap::new();
            let emails = core_text::detect_emails(self.rules.clone(), text).expect("detect_emails");
            if !emails.is_empty() {
                det.insert("Email".to_string(), sorted(emails));
            }
            let phones = core_text::detect_phone_numbers(self.rules.clone(), text)
                .expect("detect_phone_numbers");
            if !phones.is_empty() {
                det.insert("Phone Number".to_string(), sorted(phones));
            }
            // v1 keyword detection reports category names only, with no evidence string.
            let kws =
                core_text::detect_keywords(self.rules.clone(), text).expect("detect_keywords");
            (det, sorted(kws))
        }
    }
}

struct Options {
    version: bool,
    definitions: Option<String>,
}

fn parse_args() -> Result<Options, String> {
    let mut options = Options {
        version: false,
        definitions: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--version" => options.version = true,
            "--definitions" => {
                options.definitions = Some(args.next().ok_or("--definitions needs a path")?);
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    if options.definitions.is_some() && !engine::ACCEPTS_DEFINITIONS {
        return Err(format!(
            "--definitions needs core API v2; this build uses {} with rules compiled in",
            env!("AKH_CORE_API")
        ));
    }
    Ok(options)
}

fn fail(message: &str) -> ! {
    eprintln!("akh-core-rules: {message}");
    exit(2)
}

fn main() {
    let options = parse_args().unwrap_or_else(|e| fail(&e));
    if options.version {
        println!(
            "{}",
            serde_json::json!({
                "core_sha": env!("AKH_CORE_SHA"),
                "core_commit": env!("AKH_CORE_COMMIT"),
                "core_api": env!("AKH_CORE_API"),
                "definitions": env!("AKH_DEFS_PATH"),
                "accepts_definitions": engine::ACCEPTS_DEFINITIONS,
                "shim_version": env!("CARGO_PKG_VERSION"),
            })
        );
        return;
    }
    let defs_path = options
        .definitions
        .unwrap_or_else(|| env!("AKH_DEFS_PATH").to_string());
    let engine = engine::Engine::load(&defs_path).unwrap_or_else(|e| fail(&e));

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let line = line.unwrap_or_else(|e| fail(&format!("cannot read stdin: {e}")));
        if line.trim().is_empty() {
            continue;
        }
        let input: Input = serde_json::from_str(&line)
            .unwrap_or_else(|e| fail(&format!("input line must be {{\"id\", \"text\"}}: {e}")));
        let (detections, keyword_categories) = engine.run(&input.text);
        serde_json::to_writer(
            &mut out,
            &Output {
                id: &input.id,
                detections,
                keyword_categories,
            },
        )
        .and_then(|()| out.write_all(b"\n").map_err(serde_json::Error::io))
        .and_then(|()| out.flush().map_err(serde_json::Error::io))
        .unwrap_or_else(|e| fail(&format!("cannot write stdout: {e}")));
    }
}
