//! Selects which vendored copy of akhtabooti-core's `text_utils.rs` to compile.
//!
//! `AKH_CORE_VENDOR=<sha>` picks `vendor/<sha>/`, created by `vendor.sh`. The
//! core's text API changed on 2026-10-01; the build script sniffs which one the
//! vendored file exposes and sets a cfg so `main.rs` can adapt.
use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=AKH_CORE_VENDOR");
    println!("cargo:rustc-check-cfg=cfg(core_api_v1)");
    println!("cargo:rustc-check-cfg=cfg(core_api_v2)");

    let sha = env::var("AKH_CORE_VENDOR")
        .expect("set AKH_CORE_VENDOR=<core git sha>; run vendor.sh <sha> first");
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let vendor = manifest.join("vendor").join(&sha);
    let src = vendor.join("text_utils.rs");
    let defs = vendor.join("definitions.json");
    println!("cargo:rerun-if-changed={}", src.display());
    println!("cargo:rerun-if-changed={}", defs.display());

    let code =
        fs::read_to_string(&src).unwrap_or_else(|e| panic!("cannot read {}: {e}", src.display()));
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    // definitions.json is copied next to the module so `include_str!("definitions.json")`
    // in embedded-definitions revisions resolves.
    fs::copy(&defs, out.join("definitions.json")).unwrap();
    // Revisions that embed the rules use include_str!("../definitions.json"); point
    // that at the copy made above.
    let code_out = code.replace(
        "include_str!(\"../definitions.json\")",
        "include_str!(\"definitions.json\")",
    );
    fs::write(out.join("text_utils.rs"), &code_out).unwrap();

    let api = if code.contains("pub fn detect(") {
        "v2"
    } else if code.contains("pub fn detect_keywords(") {
        "v1"
    } else {
        panic!("unrecognised text_utils API in {}", src.display())
    };
    println!("cargo:rustc-cfg=core_api_{api}");
    let commit = fs::read_to_string(vendor.join("COMMIT")).unwrap_or_default();
    println!("cargo:rustc-env=AKH_CORE_SHA={sha}");
    println!("cargo:rustc-env=AKH_CORE_API={api}");
    println!("cargo:rustc-env=AKH_CORE_COMMIT={}", commit.trim());
    println!("cargo:rustc-env=AKH_DEFS_PATH={}", defs.display());
}
