//! Records the locked `image` crate version so `--version` can report it.

use std::fs;

fn locked_version(lock: &str, name: &str) -> Option<String> {
    let marker = format!("name = \"{name}\"\nversion = \"");
    let start = lock.find(&marker)? + marker.len();
    lock[start..].split('"').next().map(str::to_owned)
}

fn main() {
    println!("cargo:rerun-if-changed=Cargo.lock");
    let lock = fs::read_to_string("Cargo.lock").unwrap_or_default();
    let image = locked_version(&lock, "image").unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=AKH_IMAGE_VERSION={image}");
}
