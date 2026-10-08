#!/bin/sh
# Build akh-ocr in release mode and copy it to bin/akh-ocr.
# The crate pins rust-version 1.84; .cargo/config.toml makes cargo pick dependency versions that build on it.
set -e
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"
cargo build --release --locked
mkdir -p bin
cp target/release/akh-ocr bin/akh-ocr
echo "built bin/akh-ocr: $(bin/akh-ocr --version)"
