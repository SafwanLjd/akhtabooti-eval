#!/bin/sh
# Build the shim against one vendored core commit and copy it to bin/akh-core-rules-<sha>.
# usage: rust/core-rules/build.sh <sha>
set -e
sha=${1:?usage: build.sh <sha>}
here=$(cd "$(dirname "$0")" && pwd)
[ -f "$here/vendor/$sha/text_utils.rs" ] || "$here/vendor.sh" "$sha"
cd "$here"
AKH_CORE_VENDOR=$sha cargo build --release --locked --quiet
mkdir -p bin
cp target/release/akh-core-rules "bin/akh-core-rules-$sha"
echo "built bin/akh-core-rules-$sha: $("bin/akh-core-rules-$sha" --version)"
