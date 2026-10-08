#!/bin/sh
# Copy text_utils.rs + definitions.json of one akhtabooti-core commit into vendor/<sha>/.
# usage: rust/core-rules/vendor.sh <sha> [path-to-akhtabooti-core-checkout]
set -e
sha=${1:?usage: vendor.sh <sha> [core checkout]}
here=$(cd "$(dirname "$0")" && pwd)
core=${2:-$here/../../../akhtabooti-core}
dir=$here/vendor/$sha
mkdir -p "$dir"
trap 'rm -rf "$dir"' EXIT
git -C "$core" show "$sha:src/text_utils.rs" > "$dir/text_utils.rs"
git -C "$core" show "$sha:definitions.json" > "$dir/definitions.json"
git -C "$core" show -s --format='%H %ad %s' --date=iso "$sha" > "$dir/COMMIT"
trap - EXIT
echo "vendored $(cat "$dir/COMMIT") -> $dir"
