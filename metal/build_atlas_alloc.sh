#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${ASTC_KV_ATLAS_ALLOC_BUILD_DIR:-$project_dir/build/metal-atlas-alloc}"
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "Allocation probe build requires macOS and existing CLT." >&2
    exit 2
fi
mkdir -p "$build_dir"
build_dir="$(cd -- "$build_dir" && pwd)"
xcrun -sdk macosx clang++ -std=c++17 -O2 -fobjc-arc -mmacosx-version-min=13.0 \
    -Wall -Wextra "$project_dir/metal/probe_atlas_alloc.mm" \
    -framework Foundation -framework Metal -o "$build_dir/astc-kv-atlas-alloc"
echo "Built $build_dir/astc-kv-atlas-alloc (allocation execution remains unverified)."
