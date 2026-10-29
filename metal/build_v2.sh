#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
metal_dir="$project_dir/metal"
build_dir="${ASTC_KV_METAL_V2_BUILD_DIR:-$project_dir/build/metal-v2}"
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "Metal build requires macOS and the Command Line Tools SDK; no GPU result was produced." >&2
    exit 2
fi
mkdir -p "$build_dir"
build_dir="$(cd -- "$build_dir" && pwd)"
xcrun -sdk macosx clang++ -std=c++17 -O3 -fobjc-arc -mmacosx-version-min=13.0 \
    -Wall -Wextra "$metal_dir/probe_v2.mm" -framework Foundation -framework Metal \
    -o "$build_dir/astc-kv-probe-v2"
echo "Built $build_dir/astc-kv-probe-v2"
if xcrun --find metal >/dev/null 2>&1; then
    xcrun -sdk macosx metal -std=metal3.0 -fno-fast-math -c \
        "$metal_dir/attention_v2.metal" -o "$build_dir/attention_v2.air"
    xcrun -sdk macosx metallib "$build_dir/attention_v2.air" -o "$build_dir/attention_v2.metallib"
    shader_path="$build_dir/attention_v2.metallib"
    echo "Built $shader_path"
else
    shader_path="$metal_dir/attention_v2.metal"
    echo "Offline Metal compiler unavailable; probe will compile $shader_path at runtime."
    echo "Host build succeeded; shader compilation and GPU execution remain unverified until the probe runs."
fi
# Explicit selection avoids accidentally using a stale metallib from an older build.
printf '%s\n' "$shader_path" > "$build_dir/shader_path.txt"
