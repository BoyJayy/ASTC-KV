#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
build_dir="$project_dir/build/llama-baseline"
source_dir="$project_dir/third_party/llama.cpp-c3df4b7e22a1ac38da663552d19e649dd97ca936"
xcrun clang++ -std=c++17 -O3 -DNDEBUG -fobjc-arc -Wall -Wextra \
  -I "$source_dir/ggml/include" "$project_dir/baselines/llama/precision_probe.mm" \
  "$build_dir/ggml/src/libggml.a" "$build_dir/ggml/src/libggml-cpu.a" \
  "$build_dir/ggml/src/ggml-metal/libggml-metal.a" "$build_dir/ggml/src/libggml-base.a" \
  -framework Foundation -framework Accelerate -framework Metal -framework MetalKit -lm \
  -o "$build_dir/bin/astc-llama-precision"
echo "Built precision diagnostic; original benchmark binary/source unchanged."
