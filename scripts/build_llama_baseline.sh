#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$project_dir/third_party/llama.cpp-c3df4b7e22a1ac38da663552d19e649dd97ca936"
build_dir="$project_dir/build/llama-baseline"
cmake_bin="${CMAKE:-cmake}"
python_bin="${PYTHON:-python3}"
if [[ "$(uname -s)" != Darwin ]]; then
  echo "This pinned standalone Metal baseline requires macOS and CLT." >&2
  exit 2
fi
[[ -d "$source_dir/ggml" ]] || { echo "Pinned llama.cpp source is missing." >&2; exit 2; }
mkdir -p "$build_dir"
"$cmake_bin" -S "$project_dir/baselines/llama" -B "$build_dir" \
  -DLLAMA_SOURCE_DIR="$source_dir" -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF -DGGML_BACKEND_DL=OFF \
  -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_METAL_NDEBUG=OFF \
  -DGGML_CPU=ON -DGGML_BLAS=OFF -DGGML_OPENMP=OFF -DGGML_NATIVE=ON -DGGML_CCACHE=OFF \
  -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF \
  > "$build_dir/configure.log" 2>&1
"$cmake_bin" --build "$build_dir" --target astc-llama-baseline -j 4 \
  > "$build_dir/build.log" 2>&1
"$python_bin" "$project_dir/baselines/llama/write_build_provenance.py" "$project_dir" "$build_dir" "$cmake_bin"
echo "Built $build_dir/bin/astc-llama-baseline (GPU execution not performed)."
