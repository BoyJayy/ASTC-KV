#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ "$(uname -s)" != Darwin ]]; then
  echo "This script needs an Apple Silicon Mac with the Command Line Tools SDK." >&2
  exit 2
fi
python="${PYTHON:-python3}"
out_dir="${OUT_DIR:-$project_dir/results/mac}"
if [[ -e "$out_dir" ]]; then
  echo "Refusing to overwrite existing results: $out_dir. Set OUT_DIR to a new directory." >&2
  exit 2
fi
# Check the selected summarizer before consuming GPU time or creating output.
"$python" -c 'import numpy'
mkdir -p "$(dirname -- "$out_dir")"
mkdir "$out_dir"
out_dir="$(cd -- "$out_dir" && pwd)"
export ASTC_KV_METAL_BUILD_DIR="${ASTC_KV_METAL_BUILD_DIR:-$out_dir/build}"
mkdir -p "$ASTC_KV_METAL_BUILD_DIR"
ASTC_KV_METAL_BUILD_DIR="$(cd -- "$ASTC_KV_METAL_BUILD_DIR" && pwd)"
"$python" -c 'import json,sys,numpy; print(json.dumps({"executable":sys.executable,"version":sys.version,"numpy":numpy.__version__},indent=2))' \
  > "$out_dir/summary_python.json"
bash metal/probe_environment.sh > "$out_dir/environment_before_build.json"
bash metal/build.sh > "$out_dir/build.log" 2>&1
bash metal/probe_environment.sh > "$out_dir/environment.json"
IFS= read -r shader_path < "$ASTC_KV_METAL_BUILD_DIR/shader_path.txt"
for fixture in story_l17_v_token6 story_l17_kv_page6; do
  "$ASTC_KV_METAL_BUILD_DIR/astc-kv-probe" "results/metal_fixtures/$fixture/manifest.json" \
    "$shader_path" 30 10 256 > "$out_dir/$fixture.json" 2> "$out_dir/$fixture.stderr"
  "$python" metal/summarize.py "$out_dir/$fixture.json" > "$out_dir/${fixture}_summary.json"
done
echo "Mac diagnostic outputs saved in $out_dir. Inspect correctness before interpreting timing."
