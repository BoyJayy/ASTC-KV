#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${ASTC_KV_METAL_BUILD_DIR:-$project_dir/build/metal}"
if [[ "$(uname -s)" != "Darwin" ]]; then
    echo '{"platform":"not_macos","metal_gpu_status":"unavailable_in_current_environment"}'
    exit 2
fi
python3 - "$build_dir/astc-kv-probe" <<'PY'
import datetime
import importlib.metadata
import json
import platform
import subprocess
import sys

def command(*args):
    try:
        p = subprocess.run(args, text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc)}
    return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}

def hardware():
    value = command("system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json")
    if value["returncode"] != 0:
        return value
    def prune(obj):
        if isinstance(obj, dict):
            return {k: prune(v) for k, v in obj.items()
                    if not any(s in k.lower() for s in ("serial", "uuid", "udid"))}
        if isinstance(obj, list):
            return [prune(v) for v in obj]
        return obj
    return prune(json.loads(value["stdout"]))

versions = {}
for package in ("mlx", "mlx-lm", "numpy", "transformers", "safetensors"):
    try:
        versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        versions[package] = None
result = {
    "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "python": sys.version,
    "machine": platform.machine(),
    "macos": command("sw_vers"),
    "xcode": command("xcodebuild", "-version"),
    "sdk_version": command("xcrun", "--sdk", "macosx", "--show-sdk-version"),
    "sdk_path": command("xcrun", "--sdk", "macosx", "--show-sdk-path"),
    "metal_compiler": command("xcrun", "metal", "--version"),
    "power_settings": command("pmset", "-g", "custom"),
    "power_source": command("pmset", "-g", "batt"),
    "hardware": hardware(),
    "dependency_versions": versions,
}
try:
    probe = command(sys.argv[1], "--environment")
    result["metal_device"] = json.loads(probe["stdout"]) if probe["returncode"] == 0 else probe
except FileNotFoundError:
    result["metal_device"] = {"status": "probe_not_built"}
print(json.dumps(result, indent=2))
PY
