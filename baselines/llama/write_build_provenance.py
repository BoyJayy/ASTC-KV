"""Record local pinned-source/build identity without installing or executing GPU code."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

project, build = map(lambda s: Path(s).resolve(), sys.argv[1:3])
cmake = sys.argv[3]
revision = "c3df4b7e22a1ac38da663552d19e649dd97ca936"
upstream = project / "third_party" / f"llama.cpp-{revision}"


def digest(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def command(*args):
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return {"argv": list(args), "returncode": result.returncode,
            "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


files = [
    project / "baselines/llama/probe.mm",
    project / "baselines/llama/CMakeLists.txt",
    project / "scripts/build_llama_baseline.sh",
    project / "baselines/llama/write_build_provenance.py",
    build / "bin/astc-llama-baseline",
    build / "CMakeCache.txt",
    build / "compile_commands.json",
    upstream / "ggml/src/ggml-metal/ggml-metal-device.m",
    upstream / "ggml/src/ggml-metal/ggml-metal-device.cpp",
    upstream / "ggml/src/ggml-metal/ggml-metal-ops.cpp",
    upstream / "ggml/src/ggml-metal/kernels/fa_vec_common.metal",
    upstream / "ggml/src/ggml-metal/kernels/fa_vec_f16.metal",
    upstream / "ggml/src/ggml-metal/kernels/fa_vec_q4_0.metal",
    upstream / "ggml/src/ggml-quants.c",
]
download = json.loads((project / "third_party/llama_baseline_provenance.json").read_text())
assert download["revision"] == revision
archive = project / "third_party" / f"llama-{revision}.tar.gz"
assert digest(archive) == download["archive_sha256"]
result = {
    "schema": "astc-kv-llama-baseline-build-v1", "pinned_source": download,
    "source_modifications": "none; harness links unmodified upstream ggml",
    "sha256": {str(p.relative_to(project)): digest(p) for p in files},
    "compiler": command("xcrun", "clang++", "--version"),
    "sdk": command("xcrun", "--sdk", "macosx", "--show-sdk-version"),
    "macos": command("sw_vers"), "cmake": command(cmake, "--version"),
    "configure": "Exact options in scripts/build_llama_baseline.sh; CMakeCache and compile_commands hashes above",
    "metal_library": "Upstream GGML_METAL_EMBED_LIBRARY=ON embeds MSL sources; runtime compile, no offline metal compiler",
    "gpu_execution": "not performed by this build script",
}
(build / "build-provenance.json").write_text(json.dumps(result, indent=2) + "\n")
