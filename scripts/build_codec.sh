#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if command -v cmake >/dev/null 2>&1; then
  cmake_cmd=(cmake)
else
  cmake_cmd=("${PYTHON:-python3}" -m cmake)
fi
"${cmake_cmd[@]}" -S third_party/astc-encoder-5.7.0 -B build/astc \
  -DCMAKE_BUILD_TYPE=Release -DASTCENC_CLI=OFF -DASTCENC_SHAREDLIB=OFF \
  -DASTCENC_ISA_NATIVE=ON -DASTCENC_UNIVERSAL_BUILD=OFF \
  -DASTCENC_WERROR=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
"${cmake_cmd[@]}" --build build/astc --parallel 4
if [[ "$(uname -s)" == Darwin ]]; then
  c++ -O3 -std=c++17 -dynamiclib -fPIC astc_kv/bridge.cpp \
    -I third_party/astc-encoder-5.7.0/Source build/astc/Source/libastcenc-native-static.a \
    -o astc_kv/libakv_astc.dylib
else
  c++ -O3 -std=c++17 -shared -fPIC astc_kv/bridge.cpp \
    -I third_party/astc-encoder-5.7.0/Source build/astc/Source/libastcenc-native-static.a \
    -o astc_kv/libakv_astc.so
fi
