#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0
# GGML_SRC=/path/to/patched/ggml bash scripts/build-and-test-metal.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${GGML_SRC:-$ROOT/../ggml-src}"
SRC="$(cd "$SRC" && pwd)"
BUILD="${GGML_AUDIO_TEST_BUILD:-$ROOT/build-metal-tests}"
cmake -S "$ROOT/tests" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
    -DGGML_SOURCE_DIR="$SRC" -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_NATIVE=OFF \
    -DGGML_CCACHE=OFF -DBUILD_SHARED_LIBS=ON
cmake --build "$BUILD" --parallel
ctest --test-dir "$BUILD" --output-on-failure
# Repeat Metal execution to catch transient GPU failures.
ctest --test-dir "$BUILD" --output-on-failure -R metal
