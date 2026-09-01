#!/usr/bin/env bash
# SPDX-License-Identifier: MPL-2.0
# 一键构建 + 冒烟测试（Linux/macOS）
# 用法： bash scripts/build-and-test.sh [vulkan]
#   无参数  -> CPU-only 构建并测试
#   vulkan  -> 追加 Vulkan 后端构建与 GPU 测试（需要 Vulkan SDK）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${GGML_SRC:-$ROOT/../ggml-src}"      # 已应用 patch 的 ggml 源码目录
TEST="$ROOT/tests/test_learned_ops.c"

if [ ! -d "$SRC" ]; then
    echo "错误：找不到 ggml 源码目录 $SRC"
    echo "请先应用 patch： git clone <ggml 仓库> && git -C <目录> apply $ROOT/patches/learned-ops-ggml0190.patch"
    echo "或用环境变量 GGML_SRC 指向源码目录。"
    exit 1
fi

echo "== 1/3 配置 CPU 构建 =="
cmake -S "$SRC" -B "$SRC/build-cpu" -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=ON

echo "== 2/3 编译 =="
cmake --build "$SRC/build-cpu" --config Release --target ggml-base ggml-cpu -j

echo "== 3/3 编译并运行 CPU 冒烟测试 =="
cc -O2 -I"$SRC/include" -I"$SRC/src" "$TEST" \
   -L"$SRC/build-cpu/src" -lggml-cpu -lggml-base -lm -o "$SRC/test_learned_ops"
LD_LIBRARY_PATH="$SRC/build-cpu/src/bin:$SRC/build-cpu/src:${LD_LIBRARY_PATH:-}" \
    "$SRC/test_learned_ops" cpu

if [ "${1:-}" = "vulkan" ]; then
    VK_SDK="${VULKAN_SDK:-$HOME/VulkanSDK}"
    echo "== Vulkan 构建 =="
    cmake -S "$SRC" -B "$SRC/build-vk" -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON \
          -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=ON \
          -DCMAKE_PREFIX_PATH="$VK_SDK"
    cmake --build "$SRC/build-vk" --target ggml-base ggml-cpu ggml-vulkan -j
    # 链接 Vulkan 变体测试（USE_VULKAN 启用 vk 后端分支）
    cc -O2 -DUSE_VULKAN -I"$SRC/include" -I"$SRC/src" "$TEST" \
       -L"$SRC/build-vk/src" -L"$SRC/build-vk/src/ggml-vulkan" \
       -lggml-vulkan -lggml-cpu -lggml-base -lm -o "$SRC/test_learned_ops_vk"
    LD_LIBRARY_PATH="$SRC/build-vk/bin:$SRC/build-vk/src:$SRC/build-vk/src/ggml-vulkan:${LD_LIBRARY_PATH:-}" \
        "$SRC/test_learned_ops_vk" vk
fi

echo "全部完成。"
