# 构建须知

> **[English](building.md)** | 中文

前置依赖、patch 应用方法、分后端构建命令与已知工具链坑。

## 前置依赖

| 组件 | 用途 | 备注 |
|---|---|---|
| CMake ≥ 3.14 | 全部 | |
| C17 / C++17 编译器 | 全部 | GCC、Clang 或 MSVC（Visual Studio 2019 BuildTools 及以上） |
| [Vulkan SDK](https://vulkan.lunarg.com/) ≥ 1.3.x | `GGML_VULKAN=ON` | 需能在 `PATH`（或 SDK 的 `Bin`）中找到 `glslc` |
| CUDA 工具包 | `GGML_CUDA=ON` | 推荐 12.x；多版本共存的坑见下文 |
| Ninja | Windows 构建 | Visual Studio 的 CMake 组件自带 |

## 1. 应用 patch

```bash
git clone https://github.com/ggml-org/ggml.git ggml-src
cd ggml-src
git checkout 30bf868                       # v0.19.0
git apply <本仓库路径>/patches/learned-ops-ggml0190.patch   # 补丁一（必需）
git apply <本仓库路径>/patches/qvac-ops-ggml0190.patch      # 补丁二（可选，顺序应用）
git apply <本仓库路径>/patches/metal-ops-ggml0190.patch     # 补丁三（可选 Metal 集成）
git apply <本仓库路径>/patches/vulkan-conv-direct-1d-ggml0190.patch  # 4
git apply <本仓库路径>/patches/vulkan-pipeline-cache-ggml0190.patch  # 5
git apply <本仓库路径>/patches/metal-conv-direct-1d-ggml0190.patch   # 6 (Metal)
git apply <本仓库路径>/patches/audio-op-fixes-ggml0190.patch        # 7
```

补丁二必须**在补丁一之后**应用（两者触碰相同的枚举断言与分发代码块），补丁三必须跟在补丁二之后。只用补丁一完全可行；补丁二或补丁三都不能脱离前序补丁单独使用。

下文中所有 `cmake -S <dir>` 都指向这份打过 patch 的源码树。

补丁六需要一、二、三；只用 Metal 可按 1 → 2 → 3 → 6 应用，完整集合按 1 → 2 → 3 → 4 → 5 → 6。Metal 构建与 CPU/GPU 回归可在本仓库目录执行：

```bash
GGML_SRC=../ggml-src bash scripts/build-and-test-metal.sh
```

脚本启用嵌入 shader，运行两套 CPU/Metal 测试并重复 Metal 测试。GPU 能力门控和直接执行图的回退要求见 [Metal 直接卷积](metal-direct-conv_zh.md)。


## 2. CPU-only 构建

```bash
cmake -S ggml-src -B ggml-src/build-cpu -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=ON
cmake --build ggml-src/build-cpu --config Release --target ggml-base ggml-cpu -j
```

> `GGML_NATIVE=OFF` 生成可移植的基线 ISA 构建；只在构建机上运行可开 `ON`。本补丁集刻意锚定 v0.20 之前的布局——v0.20 把 CPU ISA 变体做成了 dlopen 的 module 插件（与 `GGML_NATIVE=ON` 互斥），这也是选择 v0.19.0 作为基线的原因。

## 3. Vulkan 构建

```bash
cmake -S ggml-src -B ggml-src/build-vk -DCMAKE_BUILD_TYPE=Release \
      -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=ON
cmake --build ggml-src/build-vk --target ggml-base ggml-cpu ggml-vulkan -j
```

**Windows 坑——请用 Ninja 生成器。** Visual Studio（MSBuild）生成器无法处理 ggml Vulkan shader 编译规则里的 `DEPFILE` 依赖：

```powershell
cmake -S ggml-src -B ggml-src/build-vk -G Ninja -DCMAKE_BUILD_TYPE=Release `
      -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=ON
```

确保 `VULKAN_SDK` 环境变量存在（官方安装器会设置），且 `glslc` 可解析。

## 4. CUDA 构建

```bash
cmake -S ggml-src -B ggml-src/build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=75
```

`-DCMAKE_CUDA_ARCHITECTURES` 按显卡选择（75 = Turing，86 = Ampere 消费级，89 = Ada，120 = Blackwell）。Linux 下不限定 Ninja。

**Windows 坑——多版本 CUDA 共存 + MSBuild 生成器。** 当系统装了多个 CUDA 版本时，MSBuild 通过 VS 的 `CUDA <ver>.targets` 文件解析 CUDA，由 `CUDA_PATH_V<major>_<minor>` 环境变量决定——而且总是挑**最新**的那套，即使你在 CMake 里另有所指。用过新的工具包配旧 MSVC 会直接编译炸（实测：CUDA 13.0 + VS2019 工具集在 `cuda_runtime` 头里报 `error C4002: __cudaLaunch` 宏参数错误）。两种修法，任选其一或叠加：

- 使用 **Ninja 生成器**（完全绕开 MSBuild 的 `.targets` 机制），**并且**
- 显式钉住编译器路径，`-D` 参数里用**正斜杠**（反斜杠会被 CMake 转义解析弄坏）：

```powershell
cmake ... -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe"
```

同一坑与解法在 [game.cpp 的 BUILDING.md](https://github.com/KakaruHayate/game.cpp/blob/main/BUILDING.md) 中有独立记录。

## 5. 冒烟测试

`tests/test_learned_ops.c` 用手写参考值核对补丁一的全部四个算子（17 个用例，容差 1e-3）。`tests/test_qvac_ops.c` 核对补丁二的十个算子（CPU：全部十个；Vulkan 下五个 shader 算子跑在设备上、五个纯 CPU 的 supertonic 算子会打印 SKIP——这就是预期的通过状态）。仓库自带脚本一键完成 配置+编译+测试：

```bash
bash scripts/build-and-test.sh            # CPU
bash scripts/build-and-test.sh vulkan     # CPU + Vulkan
```

```powershell
.\scripts\build-and-test.ps1              # CPU
.\scripts\build-and-test.ps1 -Vulkan      # CPU + Vulkan
```

手动接线方式：对着打过 patch 的 ggml 头文件编译测试源，链接 `ggml-base`、`ggml-cpu`（启用 GPU 时再加 `ggml-vulkan` / `ggml-cuda`），并保证动态库在加载路径上。测试程序接受后端名与线程数两个参数：

```
test_learned_ops [cpu | vulkan | cuda] [threads]
test_qvac_ops    [cpu | vulkan]           # vk 变体编译时加 -DUSE_VULKAN
```

每个后端的预期结尾输出：

```
== learned-ops smoke tests (backend: cpu) ==
[test] conv_1d vs conv_1d_fast_1d_im2col parity        ... done
[test] conv_transpose_1d_ext (output_padding / groups) ... done
[test] scatter_elements (overwrite / add, axis 0/1)    ... done
[test] rel_pos_bias vs naive reference                 ... done
ALL PASSED
```

```
== qvac-ops smoke tests (backend: cpu) ==
[test] supertonic_depthwise_1d / _ct / _causal_ct   ... done
[test] supertonic_layer_norm_channel / _ct          ... done
[test] supertonic_pw2_residual / _ct                ... done
[test] supertonic_bias_gelu / _ct                   ... done
[test] supertonic_edge_pad_1d / _ct                 ... done
[test] gru (forward + reverse)                      ... done
[test] zero_upsample                                ... done
[test] channel_shuffle                              ... done
[test] affine_prelu                                 ... done
[test] snake                                        ... done
ALL PASSED
```

## 6. 性能基准

`tests/bench_learned_ops.c` 与 `tests/bench_qvac_ops.c` 编译方式同上（按需加 `USE_VULKAN` / `USE_CUDA` / `USE_METAL` 宏）。运行时确保对应后端库可用：

```
bench_learned_ops [cpu | vulkan | cuda] [threads]
bench_qvac_ops    [cpu | vulkan | metal] [threads]
```

取信数据的注意事项（都是踩坑换来的，详见 [benchmarks_zh.md](benchmarks_zh.md) §方法论）：

- 每个图变体使用**独立的 backend + 图分配器生命周期**；
- Vulkan 下不要在两个不同 shape 的图之间复用同一个分配器；
- 亚毫秒的 GPU 计时不可尽信——多跑几轮取中位趋势再下结论。
