# Building

> English | **[中文](building_zh.md)**

Prerequisites, patch application, per-backend build recipes, and known toolchain pitfalls.

## Prerequisites

| Component | Required for | Notes |
|---|---|---|
| CMake ≥ 3.14 | everything | |
| C17 / C++17 compiler | everything | GCC, Clang, or MSVC (Visual Studio 2019 BuildTools or newer) |
| [Vulkan SDK](https://vulkan.lunarg.com/) ≥ 1.3.x | `GGML_VULKAN=ON` | must provide `glslc` in `PATH` (or via the SDK's `Bin`) |
| CUDA toolkit | `GGML_CUDA=ON` | 12.x recommended; see the multi-toolkit pitfall below |
| Ninja | Windows builds | bundled with Visual Studio's CMake component |

## 1. Apply the patches

```bash
git clone https://github.com/ggml-org/ggml.git ggml-src
cd ggml-src
git checkout 30bf868                       # v0.19.0
git apply <path-to>/patches/learned-ops-ggml0190.patch   # patch 1 (required)
git apply <path-to>/patches/qvac-ops-ggml0190.patch      # patch 2 (optional, sequential)
git apply <path-to>/patches/metal-ops-ggml0190.patch     # patch 3 (optional Metal integration)
git apply <path-to>/patches/vulkan-conv-direct-1d-ggml0190.patch  # 4
git apply <path-to>/patches/vulkan-pipeline-cache-ggml0190.patch  # 5
git apply <path-to>/patches/metal-conv-direct-1d-ggml0190.patch   # 6 (Metal)
git apply <path-to>/patches/audio-op-fixes-ggml0190.patch        # 7
```

Patch 2 must be applied **after** patch 1 (they touch the same enum-assert and dispatch hunks), and patch 3 must follow patch 2. Applying only patch 1 is fine; patch 2 and patch 3 cannot be applied independently of their predecessors.

All further `cmake -S <dir>` commands below point at this patched tree.

Patch 6 requires patches 1, 2 and 3. Metal-only order is 1 → 2 → 3 → 6; the full stack is 1 → 2 → 3 → 4 → 5 → 6. From this repository, build and test CPU/Metal with:

```bash
GGML_SRC=../ggml-src bash scripts/build-and-test-metal.sh
```

The script embeds shader source, runs both CPU/Metal suites, and repeats Metal tests. See [Metal direct convolution](metal-direct-conv.md) for capability gates and raw-graph fallback requirements.


## 2. CPU-only build

```bash
cmake -S ggml-src -B ggml-src/build-cpu -DCMAKE_BUILD_TYPE=Release \
      -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=ON
cmake --build ggml-src/build-cpu --config Release --target ggml-base ggml-cpu -j
```

> `GGML_NATIVE=OFF` builds a portable baseline ISA; turn it `ON` if the binary never leaves the build machine. This patch set deliberately pins the pre-v0.20 layout where CPU ISA variants are still statically linked (v0.20 turned them into dlopen'd module plugins, mutually exclusive with `GGML_NATIVE=ON`).

## 3. Vulkan build

```bash
cmake -S ggml-src -B ggml-src/build-vk -DCMAKE_BUILD_TYPE=Release \
      -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=ON
cmake --build ggml-src/build-vk --target ggml-base ggml-cpu ggml-vulkan -j
```

**Windows pitfall — use the Ninja generator.** The Visual Studio (MSBuild) generator cannot handle the `DEPFILE` dependency in ggml's Vulkan shader compilation rule:

```powershell
cmake -S ggml-src -B ggml-src/build-vk -G Ninja -DCMAKE_BUILD_TYPE=Release `
      -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=ON
```

Make sure `VULKAN_SDK` is set (the official installer does this) and `glslc` resolves.

## 4. CUDA build

```bash
cmake -S ggml-src -B ggml-src/build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=75
```

Pick `-DCMAKE_CUDA_ARCHITECTURES` for your GPU (75 = Turing, 86 = Ampere consumer, 89 = Ada, 120 = Blackwell). Non-Ninja generators work on Linux.

**Windows pitfall — multiple CUDA toolkits + MSBuild generator.** When several CUDA versions are installed side by side, MSBuild resolves CUDA through VS's `CUDA <ver>.targets` files, selected by the `CUDA_PATH_V<major>_<minor>` environment variables — and it picks the *newest* one, even when you asked CMake for another. A too-new toolkit with an older MSVC breaks the build (observed: CUDA 13.0 + VS2019 toolset fails in `crt`/`cuda_runtime` headers with `error C4002: __cudaLaunch`'s macro). Two fixes, either one works:

- use the **Ninja generator** (bypasses the MSBuild `.targets` mechanism entirely), **and**
- pin the compiler explicitly, using **forward slashes** in `-D` paths (backslashes get mangled by CMake's escape parsing):

```powershell
cmake ... -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin/nvcc.exe"
```

The same pitfall and remedy are documented independently by the [game.cpp BUILDING.md](https://github.com/KakaruHayate/game.cpp/blob/main/BUILDING.md) notes.

## 5. Smoke tests

`tests/test_learned_ops.c` checks patch-1's four operators against hand-computed references (17 cases, tolerance 1e-3). `tests/test_qvac_ops.c` checks patch-2's ten operators (CPU: all ten; on Vulkan the five shader ops run on device and the five CPU-only supertonic ops report a clean SKIP — that is the expected pass state). The bundled scripts automate configure+build+test:

```bash
bash scripts/build-and-test.sh            # CPU
bash scripts/build-and-test.sh vulkan     # CPU + Vulkan
```

```powershell
.\scripts\build-and-test.ps1              # CPU
.\scripts\build-and-test.ps1 -Vulkan      # CPU + Vulkan
```

To wire a test by hand, compile it against the patched ggml headers and link `ggml-base`, `ggml-cpu` (plus `ggml-vulkan` / `ggml-cuda` when enabled), making sure the shared libraries are on the loader path. Each test binary accepts a backend name and a thread count:

```
test_learned_ops [cpu | vulkan | cuda] [threads]
test_qvac_ops    [cpu | vulkan]           # add -DUSE_VULKAN when compiling the vk variant
```

Expected tail output per backend:

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

## 6. Benchmarks

`tests/bench_learned_ops.c` and `tests/bench_qvac_ops.c` compile the same way (`USE_VULKAN` / `USE_CUDA` / `USE_METAL` defines). Run with the matching backend libraries available:

```
bench_learned_ops [cpu | vulkan | cuda] [threads]
bench_qvac_ops    [cpu | vulkan | metal] [threads]
```

Notes for trustworthy numbers (all learned the hard way, see [benchmarks.md](benchmarks.md) §Methodology):

- give every graph variant its own backend + graph-allocator lifecycle;
- Vulkan: do not reuse one allocator across two differently-shaped graphs;
- treat sub-millisecond GPU timings with suspicion — repeat and compare medians.
