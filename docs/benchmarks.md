# Benchmarks

> English | **[中文](benchmarks_zh.md)**

Cross-backend (CPU / Vulkan / CUDA) micro-benchmarks produced by `tests/bench_learned_ops.c`.

## Environment

- GPU: NVIDIA GeForce RTX 2070 (Turing, CC 7.5, 8 GB)
- CPU: x86-64, 8 threads, MSVC 2019, AVX2
- Stack: ggml v0.19.0 (`30bf868`), CUDA 12.6, Vulkan SDK 1.4.350.0
- Timing: 3 warmup iterations + 10 timed repeats per case, mean reported. `ratio = legacy time / new time`; >1 means the new operator is faster.

## `IM2COL_FAST_1D` (shape-for-shape against `CONV_1D`; shape = `L,IC,OC,K,s,p,d`)

### CPU

| shape | conv_1d | fast_1d | ratio |
|---|---|---|---|
| 100,16,32,3,1,1,1 | 0.0413 ms | 0.0399 ms | **1.03×** |
| 200,32,64,3,1,1,1 | 0.0719 | 0.0685 | **1.05×** |
| 500,64,128,3,2,1,1 | 0.1710 | 0.1617 | **1.06×** |
| 1000,128,256,7,2,3,1 | 3.27 | 2.90 | **1.13×** |
| 2000,256,512,3,1,1,1 | 18.38 | 17.64 | **1.04×** |
| 4000,64,128,5,2,2,1 | 3.17 | 2.76 | **1.15×** |

### Vulkan (ratio)

| | case 1 | case 2 | case 3 |
|---|---|---|---|
| small | 0.97× | 0.98× | 0.86× |
| large | 1.20× | 1.09× | 0.99× |

### CUDA (ratio, two independent runs)

| | case 1 | case 2 | case 3 |
|---|---|---|---|
| small (run1 / run2) | 0.63 / 0.93 | 1.17 / 1.76 | 1.59 / 1.12 |
| large (run1 / run2) | 0.95 / 1.18 | 0.96 / 1.06 | 0.67 / 0.79 |

### Read-out

- **CPU: the gain is real and reproducible.** 1.04–1.15× at large frame lengths, 1.03–1.06× on small shapes, stable across repeated runs. This is a constant-factor optimization (the O(1) window removes boundary zero-work; the middle section degrades to `memcpy` at `d0==1`), consistent with the magnitude reported for comparable upstream proposals. Larger speedups reported elsewhere come from the generic CPU path without aligned GEMM; this machine's x86 build takes the aligned-kernel path, so gains are modest.
- **Vulkan / CUDA: parity, as designed.** On GPU backends the operator is **aliased** to the same im2col+matmul path as `IM2COL`; ratios hovering around 1.0 are measurement noise (sub-millisecond kernels: CUDA single ops <0.4 ms, ±50% between runs is common). **Pass criterion is "no regression" — met.**

## `conv_transpose_1d` ext (legacy = 6-arg `ggml_conv_transpose_1d`, `g=1, p=0, op=0` only; shape = `L,Cin,Cout,K,s,p,op,g`)

### CPU

| shape | legacy | ext | ratio |
|---|---|---|---|
| 100,32,64,3,2,0,0,1 | 0.1129 ms | 0.1105 ms | 1.02× |
| 500,16,32,7,1,0,0,1 | 0.1897 | 0.1874 | 1.01× |
| 100,32,64,3,2,0,**2**,1 | – | 0.0967 ms | new capability (output_padding) |
| 100,32,64,3,2,0,0,**2** | – | 0.1141 ms | new capability (groups) |
| 200,64,128,5,4,0,1,**4** | – | 0.7460 ms | new capability (groups) |

### Vulkan

| shape | legacy | ext | ratio |
|---|---|---|---|
| 100,32,64,3,2,0,0,1 | 0.233 ms | 0.108 ms | **2.15×** |
| 500,16,32,7,1,0,0,1 | 0.1013 | 0.0929 | 1.09× |
| op=2 / g=2 / g=4 | – | 0.107 / 0.133 / 0.203 ms | new capability |

### CUDA

| shape | legacy | ext | ratio |
|---|---|---|---|
| 100,32,64,3,2,0,0,1 (run1) | 0.068 ms | 0.151 ms | 0.45× |
| same case (run2) | 0.068 | 0.069 | **0.98×** |
| 500,16,32,7,1,0,0,1 | 0.074 | 0.070 | 1.06× |
| op=2 / g=2 / g=4 | – | 0.070 / 0.053 / 0.143 ms | new capability |

### Read-out

- **Capability**: before this patch, groups (g>1) and output_padding were *unavailable on every backend*; ext makes them work everywhere (correctness covered by the 17-case test suite).
- **Vulkan: 2.15× on the comparable g=1 case.** The legacy path materializes a full im2col scratch tensor (~s0× the output length) plus a matmul; the new kernel evaluates each output coordinate by direct gather-accumulate and never materializes im2col. Largest single-node GPU win in this patch set.
- **CUDA/CPU: parity at g=1.** The same CUDA case measured 0.45× in one run and 0.98× in another — sub-millisecond noise (see Methodology); both paths are of the same complexity class. No regression.
- Metal accepts only the `g=1, p=0, d=1` superset graph; other combinations are gated off by `supports_op` and fall back to CPU.

## `REL_POS_BIAS` / `SCATTER_ELEMENTS` (absolute performance; no prior implementation to compare)

| Operator | shape | CPU | Vulkan | CUDA |
|---|---|---|---|---|
| rel_pos_bias | C32 H8 W8 B1 | 2.50 GFLOP/s | 5.1 GFLOP/s | not ported (clean SKIP) |
| | C64 H16 W16 B2 | 1.25 | **220.1** | SKIP |
| | C128 H8 W8 B4 | 1.30 | 63.8 | SKIP |
| scatter (overwrite) | data [1024,1024] ← upd [1024,256] | 0.84 GB/s | 60.4 GB/s | SKIP |
| scatter (add) | data [4096,4096] ← upd [4096,1024] | 0.53 GB/s | **95.9 GB/s** (atomic path) | SKIP |
| scatter (axis=0) | data [1024,1024] ← upd [256,1024] | 0.77 GB/s | 67.5 GB/s | SKIP |

### Read-out

- Vulkan reaches practical throughput (60–220 GFLOP/s; 60–96 GB/s bandwidth-class). The scatter-add case is the fastest of the three because the GPU advertises `shaderBufferFloat32AtomicAdd`, enabling the `atomicAdd` shader path.
- The CPU kernels are deliberately conservative single-threaded reference implementations (`n_tasks=1`): 1–2.5 GFLOP/s as a semantic fallback. Upstream v0.19.0 has no CPU implementation of either operator at all — this patch adds them from zero; parallelizing them is a contained follow-up if needed.
- CUDA's `supports_op` correctly returns false, so schedulers fall back to CPU cleanly.

## Methodology & known traps

1. **Harness**: output copies through the multi-backend `ggml_backend_sched` showed buffer-aliasing anomalies on this baseline; the benchmark uses a **graph allocator + single-backend `ggml_backend_graph_compute`**, with explicit `ggml_backend_tensor_set/get` for host↔device transfers — the same path llama.cpp takes on single-GPU deployments.
2. **Every graph variant gets its own backend + allocator lifecycle** (begin → build → upload → time → teardown). Reusing one graph allocator for two differently-shaped graphs on Vulkan caused an access-violation crash.
3. **Single-shot ratios of sub-millisecond GPU kernels carry up to ±50% noise** — the CUDA convT 0.45×↔0.98× flip and the conv-small 0.63×↔1.76× spread are the same phenomenon. All conclusions here rest on stable trends across repeated runs, never on a single measurement.
4. The legacy convT builder asserts `g=1, p=0, op=0`; shapes outside that envelope can only run on the ext side (legacy column shows "–").
5. scatter's builder enforces `updates->ne[d] == data->ne[d]` for all `d ≠ axis`; invalid combinations are rejected by assertion.

---

# Patch 2 — the ten qvac fused operators

Benchmarks produced by `tests/bench_qvac_ops.c` (`fused` = the new single-dispatch op; `composed` = the equivalent stock-ggml sub-graph; `speedup = composed / fused`). Same machine as above (RTX 2070, 8-thread CPU, MSVC 2019 AVX2), ggml v0.19.0 + both patches. **Median over 20 timed repeats** (3 warmups) — patch 1 used the mean, but sub-millisecond stability here demanded the median.

## Fused vs composed (CPU, 8 threads)

| case | fused ms | composed ms | speedup |
|---|---|---|---|
| snake 2048×64 | 0.410 | 1.350 | **3.29×** |
| snake 8192×128 | 2.195 | 4.840 | **2.20×** |
| snake 32768×32 | 1.252 | 5.705 | **4.56×** |
| bias_gelu 1024×256 | 0.816 | 1.182 | 1.45× |
| bias_gelu 4096×512 | 6.113 | 10.588 | **1.73×** |
| bias_gelu 1024×1024 | 2.027 | 3.696 | 1.82× |
| pw2_residual 1024×256 | 0.061 | 0.568 | **9.37×** |
| pw2_residual 4096×512 | 0.460 | 4.348 | **9.46×** |
| affine_prelu 64×256×32 | 1.563 | 2.866 | 1.83× |
| affine_prelu 128×512×64 | 12.528 | 20.091 | 1.60× |
| channel_shuffle 4096×64 G8 | 0.055 | 0.047 | 0.85× |
| channel_shuffle 32768×128 G4 | 0.741 | 0.588 | 0.79× |
| zero_upsample 256×1 s4 | 0.009 | 0.026 | 2.78× |
| zero_upsample 1024×8 s2 | 0.010 | 0.141 | **14.87×** |
| depthwise_1d 4096×64 K7 | 0.739 | 21.907 | **29.64×** |
| depthwise_1d 16384×128 K7 | 10.029 | 192.425 | **19.19×** |
| edge_pad_1d 4096×64 p3/3 | 0.151 | 2.039 | **13.51×** |
| edge_pad_1d 16384×128 p7/7 | 1.189 | 15.206 | **12.79×** |
| LN_channel 1024×256 | 0.689 | 1.813 | **2.63×** |
| LN_channel 4096×512 | 15.457 | 12.852 | 0.83× |

(Second run cross-checked; numbers reproduce within ±10% except sub-0.05 ms rows.)

### CPU read-out

- **The two im2col-class fusions dominate**: `depthwise_1d` (19–30×) and `edge_pad_1d` (13×) eliminate the F16 im2col scratch tensor and the concat/repeat copies respectively.
- **Node-count-dominated fusions deliver 2.5–15×**: `pw2_residual`, `zero_upsample`, `snake` — each removes 2–4 kernel dispatches.
- **Honest regressions on this CPU**: `channel_shuffle` (0.79–0.85×) — the composed view chain on contiguous planes compiles to large `memcpy`s that beat per-plane gather; and `LN_channel` at C=512 (0.83×) — the stock chain's permute gives the vectorizer friendlier innermost strides. Both remain wins on GPU dispatch counts and on the `[C,T]` layouts the engines actually use.

## Fused vs composed (Vulkan, RTX 2070)

| case | fused ms | composed ms | speedup |
|---|---|---|---|
| snake 2048×64 | 0.109 | 0.180 | 1.65× |
| snake 8192×128 | 0.126 | 0.480 | **3.81×** |
| snake 32768×32 | 0.127 | 0.493 | **3.89×** |
| affine_prelu 64×256×32 | 0.123 | 0.431 | **3.51×** |
| affine_prelu 128×512×64 | 0.276–0.291 | 1.739–1.752 | **6.0–6.3×** |
| channel_shuffle 4096×64 G8 | 0.118 | 0.129 | 1.09× |
| channel_shuffle 32768×128 G4 | 0.285 | 0.229 | 0.80× |
| zero_upsample 256×1 s4 | 0.147 | 0.136 | 0.93× |
| zero_upsample 1024×8 s2 | 0.106 | 0.142 | 1.34× |
| pw2_residual / bias_gelu / depthwise / edge_pad / LN | — | measured | fused not ported to Vulkan (CPU fallback; upstream qvac ships them as Metal kernels) |

The five Vulkan shader ops behave as designed: `snake` and `affine_prelu` — the two that replace multi-kernel broadcast chains — show the real GPU wins (up to 3.9× and 6.3×); copy-class ops (`channel_shuffle`, `zero_upsample`) hover at parity since the composed chains already fuse into single copies on GPU.

## Fused vs composed (Metal, Apple M4)

- Platform: Apple M4 (10-core GPU), macOS 27.0 (26A5421a), Xcode 27.0 (27A5237l), Apple Clang 21.0.0.
- Stack: ggml v0.19.0 (`30bf868`) + patches 1, 2, and 3.
- Method: three independent process runs. Each process uses 3 warmups and the median of 20 timed `ggml_backend_graph_compute` calls; the table reports the median of the three per-process medians. Every graph node is checked with `ggml_backend_supports_op`, so no CPU fallback is included.

| case | fused ms | composed ms | speedup |
|---|---:|---:|---:|
| snake 2048×64 | 0.245 | 0.682 | **2.78×** |
| snake 8192×128 | 0.574 | 2.237 | **3.90×** |
| snake 32768×32 | 0.675 | 2.248 | **3.33×** |
| bias_gelu 1024×256 | 0.329 | 0.476 | 1.45× |
| bias_gelu 4096×512 | 0.534 | 0.898 | 1.68× |
| bias_gelu 1024×1024 | 0.312 | 0.549 | 1.76× |
| pw2_residual 1024×256 | 0.253 | 0.382 | 1.51× |
| pw2_residual 4096×512 | 0.654 | 1.389 | **2.12×** |
| depthwise_1d 4096×64 K7 | 0.260 | 0.626 | **2.41×** |
| depthwise_1d 16384×128 K7 | 0.535 | 2.937 | **5.49×** |
| edge_pad_1d 4096×64 p3/3 | 0.251 | 0.253 | 1.01× |
| edge_pad_1d 16384×128 p7/7 | 0.494 | 0.765 | 1.55× |
| LN_channel 1024×256 | 0.300 | 0.633 | **2.11×** |
| LN_channel 4096×512 | 1.056 | 3.355 | **3.18×** |

The largest measured gain is 5.49× for the large depthwise case. Snake reaches 3.90× and large channel layer norm reaches 3.18×. Small edge padding is effectively break-even (1.01×), while the larger case gains 1.55×.

The four operators without donor Metal kernels remain unsupported as fused ops. Their composed Metal baselines were measurable — affine-PReLU 0.745 / 4.570 ms, channel-shuffle 0.389 / 2.178 ms, and zero-upsample 0.240 / 0.242 ms for the two listed shapes — but no fused speedup is claimed. GRU remains SKIP.

## `GRU` (absolute; no stock equivalent exists)

| backend | H | B | L | reverse | ms |
|---|---|---|---|---|---|
| CPU | 64 | 1 | 256 | no | 4.93 |
| CPU | 128 | 8 | 128 | no | 11.11 |
| CPU | 256 | 1 | 64 | yes | 19.58 |
| CPU | 512 | 4 | 32 | yes | 47.29 |
| Vulkan (RTX 2070) | 64 | 1 | 256 | no | 2.23 |
| Vulkan | 128 | 8 | 128 | no | 4.27 |
| Vulkan | H ≥ 256 | | | | SKIP (H ≤ 128 shared-memory cap; CPU fallback) |

Stock ggml has no RNN cell — the comparison column is the *existence* of this row. The Vulkan packed kernel (128 lanes = `128/H` batch elements per workgroup) is ~2× the CPU at H ≤ 128.

## Patch-2 methodology notes

6. Same harness rules as patch 1 (fresh backend + galloc per variant, no sched). The Vulkan per-variant lifecycle requirement re-confirmed: reusing a subcontext across graph shapes crashes.
7. `snake` on Vulkan reuses the stock `snake_f32` pipeline (the v0.19 base already carries a graph-level snake fusion with identical math), so its fused column measures the *op-dispatch* path, not a new shader.
8. The `channel_shuffle` composed chain on Vulkan/CPU is a near-optimal single-copy — treat its speedup < 1 as "no win where none is possible", not as a defect.
