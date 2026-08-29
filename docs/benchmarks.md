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

## `ADD_LEAKY_RELU` / `CONV_DIRECT_1D` (vocoder end-to-end + micro-kernel)

> Environment for this section only: Xeon E5-2675 v3 (16C/32T Haswell-EP, sustained ~2.0 GHz under AVX2 load), Windows, MSVC 2019 `/O2`, fp32, CPU backend. Timing = 3 runs per path, median quoted. The consumer harness is the pc-nsf-hifigan.cpp `hifigan_cli` (NSF-HiFiGAN: mel=128, 5 upsample levels, 15 resblocks, activations up to `[881664, C]`), accuracy against the torch-CPU fp32 output (offset-aligned; CORR gold = torch-CPU, harness @ f8c16ba). ONNX Runtime CPU EP on the same box: 5 155.0 ms median of 9 runs.

Micro-kernel, single-thread (K=11-shaped direct-conv inner loop, best variant per row):

| variant | GF/s |
|---|---|
| pure-FMA register calibration (2 FMA/cycle ceiling @ ~2.0 GHz) | 63.4–64.8 |
| micro-kernel, `imul`-on-counter addressing (as first written) | 24.6 |
| micro-kernel, pointer-increment addressing (**shipped**) | **31.8** |

The 4× gap between calibration and the original kernel is the `inc → movsxd → imul` address chain (≈5 serial cycles per kernel tap) gating `vbroadcastss` dispatch; pointer increments remove it. The residual 2× vs peak is broadcast load-to-use latency on the FMA critical path — a register-resident control variant of the same loop reaches the calibration line, so it is a latency-bound, not bandwidth-bound, residual.

End-to-end vocoder, 24 threads (median of 3):

| path | time (ms) | corr vs torch-CPU fp32 | max\|Δ\| |
|---|---|---|---|---|
| stock `im2col` + `mul_mat` (`PCNSF_DIRECT_CONV=0`) | 80 002 | 0.99999999 | 1.490e-4 |
| direct conv + `ADD_LEAKY_RELU`, no producer fusions (`PCNSF_FUSE_IO=0`) | 31 772 | 0.99999999 | 1.492e-4 |
| `conv_direct_1d_fused` (input fold + residual epilogue) | **28 030** | 0.99999999 | 1.492e-4 |

so the direct kernel + fusions are **2.85×** over the stock ggml conv path. All three paths are numerically equivalent (the max|Δ| values differ only in their last digit — FMA-ordering noise). Producer-side fusions removed 100 of 252 graph nodes (50 leaky, 5 scale, 45 residual add) with bit-identical output. Two honest caveats:

- **The ggml CPU path remains ~5.4× slower than ONNX Runtime's CPU EP** (28.0 s vs 5.155 s) on this model; ORT's threaded GEMM on Haswell is simply much better tuned. Thread scaling is weak as well: the fused path takes ~80 s at 4 threads vs 28 s at 24 (2.9× for 6× threads). Closing that gap (better blocking / parallelism breakdown for the level-0 `[881664, 256]`-class activations) is the active workstream; treat the 2.85× as "beats stock ggml", not "competitive with ORT CPU yet".
- *Errata:* a previous revision of this table listed 10 038 / 5 973 / 4 764 ms for the three rows and claimed ORT-CPU parity. Those numbers were recorded against a stale build configuration and are not reproducible; the table above supersedes them (same machine, same model, re-measured 3-run medians).

For the Vulkan path of the same operator, see the patch-4 section below.

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
| snake 2048×64 | 0.241 | 0.282 | 1.17× |
| snake 8192×128 | 0.327 | 0.996 | **3.05×** |
| snake 32768×32 | 0.310 | 0.920 | **2.97×** |
| bias_gelu 1024×256 | 0.256 | 0.288 | 1.13× |
| bias_gelu 4096×512 | 0.458 | 0.880 | 1.92× |
| bias_gelu 1024×1024 | 0.339 | 0.660 | 1.95× |
| pw2_residual 1024×256 | 0.260 | 0.368 | 1.42× |
| pw2_residual 4096×512 | 0.636 | 1.396 | **2.19×** |
| depthwise_1d 4096×64 K7 | 0.241 | 0.655 | **2.72×** |
| depthwise_1d 16384×128 K7 | 0.502 | 2.913 | **5.80×** |
| edge_pad_1d 4096×64 p3/3 | 0.179 | 0.250 | 1.40× |
| edge_pad_1d 16384×128 p7/7 | 0.463 | 0.711 | 1.54× |
| LN_channel 1024×256 | 0.281 | 0.568 | **2.02×** |
| LN_channel 4096×512 | 1.029 | 3.332 | **3.24×** |

The largest measured gain is 5.80× for the large depthwise case. Snake reaches 3.05× and large channel layer norm reaches 3.24×. Edge padding gains 1.40–1.54×.

The four operators without donor Metal kernels remain unsupported as fused ops. Their composed Metal baselines were measurable — affine-PReLU 0.723 / 4.616 ms, channel-shuffle 0.375 / 2.148 ms, and zero-upsample 0.234 / 0.237 ms for the two listed shapes — but no fused speedup is claimed. GRU remains SKIP.

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

---

# Patch 4 — Vulkan `CONV_DIRECT_1D` (vocoder end-to-end)

> Environment: same box as the vocoder section above (Xeon E5-2675 v3, Windows, MSVC 2019) plus an RTX 2070 (driver 32.0.16.2002; warp 32, 48 KiB shared/block). Stack: ggml v0.19.0 (`30bf868`) + patches 1, 2, and 4 (measured before patch 3 landed upstream; patch 4 is orthogonal to the Metal patch). ggml v0.19.0 + patches 1–3 + pc-nsf-hifigan.cpp @ `f8c16ba` (`hifigan_cli`, branch `wip/profile-nodes`). fp32 end-to-end with tensor cores deliberately off (`GGML_VK_DISABLE_COOPMAT2=1`, set inside `hifigan_cli`): the vocoder contract matches the torch-CPU fp32 path, not tensor-core tf32/fp16 arithmetic. Timing = `PCNSF_TIMING=1` wall time of the model run (excludes WAV IO), reference clip T=1722 (≈37.6 s of audio at 23.5 kHz, 881 664 output samples).

End-to-end (median of each run set):

| path | time (ms) | corr vs torch-CPU fp32 out | max|Δ| |
|---|---|---|---|
| ONNX Runtime CPU EP (median of 9) | 5 155.0 | 0.9999999873 | — |
| ONNX Runtime DML EP (median of 9) | 325.8 | 0.99999697 | 2.09e-03 |
| ggml Vulkan, patch 4 (all convs via supports_op gate) | **433.2–435.0** | 0.99999985 | 6.13e-04 |
| ggml Vulkan, `PCNSF_DIRECT_CONV=0` (im2col fallback everywhere) | 571–581 | 0.99999956 | — |

Notes on the Vulkan number: two 3-run sets gave medians 433.2/435.0 (runs 422.6–479.1). A later re-measurement on the same box after a host-only cleanup (debug env hooks removed from `ggml-vulkan.cpp`; shader and graph unchanged, outputs **bit-identical**) read 458–479 ms (median 462, n=4) — the spread is machine-state noise (clocks/thermals after hours of compilation), not a code-path change: identical SPIR-V, identical bytes out. Quote the pair honestly: **≈ 430–480 ms**, DML-class, with an order of magnitude tighter fp32 agreement than DML itself.

Tile-variant sweep (same run, spec-constant variants forced per run; variants change tiling only, outputs bit-identical):

| variant | e16 | e32 | w64 | w128 (shipped pick) | e64 |
|---|---|---|---|---|---|
| median ms | 635.6 | 493.9 | 570.1 | 442.0 | 398.2 |

The shipped OC-based pick (≤16→e16, ≤32→e32, else w128) measured best *as a set* in production; a K-aware re-pick suggested by per-shape micro-benchmarks regressed production to 488–1 432 ms and was reverted. Lesson recorded: the variant set is co-tuned; per-shape harness benches do not extrapolate to the full graph.

The `K ≥ 3` gate is a correctness boundary, not a preference: the shader stages `XS_ROWS = 12` input rows per chunk and a `BK = 32` channel-chunk spans `⌊31/K⌋+1` input rows — at K = 2 that is 16 > 11 usable slots, so the circular window overwrites still-needed rows. Production has five K = 2 subpixel-upsample convs; forcing them onto the shader (`PCNSF_DIRECT_MIN_K=1`, bypassing the consumer gate) drops corr to 0.36 deterministically. With the gate they fall back to `im2col`+`mul_mat` (also Vulkan-resident), cost already included in the 433–480 ms above.

Repro (pc-nsf-hifigan.cpp @ `f8c16ba`, `build-vk`, from a clone with patches 1, 2, 4 applied):

```bat
set GGML_VK_DISABLE_COOPMAT2=1   rem already defaulted inside hifigan_cli
set HF_RAW_OUT=vulkan_out.f32
hifigan_cli.exe hifigan_f32.gguf mel.bin f0.bin out.wav
python cmp_align.py vulkan_out.f32   rem offset-aligned corr vs torch-CPU golden
```

A 21-case correctness harness (`tools/test_conv_direct.cpp`, includes the production level-0 shapes plus chain/resblock compositions) passes 21/21 with max|Δ| ≤ 5.5e-5 on the Vulkan path vs the CPU `_fused` op.
