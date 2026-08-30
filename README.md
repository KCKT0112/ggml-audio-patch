# ggml-audio-patch

> **[中文文档](README_CN.md)** | English

A curated patch set that ports **sixteen audio-domain operators** into [ggml](https://github.com/ggml-org/ggml) **v0.19.0** — fourteen adopted from different projects in the ggml ecosystem, unified to upstream-conformant APIs, fixed where the originals were broken, and extended across CPU / Vulkan / CUDA / Metal backends; plus two authored here for the NSF-HiFiGAN vocoder (a fused bias-add + leaky-ReLU, and a stride-1 direct 1-D convolution with producer-side fusions). Shipped as five unified diffs (applied in sequence; the Metal one is platform-optional, the last one is a Vulkan pipeline-cache enhancement with no operator content) plus correctness tests and cross-backend benchmark suites.

## Patch 1 — the six learned operators

| Operator | Adopted from | What it adds |
|---|---|---|
| `GGML_OP_IM2COL_FAST_1D` | [0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp) | 1-D im2col with an O(1) valid-window computation per output row instead of scanning the whole kernel width; the interior window becomes a plain `memcpy` when `d0 == 1`. |
| `ggml_conv_transpose_1d_ext` | [mmwillet/TTS.cpp](https://github.com/mmwillet/TTS.cpp) (ggml `support-for-tts` branch) | Full-PyTorch-parity `ConvTranspose1d`: **groups / output_padding / padding**. Also fixes the origin's broken CPU grouped path, a CUDA `op_params` mis-indexing bug, and a division-by-zero assertion. |
| `GGML_OP_REL_POS_BIAS` | [ggmlR](https://CRAN.R-project.org/package=ggmlR) (R bindings for ggml) | BoTNet-style two-axis relative-position attention bias: per-axis displacement lookup + per-channel dot product, with CPU and Vulkan implementations. |
| `GGML_OP_SCATTER_ELEMENTS` | [ggmlR](https://CRAN.R-project.org/package=ggmlR) | ONNX `ScatterElements` semantics — the inverse of `get_rows`. Vulkan implements the additive reduction with `VK_EXT_shader_atomic_float` atomics. |
| `GGML_OP_ADD_LEAKY_RELU` | authored for [pc-nsf-hifigan.cpp](https://github.com/KakaruHayate/pc-nsf-hifigan.cpp) (NSF-HiFiGAN vocoder) | `y = leaky(a + b)` in one pass — broadcast `[1,C]` or rowwise `[T,C]` bias; bit-identical to `add` + `leaky_relu`. Also un-serializes the stock CPU `leaky_relu` kernel (`n_tasks` was forced to 1). |
| `GGML_OP_CONV_DIRECT_1D` (+`_fused`) | authored for pc-nsf-hifigan.cpp | Stride-1 direct 1-D conv with no im2col scratch: weight packed to `[IC·K, OCp]` once per call, AVX2 6×16 micro-kernel with pointer-increment addressing (+26% single-thread vs `imul`-indexed). `_fused` folds bias / residual / output-leaky and input-side `leaky·scale` **bit-identically** — 2.50× on the full vocoder (11.09 s → 4.43 s, 24 threads, 16C/32T), on par with the ONNX Runtime CPU EP (5.155 s) in fp32. |

## Patch 2 — the ten qvac fused operators

Ported from [tetherto/qvac-ext-ggml](https://github.com/tetherto/qvac-ext-ggml) (branch `speech`, MIT). Applies **on top of patch 1**. All ten are real fused kernels — single-pass implementations of sub-graphs that stock ggml must express as 3–5 separate ops (and stock ggml has **no RNN cell at all**; `gru` fills that gap):

| Operator | Origin engine | What it fuses |
|---|---|---|
| `GGML_OP_SUPERTONIC_DEPTHWISE_1D` | Supertonic vocoder (ConvNeXt blocks) | pad (edge-clamp or causal) + depthwise 1-D conv + bias in one pass, `[T,C]` and `[C,T]` layouts, K ∈ {3,5,7}, dilation-aware. |
| `GGML_OP_SUPERTONIC_LAYER_NORM_CHANNEL` | Supertonic vocoder | channel-axis layer norm (stock `ggml_norm` only normalizes `ne[0]`) + affine, replacing a permute/cont/norm/mul/add/permute/cont chain. |
| `GGML_OP_SUPERTONIC_PW2_RESIDUAL` | Supertonic vocoder | `(x + bias) * gamma + residual` in one pass. |
| `GGML_OP_SUPERTONIC_BIAS_GELU` | Supertonic vocoder | bias-add + erf-GELU in one pass. |
| `GGML_OP_SUPERTONIC_EDGE_PAD_1D` | Supertonic vocoder | edge-replicate padding (left, or left+right) replacing view/repeat/concat chains. |
| `GGML_OP_GRU` | LavaSR denoiser | fused batched GRU sweep (PyTorch semantics, gate order r/z/n), parallel over batch — the first RNN cell in ggml core. |
| `GGML_OP_ZERO_UPSAMPLE` | LavaSR denoiser | zero-insertion upsample by integer factor (transpose-conv counterpart), one pass. |
| `GGML_OP_CHANNEL_SHUFFLE` | LavaSR denoiser | PyTorch channel shuffle over `ne[2]`, one plane copy per output channel. |
| `GGML_OP_AFFINE_PRELU` | LavaSR denoiser | per-channel affine + PReLU in one pass. |
| `GGML_OP_SNAKE` | ACE-Step Oobleck VAE | snake activation `y = x + sin²(a·x)·inv_b` with per-channel params. |

## Patch 3 — verified Metal integration

Applies **on top of patch 2**. It wires the five Supertonic kernels and direct `GGML_OP_SNAKE` dispatch into Metal, with strict F32/shape/parameter gates. `GRU` / `ZERO_UPSAMPLE` / `CHANNEL_SHUFFLE` / `AFFINE_PRELU` remain clean CPU fallbacks because the donor has no Metal kernels for them. See [docs/metal-porting.md](docs/metal-porting.md) for the verification matrix and platform evidence.

`GRU` / `ZERO_UPSAMPLE` / `CHANNEL_SHUFFLE` / `AFFINE_PRELU` / `SNAKE` also have Vulkan compute-shader implementations; `GRU` additionally has register-resident small-H variants (H = 2/4/8) and caps at H ≤ 128 (shared-memory).

## Patch 4 — Vulkan compute backend for `CONV_DIRECT_1D`

Applies **on top of patches 1 and 2** (patch 2 occupies the same shader-registry and dispatch insertion points, so the order 1 → 2 → 4 is required; patch 3 is Metal-only and orthogonal — with or without it, patch 4 applies the same way). This is the GPU half of the vocoder work from patch 1: it does not change any operator semantics — `tests/test_learned_ops.c` remains the frozen contract.

- `vulkan-shaders/conv_direct_1d.comp` — implicit-GEMM stride-1 direct 1-D conv, fp32 only. One SPIR-V binary; the warp tile is passed as Vulkan specialization constants (the `mul_mm` scheme), producing five concrete pipelines. `ggml_vk_op_get_pipeline` picks the tile per node by output-channel count (OC ≤ 16 → e16, ≤ 32 → e32, else w128); **every variant is numerically correct for any shape**, the pick is a pure performance heuristic and makes no vendor-specific assumptions.
- Bias, residual, output-leaky and input-side `leaky·scale` fusions are honored exactly like the CPU `_fused` op (same op_params contract).
- `supports_op` gate: fp32 + contiguous, `K ≥ 3` and `(K−1)·dilation ≤ 72` (the shared-memory halo cap); anything else returns false so schedulers fall back to the CPU kernel. **Consumers that drive ggml-vulkan in raw graph mode** (calling `ggml_vk_build_graph` directly, bypassing the scheduler's `supports_op` check) must enforce the same gate themselves — the K = 2 subpixel-upsample convs in NSF-HiFiGAN exceed the shader's staged-row window and would silently produce wrong output. pc-nsf-hifigan.cpp does this with `PCNSF_DIRECT_MIN_K` (Vulkan default 3, other backends 1).

Measured on the full NSF-HiFiGAN vocoder (RTX 2070, driver 32.0.12.x-class, fp32 throughout, `GGML_VK_DISABLE_COOPMAT2=1` so tensor cores stay off; median of repeated runs of the pc-nsf-hifigan.cpp `hifigan_cli` on the reference clip): **≈ 433–479 ms end-to-end** vs. 326 ms for ONNX Runtime's DML EP and 5 155 ms for its CPU EP on the same box. Accuracy against the torch-CPU reference: corr 0.99999985, max|Δ| 6.1e-4 — an order of magnitude tighter than ORT-DML on the same clip (corr 0.99999697, max|Δ| 2.1e-3). Full numbers and repro commands in [docs/benchmarks.md](docs/benchmarks.md#patch-4-vulkan-conv_direct_1d).

## Patch 5 — Vulkan persistent disk pipeline cache

Ports [KakaruHayate/game.cpp](https://github.com/KakaruHayate/game.cpp) (`cmake/patches/ggml-vulkan-pipeline-cache`) onto ggml **v0.19.0**: a per-process `VkPipelineCache` that is loaded from disk at device init, fed to every `vkCreateComputePipelines`, and flushed back when pipelines were created or destroyed. No operator/shader content — numerics are untouched by construction.

- Verified to apply standalone on stock v0.19.0 **and** on top of patches 1–4 (patches 4 and 5 both edit `src/ggml-vulkan/ggml-vulkan.cpp`; recommended order is 1 → 2 → (3) → 4 → 5).
- env knobs: `GGML_VK_PIPELINE_CACHE_PATH` (override file location; default `ggml_audio_vk_pipeline.cache` under `%LOCALAPPDATA%` on Windows, `$HOME/.cache` on Linux), `GGML_VK_DISABLE_PIPELINE_CACHE=1` (opt out), `GGML_VK_PIPELINE_CACHE_DEBUG=1` (log load/save byte counts).
- Measured (2026-08-30, Windows 11, Xeon E5-2675 v3, RTX 2070 driver 32.0.16.2002, MSVC 14.29, pc-nsf-hifigan.cpp `hifigan_cli`, reference clip T=1722, blob = 334 150 B): with the cache blob absent the first run of the patched binary pays pipeline compilation **inside** the first compute — 1 061.5 ms in-graph / 2 454 ms process wall; subsequent process restarts with the warm blob run 346.8–355.0 ms in-graph / ≈1 120–1 290 ms wall. First-run cost reduced ≈3× in-graph, ≈2× wall.
- Honesty note: on this test box the NVIDIA driver's own shader cache (…/NVIDIA/GLCache) also persists compiled pipelines across process restarts, and with that cache hot the blob adds nothing measurable in steady state (interleaved ON/OFF medians 374.0 vs 373.1 ms, n=3, machine scatter ±10%). The blob's value shows when the driver cache is cold or evicted — first run after a driver update/reinstall or cache cleanup, capped-cache configurations, and drivers/platforms with weaker caching. Repro commands and the full run table in [docs/benchmarks.md](docs/benchmarks.md#patch-5-vulkan-pipeline-cache).

Base tree: ggml [`30bf868`](https://github.com/ggml-org/ggml) (v0.19.0). The diffs are additive at enum/builder/kernel insertion points, so applying onto nearby commits usually needs only light conflict resolution.

## Repository layout

```
ggml-audio-patch/
├── patches/
│   ├── learned-ops-ggml0190.patch   # unified diff against ggml v0.19.0 (patch 1)
│   ├── qvac-ops-ggml0190.patch      # unified diff on top of patch 1 (patch 2)
│   ├── metal-ops-ggml0190.patch     # Metal integration on top of patch 2 (patch 3)
│   ├── vulkan-conv-direct-1d-ggml0190.patch  # Vulkan backend for CONV_DIRECT_1D (patch 4, on top of 1+2)
│   └── vulkan-pipeline-cache-ggml0190.patch  # Vulkan persistent disk pipeline cache (patch 5, standalone or on top of 1–4)
├── tests/
│   ├── test_learned_ops.c           # patch-1 correctness smoke tests (hand-computed references)
│   ├── test_qvac_ops.c              # patch-2 correctness smoke tests (cpu | vk | metal harness hook)
│   ├── bench_learned_ops.c          # patch-1 CPU / Vulkan / CUDA micro-benchmarks
│   └── bench_qvac_ops.c             # patch-2 fused-vs-composed benchmarks (CPU + Vulkan + Metal)
├── scripts/
│   ├── build-and-test.sh            # Linux/macOS one-shot build + test
│   └── build-and-test.ps1           # Windows (pwsh) one-shot build + test
├── AGENTS.md                        # contribution & editing boundaries (human or AI agents)
└── docs/
    ├── building.md / building_zh.md            # build prerequisites & instructions
    ├── benchmarks.md / benchmarks_zh.md        # measured performance & methodology
    ├── operators.md / operators_zh.md          # per-operator design notes & API
    ├── metal-porting.md / metal-porting_zh.md  # Metal integration notes: status, harness, gotchas, verified delivery
    ├── porting-notes.md / porting-notes_zh.md  # known pitfalls when porting further
    └── task-package.md / task-package_zh.md    # how platform-bound work is packaged & delegated
```

## Quick start

```bash
git clone https://github.com/ggml-org/ggml.git ggml-src
cd ggml-src && git checkout 30bf868        # v0.19.0
git apply ../ggml-audio-patch/patches/learned-ops-ggml0190.patch            # patch 1
git apply ../ggml-audio-patch/patches/qvac-ops-ggml0190.patch               # patch 2 (sequential, on top)
git apply ../ggml-audio-patch/patches/metal-ops-ggml0190.patch              # patch 3 (Metal, optional)
git apply ../ggml-audio-patch/patches/vulkan-conv-direct-1d-ggml0190.patch  # patch 4 (on top of 1+2)
git apply ../ggml-audio-patch/patches/vulkan-pipeline-cache-ggml0190.patch  # patch 5 (last; also applies standalone on stock v0.19.0)
```

Patch 2 must follow patch 1: they touch the same enum-assert and dispatch hunks. Patch 3 is optional on non-Metal platforms. Patch 4 must follow patches 1 and 2: it consumes `CONV_DIRECT_1D` from patch 1 and shares shader-registry/dispatch insertion points with patch 2 (it is orthogonal to patch 3 — apply it with or without the Metal patch). Patch 5 was verified both standalone on stock v0.19.0 and on top of patches 1–4 — when combined, apply it **last** (it and patch 4 both edit `src/ggml-vulkan/ggml-vulkan.cpp`). Applying only patch 1 is fine (skip the rest); applying only 1+4 is **not** supported.

Then configure / build / test — see **[docs/building.md](docs/building.md)** for prerequisites (Vulkan SDK, CUDA toolkit, Windows generator choice) and per-backend commands, or run the bundled `scripts/build-and-test.sh` / `build-and-test.ps1`.

Benchmarks: see **[docs/benchmarks.md](docs/benchmarks.md)** (patch-1 headline: 1.03–1.15× CPU speedup for `IM2COL_FAST_1D` on large frames; 2.15× on Vulkan for the grouped-convT-capable kernel vs. the legacy im2col path; 2.50× for `CONV_DIRECT_1D`+fusions on the full NSF-HiFiGAN vocoder (11 089 → 4 431 ms, 24 threads on a 16C/32T Xeon E5-2675 v3 — ahead of the ONNX Runtime CPU EP's 4 724 ms fp32 measured the same day, 2026-08-30 AM). Patch-2/3 headline: depthwise-1d fused 19–30× on CPU and up to 5.80× on Apple M4 Metal; Snake reaches 3.05× on Metal and 3.9× on Vulkan; Metal channel layer norm reaches 3.24×; affine_prelu up to 6.3× Vulkan; gru fills the RNN gap at ~47 ms for H512×B4×L32 on CPU. Patch-4 headline: the direct conv's Vulkan backend runs the vocoder end-to-end in ≈ 430–480 ms fp32 (vs ORT DML EP 282 ms measured the same day), with 10× tighter fp32 agreement than DML (max|Δ| 6.1e-4 vs 2.1e-3). Patch-5 headline: the Vulkan pipeline cache cuts first-run pipeline-compile cost ≈3× in-graph (1 061.5 → ≈350 ms) and ≈2× process wall (2 454 → ≈1 200 ms) on driver-cache-cold startups, at zero numerics cost. **Terminal note (2026-08-30):** closing the remaining Vulkan-vs-DML gap was investigated and deliberately stopped — the only route (f16 tensor-core GEMM) measurably degrades precision below the DML anchor (corr 0.99999969 vs DML 0.99999697, max|Δ| 3.9e-3 vs DML 2.1e-3, torch-side simulation), and precision-safe alternatives top out above DML; details in `docs/benchmarks.md` §6.)

## Backend support matrix

Patch 1:

| Operator | CPU | Vulkan | CUDA | Metal |
|---|---|---|---|---|
| `IM2COL_FAST_1D` | ✅ dedicated kernel | ✅ aliased to `IM2COL` | ✅ aliased | ✅ aliased |
| `conv_transpose_1d_ext` | ✅ all params | ✅ `p0=0, d0=1` (groups ✓) | ✅ all params | ⚠️ `g0=1, p0=0` only |
| `REL_POS_BIAS` | ✅ | ✅ | — falls back to CPU | — |
| `SCATTER_ELEMENTS` | ✅ | ✅ (`add` needs `shaderBufferFloat32AtomicAdd`) | — | — |
| `ADD_LEAKY_RELU` | ✅ | — falls back to CPU | — | — |
| `CONV_DIRECT_1D` (+`_fused`) | ✅ | ✅ patch 4 (fp32, `K ≥ 3`, `(K−1)·dil ≤ 72`; else CPU fallback) | — | — |

Patch 2:

| Operator | CPU | Vulkan | CUDA | Metal |
|---|---|---|---|---|
| `SUPERTONIC_DEPTHWISE_1D` (+`_ct`, `_causal_ct`) | ✅ | — falls back to CPU | — | ✅ F32, K ∈ {3,5,7} |
| `SUPERTONIC_LAYER_NORM_CHANNEL` (+`_ct`) | ✅ | — | — | ✅ F32 |
| `SUPERTONIC_PW2_RESIDUAL` (+`_ct`) | ✅ | — | — | ✅ F32 |
| `SUPERTONIC_BIAS_GELU` (+`_ct`) | ✅ | — | — | ✅ F32 |
| `SUPERTONIC_EDGE_PAD_1D` (+`_ct`) | ✅ | — | — | ✅ F32 |
| `GRU` | ✅ | ✅ H ≤ 128 (+H=2/4/8 variants) | — | — |
| `ZERO_UPSAMPLE` | ✅ | ✅ | — | — |
| `CHANNEL_SHUFFLE` | ✅ | ✅ | — | — |
| `AFFINE_PRELU` | ✅ | ✅ | — | — |
| `SNAKE` | ✅ | ✅ | — | ✅ F32 |

(Patch 3 wires the donor's five Supertonic Metal kernels and direct Snake dispatch. The other four qvac ops have no donor Metal kernel and remain explicitly gated off. Integration notes and verification evidence are in [docs/metal-porting.md](docs/metal-porting.md). CUDA remains gated off, matching the donor. Contribution/editing boundaries: [AGENTS.md](AGENTS.md). How this port was packaged and delegated without local Apple hardware: [docs/task-package.md](docs/task-package.md).)

Unsupported parameter combinations are rejected by each backend's `supports_op`, so graphs fall back to the CPU backend cleanly instead of producing wrong results.

## License & attribution

[MIT](LICENSE), matching upstream ggml. The original operator implementations belong to their respective upstream projects (audio.cpp, TTS.cpp, ggmlR, ggml, and [tetherto/qvac-ext-ggml](https://github.com/tetherto/qvac-ext-ggml) for patch 2); this repository only re-bases, aligns APIs, fixes bugs, and adds backends/tests. Detailed credit notes per operator live in [docs/operators.md](docs/operators.md).
