# ggml-audio-patch

> **[中文文档](README_CN.md)** | English

A curated patch set that ports **fourteen audio-domain operators** into [ggml](https://github.com/ggml-org/ggml) **v0.19.0**, adopted from different projects in the ggml ecosystem, unified to upstream-conformant APIs, fixed where the originals were broken, and extended across CPU / Vulkan / CUDA backends. Shipped as two sequential unified diffs plus correctness tests and cross-backend benchmark suites.

## Patch 1 — the four learned operators

| Operator | Adopted from | What it adds |
|---|---|---|
| `GGML_OP_IM2COL_FAST_1D` | [0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp) | 1-D im2col with an O(1) valid-window computation per output row instead of scanning the whole kernel width; the interior window becomes a plain `memcpy` when `d0 == 1`. |
| `ggml_conv_transpose_1d_ext` | [mmwillet/TTS.cpp](https://github.com/mmwillet/TTS.cpp) (ggml `support-for-tts` branch) | Full-PyTorch-parity `ConvTranspose1d`: **groups / output_padding / padding**. Also fixes the origin's broken CPU grouped path, a CUDA `op_params` mis-indexing bug, and a division-by-zero assertion. |
| `GGML_OP_REL_POS_BIAS` | [ggmlR](https://CRAN.R-project.org/package=ggmlR) (R bindings for ggml) | BoTNet-style two-axis relative-position attention bias: per-axis displacement lookup + per-channel dot product, with CPU and Vulkan implementations. |
| `GGML_OP_SCATTER_ELEMENTS` | [ggmlR](https://CRAN.R-project.org/package=ggmlR) | ONNX `ScatterElements` semantics — the inverse of `get_rows`. Vulkan implements the additive reduction with `VK_EXT_shader_atomic_float` atomics. |

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

Supertonic ops are CPU-only here (upstream qvac ships them as Metal kernels; a Metal port is future work). `GRU` / `ZERO_UPSAMPLE` / `CHANNEL_SHUFFLE` / `AFFINE_PRELU` / `SNAKE` also have Vulkan compute-shader implementations; `GRU` additionally has register-resident small-H variants (H = 2/4/8) and caps at H ≤ 128 (shared-memory).

Base tree: ggml [`30bf868`](https://github.com/ggml-org/ggml) (v0.19.0). The diffs are additive at enum/builder/kernel insertion points, so applying onto nearby commits usually needs only light conflict resolution.

## Repository layout

```
ggml-audio-patch/
├── patches/
│   ├── learned-ops-ggml0190.patch   # unified diff against ggml v0.19.0 (patch 1)
│   └── qvac-ops-ggml0190.patch      # unified diff on top of patch 1 (patch 2)
├── tests/
│   ├── test_learned_ops.c           # patch-1 correctness smoke tests (hand-computed references)
│   ├── test_qvac_ops.c              # patch-2 correctness smoke tests (CPU: all 10 ops; Vulkan: 5 shader ops)
│   ├── bench_learned_ops.c          # patch-1 CPU / Vulkan / CUDA micro-benchmarks
│   └── bench_qvac_ops.c             # patch-2 fused-vs-composed benchmarks (CPU + Vulkan)
├── scripts/
│   ├── build-and-test.sh            # Linux/macOS one-shot build + test
│   └── build-and-test.ps1           # Windows (pwsh) one-shot build + test
└── docs/
    ├── building.md / building_zh.md            # build prerequisites & instructions
    ├── benchmarks.md / benchmarks_zh.md        # measured performance & methodology
    ├── operators.md / operators_zh.md          # per-operator design notes & API
    └── porting-notes.md / porting-notes_zh.md  # known pitfalls when porting further
```

## Quick start

```bash
git clone https://github.com/ggml-org/ggml.git ggml-src
cd ggml-src && git checkout 30bf868        # v0.19.0
git apply ../ggml-audio-patch/patches/learned-ops-ggml0190.patch   # patch 1
git apply ../ggml-audio-patch/patches/qvac-ops-ggml0190.patch      # patch 2 (sequential, on top)
```

Patch 2 must follow patch 1: they touch the same enum-assert and dispatch hunks. Applying only patch 1 is fine (skip the second line).

Then configure / build / test — see **[docs/building.md](docs/building.md)** for prerequisites (Vulkan SDK, CUDA toolkit, Windows generator choice) and per-backend commands, or run the bundled `scripts/build-and-test.sh` / `build-and-test.ps1`.

Benchmarks: see **[docs/benchmarks.md](docs/benchmarks.md)** (patch-1 headline: 1.03–1.15× CPU speedup for `IM2COL_FAST_1D` on large frames; 2.15× on Vulkan for the grouped-convT-capable kernel vs. the legacy im2col path. Patch-2 headline: depthwise-1d fused 19–30× vs `conv_1d_dw`+pad+bias on CPU; snake 2.2–4.6× CPU / up to 3.9× Vulkan; affine_prelu up to 6.3× Vulkan; gru fills the RNN gap at ~47 ms for H512×B4×L32 on CPU.)

## Backend support matrix

Patch 1:

| Operator | CPU | Vulkan | CUDA | Metal |
|---|---|---|---|---|
| `IM2COL_FAST_1D` | ✅ dedicated kernel | ✅ aliased to `IM2COL` | ✅ aliased | ✅ aliased |
| `conv_transpose_1d_ext` | ✅ all params | ✅ `p0=0, d0=1` (groups ✓) | ✅ all params | ⚠️ `g0=1, p0=0` only |
| `REL_POS_BIAS` | ✅ | ✅ | — falls back to CPU | — |
| `SCATTER_ELEMENTS` | ✅ | ✅ (`add` needs `shaderBufferFloat32AtomicAdd`) | — | — |

Patch 2:

| Operator | CPU | Vulkan | CUDA | Metal |
|---|---|---|---|---|
| `SUPERTONIC_DEPTHWISE_1D` (+`_ct`, `_causal_ct`) | ✅ | — falls back to CPU | — | — |
| `SUPERTONIC_LAYER_NORM_CHANNEL` (+`_ct`) | ✅ | — | — | — |
| `SUPERTONIC_PW2_RESIDUAL` (+`_ct`) | ✅ | — | — | — |
| `SUPERTONIC_BIAS_GELU` (+`_ct`) | ✅ | — | — | — |
| `SUPERTONIC_EDGE_PAD_1D` (+`_ct`) | ✅ | — | — | — |
| `GRU` | ✅ | ✅ H ≤ 128 (+H=2/4/8 variants) | — | — |
| `ZERO_UPSAMPLE` | ✅ | ✅ | — | — |
| `CHANNEL_SHUFFLE` | ✅ | ✅ | — | — |
| `AFFINE_PRELU` | ✅ | ✅ | — | — |
| `SNAKE` | ✅ | ✅ | — | — |

(Upstream qvac implements the supertonic five as Metal kernels; the snake/shuffle/upsample/prelu/gru five as Vulkan shaders. This port keeps the Vulkan five, gates Metal off for all ten pending a kernel port, and CUDA off — matching upstream, which has no CUDA implementations either.)

Unsupported parameter combinations are rejected by each backend's `supports_op`, so graphs fall back to the CPU backend cleanly instead of producing wrong results.

## License & attribution

[MIT](LICENSE), matching upstream ggml. The original operator implementations belong to their respective upstream projects (audio.cpp, TTS.cpp, ggmlR, ggml, and [tetherto/qvac-ext-ggml](https://github.com/tetherto/qvac-ext-ggml) for patch 2); this repository only re-bases, aligns APIs, fixes bugs, and adds backends/tests. Detailed credit notes per operator live in [docs/operators.md](docs/operators.md).
