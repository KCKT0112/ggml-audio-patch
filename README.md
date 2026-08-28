# ggml-audio-patch

> **[中文文档](README_CN.md)** | English

A curated patch set that ports **four audio-domain operators** into [ggml](https://github.com/ggml-org/ggml) **v0.19.0**, each adopted from a different project in the ggml ecosystem, unified to upstream-conformant APIs, fixed where the originals were broken, and extended across CPU / Vulkan / CUDA backends. Shipped as one unified diff plus correctness tests and a cross-backend benchmark suite.

## The four operators

| Operator | Adopted from | What it adds |
|---|---|---|
| `GGML_OP_IM2COL_FAST_1D` | [0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp) | 1-D im2col with an O(1) valid-window computation per output row instead of scanning the whole kernel width; the interior window becomes a plain `memcpy` when `d0 == 1`. |
| `ggml_conv_transpose_1d_ext` | [mmwillet/TTS.cpp](https://github.com/mmwillet/TTS.cpp) (ggml `support-for-tts` branch) | Full-PyTorch-parity `ConvTranspose1d`: **groups / output_padding / padding**. Also fixes the origin's broken CPU grouped path, a CUDA `op_params` mis-indexing bug, and a division-by-zero assertion. |
| `GGML_OP_REL_POS_BIAS` | [ggmlR](https://CRAN.R-project.org/package=ggmlR) (R bindings for ggml) | BoTNet-style two-axis relative-position attention bias: per-axis displacement lookup + per-channel dot product, with CPU and Vulkan implementations. |
| `GGML_OP_SCATTER_ELEMENTS` | [ggmlR](https://CRAN.R-project.org/package=ggmlR) | ONNX `ScatterElements` semantics — the inverse of `get_rows`. Vulkan implements the additive reduction with `VK_EXT_shader_atomic_float` atomics. |

Base tree: ggml [`30bf868`](https://github.com/ggml-org/ggml) (v0.19.0). The diff is additive at enum/builder/kernel insertion points, so applying onto nearby commits usually needs only light conflict resolution.

## Repository layout

```
ggml-audio-patch/
├── patches/
│   └── learned-ops-ggml0190.patch   # unified diff against ggml v0.19.0
├── tests/
│   ├── test_learned_ops.c           # correctness smoke tests (hand-computed references)
│   └── bench_learned_ops.c          # CPU / Vulkan / CUDA micro-benchmarks
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
git apply ../ggml-audio-patch/patches/learned-ops-ggml0190.patch
```

Then configure / build / test — see **[docs/building.md](docs/building.md)** for prerequisites (Vulkan SDK, CUDA toolkit, Windows generator choice) and per-backend commands, or run the bundled `scripts/build-and-test.sh` / `build-and-test.ps1`.

Benchmarks: see **[docs/benchmarks.md](docs/benchmarks.md)** (headline: 1.03–1.15× CPU speedup for `IM2COL_FAST_1D` on large frames; 2.15× on Vulkan for the grouped-convT-capable kernel vs. the legacy im2col path).

## Backend support matrix

| Operator | CPU | Vulkan | CUDA | Metal |
|---|---|---|---|---|
| `IM2COL_FAST_1D` | ✅ dedicated kernel | ✅ aliased to `IM2COL` | ✅ aliased | ✅ aliased |
| `conv_transpose_1d_ext` | ✅ all params | ✅ `p0=0, d0=1` (groups ✓) | ✅ all params | ⚠️ `g0=1, p0=0` only |
| `REL_POS_BIAS` | ✅ | ✅ | — falls back to CPU | — |
| `SCATTER_ELEMENTS` | ✅ | ✅ (`add` needs `shaderBufferFloat32AtomicAdd`) | — | — |

Unsupported parameter combinations are rejected by each backend's `supports_op`, so graphs fall back to the CPU backend cleanly instead of producing wrong results.

## License & attribution

[MIT](LICENSE), matching upstream ggml. The original operator implementations belong to their respective upstream projects (audio.cpp, TTS.cpp, ggmlR, ggml); this repository only re-bases, aligns APIs, fixes bugs, and adds backends/tests. Detailed credit notes per operator live in [docs/operators.md](docs/operators.md).
