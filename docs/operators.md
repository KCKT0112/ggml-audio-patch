# Operator Deep-Dive & Design Notes

> English | **[中文](operators_zh.md)**

Semantics, API, per-backend implementation, and provenance of each operator.

## 1. `GGML_OP_IM2COL_FAST_1D` — O(1)-window 1-D im2col

**Origin**: `ggml_im2col_fast_1d` from [0xShug0/audio.cpp](https://github.com/0xShug0/audio.cpp).

**Core idea**: standard im2col walks the full kernel width (O(KW)) for every output position, where a large fraction lands outside the input boundary and only produces zero padding. In 1-D the valid window is a closed-form division:

```
base = iow*s0 - p0
ikw0 = ceil(max(0, -base)     / d0)   # first valid kernel tap
ikw1 = ceil(max(0, IW - base) / d0)   # window end (ceiled & clamped)
```

Head `[0, ikw0)` and tail `[ikw1, KW)` are `memset` to zero; the interior segment is **contiguous memory and copied with `memcpy`** when `d0 == 1` (degenerating to a strided loop when converting to an F16 destination). Per output position, only live input is touched.

**API**:

```c
// Drop-in equivalent of ggml_conv_1d; only the im2col node carries
// GGML_OP_IM2COL_FAST_1D instead.
struct ggml_tensor * ggml_conv_1d_fast_1d_im2col(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,   // [K, IC, OC]
        struct ggml_tensor  * b,   // [L, IC]
        int s0, int p0, int d0);
```

**Backends**: dedicated CPU kernel (`ggml_compute_forward_im2col_fast_1d`, all F16/F32 dst × src combinations); every other backend gets a one-line **alias** onto the existing `IM2COL` path — geometry and `op_params` are identical, so the alias reuses the entire GPU stack.

**Measured**: 1.03–1.15× on CPU (see [benchmarks.md](benchmarks.md)); aliased GPU paths are parity by construction.

**Gotcha**: ggml convolution weights are laid out as **[K, IC, OC]** — input channels on `ne[1]`, transposed relative to PyTorch's `[OC, IC, K]`. Swapping the two is the single most common porting mistake.

## 2. `ggml_conv_transpose_1d_ext` — full-parameter transposed convolution

**Origin**: the `conv_transpose_1d` modifications in [mmwillet/TTS.cpp](https://github.com/mmwillet/TTS.cpp)'s ggml fork (`support-for-tts` branch), made for Kokoro-style TTS vocoders.

**Stock ggml v0.19.0 ships only the 6-arg variant** (no groups / output_padding / padding), and the upstream port has three bugs — all fixed here:

1. **The CPU grouped path was broken** — acknowledged in the origin's own comment ("the CPU implementation is wrong for groups"). Rewritten: weight block selected by `i1 % cout_pg`, input channel base `(i1 / cout_pg) * ne02`, matching CUDA/PyTorch semantics.
2. **CUDA reads the wrong `op_params` slots** — the original has `const int p0 = 0; /* opts[3] */` and `const int d0 = 1; /* opts[4] */`; in the 4-slot layout `{s0,p0,d0,g0}` the correct indices are 1 and 2 (3/4 are layout-era leftovers).
3. **`GGML_ASSERT(s0 % p0 == 0)` divides by zero when `p0==0`** — now `p0 == 0 || s0 % p0 == 0`.

**New 8-arg API**; the legacy 6-arg entry point stays and delegates (`op0=0, g0=1`), keeping binary compatibility:

```c
struct ggml_tensor * ggml_conv_transpose_1d_ext(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,    // [K, Cout/g0, Cin/g0]  (PyTorch groups layout)
        struct ggml_tensor  * b,    // [L, Cin]
        int s0, int p0, int d0,     // stride / padding / dilation
        int op0,                    // output_padding
        int g0);                    // groups
```

**Layout decision**: groups use the PyTorch layout `a = [K, Cout/g0, Cin/g0]` so ecosystem weights need zero conversion. The cost: `src0->ne[1]` no longer equals the total output-channel count — anywhere treating "src shapes" as "dst shapes" must be re-audited (see the Vulkan `elements` trap in [porting-notes.md](porting-notes.md)).

**Constraints**: `d0 == 1` and (`p0 == 0 || s0 % p0 == 0`) — the common capability subset of all backend kernels.

**Backends**:

- **CPU**: grouped indexing and p0 out-of-range clamping added to both the f32 and f16_f32 paths (`o = i10*s0 + i00 - p0; if (o<0 || o>=ne0) continue;`).
- **CUDA**: the indexing bug above fixed; `cout_pg`/`cin_g` group indices added.
- **Vulkan**: `conv_transpose_1d.comp` changes only index math (`Cout_pg_idx = Cout_idx % p.Cout`, `in_c_base = (Cout_idx / p.Cout) * p.Cin`); push-constant layout untouched; dispatch `elements` fixed to `dst->ne[1]`.
- **Metal**: the kernel does not support p0/groups; `supports_op` rejects `g0>1 || p0≠0` so graphs fall back to CPU cleanly.

**Measured**: primarily a capability fix; on Vulkan the new path is **2.15× faster than legacy even at g=1** because it never materializes the im2col scratch tensor.

## 3. `GGML_OP_REL_POS_BIAS` — relative positional bias

**Origin**: [ggmlR](https://CRAN.R-project.org/package=ggmlR) (BoTNet-style relative position bias).

**Semantics**: input `x = [C, HW, B]` (C features over HW tokens per map), weight table `wcat = [rel_h + rel_w, C]` (`rel_h = 2H-1` row-difference rows, then `rel_w = 2W-1` column-difference rows). Output `[HW, HW, B]`:

```
out[k, q, b] = Σ_c x[c, q_h·W+q_w, b] · W[r_h(q_h−k_h+H−1), c]
             + Σ_c x[c, q_w·H+q_h, b] · W[rel_h + r_w(q_w−k_w+W−1), c]
```

I.e. the row/column key-query displacement each indexes one weight row, dot-producted with the features per channel, both axes summed — the classic relative-position bias generator of Swin / BoTNet-style attention.

**Backends**:

- **CPU**: direct quadruple loop + per-channel dot product (deliberately single-threaded, `n_tasks=1`, as a reference implementation).
- **Vulkan**: `rel_pos_bias.comp`, 3-D dispatch (x = width, y = height, z = b·HW+query), `local_size 8×8×4`, push constants `{H, W, B, C, rel_h, rel_w}`; dispatched through the generic `ggml_vk_op_f32` path with 3 bindings {x, wcat, dst}.

**Measured**: up to 220 GFLOP/s on Vulkan; CPU 1.2–2.5 GFLOP/s (upstream ggml had no CPU implementation of this operator at all).

## 4. `GGML_OP_SCATTER_ELEMENTS` — indexed scatter write / accumulate

**Origin**: [ggmlR](https://CRAN.R-project.org/package=ggmlR) (ONNX `ScatterElements` semantics).

**Semantics**: `dst = scatter_elements(data, updates, indices, reduction, axis)`. `updates` and `indices` share the same shape; every update element lands in `dst` at the same multi-dimensional index except that the `axis` coordinate is replaced by the corresponding `indices` value. `reduction=0` overwrites, `reduction=1` accumulates (which also fixes the repeated-index semantics). Essentially the inverse of `get_rows`.

**Builder constraints**: F32 data/updates, I32 indices, indices **shape-identical** to updates (the kernels iterate both as one flat linear sequence), `data` and `updates` may differ only along `axis`, all tensors contiguous. **Assertions must match how the kernel actually indexes, not just the operator spec** — ONNX would allow broadcast (size-1) index dims; this implementation explicitly rejects them because the kernels do not support it.

**Backends**:

- **CPU**: memcpy `data→dst`, then iterate `updates` flat, decompose the 4-D index, substitute the axis coordinate, and write back (`=` or `+=`).
- **Vulkan**: two steps — ① `buffer_copy` data into dst followed by a **pipelineBarrier** (`TransferWrite → ShaderRead|ShaderWrite`); ② run `scatter_elements.comp` in the `vk_op_binary_push_constants` layout (updates=src0, indices=src1). The accumulate variant uses `atomicAdd` from `GL_EXT_shader_atomic_float`, gated by a **device-creation-time probe** of `VK_EXT_shader_atomic_float` + `shaderBufferFloat32AtomicAdd`; `supports_op` returns false on hardware without it, falling back to CPU.

**Measured**: 60–96 GB/s on Vulkan (the accumulate path is fastest at 95.9 GB/s thanks to atomics); CPU 0.5–0.8 GB/s fallback.

## Backend support matrix (mirrors the front page)

| Operator | CPU | Vulkan | CUDA | Metal |
|---|---|---|---|---|
| `IM2COL_FAST_1D` | ✅ dedicated kernel | ✅ aliased to `IM2COL` | ✅ aliased | ✅ aliased |
| `conv_transpose_1d_ext` | ✅ all params | ✅ `p0=0, d0=1` (groups ✓) | ✅ all params | ⚠️ `g0=1, p0=0` only |
| `REL_POS_BIAS` | ✅ | ✅ | — (CPU fallback) | — |
| `SCATTER_ELEMENTS` | ✅ | ✅ (add needs the atomic ext) | — | — |

## Upstream follow-up suggestions

All four operators are suitable as standalone discussion patches against ggml main: `REL_POS_BIAS` and `SCATTER_ELEMENTS` remain absent upstream; `IM2COL_FAST_1D` should be submitted with the attached measurements; `conv_transpose_1d_ext` fits better as a **parameter-extension proposal** for `GGML_OP_CONV_TRANSPOSE_1D` than as a new op.
