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

## 5. `GGML_OP_ADD_LEAKY_RELU` — fused bias-add + leaky ReLU

**Origin**: authored for the [pc-nsf-hifigan.cpp](https://github.com/KakaruHayate/pc-nsf-hifigan.cpp) NSF-HiFiGAN vocoder port, where every `Conv1d` is followed by `bias-add → leaky ReLU` over activations up to `[881664, C]` — two dependent elementwise passes that each re-stream the full tensor.

**Semantics**: `y = leaky(a + b)` with `leaky(v) = (v > 0 ? v : 0) + slope · (v < 0 ? v : 0)` — the exact `ggml_vec_leaky_relu_f32` expression, so the op is **bit-identical** to composing `ggml_add` + `ggml_leaky_relu` (parity-tested against both). `a = [T, C]`; `b` is a broadcast bias `[1, C]` or a rowwise `[T, C]` tensor.

**API**:

```c
struct ggml_tensor * ggml_add_leaky_relu(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,      // [T, C]
        struct ggml_tensor  * b,      // [1, C] or [T, C]
        float                 slope);
```

**Backends**: CPU only. Stripes over channels when `C ≥ nth` (the bias value stays in a register, the inner loop walks contiguous `t` for both inputs and the output; auto-vectorizable as two selects + mul + add), t-split otherwise. The same change un-serializes stock `GGML_OP_LEAKY_RELU` (upstream forces `n_tasks = 1`; the patch splits rows with identical per-element arithmetic). Other backends reject the op in `supports_op` → clean CPU fallback.

**Verified**: 8 test cases (broadcast `[1,C]` and rowwise `[T,C]` bias) pass exact-reference and add+leaky parity at 1e-5/1e-4. The vocoder uses it on its im2col fallback path; the direct-conv path below folds the same pattern into the convolution epilogue.

## 6. `GGML_OP_CONV_DIRECT_1D` — stride-1 direct 1-D convolution + producer-side fusions

**Origin**: authored for the pc-nsf-hifigan.cpp NSF-HiFiGAN vocoder CPU path. Stock ggml 1-D convolution is `im2col` — materializing an F16 `[OL, IC·K]` scratch per node, i.e. input × kernel-width bytes of extra traffic on every conv — followed by a skinny-`OC`, very-tall `mul_mat`. The direct kernel packs the weight once per call and computes output tiles straight from a zero-padded input copy.

**Semantics**: stride-1 1-D convolution with optional bias, output leaky ReLU, residual add, and input-side `leaky(x)·in_scale`:

```
y[t, oc] = act( bias[oc] + Σ_{ic,kw} w[kw, ic, oc] · x'[ic, t + kw·dil] + res[t, oc] )
x'[ic, ·] = zero-padded input row, optionally pre-folded with in_slope / in_scale
```

Every fusion is **bit-identical** to the unfused graph, not an approximation: the input fold applies the same leaky expression as `ggml_vec_leaky_relu_f32` followed by a plain multiply (no fp16 round-trip) while copying rows into the padded scratch; the epilogue associates as `(bias + acc) + res`, matching `ggml_add(conv_out, res)`.

**API**:

```c
struct ggml_tensor * ggml_conv_direct_1d(        // bias + output leaky
        struct ggml_context * ctx,
        struct ggml_tensor  * w,      // [K, IC, OC]
        struct ggml_tensor  * x,      // [T, IC]
        struct ggml_tensor  * bias,   // [OC] or NULL
        int pad, int dil,
        float leaky_slope);           // 0 = no activation

struct ggml_tensor * ggml_conv_direct_1d_fused(  // + residual, input-side fold
        struct ggml_context * ctx,
        struct ggml_tensor  * w,      // [K, IC, OC]
        struct ggml_tensor  * x,      // [T, IC]
        struct ggml_tensor  * bias,   // [OC] or NULL
        struct ggml_tensor  * res,    // [OL, OC] or NULL
        int pad, int dil,
        float leaky_slope,            // output activation, 0 = none
        float in_scale,               // input fold: v = leaky(x) · in_scale
        float in_slope);              // input fold slope, 0 = none
```

`op_params` = 7 × i32: `{pad, dil, leaky_slope (f32), has_bias, in_scale (f32), in_slope (f32), has_res}`; `src[0..3] = {w, x, bias, res}`; extra work data = `(IC·K·OCp + OCp + IC·(T+2·pad))` floats. Stride is fixed at 1 — the vocoder only needs `s = 1`; general strides stay on the im2col path.

**CPU kernel** (two phases, both parallel):

- **Phase 1** packs `Wᵀ` into a `[IC·K, OCp]` blocked layout (`OCp` = OC rounded up to 16), zero-pads the bias to `[OCp]`, and copies `x` into an `[IC, T+2·pad]` scratch with both boundary pads `memset` to zero — phase 2 carries no boundary clamping at all. The producer-side input fold happens here, once per element.
- **Phase 2** hands each thread a balanced range of 6-wide t-blocks. Superblocks (oc-super × t-super, sized for an X slice ≤ 64 KB and a Wt slice ≤ 128 KB) keep both operands L2/L3-resident so X and Wt stream from RAM roughly once per call. The AVX2 micro-kernel is a 6×16 tile (12 FMAs + 6 broadcasts per `(ic, kw)` step) with **pointer-increment addressing**: the naive `imul`-on-loop-counter address chain (`inc → movsxd → imul`, ≈5 serial cycles per tap) was measured to gate broadcast dispatch below 1 FMA/cycle; incrementing `xp += dil, wp += OCp` instead raised the single-thread rate from 24.6 to 31.8 GF/s (+26%) on a Xeon E5-2675 v3. A scalar path handles the tail t-block and non-AVX2 builds.

**Backends**: CPU only by design — on Vulkan the GEMM is strong and the im2col path stays; other backends reject the op in `supports_op` → CPU fallback.

**Measured** (Xeon E5-2675 v3, 16C/32T Haswell-EP, ~2.0 GHz sustained AVX2, MSVC 2019 `/O2`, fp32, 24 threads, median of 3; harness pc-nsf-hifigan.cpp @ `f8c16ba`): full NSF-HiFiGAN inference 80 002 ms (stock im2col + `mul_mat`) → **28 030 ms** (direct + producer-side fusions), **2.85×**, at corr 0.99999999 / max|Δ| 1.49e-4 vs the torch-CPU fp32 output — the same accuracy as the im2col path (FMA-ordering noise only). The fusions removed 100 of 252 graph nodes (50 leaky, 5 scale, 45 residual add). Honest gap: ONNX Runtime CPU EP runs the same model in 5 155 ms — the ggml CPU conv path is ~5.4× behind it today (ORT-level threading/blocking is the active workstream; a previous revision of this note quoting 4 764 ms at ORT parity was recorded against a stale build and has been corrected). On the single-thread inner loop the pointer-increment lesson still stands: the residual gap to the thread-scaled rate is broadcast load-to-use latency on the FMA critical path (a register-resident control variant reaches the 2-FMA/cycle calibration line), not memory bandwidth.

## Backend support matrix (mirrors the front page)

| Operator | CPU | Vulkan | CUDA | Metal |
|---|---|---|---|---|
| `IM2COL_FAST_1D` | ✅ dedicated kernel | ✅ aliased to `IM2COL` | ✅ aliased | ✅ aliased |
| `conv_transpose_1d_ext` | ✅ all params | ✅ `p0=0, d0=1` (groups ✓) | ✅ all params | ⚠️ `g0=1, p0=0` only |
| `REL_POS_BIAS` | ✅ | ✅ | — (CPU fallback) | — |
| `SCATTER_ELEMENTS` | ✅ | ✅ (add needs the atomic ext) | — | — |
| `ADD_LEAKY_RELU` | ✅ | — (CPU fallback) | — | — |
| `CONV_DIRECT_1D` (+`_fused`) | ✅ | ✅ patch 4 (fp32, `K ≥ 3`, `(K−1)·dil ≤ 72`; else CPU fallback) | — | — |

## Upstream follow-up suggestions

All six operators are suitable as standalone discussion patches against ggml main: `REL_POS_BIAS` and `SCATTER_ELEMENTS` remain absent upstream; `IM2COL_FAST_1D` should be submitted with the attached measurements; `conv_transpose_1d_ext` fits better as a **parameter-extension proposal** for `GGML_OP_CONV_TRANSPOSE_1D` than as a new op. `ADD_LEAKY_RELU` and `CONV_DIRECT_1D` are vocoder-motivated; the direct conv's pointer-increment lesson (address-chain latency gating FMA broadcast dispatch, +26% single-thread) generalizes to any broadcast-FMA inner-product loop, and the serialized stock `leaky_relu` CPU kernel is worth fixing upstream regardless.

---

# Patch 2 — the ten qvac fused operators

**Origin**: [tetherto/qvac-ext-ggml](https://github.com/tetherto/qvac-ext-ggml), branch `speech` (MIT). These ten ops were added upstream to make three audio engines run as single-dispatch graphs: the **Supertonic** vocoder (ConvNeXt-style blocks), the **LavaSR** denoiser, and the **ACE-Step Oobleck VAE**. None of them are thin wrappers — each replaces a multi-op sub-graph whose per-node scheduling overhead dominates (see [benchmarks.md](benchmarks.md)).

**Shared design**: all ops are F32-only (CPU bring-up parity with upstream), contiguous-input, and carry layout/parameter flags in `op_params`. The five supertonic ops come in layout-0 (`[T,C]`, T inner) and layout-1 (`_ct`, `[C,T]`, C inner) variants sharing one kernel via stride flips; `depthwise_1d` additionally has a `_causal_ct` variant (causal-left padding, K ∈ {3,5,7}).

## 7. `GGML_OP_SUPERTONIC_DEPTHWISE_1D`

```
y[t,c] = bias[c] + Σ_k w[k,c] · x[clamp(t + (k + k_off)·dil, 0, L-1), c]
k_off = causal ? -(K-1) : -K/2
```

Replaces `pad → im2col → mul_mat → bias-add` (4 nodes + an F16 scratch tensor) with one pass over `[L, C]`. Weight layout matches ggml conv convention `[K, 1, C]`. CPU kernel stripes over channels (bias/w row loaded once per channel). **Measured 19–30× faster on CPU** than the composed `conv_1d_dw` chain, mostly by eliminating the F16 im2col materialization.

## 8. `GGML_OP_SUPERTONIC_LAYER_NORM_CHANNEL`

```
y[t,c] = (x[t,c] − mean_t) / sqrt(var_t + eps) · g[c] + b[c]
```

Stock `ggml_norm` normalizes along `ne[0]` only, so channel-axis norm requires permute/cont/norm/mul/add/permute/cont. One kernel does the whole thing (double-precision mean/var accumulation). CPU up to 2.6× vs the chain on wide-row shapes; on very large `C` the CPU chain's BLAS-friendly layout wins, which is why the fused kernel also matters most on GPU command-buffer counts.

## 9. `GGML_OP_SUPERTONIC_PW2_RESIDUAL`

```
y[t,c] = residual[t,c] + (x[t,c] + bias[c]) · gamma[c]
```

Three elementwise ops → one. CPU 3.9–6.8×. Trivially parallel over channels.

## 10. `GGML_OP_SUPERTONIC_BIAS_GELU`

```
y[t,c] = 0.5·v·(1 + erf(v/√2)),  v = x[t,c] + bias[c]
```

Matches `ggml_gelu_erf` bit-for-bit in op order. CPU 1.7–3.2×.

## 11. `GGML_OP_SUPERTONIC_EDGE_PAD_1D`

```
y[t,c] = x[clamp(t − pad_left, 0, L_in − 1), c]
```

Replicate/edge-clamp padding (left-only for causal vocoders, symmetric for encoders). Replaces view/repeat/concat chains. CPU 12.8–13.5×.

## 12. `GGML_OP_GRU`

Fused batched GRU over all L time-steps, PyTorch semantics: gate order r/z/n, reset applied to the hh (recurrent) new-gate, `h0 = 0`.

```
whh [H, 3H]   recurrent weight (column g = whh[.., g])
gi  [3H, B, L]  input projection PRE-COMPUTED (W_ih·x + b_ih lives outside the op)
bhh [3H]      recurrent bias
dst [H, B, L]
per step: gh = whhᵀh + bhh; r = σ(gi_r + gh_r); z = σ(gi_z + gh_z);
          n = tanh(gi_n + r·gh_n); h = n + z·(h − n)
```

`reverse` flips the time direction (BiGRU = two calls). ggml core previously had **no RNN cell of any kind** — this fills the gap (relevant to RMVPE-style BiGRU post-filters). CPU: parallel over batch, serial over time, naive inner products. Vulkan: `gru.comp` packs `128/H` batch elements per 128-lane workgroup (H ≤ 128 shared-memory cap) plus register-resident `gru_small.comp` variants for H = 2/4/8 with zero barriers in the time loop.

**Measured**: CPU H512×B4×L32 ≈ 47 ms; Vulkan (RTX 2070) H64×B1×L256 ≈ 2.2 ms.

## 13. `GGML_OP_ZERO_UPSAMPLE`

```
out[i0·s, ...] = a[i0, ...], zeros elsewhere;  out.ne0 = (a.ne0 − 1)·s + 1
```

Zero-insertion upsampling — the exact transpose-conv counterpart used by LavaSR's decoder. Replaces `upscale + mask-mul` or zero-padded convT tricks. CPU up to 15×, Vulkan ~1.3× (launch-bound at these sizes).

## 14. `GGML_OP_CHANNEL_SHUFFLE`

PyTorch channel shuffle over `ne[2]`: `in_c = (c' % G)·(C/G) + c'/G`. One plane copy per output channel instead of reshape+permute+cont (three nodes, two of them copies). CPU ~1.5×; on Vulkan the view chain is a single fused copy so gains are launch-count only.

## 15. `GGML_OP_AFFINE_PRELU`

```
out = x·aw[f,c] + ab[f,c] + max(x,0) + slope[c]·min(x,0)
```

Per-channel affine + PReLU for `[F,T,C,Bc]` spectrogram-shaped activations (LavaSR denoiser). CPU 1.6–2.1×, Vulkan 3.4–6.3× (the composed chain needs two `repeat` broadcasts and four elementwise kernels).

## 16. `GGML_OP_SNAKE`

```
y = x + sin²(a·x) · inv_b        (a, inv_b per channel)
```

Snake activation from ACE-Step's Oobleck VAE. On Vulkan this port **reuses the upstream `snake_f32` pipeline** — the base v0.19 tree already ships a graph-level snake fusion (`mul→sin→sqr→mul→add` detection) with identical math and binding layout, so the op-level path simply dispatches through it (2-D `ne0×ne1` grid, `{ne0, ne1}` push constants). CPU 2.2–4.6×, Vulkan up to 3.9×.

## Patch-2 backend support matrix

| Operator | CPU | Vulkan | CUDA | Metal |
|---|---|---|---|---|
| `SUPERTONIC_DEPTHWISE_1D` (+`_ct`, `_causal_ct`) | ✅ | — (CPU fallback) | — | — |
| `SUPERTONIC_LAYER_NORM_CHANNEL` (+`_ct`) | ✅ | — | — | — |
| `SUPERTONIC_PW2_RESIDUAL` (+`_ct`) | ✅ | — | — | — |
| `SUPERTONIC_BIAS_GELU` (+`_ct`) | ✅ | — | — | — |
| `SUPERTONIC_EDGE_PAD_1D` (+`_ct`) | ✅ | — | — | — |
| `GRU` | ✅ | ✅ H ≤ 128 (+H=2/4/8 variants) | — | — |
| `ZERO_UPSAMPLE` | ✅ | ✅ | — | — |
| `CHANNEL_SHUFFLE` | ✅ | ✅ | — | — |
| `AFFINE_PRELU` | ✅ | ✅ | — | — |
| `SNAKE` | ✅ | ✅ | — | — |

Upstream qvac implements the supertonic five as Metal kernels (`kernel_supertonic_*_f32`) and the other five as Vulkan shaders; within patch 2 this port keeps the Vulkan five and gates Metal/CUDA off for all ten (clean CPU fallback) — the donor Metal kernels are wired in separately by patch 3.

## Patch-2 upstream follow-up suggestions

`GRU` is the strongest upstream candidate (RNN gap, clean semantics, self-contained). `ZERO_UPSAMPLE` + `CHANNEL_SHUFFLE` + `AFFINE_PRELU` + `SNAKE` are a natural "audio activation/reshape pack". The supertonic five make most sense upstream as a fused-op discussion once a second backend (Metal or CUDA) exists to justify the API surface.
