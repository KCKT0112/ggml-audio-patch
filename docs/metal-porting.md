# Metal integration notes

> English | **[中文](metal-porting_zh.md)**

What the optional patch 3 (`patches/metal-ops-ggml0190.patch`) wires into
Apple's Metal backend, how to run its test harness, the pitfalls to know
when maintaining it, and the verified delivery. Editing boundaries for
contributors (human or agent) live in [/AGENTS.md](../AGENTS.md). How
this port was packaged and delegated to a macOS contributor without
local Apple hardware is recorded in [task-package.md](task-package.md).

## Current status

| Operator | CPU | Vulkan | Metal |
|---|---|---|---|
| Supertonic × 5 | ✅ | — (CPU fallback) | ✅ patch 3, F32 |
| GRU / ZERO_UPSAMPLE / CHANNEL_SHUFFLE / AFFINE_PRELU | ✅ | ✅ | ❌ no upstream kernel exists |
| SNAKE | ✅ | ✅ | ✅ patch 3, F32 |

Patch 2 intentionally ships no Metal integration. The optional sequential
`patches/metal-ops-ggml0190.patch` (patch 3) wires the six kernels that
exist in the donor (qvac-ext-ggml, MIT; per-operator credit in
[operators.md](operators.md)) and keeps the other four ops on the clean
CPU fallback. Patch 3 also restores the baseline Metal `GGML_OP_REPEAT`
gate, which patch 1 had accidentally coupled to the stricter grouped/padded
`CONV_TRANSPOSE_1D` gate; this is required for pure-Metal composed
baselines.

## Test harness

`tests/test_qvac_ops.c` accepts a `metal` backend argument. Compile the
Metal variant (macOS only), then sandwich the Metal run between green
CPU baselines:

```bash
clang -O2 -DUSE_METAL -I ggml-src/include -I ggml-src/src \
  -o test_qvac_ops_metal tests/test_qvac_ops.c \
  -L ggml-src/build-metal/src -L ggml-src/build-metal/src/ggml-metal \
  -lggml-base -lggml-cpu -lggml-metal \
  -framework Foundation -framework Metal -framework MetalKit

./test_qvac_ops_metal cpu      # CPU regression (must stay ALL PASSED)
./test_qvac_ops_cpu   cpu      # if built separately
./test_qvac_ops_metal metal    # the Metal path
```

With patch 3, the 18 Supertonic/Snake cases execute on Metal, while
the 16 GRU/zero-upsample/channel-shuffle/affine-PReLU cases print the
expected clean-fallback `SKIP` messages. Correctness needs no extra
work: the test compares the Metal op output against the hand-written
CPU reference already embedded in the test. For a delivery-grade run,
execute the Metal suite twice and require identical results (see
[Verified delivery](#verified-delivery) below).

## Known gotchas

- **kargs are positional ABI**: Metal binds `constant & args` structs by
  field order. Host struct and kernel struct must match exactly — no
  reordering, no middle-field insertion (a trailing addition is a
  coordinated change across host and shader).
- **`layer_norm_channel` threadgroup**: `nth` must be a multiple of 32
  (simdgroup) and ≤ 256; the dispatcher computes it that way. Shared
  memory is `8 * sizeof(float)` — one float per simdgroup. The
  integrated kernel adds a barrier after every simdgroup consumes the
  reduced mean and before variance partials reuse `shared[0]`. The
  donor omitted this barrier; C=256/L=4096 stress testing reproduced
  wrong results in 9/10 independent processes without it and passed
  20/20 with it.
- **`depthwise_1d` peelings are compile-time** on K ∈ {3, 5, 7}; any
  other K falls through the `else` branch treating it as K=3 — the
  supports_op gate rejects other K values, matching upstream.
- **`bias_gelu` bit-compat**: the kernel uses the baseline's
  `erf_approx` (Abramowitz–Stegun/Hastings), the same polynomial the
  baseline's `kernel_gelu_erf_f32` uses, while the CPU reference uses
  `erff`. The 1e-4 test tolerance absorbs the polynomial difference;
  do not "fix" the kernel to exact `erff` — that would break
  bit-identity with the unfused Metal gelu path.
- **`snake` baseline overlap**: v0.19.0 already contains a type-generic
  `kernel_snake` used by composed-graph fusion. Patch 3 reuses that
  identical formula and pipeline for direct `GGML_OP_SNAKE` dispatch
  instead of adding a duplicate `kernel_snake_f32` symbol.
- **Placeholder buffer binding**: `depthwise_1d` with `bias == NULL`
  binds `src[0]` at index 3 as a placeholder (Metal requires all
  declared buffers bound). Keep it.
- **Vulkan-side lessons apply** (see [porting-notes.md](porting-notes.md)):
  fresh allocator per graph variant, no sched, etc.

## Verified delivery

Verified on 2026-08-29 with macOS 27.0 (26A5421a), Apple M4 (10-core GPU),
Xcode 27.0 (27A5237l), and Apple Clang 21.0.0:

- patch 1 → patch 2 → patch 3 applies cleanly to pristine `30bf868`;
- Metal build succeeds and all 18 enabled Metal cases execute without SKIP;
- the 16 cases for the four ops without Metal kernels cleanly SKIP;
- explicit gates reject depthwise K=9, invalid layouts, mixed Snake types,
  and all four kernel-less ops;
- `test_qvac_ops_metal cpu`, `test_qvac_ops_cpu cpu`, and
  `test_learned_ops_cpu cpu` report `ALL PASSED`;
- two Metal runs have identical test-result projections and both end with:

```text
[test] metal supports_op gates / fallback envelope
  done (0 failures so far)

ALL PASSED
```
