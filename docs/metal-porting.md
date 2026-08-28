# Metal porting & verification guide

> English | **[中文](metal-porting_zh.md)**

How to bring the ten qvac operators (patch 2) to Apple's Metal backend, how
to verify the result, and the acceptance criteria a Metal port must meet
before it can be merged as `patches/metal-ops-ggml0190.patch`.

Read **[/AGENTS.md](../AGENTS.md) first** — it defines the editing
boundaries every contributor (human or agent) works under. Short version:
the CPU kernels, the test reference functions, and the
`metal-reference/*` kernel bodies are **frozen contracts**; integration
glue is yours to write.

## Current status

| Operator | CPU | Vulkan | Metal |
|---|---|---|---|
| Supertonic × 5 | ✅ | — (CPU fallback) | 📋 reference available, **not wired** |
| GRU / ZERO_UPSAMPLE / CHANNEL_SHUFFLE / AFFINE_PRELU | ✅ | ✅ | ❌ no upstream kernel exists |
| SNAKE | ✅ | ✅ | 📋 reference available, **not wired** |

Patch 2 ships **no Metal integration**: `ggml-metal-device.m`'s
`supports_op` has no cases for the ten ops, so its `default: return false`
keeps every one of them on the clean CPU fallback. macOS users lose nothing
today; they gain nothing yet either. `metal-reference/` contains the
upstream-verified kernel sources and all host-side glue, ready to be wired
in.

## Why the reference should integrate cleanly

The qvac donor tree and the v0.19.0 baseline use the **same Metal backend
architecture**, and the integration-critical names match one-for-one
(verified against both trees at extraction time):

| Item | v0.19.0 baseline | qvac donor | match |
|---|---|---|---|
| kargs typedef convention (`ggml_metal_kargs_*` in `ggml-metal-impl.h`) | ✅ | ✅ | same |
| encoder API (`ggml_metal_encoder_set_buffer` / `_bytes` / `set_pipeline` / `dispatch_threadgroups` / `set_threadgroup_memory_size`) | ✅ `ggml-metal-device.h` | ✅ | same signatures |
| pipeline lookup convention (`ggml_metal_library_get_pipeline_*` in `ggml-metal-device.cpp` + decls in `ggml-metal-device.h`) | ✅ | ✅ | same |
| op dispatch switch (`ggml_metal_op_*` in `ggml-metal-ops.cpp` + decls in `ggml-metal-ops.h`) | ✅ | ✅ | same |
| `erf_approx<T>` template + `SQRT_2_INV` in `ggml-metal.metal` | ✅ (used by `kernel_gelu_erf_f32`) | ✅ | same |
| `supports_op` switch in `ggml-metal-device.m` | ✅ `default: return false` | ✅ | same shape |

The op dispatch switch (`ggml-metal-ops.cpp`, `case GGML_OP_...:` →
`n_fuse = ggml_metal_op_...`) and the supports_op switch are the only two
places besides the three "append" locations.

## Prerequisites (the only way to verify)

- macOS 13+, Apple Silicon (or AMD) GPU, Xcode 15+ toolchain.
- The patched tree: v0.19.0 + patch 1 + patch 2 (`git apply` in order).
- A CPU build that works first: `cmake -B build-metal -DGGML_METAL=ON ...`
  and `test_qvac_ops_cpu cpu` passing — Metal work starts from a green CPU
  baseline.

## Integration steps (one op at a time, smallest first)

Recommended order — each step is independently verifiable:

1. `supertonic_bias_gelu` (simplest, has an in-graph oracle:
   `ggml_add` + `ggml_gelu_erf`)
2. `supertonic_pw2_residual` (pure elementwise)
3. `supertonic_edge_pad_1d`
4. `snake`
5. `supertonic_layer_norm_channel` (simdgroup reduction)
6. `supertonic_depthwise_1d` (three kernel-width peelings + causal flag)

For each op:

1. **Kernels**: append the kernel from `metal-reference/supertonic_ops.metal`
   to `src/ggml-metal/ggml-metal.metal` (do not modify the body —
   AGENTS.md rule 3).
2. **kargs**: copy the struct from `metal-reference/host-side.cpp`
   (PIECE 1) into `src/ggml-metal/ggml-metal-impl.h`. Field order is ABI —
   rule 4.
3. **Pipeline lookup**: copy the `ggml_metal_library_get_pipeline_*`
   function (PIECE 3, inside the `#if 0` block) into
   `ggml-metal-device.cpp`; add the matching declaration to
   `ggml-metal-device.h` (style: `struct ggml_metal_pipeline_with_params
   ggml_metal_library_get_pipeline_<name>(ggml_metal_library_t lib, const
   struct ggml_tensor * op);`).
4. **Dispatcher**: copy the `ggml_metal_op_*` function (PIECE 4) into
   `ggml-metal-ops.cpp`; declare in `ggml-metal-ops.h` (style: `int
   ggml_metal_op_supertonic_bias_gelu(ggml_metal_op_t ctx, int idx);`);
   wire the switch case: `case GGML_OP_SUPERTONIC_BIAS_GELU: { n_fuse =
   ggml_metal_op_supertonic_bias_gelu(ctx, idx); } break;`.
5. **Gate**: add the `supports_op` case in `ggml-metal-device.m` copied
   from PIECE 5 — but only for this one op.
6. **Build**: rebuild; the baseline compiles `ggml-metal.metal` into the
   metallib during the build, so a shader syntax error surfaces here.
7. **Test** (below), then move to the next op.

## Test-harness hook (already wired in patch 2)

`tests/test_qvac_ops.c` accepts a `metal` backend argument. Compile the
Metal variant (macOS only):

```bash
clang -O2 -DUSE_METAL -I ggml-src/include -I ggml-src/src \
  -o test_qvac_ops_metal tests/test_qvac_ops.c \
  -L ggml-src/build-metal/src -lggml-base -lggml-cpu -lggml-metal \
  -framework Foundation -framework Metal -framework MetalKit
./test_qvac_ops_metal metal
```

With gates off (as shipped), every case prints `SKIP: metal does not
support this op shape` and exits `ALL PASSED` — that is the correct
pre-integration state and doubles as your fallback sanity check.

## Verification procedure (per op, after enabling its gate)

Run **all three** — the Metal run is only meaningful between two green
baselines:

```bash
./test_qvac_ops_metal cpu      # CPU regression (must stay ALL PASSED)
./test_qvac_ops_cpu   cpu      # if built separately
./test_qvac_ops_metal metal    # the new path
```

Then run the composed-vs-fused cross-check on Metal itself: the test
compares the Metal op output against the hand-written CPU reference
already embedded in the test — no extra work needed.

## Acceptance criteria (all mandatory)

1. **Correctness**: `test_qvac_ops metal` reports `ALL PASSED` with the
   full case matrix for the enabled ops (depthwise: 5 cases incl. causal
   K=7 and dilation; LN: 3; pw2: 2; bias_gelu: 2; edge_pad: 4; snake: 2).
   Tolerances stay as shipped (1e-3 elementwise; 1e-4 for bias_gelu/snake
   erf paths).
2. **No CPU regression**: `test_qvac_ops cpu` and `test_learned_ops cpu`
   still `ALL PASSED` with the same binary set.
3. **Clean fallback**: for shapes you did not enable (e.g. depthwise with
   K=9, or any of the four ops without a Metal kernel),
   `ggml_backend_supports_op` returns false and the graph produces
   CPU-identical results through the fallback — demonstrate with one
   deliberately-unsupported case per op (the harness prints SKIP; add it
   as a comment in the PR).
4. **Gate discipline**: `supports_op` returns true only for the exact
   (type, shape, param) envelope that passed criterion 1 — not a superset.
5. **Kernel diff discipline**: `git diff metal-reference/` in your branch
   is empty (kernels untouched; AGENTS.md rules 3–4). All changes live in
   the four integration files + the supports_op gate.
6. **Determinism**: run the Metal suite twice; both runs identical.
7. **Platform statement**: PR description lists OS version, chip/GPU,
   Xcode version, and the verbatim tail of `test_qvac_ops_metal metal`
   (the `ALL PASSED` block).
8. **Benchmark (encouraged, not blocking)**: extend
   `tests/bench_qvac_ops.c` with a `metal` branch the same way; report
   fused-vs-composed numbers per the doc's table format, with device/OS.
   Add numbers to `docs/benchmarks*.md` only from these real runs.

## Known gotchas (learned from the donor tree)

- **kargs are positional ABI**: Metal binds `constant & args` structs by
  field order. Host struct and kernel struct must match exactly — no
  reordering, no middle-field insertion (trailing addition is a coordinated
  change per AGENTS.md rule 4).
- **`layer_norm_channel` threadgroup**: `nth` must be a multiple of 32
  (simdgroup) and ≤ 256; the reference dispatcher already computes it that
  way. Shared memory is `8 * sizeof(float)` — one float per simdgroup.
- **`depthwise_1d` peelings are compile-time** on K ∈ {3, 5, 7}; any other
  K falls through the `else` branch treating it as K=3 — the supports_op
  gate must reject other K values (upstream gates on K ∈ {3,5,7}; do the
  same).
- **`bias_gelu` bit-compat**: the kernel uses the baseline's `erf_approx`
  (Abramowitz–Stegun/Hastings), the same polynomial the baseline's
  `kernel_gelu_erf_f32` uses, while the CPU reference uses `erff`. The
  1e-4 test tolerance absorbs the polynomial difference; do not "fix" the
  kernel to exact `erff` — that would break bit-identity with the
  unfused Metal gelu path.
- **`snake` grid-stride**: dispatcher caps threadgroups at 65535 and relies
  on the grid-stride loop for larger N — don't "simplify" it to a
  one-thread-one-element dispatch.
- **Placeholder buffer binding**: `depthwise_1d` with `bias == NULL` binds
  `src[0]` at index 3 as a placeholder (Metal requires all declared
  buffers bound). Keep it.
- **Vulkan-side lessons apply** (see `docs/porting-notes.md`): fresh
  allocator per graph variant, no sched, etc.

## What NOT to do

- Do not enable all gates at once and debug six kernels together.
- Do not edit `tests/test_qvac_ops.c` reference functions (AGENTS.md rule
  2), `metal-reference/*` (rule 3/4), or patch 2 directly.
- Do not commit local paths (`/Users/...`) or non-reproducible claims.

Once your branch meets the checklist, the integration becomes
`patches/metal-ops-ggml0190.patch` (sequential on top of patch 2) via the
standard regeneration procedure (AGENTS.md artifact table).
