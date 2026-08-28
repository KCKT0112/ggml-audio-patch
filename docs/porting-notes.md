# Porting Notes: Pitfalls & Fixes

> English | **[中文](porting-notes_zh.md)**

The fixed checklist, the traps hit, and the fixes applied while porting these four operators into ggml v0.19.0. Written for anyone reproducing, extending, or upstreaming this work.

## 0. The fixed routine for adding a ggml operator

Adding an operator touches a fixed set of files, in dependency order:

1. **`include/ggml.h`** — append `GGML_OP_XXX` to the enum (**before** `GGML_OP_COUNT`) + declare the public API.
2. **`src/ggml.c`** —
   - `GGML_OP_NAME[]` / `GGML_OP_SYMBOL[]` tables + the `static_assert(GGML_OP_COUNT == N)` after each — **there are two**, one per table; both must be bumped;
   - the builder function (`ggml_set_op_params` for parameters, `result->src[i]` for inputs);
   - the backward-path `ggml_visit_parents` switch and the `ignore_src` list (if the op has no gradient).
3. **`src/ggml-cpu/ops.h` / `ops.cpp`** — kernel + dispatcher.
4. **`src/ggml-cpu/ggml-cpu.c`** — three switches: compute dispatch, `n_tasks` (thread count), wsize (if a work buffer is needed).
5. **Each backend**: `supports_op` + compute/alias:
   - CUDA: two sites in `ggml-cuda.cu` (dispatch + supports_op) + a kernel file;
   - Vulkan: a dozen sites in the monolithic `ggml-vulkan.cpp` (§4) + the `.comp` shader + registration in `vulkan-shaders-gen.cpp`;
   - Metal: `ggml-metal-ops.cpp` (dispatch), `ggml-metal-device.m` (supports_op), `ggml-metal-device.cpp` (pipeline assertions);
   - the graph splitter (`ggml-backend-meta.cpp`-style) also needs a case or graphs containing the new op will silently leak through a meta backend.

## 1. IM2COL_FAST_1D: alias-style port + one assertion trap

**Aliasing**: fast_1d shares geometry and `op_params` with im2col — only the op tag differs. GPU backends therefore alias to the existing `IM2COL` path with a one-line case each, and only CPU gets a dedicated kernel — the cheapest possible landing, since the value is the CPU kernel's O(1) window algorithm itself.

**Trap 1: im2col never reads the kernel.** The imported CPU kernel carried `GGML_ASSERT(src0->type == GGML_TYPE_F16)` from its origin, where the kernel tensor happened to always be F16. Stock ggml `ggml_conv_1d` uses `dst_type = a->type==BF16 ? F32 : F16` and **keeps the kernel (src0) in F32**; semantically im2col only expands the *input* (src1) and never touches the kernel. Dropping the assertion made F32-kernel + F16-dst pass immediately.

**Trap 2: conv_1d weights are [K, IC, OC].** Building them PyTorch-style as [K, OC, IC] trips the builder assertion `b->ne[1] == a->ne[1]`. ggml puts input channels on `ne[1]` everywhere — transposed relative to PyTorch's [OC, IC, K].

## 2. conv_transpose_1d_ext: fixing upstream + a layout choice

The three inherited bugs (broken CPU groups, CUDA `op_params` mis-indexing, division-by-zero assert) and their fixes are documented in [operators.md](operators.md) §2. Porting decisions recorded here:

**Layout decision**: groups use the PyTorch layout `a=[K, Cout_pg, Cin_g]` so existing ecosystem weights work without conversion. The cost: `src0->ne[1]` no longer equals the total output-channel count — **everywhere that treated `ne01` as Cout had to be re-audited** (which is exactly how the Vulkan `elements` dispatch bug in §4 happened).

**API compatibility**: the legacy 6-arg `ggml_conv_transpose_1d` stays and delegates to ext (`op0=0, g0=1`). `op_params` is uniformly 4 × int32.

## 3. SCATTER_ELEMENTS / REL_POS_BIAS: the builder-assertion lesson

The first scatter draft allowed `indices->ne[d] == 1` (mimicking ONNX broadcasting), but both CPU and Vulkan kernels iterate `updates` and `indices` as **one flat linear sequence** — mismatched shapes cause shifted reads/writes. The correct constraint is dimension-wise `indices->ne == updates->ne`, plus `updates->ne == data->ne` except along `axis`. **Assertions must match how the kernel actually indexes, not merely what the operator spec allows.**

**Two Vulkan-specific actions**:

- scatter's dst is not written by the shader: `buffer_copy(data → dst)` runs first, and a **pipelineBarrier** (`TransferWrite → ShaderRead|ShaderWrite`) between copy and shader is mandatory. The copy is submitted via `ggml_vk_buffer_copy_async` on the same command buffer as the compute pass, ordered by the barrier.
- The accumulate mode needs `atomicAdd`: probe the `VK_EXT_shader_atomic_float` extension *and* the `shaderBufferFloat32AtomicAdd` feature (Turing-and-newer NVIDIA, most RDNA parts support it). The probe result lives in `vk_device` and `supports_op` gates on it. The feature-chain `pNext` wiring copies the existing float8 block's style, guarded by `#if defined(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME)` so old SDKs still compile.

## 4. The Vulkan porting checklist & three big traps

`ggml-vulkan.cpp` is a ~20k-line monolith; each port walks this checklist:

```
vk_device members / pipeline members / push-constants struct
→ extension probing (inside get_device, NOT print_gpu_info!)
→ feature-chain pNext wiring + device_extensions.push_back
→ create_pipelines registration (wg_denoms must match the shader's local_size)
→ op_get_pipeline (pipeline selection by op + type)
→ the elements switch in op_f32 (decides the 3-D workgroup counts)
→ the dispatch branch in op_f32 (decides which buffers get bound)
→ the graph-compute switch (op → dispatch function)
→ supports_op (type / contiguity / parameter guards)
```

**Trap 1: `MemoryBarrier` is a Windows macro.** `vk::MemoryBarrier(a, b)` gets mangled by winnt.h's `#define MemoryBarrier ...`, yielding baffling "vk::__faststorefence does not exist" errors. Vulkan-Hpp call sites avoid it with **braced aggregate initialization** (`{ { src_access, dst_access } }`) — when vulkan.hpp throws weird errors on Windows, suspect the macro first.

**Trap 2: `elements` semantics.** In `op_f32`, `elements` isn't "element count" but a blended convention of "workgroups × wg_denoms": `dispatch_pipeline` computes `ceil(elements[i], wg_denoms[i])`. Different ops use their own conventions (some pre-divide by 512 and let the shader loop). **Copy the convention of the existing ops; don't invent one** — an early rel_pos_bias draft used a flat-x scheme (`{ne,1,1}` + `local_size 256`) and was later switched back to the 3-D dispatch (`{W, H, B*HW}` + `8×8×4`), which covers the z axis more robustly for large inputs.

**Trap 3: `src0->ne[1]` ≠ output channel count.** The original convT dispatch used `src0->ne[1]` (which equals Cout only without groups). With groups, `src0->ne[1] = Cout/g0`, so the dispatch under-allocated workgroups — **the second half of the output tensor came out all zeros while the first half was perfectly correct** (it surfaced as "values diverge starting exactly mid-tensor"). Fixed to `dst->ne[1]`. Lesson: **when aliasing/reusing an existing op's dispatch, audit every place that reads a src shape as if it were the dst shape.**

**Also**:

- `ggml_vk_op_f32` takes push constants as `PC &&`; named variables need `std::move`.
- `ggml_vk_tensor_subbuffer` has a third parameter `allow_misalign`; pass `true` for copy-style operations (a scatter dst may be misaligned).
- A new `.comp` file must also be registered in `vulkan-shaders-gen.cpp` (`string_to_spv`); CMake picks up the file itself via `file(GLOB ... CONFIGURE_DEPENDS)`.
- The VS2019 generator doesn't support the shader rule's `DEPFILE`; on Windows, Vulkan builds need the Ninja generator (see [building.md](building.md)).

## 5. Payoff of the test strategy

`tests/test_learned_ops.c` is written against **hand-computed references** rather than implementation-vs-implementation comparison (only fast_1d is checked in-graph against the original conv_1d). This caught two classes of bugs immediately:

- **Layout inversions** ([K, OC, IC] vs [K, IC, OC]) blew up on the very first CPU run;
- **Grouped-indexing errors** appeared on Vulkan as the precise signature "first half of groups correct, second half all zeros", pointing at dispatch sizing rather than shader math (the shader math is structurally identical to the CPU kernel, which passed — so the only thing that could truncate the output was the dispatch range).

**References come from three independent channels**: convT goldens re-derived from the origin project's own test values; scatter/rel_pos references hand-written from the operator semantics; fast_1d compared in-graph. Cross-sourcing makes "the reference itself was wrong" far less likely.

## 6. Harness notes

- `ggml_cgraph` is opaque to the public API: iterate with `ggml_graph_n_nodes` / `ggml_graph_node`, or recurse over tensor `src[]`.
- In v0.19, `ggml_backend_alloc_ctx_tensors_from_buft` is not public; tests use a graph allocator for intermediates plus a no-alloc context for graph construction.
- The multi-backend `ggml_backend_sched` requires the last backend in the list to be CPU; on this baseline a {GPU, CPU} sched running mixed-op graphs showed allocation anomalies (mul_mat and im2col outputs sharing a buffer, nodes apparently skipped) — the test suite sidesteps this with **galloc + single-backend graph_compute**, which is also llama.cpp's mainstream single-GPU path.
- On Windows, if a test binary can crash mid-run, add `setvbuf(stdout, NULL, _IONBF, 0)` at the top of `main()` — otherwise a crash silently discards everything still buffered.

---

# Patch 2 porting notes (qvac ops)

Porting a *batch* of ten ops from one donor tree (tetherto/qvac-ext-ggml, `speech` branch) onto an older baseline (v0.19.0) surfaced new failure classes beyond patch 1's single-op lessons:

## 7. Donor-vs-baseline drift

- **The donor tree is newer than the target baseline.** qvac's ggml is a v0.20-era dlopen-variant tree (760 files differ). Never copy whole files; port each op as a diff of insertion points. Enums land in a different numeric range (donor `GGML_OP_COUNT` = 107, baseline starts at 104 pre-patch-1) — update **both** `static_assert(GGML_OP_COUNT == N)` sites in `ggml.c`, not one.
- **Check what the baseline already has before porting.** The v0.19 baseline already shipped a *graph-level* snake fusion in its Vulkan backend (`snake_pattern` = mul→sin→sqr→mul→add, plus a `snake_f32` pipeline and `snake.comp`). Naively following the donor produced duplicate `vk_op_snake_push_constants`, a duplicate `pipeline_snake_f32` member and a second `string_to_spv("snake_f32", ...)` — redefinition errors plus a pipeline-name clash. The fix: **reuse the upstream pipeline** (its math and binding layout `{x, a, inv_b, dst}` are identical to the donor's) and delete every duplicate registration. Rule: before porting op X, `git grep -i x` the baseline for names that will collide (shader names, push-constant structs, pipeline members, `string_to_spv` keys).
- **COL2IM_1D and ROLL already existed upstream** in v0.19 — the donor's versions of those builders/kernels must *not* be ported. Only ops missing from the target's enum get ported.

## 8. Vulkan re-registration checklist (per op)

For each new op, all of these must be touched or the build fails or dispatch falls over: push-constant struct → pipeline member → `ggml_vk_create_pipeline` (matching `parameter_count`, `wg_denoms`) → `vulkan-shaders-gen.cpp` `string_to_spv` → `ggml_vk_op_get_pipeline` case → `elements` case in `ggml_vk_op_f32` → dispatch wrapper → graph-compute switch case → `supports_op` → (debug-only) `vk_check_results` clone chain. The `elements` convention note from patch 1 generalizes: the generic unary group computes a flat 512×512 split; a shader that indexes 2-D (`i0 + i1*ne0`) needs its own `elements = {ne0, ne1, 1}` case instead of joining that group.

## 9. Column-major test traps

- The CPU kernels from the donor index buffers in **ggml column-major** (`x[t + c*L]` for a `[L, C]` tensor). A reference written in row-major (`x[t*C + c]`) compiles fine, runs fine, and fails everywhere. Symptom: every case off by "some other element's value", not by magnitude. Fix the test's indexing, not the kernel.
- A multi-result test (layout-0 vs layout-1 variants of the same op) must expand **every result root** into the forward graph — `ggml_build_forward_expand` only pulls in what is reachable from the roots you pass. A tensor built but not passed as a root (or reachable from one) gets **no buffer** from the graph allocator, and the first `tensor_set` on it dies with `GGML_ASSERT(buf != NULL && "tensor buffer not set")`. With three variants (`_1d`, `_ct`, `_causal_ct`) all three roots must be expanded even though the causal one is `NULL` in non-causal cases.

## 10. PowerShell patch-generation trap

`git diff > file` from PowerShell 5.1 writes **UTF-16 LE** (the BOM `FF FE` makes `git apply` report "No valid patches in input"). Generate diffs via `cmd /c "git diff ... > file"` or `git diff --output=file`, then verify the first bytes are ASCII `d i f f`.

## 11. ggml_pad is per-dimension, not per-side

Composing comparison graphs against the fused depthwise op: `ggml_pad(x, p0, p0, 0, 0)` adds `p0` **to each of ne0 and ne1** (per-dimension sizes), not `p0` on each side of `ne0`. A same-padding 1-D conv chain should just use `ggml_conv_1d_dw(w, x, s, p0, d)` — the builder takes padding directly — rather than an explicit pad. (An explicit symmetric pad is `ggml_pad(x, 2*p0, 0, 0, 0)`.)
