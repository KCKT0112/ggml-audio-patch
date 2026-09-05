# Metal direct convolution (patch 6)

> English | **[中文](metal-direct-conv_zh.md)**

This collects the F32 Metal kernels developed in pc-nsf-hifigan.cpp
(`0abc343`) as a reusable backend patch against ggml v0.19.0 (`30bf868`).
Patch 6 is generated from committed ggml source changes, after applying the
collection's existing patches 1–5 unchanged. It also applies after 1–3 alone.

## Scope and integration

`GGML_OP_CONV_DIRECT_1D` uses the existing seven-slot parameter contract.
Weights are `[K, IC, OC]`, input `[T, IC]`, optional bias `[OC]`, optional
residual and output `[OL, OC]`, where `OL = T + 2*pad - dil*(K-1)`.
Input scale precedes input leaky-ReLU; bias, residual, then output leaky-ReLU
follow the convolution. All storage and arithmetic remain F32. Floating-point
accumulation order differs from CPU, so bit-identical CPU output is not promised.

The implementation gathers input tiles directly into threadgroup memory instead
of materializing im2col. Output-channel counts select `16×64`, `32×64`, or
`64×64` tiles, with K tile 8. `OC < 8` uses one scalar thread per time position.
Input/weight staging memory is reused for the output spill after a barrier;
the largest tile needs 16 KiB instead of 20 KiB. Bias, residual, input
scale/leaky and output leaky are fused in the same dispatch.

The patch also adds `IM2COL_FAST_1D` to Metal's existing im2col capability gate:
patch 1 already supplied the dispatch alias, but not the `supports_op` case.
CPU, Vulkan and CUDA kernels and public op semantics are unchanged.

Metal direct convolution requires `ggml_backend_metal_supports_family(backend, 7)`
(Apple7 simdgroup matrices), contiguous F32 tensors, a single sequence, valid
padding/dilation, matching dimensions, and indices fitting the shader's signed
32-bit arithmetic. This is a GPU capability gate, not an M-series name check.
Only Apple M4 has been tested; Intel/AMD Metal GPUs and other Apple devices have
not been validated.

**Raw-graph consumers must check capability before selecting this graph.**
`ggml_backend_graph_compute` does not insert a fallback. Check
`ggml_backend_supports_op` for each compute node and either rebuild the original
im2col/matmul graph or use a scheduler with an appropriate CPU backend. Checking
only a backend name beginning with `MTL` is insufficient. The historical
pc-nsf-hifigan.cpp `0abc343` consumer still has that limitation; its manual
escape hatch is `PCNSF_DIRECT_CONV=0`.

## Build and correctness reproduction

From the collection checkout, with a sibling clean ggml v0.19.0 checkout:

```bash
git -C ../ggml-src checkout 30bf868
git -C ../ggml-src apply "$PWD/patches/learned-ops-ggml0190.patch"
git -C ../ggml-src apply "$PWD/patches/qvac-ops-ggml0190.patch"
git -C ../ggml-src apply "$PWD/patches/metal-ops-ggml0190.patch"
git -C ../ggml-src apply "$PWD/patches/metal-conv-direct-1d-ggml0190.patch"
GGML_SRC=../ggml-src bash scripts/build-and-test-metal.sh
```

For the full stack, apply patches 4 and 5 between 3 and 6. Use a fresh ggml
checkout rather than layering the collection patch over a consumer's already
modified patch snapshots. Embedded Metal source avoids requiring Xcode's
separate Metal Toolchain; the runtime compiles kernels on first use.

The script builds both existing harnesses, runs CPU and Metal CTests, then
repeats the Metal tests. Reference functions and tolerances are unchanged.
The learned-op harness now checks every compute node before raw execution,
rejects unavailable backend arguments, and checks compute status.

Verified on 2026-09-05: Apple M4, macOS 27.0 build 26A5425a, AppleClang 21.0.0,
Release, `GGML_NATIVE=OFF`, `GGML_METAL_EMBED_LIBRARY=ON`:

- CPU/Metal CTest: **4/4 passed**; Metal repeat: **2/2 passed**. A separate
  `GGML_METAL=OFF` build passed both CPU tests (**2/2**).
- Direct-convolution cases: **29/29 executed on Metal**, including the original
  17 cases plus all three matrix variants, scalar outputs, odd channel counts,
  K=1/2, dilation, time tails, padding and fusion combinations.
- Rejection tests cover F16/strided weights, batches, mismatched bias/residual,
  invalid padding/dilation and output lengths.
- Existing qvac CPU/Metal references pass; unsupported ops retain explicit SKIP
  results, which are not reported as GPU execution.

Expected learned Metal tail:

```text
  case 28: OC=33 IC=13 K=3 OL=65 checked
  done (0 failures so far)
...
[test] Metal direct-conv supports_op rejection cases

ALL PASSED
```

## Historical vocoder measurements (2026-08-31)

These numbers come from the consumer benchmark, not a new full-model run of
this collection packaging. Apple M4, macOS 27.0, AppleClang 21, Release;
OpenVPI `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.ckpt`, F32 GGUF, 1722 frames,
881664 output samples at 44100 Hz (19.992 s). The graph shrank from 989 to 152
nodes, with 97 direct convolutions replacing the im2col paths.

| Path | Time | Repeat policy |
|---|---:|---|
| Original Metal im2col/matmul | 6.828–8.911 s | recorded baseline range |
| Fused Metal, fastest stable series | 0.8360 / 0.8251 / 0.8231 / 0.8228 s | four calls with the model loaded once |
| Fused Metal, later series | 1.3107 / 1.3501 / 1.3131 / 1.3103 s | four calls; independent same-source build also measured 1.3825 s |
| Fused Metal near the ORT comparison | median 1.22085 s | four calls: 1.2591 / 1.1924 / 1.2019 / 1.2398 s |
| ORT CPU EP 1.24.1, 9 threads | median 3.1560455 s | two warmups, six measured calls, session creation excluded |

The nearby-time median comparison is **2.59×** faster than ORT CPU; the fastest
Metal series is **23.9–24.3× realtime**. No cause was established for the timing
variation, so all recorded series are retained. The longer user-run sample
(16903 frames, 196.243 s output) took **8.0354 s**, **24.42× realtime** (one run).
Feature extraction and model loading are outside `hifigan_run` timing.

New Metal versus ggml CPU: corr `0.9999999999994206`, max error `2.693e-6`,
RMS `8.919e-8`, p99.9 error `7.562e-7`. Versus ORT CPU: corr
`0.9999999999991221`, max error `2.354e-6`, RMS `1.103e-7`.

Rejected tuning candidates included `64×32`, K tile 16, `128×64`, `64×128`,
and phase-interleave/crop in the epilogue; they were slower in the recorded
vocoder runs and are not part of this patch.

To reproduce full-model timing in the consumer, supply its F32 GGUF and the
same 44.1 kHz / hop512 / 128-bin natural-log Slaney mel and frame-aligned F0:

```bash
PCNSF_TIMING=1 build-metal/bin/hifigan_cli model.gguf mel.bin f0.bin output.wav
PCNSF_DIRECT_CONV=0 PCNSF_TIMING=1 build-metal/bin/hifigan_cli model.gguf mel.bin f0.bin baseline.wav
```

Model weights and private audio/features are not included in the collection.
