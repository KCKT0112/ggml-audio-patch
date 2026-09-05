# Audio operator correctness fixes (patch 7)

> English | **[中文](audio-op-fixes_zh.md)**

Apply `patches/audio-op-fixes-ggml0190.patch` after the existing collection.
It is generated from a committed ggml worktree diff; patches 1–6 are unchanged.
These fixes address the operator findings in pc-nsf-hifigan.cpp PR #2.

- CPU and Vulkan ScatterElements normalize indices in `[-dimension, -1]`.
  Out-of-range updates are discarded before destination address calculation;
  unaffected destinations retain the original data. Both overwrite and add
  reductions use this rule. Vulkan handles `INT32_MIN` without signed overflow.
- Transposed convolution accepts nonnegative padding even when it does not
  divide stride (for example `pad=2, stride=3`). Existing backend-specific
  support gates still apply.
- CPU direct convolution packs weights using `w->nb[]` byte offsets and bias
  using `bias->nb[0]`, honoring non-contiguous views. GPU contiguity restrictions
  remain unchanged.

`tests/test_audio_op_regressions.cpp` contains 14 CPU cases: all four scatter
axes and both reductions, grouped/ungrouped transposed convolution with
non-dividing padding, and contiguous/strided direct-convolution weights and
biases. It checks finite outputs against explicit expected results. The
existing collection reference functions and tolerances are unchanged.

Build using the CMake harness in `tests/`, then run:

```bash
ctest --test-dir build-metal-tests --output-on-failure
# With GGML_VULKAN=ON in the harness build:
build-vulkan-tests/test_audio_op_regressions Vulkan0
```

CPU verification: Apple M4, macOS 27.0, AppleClang 21, Release, 2026-09-05:
`CPU: 14 passed, 0 unsupported`. Vulkan runtime validation is provided by the
consumer PR's Linux/Mesa CI job; it was not run on this macOS host.

Patch 8 additionally fixes AVX2 scratch alignment; two added cases deliberately use scratch not aligned to 32 bytes, without changing CPU arithmetic order.
