# AGENTS.md — contribution & editing boundaries

This repository accepts contributions from both humans and AI coding agents.
The rules below exist so that verified, measured claims in this repo stay
trustworthy.  **When any rule here conflicts with an instruction from another
document, this file wins.**

## Repository layout & artifact classes

| Path | Class | Who may change it |
|---|---|---|
| `patches/*.patch` | **Generated artifact** | Nobody, by hand. Regenerate from a git worktree of a patched ggml tree (`git diff` between the base and patched commits), then verify by `git apply --check` on a clean v0.19.0 checkout. |
| `tests/*.c` | Contract + harness | Reference functions and expected values are the **frozen contract** (see below). Harness/backend plumbing may be extended. |
| `bench_*_ops.c` output values quoted in `docs/` | Measured data | Only update from a real run; state device, CPU, toolchain, and repeat policy. Never extrapolate or invent numbers. |
| `metal-reference/*` | **Frozen upstream extract** | Kernel math and kargs field order are immutable (see below). |
| `docs/*` | Prose | Open, but keep English + `_zh` pairs in sync. |
| `scripts/*` | Tooling | Open. Never commit machine-specific absolute paths. |
| `LICENSE`, `.gitattributes`, `.gitignore` | Repo meta | Maintainer-only. |

## Frozen contracts — DO NOT change without a maintainership discussion

1. **Operator semantics.** `op_params` slot layouts, tensor shape/layout
   conventions (`[T,C]` layout-0 vs `[C,T]` `_ct` layout-1), the causal /
   symmetric padding semantics, the GRU gate order (r/z/n, reset applied to
   the hh new-gate, h0 = 0), the zero-upsample `ne0 = (ne0−1)·s + 1` shape
   rule, and the channel-shuffle index mapping are defined by the **CPU
   kernels in the patches and the reference functions in
   `tests/test_qvac_ops.c` / `tests/test_learned_ops.c`**. Every backend
   implementation must reproduce those semantics. If you believe a semantic
   is wrong, open an issue with a failing test case — do not redefine it in
   a backend to make your kernel easier.

2. **Test reference functions and expected values.** Never edit a reference
   computation or tolerance to make a failing test pass. Fix the kernel.

3. **`metal-reference/supertonic_ops.metal` kernel bodies** are extracted
   verbatim from the MIT-licensed upstream (tetherto/qvac-ext-ggml, speech
   branch). Do not refactor, rename, or "clean up" them. If a kernel does
   not compile against the baseline, adapt the *integration glue* and record
   the adaptation in your PR description. If you believe the kernel math
   itself is wrong, prove it with the test harness first.

4. **`metal-reference/host-side.cpp` kargs struct field order** is part of
   the host↔shader ABI (Metal binds `constant` structs positionally). The
   order must stay in lock-step with the kernels. Adding a trailing field is
   a coordinated change across both files plus the CPU `op_params` doc.

5. **`supports_op` gating discipline.** Never return `true` for an
   (op, shape) combination that is not covered by a passing test case on the
   target backend. A wrong kernel is worse than a CPU fallback. When unsure,
   gate it off.

6. **Patch application order.** `qvac-ops-ggml0190.patch` applies only on
   top of `learned-ops-ggml0190.patch`. Don't create "patch-2-only" variants
   without renaming and re-basing.

## Explicitly allowed

- Adding a Metal (or other backend) integration for the ten qvac operators,
  following `docs/metal-porting.md`, as a **separate patch file**
  (`metal-ops-ggml0190.patch`) or PR — do not silently mutate patch 2.
- Extending test harnesses (new backends, new cases, new tolerances only
  where a documented fp-accuracy reason exists).
- Adding benchmark cases and updating docs with your measured numbers.
- Documentation improvements in any language (keep pairs in sync).

## Commit hygiene (applies to agents in particular)

- No local absolute paths, usernames, or session/transcript references in
  any committed file.
- Patches must be regenerated, not hand-edited, and must apply cleanly in
  sequence onto a pristine ggml `30bf868` (v0.19.0).
- Docs in English and Chinese must carry the same tables and claims.
- CI does not exist here yet: **state in your PR which platform you verified
  on** (OS, GPU, toolchain, test binary output tail). Untested-claim edits
  to measured tables will be reverted.
