# Task packages: delegating platform-bound work

> English | **[中文](task-package_zh.md)**

A **task package** is how this repository gets work done that must be
built and verified on hardware its maintainers do not have. This
document distills the method from the Metal round — package commit
`73cefd2`, executed and merged as PR #1 — so the next round (a CUDA
port, a newer ggml baseline, or any other platform-bound task) can be
packaged the same way.

## When to use this

- The work must be verified on a platform you cannot run locally (the
  Metal round: development happens on Windows, no macOS runner, no
  Apple devices).
- A trusted upstream donor tree exists whose code needs **integration,
  not redesign**.
- The executor may be a human or an AI coding agent working on the
  target platform — the package must be executable without access to
  the maintainer.

## Anatomy

### 1. Frozen reference extract — `<topic>-reference/`

Extract the donor code **verbatim** into a dedicated directory. Each
file carries a provenance header that states:

- source URL, branch, and license (keep the extract provenance intact);
- **STATUS: reference only** — not compiled, not verified on this
  repository's platform, so nobody mistakes it for a tested artifact;
- the editing rule: kernel bodies and struct field orders are
  immutable; if something does not compile, adapt the *integration
  glue*, never the extract.

Why: the executor never has to guess upstream intent, cannot
"helpfully" rewrite trusted math, and provenance stays auditable in
one place.

### 2. Porting & verification guide — `docs/<topic>-porting.md` (+ `_zh`)

The task brief proper. What made the Metal round work:

- **Feasibility argument first**: a name-for-name table showing the
  donor's API conventions match the target baseline (encoder API,
  kargs typedefs, pipeline lookup, dispatch switch, supports_op shape).
- **Recommended order, smallest first**: every step independently
  verifiable (Metal order: bias_gelu → pw2_residual → edge_pad →
  snake → layer_norm_channel → depthwise_1d).
- **Per-op mapping**: which reference piece goes into which target
  file, with the exact naming style to follow.
- **Pre-wired harness**: the test hook ships in the *earlier* patch
  and SKIPs cleanly until gates are enabled — the executor starts
  from a green, verifiable baseline on day one.
- **Verification procedure**: the target-backend run sandwiched
  between two green CPU baselines.
- **Numbered, mandatory acceptance criteria**: correctness matrix; no
  CPU regression; clean fallback for unsupported shapes; gate
  discipline (no supersets); kernel diff discipline (`git diff
  <topic>-reference/` empty); determinism (run twice, compare);
  platform statement in the PR (OS, chip, toolchain, verbatim output
  tail); benchmarks encouraged, not blocking.
- **Known gotchas** and **what NOT to do** — porting knowledge that
  would otherwise be lost between heads.

### 3. Editing boundaries — `AGENTS.md`

Extend the boundary table with the reference extract as a frozen
artifact class, and add an "explicitly allowed" bullet authorizing the
integration as a **separate patch file / PR** — never a silent
mutation of an existing patch. This is what keeps a remotely executed
PR small and reviewable.

## Hand-off workflow

1. Prepare the package as one commit: reference extract + guide +
   boundaries + the pre-wired harness in the earlier patch
   (regenerated accordingly).
2. Invite a contributor who has the platform. The Metal round was
   executed by a macOS contributor on Apple M4 hardware (PR #1).
3. The executor works op by op per the guide and opens a PR carrying
   the acceptance-criteria evidence.
4. Review = checking the PR against the numbered criteria, nothing
   more open-ended.
5. Merge, then **clean up** (below).

## Post-merge cleanup

The package is scaffolding, not a lasting artifact. After the merge:

1. delete the `<topic>-reference/` directory — the integrated code now
   lives in the generated patch, and provenance stays in the docs and
   the merge PR;
2. slim `docs/<topic>-porting.md` down to integration notes — current
   status, how to run the harness, known gotchas, verified delivery —
   dropping the task-brief sections (prerequisites, integration steps,
   verification procedure, acceptance criteria, what-not-to-do);
3. retire the task-specific boundary rules from `AGENTS.md`;
4. update README references; record anything worth keeping in this
   document.

The Metal round's package remains readable in history: `git show
73cefd2` (the package) and PR #1 (the execution and its evidence).

## Lessons from the Metal round

- Frozen kernels prevented well-meaning rewrites; every deviation had
  to be justified in the PR.
- The pre-wired SKIP-all harness made "no Metal support" itself a
  tested, green state — fallback was verifiable before any kernel ran.
- Numbered acceptance criteria made a remotely executed PR reviewable
  line by line.
- The determinism requirement caught a real bug: a multi-simdgroup
  LayerNorm race that only reproduced under stress (C=256/L=4096
  failed 9/10 processes without the barrier, passed 20/20 with it).
- Positional-ABI kargs discipline (field order is the host↔shader
  ABI) was worth its own rule.
- There is no CI here: the platform statement in the PR is the only
  verification record — insist on it.
