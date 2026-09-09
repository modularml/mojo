# Compute Sanitizer for Mojo GPU kernels

NVIDIA Compute Sanitizer is a runtime correctness suite that ships with the CUDA
Toolkit. It instruments a CUDA application at launch and watches every device
memory access and synchronization event, reporting violations the way Valgrind
or AddressSanitizer do for host code — out-of-bounds accesses, data races,
uninitialized reads, and barrier misuse. Unlike our differential tests (which
catch a bug only through its *numerical* effect, and often only intermittently),
the sanitizer catches the **mechanism** deterministically and, when the kernel
is compiled with GPU line tables, pins each finding to a Mojo `file:line`.

## The tools it provides

Compute Sanitizer runs **exactly one tool per invocation** — they cannot be
combined in a single pass, which is why each is a separate lane.

| Tool        | Detects                                                                                   | Kernel bug class for us                                                       |
|-------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| `memcheck`  | Out-of-bounds and misaligned accesses (global/local/shared), plus leaks & hardware errors | `UnsafePointer`/`LayoutTensor` stride overruns, `async_copy` over-reads       |
| `racecheck` | Shared-memory data races between threads **in the same block**                            | Missing/misplaced `barrier()` around SMEM reductions and `cp.async` tiles     |
| `initcheck` | Reads of uninitialized device **global** memory                                           | Scratch/workspace/accumulator buffers read before fully written               |
| `synccheck` | Misuse of synchronization primitives (`__syncthreads`, `__syncwarp`, named barriers)      | Divergent or mismatched-count barriers in warp-specialized SM90/SM100 kernels |

Compute Sanitizer also exposes a public **API** (callback, patching, and memory
components) for building custom tracing tools; we do not use it here.

Two limits drive the rest of this directory's design:

- **`racecheck` is intra-block only.** Inter-block races on global memory (MoE
  atomic-offset claims across blocks, split-K / stream-K reduction order,
  allreduce) are invisible to all four tools. `stress_schedule.sh` covers that
  gap (below). A green `racecheck` is **not** proof of race-freedom.
- **The MAX device caching allocator masks small OOBs and uninit reads.** Every
  buffer is carved from one ~205 MB pool, so a realistic off-by-one lands inside
  the pool and the sanitizer sees nothing. `memcheck`/`initcheck` therefore need
  the pool disabled (below).

## What's in this directory

- `run_sanitizer.sh` — local driver. Runs the `tags=["gpu"]` Mojo kernel tests
  under a chosen tool with GPU line tables on, then greps each `test.log` for
  real violation markers and writes a findings summary.
- `stress_schedule.sh` — schedule-amplification harness for the inter-block
  races `racecheck` can't see: it runs each (reference-checked) test under many
  concurrent copies and a toggled sync mode; a target that flakes only under
  amplification has a schedule-dependent race.
- `positive_control_memcheck_oob.mojo` — a kernel with a deliberate OOB global
  write. It "passes" **only** under `memcheck`, proving the lane catches real
  bugs and attributes them to a Mojo line. Tagged `manual` so it never runs in
  the normal suite.
- `positive_control_poison_uninit.mojo` — reads a never-initialized buffer; with
  the NaN-poison allocator on it surfaces a NaN, proving the uninit-read net
  works. Also `manual`.

## Running it locally

```bash
# <tool> ∈ memcheck | racecheck | initcheck | synccheck | redzone
max/kernels/test/gpu/compute_sanitizer/run_sanitizer.sh memcheck \
  //max/kernels/test/gpu/memory/...

# Validate the lane end-to-end against the positive control:
max/kernels/test/gpu/compute_sanitizer/run_sanitizer.sh memcheck \
  //max/kernels/test/gpu/compute_sanitizer:positive_control_memcheck_oob.mojo.test
```

Knobs: `CS_JOBS` (local parallelism), `CS_GPU` (pin to one physical GPU),
`CS_RESULTS_DIR`, `COMPUTE_SANITIZER` (path to the binary). For the stress tool:
`STRESS_REPS`, `STRESS_JOBS`.

## Design notes

- **Line tables, not full debug info.** The harness compiles with
  `--mojocopt=--debug-level --mojocopt=line-tables`. `full` (`-G`) disables
  `ptxas` optimization and perturbs timing, which both slows the run and can
  *hide* the very races we hunt; `line-tables` keeps `-O` and still attributes
  findings to Mojo source.
- **Pool disable for `memcheck`/`initcheck`.** These two add the landed build
  flag `--//:gpu_disable_memory_manager` (the harness does this automatically),
  which makes each buffer a 1:1 `cuMemAlloc` so the sanitizer sees true
  per-buffer bounds. `racecheck`/`synccheck` are pool-independent and don't get
  it. Caveat: pool-off can false-positive on vectorized tail over-reads and
  buffer-adjacency-assuming kernels — triage, don't gate from day one.
- **Two MAX-native complements** (no Compute Sanitizer needed, ~native speed):
  - `redzone` mode — `MODULAR_DEBUG_DEVICE_ALLOCATOR=out-of-bounds` validates
    guard regions at free time. Catches small global OOB **writes** *inside* the
    pool, so it's the fast everyday OOB sweep. Misses OOB reads.
  - NaN-poison — `MODULAR_DEBUG_DEVICE_ALLOCATOR=poison-all` fills *every*
    fresh allocation with `0xFF` (NaN), so an uninitialized read propagates NaN
    into the output and is caught by the *existing* differential tests — closing
    the uninit-**use** gap without `initcheck`'s copy-back noise. This is the
    allocator-layer, type-agnostic backstop; for graph tensors specifically the
    graph driver also offers `uninitialized-poison`, a dtype-aware, non-NaN
    sentinel paired with an instrumented Mojo load check (precise source
    location, but tensors only). The two compose — see
    `MLRT/docs/Driver/MemoryManagerOverview.md`.
- **SM100 / Blackwell coverage gap.** Compute Sanitizer can't instrument cuBLAS
  `nvjet_sm100_*` kernels ("didn't track the launch"), so GEMM tests that call a
  vendor reference emit storms of Internal Sanitizer Errors. The findings grep
  keys off specific violation phrases only — never the bare `ERROR SUMMARY: N`
  count — so those gaps don't false-positive. Prefer reference-free MAX-kernel
  tests under the sanitizer.
- **`initcheck` runs with `-D FA4_WS_POISON=1`.** `launch_workspace`
  (`sm100/dispatch.mojo`) deliberately leaves the split-K
  `o_partial`/`lse_partial` workspace unfilled — an empty partition skips its
  O store while still writing `lse_p = -inf`, and `fa4_splitk_combine`'s
  `scale != 0` select is what substitutes a literal 0 instead of evaluating
  `0 * garbage`. initcheck sees only the load, so an un-poisoned lane reports
  every empty partition as an uninitialized read. NaN-filling the workspace
  both silences that (the memory *is* initialized) and turns the lane into a
  real grader of the no-initialization contract: a dropped select or a missing
  `-inf` LSE write now propagates NaN into the output and trips the test's own
  assert.
- **`synccheck` cannot report usefully on SM100.** A warp-specialized kernel
  rendezvouses two warpgroups with `named_barrier[2 * WARPGROUP_SIZE](id)` —
  `bar.sync id, 256` — and PTX lets the participating warps arrive at
  *different* `bar.sync` instructions sharing that barrier name. `synccheck`
  only accepts arrivals from one instruction address, so it reports "Divergent
  thread(s) in block" for the whole SM100 attention/MLA family — about 48
  targets, essentially all noise. NVIDIA supports `--suppressions` for
  `memcheck`, `initcheck` and `racecheck` but **not** for `synccheck`, so there
  is no way to filter it. **The lane is expected to fail.** KERN-3536 tracks
  it; until that is resolved, read the sweep's result per lane rather than
  as a single pass/fail.
- **No known-failure list.** A lane either reports nothing or the finding gets
  fixed. There is no file of parked targets to rot, which is also why the
  `synccheck` noise is not suppressed.
- **Operational.** GPU tests need `--local_resources=gpu-memory=1000` locally;
  exclude `mojo_filecheck_test` (can't run under `--run_under`) and perf
  benchmarks (pathologically slow under `racecheck`); never
  `kill -9` a bazel test action (it crashes the bazel server).
