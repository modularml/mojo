# Kernel-level fuzzing for Mojo GPU kernels

A boundary-aware fuzzer that searches the kernel input space (shapes, value
distributions, launch configs) and feeds each generated case into a correctness
oracle, with shrinking and a replayable regression corpus. It is the
input-generation layer on top of the in-tree oracles — memory safety (the
redzone/poison device allocators and NVIDIA Compute Sanitizer), numerical
correctness (a higher-precision reference), special-value contracts, and
inter-block race checks.

## How it works

Three layers; only the top is new:

1. **Generate** (Python orchestrator + a Mojo harness) — boundary-aware specs
   from a seed, plus value-distribution fills (uniform/normal/sparse/large/
   all-equal and NaN/Inf/denormal/±0 injection).
2. **Execute** (per-kernel Mojo target) — one `run_one_case` per kernel that
   allocates, fills, launches, and (optionally) compares; reused from the
   existing per-kernel test lifecycle.
3. **Oracle** — classifies each case as PASS or a specific failure (see
   [Oracles](#oracles)).

The orchestrator runs each case in its **own subprocess with a per-case
timeout**, so a hanging case only kills its own process (it does not wedge the
run). On a failure it **shrinks** the spec to a minimal repro and writes a
corpus entry that the replay gate then locks in.

## Running

The orchestrator drives every target generically. The only required choice is
`--target` (which kernel) and `--oracle` (which bug class); everything else has
a default. Run `--help` to list the currently registered targets — each ships
with a sensible default oracle that `--oracle` overrides.

```bash
# List targets and flags:
python3 max/kernels/test/gpu/fuzz/fuzz.py --help

# Memory-safety fuzz of a kernel under Compute Sanitizer:
python3 max/kernels/test/gpu/fuzz/fuzz.py --target mha_causal \
    --oracle memcheck --budget 32 --seed 12345

# Quick diff-oracle smoke (catches hangs + crashes; no sanitizer, fast):
python3 max/kernels/test/gpu/fuzz/fuzz.py --target mha_causal \
    --oracle diff --budget 24

# Reproduce one explicit case (no generation):
python3 max/kernels/test/gpu/fuzz/fuzz.py --target mha_causal --oracle diff \
    --spec seq_len=1,num_keys=1,valid_length=0

# Replay the regression corpus (the deterministic gate):
python3 max/kernels/test/gpu/fuzz/fuzz.py --replay-corpus --timeout 30
```

Confirmed failures and their shrunk specs are recorded under `corpus/<target>/`
with their expected verdict, and the replay gate re-runs them deterministically.

The orchestrator only writes an entry for a non-PASS verdict, so a `_pass_`
entry is always hand-written: an anchor asserting a representative boundary
shape still runs clean, which is what lets the gate catch a regression rather
than only a change in a known failure. Keep them when pruning stale failures.
Note the corpus is exempted from the unreferenced-files lint by a pattern in
`bazel/internal/find_unreferenced_files.py`, and that lint also rejects an
unused pattern — so emptying the corpus entirely means dropping the exemption
in the same change.

## Oracles

Select with `--oracle`:

- `diff` — hangs (timeout) and crashes (exit code); no sanitizer.
- `ref` — numerical correctness vs a higher-precision reference (e.g. an FP64
  CPU recompute or an fp32-accum naive kernel). Emits `FUZZ_NUMERIC_FAIL`.
- `contract` — inject NaN/Inf/large and check a finiteness/propagation contract
  (for example: every softmax output is NaN or in `[0, 1]`, never Inf or
  out-of-range). Robust where a `ref` tolerance diff would false-positive on
  NaN/Inf.
- `schedule` — inter-block race check: force a split-K decomposition and re-run
  the same input N times, flagging any non-bit-exact output.
- `determinism` — run-to-run bit-stability: re-run the same input N times
  (`--rerun 8`, no forced split-K) and flag any non-bit-exact output. Catches
  races / order-dependent atomics on the kernel's default launch.
- `batch_invariance` — run a probe token under two different co-batch
  compositions (`--batch-invariance 1`) and flag if the probe's output rows
  change (`atol=rtol=0`). Locks in a same-batch-different-neighbors invariant;
  a divergence is a real batch-variance finding.
- `batch_variance` — negative control (`--batch-variance 1`), the inverse of
  `batch_invariance`: run the same probe in two batch compositions that straddle
  an M-keyed dispatch breakpoint (dense matmul M=1 GEMV vs M>1 tile GEMM;
  attention decode's default partition heuristic, which keys the count on the
  batch size) and assert the probe's output DIVERGES bit-for-bit. PASS iff
  divergence is observed (proving the invariance oracles have teeth); a
  bit-match emits `FUZZ_CONTRACT_FAIL` and is reported, not swallowed.
- `redzone` — OOB writes, ~native speed, AMD-capable (validated on MI355).
- `poison` — NaN-fills every device allocation (`MODULAR_DEBUG_DEVICE_ALLOCATOR=
  poison-all`), so an uninitialized read propagates NaN into the output and
  trips the diff/ref check. ~native speed, no kernel instrumentation. (The
  allocator's other tier, `uninitialized-poison`, covers only graph-driver
  tensors and needs an instrumented build, so it does not apply to the fuzz
  harness's directly-allocated device buffers.)
- `memcheck` / `initcheck` — Compute Sanitizer with the device pool disabled
  (exact kernel line) for OOB / uninitialized reads.
- `racecheck` / `synccheck` — intra-block shared-memory races / barrier bugs.

Not every target supports every oracle. Each target declares a default oracle
(its primary bug class) and a `ref`/`contract`/`schedule` mode only where a
reference, special-value contract, or split-K decomposition exists; likewise
`determinism`/`batch_invariance`/`batch_variance` only where the target parses
the corresponding flag (`--rerun` / `--batch-invariance` / `--batch-variance`).
Cross-run comparisons always live inside the target process — the orchestrator
issues one verdict per subprocess and never holds two cases' outputs.
The M3 decode pair carries the cross-run oracles: `msa_decode` supports
`determinism` (`--rerun`), and `sparse_indexer_decode` supports
`batch_invariance` (`--batch-invariance`) on top of its `schedule` mode. The
indexer's gate is the load-bearing one -- its scorer sizes split-K from the
batch size, so a request really is decomposed differently depending on its
neighbours, and its scores pick the blocks attention reads.

Targets that fuzz the input value distribution expose a `dist` spec field
(uniform/normal/sparse/large/all-equal); NaN/Inf specials are reachable but kept
out of the auto-mix — they drive the `contract` oracle, not `ref`.

> Oracle reality: the redzone/poison allocators catch **writes / uninitialized
> reads**; OOB **reads** need `memcheck` with the device pool disabled (the
> orchestrator sets `MEMORY_MANAGER_SIZE=0` in the subprocess env, which works
> because it runs the built binary directly rather than via `bazel test`).

## Build-time configs

A target compiles once per dtype/config and reads shapes at run time, so any
parameter the kernel specializes on is a `-D` define rather than a spec field.
Build the target by name with the define, then run with `--no-build`:

```bash
./bazelw build //max/kernels/test/gpu/fuzz:fuzz_topk_topp_sampling_dist.mojo.test \
    --mojocopt=-D --mojocopt=ttsd_bf16=1 --curses=no --noshow_progress
python3 max/kernels/test/gpu/fuzz/fuzz.py --target topk_topp_sampling_dist \
    --oracle ref --no-build --budget 24
```

The sampler targets carry a dtype axis because speculative decoding runs them
at both precisions: `float32` under `draft_proposal: argmax` and `bfloat16`
under `draft_proposal: sampled`, which switches `sampling_logits_dtype` for the
whole sampling path. The in-source default is `float32`, so the bfloat16 arm
only gets covered when it is built explicitly.

| Define        | Target                    | Effect             |
|---------------|---------------------------|--------------------|
| `ttsd_bf16=1` | `topk_topp_sampling_dist` | bfloat16 logits in |
| `ttmp_bf16=1` | `topk_topp_masked_probs`  | bfloat16 logits in |

The corpus records a spec, not a build config, so a replay entry always runs
under the in-source default. Alternate configs belong in a live sweep, not the
replay gate.

## Adding a kernel target

The `add-kernel-fuzz-target` Claude Code skill walks this end-to-end — picking
the fuzz axes, oracle, and reference; writing the target; wiring the build; and
validating locally (run `/add-kernel-fuzz-target`, or just ask to fuzz a
kernel). The summary:

1. Write `fuzz_<kernel>.mojo` with a `CaseSpec` (its fuzzable fields) and a
   `run_one_case(ctx, spec)`. Support the three argv modes — `list-specs`
   (print `FUZZ_SPEC <key>=<val> ...`), `single` (read `--<key>` per field, run
   one case, print `FUZZ_RESULT verdict=PASS`), and `fuzz` (in-process batch).
   Reuse `boundary_int` and the argv helpers from `_fuzz.mojo`.
2. Add a `mojo_test` target in `BUILD.bazel` with `srcs = ["_fuzz.mojo",
   "fuzz_<kernel>.mojo"]`, `main = "fuzz_<kernel>.mojo"`, `tags = ["gpu",
   "manual"]`.
3. Register it in `_TARGETS` in `fuzz.py` (name, bazel target, binary path,
   default oracle).

Spec field names == `FUZZ_SPEC` keys == the target's `--<key>` flags, so the
orchestrator drives any target generically.

## CI integration

The tooling is local-first (proven before any CI spend). Two lanes follow the
design's non-gating → gating rollout:

- **Presubmit (gating, fast, deterministic):** build the fuzz targets, then run
  the corpus-replay gate. Same seed/spec → same verdict, so it never flakes; it
  fails only when a verdict drifts (a regression, a fixed bug whose corpus entry
  needs updating, or a broken oracle).

  ```bash
  ./bazelw build //max/kernels/test/gpu/fuzz:all
  python3 max/kernels/test/gpu/fuzz/fuzz.py --replay-corpus --no-build --timeout 30
  ```

- **Nightly (non-gating / notify-only, slow):** a time-boxed live search per
  oracle. Because the input set is random and fresh each night, a finding is an
  EXPECTED, intermittent event; a red scheduled run means "the live search
  surfaced a candidate finding to triage" and it never gates `main`. This runs
  on GitHub Actions as `.github/workflows/nightlyKernelFuzz.yaml` (a
  `modrunner-b200` self-hosted local-GPU runner — the direct-binary home this
  section recommended), one matrix lane per oracle
  (`memcheck/initcheck/redzone/ref/determinism/contract`), driving
  `run_nightly_fuzz.sh <oracle>` over a curated per-oracle target list with a
  fresh `${{ github.run_number }}` seed. It notifies #kernel-team-notifications
  on a scheduled failure, mirroring the allocator-sweep and compute-sanitizer
  nightlies.

  `synccheck`/`racecheck` are deliberately NOT nightly lanes: `synccheck`
  false-positives on warp-specialized SM100 kernels (block-wide `__syncthreads`
  model vs warpgroup-scoped named barriers), so it is noise for a notify lane.

  GPU-locality caveat: `fuzz.py` runs the built target binaries **directly**
  (not via `bazel test`), so it needs a **local** GPU on the agent — hence the
  `modrunner-b200` runner rather than the remote-execution `persistent-b200`
  queue. Also note `manual`-tagged fuzz targets must be built by explicit name
  — `//...` and `:all` wildcards skip them (which is how the targets in the
  default lists silently bit-rotted against a `LayoutTensor` origin API change;
  give them a build-by-name presubmit to catch that).

  Validate the redzone/poison allocators on MI355 before adding an AMD lane.
