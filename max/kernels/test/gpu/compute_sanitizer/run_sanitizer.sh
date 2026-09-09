#!/usr/bin/env bash
##===----------------------------------------------------------------------===##
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##===----------------------------------------------------------------------===##
# ===----------------------------------------------------------------------=== #
# Local driver: run Mojo GPU kernel tests under NVIDIA Compute Sanitizer or the
# built-in MAX redzone debug allocator, then summarize findings.
#
#   run_sanitizer.sh <tool> <bazel target>...
#
# <tool> is one of:
#   memcheck   - out-of-bounds / misaligned global access (exact kernel line).
#                NOTE: small OOBs are masked by the device caching allocator;
#                use `redzone` to catch realistic off-by-one global overflows.
#   racecheck  - intra-block shared-memory data races (pool-independent).
#   initcheck  - uninitialized global-memory reads.
#   synccheck  - divergent / mismatched __syncthreads / named-barrier. NOTE:
#                this lane reports "Divergent thread(s) in block" for every
#                `named_barrier` in the warp-specialized SM100 attention/MLA
#                family. PTX permits warps to arrive at different `bar.sync`
#                instructions sharing a barrier name -- exactly what those
#                partial-CTA rendezvous do -- but synccheck accepts arrivals
#                from only one instruction address, and NVIDIA supports
#                `--suppressions` for memcheck/initcheck/racecheck but NOT for
#                synccheck. Expect ~48 noise findings on SM100. KERN-3536.
#   redzone    - MAX's own guard-region allocator (no compute-sanitizer): fast,
#                catches small global OOB at free time (host alloc/free trace).
#
# Env knobs: CS_JOBS (local test parallelism, default 6), COMPUTE_SANITIZER
# (path to the binary), CS_RESULTS_DIR (output dir, default
# .derived/cs-findings).
#
# compute-sanitizer is compiled with `--mojocopt=--debug-level
# --mojocopt=line-tables` (line-tables, NOT full: keeps ptxas -O optimizations
# so racecheck sees the real schedule).
# ===----------------------------------------------------------------------=== #
set -uo pipefail

usage() {
  echo "usage: $0 <memcheck|racecheck|initcheck|synccheck|redzone> <target>..." >&2
  exit 2
}
TOOL="${1:-}"
[ -n "$TOOL" ] || usage
shift
[ "$#" -gt 0 ] || usage

WORKDIR="$(git rev-parse --show-toplevel)"
cd "$WORKDIR" || exit 1
CS="${COMPUTE_SANITIZER:-/usr/local/cuda/bin/compute-sanitizer}"
RESULTS="${CS_RESULTS_DIR:-$WORKDIR/.derived/cs-findings}/$TOOL"
mkdir -p "$RESULTS"

COMMON=(
  # Build the test binaries under the shared CI GPU config (same as the
  # Nightly Kernel Allocator Sweep). This routes the heavy Mojo compiles to
  # the remote/cached builders and builds in `opt` rather than the default
  # `dbg`, so the `--debug-level line-tables` compile of large kernels (e.g.
  # test_flash_attention) no longer balloons to ~200GB RSS and OOMs the
  # runner ("lost communication"). `opt` also matches the "keep ptxas -O
  # optimizations" intent noted above.
  --config=ci-local-gpu
  --test_output=errors --curses=no --noshow_progress
  --nocache_test_results --keep_going
  "--test_timeout=900,2400,5400,10800"
  "--local_test_jobs=${CS_JOBS:-6}"
  # GPU tests declare a `gpu-memory` resource that only the remote executor
  # tracks; declare it locally so tests are schedulable on this box.
  --local_resources=gpu-memory=1000
  # Pin the sweep to one physical GPU (CS_GPU) so independent sweeps can run
  # concurrently on different B200s without contention.
  "--test_env=CUDA_VISIBLE_DEVICES=${CS_GPU:-0}"
)

# POOL: for the allocator-sensitive tools (memcheck/initcheck) we disable MAX's
# device caching allocator so each buffer is a 1:1 device allocation and the
# sanitizer sees true per-buffer bounds (small OOBs / OOB reads are otherwise
# masked inside the shared ~205MB pool). racecheck/synccheck are pool-
# independent; redzone validates *within* the pool, so neither gets the flag.
#
# MOJOCOPTS: extra `-D` defines a lane needs on top of the shared line-tables
# build.
POOL=""
MOJOCOPTS=()
case "$TOOL" in
  memcheck)  EXTRA="--leak-check no --report-api-errors no"; POOL="--//:gpu_disable_memory_manager" ;;
  racecheck) EXTRA="--racecheck-report all" ;;
  # `--track-unused-memory` takes NO argument in compute-sanitizer 2025.4.1 and
  # defaults to OFF (the noisy unused-memory check we want disabled anyway), so
  # we omit it. (The design doc's `--track-unused-memory no` is wrong: it makes
  # `no` the target application -> "Target application doesn't exist".)
  #
  # FA4_WS_POISON: `launch_workspace` (sm100/dispatch.mojo) deliberately leaves
  # the split-K `o_partial`/`lse_partial` workspace unfilled -- an empty
  # partition skips its O store while still writing `lse_p = -inf`, and
  # `fa4_splitk_combine`'s `scale != 0` select is what substitutes a literal 0
  # instead of evaluating `0 * garbage`. initcheck sees only the load, so
  # without this every empty partition reads as an uninitialized-memory
  # finding. NaN-filling the workspace both silences that (the memory IS
  # initialized) and GRADES the no-initialization contract: a dropped select or
  # a missing `-inf` LSE write now propagates NaN into the output and trips the
  # test's own assert. See KERN-3535.
  initcheck) EXTRA=""; POOL="--//:gpu_disable_memory_manager"
             MOJOCOPTS=(--mojocopt=-D --mojocopt=FA4_WS_POISON=1) ;;
  synccheck) EXTRA="" ;;
  redzone)   EXTRA="" ;;
  *) usage ;;
esac

LOG="$RESULTS/run.log"
echo ">>> tool=$TOOL targets=$# results=$RESULTS" | tee "$LOG"

# Targets go after a `--` separator so negative patterns (e.g. `-//pkg:target`)
# passed by the caller are parsed as target patterns, not flags.
if [ "$TOOL" = "redzone" ]; then
  # No compute-sanitizer; the allocator validates guard patterns at free time.
  ./bazelw test "${COMMON[@]}" \
    --test_env=MODULAR_DEBUG_DEVICE_ALLOCATOR=out-of-bounds \
    -- "$@" 2>&1 | tee -a "$LOG"
else
  RUNUNDER="$CS --tool $TOOL --target-processes all --launch-timeout 0 --error-exitcode 1 $EXTRA"
  # Exclude filecheck tests under the compute-sanitizer lanes: they can't run
  # under `--run_under`, and one (test_gather_nd_oob) is a deliberate
  # expect_crash OOB that would surface a spurious memcheck "out of bounds"
  # finding. Restrict to `gpu`-tagged tests while we're at it.
  ./bazelw test "${COMMON[@]}" $POOL \
    --test_tag_filters=gpu,-filecheck \
    --run_under="$RUNUNDER" \
    --mojocopt=--debug-level --mojocopt=line-tables \
    ${MOJOCOPTS[@]+"${MOJOCOPTS[@]}"} \
    -- "$@" 2>&1 | tee -a "$LOG"
fi

# --- Extract findings from each target's test.log -----------------------------
# Markers that indicate a *real* finding. Deliberately precise: we key off
# specific violation phrases ONLY, never the bare "ERROR SUMMARY: N" count --
# that count also tallies "Internal Sanitizer Error" (the SM100 nvjet/cuBLAS
# instrumentation gap), which would false-positive on every GEMM test that uses
# a vendor reference. Those are reported separately as coverage gaps below.
#
# NOTE: we deliberately gate on the *device-side* initcheck marker
# ("Uninitialized __global__ memory read" -- a real in-kernel read of
# uninitialized global memory) and NOT on the *host-side* "Host API memory
# access error / Uninitialized access ... on access by cudaMemcpy source".
# The latter is dominated by benign noise: kernels legitimately leave masked
# or padded output positions unwritten and the tests copy the whole buffer
# back to host, which trips initcheck on ~30 nn tests. Gating on it would make
# the lane permanently red on non-bugs; the device-side marker is the
# high-signal one (it already catches e.g. the qslice_conv3d uninit read).
MARKERS='Invalid __(global|shared|local|device)__|Race reported|Barrier error|Uninitialized __global__|misaligned address|is out of bounds|MemoryManager detected a device buffer (under|over)flow|CUDA_EXCEPTION|illegal memory access'
echo "" | tee -a "$LOG"
echo "==================== FINDINGS SUMMARY ($TOOL) ====================" | tee -a "$LOG"
found_any=0
# Expand any `//...` / `:all` wildcards to concrete test targets for scanning.
EXPANDED=()
for a in "$@"; do
  while IFS= read -r t; do [ -n "$t" ] && EXPANDED+=("$t"); done < <(./bazelw query "tests($a)" 2>/dev/null)
done
[ "${#EXPANDED[@]}" -eq 0 ] && EXPANDED=("$@")
for tgt in "${EXPANDED[@]}"; do
  # //pkg:name -> bazel-testlogs/pkg/name/test.log
  rel="${tgt#//}"; rel="${rel/://}"
  tl="bazel-testlogs/$rel/test.log"
  [ -f "$tl" ] || continue
  hits="$(grep -nE "$MARKERS" "$tl" 2>/dev/null | grep -vE 'ERROR SUMMARY: 0 errors' | head -40)"
  if [ -n "$hits" ]; then
    found_any=1
    echo "" | tee -a "$LOG"
    echo "### $tgt" | tee -a "$LOG"
    echo "$hits" | tee -a "$LOG"
    cp "$tl" "$RESULTS/$(echo "$rel" | tr '/:' '__').log"
  fi
done
[ "$found_any" = 0 ] && echo "(no findings markers in scanned test logs)" | tee -a "$LOG"

# Bazel test failures stay ungated in general -- under `--error-exitcode 1` a
# lane's benign findings turn ~33 PASSING initcheck targets into bazel FAILs,
# which is exactly why the gate keys off violation markers instead.
#
# The initcheck lane needs one addition. `-D FA4_WS_POISON=1` makes a broken
# split-K no-initialization contract surface as the test's own AssertionError
# rather than as a sanitizer marker, so the marker grep alone would report
# green over it. `_startup.mojo` prints "Unhandled exception caught during
# execution" whenever `main` raises, which distinguishes a test that actually
# raised from one that merely inherited compute-sanitizer's exit code.
#
# A raise is NOT sufficient on its own, though. `test_mla_index_kpool` raises
# `CUDA call failed: CUDA_ERROR_INVALID_VALUE` on a device-to-device copy
# under this lane on PLAIN MAIN, with no poison and no local changes -- it is
# the pool being disabled (`--//:gpu_disable_memory_manager`), not the
# workspace fill. Gating on any raise therefore reports an unrelated
# pre-existing failure as a poison finding. Driver/API errors are excluded:
# a violated no-initialization contract propagates NaN and fails the test's
# own numerical comparison, it does not make a CUDA call reject its arguments.
#
# Scoped to initcheck deliberately: memcheck, racecheck and synccheck each
# have pre-existing raisers (test_toppminp_gpu's AssertionError under warp
# serialization, test_topk_topp_sampling_with_dist's CUDA_ERROR_LAUNCH_FAILED),
# so gating them here would just trade one red lane for another. Those want
# their own tickets, not a gate flip.
if [ "$TOOL" = "initcheck" ]; then
  for tgt in "${EXPANDED[@]}"; do
    rel="${tgt#//}"; rel="${rel/://}"
    tl="bazel-testlogs/$rel/test.log"
    raise="$(grep -m1 -A1 'Unhandled exception caught during execution' "$tl" \
      2>/dev/null)"
    [ -n "$raise" ] || continue
    # Driver/API failure, not a numerical contract violation -- see above.
    case "$raise" in *"CUDA call failed"*) continue ;; esac
    found_any=1
    echo "" | tee -a "$LOG"
    echo "### $tgt raised while -D FA4_WS_POISON=1 was set" | tee -a "$LOG"
    echo "$raise" | tee -a "$LOG"
    cp "$tl" "$RESULTS/$(echo "$rel" | tr '/:' '__').log"
  done
fi
echo "==================================================================" | tee -a "$LOG"
echo ">>> full logs under $RESULTS" | tee -a "$LOG"

# Exit non-zero when a real violation marker was found, so this doubles as a CI
# gate (and gives local callers a meaningful exit status). Internal Sanitizer
# Errors and other bazel failures are intentionally NOT gated here -- they are
# coverage gaps (e.g. SM100 cuBLAS) to triage from the logs, not regressions.
exit "$found_any"
