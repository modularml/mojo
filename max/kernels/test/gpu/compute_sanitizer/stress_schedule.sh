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
# Schedule-amplification harness for inter-block global-memory races.
#
#   stress_schedule.sh <bazel target>...
#
# Compute Sanitizer's `racecheck` only sees *intra-block* shared-memory races;
# inter-block races on global memory (MoE atomic-offset claims across blocks,
# split-K / stream-K reduction order, allreduce) are invisible to all four
# tools. The classic way to expose them (Chao Peng, GPGPU '20) is *schedule
# amplification*: run the same input under many execution schedules and check
# the (reference-compared) output stays invariant.
#
# This harness amplifies the schedule with three knobs that need no kernel
# changes, leaning on the fact that every MAX kernel test already self-checks
# against a reference:
#   1. Concurrency/contention -- run N copies of the test at once on one GPU
#      (`--runs_per_test=N --local_test_jobs=N`); they fight for SMs and memory,
#      maximally perturbing block-to-SM scheduling and global-memory timing.
#   2. Repetition -- N independent runs surface timing-dependent races.
#   3. Sync-mode -- `MODULAR_DEVICE_CONTEXT_SYNC_MODE=1` changes when the host
#      synchronizes, shifting kernel overlap.
#
# A target that passes single-run but FAILS (or flakes) under amplification has
# an inter-block race or other schedule-dependent nondeterminism. Aim it at
# atomics/reduction kernels (MoE, split-K, stream-K, scan, allreduce).
#
# Env: STRESS_REPS (default 40), STRESS_JOBS (concurrent copies, default 8),
# CS_GPU (physical GPU, default 0).
# ===----------------------------------------------------------------------=== #
set -uo pipefail

[ "$#" -gt 0 ] || { echo "usage: $0 <bazel target>..." >&2; exit 2; }
WORKDIR="$(git rev-parse --show-toplevel)"; cd "$WORKDIR" || exit 1
REPS="${STRESS_REPS:-40}"
JOBS="${STRESS_JOBS:-8}"
RESULTS="${CS_RESULTS_DIR:-$WORKDIR/.derived/cs-findings}/stress"
mkdir -p "$RESULTS"; LOG="$RESULTS/run.log"

COMMON=(
  --test_output=errors --curses=no --noshow_progress --nocache_test_results
  --keep_going --runs_per_test_detects_flakes
  --local_resources=gpu-memory=1000
  "--test_env=CUDA_VISIBLE_DEVICES=${CS_GPU:-0}"
)

echo ">>> schedule amplification: reps=$REPS concurrent=$JOBS targets=$#" | tee "$LOG"

# Pass 1: concurrency/contention -- N copies fight for the GPU simultaneously.
echo "=== pass 1: concurrent contention (${JOBS} copies x ${REPS} runs) ===" | tee -a "$LOG"
./bazelw test "$@" "${COMMON[@]}" \
  --runs_per_test="$REPS" --local_test_jobs="$JOBS" 2>&1 | tee -a "$LOG"

# Pass 2: device sync-mode toggled (shifts host/device overlap), serial repeats.
echo "=== pass 2: sync-mode + repetition (${REPS} runs) ===" | tee -a "$LOG"
./bazelw test "$@" "${COMMON[@]}" \
  --runs_per_test="$REPS" --local_test_jobs=1 \
  --test_env=MODULAR_DEVICE_CONTEXT_SYNC_MODE=1 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "==================== STRESS SUMMARY ====================" | tee -a "$LOG"
# A target that flaked/failed under amplification is a candidate inter-block
# race. bazel marks these FAILED / FLAKY in its summary.
grep -E "FAILED|FLAKY|flaky|failed in|fails locally" "$LOG" 2>/dev/null | grep -vE "were skipped" | sort -u | tee -a "$LOG.summary"
if [ ! -s "$LOG.summary" ]; then
  echo "(no failures/flakes under amplification -- no schedule-dependent" \
       "race surfaced in these targets; NOTE: a clean result is not proof of" \
       "race-freedom, only that this schedule budget did not trigger one)" | tee -a "$LOG"
fi
echo "========================================================" | tee -a "$LOG"
echo ">>> logs under $RESULTS" | tee -a "$LOG"
