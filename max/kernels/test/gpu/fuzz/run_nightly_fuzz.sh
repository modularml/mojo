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
# Nightly driver: run the in-tree GPU kernel fuzzer (fuzz.py) as a time-boxed
# LIVE SEARCH over a curated target list for one oracle lane, then summarize.
#
#   run_nightly_fuzz.sh <oracle> [target ...]
#
# <oracle> is one of the fuzz.py oracles wired below into a per-oracle default
# target list (used when no explicit <target> is given):
#   memcheck     - compute-sanitizer OOB read/write (device pool disabled).
#   initcheck    - compute-sanitizer uninitialized global read.
#   redzone      - MAX guard-region allocator OOB write (native speed).
#   ref          - numeric mismatch vs a higher-precision reference.
#   determinism  - run-to-run bit-stability (races / order-dependent atomics).
#   contract     - NaN/Inf finiteness contract.
#
# Unlike the deterministic corpus-replay presubmit gate, this is a fresh random
# search every night, so a finding is an EXPECTED, intermittent outcome to
# triage -- the caller (the nightly workflow) treats a non-zero exit as a
# soft-fail + notify, not a broken build.
#
# Env knobs (all optional):
#   FUZZ_BUDGET       cases per target (default 24).
#   FUZZ_SEED         generation seed (default: date +%Y%m%d). The workflow
#                     passes $github.run_number so each night explores new
#                     cases yet a given run reproduces exactly.
#   FUZZ_TIMEOUT      per-case wall-clock seconds; exceeding it = HANG (default 120).
#   FUZZ_GPU          device index to pin the search to (default 0).
#   FUZZ_VENDOR       "nvidia" (default) or "amd". Only affects the per-oracle
#                     DEFAULT target list (below) when no explicit <target> is
#                     given -- AMD drops targets whose kernel has no CDNA4/MI355
#                     implementation. Ignored once an explicit <target> list is
#                     passed on the command line.
#   FUZZ_RESULTS_DIR  where to collect the corpus + JSONL run logs for upload
#                     (default .derived/fuzz-findings).
#
# fuzz.py builds each target itself (bazel) and runs the built binary DIRECTLY
# (not via `bazel test`), so this needs a LOCAL GPU on the runner.
#
# One broken/finding target never stops the sweep: each target runs in
# error-tolerant control flow and is classified CLEAN / FINDING / ERROR. Exits
# non-zero iff any target produced a real fuzz FINDING or a hard (build/infra)
# ERROR -- matching run_sanitizer.sh's "exits non-zero iff a real finding".
# ===----------------------------------------------------------------------=== #
# Intentionally NOT `-e`: we continue past a per-target failure and report it.
set -uo pipefail

usage() {
  echo "usage: $0 <memcheck|initcheck|redzone|ref|determinism|contract> [target ...]" >&2
  exit 2
}

ORACLE="${1:-}"
[ -n "$ORACLE" ] || usage
shift

WORKDIR="$(git rev-parse --show-toplevel)"
cd "$WORKDIR" || exit 1

FUZZ_DIR="max/kernels/test/gpu/fuzz"
FUZZ_PY="$FUZZ_DIR/fuzz.py"

FUZZ_BUDGET="${FUZZ_BUDGET:-24}"
FUZZ_SEED="${FUZZ_SEED:-$(date +%Y%m%d)}"
FUZZ_TIMEOUT="${FUZZ_TIMEOUT:-120}"
FUZZ_GPU="${FUZZ_GPU:-0}"
FUZZ_VENDOR="${FUZZ_VENDOR:-nvidia}"
RESULTS="${FUZZ_RESULTS_DIR:-$WORKDIR/.derived/fuzz-findings}/$ORACLE"
mkdir -p "$RESULTS"

# Curated per-oracle default target lists. These oracle<->target pairings are
# verified to build and run on main. Targets that are broken on main, and the
# always-red positive-control canaries (oob_canary / numeric_canary), are
# deliberately omitted so a red run means a real, new candidate finding rather
# than known noise.
#
# On AMD (FUZZ_VENDOR=amd), redzone/ref/contract additionally drop every target
# whose kernel is SM100-only (no CDNA4/MI355 implementation exists yet), plus
# attn_res_mix -- that kernel is portable, but its fuzz target arms PDL on every
# shape and has only ever been run on B200, so an AMD lane would be claiming
# coverage nobody has measured. Add it once it runs green on MI355. memcheck and
# initcheck have no AMD entry at all -- compute-sanitizer (the oracle tool
# itself, not the targets) has no AMD equivalent wired into fuzz.py. determinism
# has no AMD entry either: its only two targets (matmul, gemv_split_k) are both
# SM100-only, so there is nothing AMD-safe to run under that oracle yet.
default_targets() {
  case "$1" in
    memcheck)    echo softmax rms_norm layer_norm reductions matmul mha_nullmask block_scaled_fp4 block_scaled_mxfp8 grouped_matmul_mxfp8 grouped_matmul_sm100_w4a8 mla_decode fused_rope_rmsnorm fused_qkv_matmul_mxfp8 fused_qkv_index_matmul_mxfp8 attn_res_mix ;;
    initcheck)   echo moe_indices topk_sampling ;;
    redzone)
      if [ "$FUZZ_VENDOR" = "amd" ]; then
        echo softmax rms_norm layer_norm reductions mha_nullmask
      else
        echo softmax rms_norm layer_norm reductions matmul mha_nullmask block_scaled_fp4 attn_res_mix
      fi
      ;;
    ref)
      if [ "$FUZZ_VENDOR" = "amd" ]; then
        echo reductions moe_router msa_decode msa_prefill sparse_indexer sparse_indexer_decode sparse_indexer_prefill fused_qk_rms_norm fused_qk_rope
      else
        echo reductions block_scaled_mxfp8 grouped_matmul_mxfp8 grouped_matmul_sm100_w4a8 moe_router fused_swiglu_mxfp8 fused_swiglu_dispatch msa_decode msa_prefill sparse_indexer sparse_indexer_decode sparse_indexer_prefill fused_qk_rms_norm fused_qk_rope attn_res_mix topk_topp_masked_probs topk_topp_sampling_dist
      fi
      ;;
    # Held out until their live findings are fixed (else the notify lane is
    # always red): mha_causal (BUG-03, valid_length=0 hang) and ep_combine
    # (BUG-04, send_tokens_back OOB write). Re-add once fixed.
    determinism) echo matmul gemv_split_k attn_res_mix ;;
    contract)
      if [ "$FUZZ_VENDOR" = "amd" ]; then
        echo mxfp8_quantize moe_router sparse_indexer sparse_indexer_decode sparse_indexer_prefill
      else
        echo mxfp8_quantize moe_router sparse_indexer sparse_indexer_decode sparse_indexer_prefill attn_res_mix
      fi
      ;;
    *) return 1 ;;
  esac
}

# Validate the oracle up front (also yields the default target list).
DEFAULT_LIST="$(default_targets "$ORACLE")" || {
  echo "::error::unknown oracle '$ORACLE'" >&2
  usage
}
if [ "$FUZZ_VENDOR" = "amd" ] && [ -z "$DEFAULT_LIST" ] && [ "$#" -eq 0 ]; then
  echo "::error::oracle '$ORACLE' has no AMD-safe default target -- pass an explicit target list, or run this oracle on nvidia only" >&2
  exit 1
fi

if [ "$#" -gt 0 ]; then
  TARGETS=("$@")
else
  # shellcheck disable=SC2206  # word-splitting the space-separated list is intended.
  TARGETS=($DEFAULT_LIST)
fi

LOG="$RESULTS/run.log"
: > "$LOG"
SUMMARY_MD="$RESULTS/summary.md"
: > "$SUMMARY_MD"

echo ">>> oracle=$ORACLE vendor=$FUZZ_VENDOR targets=${#TARGETS[@]} budget=$FUZZ_BUDGET seed=$FUZZ_SEED timeout=${FUZZ_TIMEOUT}s gpu=$FUZZ_GPU" | tee -a "$LOG"
echo ">>> results=$RESULTS" | tee -a "$LOG"

{
  echo "## Nightly kernel fuzz: \`$ORACLE\` lane"
  echo ""
  echo "budget=$FUZZ_BUDGET seed=$FUZZ_SEED timeout=${FUZZ_TIMEOUT}s"
  echo ""
  echo "| target | status | verdicts |"
  echo "| --- | --- | --- |"
} >> "$SUMMARY_MD"

clean=0
findings=0
errors=0

for tgt in "${TARGETS[@]}"; do
  tgt_log="$RESULTS/${tgt}.log"
  echo "" | tee -a "$LOG"
  echo ">>> [$ORACLE] fuzz target=$tgt" | tee -a "$LOG"

  python3 "$FUZZ_PY" \
    --target "$tgt" \
    --oracle "$ORACLE" \
    --budget "$FUZZ_BUDGET" \
    --seed "$FUZZ_SEED" \
    --timeout "$FUZZ_TIMEOUT" \
    --gpu "$FUZZ_GPU" > "$tgt_log" 2>&1
  rc=$?

  # fuzz.py prints its "[fuzz] done: ..." verdict-count line only after a full
  # run. Its absence on a non-zero exit means the run itself broke (a bazel
  # build failure, a crashed orchestrator) rather than the search finding a
  # bug, so classify those apart: a broken target must not masquerade as a
  # finding, nor a finding as infra breakage.
  verdicts="$(grep -m1 '^\[fuzz\] done:' "$tgt_log" | sed 's/^\[fuzz\] done: //')"
  if [ "$rc" -eq 0 ]; then
    status="CLEAN"
    clean=$((clean + 1))
  elif [ -n "$verdicts" ]; then
    status="FINDING"
    findings=$((findings + 1))
  else
    status="ERROR"
    errors=$((errors + 1))
    verdicts="(no run summary -- build/infra error; see ${tgt}.log)"
  fi

  echo ">>> $tgt: $status  ${verdicts}" | tee -a "$LOG"
  # Surface the failing tail inline so a finding is visible in the job log
  # without opening the uploaded artifact.
  if [ "$status" != "CLEAN" ]; then
    tail -n 25 "$tgt_log" | sed 's/^/    /' | tee -a "$LOG"
  fi
  echo "| \`$tgt\` | $status | ${verdicts:-PASS} |" >> "$SUMMARY_MD"
done

echo "" | tee -a "$LOG"
echo "==================== FUZZ SUMMARY ($ORACLE) ====================" | tee -a "$LOG"
echo ">>> clean=$clean findings=$findings errors=$errors (of ${#TARGETS[@]} targets)" | tee -a "$LOG"
echo "===============================================================" | tee -a "$LOG"

{
  echo ""
  echo "clean=$clean findings=$findings errors=$errors (of ${#TARGETS[@]} targets)"
} >> "$SUMMARY_MD"

# Collect the replayable corpus + per-run JSONL logs into the results dir so the
# workflow's upload-artifact step ships them.
if [ -d "$FUZZ_DIR/corpus" ]; then
  cp -a "$FUZZ_DIR/corpus" "$RESULTS/corpus" 2>/dev/null || true
fi
if [ -d "$FUZZ_DIR/.fuzzruns" ]; then
  mkdir -p "$RESULTS/fuzzruns"
  cp -a "$FUZZ_DIR/.fuzzruns/." "$RESULTS/fuzzruns/" 2>/dev/null || true
fi

# Mirror the human-readable summary into the GitHub step summary when present.
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  cat "$SUMMARY_MD" >> "$GITHUB_STEP_SUMMARY"
fi

# Exit non-zero iff the live search surfaced a finding OR a target hit a hard
# (build/infra) error -- so this doubles as the workflow's soft-fail signal.
if [ "$findings" -gt 0 ] || [ "$errors" -gt 0 ]; then
  exit 1
fi
exit 0
