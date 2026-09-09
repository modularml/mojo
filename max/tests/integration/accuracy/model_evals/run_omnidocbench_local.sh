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
#
# One-command local run of the OmniDocBench end-to-end document-parsing eval
# against any OpenAI-compatible endpoint — a local MAX Serve, or an
# authenticated external one.
#
# OmniDocBench scores with a third-party harness (pdf_validation.py) that needs
# its own pinned checkout, a Python 3.11 venv and a LaTeX render toolchain, so
# unlike the other dataset evals it can't be a single bazel target. This
# collapses everything minimaxM3MultiModalDatasetEval.yaml used to do inline —
# clone, venv, apt deps, dataset download, endpoint patching, inference,
# empty-prediction retries and scoring — into one command. The checkout, venv
# and dataset persist under .derived/cache, so repeat runs skip straight to
# inference.
#
# Usage:
#   run_omnidocbench_local.sh [options]
#
# Options:
#   --url URL          Endpoint base URL (default: http://localhost:8000). A
#                       trailing /v1 is optional.
#   --api-key KEY      API key for an authenticated endpoint. Defaults to
#                       $OMNIDOC_API_KEY, then $OPENAI_API_KEY, then "dummy".
#                       Passed to the harness by environment, never written to
#                       disk and never logged.
#   --model MODEL      Model name for the API `model` field. Default:
#                       auto-detect via GET {url}/v1/models.
#   --limit N          Evaluate only N pages, strided across the dataset, for a
#                       smoke run. Subsets the ground truth to match, since the
#                       scorer walks the ground truth and would score an
#                       unpredicted page 0. A limited run's score has a
#                       different denominator than the published baseline, so
#                       it is reported as PARTIAL and never "passes".
#   --workers N        Concurrent inference requests (default: 32, matching the
#                       CI server batch size).
#   --seed N           Per-request seed for reproducibility (omitted when unset).
#   --max-tokens N     max_tokens per request (default: 55000).
#   --temperature F    Sampling temperature (default: 1.0).
#   --top-p F          Nucleus sampling (default: 0.95).
#   --retry-rounds N   Rounds of retrying empty predictions (default: 3).
#                       Upstream writes an empty .md on a request failure, and
#                       an empty prediction scores 0.
#   --out-dir DIR      Results directory (default: /tmp/omnidocbench-results).
#   --infer-only       Produce predictions, skip scoring. Useful when the LaTeX
#                       toolchain isn't available locally.
#   --score-only       Score the predictions already in --out-dir.
#   --refresh          Re-clone the harness and rebuild its venv.
#   -h, --help         Show this help.
#
# Reproduce a CI run against the local server:
#
#   ./bazelw run //max/tests/integration/accuracy/model_evals:omnidocbench_eval_local
#
# Quick smoke (4 pages) while iterating:
#
#   ./bazelw run //max/tests/integration/accuracy/model_evals:omnidocbench_eval_local -- \
#     --limit 4 --workers 4
#
# Score an authenticated external deployment:
#
#   ./bazelw run //max/tests/integration/accuracy/model_evals:omnidocbench_eval_local -- \
#     --url https://<endpoint> --api-key "$MY_KEY" --model MiniMaxAI/MiniMax-M3
#
# Env overrides: OMNIDOCBENCH_CACHE, OMNIDOCBENCH_PIN, OMNIDOC_API_KEY.
#
# Scoring is sensitive to the age of the render toolchain. CI scores CDM 84.8
# on Ubuntu 24.04 (ImageMagick 6.9.12, TeX Live 2023); on Ubuntu 22.04
# (ImageMagick 6.9.11, TeX Live 2022) CDM scores 0 on every sample instead,
# because the colour-coded render it measures comes back grayscale. The other
# metrics are unaffected, so a 22.04 box still gives comparable text, table and
# reading-order numbers — the overall is just ~29 points low. The scorer warns
# when it sees that signature.

set -euo pipefail

# Pinned for reproducibility. This script is the single source of truth for the
# SHA; omnidocbench-constraints.txt locks the dependency tree that matches it.
PIN="${OMNIDOCBENCH_PIN:-2b161d010d2e3aff77a0edef359ea3a6411d23cd}"
HARNESS_REPO="https://github.com/opendatalab/OmniDocBench"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${BUILD_WORKSPACE_DIRECTORY:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}"

CACHE="${OMNIDOCBENCH_CACHE:-$REPO_ROOT/.derived/cache}"
CHECKOUT="$CACHE/OmniDocBench"
DATASET="$CACHE/omnidocbench-dataset"
VENV="$CHECKOUT/.venv"

url="http://localhost:8000"
api_key="${OMNIDOC_API_KEY:-${OPENAI_API_KEY:-}}"
model=""
limit=""
workers=32
seed=""
max_tokens=55000
temperature=1.0
top_p=0.95
retry_rounds=3
out_dir="/tmp/omnidocbench-results"
infer_only=0
score_only=0
refresh=0

usage() { sed -n '15,84p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) url="$2"; shift 2 ;;
    --api-key) api_key="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --limit) limit="$2"; shift 2 ;;
    --workers) workers="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --max-tokens) max_tokens="$2"; shift 2 ;;
    --temperature) temperature="$2"; shift 2 ;;
    --top-p) top_p="$2"; shift 2 ;;
    --retry-rounds) retry_rounds="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --infer-only) infer_only=1; shift ;;
    --score-only) score_only=1; shift ;;
    --refresh) refresh=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; echo "try --help" >&2; exit 1 ;;
  esac
done

if [[ "$infer_only" == 1 && "$score_only" == 1 ]]; then
  echo "error: --infer-only and --score-only are mutually exclusive" >&2
  exit 1
fi

# External endpoints are usually quoted with the /v1 suffix already attached,
# while the harness appends its own; normalize so either form works.
url="${url%/}"
url="${url%/v1}"
url="${url%/}"

# The harness takes a directory, so the pages to run are staged as symlinks:
# a subset for --limit, and just the failures for each retry round.
PREDS="$out_dir/preds"
RUN_IMAGES="$out_dir/run-images"
RETRY_IMAGES="$out_dir/retry-images"
GT_JSON="$out_dir/ground-truth.json"
# build_save_name() in the harness derives its output name from the prediction
# directory's basename plus the match method, so this path is deterministic.
SUMMARY="$CHECKOUT/result/$(basename "$PREDS")_quick_match_run_summary.json"

# Under `bazel run` the prep module is in the runfiles tree; standalone it sits
# next to this script.
RUNFILES="${RUNFILES_DIR:-${BASH_SOURCE[0]}.runfiles}"
PREP=""
for candidate in \
  "$RUNFILES/_main/max/tests/integration/accuracy/model_evals/omnidocbench_prep.py" \
  "$SCRIPT_DIR/omnidocbench_prep.py"; do
  if [[ -f "$candidate" ]]; then
    PREP="$candidate"
    break
  fi
done
if [[ -z "$PREP" ]]; then
  echo "ERROR: omnidocbench_prep.py not found in runfiles or next to this script" >&2
  exit 1
fi
CONSTRAINTS="$REPO_ROOT/max/tests/integration/accuracy/omnidocbench-constraints.txt"
if [[ ! -f "$CONSTRAINTS" ]]; then
  echo "ERROR: dependency lock not found at $CONSTRAINTS" >&2
  exit 1
fi

# Under `bazel run` the parent leaks a bazel-runfiles PYTHONPATH; inheriting it
# into the harness venv would shadow that venv's own packages (the same hazard
# scicode_eval.py's _subprocess_env guards against).
vpy() { env -u PYTHONPATH -u PYTHONHOME "$VENV/bin/python" "$@"; }

find_uv() {
  local uv
  uv="$(command -v uv || true)"
  [[ -z "$uv" && -x "$HOME/.local/bin/uv" ]] && uv="$HOME/.local/bin/uv"
  if [[ -z "$uv" ]]; then
    echo "ERROR: uv not found; install uv to provision the harness venv." >&2
    exit 1
  fi
  echo "$uv"
}

ensure_checkout() {
  if [[ ! -d "$CHECKOUT/.git" ]]; then
    echo "[omnidoc] cloning harness into $CHECKOUT"
    rm -rf "$CHECKOUT"
    mkdir -p "$CACHE"
    git clone --filter=blob:none -q "$HARNESS_REPO" "$CHECKOUT"
    refresh=1
  fi
  # Always re-pin: a stale checkout left on another revision would silently
  # score with different metric code than the lock was generated against.
  if [[ "$(git -C "$CHECKOUT" rev-parse HEAD)" != "$PIN" ]]; then
    echo "[omnidoc] checking out pinned $PIN"
    git -C "$CHECKOUT" fetch -q --filter=blob:none origin "$PIN" 2>/dev/null || true
    git -C "$CHECKOUT" checkout -q "$PIN"
    refresh=1
  fi
}

ensure_venv() {
  if [[ "$refresh" == 1 || ! -x "$VENV/bin/python" ]]; then
    local uv
    uv="$(find_uv)"
    echo "[omnidoc] building harness venv at $VENV"
    # The harness requires Python < 3.12.
    "$uv" venv --python 3.11 "$VENV"
    "$uv" pip install --python "$VENV/bin/python" \
      "$CHECKOUT" huggingface_hub openai -c "$CONSTRAINTS"
  else
    echo "[omnidoc] reusing harness venv at $VENV"
  fi
}

# The display-formula CDM metric renders LaTeX; without it the harness leaves
# overall_notebook null and there is no score at all, so this is not optional
# for a scoring run.
ensure_latex_toolchain() {
  local missing=()
  command -v pdflatex >/dev/null 2>&1 || missing+=(texlive texlive-latex-extra)
  command -v gs >/dev/null 2>&1 || missing+=(ghostscript)
  if ! command -v magick >/dev/null 2>&1 && ! command -v convert >/dev/null 2>&1; then
    missing+=(imagemagick)
  fi

  if [[ ${#missing[@]} -gt 0 ]]; then
    local apt_cmd="sudo apt-get install -y --no-install-recommends ${missing[*]}"
    if sudo -n true 2>/dev/null; then
      echo "[omnidoc] installing scoring dependencies: ${missing[*]}"
      sudo apt-get update -qq
      sudo apt-get install -y --no-install-recommends "${missing[@]}"
    else
      echo "::warning::OmniDocBench scoring needs ${missing[*]}, and sudo is not available non-interactively." >&2
      echo "  Install them with: $apt_cmd" >&2
      echo "  Without them the display-formula CDM metric fails and no overall score is produced." >&2
      echo "  Use --infer-only to generate predictions now and score elsewhere." >&2
      return 0
    fi
  fi

  # ImageMagick 6 on Ubuntu ships `convert`; the harness calls IM7's `magick`.
  if ! command -v magick >/dev/null 2>&1 && command -v convert >/dev/null 2>&1; then
    if sudo -n true 2>/dev/null; then
      sudo ln -sf "$(command -v convert)" /usr/bin/magick
    fi
  fi

  relax_imagemagick_pdf_policy
}

# Ubuntu's ImageMagick 6 ships a Ghostscript-CVE mitigation that denies the
# PDF/PS/EPS coders. CDM renders each formula to PDF and rasterizes it with
# `convert`, so under the stock policy every sample fails ("not allowed by the
# security policy"), CDM scores 0, and that nulls the overall score. Granting
# these coders back is what makes a local run match CI, where CDM does work.
relax_imagemagick_pdf_policy() {
  local policy=/etc/ImageMagick-6/policy.xml
  [[ -f "$policy" ]] || return 0
  grep -qE 'rights="none" pattern="(PS|PS2|PS3|EPS|PDF|XPS)"' "$policy" || return 0

  if sudo -n true 2>/dev/null; then
    echo "[omnidoc] granting ImageMagick the PDF coder rights CDM needs"
    sudo sed -i -E \
      's#rights="none" pattern="(PS|PS2|PS3|EPS|PDF|XPS)"#rights="read|write" pattern="\1"#g' \
      "$policy"
  else
    echo "::warning::ImageMagick denies the PDF coder, so the CDM metric will score 0 and no overall score will be produced." >&2
    # printf, not echo: the replacement's backreference must survive verbatim.
    printf '  Fix with: sudo sed -i -E %s %s\n' \
      "'s#rights=\"none\" pattern=\"(PS|PS2|PS3|EPS|PDF|XPS)\"#rights=\"read|write\" pattern=\"\1\"#g'" \
      "$policy" >&2
  fi
}

resolve_model() {
  [[ -n "$model" ]] && return 0
  local auth=()
  [[ -n "$api_key" ]] && auth=(-H "Authorization: Bearer $api_key")
  model="$(curl -sf -m 15 "${auth[@]}" "${url}/v1/models" 2>/dev/null \
    | vpy -c 'import json,sys; print(json.load(sys.stdin).get("data",[{}])[0].get("id",""))' \
      2>/dev/null || true)"
  if [[ -z "$model" ]]; then
    model="MiniMaxAI/MiniMax-M3-MXFP8"
    echo "[omnidoc] could not auto-detect the served model; defaulting to $model"
  else
    echo "[omnidoc] auto-detected served model: $model"
  fi
}

# Point upstream's inference script at our endpoint. Restored from git first so
# a second run with a different --url/--model can't double-patch a patched file.
patch_infer_script() {
  local script="$CHECKOUT/tools/model_infer/gpt_4o_inf.py"
  git -C "$CHECKOUT" checkout -- tools/model_infer/gpt_4o_inf.py

  local seed_line=""
  [[ -n "$seed" ]] && seed_line="\\n            seed=${seed},"

  # The key is read from the environment rather than substituted in, so a
  # real credential never lands in the cached checkout. `or` rather than a
  # get() default: unauthenticated runs export the variable empty, and an
  # empty string is a present-but-invalid key that makes the OpenAI client
  # raise "Missing credentials" instead of falling back.
  sed -i \
    -e 's|api_key= "API_KEY"|api_key=os.environ.get("OMNIDOC_API_KEY") or "dummy"|' \
    -e "s|base_url=\"API_URL\"|base_url=\"${url}/v1\"|" \
    -e "s|model=\"gpt-4o\"|model=\"${model}\"|" \
    -e "s|# max_tokens=32000,|max_tokens=${max_tokens},|" \
    -e "s|# temperature=0.0 # OCR.*|temperature=${temperature},\\n            top_p=${top_p},${seed_line}|" \
    "$script"

  if grep -q '"API_URL"\|"API_KEY"' "$script"; then
    echo "ERROR: failed to patch $script — upstream's placeholders changed." >&2
    echo "  The pinned SHA ($PIN) may need updating along with the sed rules." >&2
    exit 1
  fi
}

run_inference() {
  echo "[omnidoc] running inference on $(basename "$RUN_IMAGES") with $workers workers"
  OMNIDOC_API_KEY="$api_key" vpy "$CHECKOUT/tools/model_infer/gpt_4o_inf.py" \
    --image_root "$RUN_IMAGES" \
    --save_root "$PREDS" \
    --threads "$workers"

  # gpt_4o_inf.py swallows request failures and writes an empty .md, which
  # scores 0. Retry those pages at low concurrency before scoring.
  local round empty
  for ((round = 1; round <= retry_rounds; round++)); do
    empty="$(vpy "$PREP" find-empty \
      --pred-dir "$PREDS" --link-dir "$RUN_IMAGES" --retry-dir "$RETRY_IMAGES")"
    echo "[omnidoc] retry round $round: $empty pages with empty predictions"
    [[ "$empty" -eq 0 ]] && break
    sleep 30 # let an overloaded server recover
    OMNIDOC_API_KEY="$api_key" vpy "$CHECKOUT/tools/model_infer/gpt_4o_inf.py" \
      --image_root "$RETRY_IMAGES" \
      --save_root "$PREDS" \
      --threads 4
  done
}

write_eval_config() {
  cat > "$out_dir/eval-config.yaml" << YAML
end2end_eval:
  metrics:
    text_block:
      metric:
      - Edit_dist
    display_formula:
      metric:
      - Edit_dist
      - CDM
      cdm_workers: 9
    table:
      metric:
      - TEDS
      - Edit_dist
      teds_workers: 9
    reading_order:
      metric:
      - Edit_dist
  dataset:
    dataset_name: end2end_dataset
    ground_truth:
      data_path: $GT_JSON
    prediction:
      data_path: $PREDS
    match_method: quick_match
    match_workers: 9
    quick_match_truncated_timeout_sec: 300
    match_timeout_sec: 420
    timeout_fallback_max_chunk_span: 10
    timeout_fallback_order_penalty: 0.10
YAML
}

run_scoring() {
  write_eval_config
  echo "[omnidoc] scoring"
  # ./result is written relative to the harness's working directory.
  (cd "$CHECKOUT" && env -u PYTHONPATH -u PYTHONHOME "$VENV/bin/python" \
    pdf_validation.py --config "$out_dir/eval-config.yaml") \
    2>&1 | tee "$out_dir/scoring.txt"

  [[ -f "$SUMMARY" ]] && cp "$SUMMARY" "$out_dir/run_summary.json"
  local partial=()
  [[ -n "$limit" ]] && partial=(--partial)
  vpy "$PREP" score \
    --summary "$SUMMARY" \
    --out "$out_dir/score.json" \
    --total "$(vpy -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$GT_JSON")" \
    "${partial[@]}"
}

mkdir -p "$out_dir" "$PREDS"

ensure_checkout
ensure_venv

if [[ "$score_only" == 0 ]]; then
  # Always run the download: snapshot_download is incremental and cheap once
  # complete, and it verifies every ground-truth image is present. A sentinel
  # check on OmniDocBench.json would pass on a partial download, because that
  # one small file lands long before the thousands of images do.
  echo "[omnidoc] syncing dataset into $DATASET"
  vpy "$PREP" download --local-dir "$DATASET"

  vpy "$PREP" prepare \
    --image-dir "$DATASET/images" \
    --gt "$DATASET/OmniDocBench.json" \
    --link-dir "$RUN_IMAGES" \
    --gt-out "$GT_JSON" \
    ${limit:+--limit "$limit"}

  resolve_model
  patch_infer_script
  run_inference
fi

if [[ "$infer_only" == 1 ]]; then
  echo "[omnidoc] predictions in $PREDS (scoring skipped)"
  exit 0
fi

if [[ ! -f "$GT_JSON" ]]; then
  echo "ERROR: $GT_JSON not found — run without --score-only first." >&2
  exit 1
fi

ensure_latex_toolchain
run_scoring
echo "[omnidoc] results in $out_dir"
