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
# GPQA-Diamond at OpenRouter AutoExacto parity, against any OpenAI-compatible
# endpoint — a local MAX/mach serve, or an authenticated external one.
#
# OpenRouter gates provider endpoints on its AutoExacto benchmark and deranks
# statistical outliers, so passing it is an exit criterion for a model bring-up.
# Scoring the same dataset our own way is not enough: OpenRouter runs Groq's
# openbench (which wraps UK AISI's Inspect), and model_evals/gpqa_eval.py scores
# GPQA with MiniMax's vendor prompt instead, so the two numbers are not
# comparable. This runs *their* harness with *their* config, so the score can be
# read against the other providers on the model's OpenRouter page.
#
# openbench is a third-party harness with its own dependency tree, so — like
# run_omnidocbench_local.sh — this is a shell runner around a pinned venv rather
# than a single bazel target. The venv persists under .derived/cache, so repeat
# runs skip straight to inference.
#
# Usage:
#   run_exacto_gpqa_local.sh [options]
#
# Options:
#   --url URL          Endpoint base URL (default: http://localhost:8000). A
#                       trailing /v1 is optional.
#   --api-key KEY      API key for an authenticated endpoint. Defaults to
#                       $EXACTO_API_KEY, then $OPENAI_API_KEY, then "dummy".
#                       Passed by environment, never written to disk or logged.
#   --model MODEL      Model name for the API `model` field. Default:
#                       auto-detect via GET {url}/v1/models.
#   --epochs N         Repeated passes over the dataset (default: 10, which is
#                       openbench's own gpqa_diamond default and therefore
#                       OpenRouter's). Lower only for a smoke run.
#   --limit N          Evaluate only the first N of the 198 questions, for a
#                       smoke run. A limited run's denominator differs from the
#                       published one, so it is reported PARTIAL and never
#                       "passes".
#   --max-connections N  Concurrent requests (default: 32).
#   --temperature F    Override sampling temperature. Defaults to unset, which
#                       leaves openbench's own 0.5 in place — changing it breaks
#                       comparability with OpenRouter's numbers.
#   --reference-range LOW,HIGH
#                      Optional range published providers land in for THIS
#                       model, shown next to the score as context. Provider
#                       spread is per-model, so there is no default; get it
#                       from the model's OpenRouter page.
#   --reference-source S
#                      Where --reference-range came from, e.g. an OpenRouter
#                       model slug.
#   --serve-config S   Describes the deployment under test (recipe name,
#                       quantization, KV dtype, spec-decode mode). Free text
#                       or JSON, recorded verbatim in score.json, because a
#                       score you cannot attribute to a config is not
#                       evidence.
#   --out-dir DIR      Results directory (default: /tmp/exacto-gpqa-results).
#   --refresh          Rebuild the harness venv from scratch.
#   -h, --help         Show this help.
#
# Reproduce the parity run against a local server:
#
#   ./bazelw run //max/tests/integration/accuracy/model_evals:exacto_gpqa_local
#
# Quick smoke (4 questions, 1 epoch) while iterating:
#
#   ./bazelw run //max/tests/integration/accuracy/model_evals:exacto_gpqa_local -- \
#     --limit 4 --epochs 1
#
# Score an authenticated external deployment:
#
#   ./bazelw run //max/tests/integration/accuracy/model_evals:exacto_gpqa_local -- \
#     --url https://<endpoint> --api-key "$MY_KEY" --model zai-org/GLM-5.3-Flash
#
# Env overrides: EXACTO_CACHE, EXACTO_OPENBENCH_VERSION, EXACTO_API_KEY.

set -euo pipefail

# Pinned for reproducibility. The two constraints are not optional:
#   * mcp<2 — mcp 2.x renamed FastMCP, and openbench's livemcpbench import
#     chain dies on it, taking down `bench list` and `bench eval` with it.
#   * openai<3 — openai 3.x breaks Inspect 0.3.125's timeout handling with
#     `TypeError: unsupported operand type(s) for +: 'float' and 'Timeout'`,
#     which surfaces as `APIConnectionError: Connection error.` with zero
#     requests ever reaching the server. openbench declares `openai>=2.0.0`
#     unbounded, so an unpinned install picks the break up. Do not debug that
#     symptom as a base-URL or networking problem.
OPENBENCH_VERSION="${EXACTO_OPENBENCH_VERSION:-0.5.3}"
MCP_CONSTRAINT="mcp<2"
OPENAI_CONSTRAINT="openai<3"

# openbench's own gpqa_diamond defaults, restated so a drift in the upstream
# task is visible as a mismatch in score.json rather than a silent change.
EXPECTED_DATASET="nmayorga7/gpqa_diamond"
EXPECTED_TEMPERATURE="0.5"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${BUILD_WORKSPACE_DIRECTORY:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}"
CACHE="${EXACTO_CACHE:-$REPO_ROOT/.derived/cache}"
VENV="$CACHE/exacto-openbench-venv"

url="http://localhost:8000"
api_key="${EXACTO_API_KEY:-${OPENAI_API_KEY:-}}"
model=""
epochs=10
limit=""
max_connections=32
temperature=""
reference_range=""
reference_source=""
serve_config=""
out_dir="/tmp/exacto-gpqa-results"
refresh=0

usage() { sed -n '15,85p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) url="$2"; shift 2 ;;
    --api-key) api_key="$2"; shift 2 ;;
    --model) model="$2"; shift 2 ;;
    --epochs) epochs="$2"; shift 2 ;;
    --limit) limit="$2"; shift 2 ;;
    --max-connections) max_connections="$2"; shift 2 ;;
    --temperature) temperature="$2"; shift 2 ;;
    --reference-range) reference_range="$2"; shift 2 ;;
    --reference-source) reference_source="$2"; shift 2 ;;
    --serve-config) serve_config="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --refresh) refresh=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option: $1" >&2; echo "try --help" >&2; exit 1 ;;
  esac
done

# External endpoints are usually quoted with /v1 already attached, while the
# harness appends its own; normalize so either form works.
url="${url%/}"; url="${url%/v1}"; url="${url%/}"

# Under `bazel run` the parent leaks a bazel-runfiles PYTHONPATH; inheriting it
# into the harness venv would shadow that venv's own packages.
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

# Under `bazel run` the report module is in the runfiles tree; standalone it
# sits next to this script.
RUNFILES="${RUNFILES_DIR:-${BASH_SOURCE[0]}.runfiles}"
REPORT=""
for candidate in \
  "$RUNFILES/_main/max/tests/integration/accuracy/model_evals/exacto_report.py" \
  "$SCRIPT_DIR/exacto_report.py"; do
  if [[ -f "$candidate" ]]; then REPORT="$candidate"; break; fi
done
if [[ -z "$REPORT" ]]; then
  echo "ERROR: exacto_report.py not found in runfiles or next to this script" >&2
  exit 1
fi

ensure_venv() {
  local uv; uv="$(find_uv)"
  if [[ "$refresh" == 1 ]]; then rm -rf "$VENV"; fi
  if [[ ! -x "$VENV/bin/bench" ]]; then
    echo "[exacto] provisioning openbench $OPENBENCH_VERSION in $VENV"
    mkdir -p "$CACHE"
    "$uv" venv "$VENV" >/dev/null
    "$uv" pip install --python "$VENV/bin/python" \
      "openbench==$OPENBENCH_VERSION" "$MCP_CONSTRAINT" "$OPENAI_CONSTRAINT"
  fi
  # Guard the pins on every run: a cached venv someone else has pip-installed
  # into would otherwise fail later as a bogus connection error.
  vpy - <<'PYCHECK'
import sys
from importlib.metadata import version
bad = []
if int(version("openai").split(".")[0]) >= 3:
    bad.append(f'openai {version("openai")} (need <3)')
if int(version("mcp").split(".")[0]) >= 2:
    bad.append(f'mcp {version("mcp")} (need <2)')
if bad:
    sys.exit(
        "ERROR: harness venv has incompatible pins: "
        + ", ".join(bad)
        + "\n       re-run with --refresh"
    )
PYCHECK
}

detect_model() {
  if [[ -n "$model" ]]; then return; fi
  echo "[exacto] auto-detecting served model from $url/v1/models"
  local ids count
  ids="$(curl -sf -m 30 -H "Authorization: Bearer ${api_key:-dummy}" \
    "$url/v1/models" | vpy -c '
import json, sys
data = json.load(sys.stdin).get("data") or []
print("\n".join(str(m.get("id")) for m in data if m.get("id")))
' 2>/dev/null || true)"
  count="$(printf '%s' "$ids" | grep -c . || true)"
  if [[ "${count:-0}" -eq 0 ]]; then
    cat >&2 <<MSG
ERROR: could not auto-detect a model from $url/v1/models.

       The list came back empty, which has two usual causes: the credential
       is missing or rejected (the endpoint 401s and the list reads empty),
       or this is a mach engine behind the Mammoth orchestrator, which never
       populates the list. Either way, pass --model with the served name.
MSG
    exit 1
  fi
  if [[ "$count" -gt 1 ]]; then
    # A multi-model gateway lists everything it fronts, in no useful order.
    # Guessing the first entry once picked an image-generation model and the
    # whole run 404'd, so refuse and make the caller choose.
    {
      echo "ERROR: $url/v1/models lists $count models, so there is nothing to"
      echo "       auto-detect. Pass --model with one of:"
      printf '%s\n' "$ids" | sed 's/^/         /'
    } >&2
    exit 1
  fi
  # Copy the id verbatim: MAX serve matches the OpenAI `model` field by exact
  # string equality, so a reconstructed name 400s every request.
  model="$ids"
  echo "[exacto] model: $model"
}

# A score is only evidence if you can say what produced it. Quantization, KV
# cache dtype and speculative decoding all shift accuracy, and none of them
# appear in the harness log, so probe the endpoint for whatever it volunteers
# and let the caller state the rest with --serve-config.
capture_endpoint_meta() {
  local out="$1"
  local models_json model_json
  models_json="$(curl -sf -m 30 -H "Authorization: Bearer ${api_key:-dummy}" \
    "$url/v1/models" 2>/dev/null || echo '')"
  model_json="$(curl -sf -m 30 -H "Authorization: Bearer ${api_key:-dummy}" \
    "$url/v1/models/$model" 2>/dev/null || echo '')"
  vpy - "$out" "$url" "$model" "${serve_config:-}" \
    "$models_json" "$model_json" <<'PYMETA'
import json, os, subprocess, sys

out, url, model, serve_config, models_json, model_json = sys.argv[1:7]


def maybe_json(raw):
    try:
        return json.loads(raw) if raw.strip() else None
    except json.JSONDecodeError:
        return None


def git_head():
    # The client's commit, not the server's. MAX serve exposes no build id, so
    # a score against someone else's endpoint cannot record what built it.
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


meta = {
    "base_url": url,
    "served_model": model,
    "models_endpoint": maybe_json(models_json),
    "model_endpoint": maybe_json(model_json),
    # Free-text or JSON the caller supplies: the recipe, quantization, KV dtype,
    # spec-decode mode, whatever identifies the deployment under test.
    "serve_config": maybe_json(serve_config) or (serve_config or None),
    "client_commit": git_head(),
}
with open(out, "w") as f:
    json.dump(meta, f, indent=2)
PYMETA
}

# One request before provisioning a dataset and 198 questions' worth of work.
# Without this, a wrong model name or a dead endpoint only shows up as a run
# that scores nothing, after several minutes of setup.
preflight_endpoint() {
  local resp code payload
  resp="$(curl -s -m 120 -w '\n%{http_code}' \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${api_key:-dummy}" \
    -d "{\"model\": \"$model\", \"max_tokens\": 8, \"messages\": [{\"role\": \"user\", \"content\": \"Reply with OK.\"}]}" \
    "$url/v1/chat/completions" || true)"
  code="${resp##*$'\n'}"
  payload="${resp%$'\n'*}"
  if [[ "$code" != "200" ]]; then
    cat >&2 <<MSG
ERROR: the endpoint rejected a basic chat completion (HTTP ${code:-no response})
       for model '$model'.

       Nothing would be measured, so this stops before provisioning the
       harness. Check the URL, the exact model id and the credential.
       Response: $(printf '%s' "$payload" | tr -d '\n' | head -c 300)
MSG
    exit 1
  fi
  echo "[exacto] endpoint preflight: chat completion OK for '$model'"
}

ensure_venv
detect_model
preflight_endpoint

mkdir -p "$out_dir"
# openbench resolves --logfile under a `logs/` directory relative to its cwd,
# which the run below sets to $out_dir. Clear it first: reusing an --out-dir
# would otherwise leave a previous run's log here, and the score picked up
# below would silently be the stale one.
LOG_DIR="$out_dir/logs"
rm -rf "$LOG_DIR"

args=(
  eval gpqa_diamond
  --model "openai-api/local/$model"
  --epochs "$epochs"
  --max-connections "$max_connections"
  --log-format json
  --logfile run
)
[[ -n "$limit" ]] && args+=(--limit "$limit")
[[ -n "$temperature" ]] && args+=(--temperature "$temperature")

echo "[exacto] running gpqa_diamond: epochs=$epochs limit=${limit:-198(all)} temp=${temperature:-$EXPECTED_TEMPERATURE}"
(
  cd "$out_dir"
  # Inspect resolves the endpoint from <SERVICE>_BASE_URL / <SERVICE>_API_KEY,
  # where SERVICE is the middle segment of openai-api/<service>/<model>. It
  # strips that segment before sending, so the server receives the bare id.
  env -u PYTHONPATH -u PYTHONHOME \
    LOCAL_BASE_URL="$url/v1" \
    LOCAL_API_KEY="${api_key:-dummy}" \
    "$VENV/bin/bench" "${args[@]}"
)

# Newest json under the (freshly cleared) log dir, so the exact filename
# openbench chose does not matter.
LOG_JSON="$(find "$LOG_DIR" -name '*.json' -type f -printf '%T@ %p\n' 2>/dev/null \
  | sort -rn | head -1 | cut -d' ' -f2-)"
if [[ -z "$LOG_JSON" || ! -f "$LOG_JSON" ]]; then
  echo "ERROR: no openbench JSON log found under $LOG_DIR" >&2
  exit 1
fi
echo "[exacto] harness log: $LOG_JSON"

ENDPOINT_META="$out_dir/endpoint.json"
capture_endpoint_meta "$ENDPOINT_META"

REPORT_ARGS=()
[[ -n "$reference_range" ]] && REPORT_ARGS+=(--reference-range "$reference_range")
[[ -n "$reference_source" ]] && REPORT_ARGS+=(--reference-source "$reference_source")

vpy "$REPORT" \
  --harness openbench \
  --log "$LOG_JSON" \
  --dataset gpqa_diamond \
  --out-dir "$out_dir" \
  --endpoint-meta "$ENDPOINT_META" \
  "${REPORT_ARGS[@]}" \
  --metric-prefix EXACTO_GPQA

# Comparability guard: report, don't fail. A drifted upstream default still
# produces a usable number, it just is not the number OpenRouter would get.
vpy - "$out_dir/score.json" "$EXPECTED_DATASET" "$EXPECTED_TEMPERATURE" "$limit" <<'PYWARN'
import json, sys
score_path, want_dataset, want_temp, limit = sys.argv[1:5]
s = json.load(open(score_path))
if s.get("dataset_name") != want_dataset:
    print(f"::warning::dataset is {s.get('dataset_name')}, expected {want_dataset}: "
          "score is not comparable to OpenRouter's.")
if str(s.get("temperature")) != want_temp:
    print(f"::warning::temperature is {s.get('temperature')}, expected {want_temp}: "
          "score is not comparable to OpenRouter's.")
if limit:
    print(f"::warning::PARTIAL run (--limit {limit}): a subset of the 198 "
          "questions has a different denominator than the published score.")
PYWARN

echo "[exacto] results: $out_dir/score.json"
