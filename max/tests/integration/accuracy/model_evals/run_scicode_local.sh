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
# Provisions the scicode/h5py harness venv + test_data.h5 (cached in /tmp), then
# execs scicode_eval with --test-python/--test-data injected. Extra args forward.
#   ./bazelw run //max/tests/integration/accuracy/model_evals:scicode_eval_local -- \
#     --base-url http://localhost:8000 --model MiniMaxAI/MiniMax-M3-MXFP8 \
#     --sample-size 2 --out-dir /tmp/scicode-results
# Env overrides: SCICODE_VENV, SCICODE_DATA, SCICODE_PIN.
set -euo pipefail

VENV="${SCICODE_VENV:-/tmp/scicode-venv}"
H5="${SCICODE_DATA:-/tmp/scicode-test-data.h5}"
PIN="${SCICODE_PIN:-e3158ea011d4235245a547460d3688d7ccbf9900}"

RUNFILES="${RUNFILES_DIR:-${BASH_SOURCE[0]}.runfiles}"
SCICODE_EVAL="${RUNFILES}/_main/max/tests/integration/accuracy/model_evals/scicode_eval"
if [[ ! -x "$SCICODE_EVAL" ]]; then
  echo "ERROR: scicode_eval not found in runfiles at: $SCICODE_EVAL" >&2
  exit 1
fi

UV="$(command -v uv || true)"
[[ -z "$UV" && -x "$HOME/.local/bin/uv" ]] && UV="$HOME/.local/bin/uv"

# Harness venv: git-pinned scicode + h5py.
if [[ ! -x "$VENV/bin/python" ]]; then
  [[ -z "$UV" ]] && { echo "ERROR: uv not found; install uv or set SCICODE_VENV to a prebuilt venv." >&2; exit 1; }
  echo "[scicode] building harness venv at $VENV"
  "$UV" venv "$VENV" --python 3.11
  # pyarrow/numpy pinned: see the matching comment in
  # minimaxM3ScicodeEval.yaml — an unpinned install can resolve a numpy/
  # pyarrow pair with mismatched compiled ABIs (pyarrow==14.0.2 needs
  # numpy<2), breaking every test subprocess before it runs.
  "$UV" pip install --python "$VENV/bin/python" \
    openai "datasets>=2.0" tqdm h5py requests "pyarrow==14.0.2" "numpy<2" \
    "scicode @ git+https://github.com/scicode-bench/SciCode.git@${PIN}"
else
  echo "[scicode] reusing harness venv at $VENV"
fi

# test_data.h5: S3 first, else Google Drive bootstrap.
if [[ ! -s "$H5" ]]; then
  echo "[scicode] fetching test_data.h5"
  S3_PATH="s3://modular-ci-prod-artifact-bucket/datasets/scicode/test_data.h5"
  if command -v aws >/dev/null 2>&1 && aws s3 cp "$S3_PATH" "$H5" 2>/dev/null; then
    echo "[scicode]   fetched from S3"
  else
    echo "[scicode]   S3 unavailable -- downloading from Google Drive (~GB, one-time)"
    "$VENV/bin/python" - "$H5" <<'PY'
import os, sys, requests
dst = sys.argv[1]
url = "https://drive.usercontent.google.com/download?id=17G_k65N_6yFFZ2O-jQH00Lh6iaw3z-AW&confirm=t"
r = requests.Session().get(url, stream=True); r.raise_for_status()
with open(dst, "wb") as f:
    for c in r.iter_content(65536):
        f.write(c)
print(f"[scicode]   downloaded {os.path.getsize(dst)/1e9:.2f} GB")
PY
  fi
else
  echo "[scicode] reusing test_data.h5 ($(du -h "$H5" | cut -f1))"
fi

echo "[scicode] running scicode_eval"
exec "$SCICODE_EVAL" \
  --test-python "$VENV/bin/python" \
  --test-data "$H5" \
  "$@"
