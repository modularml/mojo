# ===----------------------------------------------------------------------=== #
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
# ===----------------------------------------------------------------------=== #

"""Tests for the ``results_to_csv`` CLI front end.

The flattening / column-selection library that this CLI drives lives in
:mod:`max.benchmark.benchmark_shared.model_csv` and is covered by
``test_model_csv.py``; this file only exercises argument parsing and the
end-to-end ``main`` invocation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from max.benchmark.results_to_csv import main


def _write_blob(path: Path, **fields: object) -> None:
    path.write_text(json.dumps(fields))


def test_main_cli(tmp_path: Path) -> None:
    a = tmp_path / "results.json"
    _write_blob(
        a,
        max_concurrency=1,
        request_throughput=10.0,
        gpu_utilization=[50.0, 60.0],
    )
    out = tmp_path / "out.csv"
    rc = main([str(a), "-o", str(out), "--groups", "gpu"])
    assert rc == 0
    with open(out) as f:
        header = next(csv.reader(f))
    assert "gpu_utilization" in header
    assert "request_throughput" in header


def test_main_cli_dry_run(tmp_path: Path) -> None:
    a = tmp_path / "results.json"
    _write_blob(a, max_concurrency=1, request_throughput=10.0)
    out = tmp_path / "out.csv"
    rc = main([str(a), "-o", str(out), "--dry-run"])
    assert rc == 0
    assert not out.exists()


def _write_engine_json(path: Path) -> None:
    """Writes a minimal engine ResultSetInContext document."""
    path.write_text(
        json.dumps(
            {
                "run_context": {"config_id": None, "git_commit": "abc"},
                "results": [
                    {
                        "iteration_config": {
                            "batch_size": 4,
                            "context_len": 0,
                            "input_len": 128,
                            "output_len": 64,
                        },
                        "result": {
                            "tokens_per_second": 500.0,
                            "input_tokens_per_second": 300.0,
                            "output_tokens_per_second": 200.0,
                        },
                    }
                ],
            }
        )
    )


def test_main_cli_profile_engine(tmp_path: Path) -> None:
    src = tmp_path / "engine-results.json"
    _write_engine_json(src)
    out = tmp_path / "engine.csv"
    rc = main([str(src), "-o", str(out), "--profile", "engine"])
    assert rc == 0
    with open(out) as f:
        header = next(csv.reader(f))
    assert "result.tokens_per_second" in header
    assert "iteration_config.batch_size" in header
