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

"""Tests for the shared benchmark-result to CSV core.

Covers both projection styles hosted in
:mod:`max.benchmark.benchmark_shared.model_csv`: the data-driven JSON flattener
(used by the serving sweep and the ``results_to_csv`` CLI) and the type-driven
model flattener (used by the ``results_publication`` reporter).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

import pytest
from max.benchmark.benchmark_shared.model_csv import (
    ENGINE_PROFILE,
    CsvStreamWriter,
    JSONObject,
    build_hierarchy_key_map,
    columns_for_type,
    convert,
    flatten,
    flatten_model,
    load_result_rows,
    map_keys,
    select_columns,
    strip_leading_hierarchy_key,
)
from pydantic import BaseModel

# --- Data-driven flattening --------------------------------------------------


def test_flatten_nested_dicts_and_lists() -> None:
    obj: JSONObject = {
        "date": "2026-07-29",
        "request_throughput": 12.5,
        "server_metrics": {"prefill_batch_count": 3, "nested": {"a": 1}},
        "input_lens": [1, 2, 3],
        "max_concurrent_conversations": None,
        "steady_state_detected": True,
    }
    flat = flatten(obj)
    assert flat["date"] == "2026-07-29"
    # Non-string scalars are JSON-encoded.
    assert flat["request_throughput"] == "12.5"
    assert flat["server_metrics.prefill_batch_count"] == "3"
    assert flat["server_metrics.nested.a"] == "1"
    # Lists become a single JSON cell.
    assert flat["input_lens"] == "[1, 2, 3]"
    # None becomes an empty string.
    assert flat["max_concurrent_conversations"] == ""
    assert flat["steady_state_detected"] == "true"


def test_select_columns_default_summary_present_only() -> None:
    available = [
        "date",
        "model_id",
        "request_throughput",
        "mean_ttft_ms",
        "prefill_stats.maxserve_batch_input_tokens_mean",
        "some_unknown_column",
    ]
    selected = select_columns(available)
    # Summary columns present in the input, in summary order.
    assert selected == [
        "date",
        "model_id",
        "request_throughput",
        "mean_ttft_ms",
    ]


def test_select_columns_groups_and_columns_additive() -> None:
    available = [
        "date",
        "request_throughput",
        "prefill_stats.x",
        "decode_stats.y",
        "gpu_utilization",
        "peak_gpu_memory_mib",
        "custom_metric",
    ]
    selected = select_columns(
        available,
        groups=["prefill_decode", "gpu"],
        columns=["custom_metric"],
    )
    assert selected[:2] == ["date", "request_throughput"]
    assert "prefill_stats.x" in selected
    assert "decode_stats.y" in selected
    assert "gpu_utilization" in selected
    assert "peak_gpu_memory_mib" in selected
    # Explicit column is appended.
    assert selected[-1] == "custom_metric"


def test_select_columns_only_drops_summary() -> None:
    available = ["date", "request_throughput", "gpu_utilization"]
    selected = select_columns(available, groups=["gpu"], only=True)
    assert selected == ["gpu_utilization"]


def test_select_columns_only_with_explicit_columns() -> None:
    available = ["date", "request_throughput", "mean_ttft_ms"]
    selected = select_columns(
        available,
        columns=["mean_ttft_ms", "request_throughput"],
        only=True,
    )
    assert selected == ["mean_ttft_ms", "request_throughput"]


def test_select_columns_all() -> None:
    available = ["zeta", "date", "request_throughput"]
    selected = select_columns(available, all_columns=True)
    # Summary columns first (in summary order), then the rest sorted.
    assert selected == ["date", "request_throughput", "zeta"]


def test_select_columns_explicit_absent_is_kept() -> None:
    selected = select_columns(["date"], columns=["not_in_input"], only=True)
    assert selected == ["not_in_input"]


def test_select_columns_unknown_group_raises() -> None:
    with pytest.raises(ValueError, match="Unknown column group"):
        select_columns(["date"], groups=["nonsense"])


def test_load_result_rows_single_blob(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"date": "d", "request_throughput": 1.0}))
    rows = load_result_rows(path)
    assert rows == [{"date": "d", "request_throughput": "1.0"}]


def test_load_result_rows_array(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(json.dumps([{"a": 1}, {"a": 2}]))
    rows = load_result_rows(path)
    assert rows == [{"a": "1"}, {"a": "2"}]


def test_load_result_rows_result_set_in_context(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "run_context": {"git_commit": "abc"},
                "results": [
                    {
                        "iteration_config": {"iteration": 1},
                        "result": {"request_throughput": 5.0},
                    }
                ],
            }
        )
    )
    rows = load_result_rows(path)
    assert rows == [
        {
            "run_context.git_commit": "abc",
            "iteration_config.iteration": "1",
            "result.request_throughput": "5.0",
        }
    ]


def test_load_result_rows_bad_toplevel(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(json.dumps("just a string"))
    with pytest.raises(ValueError, match="expected a JSON object or array"):
        load_result_rows(path)


def _write_blob(path: Path, **fields: object) -> None:
    path.write_text(json.dumps(fields))


def test_convert_end_to_end_summary(tmp_path: Path) -> None:
    a = tmp_path / "results-1-median.json"
    b = tmp_path / "results-2-median.json"
    _write_blob(
        a,
        date="2026-07-29",
        model_id="m",
        max_concurrency=1,
        request_throughput=10.0,
        mean_ttft_ms=100.0,
        prefill_stats={"maxserve_batch_input_tokens_mean": 42.0},
        input_lens=[1, 2, 3],
    )
    _write_blob(
        b,
        date="2026-07-29",
        model_id="m",
        max_concurrency=2,
        request_throughput=18.0,
        mean_ttft_ms=140.0,
        prefill_stats={"maxserve_batch_input_tokens_mean": 84.0},
        input_lens=[4, 5],
    )
    out = tmp_path / "summary.csv"
    columns = convert([a, b], out)

    # Summary excludes prefill_stats.* and input_lens.
    assert "prefill_stats.maxserve_batch_input_tokens_mean" not in columns
    assert "input_lens" not in columns

    with open(out) as f:
        rows = list(csv.reader(f))
    assert rows[0] == columns
    assert len(rows) == 3
    mc_idx = columns.index("max_concurrency")
    assert [rows[1][mc_idx], rows[2][mc_idx]] == ["1", "2"]


def test_convert_with_group_adds_columns(tmp_path: Path) -> None:
    a = tmp_path / "results.json"
    _write_blob(
        a,
        max_concurrency=1,
        request_throughput=10.0,
        prefill_stats={"maxserve_batch_input_tokens_mean": 42.0},
    )
    out = tmp_path / "detailed.csv"
    columns = convert([a], out, groups=["prefill_decode"])
    assert "prefill_stats.maxserve_batch_input_tokens_mean" in columns


def test_convert_missing_cell_is_empty(tmp_path: Path) -> None:
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write_blob(a, max_concurrency=1, mean_ttft_ms=100.0)
    # b lacks mean_ttft_ms.
    _write_blob(b, max_concurrency=2)
    out = tmp_path / "out.csv"
    columns = convert([a, b], out)
    with open(out) as f:
        rows = list(csv.reader(f))
    ttft_idx = columns.index("mean_ttft_ms")
    assert rows[1][ttft_idx] == "100.0"
    assert rows[2][ttft_idx] == ""


def test_convert_no_inputs_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no input files"):
        convert([], tmp_path / "out.csv")


def test_convert_dry_run_does_not_write(tmp_path: Path) -> None:
    a = tmp_path / "results.json"
    _write_blob(a, max_concurrency=1, request_throughput=10.0)
    out = tmp_path / "out.csv"
    columns = convert([a], out, dry_run=True)
    # Selection is still computed and returned, but no file is created.
    assert "max_concurrency" in columns
    assert "request_throughput" in columns
    assert not out.exists()


# --- Engine profile (ResultSetInContext output from engine benchmarks) -------


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
                            "total_latency_ms": {
                                "mean": 10.0,
                                "std": 1.0,
                                "p50": 9.0,
                                "p90": 12.0,
                                "p95": 13.0,
                                "p99": 15.0,
                                "unit": "ms",
                                "confidence_info": None,
                            },
                            "prefill_latency_ms": {"mean": 3.0, "p90": 3.5},
                            "decode_latency_ms": {"mean": 7.0, "p90": 8.0},
                            "tokens_per_second": 500.0,
                            "input_tokens_per_second": 300.0,
                            "output_tokens_per_second": 200.0,
                            "gpu": {
                                "peak_memory_mib": 40000.0,
                                "avg_utilization_percent": 90.0,
                                "kernel_time_ms": 5.0,
                                "device_count": 1,
                                "per_gpu_metrics": None,
                            },
                        },
                    }
                ],
            }
        )
    )


def test_engine_profile_summary_selects_engine_columns() -> None:
    available = [
        "run_context.git_commit",
        "iteration_config.batch_size",
        "result.tokens_per_second",
        "result.total_latency_ms.mean",
        "result.total_latency_ms.std",  # not in summary
        "result.gpu.peak_memory_mib",
        "mean_ttft_ms",  # serving column, must NOT appear under engine profile
    ]
    selected = select_columns(available, profile=ENGINE_PROFILE)
    assert "iteration_config.batch_size" in selected
    assert "result.tokens_per_second" in selected
    assert "result.total_latency_ms.mean" in selected
    assert "result.gpu.peak_memory_mib" in selected
    # Serving-only and non-summary engine columns are excluded from summary.
    assert "mean_ttft_ms" not in selected
    assert "result.total_latency_ms.std" not in selected


def test_engine_profile_group_gpu() -> None:
    available = [
        "iteration_config.batch_size",
        "result.gpu.peak_memory_mib",
        "result.gpu.kernel_time_ms",
        "result.tokens_per_second",
    ]
    selected = select_columns(
        available, groups=["gpu"], only=True, profile=ENGINE_PROFILE
    )
    assert selected == [
        "result.gpu.kernel_time_ms",
        "result.gpu.peak_memory_mib",
    ]


def test_engine_profile_unknown_group_lists_engine_groups() -> None:
    with pytest.raises(ValueError, match="Available groups: config, gpu"):
        select_columns(["x"], groups=["prefill_decode"], profile=ENGINE_PROFILE)


def test_convert_engine_profile_end_to_end(tmp_path: Path) -> None:
    src = tmp_path / "engine-results.json"
    _write_engine_json(src)
    out = tmp_path / "engine.csv"
    columns = convert([src], out, profile=ENGINE_PROFILE)

    assert "iteration_config.batch_size" in columns
    assert "result.total_latency_ms.mean" in columns
    # Bulk / non-summary fields stay out of the default engine summary.
    assert "result.gpu.per_gpu_metrics" not in columns

    with open(out) as f:
        rows = list(csv.reader(f))
    assert rows[0] == columns
    assert len(rows) == 2
    bs_idx = columns.index("iteration_config.batch_size")
    assert rows[1][bs_idx] == "4"


# --- CsvStreamWriter (incremental / crash-resilient streaming) ---------------


def test_stream_writer_writes_header_and_rows(tmp_path: Path) -> None:
    out = tmp_path / "stream.csv"
    with CsvStreamWriter(out) as w:
        w.write_row({"a": "1", "b": "2"})
        w.write_row({"a": "3", "b": "4"})

    with open(out) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["a", "b"]
    assert rows[1] == ["1", "2"]
    assert rows[2] == ["3", "4"]


def test_stream_writer_header_fixed_by_first_row(tmp_path: Path) -> None:
    """Columns are fixed from the first row: later-only keys are dropped and
    columns absent from a later row are written empty."""
    out = tmp_path / "stream.csv"
    with CsvStreamWriter(out) as w:
        w.write_row({"a": "1", "b": "2"})
        # "c" is new (dropped); "b" is missing (empty cell).
        w.write_row({"a": "3", "c": "9"})

    with open(out) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["a", "b"]
    assert "c" not in rows[0]
    assert rows[1] == ["1", "2"]
    assert rows[2] == ["3", ""]


def test_stream_writer_is_crash_resilient(tmp_path: Path) -> None:
    """Rows written before an exception survive on disk (flushed per row)."""
    out = tmp_path / "stream.csv"
    with pytest.raises(RuntimeError):
        with CsvStreamWriter(out) as w:
            w.write_row({"a": "1", "b": "2"})
            raise RuntimeError("simulated mid-run crash")

    with open(out) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["a", "b"]
    assert rows[1] == ["1", "2"]


def test_stream_writer_write_result_from_json(tmp_path: Path) -> None:
    src = tmp_path / "results-1-median.json"
    src.write_text(
        json.dumps({"model_id": "m", "max_concurrency": 1, "nested": {"x": 5}})
    )
    out = tmp_path / "stream.csv"
    with CsvStreamWriter(out) as w:
        w.write_result(src)

    with open(out) as f:
        rows = list(csv.reader(f))
    header = rows[0]
    assert "model_id" in header
    assert "nested.x" in header
    assert rows[1][header.index("nested.x")] == "5"


def test_stream_writer_no_rows_leaves_empty_file(tmp_path: Path) -> None:
    out = tmp_path / "stream.csv"
    with CsvStreamWriter(out):
        pass
    assert out.exists()
    assert out.read_text() == ""


def test_stream_writer_requires_context_manager(tmp_path: Path) -> None:
    w = CsvStreamWriter(tmp_path / "stream.csv")
    with pytest.raises(RuntimeError, match="context manager"):
        w.write_row({"a": "1"})


# --- Type-driven flattening (pydantic / dataclass model -> CSV) --------------


class _Metrics(BaseModel):
    mean: float
    std: float


@dataclass
class _Extra:
    value: int


T = TypeVar("T")


class _Result(BaseModel, Generic[T]):
    value: int
    # Nested model -> "metrics.mean", "metrics.std".
    metrics: T | None
    # Non-structured list field -> single JSON cell.
    samples: list[float]
    # Nested dataclass -> "extra.value".
    extra: _Extra | None = None


def test_columns_for_type_expands_nested_structured_fields() -> None:
    columns = list(columns_for_type(_Result[_Metrics]))
    assert columns == [
        "value",
        "metrics.mean",
        "metrics.std",
        "samples",
        "extra.value",
    ]


def test_flatten_model_projects_instance_onto_type_columns() -> None:
    row = flatten_model(
        _Result[_Metrics],
        _Result(
            value=10,
            metrics=_Metrics(mean=1.5, std=0.5),
            samples=[1.0, 2.0, 3.0],
            extra=_Extra(value=7),
        ),
    )
    assert row == {
        "value": "10",
        "metrics.mean": "1.5",
        "metrics.std": "0.5",
        # List stays a single JSON cell rather than exploding into columns.
        "samples": "[1.0, 2.0, 3.0]",
        "extra.value": "7",
    }


def test_flatten_model_none_nested_yields_stable_empty_subcolumns() -> None:
    row = flatten_model(
        _Result[_Metrics],
        _Result(value=20, metrics=None, samples=[4.0]),
    )
    # A None nested model still yields its sub-columns (as empty cells) so the
    # header stays stable across rows.
    assert row["metrics.mean"] == ""
    assert row["metrics.std"] == ""
    assert row["extra.value"] == ""


def test_build_hierarchy_key_map_strips_unique_and_keeps_collisions() -> None:
    key_map = build_hierarchy_key_map(
        [
            "run_context.git_commit",
            "iteration_config.max_concurrency",
            "result.max_concurrency",
        ]
    )
    # Unique after stripping -> stripped.
    assert key_map["run_context.git_commit"] == "git_commit"
    # Collision after stripping -> both keep their full dotted path.
    assert key_map["iteration_config.max_concurrency"] == (
        "iteration_config.max_concurrency"
    )
    assert key_map["result.max_concurrency"] == "result.max_concurrency"


def test_strip_leading_hierarchy_key() -> None:
    assert strip_leading_hierarchy_key("result.gpu.peak_memory_mib") == (
        "gpu.peak_memory_mib"
    )
    assert strip_leading_hierarchy_key("nohierarchy") == "nohierarchy"


def test_map_keys_raises_on_collision() -> None:
    with pytest.raises(KeyError, match="Collision"):
        map_keys(strip_leading_hierarchy_key, {"a.x": "1", "b.x": "2"})
