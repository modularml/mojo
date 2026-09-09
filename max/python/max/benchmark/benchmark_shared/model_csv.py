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

"""Shared benchmark-result to CSV core used across the benchmarking stack.

Every benchmark surface projects the same underlying result data into a CSV,
and this module is the single home for that projection so the serving sweep,
the ``results_to_csv`` post-processing CLI, and the ``results_publication``
reporter (which backs the engine and unified serving benchmarks) share one
implementation instead of three.

Two projection styles live here because the surfaces consume different inputs,
and each must keep its exact column layout:

- *Data-driven* (:func:`flatten`, :func:`load_result_rows`, :func:`select_columns`,
  :class:`CsvStreamWriter`, :func:`convert`): flattens an already-serialized
  result JSON blob into a flat ``dotted.key`` namespace, with a curated
  :class:`ColumnProfile` summary plus opt-in groups. Used by the serving sweep
  (which streams ``results.csv`` from per-iteration JSON blobs) and by the
  ``results_to_csv`` CLI. Lists and non-string scalars become a single
  JSON-encoded cell.
- *Type-driven* (:func:`columns_for_type`, :func:`flatten_model`, and the
  hierarchy-key helpers): derives a stable column header from a pydantic /
  dataclass *type* and projects each instance's ``model_dump(mode="json")``
  onto it. Unlike the data-driven flattener it recurses only into fields whose
  *type* is structured, so a plain ``dict`` / ``list`` field stays one opaque
  JSON cell (a stable schema regardless of a row's runtime contents). Used by
  the ``results_publication`` streaming reporter.

Neither style recomputes metrics; both only project columns already present in
the data.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import logging
import types
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    NamedTuple,
    Protocol,
    TextIO,
    TypeGuard,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel
from typing_extensions import TypedDict, TypeIs

if TYPE_CHECKING:
    from _csv import Writer as _CsvWriter

logger = logging.getLogger(__name__)

# A JSON value as produced by ``json.load``. Recursive: objects nest objects
# and lists. Using this instead of a bare ``object`` lets the flattener's
# branches typecheck (``json.dumps`` only ever sees a non-str scalar or list).
JSONValue = (
    str
    | int
    | float
    | bool
    | None
    | list["JSONValue"]
    | Mapping[str, "JSONValue"]
)
# A JSON object: the top level of a single result blob.
JSONObject = Mapping[str, JSONValue]
# One flattened output row: a string cell per (flattened) column name.
Row = dict[str, str]


class ResultEntry(TypedDict, total=False):
    """One ``results`` entry of a ``ResultSetInContext`` document."""

    iteration_config: JSONObject
    result: JSONValue


class ResultSetDocument(TypedDict, total=False):
    """The ``{run_context, results}`` envelope emitted by the JSON reporter."""

    run_context: JSONObject
    results: list[ResultEntry]


class ColumnProfile(NamedTuple):
    """A domain-specific column-selection policy.

    Bundles the curated default ``summary`` columns (in display order, emitted
    only when present in the input) with the named opt-in ``groups``, each a
    predicate matching a flattened column name.
    """

    summary: tuple[str, ...]
    groups: Mapping[str, Callable[[str], bool]]


# ---------------------------------------------------------------------------
# Serving profile (default) — save_result_json / serving-sweep column layout.
# ---------------------------------------------------------------------------

# Curated, ordered default columns. Only those actually present in the input
# are emitted, so this same list works for text-generation and pixel-generation
# runs (missing keys are silently skipped).
_SERVING_SUMMARY_COLUMNS: tuple[str, ...] = (
    # Run identity / configuration.
    "date",
    "model_id",
    "backend",
    "benchmark_task",
    "dataset_name",
    "num_prompts",
    "max_concurrency",
    "max_concurrent_conversations",
    "request_rate",
    # Top-line throughput / duration.
    "duration",
    "completed",
    "failures",
    "request_throughput",
    "total_input_tokens",
    "total_output_tokens",
    "total_generated_outputs",
    "aggregate_tokens_per_minute",
    "mean_input_throughput",
    "mean_output_throughput",
    # Latency headlines (mean + median + p99 for the metrics people quote).
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_step_tpot_ms",
    "median_step_tpot_ms",
    "p99_step_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
    "mean_latency_ms",
    "median_latency_ms",
    "p99_latency_ms",
    # Cache + speculative-decode headlines.
    "global_cached_token_rate",
    "spec_decode_acceptance_rate",
    "spec_decode_acceptance_length",
)


def _group_prefill_decode(key: str) -> bool:
    return key.startswith(("prefill_stats.", "decode_stats.")) or (
        key.startswith("server_metrics")
        and ("prefill" in key or "decode" in key)
    )


def _group_gpu(key: str) -> bool:
    return key in (
        "peak_gpu_memory_mib",
        "available_gpu_memory_mib",
        "gpu_utilization",
    )


def _group_cpu(key: str) -> bool:
    return key.startswith("cpu_metrics")


def _group_per_turn(key: str) -> bool:
    return "per_turn" in key


def _group_server_metrics(key: str) -> bool:
    return key.startswith("server_metrics")


def _group_spec_decode(key: str) -> bool:
    return key.startswith("spec_decode")


def _group_confidence(key: str) -> bool:
    return (
        key.endswith(("_confidence", "_sample_size"))
        or "_ci_lower" in key
        or "_ci_upper" in key
        or "_ci_relative_width" in key
    )


def _group_client_args(key: str) -> bool:
    return key.startswith("client_args")


def _group_steady_state(key: str) -> bool:
    return key.startswith("steady_state") or key == "num_outliers_rejected"


def _group_raw(key: str) -> bool:
    # Per-request sample arrays and other bulk lists. Kept out of the summary
    # because each is a single opaque JSON cell that bloats the CSV.
    return key in (
        "input_lens",
        "output_lens",
        "ttfts",
        "latencies",
        "num_generated_outputs",
        "errors",
        "request_submit_times",
        "request_complete_times",
        "per_turn_cached_token_rates",
        "per_turn_cache_retentions",
        "request_records",
        "session_server_stats",
        "aggregate_server_stats",
    )


# Named opt-in column groups. Each predicate matches a *flattened* column name.
# Keyed by ``str`` (not a ``Literal``) so the dict fits ``ColumnProfile.groups``:
# ``Mapping`` is invariant in its key, so a ``Literal``-keyed map is not
# assignable to the shared ``Mapping[str, ...]`` field.
_SERVING_COLUMN_GROUPS: Mapping[str, Callable[[str], bool]] = {
    "prefill_decode": _group_prefill_decode,
    "gpu": _group_gpu,
    "cpu": _group_cpu,
    "per_turn": _group_per_turn,
    "server_metrics": _group_server_metrics,
    "spec_decode": _group_spec_decode,
    "confidence": _group_confidence,
    "client_args": _group_client_args,
    "steady_state": _group_steady_state,
    "raw": _group_raw,
}

SERVING_PROFILE = ColumnProfile(
    summary=_SERVING_SUMMARY_COLUMNS, groups=_SERVING_COLUMN_GROUPS
)


# ---------------------------------------------------------------------------
# Engine profile — ResultSetInContext layout emitted by the engine benchmark
# (``iteration_config`` = IndividualShapeConfig, ``result`` =
# EngineBenchmarkMetrics), where nested pydantic fields flatten to dotted keys.
# ---------------------------------------------------------------------------

_ENGINE_SUMMARY_COLUMNS: tuple[str, ...] = (
    # Run identity / shape configuration.
    "run_context.git_commit",
    "iteration_config.batch_size",
    "iteration_config.context_len",
    "iteration_config.input_len",
    "iteration_config.output_len",
    # Throughput.
    "result.tokens_per_second",
    "result.input_tokens_per_second",
    "result.output_tokens_per_second",
    # Latency headlines.
    "result.total_latency_ms.mean",
    "result.total_latency_ms.p90",
    "result.total_latency_ms.p99",
    "result.prefill_latency_ms.mean",
    "result.decode_latency_ms.mean",
    # GPU headlines.
    "result.gpu.peak_memory_mib",
    "result.gpu.avg_utilization_percent",
    "result.gpu.kernel_time_ms",
    "result.gpu.device_count",
)


def _engine_group_gpu(key: str) -> bool:
    return key.startswith("result.gpu")


def _engine_group_phase(key: str) -> bool:
    return key.startswith(
        ("result.prefill_latency_ms", "result.decode_latency_ms")
    )


def _engine_group_latency(key: str) -> bool:
    return key.startswith("result.total_latency_ms")


def _engine_group_config(key: str) -> bool:
    return key.startswith(("iteration_config", "run_context"))


_ENGINE_COLUMN_GROUPS: Mapping[str, Callable[[str], bool]] = {
    "gpu": _engine_group_gpu,
    "phase": _engine_group_phase,
    "latency": _engine_group_latency,
    "config": _engine_group_config,
}

ENGINE_PROFILE = ColumnProfile(
    summary=_ENGINE_SUMMARY_COLUMNS, groups=_ENGINE_COLUMN_GROUPS
)


# Named profiles selectable on the CLI via ``--profile``.
PROFILES: Mapping[str, ColumnProfile] = {
    "serving": SERVING_PROFILE,
    "engine": ENGINE_PROFILE,
}


# ---------------------------------------------------------------------------
# Data-driven flattening (JSON blob -> flat dotted.key row).
# ---------------------------------------------------------------------------


def flatten(obj: JSONObject) -> Row:
    """Flattens a nested JSON object into a flat ``dotted.key -> cell`` mapping.

    Nested objects are expanded recursively with dot-separated keys. Lists and
    other non-string scalars are JSON-encoded into a single cell (matching the
    streaming CSV reporter's convention), ``None`` becomes an empty string, and
    strings pass through unchanged.

    Args:
        obj:
            The parsed JSON object (a single result blob) to flatten.

    Returns:
        A mapping from flattened column name to its string cell value.
    """
    flat: Row = {}

    def _walk(value: JSONValue, prefix: str) -> None:
        if isinstance(value, Mapping):
            if not value:
                flat[prefix] = "{}"
                return
            for k, v in value.items():
                child = f"{prefix}.{k}" if prefix else str(k)
                _walk(v, child)
        elif value is None:
            flat[prefix] = ""
        elif isinstance(value, str):
            flat[prefix] = value
        else:
            # Lists, ints, floats, bools: a single JSON-encoded cell. Lists are
            # variable-length per row and cannot become their own columns.
            flat[prefix] = json.dumps(value)

    _walk(obj, "")
    return flat


def _is_result_set_document(
    data: Mapping[str, object],
) -> TypeGuard[ResultSetDocument]:
    """Returns True when ``data`` is a ``ResultSetInContext`` envelope.

    Detected by the presence of a ``results`` list; ``run_context`` and each
    entry's fields are validated where they are read.
    """
    return isinstance(data.get("results"), list)


def _rows_from_result_set(document: ResultSetDocument) -> list[Row]:
    """Flattens a ``ResultSetInContext`` document into one row per result.

    Each row merges the shared ``run_context`` with the entry's
    ``iteration_config`` and ``result``.
    """
    run_context = document.get("run_context")
    rows: list[Row] = []
    for entry in document["results"]:
        record: dict[str, JSONValue] = {}
        if isinstance(run_context, Mapping):
            record["run_context"] = run_context
        iteration_config = entry.get("iteration_config")
        if isinstance(iteration_config, Mapping):
            record["iteration_config"] = iteration_config
        if "result" in entry:
            record["result"] = entry["result"]
        rows.append(flatten(record))
    return rows


def load_result_rows(path: Path) -> list[Row]:
    """Loads a result JSON file and returns one flattened row per result.

    Supports the shapes the benchmarks emit:

    - a single ``save_result_json`` blob (one row);
    - a JSON array of such blobs (one row each);
    - a ``ResultSetInContext`` document with a ``results`` list, where each
      entry is merged with the shared ``run_context`` and its
      ``iteration_config`` (one row each).

    Args:
        path:
            The path to the result JSON file.

    Returns:
        A list of flattened rows (each a ``column -> cell`` mapping).

    Raises:
        ValueError: If the top-level JSON is not an object or array.
    """
    with open(path) as f:
        data: object = json.load(f)

    if isinstance(data, list):
        return [flatten(item) for item in data]
    if not isinstance(data, Mapping):
        raise ValueError(
            f"{path}: expected a JSON object or array at the top level, "
            f"got {type(data).__name__}"
        )
    if _is_result_set_document(data):
        return _rows_from_result_set(data)
    return [flatten(data)]


def ordered_unique(items: Iterable[str]) -> list[str]:
    """Returns ``items`` with duplicates removed, preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def select_columns(
    available: Sequence[str],
    *,
    groups: Sequence[str] = (),
    columns: Sequence[str] = (),
    all_columns: bool = False,
    only: bool = False,
    profile: ColumnProfile = SERVING_PROFILE,
) -> list[str]:
    """Chooses the ordered CSV columns from those available in the input.

    The default (no options) is ``profile.summary``, restricted to columns
    actually present in the input. ``groups`` and ``columns`` add to that
    summary; ``only`` drops the summary so just the requested groups/columns are
    emitted; ``all_columns`` emits every available column (summary first, then
    the rest sorted).

    For example:

    .. code-block:: python

        available = [
            "date", "mean_ttft_ms", "prefill_stats.x", "gpu_utilization",
        ]
        # Default: the summary columns present in the input, in summary order.
        select_columns(available)
        # -> ["date", "mean_ttft_ms"]
        # Opt-in groups add to the summary (group columns in definition order):
        select_columns(available, groups=["prefill_decode", "gpu"])
        # -> ["date", "mean_ttft_ms", "prefill_stats.x", "gpu_utilization"]
        # An exact, ordered set with no summary:
        select_columns(available, only=True, columns=["mean_ttft_ms", "date"])
        # -> ["mean_ttft_ms", "date"]

    Args:
        available:
            All flattened column names present across the input rows.
        groups:
            Names of opt-in ``profile.groups`` to include.
        columns:
            Exact column names to include, in the given order. Included even if
            absent from ``available`` (they produce empty cells).
        all_columns:
            Emit every available column.
        only:
            Emit only the requested ``groups`` / ``columns`` (no summary).
        profile:
            The :class:`ColumnProfile` supplying the summary columns and named
            groups. Defaults to :data:`SERVING_PROFILE`.

    Returns:
        The ordered list of column names to write.

    Raises:
        ValueError: If a requested group name is unknown to ``profile``.
    """
    available_set = set(available)

    if all_columns:
        summary_present = [c for c in profile.summary if c in available_set]
        rest = sorted(c for c in available_set if c not in set(summary_present))
        return ordered_unique([*summary_present, *rest])

    unknown = [g for g in groups if g not in profile.groups]
    if unknown:
        raise ValueError(
            f"Unknown column group(s): {', '.join(sorted(unknown))}. "
            f"Available groups: {', '.join(sorted(profile.groups))}."
        )

    selected: list[str] = []
    if not only:
        selected.extend(c for c in profile.summary if c in available_set)

    # Iterate the group definitions (not the caller-supplied strings) so
    # requested groups are emitted in definition order.
    requested = set(groups)
    for name, predicate in profile.groups.items():
        if name in requested:
            selected.extend(sorted(c for c in available_set if predicate(c)))

    # Explicit columns keep the caller's order and are emitted even if absent.
    selected.extend(columns)

    return ordered_unique(selected)


class CsvStreamWriter:
    """Incrementally writes result rows to a CSV as they are produced.

    Unlike :func:`convert` (which buffers every input, takes the union of all
    columns, then writes once), this writer commits each row to disk as it
    arrives so a crash mid-run still leaves the rows written so far -- restoring
    the streaming behavior the serving sweep had before it moved to per-run JSON
    blobs.

    Every column found in the first row is emitted (summary columns first, then
    the rest sorted). The header is fixed from that *first* row because a CSV
    header cannot be revised once written, so keys appearing only in later rows
    are dropped; for the homogeneous per-iteration blobs a sweep emits the first
    row already carries the column set, and the authoritative superset can
    always be regenerated from the JSON blobs via :func:`convert` /
    ``--all``. Use as a context manager so the file is always closed::

        with CsvStreamWriter(path) as w:
            for json_path in produced_json_paths:
                w.write_result(json_path)
    """

    def __init__(self, output_file: Path) -> None:
        self._output_file = output_file
        self._file: TextIO | None = None
        self._writer: _CsvWriter | None = None
        self._header: list[str] | None = None
        self._row_count = 0

    def __enter__(self) -> CsvStreamWriter:
        self._file = open(self._output_file, "w", newline="")
        self._writer = csv.writer(self._file)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._file is not None:
            self._file.close()
        if self._header is not None:
            logger.info(
                "Streamed %d row(s) and %d column(s) to %s",
                self._row_count,
                len(self._header),
                self._output_file,
            )

    def write_result(self, path: Path) -> None:
        """Flattens a result JSON file and streams each of its rows to the CSV."""
        for row in load_result_rows(path):
            self.write_row(row)

    def write_row(self, row: Row) -> None:
        """Streams a single flattened row, fixing the header on the first call."""
        if self._writer is None or self._file is None:
            raise RuntimeError(
                "CsvStreamWriter must be used as a context manager"
            )
        if self._header is None:
            self._header = select_columns(list(row), all_columns=True)
            self._writer.writerow(self._header)
        self._writer.writerow([row.get(column, "") for column in self._header])
        # Flush per row so a crash mid-run leaves the completed rows on disk.
        self._file.flush()
        self._row_count += 1


def convert(
    inputs: Sequence[Path],
    output_file: Path,
    *,
    groups: Sequence[str] = (),
    columns: Sequence[str] = (),
    all_columns: bool = False,
    only: bool = False,
    profile: ColumnProfile = SERVING_PROFILE,
    dry_run: bool = False,
) -> list[str]:
    """Reads result JSON ``inputs`` and writes a CSV with selected columns.

    Args:
        inputs:
            The result JSON files to read (each may hold one or many results).
        output_file:
            The CSV path to write.
        groups:
            Names of opt-in ``profile.groups`` to include.
        columns:
            Exact column names to include, in order.
        all_columns:
            Emit every available column.
        only:
            Emit only the requested groups/columns (no summary).
        profile:
            The :class:`ColumnProfile` supplying the summary/groups. Defaults to
            :data:`SERVING_PROFILE`.
        dry_run:
            Compute the selected columns and report what would be written, but
            do not create ``output_file``.

    Returns:
        The ordered list of columns that were (or would be) written.

    Raises:
        ValueError: If no inputs are given or a group name is unknown.
    """
    if not inputs:
        raise ValueError("no input files given")

    rows: list[Row] = []
    for path in inputs:
        rows.extend(load_result_rows(path))

    # Union of keys across rows, preserving first-seen order for stable output.
    available = ordered_unique(key for row in rows for key in row)
    selected = select_columns(
        available,
        groups=groups,
        columns=columns,
        all_columns=all_columns,
        only=only,
        profile=profile,
    )
    if dry_run:
        logger.info(
            "[dry run] would write %d row(s) and %d column(s) to %s: %s",
            len(rows),
            len(selected),
            output_file,
            ", ".join(selected),
        )
        return selected
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(selected)
        for row in rows:
            writer.writerow([row.get(column, "") for column in selected])
    logger.info(
        "Wrote %d row(s) and %d column(s) to %s",
        len(rows),
        len(selected),
        output_file,
    )
    return selected


# ---------------------------------------------------------------------------
# Type-driven flattening (pydantic / dataclass model -> flat dotted.key row).
#
# Column structure is derived from the *type* so headers stay stable across
# rows, and a field whose type is a plain ``dict`` / ``list`` stays a single
# opaque JSON cell rather than being exploded by a row's runtime contents.
# ---------------------------------------------------------------------------

T = TypeVar("T")
ModelT = TypeVar("ModelT", bound="BaseModel | _DataclassInstance")


class _DataclassInstance(Protocol):
    """A structural ``Protocol`` for identifying dataclass instances."""

    __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]


def _is_structured_type(
    t: object,
) -> TypeIs[type[BaseModel | _DataclassInstance]]:
    """Returns ``True`` if ``t`` is a ``BaseModel`` subclass or a dataclass."""
    return isinstance(t, type) and (
        issubclass(t, BaseModel) or dataclasses.is_dataclass(t)
    )


def _unwrap_optional_structured_type(
    annotation: object,
) -> type[BaseModel | _DataclassInstance] | None:
    """Returns the inner structured type when ``annotation`` wraps one, or ``None``.

    A structured type is a ``BaseModel`` subclass or a dataclass.
    Handles both ``X | None`` (Python 3.10+ native union) and
    ``Optional[X]`` / ``Union[X, None]`` (typing module).

    Args:
        annotation:
            The type annotation to inspect.

    Returns:
        The unwrapped structured type, or ``None`` if the annotation is not a
        structured type or an optional wrapping of one.
    """
    if _is_structured_type(annotation):
        return annotation
    origin = get_origin(annotation)
    is_union = origin is Union or (
        hasattr(types, "UnionType") and isinstance(annotation, types.UnionType)
    )
    if is_union:
        non_none_args = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none_args) == 1 and _is_structured_type(non_none_args[0]):
            return non_none_args[0]
    return None


def _iter_fields(model_type: type[ModelT]) -> Iterable[tuple[str, object]]:
    """Yields ``(field_name, annotation)`` for each field of a ``BaseModel`` or dataclass."""
    if issubclass(model_type, BaseModel):
        for field_name, field_info in model_type.model_fields.items():
            yield field_name, field_info.annotation
    elif dataclasses.is_dataclass(model_type):
        try:
            hints: Mapping[str, object] = get_type_hints(model_type)
        except Exception:
            hints = {f.name: f.type for f in dataclasses.fields(model_type)}
        for f in dataclasses.fields(model_type):
            yield f.name, hints.get(f.name, f.type)
    else:
        raise TypeError(f"Expected BaseModel or dataclass, got {model_type}")


def _csv_mode_opaque(model_type: type[ModelT], field_name: str) -> bool:
    """Returns True when a BaseModel field opts out of structured CSV expansion.

    Set ``Field(json_schema_extra={"csv_mode": "opaque"})`` on presentation
    views that duplicate other structured fields (e.g. ``result_groups``) so
    they stay one JSON cell instead of exploding into duplicate columns.
    """
    if not isinstance(model_type, type) or not issubclass(
        model_type, BaseModel
    ):
        return False
    field_info = model_type.model_fields.get(field_name)
    if field_info is None:
        return False
    extra = field_info.json_schema_extra
    return isinstance(extra, dict) and extra.get("csv_mode") == "opaque"


def columns_for_type(
    model_type: type[BaseModel | _DataclassInstance],
) -> Iterable[str]:
    """Yields flattened column names for a ``BaseModel`` or dataclass type.

    Nested structured fields are recursively expanded with dot-separated names.
    """
    for field_name, annotation in _iter_fields(model_type):
        if _csv_mode_opaque(model_type, field_name):
            yield field_name
            continue
        nested_type = _unwrap_optional_structured_type(annotation)
        if nested_type is not None:
            for column in columns_for_type(nested_type):
                yield f"{field_name}.{column}"
        else:
            yield field_name


def flatten_model(model_type: type[ModelT], model: ModelT) -> dict[str, str]:
    """Flattens a ``BaseModel`` or dataclass instance into a dict of string values.

    Every value is sourced from the model's JSON serialization
    (``model_dump(mode="json")``) rather than from the live field objects, so
    the CSV is a flat projection of the same JSON/schema blob the JSON reporter
    emits — one serialization source of truth, no independent field walk. This
    also means fields whose runtime value is a non-JSON-native object (e.g. the
    percentile-metric containers) serialize via pydantic instead of tripping
    ``json.dumps``.
    """
    if isinstance(model, BaseModel):
        dumped = model.model_dump(mode="json")
    else:
        dumped = dataclasses.asdict(model)
    return _flatten_jsonable(model_type, dumped)


def _flatten_jsonable(
    model_type: type[ModelT], dumped: Mapping[str, Any]
) -> dict[str, str]:
    """Flattens a model's JSON-serialized ``dumped`` mapping into CSV cells.

    Column structure is still driven by the *type* (via
    :func:`_unwrap_optional_structured_type`) so headers stay stable across
    rows: structured nested fields recurse into dot-separated sub-columns,
    while everything else — including data ``dict``/``list`` fields — stays a
    single JSON-encoded cell. ``None`` values produce an empty string.
    """
    flattened: dict[str, str] = {}
    for field_name, annotation in _iter_fields(model_type):
        field_value = dumped.get(field_name)
        if _csv_mode_opaque(model_type, field_name):
            nested_type = None
        else:
            nested_type = _unwrap_optional_structured_type(annotation)
        if nested_type is not None:
            if field_value is None:
                for col in columns_for_type(nested_type):
                    flattened[f"{field_name}.{col}"] = ""
            else:
                assert isinstance(field_value, Mapping)
                sub_flattened = _flatten_jsonable(nested_type, field_value)
                for sub_field_name, sub_field_value in sub_flattened.items():
                    flattened[f"{field_name}.{sub_field_name}"] = (
                        sub_field_value
                    )
        elif field_value is None:
            flattened[field_name] = ""
        elif isinstance(field_value, str):
            flattened[field_name] = field_value
        else:
            flattened[field_name] = json.dumps(field_value)
    return flattened


def strip_leading_hierarchy_key(key: str) -> str:
    """Strips the leading hierarchy level from a dotted key."""
    return key.split(".", 1)[-1]


def build_hierarchy_key_map(original_keys: Iterable[str]) -> dict[str, str]:
    """Maps each dotted key to its display column name.

    The leading hierarchy level (``run_context`` / ``iteration_config`` /
    ``result``) is stripped for readability, but only when the stripped name is
    unique. When two source keys strip to the same name — e.g.
    ``iteration_config.max_concurrency`` and ``result.max_concurrency`` — both
    keep their full dotted path so the columns stay distinct instead of
    colliding (which previously raised in :func:`map_keys`).
    """
    keys = list(original_keys)
    stripped_counts: dict[str, int] = {}
    for key in keys:
        stripped = strip_leading_hierarchy_key(key)
        stripped_counts[stripped] = stripped_counts.get(stripped, 0) + 1

    key_map: dict[str, str] = {}
    for key in keys:
        stripped = strip_leading_hierarchy_key(key)
        key_map[key] = stripped if stripped_counts[stripped] == 1 else key
    return key_map


def map_keys(f: Callable[[str], str], mapping: Mapping[str, T]) -> dict[str, T]:
    """Applies a function to every key in a mapping and returns a new dict.

    Args:
        f:
            The function to apply to each key.
        mapping:
            The source mapping whose keys are transformed.

    Returns:
        A new dict with the same values as ``mapping`` but with keys
        transformed by ``f``.

    Raises:
        KeyError: If two different keys in ``mapping`` map to the same
            transformed key.
    """
    mapped: dict[str, T] = {}
    original_keys: dict[str, str] = {}
    for key, value in mapping.items():
        mapped_key = f(key)
        if mapped_key in mapped:
            raise KeyError(
                f"Collision: {mapped_key!r} is mapped to "
                f"by both {original_keys[mapped_key]!r} and {key!r}"
            )
        mapped[mapped_key] = value
        original_keys[mapped_key] = key
    return mapped
