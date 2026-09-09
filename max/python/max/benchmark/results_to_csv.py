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

"""Post-process benchmark result JSON blobs into a CSV with selectable columns.

The benchmarks treat their JSON output as the single source of truth: a
``save_result_json`` blob (serving) or a ``ResultSetInContext`` document (the
``results_publication`` reporters, used by both serving sweeps and engine
benchmarks) carries *every* metric the run produced. A CSV, by contrast, is a
presentation of that data — most consumers care about a small "summary" slice,
while power users opt into complementary groups (prefill / decode batch stats,
per-turn cache rates, GPU stats, ...) or name exact columns.

This module is the command-line front end for that projection. The flattening,
column selection, and streaming/writing logic live in the shared
:mod:`max.benchmark.benchmark_shared.model_csv` core (which the serving sweep
and the ``results_publication`` reporter also build on); this file only parses
arguments and calls into it. It never recomputes metrics; it only projects the
columns already present in the JSON.

The curated summary and opt-in groups are supplied by a
:class:`~max.benchmark.benchmark_shared.model_csv.ColumnProfile`, selected with
``--profile`` (``serving`` by default, ``engine`` for engine benchmark output).
``--all`` / ``--columns`` remain profile-independent and work on any JSON.

Run it through Bazel. Under ``bazel run`` the working directory is the
workspace root, so input/output paths are resolved relative to the repo root
(pass absolute paths to target files elsewhere)::

    ./bazelw run //max/python/max/benchmark:results_to_csv -- \\
        results-1-median.json results-2-median.json -o summary.csv

    # engine benchmark output (ResultSetInContext JSON):
    ./bazelw run //max/python/max/benchmark:results_to_csv -- \\
        engine-results.json -o engine.csv --profile engine

    # add the prefill/decode batch stats and GPU columns:
    ./bazelw run //max/python/max/benchmark:results_to_csv -- \\
        sweep-*/results-*.json -o detailed.csv --groups prefill_decode,gpu

    # emit every column found in the JSON:
    ./bazelw run //max/python/max/benchmark:results_to_csv -- \\
        results.json -o full.csv --all

    # pick an exact set of columns (no summary):
    ./bazelw run //max/python/max/benchmark:results_to_csv -- \\
        results.json -o custom.csv \\
        --only --columns max_concurrency,mean_ttft_ms,request_throughput
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from max.benchmark.benchmark_shared.model_csv import (
    PROFILES,
    Row,
    convert,
    load_result_rows,
    ordered_unique,
)

logger = logging.getLogger(__name__)


def _expand_inputs(patterns: Sequence[str]) -> list[Path]:
    """Expands CLI input arguments, globbing any that contain wildcards."""
    paths: list[Path] = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            matches = sorted(glob.glob(pattern))
            if not matches:
                logger.warning("No files matched pattern: %s", pattern)
            paths.extend(Path(m) for m in matches)
        else:
            paths.append(Path(pattern))
    return paths


def _parse_csv_list(value: str | None) -> list[str]:
    """Splits a comma-separated CLI value into a list, trimming whitespace."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for JSON-to-CSV benchmark result post-processing."""
    # Under ``bazel run`` the process starts in the runfiles tree; hop back to
    # the workspace root so relative input/output paths (and globs) resolve
    # where the user invoked Bazel. Unset outside Bazel, so this is a no-op.
    if workspace := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(workspace)

    parser = argparse.ArgumentParser(
        prog="results-to-csv",
        description=(
            "Flatten benchmark result JSON blobs into a CSV with a curated "
            "summary column set plus opt-in groups and columns."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Column groups depend on --profile; run with --list-columns to see "
            "the columns available in a given input."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Result JSON file(s). Glob patterns are expanded.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="serving",
        help=(
            "Column profile supplying the default summary and opt-in groups "
            "(default: serving)."
        ),
    )
    parser.add_argument(
        "--groups",
        default="",
        help=(
            "Comma-separated opt-in column groups to add to the summary "
            "(e.g. prefill_decode,gpu)."
        ),
    )
    parser.add_argument(
        "--columns",
        default="",
        help="Comma-separated exact column names to add, in order.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_columns",
        help="Emit every column found in the input.",
    )
    parser.add_argument(
        "--only",
        action="store_true",
        help=(
            "Emit only the requested --groups / --columns (drop the default "
            "summary)."
        ),
    )
    parser.add_argument(
        "--list-columns",
        action="store_true",
        help="Print the columns available in the input and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Report the rows/columns that would be written without creating "
            "the output CSV."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    inputs = _expand_inputs(args.inputs)
    if not inputs:
        parser.error("no input files found")

    if args.list_columns:
        rows: list[Row] = []
        for path in inputs:
            rows.extend(load_result_rows(path))
        for column in ordered_unique(key for row in rows for key in row):
            print(column)
        return 0

    try:
        convert(
            inputs,
            args.output,
            groups=_parse_csv_list(args.groups),
            columns=_parse_csv_list(args.columns),
            all_columns=args.all_columns,
            only=args.only,
            profile=PROFILES[args.profile],
            dry_run=args.dry_run,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
