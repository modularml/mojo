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

"""Run a suite of sweep serving benchmarks.

A thin orchestration layer on top of :func:`benchmark_serving.main_with_parsed_args`
that adds a log directory, a CSV writer, and an optional caller-supplied
:class:`SweepUploader` for per-iteration result-JSON upload.  The actual
range iteration, workload YAML loading, ``num_iters`` / median selection,
and prefix-cache flushing are all handled inside ``benchmark_serving``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from max.benchmark.benchmark_serving import (
    BenchmarkRunResult,
    save_result_json,
)
from max.benchmark.benchmark_serving import (
    main_with_parsed_args as benchmark_serving_main,
)
from max.benchmark.benchmark_serving import (
    parse_args as _parse_serving_args,
)
from max.benchmark.benchmark_shared.config import (
    ServingBenchmarkConfig,
)
from max.benchmark.benchmark_shared.model_csv import CsvStreamWriter
from max.benchmark.sweep_benchmark_serving_result_utils import (
    SweepUploader,
    validate_sweep_serving_percentiles,
)

DESCRIPTION = "Run a suite of sweep serving benchmarks."

logger = logging.getLogger("sweep-benchmark-serving")


def parse_args(
    args: Sequence[str] | None = None,
    *,
    app_name: str = "sweep-benchmark-serving",
    description: str = DESCRIPTION,
) -> ServingBenchmarkConfig:
    """Parse command line arguments into a ServingBenchmarkConfig."""
    return _parse_serving_args(
        args,
        app_name=app_name,
        description=description,
    )


def main(
    args: Sequence[str] | None = None,
    *,
    uploader: SweepUploader | None = None,
    app_name: str = "sweep-benchmark-serving",
    description: str = DESCRIPTION,
) -> None:
    """CLI entry point.

    Args:
        args: CLI argv (defaults to ``sys.argv[1:]``).
        uploader: Optional :class:`SweepUploader` invoked with the
            per-iteration result JSON path whenever
            ``--upload-results`` is set and the run is not a dry run.
            Intentionally not wired by default so this module has no
            hard dependency on any specific result-ingestion backend.
        app_name: Name shown in ``--help`` output.
        description: Description shown in ``--help`` output.
    """
    try:
        config = parse_args(args, app_name=app_name, description=description)
    except SystemExit as e:
        if e.code == 0:
            return
        raise

    if not config.model:
        raise SystemExit("error: the following arguments are required: --model")

    if config.upload_results and not config.workload_config:
        logger.warning(
            "--workload-config is not set while --upload-results is set. "
            "Run results will be recorded, but will not include any workload name, "
            "and may not be picked up by dashboards."
        )

    run_sweep(config, uploader=uploader)


def run_sweep(
    config: ServingBenchmarkConfig,
    *,
    uploader: SweepUploader | None = None,
    report_result: Callable[[BenchmarkRunResult], None] | None = None,
    server_liveness: Callable[[], bool] | None = None,
) -> list[BenchmarkRunResult]:
    """Set up CSV + upload infrastructure and delegate benchmarking to the library.

    The actual range iteration, workload YAML loading, num_iters / median
    selection, and prefix-cache flushing are all handled by
    :func:`benchmark_serving.main_with_parsed_args`.  This function provides
    the thin orchestration layer on top: log directory, CSV writer, and
    optional uploader invocation.

    Args:
        config: Parsed :class:`ServingBenchmarkConfig`.
        uploader: Optional :class:`SweepUploader` invoked with the
            per-iteration result JSON path.  Only consulted when
            ``config.upload_results`` is True and the run is not a dry
            run.
        report_result: Optional callback invoked once per sweep iteration as
            soon as that iteration's :class:`BenchmarkRunResult` is
            produced. Used by the unified ``benchmark_serving`` binary to
            stream rows into ``utils/benchmarking/results_publication``
            during the run rather than after — preserves "live progress"
            visibility and ensures partial results survive a mid-sweep
            crash.
        server_liveness: Optional predicate forwarded to
            :func:`benchmark_serving.main_with_parsed_args`. When the
            orchestrator launched the server it passes a process-liveness
            check so the server-ready wait aborts promptly on a crashed
            bring-up instead of polling until the timeout.

    Returns:
        The per-iteration :class:`BenchmarkRunResult` list produced by
        :func:`benchmark_serving.main_with_parsed_args`, so legacy callers
        that haven't migrated to ``report_result`` can still iterate the
        whole sweep at once.
    """
    if config.upload_results and config.cluster_information_path is None:
        logger.warning("Warning: uploading results without cluster information")

    # ---- Log directory ----
    # Skip auto-creating a sweep-serving-* directory under --dry-run when
    # the user didn't explicitly ask for one. Tests that pass --log-dir
    # still get a real on-disk CSV.
    if config.log_dir:
        log_dir = Path(config.log_dir)
        os.makedirs(log_dir, exist_ok=True)
        config.log_dir = str(log_dir)
        print(f"Saving logs to: {log_dir}")
    elif not config.dry_run:
        timestamp = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
        log_dir = Path(f"sweep-serving-{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        config.log_dir = str(log_dir)
        print(f"Saving logs to: {log_dir}")
    else:
        log_dir = None

    # ---- Percentiles ----
    percentiles = [
        int(x.strip()) for x in config.latency_percentiles.split(",")
    ]
    validate_sweep_serving_percentiles(percentiles)

    upload_active = uploader is not None and config.upload_results

    # ---- CSV output + upload ----
    # Stream per-iteration: ``benchmark_serving_main`` yields one result per
    # ``(max_concurrency, request_rate)`` step, and we drive CSV writes,
    # upload side effects, and the optional ``report_result`` callback as
    # each yields. Don't materialize the iterator here — batching at end
    # would defeat the streaming guarantees of
    # ``utils/benchmarking/results_publication`` (live BigQuery rows during
    # the run + partial results surviving a mid-sweep crash).
    results: list[BenchmarkRunResult] = []

    # No log directory configured (dry-run without --log-dir): skip CSV
    # and per-result JSON output but still drive the benchmarks.
    if log_dir is None:
        for result in benchmark_serving_main(
            config, server_liveness=server_liveness
        ):
            results.append(result)
            if report_result is not None:
                report_result(result)
        return results

    results_csv_path = log_dir / "results.csv"
    # results.csv streams one row per iteration (all columns) as that
    # iteration's JSON blob is written, so a crash mid-sweep still leaves the
    # completed rows on disk. The per-concurrency JSON blobs remain the source
    # of truth; the authoritative all-columns superset can be regenerated from
    # them via results_to_csv if a later iteration surfaces new columns.
    with CsvStreamWriter(results_csv_path) as csv_writer:
        for result in benchmark_serving_main(
            config, server_liveness=server_liveness
        ):
            results.append(result)
            # Save per-concurrency JSON with full metrics, then stream its row.
            if result.result is not None:
                assert config.model is not None
                json_path = (
                    log_dir / f"results-{result.max_concurrency}-median.json"
                )
                save_result_json(
                    str(json_path),
                    config,
                    result.result,
                    benchmark_task=config.benchmark_task,
                    model_id=config.model,
                    tokenizer_id=config.tokenizer or config.model,
                    request_rate=result.request_rate,
                    record_max_concurrency=result.max_concurrency,
                )
                csv_writer.write_result(json_path)
                if upload_active and uploader is not None:
                    uploader.upload(str(json_path))

            # Stream the row to the results-publication reporter, if the caller
            # wired one in — so a crash mid-sweep still leaves rows 1..N-1
            # published downstream.
            if report_result is not None:
                report_result(result)

    result_file_path = results_csv_path.resolve()
    logger.info(
        f"All concurrency sweep results have been written to: {result_file_path}"
    )
    return results


if __name__ == "__main__":
    main()
