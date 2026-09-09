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

"""Tool to replay recorded HTTP transactions."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
import httpx
from max.serve.recordreplay import jsonl, replay


@click.command
@click.argument(
    "recording_file",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, path_type=Path
    ),
)
@click.option(
    "--base-url",
    required=True,
    metavar="URL",
    help="Base URL to send requests to.",
)
@click.option(
    "--concurrency",
    type=click.IntRange(min=1),
    default=1,
    metavar="REQUESTS",
    help="Number of requests to send in parallel.",
)
@click.option(
    "--read-timeout",
    type=click.FloatRange(min=0),
    default=120,
    metavar="SECONDS",
    help="Maximum seconds to wait for a response",
)
def main(
    recording_file: Path,
    *,
    base_url: str,
    concurrency: int,
    read_timeout: float,
) -> None:
    """Replay HTTP transactions recorded from MAX Serve.

    MAX Serve can record HTTP transactions with the
    MAX_SERVE_TRANSACTION_RECORDING_FILE environment variable.  The resulting
    HTTP transaction recording can be replayed against an arbitrary endpoint
    with this command.

    This can be used to, for example, capture the requests of a workload not
    designed to use concurrent requests, and later replay these against a
    server with higher request concurrency for load testing.
    """

    if not recording_file.name.endswith(".rec.jsonl"):
        raise click.BadParameter(
            "Only JSONL recording files are currently supported."
        )

    recording = list(jsonl.read_jsonl_recording(recording_file))

    async def run_inner() -> replay.ProgressSnapshot:
        async with (
            httpx.AsyncClient(
                base_url=base_url,
                timeout=httpx.Timeout(5, read=read_timeout, pool=None),
            ) as client,
            replay.TerminalProgressNotifier() as progress_notifier,
        ):
            await replay.replay_recording(
                recording,
                concurrency=concurrency,
                client=client,
                notifier=progress_notifier,
            )
            return progress_notifier.current_progress

    final_progress = asyncio.run(run_inner())
    _report_metrics(final_progress)


def _report_metrics(progress: replay.ProgressSnapshot) -> None:
    rps = progress.completed_transactions / progress.elapsed_seconds
    print(f"Throughput: {rps} RPS")
    print(f"Average latency: {progress.latency_distribution.mean} seconds")
    print(f"Latency stdev: {progress.latency_distribution.stdev} seconds")


if __name__ == "__main__":
    main()
