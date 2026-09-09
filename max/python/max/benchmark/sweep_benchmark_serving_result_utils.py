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

"""Uploader protocol and percentile validation for serving sweep benchmarks.

Used by :mod:`max.benchmark.sweep_benchmark_serving`. The sweep's ``results.csv``
is streamed from the per-iteration JSON blobs by the shared
:class:`~max.benchmark.benchmark_shared.model_csv.CsvStreamWriter`; this module
carries only the upload protocol (so callers can plug in a concrete result
sink) and the supported-percentile validation.
"""

from __future__ import annotations

from typing import Protocol


class SweepUploader(Protocol):
    """An uploader for sweep benchmark result JSONs.

    The sweep is deliberately agnostic about *where* each result is ingested.
    Callers plug in a concrete implementation of this protocol; the sweep calls
    :meth:`upload` with the path of each per-iteration result JSON it writes.

    Implementations can do anything from writing to a local file, to pushing to
    cloud storage, to inserting into a database — the sweep only cares that
    :meth:`upload` exists and accepts a string path.
    """

    def upload(self, result_filename: str) -> None:
        """Ingests the benchmark result JSON at ``result_filename``.

        Args:
            result_filename: Path to a per-iteration result JSON file
                previously written by ``save_result_json``.  The
                implementation decides whether to read it, transform it,
                forward it to a service, or drop it entirely (e.g.
                dry-run mode).
        """
        ...


SUPPORTED_SWEEP_SERVING_PERCENTILES: frozenset[int] = frozenset(
    (50, 90, 95, 99)
)


def validate_sweep_serving_percentiles(percentiles: list[int]) -> None:
    """Raises ``ValueError`` if any percentile is not supported for sweep CSV output."""
    unsupported = set(percentiles) - SUPPORTED_SWEEP_SERVING_PERCENTILES
    if unsupported:
        raise ValueError(
            f"Unsupported percentiles: {sorted(unsupported)}. "
            "Only P50, P90, P95, and P99 are supported."
        )
