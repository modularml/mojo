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

import csv
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BenchMetric:
    code: int
    """Op-code of the Metric."""
    name: str
    """Metric's name."""
    unit: str
    """Metric's throughput rate unit (count/second)."""


@dataclass
class ThroughputMeasure:
    """Records a throughput metric of metric BenchMetric and value."""

    metric: BenchMetric
    """Type of throughput metric."""
    value: int
    """Measured count of throughput metric."""

    def compute(self, elapsed_sec: float) -> float:
        """Computes throughput rate for this metric per unit of time (second).

        Args:
            elapsed_sec: Elapsed time measured in seconds.

        Returns:
            The throughput values as a floating point 64.
        """
        # TODO: do we need support other units of time (ms, ns)?
        if elapsed_sec == 0:
            return 0.0
        return (self.value * 1e-9) / elapsed_sec


@dataclass
class Bench:
    """Constructs a Benchmark object, used for running multiple benchmarks
    and comparing the results.

    Args:
        name: Name of benchmark entry (string).
        met: Measured execution time for the entry in seconds (float).
        iters: Number of iterations in the measurement (int).
        metric_list: List of ThroughputMeasure's.
    """

    name: str
    met: float
    iters: int

    metric_list: list[ThroughputMeasure] = field(default_factory=list)

    elements = BenchMetric(0, "throughput", "GElems/s")
    bytes = BenchMetric(1, "DataMovement", "GB/s")
    flops = BenchMetric(2, "Arithmetic", "GFLOPS/s")
    theoretical_flops = BenchMetric(3, "TheoreticalArithmetic", "GFLOPS/s")

    BENCH_LABEL = "name"
    MET_LABEL = "met (s)"
    ITERS_LABEL = "iters"

    def dump_report(self, output_path: Path | None = None) -> None:
        if self.met == 0:
            repro = " ".join(sys.argv)
            print(
                f"WARNING: '{self.name}' reported elapsed_sec=0; benchmark"
                f" likely failed. Reporting 0 throughput.\nRepro: {repro}",
                file=sys.stderr,
            )
        metrics = [self.BENCH_LABEL, self.MET_LABEL, self.ITERS_LABEL] + [
            f"{m.metric.name} ({m.metric.unit})" for m in self.metric_list
        ]
        vals = ['"' + self.name + '"', self.met, self.iters] + [
            f"{m.compute(self.met)}" for m in self.metric_list
        ]
        rows = (metrics, vals)

        if output_path:
            with open(output_path, "w") as f:
                w = csv.writer(f, delimiter=",", quotechar="'")
                w.writerows(rows)
        with sys.stdout as f:
            w = csv.writer(f, delimiter=",", quotechar="'")
            w.writerows(rows)


def arg_parse(handle: str, default: Any = None, short_handle: str = "") -> str:
    # TODO: add constraints on dtype of return value

    handle = handle.lstrip("-")
    short_handle = short_handle.lstrip("-")
    args = sys.argv
    for i in range(len(args)):
        if handle and args[i].startswith("--" + handle):
            if "=" in args[i]:
                name_val = args[i].split("=")
                return name_val[1]
            else:
                return args[i + 1]
        elif short_handle and args[i].startswith("-" + short_handle):
            return args[i + 1]
    return default


def check_mpirun() -> int:
    """
    Check environment to examine whether the benchmark is called via mpirun.
    If so, use pe_rank=OMPI_COMM_WORLD_RANK as a suffix for output file.

    Raises:
        If the operation fails.

    Returns:
        An integer representing pe rank (default=-1).
    """
    pe_rank = -1
    if (
        "OMPI_COMM_WORLD_RANK" in os.environ
        and "OMPI_COMM_WORLD_SIZE" in os.environ
    ):
        TORCHRUN_DEFAULT_PORT = "29500"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = TORCHRUN_DEFAULT_PORT
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        pe_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))
    return pe_rank
