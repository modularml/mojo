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


"""Metric-gathering utilities for the pipelines."""

import time
from typing import Any

import psutil


class TextGenerationMetrics:
    """Metrics capturing and reporting for a text generation pipeline."""

    prompt_size: int
    output_size: int
    startup_time: float | str
    time_to_first_token: float | str
    prompt_eval_throughput: float | str
    eval_throughput: float | str

    _start_time: float
    _signposts: dict[str, float]
    _mem_usage_marker: dict[str, float]
    _should_print_report: bool
    _process: psutil.Process
    _print_raw: bool

    def __init__(
        self, print_report: bool = False, print_raw: bool = False
    ) -> None:
        self._signposts = {}
        self._mem_usage_marker = {}
        self.batch_size = 1
        self.prompt_size = 0
        self.output_size = 0
        self._should_print_report = print_report
        self._start_time = time.time()
        self._process = psutil.Process()
        self._print_raw = print_raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._calculate_results()
        if self._should_print_report:
            self._print_report(self._print_raw)

    def signpost(self, name: str) -> None:
        """Measure the current time and memory usage, tagging it with a name for later reporting."""
        self._signposts[name] = time.time()
        self._mem_usage_marker[name] = (self._process.memory_info().rss) / (
            1024 * 1024 * 1024
        )

    def new_token(self) -> None:
        """Report that a new token has been generated."""
        self.new_tokens(1)

    def new_tokens(self, num_tokens: int) -> None:
        """Report that a num_tokens tokens have been generated."""
        self.output_size += num_tokens

    def _calculate_results(self) -> None:
        begin_generation = self._signposts.get("begin_generation")
        if begin_generation:
            self.startup_time = (
                self._signposts["begin_generation"] - self._start_time
            ) * 1000.0
        else:
            self.startup_time = "n/a"

        # Calculate TTFT & context-encoding throughput
        first_token = self._signposts.get("first_token")
        if first_token and begin_generation:
            self.time_to_first_token = (
                self._signposts["first_token"]
                - self._signposts["begin_generation"]
            ) * 1000.0
            self.prompt_eval_throughput = (
                self.prompt_size
                * self.batch_size
                / (self.time_to_first_token / 1000.0)
            )
        else:
            self.time_to_first_token = "n/a"
            self.prompt_eval_throughput = "n/a"

        # Calculate TPOT & token-gen throughput
        end_generation = self._signposts.get("end_generation")
        if end_generation and first_token and self.output_size > 1:
            generation_time = (
                self._signposts["end_generation"]
                - self._signposts["first_token"]
            )
            self.eval_throughput = (
                (self.output_size - 1) * self.batch_size / generation_time
            )
            self.time_per_output_token: Any = (
                generation_time * 1000.0 / (self.output_size - 1)
            )
        else:
            self.eval_throughput = "n/a"
            self.time_per_output_token = "n/a"

        if end_generation and begin_generation:
            total_batch_time = (
                self._signposts["end_generation"]
                - self._signposts["begin_generation"]
            )
            self.requests_per_second: Any = self.batch_size / total_batch_time
            self.total_exe_time: Any = total_batch_time * 1000
        else:
            self.total_exe_time = "n/a"
            self.requests_per_second = "n/a"

    def _print_report(self, print_raw: bool = False) -> None:
        print()
        print("Prompt size:", self.prompt_size)
        print("Output size:", self.output_size)
        print("Startup time:", self.startup_time, "ms")
        print("Time to first token:", self.time_to_first_token, "ms")
        print(
            "Prompt eval throughput (context-encoding):",
            self.prompt_eval_throughput,
            "tokens per second",
        )
        print("Time per Output Token:", self.time_per_output_token, "ms")
        print(
            "Eval throughput (token-generation):",
            self.eval_throughput,
            "tokens per second",
        )
        print("Total Latency:", self.total_exe_time, "ms")
        print("Total Throughput:", self.requests_per_second, "req/s")
        if print_raw:
            print("=============raw stats=================")
            for k, v in self._signposts.items():
                print(
                    f"Started {k} at {v} with memory"
                    f" {self._mem_usage_marker[k]} GB"
                )


class EmbeddingsMetrics:
    prompt_size: int
    startup_time: float | str

    _start_time: float
    _signposts: dict[str, float]
    _mem_usage_marker: dict[str, float]
    _should_print_report: bool
    _process: psutil.Process
    _print_raw: bool

    def __init__(
        self, print_report: bool = False, print_raw: bool = False
    ) -> None:
        self._signposts = {}
        self._mem_usage_marker = {}
        self.batch_size = 1
        self.prompt_size = 0
        self._should_print_report = print_report
        self._start_time = time.time()
        self._process = psutil.Process()
        self._print_raw = print_raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._calculate_results()
        if self._should_print_report:
            self._print_report(self._print_raw)

    def signpost(self, name: str) -> None:
        """Measure the current time and memory usage, tagging it with a name for later reporting."""
        self._signposts[name] = time.time()
        self._mem_usage_marker[name] = (self._process.memory_info().rss) / (
            1024 * 1024 * 1024
        )

    def _calculate_results(self) -> None:
        begin_encoding = self._signposts.get("begin_encoding")
        if begin_encoding:
            self.startup_time = (begin_encoding - self._start_time) * 1000.0
        else:
            self.startup_time = "n/a"

        # Calculate total latency and throughput
        end_encoding = self._signposts.get("end_encoding")
        if end_encoding and begin_encoding:
            total_batch_time = end_encoding - begin_encoding
            self.requests_per_second: Any = self.batch_size / total_batch_time
            self.total_exe_time: Any = total_batch_time * 1000
        else:
            self.requests_per_second = "n/a"
            self.total_exe_time = "n/a"

    def _print_report(self, print_raw: bool = False) -> None:
        print()
        print("Prompt size:", self.prompt_size)
        print("Startup time:", self.startup_time, "ms")
        print("Total Latency:", self.total_exe_time, "ms")
        print("Total Throughput:", self.requests_per_second, "req/s")
        if print_raw:
            print("=============raw stats=================")
            for k, v in self._signposts.items():
                print(
                    f"Started {k} at {v} with memory"
                    f" {self._mem_usage_marker[k]} GB"
                )
