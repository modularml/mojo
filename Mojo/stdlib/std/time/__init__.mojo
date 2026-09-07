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
"""Timing operations: monotonic clocks, performance counters, sleep, time_function.

The `time` package provides utilities for measuring elapsed time, benchmarking
code performance, and introducing delays. It offers monotonic clocks that are
unaffected by system clock adjustments, making them suitable for measuring
intervals and profiling execution time.

Use this package for performance measurement, benchmarking, profiling, or when
you need to introduce delays in your code.
"""

from .time import (
    global_perf_counter_ns,
    monotonic,
    perf_counter,
    perf_counter_ns,
    sleep,
    time_function,
)
