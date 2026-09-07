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
"""Benchmarks for Error and StackTrace performance.

These benchmarks measure the cost of error handling with stack trace collection.

Key metrics:
- error_catch_no_print: The common case - catch error without printing stack trace
- error_catch_depth_*: Shows impact of call stack depth on capture time
"""

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep


# ===-----------------------------------------------------------------------===#
# Helper functions to create errors at various call depths
# ===-----------------------------------------------------------------------===#


@no_inline
def create_error_depth_1() raises:
    """Create and raise an error at depth 1."""
    raise Error("test error")


@no_inline
def create_error_depth_2() raises:
    """Create and raise an error at depth 2."""
    create_error_depth_1()


@no_inline
def create_error_depth_3() raises:
    """Create and raise an error at depth 3."""
    create_error_depth_2()


@no_inline
def create_error_depth_4() raises:
    """Create and raise an error at depth 4."""
    create_error_depth_3()


@no_inline
def create_error_depth_5() raises:
    """Create and raise an error at depth 5."""
    create_error_depth_4()


@no_inline
def create_error_depth_10() raises:
    """Create and raise an error at depth 10."""

    @no_inline
    def d6() raises:
        create_error_depth_5()

    @no_inline
    def d7() raises:
        d6()

    @no_inline
    def d8() raises:
        d7()

    @no_inline
    def d9() raises:
        d8()

    d9()


# ===-----------------------------------------------------------------------===#
# Benchmarks
# ===-----------------------------------------------------------------------===#


def bench_error_catch_no_print_depth3(mut b: Bencher) raises:
    """Benchmark catching errors without printing the stack trace.

    This is the common case in production code - errors are caught and handled
    without printing.
    """

    @always_inline
    def call_fn():
        for _ in range(100):
            try:
                create_error_depth_3()
            except e:
                keep(e)

    b.iter(call_fn)


def bench_error_catch_depth1(mut b: Bencher) raises:
    """Benchmark with shallow call stack (depth 1)."""

    @always_inline
    def call_fn():
        for _ in range(100):
            try:
                create_error_depth_1()
            except e:
                keep(e)

    b.iter(call_fn)


def bench_error_catch_depth5(mut b: Bencher) raises:
    """Benchmark with medium call stack (depth 5)."""

    @always_inline
    def call_fn():
        for _ in range(100):
            try:
                create_error_depth_5()
            except e:
                keep(e)

    b.iter(call_fn)


def bench_error_catch_depth10(mut b: Bencher) raises:
    """Benchmark with deeper call stack (depth 10)."""

    @always_inline
    def call_fn():
        for _ in range(100):
            try:
                create_error_depth_10()
            except e:
                keep(e)

    b.iter(call_fn)


def bench_error_create_only(mut b: Bencher) raises:
    """Benchmark just creating Error objects (no raise/catch overhead)."""

    @always_inline
    def call_fn():
        for _ in range(100):
            var e = Error("test error")
            keep(e)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#


def main() raises:
    var m = Bench(BenchConfig(num_repetitions=3))

    m.bench_function(bench_error_create_only, BenchId("error_create_only"))
    m.bench_function(
        bench_error_catch_no_print_depth3,
        BenchId("error_catch_no_print_depth3"),
    )
    m.bench_function(bench_error_catch_depth1, BenchId("error_catch_depth1"))
    m.bench_function(bench_error_catch_depth5, BenchId("error_catch_depth5"))
    m.bench_function(bench_error_catch_depth10, BenchId("error_catch_depth10"))

    m.dump_report()
