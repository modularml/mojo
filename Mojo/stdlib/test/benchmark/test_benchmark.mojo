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

from std.time import sleep, time_function

from std.benchmark import Batch, Report, clobber_memory, keep, run
from std.benchmark.bencher import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    Format,
    ThroughputMeasure,
)
from std.testing import (
    TestSuite,
    assert_equal,
    assert_not_equal,
    assert_raises,
    assert_true,
    assert_false,
)
from test_utils import check_write_to


def test_stopping_criteria() raises:
    # Stop when min_runtime_secs has elapsed and either max_runtime_secs or max_iters
    # is reached

    @always_inline
    def time_me():
        sleep(0.002)
        clobber_memory()
        return

    var lb = 0.02  # 20ms
    var ub = 0.1  # 100ms

    # stop after ub (max_runtime_secs)
    var max_iters_1 = 1000_000_000

    def timer() raises {var lb, var ub, mut max_iters_1}:
        var report = run(
            time_me,
            max_iters=max_iters_1,
            min_runtime_secs=lb,
            max_runtime_secs=ub,
        )
        assert_true(report.mean() > 0)
        assert_true(report.iters() != max_iters_1)

    var t1 = time_function(timer)
    assert_true(Float64(t1) / 1e9 >= ub)

    # stop after lb (min_runtime_secs)
    var ub_big = 1  # 1s
    var max_iters_2 = 1

    def timer2() raises {var ub_big, var lb, mut max_iters_2}:
        var report = run(
            time_me,
            max_iters=max_iters_2,
            min_runtime_secs=lb,
            max_runtime_secs=Float64(ub_big),
        )
        assert_true(report.mean() > 0)
        assert_true(report.iters() >= max_iters_2)

    var t2 = time_function(timer2)

    assert_true(
        Float64(t2) / 1e9 >= lb and Float64(t2) / 1e9 <= Float64(ub_big)
    )

    # stop on or before max_iters
    var max_iters_3 = 3

    def timer3() raises {var ub_big, mut max_iters_3}:
        var report = run(
            time_me,
            max_iters=max_iters_3,
            min_runtime_secs=0,
            max_runtime_secs=Float64(ub_big),
        )
        assert_true(report.mean() > 0)
        assert_true(report.iters() <= max_iters_3)

    var t3 = time_function(timer3)

    assert_true(Float64(t3) / 1e9 <= Float64(ub_big))


struct SomeStruct(TrivialRegisterPassable):
    var x: Int
    var y: Int

    @always_inline
    def __init__(out self):
        self.x = 5
        self.y = 4


struct SomeTrivialStruct(TrivialRegisterPassable):
    var x: Int
    var y: Int

    @always_inline
    def __init__(out self):
        self.x = 3
        self.y = 5


# There is nothing to test here other than the code executes and does not crash.
def test_keep() raises:
    keep(False)
    keep(33)

    var val = SIMD[.int, 4](1, 2, 3, 4)
    keep(val)

    var ptr = Pointer(to=val)
    keep(ptr)

    var s0 = SomeStruct()
    keep(s0)

    var s1 = SomeTrivialStruct()
    keep(s1)


def sleeper():
    sleep(0.001)


def test_non_capturing() raises:
    var report = run(sleeper, min_runtime_secs=0.1, max_runtime_secs=0.3)
    assert_true(report.mean() > 0.001)


def test_change_units() raises:
    var report = run(sleeper, min_runtime_secs=0.1, max_runtime_secs=0.3)
    assert_true(report.mean("ms") > 1.0)
    assert_true(report.mean("us") > 1_000)
    assert_true(report.mean("ns") > 1_000_000.0)


def test_report() raises:
    var report = run(sleeper, min_runtime_secs=0.1, max_runtime_secs=0.3)

    var report_string = report.as_string()
    assert_true("Benchmark Report (s)" in report_string)
    assert_true("Mean: " in report_string)
    assert_true("Total: " in report_string)
    assert_true("Iters: " in report_string)
    assert_true("Warmup Total: " in report_string)
    assert_true("Fastest Mean: " in report_string)
    assert_true("Slowest Mean: " in report_string)


def test_bench_metric_write_repr_to() raises:
    var s = String()
    BenchMetric.elements.write_repr_to(s)
    assert_true(s.startswith("BenchMetric("))
    assert_true("code=0" in s)
    assert_true("name=" in s)
    assert_true("unit=" in s)


def test_format_write_repr_to() raises:
    var s = String()
    Format.csv.write_repr_to(s)
    assert_equal(s, "Format('csv')")

    s = String()
    Format.table.write_repr_to(s)
    assert_equal(s, "Format('table')")


def test_throughput_measure_write_repr_to() raises:
    var m = ThroughputMeasure(BenchMetric.elements, 1024)
    var s = String()
    m.write_repr_to(s)
    assert_true(s.startswith("ThroughputMeasure("))
    assert_true("metric=" in s)
    assert_true("value=1024" in s)


def test_batch_write_to() raises:
    var b = Batch(duration=1000, iterations=10, _is_significant=True)
    check_write_to(
        b,
        expected="Batch(duration=1000ns, iterations=10, significant=True)",
        is_repr=False,
    )


def test_batch_write_repr_to() raises:
    var b = Batch(duration=2000, iterations=5, _is_significant=False)
    check_write_to(
        b,
        expected="Batch(duration=2000, iterations=5, _is_significant=False)",
        is_repr=True,
    )


def test_bencher_iter_unified() raises:
    """Tests Bencher.iter with a unified closure."""
    var bencher = Bencher(3)

    var count = 0

    @always_inline
    def increment() {
        mut count,
    }:
        count += 1

    bencher.iter(increment)
    assert_equal(count, 3)


def test_bencher_iter_unified_raising() raises:
    """Tests Bencher.iter with a raising unified closure."""
    var bencher = Bencher(3)

    var count = 0
    var data = [1, 2, 3]

    @always_inline
    def increment() raises {
        mut count,
        var data^,
    }:
        count += len(data)

    # `data` is owned by the closure, so it stays alive for the whole measured
    # loop; an implicit by-reference capture would not keep it alive.
    bencher.iter(increment)
    assert_equal(count, 9)


def test_bencher_iter_unified_raising_propagates() raises:
    """Tests Bencher.iter propagating an error out of a unified closure."""
    var bencher = Bencher(3)

    var count = 0

    @always_inline
    def fail_on_second() raises {
        mut count,
    }:
        count += 1
        if count == 2:
            raise Error("boom")

    with assert_raises(contains="boom"):
        bencher.iter(fail_on_second)

    assert_equal(count, 2)


def test_bencher_iter_preproc_unified() raises:
    """Tests Bencher.iter_preproc with unified closures."""
    var bencher = Bencher(2)

    var count = 0
    var state = 0

    @always_inline
    def work(mut state: Int) {mut count}:
        count += 1
        state += 1

    @always_inline
    def preproc(mut state: Int):
        state += 10

    bencher.iter_preproc(state, work, preproc)
    assert_equal(count, 2)
    assert_equal(state, 22)


def test_bencher_iter_custom_unified() raises:
    """Tests Bencher.iter_custom with a unified closure."""
    var bencher = Bencher(5)

    @always_inline
    def custom_timer(num_iters: Int) -> Int:
        return num_iters * 100

    bencher.iter_custom(custom_timer)
    assert_equal(bencher.elapsed, 500)


def test_bench_function_unified() raises:
    """Tests Bench.bench_function with a unified closure taking mut Bencher."""
    var bench = Bench(BenchConfig(max_iters=2, max_runtime_secs=0.01))

    var call_count = 0

    @always_inline
    def noop():
        pass

    @always_inline
    def my_bench(
        mut b: Bencher,
    ) {mut call_count,}:
        call_count += 1
        b.iter(noop)

    bench.bench_function(my_bench, BenchId("test_unified"))
    assert_true(call_count > 0)
    assert_equal(len(bench.info_vec), 1)


def test_bench_with_input_unified() raises:
    """Tests Bench.bench_with_input with a unified closure."""
    var bench = Bench(BenchConfig(max_iters=2, max_runtime_secs=0.01))

    var call_count = 0

    @always_inline
    def my_bench(
        mut b: Bencher,
        input: Int,
    ) {mut call_count,}:
        call_count += 1

        @always_inline
        def noop():
            pass

        b.iter(noop)

    bench.bench_with_input(my_bench, BenchId("test_with_input_unified"), 42)
    assert_true(call_count > 0)
    assert_equal(len(bench.info_vec), 1)


def test_bench_function_no_arg_unified() raises:
    """Tests Bench.bench_function with a no-arg unified closure."""
    var bench = Bench(BenchConfig(max_iters=2, max_runtime_secs=0.01))

    var count = 0

    @always_inline
    def my_func() {
        mut count,
    }:
        count += 1

    bench.bench_function(my_func, BenchId("test_noarg_unified"))
    assert_true(count > 0)


def test_bench_function_no_arg_raising_unified() raises:
    """Tests Bench.bench_function with a raising no-arg unified closure."""
    var bench = Bench(BenchConfig(max_iters=2, max_runtime_secs=0.01))

    var count = 0

    @always_inline
    def my_func() raises {
        mut count,
    }:
        count += 1

    bench.bench_function(my_func, BenchId("test_noarg_raising_unified"))
    assert_true(count > 0)


def test_bench_id_hash() raises:
    var bench_id1 = BenchId("foo()", "123")

    assert_equal(hash(bench_id1), hash(bench_id1))
    assert_not_equal(hash(bench_id1), hash(BenchId("bar()")))
    assert_not_equal(hash(bench_id1), hash(BenchId("bar()", "123")))


def test_bench_id_eq() raises:
    var bench_id1 = BenchId("foo()", "123")

    assert_equal(bench_id1, bench_id1)
    assert_not_equal(bench_id1, BenchId("foo()", "456"))
    assert_not_equal(bench_id1, BenchId("bar()"))
    assert_not_equal(bench_id1, BenchId("bar()", "123"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
