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
"""Microbenchmarks for `PythonObject` attribute access and operator dispatch.

These cover the Python interop hot path:

- `getattr`  : `obj.__getattr__("...")` attribute lookup.
- `op_add`   : `a + b`, dispatched through CPython's `PyNumber_Add`.
- `op_iadd`  : `a += b`, dispatched through `PyNumber_InPlaceAdd`.
- `op_lt`    : `a < b`, dispatched through `PyObject_RichCompare`.
- `op_in`    : `x in lst`, dispatched through `PySequence_Contains`.

Each benchmark holds the GIL across its whole measured region via `GILAcquired`
so the timing is robust regardless of which thread the `std.benchmark` harness
runs the closure on. The acquire/release happens once per run, outside the
measured loop, so it does not perturb the per-call numbers.

Operands go in an explicit capture list, which keeps them alive for as long as
the closure is. The implicit by-reference capture of an `@__parameter` closure
does not: it is not a use, so an operand whose last use precedes `b.iter` is
destroyed before the measured closure runs, leaving the closure to operate on a
freed `PyObject`.
"""

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep
from std.python import Python, PythonObject
from std.python._cpython import GILAcquired

comptime LOOP_SIZE = 1000
"""Number of interop calls per bench iteration."""


def bench_getattr(mut b: Bencher) raises:
    var py = Python()
    with GILAcquired(py):
        var obj = PythonObject(42)

        @always_inline
        def call_fn() raises {var obj}:
            for _ in range(LOOP_SIZE):
                var r = obj.__getattr__("numerator")
                keep(r)

        b.iter(call_fn)


def bench_op_add(mut b: Bencher) raises:
    var py = Python()
    with GILAcquired(py):
        var a = PythonObject(42)
        var c = PythonObject(10)

        @always_inline
        def call_fn() raises {var a, var c}:
            for _ in range(LOOP_SIZE):
                var r = a + c
                keep(r)

        b.iter(call_fn)


def bench_op_iadd(mut b: Bencher) raises:
    var py = Python()
    with GILAcquired(py):
        var a = PythonObject(42)
        # A zero step keeps the running total inside CPython's small-integer
        # cache, so the measurement stays on dispatch instead of drifting into
        # `int` allocation as the value grows.
        var c = PythonObject(0)

        @always_inline
        def call_fn() raises {mut a, var c}:
            for _ in range(LOOP_SIZE):
                a += c

        b.iter(call_fn)


def bench_op_lt(mut b: Bencher) raises:
    var py = Python()
    with GILAcquired(py):
        var a = PythonObject(42)
        var c = PythonObject(10)

        @always_inline
        def call_fn() raises {var a, var c}:
            for _ in range(LOOP_SIZE):
                var r = a < c
                keep(r)

        b.iter(call_fn)


def bench_op_in(mut b: Bencher) raises:
    var py = Python()
    with GILAcquired(py):
        var lst = Python.list(1, 2, 3, 4, 5)
        var needle = PythonObject(5)

        @always_inline
        def call_fn() raises {var lst, var needle}:
            for _ in range(LOOP_SIZE):
                var r = needle in lst
                keep(r)

        b.iter(call_fn)


def main() raises:
    var m = Bench(BenchConfig(num_repetitions=3))

    m.bench_function(bench_getattr, BenchId("bench_getattr"))
    m.bench_function(bench_op_add, BenchId("bench_op_add"))
    m.bench_function(bench_op_iadd, BenchId("bench_op_iadd"))
    m.bench_function(bench_op_lt, BenchId("bench_op_lt"))
    m.bench_function(bench_op_in, BenchId("bench_op_in"))

    m.dump_report()
