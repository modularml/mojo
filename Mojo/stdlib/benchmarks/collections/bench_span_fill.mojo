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

# Checks that steering byte-fill callers to the safe `Span[Byte].fill` costs
# nothing versus the `unsafe_memset` it replaces (and a naive element-wise loop).
#
# The element count goes through `black_box` so the compiler sees the fill
# length as a runtime value. With a compile-time-constant length it would
# inline-expand/hoist the work and misrepresent the cost of a genuinely
# dynamic-sized fill. The fill value is likewise opaque, and the trailing
# `keep` stops the stores from being dead-code eliminated.
#
# Each closure captures the `Allocation` and derives its view inside the body
# rather than capturing a `Span`/`Pointer` from the enclosing scope: a capture
# whose type names an interior origin (`origin_of(allocation._alloc)`) fails to
# type check (MOCO-4433).

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, black_box, keep
from std.memory import Layout, alloc, dealloc, unsafe_memset

comptime SIZES = [64, 256, 4096, 65536, 1048576]
comptime FILL_VALUE = Byte(0xA5)


def bench_span_fill[size: Int](mut b: Bencher) raises:
    var allocation = alloc(Layout[Byte](count=black_box(size)))

    @always_inline
    def call_fn() {mut allocation}:
        var s = allocation.unsafe_span()
        s.fill(black_box(FILL_VALUE))
        keep(s)

    b.iter(call_fn)
    dealloc(allocation^)


def bench_unsafe_memset[size: Int](mut b: Bencher) raises:
    var allocation = alloc(Layout[Byte](count=black_box(size)))

    @always_inline
    def call_fn() {mut allocation}:
        var ptr = allocation.unsafe_ptr()
        unsafe_memset(ptr, black_box(FILL_VALUE), allocation.layout().count())
        keep(ptr)

    b.iter(call_fn)
    dealloc(allocation^)


def bench_span_fill_elementwise[size: Int](mut b: Bencher) raises:
    var allocation = alloc(Layout[Byte](count=black_box(size)))

    @always_inline
    def call_fn() {mut allocation}:
        var s = allocation.unsafe_span()
        ref value = black_box(FILL_VALUE)
        for ref element in s:
            var p = Pointer(to=element).unsafe_mut_cast[True]()
            p[] = value
        keep(s)

    b.iter(call_fn)
    dealloc(allocation^)


def main() raises:
    # A small fill is a handful of nanoseconds, so let the harness batch until
    # a run is long enough to time; the default 1_000-iteration cap leaves the
    # 64- and 256-byte rows below the clock's resolution.
    var m = Bench(BenchConfig(min_runtime_secs=0.01, max_iters=10_000_000))
    comptime for size in SIZES:
        m.bench_function(
            bench_span_fill[size], BenchId("span_fill/" + String(size))
        )
        m.bench_function(
            bench_unsafe_memset[size], BenchId("unsafe_memset/" + String(size))
        )
        m.bench_function(
            bench_span_fill_elementwise[size],
            BenchId("span_fill_elementwise/" + String(size)),
        )
    m.dump_report()
