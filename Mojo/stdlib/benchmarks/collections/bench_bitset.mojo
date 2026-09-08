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

from std.collections import BitSet
from std.random import *

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep

comptime INIT_LOOP_SIZE = 1000000
"""Bench loop size for BitSet init tests."""

comptime OP_LOOP_SIZE = 1000
"""Bench loop size for BitSet operation tests."""


def bench_empty_bitset_init[size: Int](mut b: Bencher) raises:
    @always_inline
    def call_fn():
        for _ in range(0, INIT_LOOP_SIZE):
            var b = BitSet[size]()
            keep(len(b))

    b.iter(call_fn)


def bench_bitset_init_from[width: Int](mut b: Bencher) raises:
    var initial = SIMD[.bool, width](fill=True)

    @always_inline
    def call_fn() {var initial}:
        for _ in range(0, INIT_LOOP_SIZE):
            var b = BitSet(initial)
            keep(len(b))

    b.iter(call_fn)


def bench_bitset_set[size: Int](mut b: Bencher) raises:
    @always_inline
    def call_fn():
        var bitset = BitSet[size]()
        for _ in range(0, OP_LOOP_SIZE):
            comptime for i in range(0, bitset.size):
                bitset.set(i)
        keep(len(bitset))

    b.iter(call_fn)


def bench_bitset_clear[width: Int](mut b: Bencher) raises:
    var initial = SIMD[.bool, width](fill=True)

    @always_inline
    def call_fn() {var initial}:
        var bitset = BitSet[width](initial)
        for _ in range(0, OP_LOOP_SIZE):
            comptime for i in range(0, bitset.size):
                bitset.clear(i)

        keep(len(bitset))

    b.iter(call_fn)


def bench_bitset_toggle[width: Int](mut b: Bencher) raises:
    var initial = SIMD[.bool, width](fill=True)

    @always_inline
    def call_fn() {var initial}:
        var bitset = BitSet[width](initial)
        for _ in range(0, OP_LOOP_SIZE):
            comptime for i in range(0, bitset.size):
                bitset.toggle(i)

        keep(len(bitset))

    b.iter(call_fn)


def bench_bitset_test[width: Int](mut b: Bencher) raises:
    var initial = SIMD[.bool, width](fill=True)

    @always_inline
    def call_fn() {var initial}:
        var bitset = BitSet[width](initial)
        for _ in range(0, OP_LOOP_SIZE):
            comptime for i in range(0, bitset.size):
                keep(bitset.test(i))

    b.iter(call_fn)


def bench_bitset_union[width: Int](mut b: Bencher) raises:
    var lhs_init = SIMD[.bool, width](fill=True)
    var rhs_init = SIMD[.bool, width](fill=False)

    @always_inline
    def call_fn() {var lhs_init, var rhs_init}:
        var lhs = BitSet[width](lhs_init)
        var rhs = BitSet[width](rhs_init)

        for _ in range(0, OP_LOOP_SIZE):
            var new = lhs.union(rhs)
            keep(len(new))

    b.iter(call_fn)


def bench_bitset_intersection[width: Int](mut b: Bencher) raises:
    var lhs_init = SIMD[.bool, width](fill=True)
    var rhs_init = SIMD[.bool, width](fill=False)

    @always_inline
    def call_fn() {var lhs_init, var rhs_init}:
        var lhs = BitSet[width](lhs_init)
        var rhs = BitSet[width](rhs_init)

        for _ in range(0, OP_LOOP_SIZE):
            var new = lhs.intersection(rhs)
            keep(len(new))

    b.iter(call_fn)


def bench_bitset_difference[width: Int](mut b: Bencher) raises:
    var lhs_init = SIMD[.bool, width](fill=True)
    var rhs_init = SIMD[.bool, width](fill=False)

    @always_inline
    def call_fn() {var lhs_init, var rhs_init}:
        var lhs = BitSet[width](lhs_init)
        var rhs = BitSet[width](rhs_init)

        for _ in range(0, OP_LOOP_SIZE):
            var new = lhs.difference(rhs)
            keep(len(new))

    b.iter(call_fn)


def main() raises:
    seed()
    comptime widths = (1, 2, 4, 8, 16)
    comptime sizes = (10, 30, 50, 100, 1000)
    var m = Bench(BenchConfig(num_repetitions=1))

    comptime for i in range(len(sizes)):
        comptime size = rebind[Int](sizes[i])
        m.bench_function(
            bench_empty_bitset_init[size],
            BenchId(String("bench_empty_bitset_init[", size, "]")),
        )

    comptime for width_idx in range(0, len(widths)):
        comptime width = rebind[Int](widths[width_idx])
        m.bench_function(
            bench_bitset_init_from[width],
            BenchId(String("bench_bitset_init_from[", width, "]")),
        )

    comptime for width_idx in range(0, len(widths)):
        comptime width = rebind[Int](widths[width_idx])
        m.bench_function(
            bench_bitset_set[width],
            BenchId(String("bench_bitset_set[", width, "]")),
        )

    comptime for width_idx in range(0, len(widths)):
        comptime width = rebind[Int](widths[width_idx])
        m.bench_function(
            bench_bitset_clear[width],
            BenchId(String("bench_bitset_clear[", width, "]")),
        )

    comptime for width_idx in range(0, len(widths)):
        comptime width = rebind[Int](widths[width_idx])
        m.bench_function(
            bench_bitset_test[width],
            BenchId(String("bench_bitset_test[", width, "]")),
        )

    comptime for width_idx in range(0, len(widths)):
        comptime width = rebind[Int](widths[width_idx])
        m.bench_function(
            bench_bitset_toggle[width],
            BenchId(String("bench_bitset_toggle[", width, "]")),
        )

    comptime for width_idx in range(0, len(widths)):
        comptime width = rebind[Int](widths[width_idx])
        m.bench_function(
            bench_bitset_union[width],
            BenchId(String("bench_bitset_union[", width, "]")),
        )

    comptime for width_idx in range(0, len(widths)):
        comptime width = rebind[Int](widths[width_idx])
        m.bench_function(
            bench_bitset_intersection[width],
            BenchId(String("bench_bitset_intersection[", width, "]")),
        )

    comptime for width_idx in range(0, len(widths)):
        comptime width = rebind[Int](widths[width_idx])
        m.bench_function(
            bench_bitset_difference[width],
            BenchId(String("bench_bitset_difference[", width, "]")),
        )

    m.dump_report()
