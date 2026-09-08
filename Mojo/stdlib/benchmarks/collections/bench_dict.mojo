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

from std.collections.dict import DictEntry
from std.hashlib import Hasher
from std.random.random import random_si64, seed
from std.sys import size_of

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, black_box, keep


# ===-----------------------------------------------------------------------===#
# Benchmark Data
# ===-----------------------------------------------------------------------===#
def make_dict[size: Int, *, random: Bool = False]() -> Dict[Int, Int]:
    var d = Dict[Int, Int]()
    for i in range(0, size):
        comptime if random:
            d[i] = Int(random_si64(0, Int64(size)))
        else:
            d[i] = i
    return d^


# ===-----------------------------------------------------------------------===#
# Benchmark Dict init
# ===-----------------------------------------------------------------------===#
def bench_dict_init(mut b: Bencher) raises:
    @always_inline
    def call_fn():
        for _ in range(1000):
            var d = Dict[Int, Int]()
            keep(d)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Dict Insert
# ===-----------------------------------------------------------------------===#
def bench_dict_insert[size: Int](mut b: Bencher) raises:
    """Insert 10 new items 100_000 times."""
    var items = make_dict[size]()

    @always_inline
    def call_fn() raises {mut items}:
        for _ in range(10_000):
            for key in range(size, size + 10):
                items[key] = Int(random_si64(0, Int64(size)))

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Dict Lookup
# ===-----------------------------------------------------------------------===#
def bench_dict_lookup[size: Int](mut b: Bencher) raises:
    """Lookup 10 items 100_000 times."""
    var items = make_dict[size]()

    @always_inline
    def call_fn() raises {imm items}:
        for _ in range(10_000):
            for key in range(10):
                var res = items[key]
                keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Dict contains
# ===-----------------------------------------------------------------------===#
def bench_dict_contains[size: Int](mut b: Bencher) raises:
    """Check if the dict contains 10 keys 100_000 times."""
    var items = make_dict[size]()

    @always_inline
    def call_fn() raises {imm items}:
        for _ in range(100_000):
            for key in range(10):
                var res = key in items
                keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Dict Lookup Miss
# ===-----------------------------------------------------------------------===#
def bench_dict_lookup_miss[size: Int](mut b: Bencher) raises:
    """Lookup 10 missing keys 100_000 times."""
    var items = make_dict[size]()

    @always_inline
    def call_fn() raises {imm items}:
        for _ in range(10_000):
            for key in range(size, size + 10):
                var res = black_box(key) in items
                keep(res)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Dict Defaulting Get / Find
# ===-----------------------------------------------------------------------===#
def bench_dict_get_default_miss[size: Int](mut b: Bencher) raises:
    """Look up 10 missing keys through the defaulting `get` 10_000 times."""
    var items = make_dict[size]()

    @always_inline
    def call_fn() {imm items}:
        var total = 0
        for _ in range(10_000):
            for key in range(size, size + 10):
                total += items.get(black_box(key), 0)
        keep(total)

    b.iter(call_fn)


def bench_dict_find_hit[size: Int](mut b: Bencher) raises:
    """Look up 10 present keys through `find` 10_000 times."""
    var items = make_dict[size]()

    @always_inline
    def call_fn() {imm items}:
        var total = 0
        for _ in range(10_000):
            for key in range(10):
                total += items.find(black_box(key)).value()
        keep(total)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Dict Insert/Delete
# ===-----------------------------------------------------------------------===#
def bench_dict_insert_delete[size: Int](mut b: Bencher) raises:
    """Insert and immediately delete 10_000 times."""
    var items = make_dict[size]()

    @always_inline
    def call_fn() raises {mut items}:
        for i in range(10_000):
            var key = black_box(size + i)
            items[key] = i
            var result = items.pop(key, 0)
            keep(result)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Dict Iteration
# ===-----------------------------------------------------------------------===#
def bench_dict_iter[size: Int](mut b: Bencher) raises:
    """Iterate over all keys."""
    var items = make_dict[size]()

    @always_inline
    def call_fn() raises {imm items}:
        for key in black_box(items):
            keep(key)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark whole-dictionary operations
# ===-----------------------------------------------------------------------===#
# The benchmarks above measure per-key operations against a dictionary that
# lives across iterations. These measure operations whose cost is proportional
# to the whole dictionary, so each one builds (and for the destructive cases,
# destroys) its subject inside the body. `bench_dict_build` is the control:
# subtract it from `bench_dict_popitem_drain` to isolate the drain itself.


def bench_dict_build[size: Int](mut b: Bencher) raises:
    """Build a dictionary of `size` entries from empty."""

    @always_inline
    def call_fn():
        var items = Dict[Int, Int]()
        for i in range(black_box(size)):
            items[i] = i
        keep(items)

    b.iter(call_fn)


def bench_dict_accumulate[size: Int](mut b: Bencher) raises:
    """Tally a stream of `4 * size` draws over a keyspace of `size`.

    This is the shape accumulating code has — word counts, grouping,
    histograms: the dictionary grows from empty while most draws land on a
    key that is already resident, and every draw reads a key before writing
    it back. `bench_dict_insert` re-inserts a key set that is resident from
    its first iteration onward, so it never exercises growth.

    The draw sequence is a xorshift so the keys arrive scattered rather than
    in the ascending order the rest of the file uses, and it is seeded to a
    constant so every run tallies the same stream.
    """

    @always_inline
    def call_fn():
        var n = black_box(size)
        var counts = Dict[Int, Int]()
        var x = UInt64(0x9E3779B97F4A7C15)
        for _ in range(n * 4):
            x ^= x >> 12
            x ^= x << 25
            x ^= x >> 27
            var key = Int(x % UInt64(n))
            counts[key] = counts.get(key, 0) + 1
        keep(counts)

    b.iter(call_fn)


def bench_dict_popitem_drain[size: Int](mut b: Bencher) raises:
    """Drain a dictionary entirely through `popitem`.

    `popitem` scans the insertion-order array backwards for a live slot, so a
    full drain is the shape that turns a failure to discard the dead tail into
    quadratic work.
    """

    @always_inline
    def call_fn() raises:
        var items = Dict[Int, Int]()
        for i in range(black_box(size)):
            items[i] = i
        var total = 0
        for _ in range(size):
            total += items.popitem().value
        keep(total)

    b.iter(call_fn)


def bench_dict_update[size: Int](mut b: Bencher) raises:
    """Merge a dictionary of `size` entries into an empty one."""
    var other = make_dict[size]()

    @always_inline
    def call_fn() {imm other}:
        var items = Dict[Int, Int]()
        items.update(other)
        keep(items)

    b.iter(call_fn)


def bench_dict_eq[size: Int](mut b: Bencher) raises:
    """Compare two equal dictionaries, which probes every key."""
    var left = make_dict[size]()
    var right = make_dict[size]()

    @always_inline
    def call_fn() {imm left, imm right}:
        keep(left == right)

    b.iter(call_fn)


# ===-----------------------------------------------------------------------===#
# Benchmark Dict Memory Footprint
# ===-----------------------------------------------------------------------===#


def total_bytes_used[H: Hasher](items: Dict[Int, Int, H]) -> Int:
    # The SIMD group width used internally by Dict's Swiss Table.
    comptime _BENCH_GROUP_WIDTH: Int = 16
    # ctrl bytes: capacity + GROUP_WIDTH for SIMD mirroring
    var ctrl_bytes = (items._reserved() + _BENCH_GROUP_WIDTH) * size_of[UInt8]()
    # slot storage: one DictEntry per capacity slot
    var slot_bytes = items._reserved() * size_of[DictEntry[Int, Int, H]]()
    # struct overhead (includes _order List inline storage)
    var struct_bytes = size_of[Dict[Int, Int, H]]()
    return ctrl_bytes + slot_bytes + struct_bytes


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main() raises:
    seed()
    var m = Bench(BenchConfig(num_repetitions=5))
    m.bench_function(bench_dict_init, BenchId("bench_dict_init"))
    comptime sizes = (10, 30, 50, 100, 1000, 10_000, 100_000, 1_000_000)

    comptime for i in range(len(sizes)):
        comptime size = rebind[Int](sizes[i])
        m.bench_function(
            bench_dict_insert[size],
            BenchId(String("bench_dict_insert[", size, "]")),
        )
        m.bench_function(
            bench_dict_lookup[size],
            BenchId(String("bench_dict_lookup[", size, "]")),
        )
        m.bench_function(
            bench_dict_contains[size],
            BenchId(String("bench_dict_contains[", size, "]")),
        )
        m.bench_function(
            bench_dict_lookup_miss[size],
            BenchId(String("bench_dict_lookup_miss[", size, "]")),
        )
        m.bench_function(
            bench_dict_get_default_miss[size],
            BenchId(String(t"bench_dict_get_default_miss[{size}]")),
        )
        m.bench_function(
            bench_dict_find_hit[size],
            BenchId(String(t"bench_dict_find_hit[{size}]")),
        )
        m.bench_function(
            bench_dict_insert_delete[size],
            BenchId(String("bench_dict_insert_delete[", size, "]")),
        )
        m.bench_function(
            bench_dict_iter[size],
            BenchId(String("bench_dict_iter[", size, "]")),
        )

    # Whole-dictionary operations cost O(size) per iteration rather than the
    # fixed batch of key operations the sweep above measures, so they get a
    # smaller sweep of their own to keep the benchmark's runtime bounded.
    comptime whole_sizes = (1000, 10_000)

    comptime for i in range(len(whole_sizes)):
        comptime size = rebind[Int](whole_sizes[i])
        m.bench_function(
            bench_dict_build[size],
            BenchId(String(t"bench_dict_build[{size}]")),
        )
        m.bench_function(
            bench_dict_accumulate[size],
            BenchId(String(t"bench_dict_accumulate[{size}]")),
        )
        m.bench_function(
            bench_dict_popitem_drain[size],
            BenchId(String(t"bench_dict_popitem_drain[{size}]")),
        )
        m.bench_function(
            bench_dict_update[size],
            BenchId(String(t"bench_dict_update[{size}]")),
        )
        m.bench_function(
            bench_dict_eq[size],
            BenchId(String(t"bench_dict_eq[{size}]")),
        )

    var results = Dict[String, Tuple[Float64, Int]]()
    for info in m.info_vec:
        var n = info.name
        var time = info.result.mean("ms")
        var avg, amnt = results.get(n, (Float64(0), 0))
        results[n] = (
            (avg * Float64(amnt) + time) / Float64((amnt + 1)),
            amnt + 1,
        )
    print("")
    for k_v in results.items():
        print(k_v.key, k_v.value[0], sep=",")

    comptime for i in range(len(sizes)):
        comptime size = rebind[Int](sizes[i])
        var mem_s = total_bytes_used(make_dict[size]())
        print("dict_memory_size[", size, "]: ", mem_s, sep="")
