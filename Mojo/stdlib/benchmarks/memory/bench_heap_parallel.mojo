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
"""Parallel heap microbenchmark using ``alloc`` / ``free`` and ``parallelize``.

Each timed iteration runs one parallel task per physical core (via
``num_physical_cores()``). Every task performs many heap allocations. For each
slab, the task **writes every element**, then **reads every element** into a
running XOR fold (so the full footprint is touched twice and the allocation
cannot be eliminated). Slabs are freed immediately after. Task checksums merge
via ``Atomic.fetch_add``, then ``keep`` preserves the final value.

Intended to stress multithreaded heap **placement**, **first-touch**, and
**sequential access bandwidth** (e.g. NUMA-local vs remote DRAM), not just
malloc/free metadata.
"""

from max.algorithm import parallelize
from std.benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    Unit,
    black_box,
    keep,
)
from std.math import ceildiv
from std.memory import Allocation, alloc, dealloc
from std.atomic import Atomic
from std.sys.info import num_physical_cores

# Elements per allocated slab; ``black_box`` keeps the count opaque.
comptime ELEMENTS_PER_ALLOC = 4096
# Total heap allocations per timed ``call_fn`` (split across parallel tasks).
comptime ALLOCS_PER_ITER = 2048


def bench_heap_alloc_parallel(mut b: Bencher) raises:
    @always_inline
    def call_fn():
        var num_tasks = num_physical_cores()
        if num_tasks < 1:
            num_tasks = 1

        var per_task = ceildiv(ALLOCS_PER_ITER, num_tasks)
        var checksum = Atomic[Int64](0)

        @always_inline
        def task_body(
            task_id: Int,
        ) {mut checksum, imm per_task,}:
            var acc = Int64(0)
            var j = 0
            while j < per_task:
                var n_elems = Int(black_box(ELEMENTS_PER_ALLOC))
                if n_elems < 1:
                    n_elems = 1

                var allocation = alloc[Int]({count = n_elems}).into_managed()
                var p = allocation.unsafe_ptr()
                var seed = Int(black_box(task_id + j))

                var k = 0
                while k < n_elems:
                    p[unsafe_offset=k] = seed ^ k
                    k += 1

                var fold = Int64(0)
                k = 0
                while k < n_elems:
                    fold ^= Int64(p[unsafe_offset=k])
                    k += 1
                acc += fold

                j += 1

            _ = checksum.fetch_add(acc)

        parallelize(task_body, num_tasks)
        keep(checksum.load())

    b.iter(call_fn)


def main() raises:
    var m = Bench(BenchConfig())
    m.bench_function(
        bench_heap_alloc_parallel, BenchId("bench_heap_alloc_parallel")
    )
    m.dump_report()
