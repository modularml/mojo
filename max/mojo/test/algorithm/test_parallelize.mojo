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

from std.math import ceildiv
from std.sys.info import num_physical_cores

from std.algorithm import (
    map,
)

from max.algorithm import (
    parallelize,
    sync_parallelize,
    parallelize_over_rows,
)
from std.testing import assert_equal, assert_false, TestSuite
from std.utils import IndexList


def test_sync_parallelize() raises:
    var num_work_items = 4

    var vector_stack = Array[Int, 20](fill={})
    var vector = Span(vector_stack)

    for i in range(len(vector)):
        vector[i] = Int(i)

    var chunk_size = ceildiv(len(vector), num_work_items)

    @always_inline
    def parallel_fn(thread_id: Int) {var vector, var chunk_size}:
        var start = thread_id * chunk_size
        var end = min(start + chunk_size, len(vector))

        @always_inline
        def add_two(idx: Int) {var}:
            vector[start + idx] = vector[start + idx] + 2

        map(end - start, add_two)

    sync_parallelize(parallel_fn, num_work_items)

    for i in range(len(vector)):
        assert_equal(vector[i], Int(i + 2))


def test_parallelize() raises:
    var num_work_items = num_physical_cores()

    var vector_stack = Array[Int, 20](fill={})
    var vector = Span(vector_stack)

    for i in range(len(vector)):
        vector[i] = Int(i)

    var chunk_size = ceildiv(len(vector), num_work_items)

    @always_inline
    def parallel_fn(thread_id: Int) {var vector, var chunk_size}:
        var start = thread_id * chunk_size
        var end = min(start + chunk_size, len(vector))

        @always_inline
        def add_two(idx: Int) {var}:
            vector[start + idx] = vector[start + idx] + 2

        map(end - start, add_two)

    parallelize(parallel_fn, num_work_items)


def printme(i: Int):
    print(i, end="")


def test_parallelize_no_workers() raises:
    # With ASSERT=warn, this prints a warning but doesn't abort.
    parallelize(printme, 10, 0)


def test_parallelize_negative_workers() raises:
    # With ASSERT=warn, this prints a warning but doesn't abort.
    parallelize(printme, 10, -1)


def test_parallelize_negative_work() raises:
    # This should do nothing
    parallelize(printme, -1, 4)


def test_parallelize_over_rows_zero_work() raises:
    # This should do nothing
    def noop(start: Int, end: Int) {}:
        pass

    parallelize_over_rows(noop, IndexList[1](0), 0, 1)


def test_parallelize_unified() raises:
    var num_work_items = num_physical_cores()

    var vector_stack = Array[Int, 20](fill={})
    var vector = Span(vector_stack)

    for i in range(len(vector)):
        vector[i] = Int(i)

    var chunk_size = ceildiv(len(vector), num_work_items)

    @always_inline
    def parallel_fn(
        thread_id: Int,
    ) {imm vector, imm chunk_size,}:
        var start = thread_id * chunk_size
        var end = min(start + chunk_size, len(vector))

        for idx in range(end - start):
            vector[start + idx] = vector[start + idx] + 2

    parallelize(parallel_fn, num_work_items)

    for i in range(len(vector)):
        assert_equal(vector[i], Int(i + 2))


def test_sync_parallelize_unified() raises:
    var num_work_items = 4

    var vector_stack = Array[Int, 20](fill={})
    var vector = Span(vector_stack)

    for i in range(len(vector)):
        vector[i] = Int(i)

    var chunk_size = ceildiv(len(vector), num_work_items)

    @always_inline
    def add_two_parallel(
        thread_id: Int,
    ) {imm vector, imm chunk_size,}:
        var start = thread_id * chunk_size
        var end = min(start + chunk_size, len(vector))

        for idx in range(end - start):
            vector[start + idx] = vector[start + idx] + 2

    sync_parallelize(add_two_parallel, num_work_items)

    for i in range(len(vector)):
        assert_equal(vector[i], Int(i + 2))


def test_sync_parallelize_unified_single_item() raises:
    var result = 0

    @always_inline
    def set_result(
        i: Int,
    ) {mut result,}:
        result = 42

    sync_parallelize(set_result, 1)

    assert_equal(result, 42)


def test_sync_parallelize_unified_zero_items() raises:
    var called = False

    @always_inline
    def should_not_run(
        i: Int,
    ) {mut called,}:
        called = True

    sync_parallelize(should_not_run, 0)

    assert_false(called)


def test_parallelize_over_rows() raises:
    var shape = IndexList[2](4, 8)
    var num_rows = shape[0]
    var row_size = shape[1]

    var data_stack = Array[Int, 32](fill={})
    var data = Span(data_stack)

    for i in range(num_rows * row_size):
        data[i] = Int(0)

    def process_rows(start_row: Int, end_row: Int) {imm}:
        for row in range(start_row, end_row):
            for col in range(row_size):
                data[row * row_size + col] = Int(row + 1)

    parallelize_over_rows(process_rows, shape, 1, 1)

    for row in range(num_rows):
        for col in range(row_size):
            assert_equal(data[row * row_size + col], Int(row + 1))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
