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
"""Tests for by-reference iteration over move-only types."""

from std.iter import StopIteration
from std.memory import Allocation, alloc, dealloc
from std.testing import TestSuite, assert_equal
from test_utils import ExplicitCopyOnly


# ===-----------------------------------------------------------------------===#
# Test types
# ===-----------------------------------------------------------------------===#


struct MoveOnlyInt(Movable):
    """A move-only integer wrapper."""

    var value: Int

    def __init__(out self, value: Int):
        self.value = value

    def __init__(out self, *, other: Self):
        self.value = other.value


# ===-----------------------------------------------------------------------===#
# Ref iterator
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _RefIter[
    mut: Bool,
    //,
    T: Movable,
    origin: Origin[mut=mut],
    forward: Bool = True,
](ImplicitlyCopyable):
    """A ref-returning iterator that works with move-only types."""

    var index: Int
    var data: Pointer[Self.T, Self.origin]
    var length: Int

    def __iter__(ref self) -> Self:
        return self.copy()

    def __next__(
        mut self,
    ) raises StopIteration -> ref[Self.origin] Self.T:
        comptime if Self.forward:
            if self.index >= self.length:
                raise StopIteration()
            self.index += 1
            return self.data[unsafe_offset=self.index - 1]
        else:
            if self.index <= 0:
                raise StopIteration()
            self.index -= 1
            return self.data[unsafe_offset=self.index]


# ===-----------------------------------------------------------------------===#
# A simple collection using the ref iterator
# ===-----------------------------------------------------------------------===#


struct MoveOnlyList[T: Movable & Deinitable]:
    """A simple list that holds move-only types."""

    var _data: Pointer[Self.T, MutUntrackedOrigin]
    var _len: Int
    var _capacity: Int

    def __init__(out self):
        self._data = Pointer[Self.T, MutUntrackedOrigin].unsafe_dangling()
        self._len = 0
        self._capacity = 0

    def __deinit__(deinit self):
        for i in range(self._len):
            self._data.unsafe_offset(i).unsafe_deinit_pointee()
        if self._capacity > 0:
            dealloc(
                Allocation(
                    unsafe_owned_ptr=self._data, layout={count = self._capacity}
                )
            )

    def __len__(self) -> Int:
        return self._len

    def append(mut self, var value: Self.T):
        if self._len >= self._capacity:
            var new_cap = self._capacity * 2 if self._capacity > 0 else 4
            var new_data: Pointer[Self.T, MutUntrackedOrigin] = alloc[Self.T](
                {count = new_cap}
            ).unsafe_leak()
            for i in range(self._len):
                new_data.unsafe_offset(i).unsafe_write(
                    self._data.unsafe_offset(i).unsafe_take_pointee()
                )
            if self._capacity > 0:
                dealloc(
                    Allocation(
                        unsafe_owned_ptr=self._data,
                        layout={count = self._capacity},
                    )
                )
            self._data = new_data
            self._capacity = new_cap
        self._data.unsafe_offset(self._len).unsafe_write(value^)
        self._len += 1

    def __getitem__(ref self, idx: Int) -> ref[self] Self.T:
        return self._data[unsafe_offset=idx]

    def unsafe_ptr[
        origin: Origin, //
    ](ref[origin] self) -> Pointer[Self.T, origin]:
        return self._data.unsafe_mut_cast[origin.mut]().unsafe_origin_cast[
            origin
        ]()

    def __iter__(ref self) -> _RefIter[Self.T, origin_of(self)]:
        return _RefIter(index=0, data=self.unsafe_ptr(), length=self._len)

    def __reversed__(
        ref self,
    ) -> _RefIter[Self.T, origin_of(self), forward=False]:
        return _RefIter[forward=False](
            index=self._len, data=self.unsafe_ptr(), length=self._len
        )


# ===-----------------------------------------------------------------------===#
# Tests
# ===-----------------------------------------------------------------------===#


def test_ref_iteration_implicitly_copyable_type() raises:
    var list = MoveOnlyList[Int]()
    list.append(10)
    list.append(20)
    list.append(30)

    var sum = 0
    for x in list:
        sum += x
    assert_equal(sum, 60)


def test_ref_iteration_copyable_type() raises:
    var list = MoveOnlyList[ExplicitCopyOnly]()
    list.append(ExplicitCopyOnly(10))
    list.append(ExplicitCopyOnly(20))
    list.append(ExplicitCopyOnly(30))

    var sum = 0
    for x in list:
        sum += x.value
    assert_equal(sum, 60)


def test_ref_iteration_move_only_type() raises:
    var list = MoveOnlyList[MoveOnlyInt]()
    list.append(MoveOnlyInt(100))
    list.append(MoveOnlyInt(200))
    list.append(MoveOnlyInt(300))

    var sum = 0
    for x in list:
        sum += x.value
    assert_equal(sum, 600)


def test_mutable_ref_iteration() raises:
    var list = MoveOnlyList[Int]()
    list.append(1)
    list.append(2)
    list.append(3)

    for ref x in list:
        x += 10

    var sum = 0
    for x in list:
        sum += x
    assert_equal(sum, 36)


def test_mutable_ref_iteration_move_only_type() raises:
    var list = MoveOnlyList[MoveOnlyInt]()
    list.append(MoveOnlyInt(100))
    list.append(MoveOnlyInt(200))
    list.append(MoveOnlyInt(300))

    for ref x in list:
        x.value += 1

    var sum = 0
    for x in list:
        sum += x.value
    assert_equal(sum, 603)


def test_reversed_ref_iteration() raises:
    var list = MoveOnlyList[Int]()
    list.append(1)
    list.append(2)
    list.append(3)

    var result = List[Int]()
    for x in list.__reversed__():
        result.append(x)

    assert_equal(len(result), 3)
    assert_equal(result[0], 3)
    assert_equal(result[1], 2)
    assert_equal(result[2], 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
