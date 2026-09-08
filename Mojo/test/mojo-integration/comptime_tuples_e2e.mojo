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

# RUN: mojo -O0 %s | FileCheck %s

from std.utils.variant import Variant
from std.memory import ThinAllocation, alloc, dealloc


# NOTE: This struct uses Pointer-based storage instead of List because
# List requires T: Copyable at the field declaration site, which creates a
# circular dependency with Variant's conditional Copyable conformance:
#   MTuple contains List[Variant[T, MTuple[T]]]
#   -> List requires Variant[T, MTuple[T]]: Copyable
#   -> requires MTuple[T]: Copyable (conditional on Variant)
#   -> MTuple's fields include List[Variant[T, MTuple[T]]]... (cycle)
# Pointer avoids this because it doesn't constrain its element type,
# and Copyable checks in method bodies resolve after the struct is defined.
struct MTuple[T: ImplicitlyCopyable & Deinitable](ImplicitlyCopyable, Writable):
    comptime Element = Variant[Self.T, Self]
    var _data: Pointer[Self.Element, MutUntrackedOrigin]
    var _len: Int
    var _cap: Int

    @always_inline
    def __init__(out self):
        self._data = Pointer[Self.Element, MutUntrackedOrigin].unsafe_dangling()
        self._len = 0
        self._cap = 0

    @always_inline
    def __init__(out self, var value: Self.Element):
        self._cap = 4
        self._data = alloc[Self.Element]({count = self._cap}).unsafe_leak()
        self._data.unsafe_write(value^)
        self._len = 1

    def __init__(out self, *, copy: Self):
        self._len = copy._len
        self._cap = copy._len
        if copy._len > 0:
            self._data = alloc[Self.Element]({count = copy._len}).unsafe_leak()
            for i in range(copy._len):
                self._data.unsafe_offset(i).unsafe_write(
                    copy._data[unsafe_offset=i]
                )
        else:
            self._data = Pointer[
                Self.Element, MutUntrackedOrigin
            ].unsafe_dangling()

    def __deinit__(deinit self):
        for i in range(self._len):
            self._data.unsafe_offset(i).unsafe_deinit_pointee()
        if self._cap > 0:
            dealloc(
                ThinAllocation(unsafe_owned_ptr=self._data).unsafe_with_layout(
                    {count = self._cap}
                )
            )

    def _grow_if_needed(mut self):
        if self._len >= self._cap:
            var new_cap = self._cap * 2 if self._cap > 0 else 4
            var new_data = alloc[Self.Element]({count = new_cap}).unsafe_leak()
            for i in range(self._len):
                new_data.unsafe_offset(i).unsafe_write(
                    self._data.unsafe_offset(i).unsafe_take_pointee()
                )
            if self._cap > 0:
                dealloc(
                    ThinAllocation(
                        unsafe_owned_ptr=self._data
                    ).unsafe_with_layout({count = self._cap})
                )
            self._data = new_data
            self._cap = new_cap

    def _append(mut self, value: Self.Element):
        self._grow_if_needed()
        self._data.unsafe_offset(self._len).unsafe_write(value)
        self._len += 1

    @always_inline
    def cons(self, var other: Self) -> Self:
        var new = self
        for i in range(other._len):
            new._append(other._data[unsafe_offset=i])
        return new

    @always_inline
    def __add__(self, var other: Self) -> Self:
        var new = Self()
        for i in range(self._len):
            new._append(self._data[unsafe_offset=i])
        for i in range(other._len):
            new._append(other._data[unsafe_offset=i])
        return new

    def write_to(self, mut writer: Some[Writer]):
        writer.write("(")

        for i in range(self._len):
            if self._data[unsafe_offset=i].isa[Int]():
                var value = self._data[unsafe_offset=i]
                writer.write(value[Int])
            elif self._data[unsafe_offset=i].isa[MTuple[Self.T]]():
                var value = self._data[unsafe_offset=i]
                writer.write(value[MTuple[Self.T]])
            else:
                writer.write("?")
            if i < self._len - 1:
                writer.write(", ")

        writer.write(")")


comptime IntTuple = MTuple[Int]


def main():
    comptime tup = IntTuple(IntTuple(3) + IntTuple(4))
    # CHECK: (3, 4)
    print(tup)
    add_print[tup]()


def add_print[x: IntTuple]():
    comptime tup = x + IntTuple(4)
    # CHECK: ((3, 4), 4)
    print(tup)
