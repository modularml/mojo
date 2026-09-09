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
# RUN: %mojo %s | FileCheck %s

from std.memory import ThinAllocation, dealloc
from std.sys import llvm_intrinsic, size_of


def memcpy(
    dst: Pointer[mut=True, Int, _],
    src: Pointer[Int, _],
    count: Int,
):
    var byte_count = count * size_of[Int]()

    if __mlir_op.`kgen.is_run_in_comptime_interpreter`[
        _type=__mlir_type.`!kgen.scalar<bool>`
    ]():
        llvm_intrinsic["llvm.memcpy", NoneType](
            dst.unsafe_bitcast[Byte](),
            src.unsafe_bitcast[Byte](),
            byte_count.__mlir_index__(),
        )
    else:
        # Intentionally mis-match behavior between then and else
        # here for testing.
        pass


struct Data(ImplicitlyCopyable, Writable):
    var _data: Pointer[Int, MutUntrackedOrigin]
    var _size: Int

    def __init__(out self, *, size: Int):
        self._data = alloc[Int]({count = size}).unsafe_leak()
        self._size = size
        for i in range(size):
            self._data[unsafe_offset=i] = 0

    def __init__(out self, *data: Int):
        var num_elems = len(data)
        self._data = alloc[Int]({count = num_elems}).unsafe_leak()
        self._size = num_elems
        for i in range(num_elems):
            self._data[unsafe_offset=i] = data[i]

    def __init__(out self, *, copy: Self):
        self._size = copy._size
        self._data = alloc[Int]({count = self._size}).unsafe_leak()
        for i in range(self._size):
            self._data[unsafe_offset=i] = copy._data[unsafe_offset=i]

    def write_to(self, mut writer: Some[Writer]):
        for i in range(self._size):
            t"data[{i}] = {self._data[unsafe_offset=i]}\n".write_to(writer)

    def __add__(self, rhs: Self) -> Self:
        var size = self._size + rhs._size
        var result = Self(size=size)
        memcpy(result._data, self._data, self._size)
        memcpy(result._data.unsafe_offset(self._size), rhs._data, rhs._size)
        return result

    def __deinit__(deinit self):
        dealloc(
            ThinAllocation(unsafe_owned_ptr=self._data).unsafe_with_layout(
                {count = self._size}
            )
        )


def main():
    comptime d1 = Data(4, 2)
    comptime d2 = Data(2, 8)
    comptime d3 = d1 + d2
    var d4 = d1 + d2

    # CHECK: data[0] = 4
    # CHECK: data[1] = 2
    # CHECK: data[2] = 2
    # CHECK: data[3] = 8
    print(String(d3))

    # CHECK: data[0] = 0
    # CHECK: data[1] = 0
    # CHECK: data[2] = 0
    # CHECK: data[3] = 0
    print(String(d4))
