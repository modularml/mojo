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
from std.memory.alloc import alloc, dealloc, ThinAllocation, Layout


struct ParameterizedArray[T: Copyable & Deinitable](
    Writable where conforms_to(T, Writable)
):
    var _data: ThinAllocation[Self.T]
    var _size: Int

    def __init__(out self, var *elements: Self.T):
        self._size = len(elements)
        self._data = alloc[Self.T]({count = self._size}).into_thin()
        var ptr = self._data.unsafe_ptr()
        for i in range(self._size):
            ptr.unsafe_offset(i).unsafe_write(elements[i].copy())

    def __init__(out self, *, count: Int, value: Self.T):
        self._size = count
        self._data = alloc[Self.T]({count = count}).into_thin()
        var ptr = self._data.unsafe_ptr()
        for i in range(self._size):
            ptr.unsafe_offset(i).unsafe_write(copy=value)

    def __deinit__(deinit self):
        var ptr = self._data.unsafe_ptr()
        for i in range(self._size):
            ptr.unsafe_offset(i).unsafe_deinit_pointee()
        dealloc(self._data^.unsafe_with_layout({count = self._size}))

    def __getitem__(self, i: Int) raises -> ref[self] Self.T:
        if i < self._size:
            return self._data.unsafe_ptr().unsafe_origin_cast[
                origin_of(self)
            ]()[unsafe_offset=i]
        else:
            raise Error("Out of bounds")

    def write_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.T, Writable):
        writer.write("[")
        var ptr = self._data.unsafe_ptr()
        for i in range(self._size):
            writer.write(ptr[unsafe_offset=i])
            if i < self._size - 1:
                writer.write(", ")
        writer.write("]")

    @staticmethod
    def splat(count: Int, value: Self.T) -> Self:
        # Create a new array with count instances of the given value
        return Self(count=count, value=value)


def main() raises:
    # start-usage
    var array = ParameterizedArray(1, 2, 3)
    print(array)
    # end-usage

    # start-splat-usage
    var float_array = ParameterizedArray[Float64].splat(8, 0)
    print(float_array)
    # end-splat-usage
