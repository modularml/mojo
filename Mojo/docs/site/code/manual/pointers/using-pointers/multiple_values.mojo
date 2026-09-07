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
from std.memory.alloc import alloc, dealloc, Layout


def main():
    # start-alloc-multiple
    var allocation = alloc(Layout[Float64](count=6))
    var float_ptr = allocation.unsafe_ptr()
    for offset in range(6):
        float_ptr.unsafe_offset(offset).unsafe_write(0.0)
    # end-alloc-multiple

    # start-subscript-access
    float_ptr[unsafe_offset=2] = 3.0
    for offset in range(6):
        print(float_ptr[unsafe_offset=offset], end=", ")
    # end-subscript-access
    print()

    # Pointer arithmetic: offset from an existing pointer
    var first_ptr = float_ptr
    # start-pointer-offset
    var third_ptr = first_ptr.unsafe_offset(2)
    # end-pointer-offset

    print(third_ptr[])

    # Advance a pointer to the next element
    var ptr = float_ptr
    # start-pointer-advance
    # Advance the pointer one element:
    ptr = ptr.unsafe_offset(1)
    # end-pointer-advance

    print(ptr[])

    for offset in range(6):
        float_ptr.unsafe_offset(offset).unsafe_deinit_pointee()
    dealloc(allocation^)
