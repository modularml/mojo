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
from std.testing import assert_equal


def inline_allocation() raises:
    # Include following line uncommented
    # from std.memory.alloc import alloc, dealloc, Layout
    var allocation = alloc(Layout[Int](count=4))
    # Use allocation
    var ptr = allocation.unsafe_ptr()
    for i in range(4):
        ptr.unsafe_offset(i).unsafe_write(i)
    # Omit next line
    var tmp = ptr[unsafe_offset=3]
    # Release allocation
    dealloc(allocation^)
    # Delay assert until after allocation is released
    assert_equal(tmp, 3)


def raising_function(ptr: Pointer[Float64, _]) raises:
    pass


def allocating_function() raises:
    var data = alloc[Float64]({count = 64})
    # ...
    try:
        raising_function(data.unsafe_ptr())
    except e:
        dealloc(data^)
        raise e^  # propagate the error
    dealloc(data^)


def leaky_function() raises:
    var data_ptr = alloc[Float64]({count = 64}).unsafe_leak()
    # ...
    raising_function(data_ptr)
    dealloc(
        ThinAllocation(unsafe_owned_ptr=data_ptr).unsafe_with_layout(
            {count = 64}
        )
    )


struct Counter:
    comptime _layout = Layout[Int].single()
    var _alloc: ThinAllocation[Int]

    def __init__(out self, value: Int):
        self._alloc = alloc(Self._layout).into_thin()
        self._alloc.unsafe_ptr().unsafe_write(value)

    def increment(mut self):
        self._alloc.unsafe_ptr()[] += 1

    def get(self) -> Int:
        return self._alloc.unsafe_ptr()[]

    def __deinit__(deinit self):
        # Convert ThinAllocation back into Allocation
        dealloc(self._alloc^.unsafe_with_layout(Self._layout))


def use_counter() raises:
    var c = Counter(0)
    c.increment()
    c.increment()
    assert_equal(c.get(), 2)


def main() raises:
    inline_allocation()
    use_counter()
    allocating_function()
    leaky_function()
