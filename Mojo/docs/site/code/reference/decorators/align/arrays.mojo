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

from std.sys import align_of, size_of
from std.memory import alloc, dealloc
from std.testing import *


@align(64)
struct CacheAligned:
    var data: Int  # 8 bytes


def demonstrate_array_stride() raises:
    var allocation = alloc[CacheAligned]({count = 4})
    var _ = allocation.unsafe_ptr()  # `arr`

    # print(align_of[CacheAligned]())  # 64
    # print(size_of[CacheAligned]())  # 64

    var align = align_of[CacheAligned]()
    var size = size_of[CacheAligned]()

    # All elements of arr are guaranteed to be 64-byte aligned

    dealloc(allocation^)
    assert_equal(64, align, "align should be 64")
    assert_equal(64, size, "size_of should be 64")


def main() raises:
    demonstrate_array_stride()
