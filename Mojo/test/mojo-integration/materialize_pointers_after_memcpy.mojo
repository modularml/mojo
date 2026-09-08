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

# RUN: %mojo %s

# A comptime value whose pointer fields were written by a raw byte copy still
# materializes with usable pointers. The interpreter tracks which bytes of a
# blob hold pointers so that materialization can rebase them; a copy out of
# non-heap storage into heap storage has to carry that tracking with it, or the
# pointers reach the binary as raw interpreter addresses and crash on use.

from std.memory import unsafe_uninit_copy_n
from std.testing import assert_true


def copied_by_memcpy() -> List[StaticString]:
    """Copies pointer-bearing values out of stack storage into heap storage."""
    var out = List[StaticString]()
    for _ in range(4):
        out.append("000")
    # `StaticString` is trivially copyable, so this lowers to one memcpy whose
    # source is the stack-resident array and whose destination is the list's
    # heap buffer.
    var words: Array[StaticString, 4] = ["905", "906", "907", "908"]
    unsafe_uninit_copy_n[overlapping=False](
        dest=out.unsafe_ptr(), src=words.unsafe_ptr(), count=4
    )
    return out^


# `Array` only routes its copy through a memcpy once the array is larger than
# `TRIVIAL_FAST_PATH_MAX_BYTES` (1024); below that it copies field-wise and
# never reaches the path above. 65 sixteen-byte elements clears that, so this
# reaches the same code through ordinary user-level code rather than an explicit
# `unsafe_uninit_copy_n` call.
comptime WIDE = 65


struct Wide(Copyable, Movable):
    var words: Array[StaticString, WIDE]

    def __init__(out self, words: Array[StaticString, WIDE]):
        self.words = words.copy()


def nested_in_list_element() -> List[Wide]:
    var out = List[Wide]()
    out.append(Wide(Array[StaticString, WIDE](fill="905")))
    out.append(Wide(Array[StaticString, WIDE](fill="906")))
    out.append(Wide(Array[StaticString, WIDE](fill="907")))
    return out^


comptime COPIED: List[StaticString] = copied_by_memcpy()
comptime NESTED: List[Wide] = nested_in_list_element()


def main() raises:
    var copied = materialize[COPIED]()
    assert_true(copied[0] == "905", "first value copied by memcpy")
    assert_true(copied[1] == "906", "second value copied by memcpy")
    assert_true(copied[2] == "907", "third value copied by memcpy")
    assert_true(copied[3] == "908", "fourth value copied by memcpy")

    var nested = materialize[NESTED]()
    assert_true(nested[0].words[0] == "905", "first list element")
    assert_true(nested[1].words[0] == "906", "second list element")
    assert_true(nested[2].words[0] == "907", "third list element")
