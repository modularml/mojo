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

from debug_test_utils import keep_alive


struct MyPair:
    var first: Int
    var second: Int

    # Make the struct go thru SROA by inlining its init.
    @always_inline("nodebug")
    def __init__(out self, first: Int, second: Int):
        self.first = first
        self.second = second


struct MyPairPair:
    var first: MyPair
    var second: MyPair

    @always_inline("nodebug")
    def __init__(out self, a: Int, b: Int, c: Int, d: Int):
        self.first = MyPair(a, b)
        self.second = MyPair(c, d)


def use_address(ptr: Pointer[Int, _]):
    print(ptr[])


def main():
    var p = MyPair(1, 2)
    print(p.first, p.second)  # breakpoint
    p.first = 3
    p.second = 4
    print(p.first, p.second)  # breakpoint
    use_address(Pointer(to=p.first))

    var pp = MyPairPair(5, 6, 7, 8)
    print(pp.second.first)  # breakpoint

    keep_alive(p, pp)
