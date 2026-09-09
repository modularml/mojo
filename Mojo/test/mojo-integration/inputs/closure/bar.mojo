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


@no_inline
def takeIt[F: def[width: Int](idx: Int) -> Int](impl: F):
    print(impl.__call__[1](0))


def emitLoad(x: Int):
    var ptr = alloc[Int]({count = 1}).unsafe_leak()
    ptr.store(x)
    var count = Int(0)

    @no_inline
    def foo[width: Int](idx: Int) {mut count, imm ptr} -> Int:
        var vec = ptr.load[width=width](idx).cast[.int]()
        count = count + rebind[type_of(count)](vec)
        return count

    takeIt(foo)
