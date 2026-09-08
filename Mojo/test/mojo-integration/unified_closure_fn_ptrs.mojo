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
# RUN: %mojo %s 3 4 | FileCheck %s

from std.sys import argv


def top_level(x: Int) -> Int:
    return x


def takeIt[T: def(Int) -> Int](cb: T, x: Int):
    print(cb(x))


@fieldwise_init
struct HasParam[N: Int](ImplicitlyCopyable):
    comptime rank = Self.N + Self.N

    def printMe(self):
        print(Self.rank)


def consume[
    c: Int, r: Int, FuncType: def(a: HasParam[c]) -> HasParam[r]
](impl: FuncType):
    var p = impl(HasParam[c]())
    p.printMe()


def foo[a: Int, b: Int, c: Int]():
    def nested(a: HasParam[a + b]) -> HasParam[b + c]:
        return HasParam[b + c]()

    comptime parameter_domain = nested(HasParam[a + b]())
    materialize[parameter_domain]().printMe()
    consume[a + b, b + c, type_of(nested)](nested)


def main() raises:
    var x = atol(argv()[1])
    var y = atol(argv()[2])

    # CHECK: 3
    takeIt(top_level, x)

    # CHECK: 4
    takeIt(top_level, y)

    # CHECK: 10
    # CHECK: 10
    foo[1, 2, 3]()
