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

# RUN: kgen-doc %s | FileCheck %s
from std.collections import List


# CHECK: "name": "foo"
# CHECK: "type": "List[Int]"
# CHECK: "signature": "def foo(x: List[Int])"
def foo(x: List[Int]):
    pass


# CHECK: "name": "bar"
# CHECK: "type": "List[Int]"
# CHECK: "signature": "def bar[x: List[Int]]()"
def bar[x: List[Int]]():
    pass


# CHECK: "name": "baz"
# CHECK: "returns": {
# CHECK:    "type": "List[Int]"
# CHECK: }
# CHECK: "signature": "def baz() -> List[Int]"
def baz() -> List[Int]:
    return List[Int]()


# CHECK: "name": "higher_order"
# CHECK: "type": "def(List[Int]) thin -> Int"
# CHECK: "signature": "def higher_order(f: def(List[Int]) thin -> Int) -> def(List[Int]) thin -> Int"
def higher_order(
    f: def(List[Int]) thin -> Int,
) -> def(List[Int]) thin -> Int:
    return f


# Self-substitution at sub-expression depth: when the self type appears as a
# parameter of an outer parametric type, the printer should still emit `Self`
# rather than the fully-expanded form.


struct Wrap[T: AnyType]:
    pass


struct SelfSubHost[N: Int]:
    # CHECK: "name": "takes_wrap_of_self"
    # CHECK: "type": "Wrap[Self]"
    # CHECK: "signature": "def takes_wrap_of_self(self, w: Wrap[Self])"
    def takes_wrap_of_self(self, w: Wrap[Self]):
        pass

    # CHECK: "name": "returns_wrap_of_self"
    # CHECK: "type": "Wrap[Self]"
    # CHECK: "signature": "def returns_wrap_of_self(self) -> Wrap[Self]"
    def returns_wrap_of_self(self) -> Wrap[Self]:
        pass
