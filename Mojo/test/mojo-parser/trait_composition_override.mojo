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

# Regression test for MOCO-3332: trait composition should accept valid code
# where a child trait re-declares a parent's abstract requirement and both
# traits appear in a composition constraint (e.g. struct S[T: A & B]() where
# B(A)).

# RUN: %parse-mojo-isolated %s --mojo-disable-builtins | FileCheck %s


trait A:
    def foo(self):
        ...


trait B(A):
    def foo(self):
        ...


# CHECK: lit.struct.decl @S<T: !A_B>
struct S[T: A & B]():
    var _value: Self.T

    def test_foo(self):
        # This should not trigger ambiguous call.
        self._value.foo()
