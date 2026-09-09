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

# Regression test for MOCO-4671: binding `T` to a type an enclosing `where`
# clause refined past the `AnyType` floor of `Ts` yields `T(Movable)`, and
# `Self.Ts.contains[T]()` has to see through that downcast to hold.

from std.testing import assert_true


struct MyVariant[*Ts: AnyType]:
    @staticmethod
    def put[T: Movable & Deinitable]() -> Bool where Self.Ts.contains[T]():
        return True


def refined[T: AnyType]() -> Bool where conforms_to(T, Movable & Deinitable):
    return MyVariant[T].put[T]()


def main() raises:
    assert_true(refined[Int]())
