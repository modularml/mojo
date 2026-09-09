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

# RUN: not %parse-mojo-isolated %s 2>&1 | FileCheck %s

# Regression test for MOCO-4400.

# CHECK-NOT: Failed to specialize generator
# CHECK: error: 'S[T]' does not implement all requirements for 'Iterable'


@fieldwise_init
struct S[T: AnyType](
    Copyable where conforms_to(T, Copyable),
    Iterable where conforms_to(T, Movable),
    Iterator where conforms_to(T, Movable),
    Movable,
):
    comptime IteratorType[
        mut: Bool, //, origin: Origin[mut=mut]
    ]: Iterator where conforms_to(Self.T, Movable) = Self
    comptime IteratorOwnedType: Iterator where conforms_to(
        Self.T, Movable
    ) = Self
    comptime Element: Movable where conforms_to(Self.T, Movable) = Self.T

    var _v: Self.T

    def __iter__(
        ref self,
    ) -> Self.IteratorType[origin_of(self)] where conforms_to(Self.T, Copyable):
        return self.copy()

    def __next__(
        mut self,
    ) raises StopIteration -> Self.Element where conforms_to(Self.T, Movable):
        raise StopIteration()


@fieldwise_init
struct NotMovable(Deinitable):
    var x: Int


def needs_iterator[I: Iterator](x: I):
    pass


def foo():
    var s = S(NotMovable(1))
    needs_iterator(s)
