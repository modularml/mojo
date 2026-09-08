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

# RUN: %parse-mojo-isolated %s | FileCheck %s

# Regression test: capturing a value whose type mentions a trait-associated
# parametric alias (like `IteratorType` below) must not crash.
#
# `IteratorType`'s own declaration is self-referential: `origin`'s type has to
# name `mut` positionally within that same declaration, since it depends on it.
# That positional reference is only meaningful while reading `IteratorType`'s
# declaration in isolation; it doesn't name anything once lifted out into a
# different scope, such as a closure's capture list.
#
# `scanForOrigins` used to walk into that positional reference and report it as
# one of the closure's real captured origins. A later substitution pass then
# treated it as if it referred to one of the *closure's own* parameters and
# indexed out of bounds.


trait Iterable:
    comptime IteratorType[mut: Bool, //, origin: Origin[mut=mut]]: AnyType

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        ...


# CHECK-LABEL: lit.fn @"outer{{.*}}"
def outer[T: Iterable](ref x: T):
    var it = x.__iter__()

    # The closure's capture set must be its actual captures, not an abstract
    # reference lifted out of `IteratorType`.
    # CHECK: lit.fn *"capture_it{{.*}}":{(*"x_is_mut`") *"x_is_origin`1", mut *"it`{{[0-9]+}}"}:
    @__parameter
    def capture_it():
        _ = it

    capture_it()
