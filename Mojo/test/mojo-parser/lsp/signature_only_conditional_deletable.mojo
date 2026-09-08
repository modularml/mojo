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

# ===----------------------------------------------------------------------=== #
#
# Destructor discharge for a conditionally `Deinitable` imported
# generic type must work under LSP mode, where the imported struct's body is
# left unresolved.
#
# ===----------------------------------------------------------------------=== #

# RUN: %parse-mojo-isolated -lsp -I %S/inputs %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -verify-diagnostics
# RUN: %parse-mojo-isolated -I %S/inputs %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -verify-diagnostics

from conditional_helper import ConditionalHelper


struct PlainThing(Movable):
    var x: Int

    def __init__(out self, x: Int):
        self.x = x


struct NeverDeletable(Deinitable where False, Movable):
    var x: Int

    def __init__(out self, x: Int):
        self.x = x


# `PlainThing` is (implicitly) `Deinitable`, so `h` should be
# destroyed with no diagnostic.
def use_conforming(var h: ConditionalHelper[PlainThing]):
    pass


# `NeverDeletable` opts out of `Deinitable`, so `ConditionalHelper`
# genuinely cannot destroy `h` implicitly.
# expected-error @below {{'h' abandoned without being explicitly destroyed}}
def use_non_conforming(var h: ConditionalHelper[NeverDeletable]):
    pass


# The enclosing `where` bound discharges `ConditionalHelper[T]`'s conditional
# conformance from context, so `h` is destroyed with no diagnostic.
def use_generic_conforming[
    T: Movable
](var h: ConditionalHelper[T]) where conforms_to(T, Deinitable):
    pass


# Without the bound there is no assumption to satisfy `ConditionalHelper[T]`'s
# conformance condition for an arbitrary `T`, so `h` is conditionally linear and
# cannot be destroyed implicitly.
# expected-error @below {{'h' abandoned without being explicitly destroyed}}
def use_generic_unconstrained[T: Movable](var h: ConditionalHelper[T]):
    pass


struct CondDeletable[flag: Bool](Deinitable where flag, Movable):
    var x: Int

    def __init__(out self, x: Int):
        self.x = x


# `CondDeletable[True]` conforms (`where flag`, flag=True), so
# `ConditionalHelper` discharges its conformance and `h` is destroyed cleanly.
def use_cond_true(var h: ConditionalHelper[CondDeletable[True]]):
    pass


# `CondDeletable[False]` does not conform (`where flag`, flag=False), so `h` is
# conditionally linear and cannot be destroyed implicitly.
# expected-error @below {{'h' abandoned without being explicitly destroyed}}
def use_cond_false(var h: ConditionalHelper[CondDeletable[False]]):
    pass


# The enclosing `where flag` bound proves `CondDeletable[flag]` deletable, which
# discharges `ConditionalHelper`'s conformance from context; no diagnostic.
def use_generic_cond[
    flag: Bool
](var h: ConditionalHelper[CondDeletable[flag]]) where flag:
    pass


# Without the `where flag` bound, `CondDeletable[flag]` is not provably
# deletable for an arbitrary `flag`, so `h` is conditionally linear.
# expected-error@+3 {{'h' abandoned without being explicitly destroyed}}
def use_generic_cond_unconstrained[
    flag: Bool
](var h: ConditionalHelper[CondDeletable[flag]]):
    pass
