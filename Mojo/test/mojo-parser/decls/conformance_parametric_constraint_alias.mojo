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

# Discharging parametric `where` clauses: witness selection against a
# trait-alias bound and against a same-named method that stays partially bound,
# and a clause that can be neither proven nor refuted at a conversion site.

# RUN: %parse-mojo-isolated -verify-diagnostics %s


# A same-named `__init__` that stays partially bound, with a parametric `where`
# over a dependent member alias, must not be the `Copyable` witness. Checking
# that clause must not require a specialization with an unbound hole in a nested
# `bind_params`.
struct Wrapper[T: Movable, /](
    Movable,
    Copyable where conforms_to(T, Copyable),
):
    # Cannot be the `Copyable` witness: the requirement is
    # `__init__(out self, *, copy: Self)`, which offers nothing to infer
    # `IterableType` or the origin of `iterable` from. Its `where` clause stays
    # parametric, and names `IteratorType[origin_of(iterable)]` -- whose own
    # parameter list is a dependent chain (mutability, then an origin whose type
    # mentions it).
    def __init__[
        IterableType: Iterable,
    ](
        ref iterable: IterableType,
        out self: Wrapper[IterableType.IteratorType[origin_of(iterable)].Element],
    ) where conforms_to(
        IterableType.IteratorType[origin_of(iterable)].Element, Copyable
    ):
        pass

    # The actual witness: provable directly from the conformance constraint.
    def __init__(out self, *, copy: Self) where conforms_to(Self.T, Copyable):
        pass


# A parameter bound through a parametric trait alias (`F: _fn_trait[dtype]`)
# contributes `identical(F.dtype, dtype)` on the struct. Generated lifecycle
# methods inherit that constraint as a decl-ref, so witness selection must
# assume the remapped struct-parameter facts -- an index-ref never matches
# the decl-ref naming the same parameter.
comptime _fn_trait[dtype: DType] = ImplicitlyCopyable & Deinitable & (
    def[w: Int]() -> SIMD[dtype, w]
)


@fieldwise_init
struct HoldsFn[dtype: DType, F: _fn_trait[dtype]](
    Deinitable, ImplicitlyCopyable
):
    var f: Self.F


# Both kinds of struct-scope assumption at once: an explicit `where` clause and
# a parametric-trait-alias capture constraint. Witness selection needs the
# capture constraint, and the two reach the struct's scope by different routes,
# so both have to survive.
@fieldwise_init
struct HoldsFnBounded[n: Int, dtype: DType, F: _fn_trait[dtype]](
    Deinitable, ImplicitlyCopyable
) where n > 0:
    var f: Self.F


# A trait method with a *default* implementation, overridden by a struct method
# whose `where` clause follows from the conformance. The default stub must be
# disabled and the struct's method chosen; leaving the stub in place makes both
# candidates viable and the reference ambiguous.
trait DefaultedGreeter:
    def greet(self) -> Int:
        return 0


struct ConditionalGreeter[T: Movable](
    Movable, DefaultedGreeter where conforms_to(T, Copyable)
):
    def greet(self) -> Int where conforms_to(Self.T, Copyable):
        return 1


# An `@implicit` constructor whose `where` clause is unprovable in the calling
# scope leaves the conversion undecided rather than impossible. Recording that
# as a definitive "not convertible" for the `(Int, Constrained[m])` pair
# poisons the query below, which asks about the same pair from a scope that can
# prove the clause.
struct Constrained[n: Int]:
    var v: Int

    @implicit
    def __init__(out self, x: Int) where Self.n > 0:
        self.v = x


# expected-note @below {{function declared here}}
def takes_constrained[n: Int](imm s: Constrained[n]):
    pass


def unprovable_at_call[m: Int]():
    # expected-error @below {{invalid call to 'takes_constrained': value passed to 's' cannot be converted from 'Int' to 'Constrained[m]'}}
    takes_constrained[m](Int(1))


def provable_at_call[m: Int]() where m > 0:
    takes_constrained[m](Int(1))
