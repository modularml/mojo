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

# Test parsing of the `not Trait` entry in a struct's conformance list. It is
# surface syntax for `Trait where False` (see `never_conforming_no_synthesis.mojo`
# for the shared semantics matrix): the slot's condition is trivially false, so
# it is erased from the struct's canonical trait and no witness is owed.
#
# `not` is a hard keyword, so it cannot begin a trait name and needs no
# lookahead to tell apart from a trait expression.
#
# The load-bearing check here is `NotSpelling` / `WhereFalseSpelling`: both
# print the same canonical trait, which is what makes the two spellings
# interchangeable rather than merely similar.

# RUN: %parse-mojo-isolated %s | FileCheck %s


trait Base:
    pass


trait Refines(Base):
    pass


trait Other:
    pass


##===----------------------------------------------------------------------===##
# The two spellings produce the same canonical trait
##===----------------------------------------------------------------------===##
# CHECK-LABEL: lit.struct.decl @NotSpelling
# CHECK-SAME: (!AnyType_Deinitable)
struct NotSpelling(not Movable):
    var x: Int


# CHECK-LABEL: lit.struct.decl @WhereFalseSpelling
# CHECK-SAME: (!AnyType_Deinitable)
struct WhereFalseSpelling(Movable where False):
    var x: Int


##===----------------------------------------------------------------------===##
# Opting out of the other two injected traits
##===----------------------------------------------------------------------===##
# `Deinitable` is injected unconditionally, so opting out makes the struct
# linear -- the same outcome the `where False` spelling reaches.
# CHECK-LABEL: lit.struct.decl @NotDeinitable
# CHECK-SAME: does not conform to 'Deinitable' and must be explicitly destroyed
struct NotDeinitable(not Deinitable):
    var x: Int


# `Copyable` is not injected, so opting out of it is a no-op on the canonical
# trait -- it just states the default.
# CHECK-LABEL: lit.struct.decl @NotCopyable
# CHECK-SAME: (!AnyType_Deinitable_Movable)
struct NotCopyable(not Copyable):
    var x: Int


##===----------------------------------------------------------------------===##
# A `not` entry mixes freely with plain and `where`-constrained entries
##===----------------------------------------------------------------------===##
# CHECK-LABEL: lit.struct.decl @MixedList
# CHECK-SAME: (!constrained_Other_AnyType_Deinitable)
struct MixedList[T: AnyType](
    Other,
    not Movable,
    Deinitable where conforms_to(T, Deinitable),
):
    var x: Int


##===----------------------------------------------------------------------===##
# `not` applies to every symbol of a trait composition
##===----------------------------------------------------------------------===##
# Both `Base` and `Other` are dropped, leaving only the injected traits.
# CHECK-LABEL: lit.struct.decl @NotComposition
# CHECK-SAME: (!AnyType_Deinitable_Movable)
struct NotComposition(not (Base & Other)):
    var x: Int


##===----------------------------------------------------------------------===##
# Opting out of a derived trait keeps its ancestor
##===----------------------------------------------------------------------===##
# `Refines` refines `Base`, and `False` implies anything, so listing `Base`
# unconditionally alongside `not Refines` is consistent, not contradictory.
# CHECK-LABEL: lit.struct.decl @NotDerivedKeepsAncestor
# CHECK-SAME: (!Base_AnyType_Deinitable_Movable)
struct NotDerivedKeepsAncestor(not Refines, Base):
    var x: Int


##===----------------------------------------------------------------------===##
# `else "<reason>"` records why the trait is opted out
##===----------------------------------------------------------------------===##
# The reason lands on the synthesized `Trait where False` constraint, and both
# spellings again print the same canonical trait.
# CHECK-LABEL: lit.struct.decl @NotSpellingWithReason
# CHECK-SAME: (!AnyType_Deinitable)
struct NotSpellingWithReason(not Movable else "a Handle is pinned to its port"):
    var x: Int


# CHECK-LABEL: lit.struct.decl @WhereFalseSpellingWithReason
# CHECK-SAME: (!AnyType_Deinitable)
struct WhereFalseSpellingWithReason(
    Movable where False else "a Handle is pinned to its port"
):
    var x: Int


# A parenthesized reason wraps across lines, as a `where ... else` message does.
# CHECK-LABEL: lit.struct.decl @NotWrappedReason
# CHECK-SAME: (!AnyType_Deinitable)
struct NotWrappedReason(
    not Movable else (
        "a Handle is pinned to the port it was opened on "
        "and cannot be moved between them"
    )
):
    var x: Int


# The reason applies to every symbol of a composition, as the opt-out does.
# CHECK-LABEL: lit.struct.decl @NotCompositionWithReason
# CHECK-SAME: (!AnyType_Deinitable_Movable)
struct NotCompositionWithReason(not (Base & Other) else "not a plain payload"):
    var x: Int


# Opting out of `Deinitable` makes the struct linear, and the reason becomes
# the message for abandoning one -- the replacement for `@explicit_destroy`.
# See `explicit_destroy_errors.mojo` for the diagnostic it reaches.
# CHECK-LABEL: lit.struct.decl @NotDeinitableWithReason
# CHECK-SAME: call `close()` instead
struct NotDeinitableWithReason(not Deinitable else "call `close()` instead"):
    var x: Int


##===----------------------------------------------------------------------===##
# `not` on a trailing-`where` struct
##===----------------------------------------------------------------------===##
# The opt-out is independent of the struct's own clause; neither erases the
# other.
# CHECK-LABEL: lit.struct.decl @NotWithTrailingWhere
# CHECK-NOT: __init__(move:
struct NotWithTrailingWhere[n: Int](not Movable) where n > 0:
    var x: Int


# Sentinel: bounds the preceding `CHECK-NOT` region before the stdlib decls
# (which legitimately conform to `Movable`) appear in the dump.
# CHECK-LABEL: lit.struct.decl @Sentinel
struct Sentinel:
    pass
