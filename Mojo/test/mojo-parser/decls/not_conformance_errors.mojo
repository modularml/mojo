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

# Errors for the `not Trait` conformance-list entry (see `not_conformance.mojo`
# for the accepted forms).
#
# A trait opted out with `not` and then mentioned again in the same list is
# diagnosed as a repeated trait, not as disagreeing constraints: the user wrote
# no constraint to reconcile, and one of the two entries is dead either way.
# The transitive case needs resolved trait symbols to see that a derived trait
# drags its ancestor in, so it is caught alongside the ordinary
# derived-implies-ancestor check rather than during parsing.

# RUN: %parse-mojo-isolated -verify-diagnostics %s


trait Marker:
    pass


trait Refines(Marker):
    pass


# ===========================================================================
# Opted out and listed again, explicitly
# ===========================================================================


struct OptOutThenList(
    not Marker,  # expected-note {{opted out here}}
    # expected-error @below {{trait 'Marker' must not be listed and also opted out with 'not'; remove one of the entries}}
    Marker,
):
    var x: Int


# The order of the two entries does not matter: the error always lands on the
# plain entry and the note on the `not`.
struct ListThenOptOut(
    # expected-error @below {{trait 'Marker' must not be listed and also opted out with 'not'; remove one of the entries}}
    Marker,
    not Marker,  # expected-note {{opted out here}}
):
    var x: Int


# The two spellings agreeing on false is still a repeated trait. This is the
# case the pre-existing "different constraints" check cannot see, because after
# canonicalization the propositions are equal.
struct OptOutThenWhereFalse(
    not Marker,  # expected-note {{opted out here}}
    # expected-error @below {{trait 'Marker' must not be listed and also opted out with 'not'; remove one of the entries}}
    Marker where False,
):
    var x: Int


# ===========================================================================
# Opted out twice
# ===========================================================================


struct OptOutTwice(
    not Marker,  # expected-note {{first opted out here}}
    # expected-error @below {{trait 'Marker' must be opted out at most once; remove the duplicate 'not'}}
    not Marker,
):
    var x: Int


# ===========================================================================
# Opted out and pulled back in transitively
# ===========================================================================
# `Refines` refines `Marker`, so conforming to it unconditionally requires
# `Marker` -- which this list opts out of.


struct OptOutAncestorOfListed(
    not Marker,  # expected-note {{opted out here}}
    # expected-error @below {{trait 'Refines' requires ancestor trait 'Marker', which is opted out with 'not'; remove one of the entries}}
    Refines,
):
    var x: Int


# Same shape through the injected traits: `Copyable` refines `Movable`.
struct OptOutMovableKeepCopyable(
    not Movable,  # expected-note {{opted out here}}
    # expected-error @below {{trait 'Copyable' requires ancestor trait 'Movable', which is opted out with 'not'; remove one of the entries}}
    Copyable,
):
    var x: Int


# ===========================================================================
# `not` combined with a `where` clause
# ===========================================================================
# `not Trait` already fixes the condition to false, so a second condition could
# only restate it or contradict it.


struct NotWithWhere[T: AnyType](
    # expected-error @below {{'not' conformance does not support a 'where' clause}}
    not Marker where conforms_to(T, Marker),
):
    var x: Int


# ===========================================================================
# Malformed `not` entries
# ===========================================================================


struct RepeatedNot(
    # expected-error @below {{'not' must not be repeated in a conformance entry}}
    not not Marker,
):
    var x: Int


struct NotAValue(
    # expected-error @below {{expected a type, not a value}}
    not 42,
):
    var x: Int


# ===========================================================================
# Malformed `else "<reason>"` on a `not` entry
# ===========================================================================
# The reason is parsed by the same code as a `where ... else` message, so it
# carries the same restriction to string literals.


struct NotElseNonLiteral(
    # expected-error @below {{the message in a 'not' conformance must be a string literal}}
    not Marker else 42,
):
    var x: Int


struct NotElseMissingMessage(
    # expected-error @below {{expected a string literal message after 'else'}}
    not Marker else,
):
    var x: Int


# A reason needs something to explain. Without a `where` clause or a `not`,
# the entry is an unconditional conformance that cannot fail.
struct ElseWithoutCondition(
    # expected-error @below {{a conformance message requires a 'where' clause or a 'not' conformance}}
    Marker else "Marker is required",
):
    var x: Int


# ===========================================================================
# A reason on `not Deinitable` competes with `@explicit_destroy`
# ===========================================================================
# Both name the message for abandoning the linear type the opt-out creates,
# so writing both is ambiguous rather than redundant.


# expected-error @below {{@explicit_destroy and the 'Deinitable' opt-out both give a message; keep only one}}
@explicit_destroy("use consume()")
struct DoubleLinearMessage(
    # expected-note @below {{opt-out message written here}}
    not Deinitable else "use release()",
):
    var x: Int


# ===========================================================================
# `not` is a conformance condition, so it is struct-only
# ===========================================================================


# expected-error @below {{'not' conformances are only supported on structs}}
trait RefinementOptOut(not Marker):
    pass


struct Extended:
    var x: Int


# expected-error @below {{'not' conformances are only supported on structs}}
__extension Extended(not Marker):
    pass


# A reason is a conformance condition too, so it is rejected before the entry
# gets far enough to be a `not`.
# expected-error @below {{conformance messages are only supported on structs}}
trait MessageOnRefinement(Marker else "Marker is required"):
    pass
