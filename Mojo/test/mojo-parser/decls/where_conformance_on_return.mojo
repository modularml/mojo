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

# A conformance proved by a `where` clause must be honored everywhere, not only
# where type refinement folded it into the parameter's bound. The function result
# type is declared, never refined, so `return x^` used to consult the unrefined
# `T` and report a missing `Movable` conformance while every body-level transfer
# of the same value succeeded.

# RUN: %parse-mojo-isolated -verify-diagnostics %s


##===----------------------------------------------------------------------===##
# The reported shape: transfer out through `return`.
##===----------------------------------------------------------------------===##


def ret_move[T: Deinitable](var x: T) -> T where conforms_to(T, Movable):
    return x^


# Via a local, so the transferred value's type comes from the body rather than
# from the signature.
def ret_move_via_local[
    T: Deinitable
](var x: T) -> T where conforms_to(T, Movable):
    var y = x^
    return y^


# `Copyable` refines `Movable`, so it proves the transfer too.
def ret_move_from_copyable[
    T: Deinitable
](var x: T) -> T where conforms_to(T, Copyable):
    return x^


# Exact shape from the bug report: stacked where-clauses including
# `Deinitable` alongside `Copyable`.
def ret_move_from_copyable_and_deletable[
    T: Deinitable
](var x: T) -> T where conforms_to(T, Copyable) where conforms_to(
    T, Deinitable
):
    return x^


##===----------------------------------------------------------------------===##
# Not specific to `Movable`: the implicit-copy check took the same path.
##===----------------------------------------------------------------------===##


def ret_implicit_copy[
    T: Deinitable
](var x: T) -> T where conforms_to(T, ImplicitlyCopyable):
    return x


##===----------------------------------------------------------------------===##
# Controls: these already worked through type refinement and must keep working.
##===----------------------------------------------------------------------===##


def sink[U: Deinitable & Movable](var v: U):
    pass


def body_move[T: Deinitable](var x: T) where conforms_to(T, Movable):
    var y = x^
    _ = y^


def move_assign[
    T: Deinitable
](var x: T, var y: T) where conforms_to(T, Movable):
    y = x^
    _ = y^


def transfer_as_argument[T: Deinitable](var x: T) where conforms_to(T, Movable):
    sink(x^)


def body_implicit_copy[
    T: Deinitable
](var x: T) where conforms_to(T, ImplicitlyCopyable):
    var y = x
    _ = y^
    _ = x^


# An unconditional bound, not an assumption.
def direct_bound[T: Movable](var x: T) -> T:
    return x^


##===----------------------------------------------------------------------===##
# Negative guards: a proven-absent conformance must still be an error. These
# catch a fix that promotes every unprovable conformance to "proven".
##===----------------------------------------------------------------------===##


# A concrete type that opts out of `Movable`. The metatype guard must keep
# answering `no` here rather than reaching for the assumptions.
struct NotMovable(Movable where False):
    def __init__(out self):
        pass


def concrete_not_movable(var m: NotMovable) -> NotMovable:
    # expected-error @below {{cannot transfer value into destination, because 'NotMovable' doesn't conform to 'Movable'}}
    return m^


# Generic with no assumption at all: still unprovable.
def generic_no_assumption[T: Deinitable](var x: T) -> T:
    # expected-error @below {{cannot transfer value into destination, because 'T' doesn't conform to 'Movable'}}
    return x^


# An assumption about an unrelated trait must not prove `Movable`.
trait Unrelated:
    pass


def wrong_assumption[
    T: Deinitable
](var x: T) -> T where conforms_to(T, Unrelated):
    # expected-error @below {{cannot transfer value into destination, because 'T' doesn't conform to 'Movable'}}
    return x^
