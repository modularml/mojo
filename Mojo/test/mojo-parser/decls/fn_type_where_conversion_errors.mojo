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

# A function type's trailing `where` clauses are an obligation, not part of its
# representation, so a constrained function is not interchangeable with an
# unconstrained one of the same shape. Dropping the clause has to be refused
# unless it can be proven: the value would be rebound to a type that no longer
# mentions the obligation, and a later call through it would never recheck it.

# RUN: %parse-mojo-isolated -verify-diagnostics %s


trait Marker:
    pass


def marked_impl[T: AnyType](x: Int) where conforms_to(T, Marker):
    pass


# expected-note @below {{function declared here}}
def takes_unmarked[F: def[T: AnyType](Int) thin -> None]():
    pass


# Nothing in this scope proves `conforms_to(T, Marker)` for an arbitrary `T`, so
# the constraint cannot be shed to reach the unconstrained parameter type.
def shed_without_evidence():
    # expected-error @below {{parameter 'F' has 'def[T: AnyType](Int) thin -> None' type, but value has type 'def marked_impl[T: AnyType](x: Int) thin -> None where conforms_to(T, Marker)'}}
    takes_unmarked[marked_impl]()


# The same refusal applies to non-function generator types: a `where` clause on
# a `comptime` generator alias is a body constraint, and shedding it to reach an
# unconstrained generator type is refused unless this scope proves it.
comptime plain_gen[U: AnyType] = U


comptime marked_gen[U: AnyType] where conforms_to(U, Marker) = U


comptime PlainGenType = type_of(plain_gen)


# expected-note @+1 {{function declared here}}
def takes_plain_gen[F: PlainGenType]():
    pass


def shed_nonfn_without_evidence():
    # expected-error @+2 {{parameter 'F' has 'PlainGenType' type, but value has type '__generator_type[U: AnyType] AnyType where conforms_to(U, Marker)'}}
    # expected-note @+1 {{'PlainGenType' is aka '__generator_type[U: AnyType] AnyType'}}
    takes_plain_gen[marked_gen]()
