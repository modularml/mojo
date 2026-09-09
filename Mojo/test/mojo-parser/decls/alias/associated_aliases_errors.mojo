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

# RUN: %parse-mojo-isolated -split-input-file -verify-diagnostics %s -o /dev/null

# expected-error @below {{only traits may contain a comptime member without an initializer}}
comptime K: Int

trait MyTrait:  # expected-note {{trait 'MyTrait' declared here}}
    # expected-note @below {{required member 'N' is not specified}}
    # expected-note @below {{comptime member 'N' type 'Bool' does not conform to trait's required type 'Int'}}
    comptime N: Int


# expected-error @below {{'StructConformingExplicitlyWithNoMatchingAlias' does not implement all requirements for 'MyTrait'}}
struct StructConformingExplicitlyWithNoMatchingAlias(MyTrait, Movable where False):
    pass


# expected-error @below {{'StructConformingExplicitlyWithMismatchedAlias' does not implement all requirements for 'MyTrait'}}
struct StructConformingExplicitlyWithMismatchedAlias(MyTrait, Movable where False):
    comptime N: Bool = Bool()


# expected-error @below {{'StructConformingExplicitlyWithMemberSameName' does not implement all requirements for 'MyTrait'}}
struct StructConformingExplicitlyWithMemberSameName(MyTrait, Movable where False):
    var N: Int


@fieldwise_init
struct StructWithNoMatchingAlias(Movable where False):
    pass


@fieldwise_init
struct StructWithMismatchedAlias(Movable where False):
    comptime N: Bool = Bool()


struct StructWithUninitializedAlias(Movable where False):
    # expected-error @below {{only traits may contain a comptime member without an initializer}}
    comptime N: Bool


struct StructWithTypelessUninitializedAlias(Movable where False):
    # This makes sure we print out this error, rather than the also-relevant "alias without initial value must have a type" error
    # expected-error @below {{expected '=' after comptime declaration}}
    comptime N


# expected-note @below {{function declared here}}
def funcForMyTrait[T: MyTrait](t: T) -> Int:
    comptime X = T.N
    return X


def testError1():
    # TODO(MOCO-1152): Add more detailed errors for this
    # expected-error @below {{invalid call to 'funcForMyTrait': value passed to 't' cannot be converted from 'StructWithNoMatchingAlias' to 'T', argument type 'StructWithNoMatchingAlias' does not conform to trait 'MyTrait'}}
    var whatev: Int = funcForMyTrait(StructWithNoMatchingAlias())


def testError2():
    # TODO(MOCO-1152): Add more detailed errors for this
    # expected-error @below {{invalid call to 'funcForMyTrait': value passed to 't' cannot be converted from 'StructWithMismatchedAlias' to 'T', argument type 'StructWithMismatchedAlias' does not conform to trait 'MyTrait'}}
    var whatev: Int = funcForMyTrait(StructWithMismatchedAlias())


# // -----


struct TensorIndex[rank: Int](Movable where False):
    pass


trait Stencil:
    comptime rank: Int


# // -----

# Tests that we get a nice error when an override alias has an incompatible
# type.


struct ZInt:
    pass


struct ZBool:
    pass


trait TraitWithTypeAlias:
    # expected-note @below {{the other trait's member defined here}}
    comptime T: ZBool


trait TraitWithSameTypeAlias(TraitWithTypeAlias):
    # expected-error @below {{invalid redefinition of 'T': cannot convert 'ZInt' to the other trait's member's type 'ZBool'}}
    comptime T: ZInt


# // -----

# Makes sure that we don't crash if there are multiple overrides.


struct ZInt:
    pass


struct ZBool:
    pass


struct ZFloat(Movable where False):
    pass


trait SuperTrait:
    comptime T: ZFloat


trait TraitWithTooManyAliases(SuperTrait):
    # expected-note @below {{previous definition here}}
    comptime T: ZBool
    # expected-error @below {{invalid redefinition of 'T'}}
    comptime T: ZInt


# // -----

trait TraitWithComptime:
    # expected-note @+1 {{conflicting comptime alias declared here}}
    comptime MyType = Int


trait ChildOverridesComptimeWithFn(TraitWithComptime):
    # expected-error @+1 {{invalid redefinition of 'MyType'}}
    def MyType(self):
        pass
