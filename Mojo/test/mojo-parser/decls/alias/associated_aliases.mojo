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
# RUN: %parse-mojo-isolated -split-input-file %s | FileCheck %s


# Tests that we correctly call get_witness when looking up a trait's alias.


@fieldwise_init
struct ZInt(TrivialRegisterPassable):
    pass


# CHECK-LABEL: lit.trait.decl @TraitWithAlias
trait TraitWithAlias:
    # CHECK-NEXT: lit.alias.decl *"N`1": !ZInt
    comptime N: ZInt


# CHECK-LABEL: lit.struct.decl @StructWithMatchingAlias
struct StructWithMatchingAlias(TraitWithAlias):
    # CHECK: kgen.conformance {{.*}}::@TraitWithAlias
    # CHECK: kgen.witness "N" : !ZInt = apply{{.*}}ZInt::@"__init__()"
    comptime N: ZInt = ZInt()

    def __init__(out self):
        pass


struct S[n: Int](Movable where False):
    def __init__(out self):
        pass


# CHECK-LABEL: lit.trait.decl @TraitWithDependentAlias
trait TraitWithDependentAlias:
    # CHECK-NEXT: lit.alias.decl *"N`1": !alias_Int1
    comptime N: Int
    # CHECK-NEXT: lit.alias.decl *"depend_on_N`2": {{.*}}#S <:!Int {{.*}}#kgen.get_witness<:!{{.*}} *"_Self`", @{{.*}}::@{{.*}}, "N">{{.*}}
    comptime depend_on_N: S[Self.N]


# CHECK-LABEL: lit.struct.decl @StructWithMatchingDependentAlias1
struct StructWithMatchingDependentAlias1(TraitWithDependentAlias, Movable where False):
    # CHECK: lit.alias.decl *"N`": !alias_Int1 = <rebind(:!Int {:scalar<index> 1})>
    comptime N: Int = 1
    # CHECK-NEXT: lit.alias.decl *"depend_on_N`1": {{.*}}#S <:!Int {:scalar<index> 1}>> =
    comptime depend_on_N = S[1]()


# CHECK-LABEL: lit.struct.decl @StructWithMatchingDependentAlias2
struct StructWithMatchingDependentAlias2(TraitWithDependentAlias, Movable where False):
    # CHECK: lit.alias.decl *"N`": !alias_Int1 = <rebind(:!Int {:scalar<index> 1})>
    comptime N: Int = 1
    # CHECK-NEXT: lit.alias.decl *"depend_on_N`1": {{.*}}#S <:!Int {:scalar<index> 1}>> =
    comptime depend_on_N = S[Self.N]()


# This tests that we correctly call get_witness when looking up a trait's
# alias.
# CHECK-LABEL: lit.fn @"getNFromTraitWithAlias
def getNFromTraitWithAlias[T: TraitWithAlias](t: T) -> ZInt:
    # CHECK-NEXT: lit.alias.decl [[X:.*]]: !ZInt = <{{.*}}#kgen.get_witness<:!TraitWithAlias_AnyType T, @associated_aliases::@TraitWithAlias, "N">
    # CHECK-NEXT: kgen.param.constant: !ZInt = <#alias_X>
    comptime X = T.N
    return X


# This tests that we create a #kgen.type for StructWithMatchingAlias for
# TraitWithAlias, and it contains an entry for `N` of the right type.`
# CHECK-LABEL: lit.fn export @"testTraitWithAliasAndStructWithMatchingAlias
@export
def testTraitWithAliasAndStructWithMatchingAlias() abi("Mojo"):
    # CHECK: {{.*}} = lit.call @associated_aliases::@"getNFromTraitWithAlias{{.*}}<:!TraitWithAlias_AnyType !StructWithMatchingAlias>(%1)
    _ = getNFromTraitWithAlias(StructWithMatchingAlias())


# // -----

# Tests that we correctly call get_witness when looking up a trait's alias,
# even when we're looking up an alias that originally came from a grandparent.
# (See also MOCO-1992)


@fieldwise_init
struct ZInt(TrivialRegisterPassable):
    pass


# CHECK-LABEL: lit.trait.decl @TraitWithAlias
trait TraitWithAlias:
    # CHECK-NEXT: lit.alias.decl *"N`1": !ZInt
    comptime N: ZInt


trait TraitWithTypeAlias:
    comptime T: TraitWithAlias


trait TraitWithSameTypeAlias(TraitWithTypeAlias):
    pass


# CHECK-LABEL: lit.fn @"testTraitWithRefinedTypeAlias
def testTraitWithRefinedTypeAlias[T: TraitWithSameTypeAlias]():
    # CHECK-NEXT: !TraitWithAlias_AnyType = <{{.*}}#kgen.get_witness<:!TraitWithSameTypeAlias_TraitWithTypeAlias_AnyType T, @associated_aliases::@TraitWithTypeAlias, "T">
    comptime MyT: TraitWithAlias = T.T


# // -----

# Tests that we can upcast a generic struct to a trait, when the generic struct
# uses an input-parameter in a method override for a trait method that mentions
# a trait alias in an argument.


@fieldwise_init
struct ZInt(ATrait, TrivialRegisterPassable):
    pass


trait ATrait:
    pass


struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasArgMethod:
    comptime T: ATrait

    def lork(self, thing: SIMD[Self.T]):
        ...


@fieldwise_init
struct StructWithAliasArgMethod(TraitWithAliasArgMethod):
    # CHECK: kgen.conformance {{.*}}::@TraitWithAliasArgMethod
    # CHECK: kgen.witness "T" : !ATrait_AnyType = !ZInt
    # CHECK: kgen.witness "lork{{.*}}" : {{.*}} = {{.*}}::@StructWithAliasArgMethod::@"lork{{.*}}"
    comptime T: ATrait = ZInt

    def lork(self, thing: SIMD[ZInt]):
        pass


def receiveTraitWithAliasArgMethod[X: TraitWithAliasArgMethod](t: X):
    pass


# CHECK-LABEL: lit.fn @"testUpcastingStructWithAliasArgMethod
def testUpcastingStructWithAliasArgMethod():
    # CHECK: {{.*}}lit.call @associated_aliases::@"receiveTraitWithAliasArgMethod{{.*}}<:!TraitWithAliasArgMethod_AnyType !StructWithAliasArgMethod>
    receiveTraitWithAliasArgMethod(StructWithAliasArgMethod())


# // -----

# Tests that we can upcast a generic struct to a trait, when the generic struct
# uses an input-parameter in a STATIC method override for a trait method that
# mentions a trait alias in an argument. (Similar to the last test but static)


@fieldwise_init
struct ZInt(ATrait, TrivialRegisterPassable):
    pass


trait ATrait:
    pass


struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasArgMethod:
    comptime T: ATrait

    @staticmethod
    def lork(thing: SIMD[Self.T]):
        ...


@fieldwise_init
struct StructWithAliasArgMethod(TraitWithAliasArgMethod):
    comptime T: ATrait = ZInt

    @staticmethod
    def lork(thing: SIMD[ZInt]):
        pass


def receiveTraitWithAliasArgMethod[X: TraitWithAliasArgMethod](t: X):
    pass


# CHECK-LABEL: lit.fn @"testUpcastingStructWithAliasArgMethod
def testUpcastingStructWithAliasArgMethod():
    # CHECK: {{.*}}lit.call @associated_aliases::@"receiveTraitWithAliasArgMethod{{.*}}<:!TraitWithAliasArgMethod_AnyType !StructWithAliasArgMethod>
    receiveTraitWithAliasArgMethod(StructWithAliasArgMethod())


# // -----

# Tests that we can call a trait's method, when it mentions a trait alias in
# an argument type.


@fieldwise_init
struct ZInt(TrivialRegisterPassable):
    pass


trait ATrait:
    pass


struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasArgMethod:
    comptime T: ATrait

    def lork(self, thing: SIMD[Self.T]):
        ...


# CHECK-LABEL: lit.fn @"callTraitMethodWithAliasArg
def callTraitMethodWithAliasArg[
    X: TraitWithAliasArgMethod
](t: X, thing: SIMD[X.T]):
    # CHECK:  %0 = lit.call
    # CHECK-SAME: "thing": !lit.ref<{{.*}}#SIMD <:!ATrait_AnyType {{.*}}#kgen.get_witness<:!TraitWithAliasArgMethod_AnyType X, @associated_aliases::@TraitWithAliasArgMethod, "T">)>
    # CHECK-SAME: #kgen.get_witness<:!TraitWithAliasArgMethod_AnyType X, @associated_aliases::@TraitWithAliasArgMethod, "lork
    t.lork(thing)


# // -----

# Tests that we can call a trait's STATIC method, when it mentions a trait alias
# in an argument type. Similar to the previous test but with static.


@fieldwise_init
struct ZInt(TrivialRegisterPassable):
    pass


trait ATrait:
    pass


struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasArgMethod:
    comptime T: ATrait

    @staticmethod
    def lork(thing: SIMD[Self.T]):
        ...


# CHECK-LABEL: lit.fn @"callTraitMethodWithAliasArg
def callTraitMethodWithAliasArg[
    X: TraitWithAliasArgMethod
](t: X, thing: SIMD[X.T]):
    # CHECK:  %0 = lit.call
    # CHECK-SAME: "thing": !lit.ref<{{.*}}#SIMD <:!ATrait_AnyType {{.*}}#kgen.get_witness<:!TraitWithAliasArgMethod_AnyType X, @associated_aliases::@TraitWithAliasArgMethod, "T">)>
    # CHECK-SAME: #kgen.get_witness<:!TraitWithAliasArgMethod_AnyType X, @associated_aliases::@TraitWithAliasArgMethod, "lork
    t.lork(thing)


# // -----

# Tests that a trait can have a method that returns a generic struct with an
# input parameter-value that's a trait alias.


@fieldwise_init
struct ZInt(ATrait, TrivialRegisterPassable):
    pass


trait ATrait:
    pass


@fieldwise_init
struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasReturnMethod:
    comptime T: ATrait

    def bork(self) -> SIMD[Self.T]:
        ...


# CHECK-LABEL: lit.struct.decl @ExplicitStructWithAliasMethod
@fieldwise_init
struct ExplicitStructWithAliasMethod(TraitWithAliasReturnMethod):
    # CHECK: kgen.conformance {{.*}}::@TraitWithAliasReturnMethod
    # CHECK: kgen.witness "T" : !ATrait_AnyType = !ZInt
    # CHECK: kgen.witness "bork{{.*}}" : {{.*}} = {{.*}}::@ExplicitStructWithAliasMethod::@"bork{{.*}}"
    comptime T: ATrait = ZInt

    def bork(self) -> SIMD[ZInt]:
        return SIMD[ZInt]()


# CHECK-LABEL: lit.fn @"testUpcastingExplicitStructWithAliasMethod
def testUpcastingExplicitStructWithAliasMethod():
    # CHECK:       {{.*}}lit.call @associated_aliases::@"receiveTraitWithAliasReturnMethod{{.*}}<:!TraitWithAliasReturnMethod_AnyType !ExplicitStructWithAliasMethod>
    receiveTraitWithAliasReturnMethod(ExplicitStructWithAliasMethod())


def receiveTraitWithAliasReturnMethod[X: TraitWithAliasReturnMethod](t: X):
    pass


# // -----

# Tests that a trait can have a STATIC method that returns a generic struct with
# an input parameter-value that's a trait alias. Similar to the previous test
# but static.


@fieldwise_init
struct ZInt(ATrait, TrivialRegisterPassable):
    pass


trait ATrait:
    pass


@fieldwise_init
struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasReturnMethod:
    comptime T: ATrait

    @staticmethod
    def bork() -> SIMD[Self.T]:
        ...


# CHECK-LABEL: lit.struct.decl @ExplicitStructWithAliasMethod
@fieldwise_init
struct ExplicitStructWithAliasMethod(TraitWithAliasReturnMethod):
    comptime T: ATrait = ZInt

    @staticmethod
    def bork() -> SIMD[ZInt]:
        return SIMD[ZInt]()


# CHECK-LABEL: lit.fn @"testUpcastingExplicitStructWithAliasMethod
def testUpcastingExplicitStructWithAliasMethod():
    # CHECK:       {{.*}}lit.call @associated_aliases::@"receiveTraitWithAliasReturnMethod{{.*}}<:!TraitWithAliasReturnMethod_AnyType !ExplicitStructWithAliasMethod>
    receiveTraitWithAliasReturnMethod(ExplicitStructWithAliasMethod())


def receiveTraitWithAliasReturnMethod[X: TraitWithAliasReturnMethod](t: X):
    pass


# // -----

# Tests that we can call an alias-returning method on a given trait instance.


trait ATrait:
    pass


struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasReturnMethod:
    comptime T: ATrait

    def bork(self) -> SIMD[Self.T]:
        ...


# CHECK-LABEL: lit.fn @"callTraitWithAliasReturnMethod
def callTraitWithAliasReturnMethod[X: TraitWithAliasReturnMethod](t: X):
    # CHECK: {{.*}}lit.call
    # CHECK-SAME: "__result__": !lit.ref<{{.*}}#SIMD <:!ATrait_AnyType {{.*}}#kgen.get_witness<:!TraitWithAliasReturnMethod_AnyType X, @associated_aliases::@TraitWithAliasReturnMethod, "T">)>
    # CHECK-SAME: #kgen.get_witness<:!TraitWithAliasReturnMethod_AnyType X, @associated_aliases::@TraitWithAliasReturnMethod, "bork($0)"
    _ = t.bork()


# // -----

# Tests that we can call an alias-returning STATIC method on a given trait
# instance. Similar to the previous test but with static.


trait ATrait:
    pass


struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasReturnMethod:
    comptime T: ATrait

    @staticmethod
    def bork() -> SIMD[Self.T]:
        ...


# CHECK-LABEL: lit.fn @"callTraitWithAliasReturnMethod
def callTraitWithAliasReturnMethod[X: TraitWithAliasReturnMethod](t: X):
    # CHECK: {{.*}}lit.call
    # CHECK-SAME: "__result__": !lit.ref<{{.*}}#SIMD <:!ATrait_AnyType {{.*}}#kgen.get_witness<:!TraitWithAliasReturnMethod_AnyType X, @associated_aliases::@TraitWithAliasReturnMethod, "T">)>
    # CHECK-SAME: #kgen.get_witness<:!TraitWithAliasReturnMethod_AnyType X, @associated_aliases::@TraitWithAliasReturnMethod, "bork()">
    _ = t.bork()


# // -----

# Tests that we can upcast a generic struct to a trait, when the generic struct
# uses an input-parameter in a method override for a trait method that mentions
# a trait alias in the return.


@fieldwise_init
struct ZInt(ATrait, TrivialRegisterPassable):
    pass


trait ATrait:
    pass


@fieldwise_init
struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasReturnMethod:
    comptime T: ATrait

    def bork(self) -> SIMD[Self.T]:
        ...


# TODO(MOCO-1109): also check that this works with the thunk generation for RegisterPassable methods
# CHECK-LABEL: lit.struct.decl @GenericStructWithAliasMethod<Z: !ATrait_AnyType>
@fieldwise_init
struct GenericStructWithAliasMethod[Z: ATrait](TraitWithAliasReturnMethod):
    # CHECK: kgen.conformance {{.*}}::@TraitWithAliasReturnMethod
    # CHECK: kgen.witness "T" : !ATrait_AnyType = Z
    # CHECK: kgen.witness "bork{{.*}}" : {{.*}} = {{.*}}::@GenericStructWithAliasMethod::@"bork{{.*}}"
    comptime T: ATrait = Self.Z

    def bork(self) -> SIMD[Self.Z]:
        return SIMD[Self.Z]()


# CHECK-LABEL: lit.fn @"testUpcastingGenericStructWithAliasMethod
def testUpcastingGenericStructWithAliasMethod():
    # CHECK: {{.*}}lit.call @associated_aliases::@"receiveTraitWithAliasReturnMethod{{.*}}<:!TraitWithAliasReturnMethod_AnyType @associated_aliases::@GenericStructWithAliasMethod<:!ATrait_AnyType !ZInt>>
    receiveTraitWithAliasReturnMethod(GenericStructWithAliasMethod[ZInt]())


def receiveTraitWithAliasReturnMethod[X: TraitWithAliasReturnMethod](t: X):
    pass


# // -----

# Tests that we can upcast a generic struct to a trait, when the generic struct
# uses an input-parameter in a STATIC method override for a trait method that
# mentions a trait alias in the return. Same as the last test but with static.


@fieldwise_init
struct ZInt(ATrait, TrivialRegisterPassable):
    pass


trait ATrait:
    pass


@fieldwise_init
struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasReturnMethod:
    comptime T: ATrait

    @staticmethod
    def bork() -> SIMD[Self.T]:
        ...


# TODO(MOCO-1109): also check that this works with the thunk generation for RegisterPassable methods
@fieldwise_init
struct GenericStructWithAliasMethod[Z: ATrait](TraitWithAliasReturnMethod):
    comptime T: ATrait = Self.Z

    @staticmethod
    def bork() -> SIMD[Self.Z]:
        return SIMD[Self.Z]()


# CHECK-LABEL: lit.fn @"testUpcastingGenericStructWithAliasMethod
def testUpcastingGenericStructWithAliasMethod():
    # CHECK: {{.*}}lit.call @associated_aliases::@"receiveTraitWithAliasReturnMethod{{.*}}<:!TraitWithAliasReturnMethod_AnyType @associated_aliases::@GenericStructWithAliasMethod<:!ATrait_AnyType !ZInt>>
    receiveTraitWithAliasReturnMethod(GenericStructWithAliasMethod[ZInt]())


def receiveTraitWithAliasReturnMethod[X: TraitWithAliasReturnMethod](t: X):
    pass


# // -----

# Tests that we can upcast a struct to a trait, when the trait method mentions
# a trait alias in the return type, specifically with `Self.`.


@fieldwise_init
struct ZInt(ATrait, TrivialRegisterPassable):
    pass


trait ATrait:
    pass


@fieldwise_init
struct SIMD[T: ATrait]:
    pass


# Tests explicit mentions of `Self.T` where `T` is an associated alias.
# See MOCO-1438.
trait TraitWithSelfDotAliasReturnMethod:
    comptime T: ATrait

    def bork(self) -> SIMD[Self.T]:
        ...


# CHECK-LABEL: lit.struct.decl @StructWithSelfDotAliasReturnMethod
struct StructWithSelfDotAliasReturnMethod(TraitWithSelfDotAliasReturnMethod):
    # CHECK: kgen.conformance {{.*}}::@TraitWithSelfDotAliasReturnMethod
    # CHECK: kgen.witness "T" : !ATrait_AnyType = !ZInt
    # CHECK: kgen.witness "bork{{.*}}" : {{.*}} = {{.*}}::@StructWithSelfDotAliasReturnMethod::@"bork{{.*}}"
    comptime T: ATrait = ZInt

    def bork(self) -> SIMD[Self.T]:
        return SIMD[Self.T]()


def receiveTraitWithSelfDotAliasReturnMethod[
    T: TraitWithSelfDotAliasReturnMethod
](z: T):
    _ = z.bork()


def upcastStructWithSelfDotAliasReturnMethod(
    x: StructWithSelfDotAliasReturnMethod,
):
    receiveTraitWithSelfDotAliasReturnMethod(x)


# // -----

# Tests that we can upcast a struct to a trait, when the trait STATIC method
# mentions a trait alias in the return type, specifically with `Self.`. Same as
# the last test but with static.


@fieldwise_init
struct ZInt(ATrait, TrivialRegisterPassable):
    pass


trait ATrait:
    pass


@fieldwise_init
struct SIMD[T: ATrait]:
    pass


# Tests explicit mentions of `Self.T` where `T` is an associated alias.
# See MOCO-1438
trait TraitWithSelfDotAliasReturnMethod:
    comptime T: ATrait

    @staticmethod
    def bork() -> SIMD[Self.T]:
        ...


struct StructWithSelfDotAliasReturnMethod(TraitWithSelfDotAliasReturnMethod):
    comptime T: ATrait = ZInt

    @staticmethod
    def bork() -> SIMD[Self.T]:
        return SIMD[Self.T]()


def receiveTraitWithSelfDotAliasReturnMethod[
    T: TraitWithSelfDotAliasReturnMethod
](z: T):
    _ = z.bork()


def upcastStructWithSelfDotAliasReturnMethod(
    x: StructWithSelfDotAliasReturnMethod,
):
    receiveTraitWithSelfDotAliasReturnMethod(x)


# // -----

# Tests that we can upcast a generic struct to a trait, when the generic struct
# uses an input-parameter in a method override for a trait method that mentions
# a trait alias in the return.
# This is like the above test, but explicitly mentions `Self.` in the return.


@fieldwise_init
struct ZInt(ATrait, TrivialRegisterPassable):
    pass


trait ATrait:
    pass


@fieldwise_init
struct SIMD[T: ATrait]:
    pass


# Tests explicit mentions of `Self.T` where `T` is an associated alias.
# See MOCO-1438
trait TraitWithSelfDotAliasReturnMethod:
    comptime T: ATrait

    def bork(self) -> SIMD[Self.T]:
        ...


# TODO(MOCO-1109): also check that this works with the thunk generation for RegisterPassable methods
# CHECK-LABEL: lit.struct.decl @GenericStructWithSelfDotAliasReturnMethod<Z: !ATrait_AnyType>
@fieldwise_init
struct GenericStructWithSelfDotAliasReturnMethod[Z: ATrait](
    TraitWithSelfDotAliasReturnMethod
):
    # CHECK: kgen.conformance {{.*}}::@TraitWithSelfDotAliasReturnMethod
    # CHECK: kgen.witness "T" : !ATrait_AnyType = Z
    # CHECK: kgen.witness "bork{{.*}}" : {{.*}} = {{.*}}::@GenericStructWithSelfDotAliasReturnMethod::@"bork{{.*}}"
    comptime T: ATrait = Self.Z

    def bork(self) -> SIMD[Self.T]:
        return SIMD[Self.T]()


def receiveTraitWithSelfDotAliasReturnMethod[
    T: TraitWithSelfDotAliasReturnMethod
](z: T):
    _ = z.bork()


def testUpcastingGenericStructWithSelfDotAliasReturnMethod():
    receiveTraitWithSelfDotAliasReturnMethod(
        GenericStructWithSelfDotAliasReturnMethod[ZInt]()
    )


# // -----

# Tests that we can upcast a generic struct to a trait, when the generic struct
# uses an input-parameter in a STATIC method override for a trait method that
# mentions a trait alias in the return.
# This is like some above tests, but explicitly mentions `Self.` in the return
# and with the method being static.


@fieldwise_init
struct ZInt(ATrait, TrivialRegisterPassable):
    pass


trait ATrait:
    pass


@fieldwise_init
struct SIMD[T: ATrait]:
    pass


# Tests explicit mentions of `Self.T` where `T` is an associated alias.
# See MOCO-1438
trait TraitWithSelfDotAliasReturnMethod:
    comptime T: ATrait

    @staticmethod
    def bork() -> SIMD[Self.T]:
        ...


# TODO(MOCO-1109): also check that this works with the thunk generation for RegisterPassable methods
@fieldwise_init
struct GenericStructWithSelfDotAliasReturnMethod[Z: ATrait](
    TraitWithSelfDotAliasReturnMethod
):
    comptime T: ATrait = Self.Z

    @staticmethod
    def bork() -> SIMD[Self.T]:
        return SIMD[Self.T]()


def receiveTraitWithSelfDotAliasReturnMethod[
    T: TraitWithSelfDotAliasReturnMethod
](z: T):
    _ = z.bork()


def testUpcastingGenericStructWithSelfDotAliasReturnMethod():
    receiveTraitWithSelfDotAliasReturnMethod(
        GenericStructWithSelfDotAliasReturnMethod[ZInt]()
    )


# // -----

# Tests that we correctly handle substituting struct alias into the "needle"
# signature when confirming that a trait's method exists in the struct (see
# SAVMBCTATBS).


trait ATrait:
    pass


trait ASubTrait(ATrait):
    pass


struct ZInt(ASubTrait, ATrait, TrivialRegisterPassable):
    pass


struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasReturnMethod:
    comptime T: ATrait

    def bork(self) -> SIMD[Self.T]:
        ...


# CHECK-LABEL: lit.struct.decl @ExplicitStructWithAliasMethod
@fieldwise_init
struct ExplicitStructWithAliasMethod(TraitWithAliasReturnMethod):
    comptime T: ASubTrait = ZInt

    # If we didn't follow SAVMBCTATBS, then verifyConformance would be
    # incorrectly checking for the existence of
    # `def bork(self) -> SIMD[:ASubTrait ZInt]:` which is actually malformed
    # because SIMD takes an ATrait, not a ASubTrait.
    def bork(self) -> SIMD[ZInt]:
        ...


# // -----

# Tests that we correctly handle substituting struct alias into the "needle"
# signature when confirming that a trait's STATIC method exists in the struct
# (see SAVMBCTATBS). Same as the previous test but with static.


trait ATrait:
    pass


trait ASubTrait(ATrait):
    pass


struct ZInt(ASubTrait, ATrait, TrivialRegisterPassable):
    pass


struct SIMD[T: ATrait]:
    pass


trait TraitWithAliasReturnMethod:
    comptime T: ATrait

    @staticmethod
    def bork() -> SIMD[Self.T]:
        ...


# CHECK-LABEL: lit.struct.decl @ExplicitStructWithAliasMethod
@fieldwise_init
struct ExplicitStructWithAliasMethod(TraitWithAliasReturnMethod):
    comptime T: ASubTrait = ZInt

    # If we didn't follow SAVMBCTATBS, then verifyConformance would be
    # incorrectly checking for the existence of
    # `def bork(self) -> SIMD[:ASubTrait ZInt]:` which is actually malformed
    # because SIMD takes an ATrait, not a ASubTrait.
    @staticmethod
    def bork() -> SIMD[ZInt]:
        ...


# // -----

# Tests that we can call a static method that has an associated alias in it.


struct ZInt:
    pass


@fieldwise_init
struct Zcalar[X: ZInt]:
    pass


trait FooTrait:
    comptime dtype: ZInt

    @staticmethod
    def foo(x: Zcalar[Self.dtype]):
        ...


def bar[foo: FooTrait]():
    var p0 = Zcalar[foo.dtype]()
    foo.foo(p0)


# // -----

# Sub-Trait Alias Type Can Be More Specific (STATCBMS):
# This test shows that a sub-trait's alias can have a more specific type than
# the super-trait's alias.
# See original bug:
# https://linear.app/modularml/issue/MOCO-1869/bug-trait-refinement-does-not-correctly-propagate-associated-type


# This is the important check, it makes sure that we:
#  * Pull it out of the witness table as a BB
#  * Upcast it to an AA


trait AA(Movable):
    def __init__(out self):
        ...


trait BB(AA):
    def __init__(out self):
        ...


trait A:
    comptime Type: AA


trait B(A):
    comptime Type: BB


def fa[T: A]() -> T.Type:
    return T.Type()


# CHECK-LABEL: lit.fn @"fb
def fb[T: B]() -> T.Type:
    # CHECK: [[REBIND:%.*]] = kgen.rebind %__result__ : {{.*}} to !lit.ref<:!AA_AnyType_Movable sugar_member_alias{{.*}}#kgen.get_witness{{.*}}@associated_aliases::@A, "Type">
    # CHECK-NEXT: lit.call{{.*}}fa{{.*}}([[REBIND]])
    return fa[T]()


# Seeing two resolvable conflicting aliases in trait type should work too.
# CHECK-LABEL: lit.fn @"fb_trait_composition
def fb_trait_composition[T: B & A]() -> T.Type:
    # CHECK: [[REBIND:%.*]] = kgen.rebind %__result__ : {{.*}} to !lit.ref<:!AA_AnyType_Movable sugar_member_alias{{.*}}#kgen.get_witness{{.*}}@associated_aliases::@A, "Type">
    # CHECK-NEXT: lit.call{{.*}}fa{{.*}}([[REBIND]])
    return fa[T]()


# // -----

# Tests that a trait or struct can declare an alias that is a more specific type
# of *both* of its parent traits' aliases' types.
# This is like the above STATCBMS test but with two parent traits.


trait AA:
    def __init__(out self):
        ...


trait BB:
    def __init__(out self):
        ...


trait CC(AA, BB):
    def __init__(out self):
        ...


trait A:
    comptime Type: AA


trait B:
    comptime Type: BB


trait TraitWithExplicitOverride(A, B):
    comptime Type: CC


def receiveTraitWithExplicitOverride[T: TraitWithExplicitOverride]():
    comptime cc: CC = T.Type


struct StructImplCC(CC):
    def __init__(out self):
        pass


struct StructWithExplicitOverride(A, B):
    comptime Type: CC = StructImplCC


def receiveStructWithExplicitOverride[T: StructWithExplicitOverride]():
    comptime cc: CC = T.Type


# // -----


# Tests that a bug originally introduced by (#65985) is no longer present.
struct ZInt:
    pass


trait A:
    comptime foo: ZInt


trait B(A):
    comptime foo: ZInt


trait C(B):
    pass


# TODO(MOCO-2123): Make this work:
# trait TraitWithNoOverride(A, B):
#     pass
# def receiveTraitWithNoOverride[T: TraitWithNoOverride]():
#     alias Something = T.Type


# TODO(MOCO-1143): Make this work:
# struct StructWithParam[X: ZInt]:
#     pass
# # HECK-LABEL: lit.trait.decl @Spork<Self: type> {
# trait TraitWithStaticMethodUsingAlias:
#     # HECK-NEXT: lit.alias.decl N = <?>
#     alias N: ZInt
#     # HECK-LABEL: lit.fn @foo(%x: !kgen.simd<N, f32>) { // #kgen.param.decl.ref<"N"> : index
#     @staticmethod
#     def foo(x: StructWithParam[N]):
#         pass
# struct StructWithStaticMethod:
#     @staticmethod
#     def foo(x: StructWithParam[5]):
#         pass
# def sporkify[T: TraitWithStaticMethodUsingAlias]() -> ZInt:
#    return T.N # emits a get_witness
# @export
# def testSomething():
#     # And maybe add a test for sporkify[TraitWithStaticMethodUsingAlias]()
#     sporkify[StructWithStaticMethod]()

# // -----

# Tests that we can correctly access associated aliases via an instance of a
# parametric type whose trait is known.


@fieldwise_init
struct ZInt(TrivialRegisterPassable):
    pass


trait MyTrait:
    comptime BIT_WIDTH: ZInt


trait MyTrait2:
    pass


@fieldwise_init
struct MyStruct(MyTrait, MyTrait2, Movable where False):
    comptime BIT_WIDTH = ZInt()


# CHECK-LABEL: lit.fn @"bitwidth_from_instance
def bitwidth_from_instance[T: MyTrait, Inst: T]() -> ZInt:
    # CHECK-NEXT: #kgen.get_witness<:!MyTrait_AnyType T, @{{.*}}::@MyTrait, "BIT_WIDTH">
    return Inst.BIT_WIDTH


# CHECK-LABEL: lit.fn @"bitwidth_from_composition_instance
def bitwidth_from_composition_instance[
    T: MyTrait & MyTrait2, Inst: T
]() -> ZInt:
    # CHECK-NEXT: #kgen.get_witness<:!MyTrait_MyTrait2_AnyType T, @associated_aliases::@MyTrait, "BIT_WIDTH">
    return Inst.BIT_WIDTH


# // -----


trait DependentAssociatedTypeInDefault:
    comptime T1: AnyType
    comptime V1: Self.T1


# CHECK-LABEL: lit.fn @"foo
def foo[
    T: DependentAssociatedTypeInDefault,
    # CHECK: V: !kgen.param<:!AnyType {{.*}}#kgen.get_witness<:!{{.*}} T, @{{.*}}, "T1">)> = {{.*}}#kgen.get_witness<:!{{.*}} T, @{{.*}}, "V1">)
    V: T.T1 = T.V1,
]():
    pass
