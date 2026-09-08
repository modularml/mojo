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

# RUN: %parse-mojo-isolated -verify-diagnostics %s

##===----------------------------------------------------------------------===##
# declarations
##===----------------------------------------------------------------------===##

# expected-error @below {{use of unknown declaration 'y'}}
comptime unknownDecl[x: Int] = y

# expected-error @below {{cannot implicitly convert 'Int' value to 'String'}}
comptime wrongType[x: Int]: String = x

comptime myIntAdd[x: Int, y: Int] = x + y

# expected-error @+2 {{'where' clauses inside parameter lists are no longer supported}}
# expected-note @+1 {{use a trailing 'where' clause after the signature instead}}
comptime inlineWhereIsError[x: Int where x > 0] = x

# expected-error @below {{cannot implicitly convert 'SIMD[.int, x]' value to 'SIMD[.int, (x * Int(20))]'}}
comptime ComptimeWithParametricType2[x: Int]: SIMD[.int, x*20] = SIMD[.int, x]()

def implicit_generator_constraint_drop_error[cond: Bool]():
    comptime constrained[x: Int]: AnyType where cond = Int
    # expected-error @below {{cannot implicitly convert '__generator_type[x: Int] AnyType where cond' value to '__generator_type[x: Int] AnyType' in comptime initializer}}
    comptime dropped: __generator_type[x: Int] AnyType = constrained

def implicit_generator_constraint_drop_env_mismatch_error[
    cond: Bool, other: Bool
]() where other:
    comptime constrained[x: Int]: AnyType where cond = Int
    # expected-error @below {{cannot implicitly convert '__generator_type[x: Int] AnyType where cond' value to '__generator_type[x: Int] AnyType' in comptime initializer}}
    comptime dropped: __generator_type[x: Int] AnyType = constrained

# Discharging a generator body constraint depends on the assumptions in scope,
# so its convertibility result must never be cached: the convertibility cache is
# keyed only on the (value, required) type pair. The two scopes above failed to
# discharge `where cond` for the `[x: Int] AnyType` generator/body pair; this
# scope proves `cond` and must still succeed for the *same* type pair. If a
# scope-dependent result were cached, the failures above would poison this query
# and produce a spurious "cannot implicitly convert" diagnostic here.
def implicit_generator_constraint_drop_cross_scope[cond: Bool]() where cond:
    comptime constrained[x: Int]: AnyType where cond = Int
    comptime dropped: __generator_type[x: Int] AnyType = constrained


# A target that keeps `condB` is reachable: the source's constraints are all
# dropped and the value is rebound to the target type, so nothing retains a
# mismatched source location. `condA` is provable in this scope, and `condB` is
# assumed because the target promises it.
def implicit_generator_constraint_drop_partial[
    condA: Bool, condB: Bool
]() where condA where condB:
    comptime keepsB[x: Int]: AnyType where condB = Int
    comptime bothConstraints[x: Int]: AnyType where condA where condB = Int
    comptime r: type_of(keepsB) = bothConstraints


# Adding a body constraint the source lacks is fine: constraints are
# contravariant, so a source that demands less satisfies a target that promises
# more.
def implicit_generator_constraint_add[condA: Bool, condB: Bool]() where condA where condB:
    comptime onlyA[x: Int]: AnyType where condA = Int
    comptime bothConstraints[x: Int]: AnyType where condA where condB = Int
    comptime r: type_of(bothConstraints) = onlyA


# The target's constraints are only assumptions, not a licence to ignore the
# source's. `condB` is unprovable here, so the source's `where condA, condB`
# cannot be discharged against a target that only promises `condA`.
def implicit_generator_constraint_unprovable_error[
    condA: Bool, condB: Bool
]() where condA:
    comptime onlyA[x: Int]: AnyType where condA = Int
    comptime bothConstraints[x: Int]: AnyType where condA where condB = Int
    # expected-error @below {{cannot implicitly convert '__generator_type[x: Int] AnyType where condA, condB' value to '__generator_type[x: Int] AnyType where condA' in comptime initializer}}
    comptime r: type_of(onlyA) = bothConstraints

comptime myCurriedIntAdd[x: Int] = myIntAdd[x, ...]

# expected-error @below {{unexpected keyword parameter 'y'}}
comptime myCurriedIntAdd2 = myCurriedIntAdd[y=2]

comptime myRenamedCurriedIntAdd[a: Int] = myCurriedIntAdd[a]

# expected-error @below {{unexpected keyword parameter 'x'}}
comptime myRenamedCurriedIntAdd2 = myRenamedCurriedIntAdd[x=2]

# expected-error @below {{'Int' is not subscriptable}}
comptime mySix = myCurriedIntAdd[2][4][6]

comptime myIntAddTooManyParams = myIntAdd[1, 2,
   3]  # expected-error {{unexpected parameter}}


# COM: A type with dependent parameters.
struct Dep[T: AnyType, v: T](Movable where False):
    pass


comptime MyDep[T: AnyType, v: T] = Dep[T, v]

# expected-error @below {{'T' refers to an unbound parameter in 'MyDep'}}
# expected-note @below {{'MyDep' is aka 'comptime[T: AnyType, v: T] Dep[T, v]'}}
comptime MyDepDotT = MyDep.T

# expected-error @below {{'Dep[_, _]' value has no attribute 'hello'}}
comptime MyDepGetAlias0 = MyDep.hello

# expected-error @below {{'Dep[Int, _]' value has no attribute 'hello'}}
comptime MyDepGetAlias1 = MyDep[Int].hello

# expected-error @below {{'Dep[Int, Int(2)]' value has no attribute 'hello'}}
comptime MyDepGetAlias2 = MyDep[Int, 2].hello


# COM: Using a generator as a struct field type should be rejected (MOCO-3514).
struct FieldWithUnboundAlias(Movable where False):
    # expected-error @below {{'MyDep' is not a concrete type, use '[]' to bind missing parameters}}
    # expected-note @below {{'MyDep' is aka 'comptime[T: AnyType, v: T] Dep[T, v]'}}
    var f: MyDep


def test_variable_type_parameterization():
    # Store an unparameterized struct type in a variable...
    # expected-error @below {{dynamic type values not permitted yet; try creating a 'comptime' instead of a 'var'}}
    var struct_type = Dep

    # .. and try to parameterize it.
    # expected-error @below {{types are not subscriptable}}
    var instance: struct_type[Int]


##===----------------------------------------------------------------------===##
# Trailing 'where' constraints are enforced when the alias's parameters are
# inferred during auto-parameterization, not just for explicit bindings
# (MOCO-4081). This covers every form that auto-parameterizes the alias
# generator: argument types, value-parameter types, and variadic element types.
##===----------------------------------------------------------------------===##


@fieldwise_init
struct Tag[n: Int](Copyable, Movable):
    pass


# A distinct alias is used per form so each violated call's note points at its
# own 'where' clause (identical notes at one location are coalesced by the
# diagnostic verifier).

# Alias used as a function argument type.
# expected-note @+1 {{constraint declared here evaluated to False, expected '(n > Int(0))'}}
comptime PositiveArg[n: Int, //] where n > 0 = Tag[n]


# expected-note @+1 {{function declared here}}
def take_arg(p: PositiveArg):
    pass


# Alias used as a value-parameter type.
# expected-note @+1 {{constraint declared here evaluated to False, expected '(n > Int(0))'}}
comptime PositiveParam[n: Int, //] where n > 0 = Tag[n]


# expected-note @+1 {{function declared here}}
def take_param[p: PositiveParam]():
    pass


# Alias used as a variadic argument element type.
# expected-note @+1 {{constraint declared here evaluated to False, expected '(n > Int(0))'}}
comptime PositiveVar[n: Int, //] where n > 0 = Tag[n]


# expected-note @+1 {{function declared here}}
def take_variadic(*p: PositiveVar):
    pass


def use_ok():
    # Inferred bindings that satisfy the constraint are accepted in every form.
    take_arg(Tag[1]())
    take_param[Tag[1]()]()
    take_variadic(Tag[1](), Tag[1]())


def use_arg_bad():
    # expected-error @+1 {{invalid call to 'take_arg': violated constraint}}
    take_arg(Tag[-1]())


def use_param_bad():
    # expected-error @+1 {{invalid call to 'take_param': violated constraint}}
    take_param[Tag[-1]()]()


def use_variadic_bad():
    # expected-error @+1 {{invalid call to 'take_variadic': violated constraint}}
    take_variadic(Tag[-1]())


##===----------------------------------------------------------------------===##
# Handle type alias with extra constraints
##===----------------------------------------------------------------------===##

struct Iter[Cond: Bool](Movable where False):
    # expected-note @+1 {{function declared here}}
    def __init__(out self: Iter[False]):
        pass


struct Collection[Cond: Bool]:
    # expected-note @+1 {{constraint declared here evaluated to False, expected 'Cond'}}
    comptime Alias[Cond: Bool]: AnyType where Cond = Iter[Cond]

    def iter(self) where Self.Cond:
        # Inferring to `Iter[False]` violates the constraint imposed on `Alias`
        #
        # expected-error @+1 {{invalid initialization: violated constraint}}
        _ = Self.Alias()
