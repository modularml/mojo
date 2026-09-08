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

# RUN: %parse-mojo-isolated -verify-diagnostics %s -I=%S/inputs

struct HasIntParam[x: Int](Movable where False):
  def __init__(out self): pass


struct _MLIR(Movable where False):
    comptime KGENParamListType[elt_type: AnyType] = __mlir_type[
        `!kgen.param_list<`, elt_type, `>`
    ]

##===----------------------------------------------------------------------===##
# Functions
##===----------------------------------------------------------------------===##

def missing_raises_is_ok():
  pass

struct NotBoolConvertible(Movable where False): pass
# expected-note @+1 {{function declared here}}
def test_bool_context(a: NotBoolConvertible): pass

def voidReturningFn(): pass
def badCall():
  # expected-error @+1 {{invalid call to 'test_bool_context': value passed to 'a' cannot be converted from 'None' to 'NotBoolConvertible'}}
  test_bool_context(voidReturningFn())


def missing_ret_val() -> __mlir_type.index:
  return # expected-error {{cannot implicitly convert 'None' value to '__mlir_type.index' in return value}}

def ret_type_mismatch() -> __mlir_type.index:
  return 4.0 # expected-error {{cannot implicitly convert 'FloatLiteral[4]' value to '__mlir_type.index' in return value}}

async def testAsyncVoid(): pass
async def testAsyncInt() -> Int: return 42

def callsWith():
  testAsyncVoid() # expected-warning {{'Coroutine[None, {}]' value is not awaited; use 'await' to get its result}}
  testAsyncInt() # expected-warning {{'Coroutine[Int, {}]' value is not awaited; use 'await' to get its result}}


struct ThingWithStaticMethod(Movable where False):
   @staticmethod
   def splat(x: Int):
     pass

def testThingWithStaticMethod():
  ThingWithStaticMethod.splat(4.0)


def top_level_fn(a: Int) raises:
    def bar[b: Int]() -> Int:
      # expected-error @below {{Could not infer capture convention of the captured value a}}
      return a

def use_non_copyable_type(a: ThingWithStaticMethod) raises:
  pass

def test_use_non_copyable_type(var b: ThingWithStaticMethod) raises:
  use_non_copyable_type(b^)


@fieldwise_init
struct MemType(ImplicitlyCopyable):
    pass

# COM: Issue https://github.com/modularml/modular/issues/37758 where the
# COM: key test is that the below is not crashing due to assertion violation.
# expected-error @+1 {{expected ':' in function definition}}
def missingColon(x: Int)
  return x

def out1(a: Int, out b: Int): pass

# expected-error @+1 {{'out' convention must not be variadic}}
def bad_out2(out *b: Int): pass

# expected-error @+1 {{expected ']' for parameter list}}
def bad_out3[out x: Int](): pass

# expected-error @+1 {{function may not have multiple 'out' arguments}}
def bad_out4(out a: Int, out b: Int): pass

# expected-error @+1 {{functions must not declare both an 'out' argument and a return type}}
def bad_out5(out a: Int) -> Int: pass

# expected-error @+1 {{functions must not declare both an 'out' argument and a return type; remove the '-> None' to fix it}}
def bad_out6(out self) -> None: pass

struct BadInitResult(Movable where False):
  # expected-error @+1 {{__init__ method must return Self type with 'out' argument}}
  def __init__(mut self) raises -> None:
    pass

struct BadInitType(Movable where False):
    # expected-error @below {{__init__ method must return Self type with 'out' argument}}
    # expected-error @below {{self argument must be present in instance method}}
    def __init__():
        pass

# expected-error @+1 {{argument type must be specified}}
def defaultArgumentUntyped(a=1) raises:
    pass

# expected-error @+1 {{'abi("C")' function may not be marked 'raises'; remove 'raises' or use 'abi("Mojo")'}}
def c_raises() abi("C") raises:
    pass

##===----------------------------------------------------------------------===##
# Default Arguments, VarArgs, and Packs
##===----------------------------------------------------------------------===##

# COM: Issue https://github.com/modular/mojo/issues/1091
def missing_arg_type_or_default(
    a: Int = 9,
    # expected-error @+2 {{required positional argument follows optional positional argument; change the ordering}}
    # expected-error @+1 {{argument type must be specified}}
    b,
    c: Int,  # expected-error {{required positional argument follows optional positional argument; change the ordering}}
    d: Int = 0,
    # expected-error @+2 {{required positional argument follows optional positional argument; change the ordering}}
    # expected-error @+1 {{argument type must be specified}}
    e,
    # expected-error @+1 {{argument type must be specified}}
    var **kwargs,
):
    pass

def missing_default(
    a: Int=9,
    b: Int,  # expected-error {{required positional argument follows optional positional argument; change the ordering}}
    c: Int=0,
    d: Int,  # expected-error {{required positional argument follows optional positional argument; change the ordering}}
) raises:
    pass

# expected-error @+1 {{use of unknown declaration 'unknown'}}
def defaultArgumentUnknownDeclaration(a: Int = unknown): pass

# expected-error @+1 {{cannot use a dynamic value in default argument}}
def defaultArgumentReferencesArgument(a: Int = 0, b: Int = a): pass

def defaultArgumentBadType(a: Int = 1.0): pass

def defaultArgumentBadType2[T: AnyType](a: T = 1.0): pass
def callDefaultArgumentBadType2():
  defaultArgumentBadType2[Int]()
  defaultArgumentBadType2[Float64]()

# expected-error @+1 {{invalid call to '__init__': no candidates found}}
def test_contextual_default[T: AnyType](copies: T = {}):
    pass

# this should work.
def test_contextual_default2[i: Int](x: HasIntParam[i] = {}):
    pass

# expected-error @+1 {{'mut' arguments must not have defaults}}
def byref_default(mut x: Int = 2): pass

# expected-error @below {{'**' marker must be at end of argument list}}
def starStarLast(var **a: Int, b: Int): pass

# expected-error @below {{'**' marker must be at end of argument list}}
def twoStarStar(var **a: Int, var **b: Int): pass

# expected-error @+1 {{expected argument name}}
def starSpaceStar(* *a: Int): pass

# expected-error @+1 {{variadic arguments must not have defaults}}
def noDefaultVariadics(*a: Int = 42): pass

def exampleVariadic(a: FloatLiteral, *b: Int): pass
def exampleVariadicAndKeyword(*a: Int, b: Int): pass
# expected-note @+1 {{function declared here}}
def exampleByRefVariadic(a: FloatLiteral, mut *b: Int): pass
# expected-note @+1 {{function declared here}}
def parameterizedVariadic[T: TrivialRegisterPassable](*args: T): pass

def ownedPack[*Ts: AnyType](var *args: *Ts): pass
def ownedVariadic(var *args: Inner): pass
def ownedVariadicReg(var *args: WrongType): pass


# expected-note @+1 {{struct declared here}}
struct ParameterizedStruct[T: TrivialRegisterPassable](Movable where False):
    # expected-note @+1 {{function declared here}}
    def __init__(out self, *args: Self.T) raises:
        pass

struct BadResultParams[a: Int](Movable where False):
  def __init__(out self: Self): pass # Ok.
  # expected-error @+1 {{cannot use Self parameter 'a' in constructor whose result defines it to '(a + Int(1))'}}
  def __init__(x: Int, out self: BadResultParams[Self.a + 1]): pass
  # expected-error @+1 {{cannot use Self parameter 'a' in constructor whose result defines it to 'Int(4)'}}
  def __init__(x: BadResultParams[Self.a], out self: BadResultParams[4]): pass
  # expected-error @+1 {{cannot use Self parameter 'a' in constructor whose result defines it to 'Int(7)'}}
  def __init__[p: BadResultParams[Self.a]](*, y: Int, out self: BadResultParams[7]):
      pass

def useBadResultParams():
  # These shouldn't generate errors.
  _ = BadResultParams[1](1)
  _ = BadResultParams()
  _ = BadResultParams[1]()

@fieldwise_init
struct TestTuple[*Ts: __mlir_type.`!kgen.type`](Movable where False):
    # expected-note @+1 {{function declared here}}
    def test[i: Int, j: Int](self):
        pass

def badCalls(arg: Int):
  exampleVariadic(1.0, 1.0)
  exampleVariadic(1.0, 1, 2, 1.0)
  exampleVariadicAndKeyword(1, 2, 3, b=4.0)

  var x: Int
  var y : FloatDyn
  # expected-error @+1 {{invalid call to 'exampleByRefVariadic': value passed to mutable argument 'b' must be mutable}}
  exampleByRefVariadic(1.0, x, arg)
  # expected-error-re @+1 {{invalid call to 'exampleByRefVariadic': l-value of type 'FloatDyn' cannot be converted to reference of type 'Int'}}
  exampleByRefVariadic(1.0, x, y)
  # expected-error @+1 {{value passed to mutable argument 'b' must be mutable}}
  exampleByRefVariadic(1.0, x, 1)

  # The user hasn't provided any arguments that could be used to infer `T`.
  # expected-error @below {{failed to infer parameter 'T'}}
  parameterizedVariadic()
  # expected-error @below {{failed to infer parameter 'T' of parent struct 'ParameterizedStruct'}}
  var z = ParameterizedStruct()

  # We can't infer `T` with two arguments of different types.
  parameterizedVariadic(1, 2.0)

  # expected-error @below {{invalid call to 'test': failed to infer parameter 'j'}}
  TestTuple[Int, FloatLiteral]().test[1]()

def badError(a: ParameterizedStruct[Int]):
  # expected-error @+1 {{cannot implicitly convert 'ParameterizedStruct[Int]' value to 'ParameterizedStruct[Bool]'}}
  var b: ParameterizedStruct[Bool] = a

# expected-note @below {{candidate declared here}}
def overloadedFunc(x: Int): pass
# expected-note @below {{candidate declared here}}
def overloadedFunc(x: Int, y: Int): pass

# expected-note @below {{function declared here}}
def takeFuncArgument(f: Int): pass

def callWithOverloadedArg():
  # expected-error @below {{cannot convert function to non-function type 'Int'}}
  takeFuncArgument(overloadedFunc)

# expected-error @+1 {{unexpected token in expression}}
def invalidStarExpression(*x: *): pass

# expected-error @+1 {{pack argument type list must reference a variadic list}}
def invalidPackType(*x: *Int): pass

def invalidParameterPack[*Ts: AnyType]():
  # expected-error @+1 {{parameters must not be variadic packs}}
  def invalid[*Us: *Ts](): pass

# expected-error @+2 {{variadic unpacking with '*' requires a variadic argument}}
# expected-note @+1 {{'x' is not a variadic argument}}
def invalidArgumentUnpack[*Ts: AnyType](x: *Ts): pass

# expected-error @+1 {{argument already has a convention specified}}
def invalidOwned(var var x: Int): pass

# expected-note @+1 {{function declared here}}
def examplePack[*Ts: AnyType](*args: *Ts):
  pass

def packArgOverload():
  pass

def packArgOverload(x: Int):
  pass


# expected-note @+1 {{function declared here}}
def first_and_rest[T: TrivialRegisterPassable, *Ts: AnyType](*values: *Ts):
    pass


def unresolvedPackCall[*t : AnyType](var *args: *t):
  # expected-error @below {{assigning 0 operands to an unresolvable variadic pack argument}}
  var _ = examplePack[*t]()

def badPackCalls(value: Int):
  # expected-error @+1 {{expected 1 element in variadic pack, got 2 argument values}}
  examplePack[Int](1, 2)
  # expected-error @+1 {{expected 2 elements in variadic pack, got 1 argument value}}
  examplePack[Int, FloatDyn](1)
  # expected-error-re @below {{invalid call to 'examplePack': value passed to 'args' cannot be converted from '__mlir_type.`!kgen.scalar<index>`' to 'FloatDyn'}}
  examplePack[Int, FloatDyn](1, Int(2)._mlir_value)
  # expected-error @below {{invalid call to 'examplePack': could not infer type of parameter pack 'args' given value with unresolved type}}
  examplePack(packArgOverload)
  # expected-error @below {{invalid call to 'first_and_rest': failed to infer parameter 'T'}}
  first_and_rest(value)

struct TestPackErrorMessage[*Ts: AnyType](Movable where False):
    # expected-error @+3 {{'self' argument must have type 'TestPackErrorMessage[Ts]', but actually has type 'VariadicPack[False, Ts]'}}
    # expected-error @+2 {{__init__ method must return Self type with 'out' argument}}
    @__allow_legacy_custom_self_type
    def __init__(*args: *Self.Ts):
         pass

# always_inline("builtin")

# expected-error @+2 {{'@always_inline("builtin")' does not support this argument convention}}
@always_inline("builtin")
async def always_inline_builtin_1(): pass

# expected-error @+2 {{'@always_inline("builtin")' does not support this argument convention}}
@always_inline("builtin")
def always_inline_builtin_2(a: MemType): pass

# expected-error @+2 {{'@always_inline("builtin")' does not support this argument convention}}
@always_inline("builtin")
def always_inline_builtin_3() raises: pass

@always_inline("builtin")
def always_inline_builtin_4(a: Bool):
  # expected-error @+1 {{'@always_inline("builtin")' can only handle single-result if}}
  if a:
     pass

# expected-error @+1 {{'where' clauses must be used with parameters and cannot be used with arguments}}
def illegal_runtime_where[x: Int](a: Int where a > 1):
  pass

# expected-note @+1 {{function declared here}}
def simple_constraints[x: Int, y: Int]()
  # expected-note @+1 {{constraint declared here}}
  where x > 1
  # expected-note @+1 {{constraint declared here}}
  where y < 10:
    pass

# expected-note @below {{cannot evaluate call to non-builtin function declared here}}
def unfoldable_predicate(y: Int) -> Bool:
  return y > 2

@always_inline("builtin")
def foldable_predicate(y: Int) -> Bool:
  return y > 10

# expected-note @below {{cannot prove constraint for candidate}}
# expected-note @below {{constraint declared here needs evidence for 'foldable_predicate(x)'}}
def need_foldable_predicate[a: Int]() where foldable_predicate(a):
  pass

def call_need_foldable_predicate[x: Int]():
  # expected-error @below {{invalid call to 'need_foldable_predicate': lacking evidence to prove correctness}}
  # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
  need_foldable_predicate[x]()

# expected-note @below {{function declared here}}
# expected-note @below {{cannot prove constraint}}
def unprovable_constraints[x: Int, y: Int]()
  # expected-note @+1 {{constraint declared here evaluated to False, expected '(x > Int(1))'}}
  where x > 1
  # expected-note @+1 {{constraint declared here needs evidence for 'unfoldable_predicate(Int(0))'}}
  where unfoldable_predicate(y):
    pass

comptime IntSumReducerdef[Prev: Int, From: _MLIR.KGENParamListType[Int], Idx: __mlir_type.index]: Int = Prev + ParameterList[From]()[Int(mlir_value=Idx)]

comptime IntSumReducer[Variadic: _MLIR.KGENParamListType[Int]] = __mlir_attr[
  `#kgen.param_list.reduce<`,
  Int(0),  # base
  `,`,
  Variadic,
  `,`,
  IntSumReducerdef,
  `> : `,
  Int,
]

# expected-note @below {{function declared here}}
# expected-note @below {{constraint declared here evaluated to False}}
def failing_constraint_with_depth[*vals: Int]() where IntSumReducer[vals.values] == 9:
  pass

def test_constraints():
  # expected-error @+1 {{violated constraint}}
  simple_constraints[0, 0]()
  # expected-error @+1 {{violated constraint}}
  simple_constraints[2, 11]()
  # expected-error @+1 {{violated constraints}}
  simple_constraints[0, 11]()

  # expected-error @+1 {{violated constraint}}
  unprovable_constraints[0, 0]()
  # expected-error @below {{invalid call to 'unprovable_constraints': lacking evidence to prove correctness}}
  # expected-note @below {{provide evidence for the constraint here to aid in candidate selection}}
  unprovable_constraints[2, 0]()

  # expected-error @below {{invalid call to 'failing_constraint_with_depth': violated constraint}}
  failing_constraint_with_depth[1,2,3]()

# expected-note @below {{function declared here}}
def unprovable_param_constraints[x: Int]()
  # expected-note @below {{constraint declared here evaluated to False, expected '(x > Int(1))'}}
  where x > 1:
  pass

def test_param_constraints():
  # expected-error @below {{invalid call to 'unprovable_param_constraints': violated constraint}}
  unprovable_param_constraints[0]()

# expected-error @below {{'where' clauses inside parameter lists are no longer supported}}
# expected-note @below {{use a trailing 'where' clause after the signature instead}}
def inline_where_is_error[x: Int where x > 0]():
  pass

# A function type carries its constraints after the result type, so the note
# points there rather than at the enclosing signature.
def fn_type_inline_where[
  # expected-error @below {{'where' clauses inside parameter lists are no longer supported}}
  # expected-note @below {{use a trailing 'where' clause after the result type of a 'thin' function type instead}}
  f: def[a: Int where a > 0]() thin -> None
]():
    pass

# expected-note @below {{'ConstraintStruct' declared here}}
# expected-note @below {{constraint declared here needs evidence for '(x > Int(0))'}}
# expected-note @below {{constraint declared here evaluated to False, expected '(a > Int(0))'}}
struct ConstraintStruct[a: Int] (Movable where False) where a > 0:
    pass

# expected-note @below {{add a trailing 'where' clause that requires '(x > Int(0))'}}
def use_constraint_struct[x: Int, cs: ConstraintStruct[x]]():
    # expected-error @-1 {{invalid bindings in signature: lacking evidence to prove correctness}}
    pass

# expected-error @below {{violated constraint}}
def violate_constraint_struct[cs: ConstraintStruct[0]]():
    pass

# expected-note @below {{function declared here}}
# expected-note @below {{constraint declared here evaluated to False, expected '(a > Int(0))'}}
# expected-note @below {{cannot prove constraint}}
# expected-note @below {{constraint declared here needs evidence for '(x > Int(0))'}}
def constraint_fn[a: Int, b: Int]() where a > 0:
    pass

def use_constraint_fn[x: Int]():
    # expected-error @below {{violated constraint}}
    comptime f0 = constraint_fn[0, 1]
    # expected-error @below {{invalid bindings for 'constraint_fn': lacking evidence to prove correctness}}
    comptime fx = constraint_fn[x, 1]

# Defaults are checked against a trailing 'where' clause when the function is
# instantiated using them. A default that can never satisfy the constraint is
# reported at the call that relies on it.
# expected-note @below {{function declared here}}
# expected-note @below {{constraint declared here evaluated to False, expected '(x > Int(3))'}}
def violated_default_constraint[x: Int = 1]() where x > 3:
    pass

def use_violated_default_constraint():
    # expected-error @below {{invalid call to 'violated_default_constraint': violated constraint}}
    violated_default_constraint()

# A default that depends on another parameter cannot be rejected at declaration
# time; using the defaults here satisfies the constraint, so there should NOT be
# any errors.
def unprovable_default_constraint[x: Int = 3, y: Int = 1]() where x + y > 3:
    pass

def use_unprovable_default_constraint():
    unprovable_default_constraint()

##===----------------------------------------------------------------------===##
# Function Overloading
##===----------------------------------------------------------------------===##

# expected-note @+1 {{previous definition here}}
def fn_redecl(): pass
# expected-error @+1 {{redefinition of function 'fn_redecl' with identical signature}}
def fn_redecl(): pass

# expected-note @+1 {{previous definition here}}
def fn_redecl2() raises -> Int: pass
# expected-error @+1 {{redefinition of function 'fn_redecl2', cannot overload on return type}}
def fn_redecl2() -> FloatDyn: pass

# expected-note @below {{candidate declared here}}
# expected-note @below {{candidate not viable: value passed to 'a' cannot be converted from 'TestOverloading' to 'Int'}}
# expected-note @below {{candidate not viable: unexpected argument}}
def overloadIntFloat32(a: Int): pass

# expected-note @below {{candidate declared here}}
# expected-note-re @below {{candidate not viable: value passed to 'a' cannot be converted from 'TestOverloading' to 'FloatDyn'}}
# expected-note @below {{candidate not viable: unexpected argument}}
def overloadIntFloat32(a: FloatDyn): pass

# expected-note @below {{candidate declared here}}
# expected-note @below {{candidate not viable: missing required argument: 'b'}}
# expected-note-re @below {{candidate not viable: value passed to 'b' cannot be converted from 'FloatDyn' to 'Int'}}
def overloadIntFloat32(a: Int, b: Int): pass

# expected-note @below {{candidate declared here}}
# expected-note @below {{candidate not viable: missing required argument: 'b'}}
# expected-note @below {{value passed to mutable argument 'b' must be mutable}}
def overloadIntFloat32(a: Int, mut b: FloatDyn): pass

# expected-note @below {{candidate not viable: missing required argument: 'b'}}
# expected-note @below {{candidate not viable: missing required argument: 'c'}}
# expected-note @below {{candidate declared here}}
def overloadIntFloat32(a: Int, mut b: FloatDyn, c: Int, *args: Int): pass

struct TestOverloading(Movable where False):
  var a: Int   # expected-note {{cannot overload with this non-function definition}}
  def a(self):  # expected-error {{invalid redefinition of 'a'}}
    pass

  def test(self, a: Int, b: FloatDyn) raises:
    # expected-note @below {{add '()' to call the function}}
    # expected-error @below {{cannot form a reference to overloaded declaration}}
    var bad = overloadIntFloat32

    # expected-error @+1 {{no matching function in call}}
    overloadIntFloat32(self)
    # expected-error @+1 {{no matching function in call}}
    overloadIntFloat32(a, b)

@fieldwise_init
struct OverloadedKwArgs(Movable where False):
    var vals: List[Int]

    # expected-note @below {{previous definition here}}
    def __getitem__(self, idx: Int) -> Int:
        return self.vals[idx]

    # expected-error @below {{redefinition of function '__getitem__', cannot overload on return type}}
    def __getitem__(self, idx: Int) -> Bool:
        return self.vals[idx] > 0

# Test that static methods don't get dispatched if their first arg is self type.
struct StructWithStaticMethod(Movable where False):
    def __init__(out self): pass

    # expected-note @+2 {{function declared here}}
    @staticmethod
    def bar(mut f: StructWithStaticMethod): pass


def test_static_overload():
    var a = StructWithStaticMethod()
    # expected-error @below {{invalid call to 'bar': missing required argument: 'f'}}
    a.bar()


# expected-note @+1 {{function declared here}}
def takesAtLeastOneInt(x: Int, *y: Int): pass
def badTakesAtLeastOneInt():
  # expected-error @+1 {{invalid call to 'takesAtLeastOneInt': missing required argument: 'x'}}
  takesAtLeastOneInt()


# COM: Issue #23007
# expected-note @+1 {{function declared here}}
def too_few_pos_only(a: Int, b: Int, /, msg: Int = 2): pass

def test_too_few_pos_only(a: Int, msg: Int = 3):
  # expected-error @+1 {{invalid call to 'too_few_pos_only': missing required argument: 'b'}}
  too_few_pos_only(a, msg=msg)


# COM: Issue #23007
# expected-note @+1 {{function declared here}}
def missing_args(a: Int, b: Int, c: Int = 2, d: Int = 2): pass

def test_missing_args():
  # expected-error @+1 {{invalid call to 'missing_args': missing required argument: 'a'}}
  _ = missing_args(c=1, d=1)


struct ConvertibleFromInt(Movable where False):
  @implicit
  def __init__(out self, value: Int):
    pass

# expected-note @below {{candidate declared here}}
# expected-note @below {{candidate not viable: value passed to 'b' cannot be converted from 'ConvertibleFromInt' to 'Int'}}
def ambiguousConversions(a: ConvertibleFromInt, b: Int): pass
# expected-note @below {{candidate declared here}}
# expected-note @below {{candidate not viable: value passed to 'a' cannot be converted from 'ConvertibleFromInt' to 'Int'}}
def ambiguousConversions(a: Int, b: ConvertibleFromInt): pass

def testAmbiguousConversions(a: Int, b: ConvertibleFromInt):
  ambiguousConversions(a, b) # ok
  ambiguousConversions(b, a) # ok
  # expected-error @+1 {{ambiguous call to 'ambiguousConversions', each candidate requires 1 implicit conversion, disambiguate with an explicit cast}}
  ambiguousConversions(a, a)
  # expected-error @+1 {{no matching function in call}}
  ambiguousConversions(b, b)

  var localdef = testAmbiguousConversions  # ok
  localdef(a, b)
  localdef(a, a)
  localdef(1, b)
  # expected-error @+1 {{invalid indirect call: value passed to 'a' cannot be converted from 'ConvertibleFromInt' to 'Int'}}
  localdef(b, b)

# COM: https://github.com/modular/mojo/issues/1530
# COM: Do not crash when explicitly unbound parameter cannot be deduced due to missing arguments.
struct Parametric[a: Int](Movable where False): pass

# expected-note @below {{function declared here}}
def takes_same_arg_types[x: Int](a: Parametric[x], b: Parametric[x]): pass

def test_param_deduction_failure[
    func: def[y: Int] (c: Parametric[y], d: Parametric[y]) thin -> None,
](u: Int, v: Int):
    # expected-error @+1 {{invalid call to 'takes_same_arg_types': missing required argument: 'b'}}
    takes_same_arg_types[_](u)

    # expected-error @below {{invalid call to 'takes_same_arg_types': value passed to 'a' cannot be converted from 'Int' to 'Parametric[x]', it depends on an unresolved parameter 'x'}}
    takes_same_arg_types[_](u, v)

    # expected-error @+1 {{invalid indirect call: missing required argument: 'd'}}
    func[_](u)

    # TODO: This is because we're not inferring signatures correctly
    # expected-error @below {{invalid indirect call: value passed to 'c' cannot be converted from 'Int' to 'Parametric[func]', it depends on an unresolved parameter 'func'}}
    func[_](u, v)

struct InitOverloaded(Movable where False):
  # expected-note @below {{value passed to 'a' cannot be converted from 'StringLiteral["foo"]' to 'Int'}}
  # expected-note @below {{value passed to 'a' cannot be converted from 'Parametric[Int(1)]' to 'Int'}}
  def __init__(out self, a: Int): pass
  # expected-note @below {{value passed to 'a' cannot be converted from 'StringLiteral["foo"]' to '__mlir_type.index'}}
  # expected-note @below {{value passed to 'a' cannot be converted from 'Parametric[Int(1)]' to '__mlir_type.index'}}
  def __init__(out self, a: __mlir_type.index): pass

def testOverloadInitError(a: InitOverloaded, b: Parametric[1], c: Int):
  # expected-error @+1 {{cannot construct 'InitOverloaded' with itself, you can remove the constructor call}}
  _ = InitOverloaded(a)

  # expected-error @+1 {{no matching function in initialization}}
  _ = InitOverloaded(b)

  # This is ok
  _ = InitOverloaded(c)

  # expected-error @+1 {{no matching function in initialization}}
  _ = InitOverloaded("foo")

  # Ambiguous initializer list assigning to discard pattern needs to be an error.
  # expected-error @+1 {{cannot emit initializer list without a contextual type}}
  _ = {a = 1, b = 2}


##===----------------------------------------------------------------------===##
# Structs
##===----------------------------------------------------------------------===##

# expected-note @below {{by declaration 'Rec'}}
# expected-error @below {{attempt to resolve a recursive reference to declaration 'Rec'}}
# expected-note @below {{referenced through this use}}
struct Rec[
  # expected-note @below {{referenced from here}}
  param: Rec]:
  pass

struct EarlySelf[
  # expected-error @below {{'Self' type is not available in this context}}
  param: Self]:
  pass

# expected-error @+2 {{attempt to resolve a recursive reference to declaration 'Rec1'}}
# expected-note @+1 {{referenced through this use}}
struct Rec1[# expected-note {{by declaration 'Rec1'}}
 # expected-note @below {{referenced through this use}}
  p1: Rec2]: pass

# expected-note @below {{by declaration 'Rec2'}}
struct Rec2[
  # expected-note @below {{referenced from here}}
  p2: Rec1]:


# expected-error @+1 {{'def' statement must be on its own line}}
struct Struct(Movable where False): def foo(mut self): pass

struct ReturnFromStruct(Movable where False):
  # expected-error @+1 {{'return' must be inside a function; move this into a function body}}
  return 42

struct ReDef: pass # expected-note {{conflicts with this previous struct declaration}}
struct ReDef: pass # expected-error {{invalid redefinition of 'ReDef'}}

# Ambiguous Lookup Case for Referencing Redefined Struct (ALCFRRS)
# This tests that we don't crash or anything when we reference a redefined
# struct.
def reference_redefined_struct(arg: ReDef):
  pass

struct StructMemberRedefinition(Movable where False):
  var x : __mlir_type.index  # expected-note {{previous definition here}}
  var x : __mlir_type.index  # expected-error {{invalid redefinition of 'x'}}

struct SpecialFunctions(Movable where False):
  # expected-error @+1 {{'__new__' is not supported on structs}}
  def __new__() -> Self:
    pass

  # expected-error @+1 {{'__add__' requires 2 operands}}
  def __add__(self):
    pass

  # Issue #7573: an LValue/RValue pair of a special method differs only in
  # argument conventions, so the declarations collide instead of forming an
  # overload set.
  # expected-note @+1 {{previous definition here}}
  def __iadd__(a: SpecialFunctions, b: SpecialFunctions): pass

  # expected-error @+2 {{'__iadd__' result type must be elided (or None)}}
  # expected-error @+1 {{redefinition of function '__iadd__', cannot overload on return type}}
  def __iadd__(mut self, rhs: SpecialFunctions) -> SpecialFunctions: pass

  def failures(self):
    self+self # Supports this, even though it isn't valid.  Shouldn't crash.
    self*self # expected-error {{'SpecialFunctions' does not implement the '__mul__' method}}

  # expected-error @+1 {{destructor must not declare 'raises'; remove the 'raises' keyword}}
  def __deinit__(self) raises:
     pass

struct TestOwnedDeinitErrors(Movable where False):
  # 'owned' is no longer a keyword; it is parsed as parameter name, then 'self' is unexpected
  # expected-error @+1 {{expected ')' in argument list}}
  def __deinit__(owned self): pass

  # expected-error @+1 {{expected ')' in argument list}}
  def __moveinit__(out self, owned x: TestOwnedDeinitErrors): pass

  # expected-error @+1 {{expected ')' in argument list}}
  def method(owned x): pass

# expected-note @+1 {{previous definition here}}
struct WrongType(RegisterPassable):
  # expected-error @+3 {{__init__ method must return Self type with 'out' argument}}
  # expected-error @+2 {{'self' argument must have type 'WrongType', but actually has type 'None'}}
  @__allow_legacy_custom_self_type
  def __init__(self: None) raises: pass

  # expected-error @+1 {{'self' argument must have type 'WrongType', but actually has type 'Int'}}
  def __init__(out self: Int): pass

  # TODO: Should err.
  def __init__(out self, *, copy: Int): pass

  # expected-error @+2 {{redefinition of function '__init__', cannot overload on return type}}
  # expected-error @+1 {{'RegisterPassable' types must not declare explicit move constructors; values of these types have no identity, and the compiler can freely move them between registers}}
  def __init__(out self, *, deinit move: Self): pass


struct WrongSelfType[a: Int](Movable where False):
  # expected-error @+2 {{'self' argument must have type 'WrongSelfType[a]', but actually has type 'Int'}}
  @__allow_legacy_custom_self_type
  def badMethod(self: Int): pass
  def goodMethod(mut self: WrongSelfType[Self.a]): pass

  # Issue #13358
  def __init__(out self, *, copy: Self, moar: Int): pass

  # expected-error @+1 {{'__add__' requires 2 operands}}
  def __add__(self): pass

  def __pow__(self, exp: Int): pass

  def __pow__(self, exp: Int, mod: Int): pass

  # expected-error @+1 {{'__pow__' requires at least 2 operands}}
  def __pow__(self): pass

  # expected-error @+1 {{'__pow__' requires at most 3 operands}}
  def __pow__(self, exp: Int, mod: Int, extra: Int): pass

# Issue #6587: [Lit] Recursive constructors crash kgen
struct BadInit[size: __mlir_type.index](Movable where False):
  @implicit
  def __init__(out self, elem: BadInit[Int(1).__mlir_index__()]) raises:
    var x : __mlir_type[`!kgen.simd<`, Self.size, `, FloatDyn>`]
    # expected-error @+1 {{cannot implicitly convert '__mlir_type.`!kgen.simd<size, FloatDyn>`' value to 'BadInit[size]'}}
    self = x

struct MLIRAttrWithinStruct(Movable where False):
  # expected-error @below {{bare expressions must not appear within structs; use 'var' for a field or move this into a method body}}
  __mlir_attr.`#index.cmp_predicate<eq>`

trait BareExprInTrait:
  # expected-error @below {{bare expressions must not appear within traits; use 'comptime' for an associated type or move this into a method body}}
  1 + 1

struct BareExprInExtension(Movable where False):
  pass

__extension BareExprInExtension:
  # expected-error @below {{bare expressions must not appear within extensions; move this into a method body}}
  1 + 1


# In register structs may only have stored properties of other in-reg values.
struct InMemStruct(Movable where False): pass

# expected-error @+1 {{all members of 'RegisterPassable' struct must themselves be 'RegisterPassable'}}
struct InRegStruct(RegisterPassable):
  var x: Int # ok
  # expected-error @+1 {{cannot synthesize move constructor because field 'y' has non-movable and non-implicitly-copyable type 'InMemStruct'}}
  var y: InMemStruct # expected-note {{'y' declared with type 'InMemStruct'}}

struct OtherInMemStruct(Movable where False):
  var x: Int # ok
  var y: InMemStruct # ok


# expected-note @+1 {{previous definition here}}
struct InvalidMember(TrivialRegisterPassable):
  var x: __mlir_type.index
  # expected-error @+1 {{'RegisterPassable' types must not declare explicit move constructors; values of these types have no identity, and the compiler can freely move them between registers}}
  def __init__(out self, *, deinit move: Self): pass
  # expected-error @+2 {{redefinition of function '__init__', cannot overload on return type}}
  # expected-error @+1 {{trivial types must not declare explicit copy constructors; they are trivially copyable}}
  def __init__(out self, *, copy: Self): pass
  # expected-error @+2 {{trivial types must not declare '__deinit__' methods; they are trivially destroyable}}
  # expected-note @+1 {{trivial types have no identity; the compiler destroys them automatically with no observable effect}}
  def __deinit__(self): pass

def noop():  # expected-error {{body must not be empty; use 'pass' or check that the lines below are indented}}

struct BadDtor1(Movable where False):
  def __deinit__(mut self): # expected-error {{'self' argument must be passed as 'deinit'}}
    pass

  # expected-error @+1 {{'deinit' must only be applied to arguments of Self type}}
  def bad1(self, deinit x: Int): pass
  # expected-error @+1 {{'self' argument must not be variadic; remove '*' before 'self'}}
  def bad2(deinit *self): pass
  # expected-error @+1 {{'self' argument must not be variadic; remove '*' before 'self'}}
  def bad3(*self): pass

# expected-error @+1 {{'deinit' convention is only valid on struct method arguments}}
def invalid_deinit(deinit self: Int) raises:
    pass

struct GoodDtor(Movable where False):
   def __deinit__(deinit self): pass
   def explicit_dtor(deinit self): pass
   def explicit_dtor2(deinit self, deinit other: Self): pass
   def normal_var(var self): pass

struct BadDtorImplicitSelf(Movable where False):
  # expected-error @+1 {{'self' argument must be passed as 'deinit'}}
  def __deinit__(self): pass

struct GoodDtor2[A: Int](Movable where False):
   def explicit_dtor(deinit self, deinit other: GoodDtor2[0]): pass

def test_deinit_fn_types():
  var fp1 : def(var self: GoodDtor) thin -> None
  fp1 = GoodDtor.__deinit__
  fp1 = GoodDtor.explicit_dtor
  fp1 = GoodDtor.normal_var

  # expected-error @+1 {{function types do not support 'deinit'; replace with 'var'}}
  var fp2 : def(deinit self: GoodDtor) thin -> None

# An initializer with a lone Self-typed 'deinit' argument has move-constructor
# shape, but only the keyword-only name 'move' defines one. Anything else is
# silently shadowed by the synthesized default move constructor, most commonly
# code predating the rename of the 'take' argument to 'move'.
struct MoveCtorShapedInit(Movable where False):
  def __init__(out self): pass
  # expected-warning @+1 {{'deinit' argument 'take' does not define a move constructor; declare it as '__init__(*, deinit move)'}}
  def __init__(out self, *, deinit take: Self): pass

struct MoveCtorShapedPositionalInit(Movable where False):
  # expected-warning @+1 {{'deinit' argument 'move' does not define a move constructor; declare it as '__init__(*, deinit move)'}}
  def __init__(out self, deinit move: Self): pass

struct MoveCtorWrongConvention(Movable where False):
  # expected-error @+1 {{'move' argument must be passed as 'deinit'}}
  def __init__(out self, *, move: Self): pass

struct MoveCtorShapeNegatives(Movable where False):
  # A properly-spelled move constructor: no warning.
  def __init__(out self, *, deinit move: Self): pass
  # 'deinit' on a non-init consuming method: no warning.
  def consume(self, deinit other: Self): pass
  # Additional arguments break move-constructor shape: no warning.
  def __init__(out self, *, deinit a: Self, x: Int): pass

@fieldwise_init
struct CantSynthesize(ImplicitlyCopyable):
# expected-error @below {{cannot synthesize fieldwise init because field 'x' has non-copyable and non-movable type 'InMemStruct'}}
# expected-error @below {{cannot synthesize move constructor because field 'x' has non-movable and non-implicitly-copyable type 'InMemStruct'}}
# expected-error @below {{cannot synthesize implicit copy constructor because field 'x' has non-implicitly-copyable type 'InMemStruct'}}
  var x : InMemStruct


@fieldwise_init
struct ResolveErrorIsBubbled(Movable where False):
   var x: Int
   @implicit
   def __init__(out self, x: unknown): # expected-error {{use of unknown declaration 'unknown'}}
      pass

def function_with_struct():
  struct Foo: # expected-error {{struct inside a function not supported here}}
    var x: Int

# https://github.com/modularml/modular/issues/12598
struct not_nested_struct[*Ts: AnyType](Movable where False):
    @implicit
    def __init__(out self, *args: *Self.Ts):
        pass
def function_with_struct2():
    var s1 = not_nested_struct()  # ok
    struct S2[*Ts: AnyType]: # expected-error {{struct inside a function not supported here}}
        @implicit
        def __init__(out self, *args: *Ts):
            pass
    var s2 = S2() # In issue https://github.com/modularml/modular/issues/12598 this was crashing.

# https://github.com/modularml/modular/issues/33557
struct HasBadCtor(Movable where False):
    var v: Int
    # expected-error @below {{functions must not declare both an 'out' argument and a return type}}
    def __init__(out self, v: Int) -> Self:
        self.v = v
def useBadCtor() raises:
    # Note that the key thing we're checking for here is that this does NOT have
    # a spurious error about HasBadCtor not being constructable from IntLiteral
    var fromBadCtor = HasBadCtor(123)

struct NotRegisterPassable(Movable where False):
    def __init__(out self):
        pass

# https://github.com/modularml/modular/issues/34551
# Don't crash on emitting methods when the struct itself is erroneous.

@fieldwise_init
struct Outer34551(ImplicitlyCopyable, RegisterPassable): # expected-error {{all members of 'RegisterPassable' struct must themselves be 'RegisterPassable'}}
    # expected-error @below {{cannot synthesize move constructor because field '_inner' has non-movable and non-implicitly-copyable type 'NotRegisterPassable'}}
    # expected-error @below {{cannot synthesize copy constructor because field '_inner' has non-copyable type 'NotRegisterPassable'}}
    # expected-note @below {{'_inner' declared with type 'NotRegisterPassable'}}
    var _inner: NotRegisterPassable
    def __init__(out self):
        self._inner = NotRegisterPassable()
    # The key point of this test is that these errors break an invariant needed
    # for emission, so previously it would crash while emitting this __deinit__.
    def __deinit__(deinit self):
        _ = self._inner ^

struct StructWithoutBody(RegisterPassable):
    pass

@fieldwise_init
struct OkayStruct(ImplicitlyCopyable, RegisterPassable):
# expected-error @below {{cannot synthesize implicit copy constructor because field 'begin' has non-implicitly-copyable type 'StructWithoutBody'}}
    var begin: StructWithoutBody


@fieldwise_init
struct ExplicitlyCopyableStructWithNonCopyableBody(Copyable, RegisterPassable):
# expected-error @below {{cannot synthesize copy constructor because field 'begin' has non-copyable type 'StructWithoutBody'}}
    var begin: StructWithoutBody


struct ExplicitlyCopyableStructWithoutBody(Copyable, RegisterPassable):
    pass

@fieldwise_init
struct ImplicitCopyableStructWithExplicitBody(ImplicitlyCopyable, RegisterPassable):
  # expected-error @below {{cannot synthesize implicit copy constructor because field 'begin' has non-implicitly-copyable type 'ExplicitlyCopyableStructWithoutBody'}}
    var begin: ExplicitlyCopyableStructWithoutBody


# A parameterized struct declaring `ImplicitlyCopyable` whose field is a type
# parameter constrained only to `Copyable` must not synthesize an implicit copy
# ctor.
@fieldwise_init
struct OnlyCopyableField[T: Copyable & Deinitable](ImplicitlyCopyable where conforms_to(T, Copyable)):
    # expected-error @below {{cannot synthesize implicit copy constructor because field 'f' has non-implicitly-copyable type 'T'}}
    var f: Self.T


# MOCO-2186: Initializer syntax should reject incorrect result type
struct StructWithSpecificInit[X: Int](Movable where False):
    def __init__(out self: StructWithSpecificInit[4]): # expected-note {{function declared here}}
        pass
def testStructWithSpecificInit() raises:
    # expected-error @+1 {{invalid initialization: return type 'StructWithSpecificInit[Int(4)]' parameter 'X' value 'Int(4)' doesn't match expected value 'Int(1)'}}
    var a = StructWithSpecificInit[1]()  # Infers to A[4]

    # This is ok.
    var b = StructWithSpecificInit[4]()


##===----------------------------------------------------------------------===##
# Traits
##===----------------------------------------------------------------------===##

# MOCO-2391
# expected-error @+1 {{use of unknown declaration 'UnknownTrait'}}
struct StructWithUnknownTrait(UnknownTrait):
    pass


trait EverythingIsWrongTrait:
    var value: Int # expected-error {{traits do not support 'var' fields; use 'comptime' to declare associated types}}

    def trait_fn_no_dot_dot_dot(self): # expected-error {{body must not be empty; use 'pass' or check that the lines below are indented}}

    trait NestedTrait: # expected-error {{nested trait not supported here}}
        ...

    def parametric[x: Int](self): ...

    struct NestedStruct: # expected-error {{nested struct in a trait not supported here}}
        pass

trait TraitWithParams[T: TrivialRegisterPassable]: # expected-error {{trait declarations do not support parameters; remove the parameter list}}
    ...

# Errors on emitting trait type for `EverythingIsWrongTrait`
def bad_trait_params[T: EverythingIsWrongTrait](x: T):
  pass

trait Shape(ImplicitlyCopyable):
	def area(self) -> Int:
	    ...

@fieldwise_init
struct ShapeContainer(Movable where False):
    var shape: Shape # expected-error {{struct fields do not support trait types; 'Shape' is a trait, use a concrete type or compile-time generic}}

# MSTDL-2267: Structs with "AnyOrigin" fields stop keeping things alive when passed to a function
struct SwallowAnyOrigin(Movable where False):
    # expected-error @below {{struct fields cannot expose AnyOrigin in their type; foo has type 'Pointer[Int, MutAnyOrigin]'}}
    # expected-note @below {{consider parameterizing enclosing struct with an Origin}}
    # expected-note @below {{alternatively, use UntrackedOrigin if lifetime is managed explicitly}}
    var foo: UnsafePointer[Int, MutAnyOrigin]

# A field that opts into the legacy behavior with the
# @__allow_legacy_any_origin_fields decorator is accepted without error.
struct AllowedLegacyAnyOrigin(Movable where False):
    @__allow_legacy_any_origin_fields
    var foo: UnsafePointer[Int, MutAnyOrigin]

# A function-pointer field whose signature mentions AnyOrigin in an argument or
# result position does not expose AnyOrigin in the struct's own storage (the
# struct holds a function, not a reference into that memory), so it is accepted
# without the decorator.
struct FnPtrAnyOriginArg(Movable where False):
    var cb_arg: def(UnsafePointer[Int, MutAnyOrigin]) thin -> None
    var cb_ret: def() thin -> UnsafePointer[Int, MutAnyOrigin]

# A struct parameter bound by a function type that mentions AnyOrigin, with a
# field of that parameter type, is likewise accepted: the AnyOrigin lives in the
# parameter's bound, not in the field's own storage type.
struct FnTypeParamBound[T: def(UnsafePointer[Int, MutAnyOrigin])](Movable where False):
    var t: Self.T

##===----------------------------------------------------------------------===##
# Struct/Trait conformance check failure
##===----------------------------------------------------------------------===##

trait CFMTrait: # expected-note {{trait 'CFMTrait' declared here}}
    # expected-note @below {{no 'f1' candidates have type 'def(self: CFMStructFail) thin -> None'}}
    def f1(self):
        ...

    @staticmethod
    def f2(): # expected-note {{required function 'f2' is not implemented}}
        ...

# struct implements CFMTrait but does not have f2().
struct CFMStructFail(TrivialRegisterPassable, CFMTrait): # expected-error {{'CFMStructFail' does not implement all requirements for 'CFMTrait'}}
  def f1(self, x: Int): # expected-note {{candidate declared here with type 'def(self: CFMStructFail, x: Int) thin -> None'}}
    pass

struct NoTraits(TrivialRegisterPassable):
    pass

# expected-note @+1 {{function declared here}}
def trait_fn[T: CFMTrait]():
    pass

def invalid_trait_bind():
    trait_fn[NoTraits]() # expected-error {{invalid call to 'trait_fn': 'trait_fn' parameter 'T' has 'CFMTrait' type, but value has type 'AnyStruct[NoTraits]'}}

def non_copyable_trait[T: CFMTrait](value: T):
    var copy = value # expected-error {{value of type 'T' cannot be implicitly copied, it does not conform to 'ImplicitlyCopyable'}}


def trait_fn_infer[T: CFMTrait](x: T):
    pass

def dont_crash_pvalue_convert(x: CFMStructFail):
    # This will succeed, the error will be raised when resolving `CFMStructFail`.
    trait_fn_infer(x)

trait GrandFather: # expected-note {{trait 'GrandFather' declared here}}
    def foo(self): # expected-note {{required function 'foo' is not implemented}}
        ...

trait Father(GrandFather): # expected-note {{inherited through 'Father' here}}
    pass

# expected-error @below {{'MissingInheriteddef' does not implement all requirements for 'GrandFather'}}
struct MissingInheriteddef(Father, GrandFather, Movable where False):
    pass

struct InheritsTwice(Father, Father, Movable where False):
    def foo(self):
        pass


# https://github.com/modular/mojo/issues/1399
# Parser crash when trait implementation parameters don't match the definition
# expected-note @below {{trait 'TraitWithIntParamOnMethodReallyLongName' declared here}}
trait TraitWithIntParamOnMethodReallyLongName:
  # expected-note @below {{no 'f' candidates have type 'def[n: Int](self: UseTraitWithIntParamOnMethodReallyLongName) thin -> None'}}
  def f[n: Int](self):
    ...
# expected-error @below {{'UseTraitWithIntParamOnMethodReallyLongName' does not implement all requirements for 'TraitWithIntParamOnMethodReallyLongName'}}
struct UseTraitWithIntParamOnMethodReallyLongName(TraitWithIntParamOnMethodReallyLongName, Movable where False):
  # expected-note @below {{candidate declared here with type 'def[n: Bool](self: UseTraitWithIntParamOnMethodReallyLongName) thin -> None'}}
  # expected-note @below {{.n of the first type is 'Int' but the second type is 'Bool'}}
  def f[n: Bool](self):
    pass

##===----------------------------------------------------------------------===##
# Class
##===----------------------------------------------------------------------===##

class SomeClass:  # expected-error {{classes are not supported yet}}
  pass


# Issue #12090
from imported_module import DTypePointer # expected-note {{conflicts with this previous declaration}}
struct DTypePointer: # expected-error {{cannot define a struct here with name 'DTypePointer'}}
    pass

# Issue #13321.
struct copy_init_def(Movable where False):
  var field: Int

  # expected-error @+1 {{copy constructor must not declare 'raises'; remove the 'raises' keyword}}
  def __init__(out self, *, copy: Self) raises:
    self.field = copy.field

struct copy_init_raises(Movable where False):
  # expected-error @+1 {{copy constructor must not declare 'raises'; remove the 'raises' keyword}}
  def __init__(out self, *, copy: Self) raises:
     pass


# Order of declaration processing.
# https://github.com/modular/mojo/issues/235
@fieldwise_init
struct Inner(Movable where False):
    pass

@fieldwise_init
struct Outer(RegisterPassable): # expected-error {{all members of 'RegisterPassable' struct must themselves be 'RegisterPassable'}}
    # expected-error @+1{{cannot synthesize move constructor because field 'inner' has non-movable and non-implicitly-copyable type 'Inner'}}
    var inner: Inner # expected-note {{'inner' declared with type 'Inner'}}


# expected-warning @+2 {{redundant trait composition: 'Deinitable' already implies 'AnyType'}}
@fieldwise_init
struct AnyTypeMember[T: AnyType & Deinitable](ImplicitlyCopyable):
# expected-error @below {{cannot synthesize fieldwise init because field 'value' has non-copyable and non-movable type 'T'}}
# expected-error @below {{cannot synthesize move constructor because field 'value' has non-movable and non-implicitly-copyable type 'T'}}
# expected-error @below {{cannot synthesize implicit copy constructor because field 'value' has non-implicitly-copyable type 'T'}}
    var value: Self.T


# Issue https://github.com/modular/mojo/issues/1675
# Ensure @fieldwise_init fails gracefully in the presence of duplicate field names.
@fieldwise_init
struct BadStruct(Movable where False):
    var b: Int  # expected-note {{previous definition here}}
    var b: Int  # expected-error {{invalid redefinition of 'b'}}


# Also ensure that @fieldwise_init doesn't fail if a method/alias shadows it.
@fieldwise_init
struct OtherBadStruct(Movable where False):
    # expected-note @below {{previous definition here}}
    # expected-note @below {{cannot overload with this non-function definition}}
    var b: Int
    comptime b = 0  # expected-error {{invalid redefinition of 'b'}}

    def b(mut self):  # expected-error {{invalid redefinition of 'b'}}
        pass


def test_bad_struct():
    _ = BadStruct(1)
    _ = OtherBadStruct(2)

##===----------------------------------------------------------------------===##
# Bad implicit conversions.
##===----------------------------------------------------------------------===##


@fieldwise_init
struct HasBoolParam[a: Bool](Movable where False):
   pass

def test(arg: HasBoolParam[True]):
  # expected-error @+1 {{cannot implicitly convert 'HasBoolParam[True]' value to 'HasBoolParam[False]'}}
  var bad : HasBoolParam[False] = arg


@fieldwise_init
struct Foo(ImplicitlyCopyable):
    var val: Int

@fieldwise_init
struct ContainsFoo(Movable where False):
    var foo: Foo

# expected-note @+1 {{function declared here}}
def take_foo(x: Foo): pass

def return_foo(x: Int) -> Foo:
    return x # expected-error {{cannot implicitly convert 'Int' value to 'Foo'}}

    return 1.2 # expected-error {{cannot implicitly convert 'FloatLiteral[1.2]' value to 'Foo'}}

# When attempting to do implicit conversions without an @implicit decorator
def implicit_conversions():
    # assigning to expected type
    var x = 42
    var a: Foo = x # expected-error {{cannot implicitly convert 'Int' value to 'Foo'}}

    # # reassigning
    var b = Foo(42)
    b = 42 # expected-error {{cannot implicitly convert 'IntLiteral[42]' value to 'Foo'}}

    # # assigning to uninitialized
    var c: Foo
    c = 42 # expected-error {{cannot implicitly convert 'IntLiteral[42]' value to 'Foo'}}

    # # assigning to member
    var d = ContainsFoo(Foo(24))
    d.foo = 42 # expected-error {{cannot implicitly convert 'IntLiteral[42]' value to 'Foo'}}

    # # returning conversions
    var e = return_foo(42)

    take_foo(24) # expected-error {{invalid call to 'take_foo': value passed to 'x' cannot be converted from 'IntLiteral[24]' to 'Foo'}}

##===----------------------------------------------------------------------===##
# Top Level Code
##===----------------------------------------------------------------------===##

def top_level_func() raises -> Int:
   pass

def use_error(e: Error):
   pass

# expected-error @below {{expressions must not appear at file scope; move this into a function body}}
_ = top_level_func()

# NOTE: try/except at module scope is tested in module_scope_try_error.mojo
# Removing inline test here because parser error recovery skips subsequent code


def top_level_func_param[p: Int]():
    pass

comptime a = 100
# expected-error @below {{expressions must not appear at file scope; move this into a function body}}
top_level_func_param[a]()

# expected-error @below {{global variables are not supported; move this into a function body or use 'comptime' to declare a constant}}
var globalVar = 1


struct S[param: Int](Movable where False): #expected-note {{previous definition here}}
  def method[param: Int](self): # expected-error {{invalid redefinition of 'param'}}
    pass

struct MyParam[p: Int](Movable where False):
  pass


#expected-note @below {{previous definition here}}
struct MyStruct[p: Int, m1: MyParam[_], m2: MyParam[_]](Movable where False):
  def method[p: Int](self): # expected-error {{invalid redefinition of 'p'}}
    pass

# https://github.com/modular/modular/issues/5479
def __deinit__(): # expected-error {{'__deinit__' must be a method, not a global function}}
  pass

def raises_int() raises Int:
  raise 1

  raise 4.0

  # expected-error @+1 {{cannot implicitly convert 'Error' value to 'Int'}}
  raise Error()

def bad_raises_fn2() raises:
  # expected-error @+1 {{cannot call function that may raise 'Int' in context that supports an error type of 'Error'}}
  raises_int()

  # expected-error @+1 {{cannot implicitly convert 'FloatLiteral[4]' value to 'Error'}}
  raise 4.0

  try:
    raises_int()

    # expected-error @+1 {{cannot implicitly convert 'Error' value to 'Int'}}
    raise Error()
  except e: # 'e' inferred to Int.
    var x: Int = e
    # expected-error @+1 {{cannot implicitly convert 'Int' value to 'Error'}}
    var y: Error = e

  try:
    raise 1 # Should infer error to Int, not IntLiteral
  except e:
    # expected-error @+1 {{cannot implicitly convert 'Int' value to 'String'}}
    var x: String = e


# expected-error @+1 {{@export can not be applied on parametric functions}}
@export
def foo(s: SIMD) abi("Mojo"):
    pass

# expected-error @+1 {{@export can not be applied on parametric functions}}
@export
def bar[n: Int]() abi("Mojo"):
    pass

@export
def exported_with_origin_param[a: ImmOrigin](ref [a] x: Int) abi("Mojo"):
  pass



comptime _something = MyParam.p # expected-error {{'p' refers to an unbound parameter in 'MyParam[_]'}}

def pack_error[Trait: type_of(AnyType), //, *element_types: Trait]():
    pass


# expected-error @+1 {{cannot construct a value with parametric type: 'Origin[mut=_]'}}
comptime A: Origin = AnyOrigin


# Argument conventions are not part of the mangled name, so they cannot
# distinguish two otherwise-identical signatures.
# expected-note @+1 {{previous definition here}}
def overload_on_arg_conv(mut x: Int):
    pass


# expected-error @+1 {{redefinition of function 'overload_on_arg_conv', cannot overload on argument conventions}}
def overload_on_arg_conv(imm x: Int):
    pass

# expected-error @+1 {{'List[_]' is not concrete, use '[]' to bind missing parameters}}
comptime _list_eq_int = List == Int
