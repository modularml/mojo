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

# RUN: %parse-mojo-isolated %s -verify-diagnostics

struct SomeNonTrivRegPassable(RegisterPassable): pass

struct MemType(Movable where False):
  def __init__(out self):
    pass

  def consume(var self): pass


def takes_pos_or_kw_arg(i: Int, j: Int):
    pass


def test_duplicate_kw_arg(x: Int):
    takes_pos_or_kw_arg(
        j=x,  # expected-note {{previously specified here}}
        j=x,  # expected-error {{keyword argument 'j' was already used; remove the duplicate}}
    )


def test_pos_after_kw_arg(x: Int):
    takes_pos_or_kw_arg(
        j=x,
        x,  # expected-error {{positional argument must not follow a keyword argument; move it before or convert to a keyword argument}}
    )


def takes_pos_or_kw_param[i: Int, j: Int]():
    pass


def test_duplicate_kw_param[x: Int]():
    takes_pos_or_kw_param[
        j=x,  # expected-note {{previously specified here}}
        j=x,  # expected-error {{keyword parameter 'j' was already used; remove the duplicate}}
    ]


def test_pos_after_kw_param[x: Int]():
    takes_pos_or_kw_param[j=x, x]()


def invalid_with():
    # expected-error @below {{use of unknown declaration 'bogus'}}
    with bogus() as foo:
        foo.something()


struct SomeType(Movable where False):
    pass


def mem_type_var():
    # expected-error @below {{dynamic type values not permitted yet; try creating a 'comptime' instead of a 'var}}
    var type = SomeType


def reg_type_var():
    # expected-error @below {{dynamic type values not permitted yet; try creating a 'comptime' instead of a 'var}}
    var type = Int


trait SomeTrait:
    pass


def trait_var():
    # expected-error @below {{dynamic type values not permitted yet; try creating a 'comptime' instead of a 'var}}
    var type = SomeTrait


def reg_type_func() -> TrivialRegisterPassable:
    # expected-error @below {{dynamic type values not permitted yet}}
    return Int


def mem_type_func() -> AnyType:
    # expected-error @below {{dynamic type values not permitted yet}}
    return SomeType


def takes_reg_type(t: TrivialRegisterPassable):
    pass


def test_takes_reg_type():
    # expected-error @below {{use of unknown declaration 'takes_type'}}
    takes_type(Int)


def takes_mem_type(t: AnyType):
    pass


def test_takes_mem_type():
    # expected-error @below {{use of unknown declaration 'takes_type'}}
    takes_type(SomeType)

# MOCO-56: Mojo produces weird error when mut function is used in non mutating function
struct SomethingWithInferredParam[T: ImplicitlyCopyable](Movable where False):
  pass
# expected-note @+1 {{function declared here}}
def SomethingWithInferredParamCallee(mut v: SomethingWithInferredParam):
  pass

def SomethingWithInferredParamCaller(v: SomethingWithInferredParam):
  # expected-error @+1 {{value passed to mutable argument 'v' must be mutable}}
  SomethingWithInferredParamCallee(v)

##===----------------------------------------------------------------------===##
# Variable declarations
##===----------------------------------------------------------------------===##

def test_var_decl_patterns(cond: Bool):
  # expected-note @+1 {{previous definition here}}
  var x = 42 # ok of course.

  (var x) = 42 # expected-error {{invalid redefinition of 'x'}}

  if cond:
    (var x) = 42

  y = (var 42) # expected-error {{'var' patterns are only valid on the left side of an assignment}}

import std.builtin
def test_member_access() raises:
    # MOCO-2006: This crashed because it was trying to synthesize the vardecl in
    # the package.
    # expected-error @+2 {{dynamic type values not permitted yet}}
    # expected-error @+1 {{implicit declaration of 'localvar' is not allowed; add 'var' to declare a new name}}
    localvar = std.builtin.Int

##===----------------------------------------------------------------------===##
# Conversions
##===----------------------------------------------------------------------===##

def invalid_conversion(a: Int) raises:
  var b: __mlir_type.index = a # expected-error {{implicitly convert 'Int' value to '__mlir_type.index' in 'var' initializer}}

  # expected-error @+1 {{cannot construct type '__mlir_type.index'}}
  _ = __mlir_type.index(4)

  # expected-error @+1 {{cannot construct type 'Never'}}
  _ = Never()

struct NotBoolConvertible(Movable where False):
  pass

# Issue #6600
def negBuiltinType(x: __mlir_type.f64) :
    # expected-error @+1 {{'__mlir_type.f64' does not implement the '__neg__' method}}
    _ = -x

# expected-note @+1 {{function declared here}}
def some_fn_take_int(a: Int): pass
def some_fn_ret_int() -> Int: return 42

# Issue #11288
def test_overload_set():
  # expected-error @+1 {{invalid call to 'some_fn_take_int': value passed to 'a' cannot be converted from 'def some_fn_ret_int() thin -> Int' to 'Int'}}
  some_fn_take_int(some_fn_ret_int)

def overloaded_arg(x: Int): pass # expected-note {{candidate declared here}}
def overloaded_arg(x: String): pass # expected-note {{candidate declared here}}
def test_overloaded_arg_ambiguity() :
  # expected-error @below {{cannot form a reference to overloaded declaration of 'overloaded_arg'}}
  # expected-note @below {{add '()' to call the function}}
  (var xxx) = overloaded_arg

def throws_int() raises Int:
    pass

def test_func_type():
    # expected-error @below {{def(Int) thin -> Int}}
    comptime float0: def(Int) thin -> Int = test_func_type
    # expected-error @below {{async def() thin -> None}}
    comptime float1: async def() thin -> None = test_func_type
    # expected-error @below {{def[a: Int]() thin -> MemType}}
    comptime float2: def[a: Int]() thin -> MemType = test_func_type
    # expected-error @below {{def[a: Int](var Int) thin -> MemType}}
    comptime float3: def[a: Int](var Int) thin -> MemType = test_func_type
    # expected-error @below {{def[a: Int](mut *Int) thin -> None}}
    comptime float4: def[a: Int](mut *Int) thin -> None = test_func_type
    # expected-error @below {{'def(*MemType) raises capturing thin -> None'}}
    comptime float5: def(*MemType) raises capturing -> None = test_func_type
    # expected-error @below {{'def[*Ts: AnyType](var * *Ts) capturing thin -> None'}}
    comptime float6: def[*Ts: AnyType](var* *Ts) capturing -> None = test_func_type
    # expected-error @below {{'def[*Ts: AnyType](var * *Ts) capturing thin -> None'}}
    comptime float6a: def[*Ts: AnyType](var* *Ts) capturing -> None = test_func_type
    # expected-error @below {{'def[T: TrivialRegisterPassable](mut *T) capturing thin -> None'}}
    comptime float7: def[T: TrivialRegisterPassable](mut *T) capturing -> None = test_func_type
    # expected-error @below {{'def(var **args: Int) -> None'}}
    comptime float8: def(var **args: Int) = test_func_type

    # expected-error @below {{'def(a1: Int, /, *, a2: Int) -> None'}}
    comptime float9: def(a1: Int, /, *, a2: Int) = test_func_type
    # expected-error @below {{'def(a1: Int, /, a2: Int) -> None'}}
    comptime float10: def(a1: Int, /, a2: Int) = test_func_type

    def passing_kinds_1(a1: Int, /, *, a2: Int): pass
    # expected-error @below {{'def passing_kinds_1(a1: Int, /, *, a2: Int) thin -> None'}}
    _ : Int = passing_kinds_1
    def passing_kinds_2(a1: Int, /, a2: Int): pass
    # expected-error @below {{'def passing_kinds_2(a1: Int, /, a2: Int) thin -> None'}}
    _ : Int = passing_kinds_2
    def passing_kinds_3(a1: Int, *, a2: Int): pass
    # expected-error @below {{'def passing_kinds_3(a1: Int, *, a2: Int) thin -> None'}}
    _ : Int = passing_kinds_3


    # expected-error @below {{unnamed argument cannot follow named argument}}
    comptime f1: def (a: Int, Int) thin -> Int
    # expected-error @below {{unnamed argument cannot follow '/' or '*'}}
    comptime f2: def (Int, /, Int) thin -> Int
    # expected-error @below {{unnamed argument cannot follow '/' or '*'}}
    comptime f3: def (*, Int) thin -> Int
    # expected-error @below {{unnamed argument must be positional-only}}
    comptime f4 = def (Int, b: Int) capturing -> Int

    # expected-error @below {{unnamed parameter cannot follow named parameter}}
    comptime f5: def [a: Int, Int]() thin -> Int
    # expected-error @below {{unnamed parameter cannot follow '/' or '*'}}
    comptime f6: def [Int, /, Int] thin -> Int
    # expected-error @below {{unnamed parameter cannot follow '/' or '*'}}
    comptime f7: def [*, Int] thin -> Int = test_func_type
    # expected-error @below {{unnamed parameter must be positional-only}}
    comptime f8 = def [Int, b: Int] capturing -> Int
    # expected-error @below {{'def throws_int() raises Int thin -> None' value to 'def() raises String thin -> None'}}
    # expected-note @below {{error type of the first type is 'Int' but the second type is 'String'}}
    comptime f9: def () thin raises String = throws_int

    def has_foo_kw(*, foo: Int): pass
    # expected-error @+1 {{cannot implicitly convert 'def has_foo_kw(*, foo: Int) thin -> None' value to 'def(*, bar: Int) thin -> None'}}
    var f10: def (*, bar: Int) thin -> None = has_foo_kw

def param_passing_kinds():
    def f1[a: Int, //, b: Int]() -> None: pass
    _ : Int = f1 # expected-error {{'def f1[a: Int, //, b: Int]() thin -> None'}}
    def f2[a: Int, *, b: Int]() -> None: pass
    _ : Int = f2 # expected-error {{'def f2[a: Int, *, b: Int]() thin -> None'}}
    def f3[a: Int, *b: Int]() -> None: pass
    _ : Int = f3 # expected-error {{'def f3[a: Int, *b: Int]() thin -> None'}}
    def f4(*, a: Int) -> None: pass
    _ : Int = f4 # expected-error {{'def f4(*, a: Int) thin -> None'}}
#===----------------------------------------------------------------------===##
# LValue and RValues
##===----------------------------------------------------------------------===##

def mutArg(a: Int):
  a = a  # expected-error {{expression must be mutable in assignment}}

def assignRValue():
  42 = 17 # expected-error {{expression must be mutable in assignment}}

struct LValuesRvalues(Movable where False):
  def __init__(out self): pass
  def __init__(out self, *, copy: Self): pass

  def normalMethod(self) raises: pass
  # expected-note @+1 {{function declared here}}
  def mutatingMethod(mut self) raises -> None: pass
  # expected-note @+1 {{function declared here}}
  def takesByRef(self, mut x: LValuesRvalues) raises: pass

  def normalMethod3(self, a: FloatDyn): pass

struct MemoryOnlyPair(Copyable):
  var x: Int
  var y: Int
  def __init__(out self):
    self.x = 0
    self.y = 0

struct NonCopyable(Movable where False):
  def __init__(out self): pass

def generic_on_type_ok[T: TrivialRegisterPassable](): pass

def testLValuesRvalues() raises -> None:
  # Test with lvalues
  var lv: LValuesRvalues
  lv.normalMethod()
  lv.mutatingMethod()

  # Partial application.
  # expected-error @below {{member method closures are not supported; add '()' to call 'mutatingMethod'}}
  lv.mutatingMethod

  # Test with rvalues
  LValuesRvalues().normalMethod()
  LValuesRvalues().mutatingMethod()  # expected-error {{invalid use of mutating method on rvalue of type 'LValuesRvalues'}}

  # expected-error @+1 {{value passed to mutable argument 'x' must be mutable}}
  LValuesRvalues().takesByRef(LValuesRvalues())

  # We can not implicitly declare things on the RHS
  lv += unknown2 # expected-error {{use of unknown declaration 'unknown2'}}

  lv.normalMethod3(1.0)

  var nc1 = NonCopyable()
  var nc2 = NonCopyable()

  # expected-error @below {{value of type 'NonCopyable' cannot be implicitly copied, it does not conform to 'ImplicitlyCopyable'}}
  # expected-note @below {{consider transferring the value with '^'}}
  var nc3 = nc1
  # expected-error @below {{value of type 'NonCopyable' cannot be implicitly copied, it does not conform to 'ImplicitlyCopyable'}}
  # expected-note @below {{consider transferring the value with '^'}}
  var nc4 = nc2

  var mpPair = MemoryOnlyPair()

  # expected-error @+1 {{cannot bind type 'MemoryOnlyPair' to trait 'TrivialRegisterPassable'}}
  comptime T: TrivialRegisterPassable = MemoryOnlyPair

# expected-note @+1 {{function declared here}}
def badRef(mut val: Int):
  var x = FloatDyn(1.0)
  # expected-error @+1 {{invalid call to 'badRef': l-value of type 'FloatDyn' cannot be converted to reference of type 'Int'}}
  badRef(x)

struct PythonObject(Movable where False): pass
def getPythonObject() -> PythonObject: pass

def unused_values():
  var x : Int = 42

  _ = 4+4 # OK: Explicitly ignored.
  # expected-warning @+1 {{'Int' value is unused; assign to '_' to discard the result}}
  4+4  # MValue

  _ = x # OK: Explicitly ignored.
  # expected-warning @+1 {{'Int' value is unused; assign to '_' to discard the result}}
  x  # LValue

  _ = x+1 # OK: Explicitly ignored.
  # expected-warning @+1 {{'Int' value is unused; assign to '_' to discard the result}}
  x+1 # DRValue

  # expected-warning @+1 {{function is not called; add '()' after name}}
  testLValuesRvalues
  _ = testLValuesRvalues # OK

  # No warning.
  getPythonObject()

  # No warning.
  try:
    unused_values2()
  except e:
    pass

def unused_values2() raises:
  # No warning.
  getPythonObject()

  # expected-warning @+1 {{'Int' value is unused; assign to '_' to discard the result}}
  4+1

def no_unused_values_in_def() raises:
  var x : Int = 42
  4+4  # expected-warning {{'Int' value is unused; assign to '_' to discard the result}}
  x    # expected-warning {{'Int' value is unused; assign to '_' to discard the result}}
  x+1  # expected-warning {{'Int' value is unused; assign to '_' to discard the result}}
  testLValuesRvalues # expected-warning {{function is not called; add '()' after name}}

  _ # expected-error {{cannot read from discard pattern '_'}}

  # expected-error @+1 {{cannot read from discard pattern '_'}}
  var abc = _

  # expected-error @+1 {{cannot read from discard pattern '_'}}
  var bcd = *_

  _ = *x # expected-error {{can't use starred expression here}}

# expected-note @+1 {{function declared here}}
def func_with_static_param[x: Int]() -> Int:
  return x

def dynamic_used_as_param() -> Int:
  var x = 5
  # expected-error @+1 {{cannot use a dynamic value in a parameter list}}
  return func_with_static_param[x]()

@fieldwise_init
struct StructWithField(Movable where False):
  var x : Int

def dynamic_used_as_param_2() -> Int:
  var w = StructWithField(3)
  # expected-error @+1 {{cannot use a dynamic value in type parameter}}
  return func_with_static_param[w.x]()

def test_ref_decl_patterns(a: List[Int], mut b: List[Int]):
    ref r = a[0]
    r += 1 # expected-error {{expression must be mutable for in-place operator destination}}

    ref r2 = 42 # expected-error {{value of type 'Int' doesn't have a memory origin in 'ref' binding}}
    ref r3 = r2 # no follow-on error.


##===----------------------------------------------------------------------===##
# Tuples
##===----------------------------------------------------------------------===##

def bad_tuple(a: Int):
  _ = (a, a, b)  # expected-error {{use of unknown declaration 'b'}}

  var c: Int
  var d: Int
  # expected-error @+1 {{cannot implicitly convert 'Tuple[Int, Int, Int]' value to 'Tuple[Int, Int]'}}
  (c, d) = (a, a, a)
  # expected-error @+1 {{cannot implicitly convert 'Tuple[Int]' value to 'Tuple[Int, Int]'}}
  (c, d) = (a,)
  # expected-error @+1 {{cannot implicitly convert 'Int' value to 'Tuple[Int, Int]'}}
  (c, d) = a

  var iTup : Tuple[Int, Int]
  iTup = (1, 2.0)


def tuple_return() raises -> Int:
  # expected-error @+1 {{cannot implicitly convert 'Tuple[Int, Int]' value to 'Int'}}
  return 32, 17

def tuple_pattern(a: Int) raises:
  # expected-error @+1 {{cannot unpack value of type 'Int' into 2 values}}
  (b, c) = a

def list_pattern():
  # expected-error @+1 {{'Int' value has no attribute '__len__'}}
  [_] = 4


# Issue https://github.com/modular/mojo/issues/1917
# Do not crash in tuple creation if element has syntax error.
# expected-error @below {{expected '(' for argument list}}
def bad_func return def() -> __mlir_type.index


##===----------------------------------------------------------------------===##
# Other Specific expression forms
##===----------------------------------------------------------------------===##


# expected-error @+1 {{'Self' type may only be used inside a struct, trait, or extension}}
def badSelf(a: Self):
  var x: Self.field

# Structs convertible to each other.
struct Conv1(Movable where False):
  @implicit
  def __init__(out self, value: Conv2): pass
struct Conv2(Movable where False):
  @implicit
  def __init__(out self, value: Conv1): pass

struct MyIntPair(RegisterPassable):
  var a: Int
  var b: Int

struct TwoAndThreeList(Movable where False):

   # expected-note @below {{candidate not viable: unexpected argument}}
   # expected-note @below {{candidate not viable: missing required argument: 'a'}}
   def __init__(out self, a: Int, b: Int, __list_literal__: NoneType): pass
   # expected-note @below {{candidate not viable: unexpected keyword argument '__list_literal__'}}
   # expected-note @below {{candidate not viable: missing required argument: 'a'}}
   def __init__(out self, a: Int, b: Int, c: Int, __list_literal__: NoneType): pass

struct SimpleRange(TrivialRegisterPassable):
    def __init__(out self): pass
    def __len__(self) -> Int:
        pass
    def __next__(mut self) raises StopIteration -> Int:
        pass
    def __iter__(self) -> Self:
        pass

def list_literals():
  # expected-error @+1 {{cannot emit an empty list without a contextual type}}
  _ = []
  _ = [1, 2]

  var a: TwoAndThreeList = [1, 2]
  var b: TwoAndThreeList = [1, 2, 3]

  # expected-error @+1 {{no matching function in initialization}}
  var c: TwoAndThreeList = [1, 2, 3, 4]
  # expected-error @+1 {{no matching function in initialization}}
  var d: TwoAndThreeList = []

  # expected-error @+1 {{list comprehension must have a single expression before 'for'; remove extra expressions}}
  _ = [x, x+1 for x in SimpleRange()]

  _ = [x for x in SimpleRange() if x * 2 == 0]

  # expected-error @+1 {{list comprehension must execute in runtime contexts; remove 'comptime' and move this into a function body}}
  comptime some_alias = [1 for x in range(10)]


def set_parse_errors(a: Int):
  _ = {key for key in SimpleRange() if key == 0}

  # expected-error @+1 {{cannot use keyword argument in set comprehension}}
  _ = {keyword=key for key in SimpleRange()}

def dict_expression(a: Int):
  # expected-error @+1 {{cannot emit initializer list without a contextual type}}
  _ = {}
  _ = {a: 4}
  # expected-error @+1 {{TODO: unpack emission in dict literal not supported yet}}
  _ = {a: 4, **dict, "b": 17}

  var comprehension = {elt:elt+1 for elt in SimpleRange()}

def dict_parse_errors(a: Int):
  # expected-error @+1 {{comprehension must have a single expression before 'for'; remove extra expressions}}
  _ = {elt:elt+1, 1:2 for elt in SimpleRange()}



def bad_exprs(cond: Bool, x: Error, c1: Conv1, c2: Conv2):
  # expected-error @+1 {{value of type 'Error' is not compatible with value of type 'Conv1'}}
  _ = x if cond else c1

  # expected-error @below {{ambiguous merge: left value has type 'Conv1' and right value has type 'Conv2', and both convert to each other}}
  # expected-note @below {{you could disambiguate by casting the left value to 'Conv2'}}
  # expected-note @below {{or cast the right value to 'Conv1'}}
  _ = c1 if cond else c2

def bad_assignment0() raises:
   var a: Int
   var b: Int
   # expected-error @+1 {{cannot implicitly convert 'None' value to 'Int'}}
   a = b += b

def bad_assignment1(a: Int, b: Int) raises:
   # expected-error @+1 {{expected ')' in parenthesized expression}}
   a = (b += b)

# MOCO-1936 / Issue #4501: Incorrect parsing of incomplete assignment
def bad_assignment2():
  # expected-error @+1 {{'=' must be followed by an expression on the same line}}
  _ =    # should error here
  a = 1


def bad_walrus_undeclared():
  # Walrus requires an existing LValue; it does not implicitly declare.
  if a := 4: # expected-error {{use of unknown declaration 'a'}}
    pass

def unused_assignments():
  var a = 1
  a = a  # ok of course.
  a := a # expected-warning {{'Int' value is unused; assign to '_' to discard the result}}

async def async_function() -> Int:
    return 0

# See Issue #15578
def doIs(a: Int, b: Int) raises:
  # expected-error @+1 {{'Int' does not implement the '__is__' method}}
  if a is b:
    pass

def doIsNot(a: Int, b: Int) raises:
  # expected-error @+1 {{'Int' does not implement the '__isnot__' method}}
  if a is not b:
    pass

def testInExpr(x: Int, y: Int) raises:
  # expected-error @+1 {{'Int' does not implement the '__contains__' method}}
  _ = x in y
  # expected-error @+1 {{'Int' does not implement the '__contains__' method}}
  _ = x not in y

##===----------------------------------------------------------------------===##
# Computed Properties and Subscripts
##===----------------------------------------------------------------------===##

struct ConvertFromInt(Movable where False):
    def __init__(out self): pass
    @implicit
    def __init__(out self, value: Int): pass

struct IncompatElementTypes(Movable where False):
  def __getitem__(self, x: Int) -> Int: pass
  def __setitem__(self, x: Int, y: ConvertFromInt): pass

def test_subscript_implicit_conversion(c: IncompatElementTypes):
  var tmp : Int = c[1]
  # expected-error @+1 {{cannot implicitly convert 'ConvertFromInt' value to 'Int'}}
  c[1] = ConvertFromInt()  # FIXME: This should work
  c[1] = tmp

struct GetAttrNotString(Movable where False):
    # expected-note @below {{function declared here}}
    def __init__(out self):
        pass

    # expected-note @below {{function declared here}}
    def __getattr__(self, idx: Int) -> Int:
        return 0

def invalid_getattr():
    var obj = GetAttrNotString()
    # expected-error @below {{invalid call to '__getattr__': value passed to 'idx' cannot be converted from 'StringLiteral["attr"]' to 'Int}}
    obj.attr


struct GetSettable(Movable where False):
  def __getitem__(self, x: Int) -> Int: pass
  def __setitem__(self, x: Int, y: Int): pass

struct NoSelfCtor(Movable where False):
  var x: Int
  def __init__(out self, x: Int):
    self.x = x

def test_int_to_int_error(a: Int, b: NoSelfCtor):
  # expected-error @+1 {{cannot construct 'NoSelfCtor' with itself, you can remove the constructor call}}
  _ = NoSelfCtor(NoSelfCtor(a))

  # expected-error @+1 {{invalid initialization: unexpected argument}}
  _ = GetAttrNotString(a)


trait t:
  pass

def type_subscript(t0 : t):
# expected-error @below {{types are not subscriptable}}
  t[]

##===----------------------------------------------------------------------===##
# lambda errors. The capture list and return type may be elided (omitted capture
# list -> {imm}/thin; omitted return type -> None); arguments must be
# parenthesized and typed. The cases below are forms that remain rejected;
# successful construction is covered by mojo-parser/exprs/lambda.mojo.
##===----------------------------------------------------------------------===##

# Unparenthesized (Python-style) arguments are guided to the parenthesized form.
# This fires at parse time, before any elision or type check.
def testLambdaUnparenthesizedArgs() raises:
  # expected-error @+1 {{unparenthesized lambda arguments are not supported}}
  _ = lambda x, y: x + y

# Adding types to unparenthesized arguments does not help.
def testLambdaTypedArgWithoutParens() raises:
  # expected-error @+1 {{unparenthesized lambda arguments are not supported}}
  _ = lambda x: Int: x + 1

# A parenthesized but untyped arg keeps its own clear error (shown with an
# explicit `{}`/`-> Int` so the untyped arg is what is rejected).
def testLambdaParenthesizedUntypedArg() raises:
  # expected-error @+1 {{argument type must be specified}}
  _ = lambda (x) {} -> Int: x

# Fully bare `lambda: EXPR` is arg-less, thin, and `None`-returning, so a non-None
# body is rejected (like an elided-return `def`).
def testLambdaBareNonNoneBody() raises:
  # expected-error @+1 {{cannot implicitly convert 'IntLiteral[5]' value to 'None' in return value}}
  _ = lambda: 5

# Elided return type defaults to None (like a `def` with no `->`), so a non-None body is
# rejected just as `def f(x: Int): return x + 1` is.
def testLambdaElidedReturnNonNoneBody() raises:
  # expected-error @+1 {{cannot implicitly convert 'Int' value to 'None' in return value}}
  _ = lambda (x: Int) {}: x + 1

# A specific capture list names some outer vars but not others; a used-but-unnamed
# outer var has no capture convention (and there's no capture-all), so it is rejected.
def testLambdaUnnamedUsedCapture() raises:
  var a = 1
  var b = 2
  # expected-error @+1 {{Could not infer capture convention of the captured value b}}
  _ = lambda (x: Int) {imm a} -> Int: x + a + b

# An explicit `{}` means "no captures" and stays distinct from an elided list: a
# free variable used under `{}` still errors (it is NOT imm-captured by default,
# unlike the same body with the capture list omitted -- cf. withOmittedCaptures).
def testLambdaEmptyCaptureFreeVar() raises:
  var z = 10
  # expected-error @+1 {{Could not infer capture convention of the captured value z}}
  _ = lambda (x: Int) {} -> Int: x + z

# An `imm` capture is immutable, so calling a mutating method on it is rejected.
struct IntList:
  def __init__(out self): pass
  # expected-note @+1 {{function declared here}}
  def append(mut self, value: Int): pass

def testLambdaReadCaptureMutated() raises:
  var lst = IntList()
  # expected-error @+1 {{invalid use of mutating method on rvalue of type 'IntList'}}
  _ = lambda (x: Int) {imm lst} -> None: lst.append(x)

##===----------------------------------------------------------------------===##
# Parameter-context lambdas: a thin lambda folds like a `def` reference; a
# capturing lambda is a runtime value and is rejected up front.
##===----------------------------------------------------------------------===##

# An explicit capture list (a capture-all convention or a named capture) makes
# the lambda a stateful runtime value, so binding it to a `comptime` is rejected
# -- as a capturing `def` closure bound to a `comptime` is too.
def testLambdaComptimeCapturing() raises:
  var z = 10
  # expected-error @+1 {{cannot use a capturing lambda in comptime initializer}}
  comptime f = lambda (x: Int) {imm} -> Int: x + z

def testLambdaComptimeNamedCapture() raises:
  var z = 10
  # expected-error @+1 {{cannot use a capturing lambda in comptime initializer}}
  comptime f = lambda (x: Int) {imm z} -> Int: x + z

# A capture spec reifies even with no free variable: a `{mut}` capture-all whose
# body captures nothing is still a capturing (runtime) lambda, rejected up front.
def testLambdaComptimeCapturingNoFreeVar() raises:
  # expected-error @+1 {{cannot use a capturing lambda in comptime initializer}}
  comptime f = lambda (x: Int) {mut} -> Int: x + 1

# Same for a written `{imm}` -- the convention the elided default uses. Eliding
# is not sugar for writing `{imm}`: the written form reifies (rejected here),
# while the elided form stays thin when nothing is captured (cf. the positive
# withComptimeBoundElided).
def testLambdaComptimeReadNoFreeVar() raises:
  # expected-error @+1 {{cannot use a capturing lambda in comptime initializer}}
  comptime f = lambda (x: Int) {imm} -> Int: x + 1

# A free variable under the default (elided) capture list is a runtime `imm`
# capture, so the lambda is dynamic and rejected in a comptime initializer --
# exercising the post-body capture check.
def testLambdaComptimeElidedFreeVar() raises:
  var z = 10
  # expected-error @+1 {{cannot use a capturing lambda in comptime initializer}}
  comptime f = lambda (x: Int) -> Int: x + z

# A thin lambda in a parameter position folds to a function literal, as a `def`
# reference does. Against non-`thin` `def(x: Int) -> Int` (a trait, not a type)
# it fails with the same error a `def` reference gets -- def-parity.
# expected-note @+1 {{function declared here}}
def takesFnParam[F: def(x: Int) -> Int]():
  pass

def testLambdaNonThinCallParam() raises:
  # expected-error @+2 {{'takesFnParam' parameter 'F' has 'def(x: Int) -> Int' type}}
  # expected-note @+1 {{a thin function cannot bind to a closure trait}}
  takesFnParam[lambda (x: Int) {} -> Int: x + 1]()

# A `def` reference is rejected identically -- def-parity, not a lambda-specific
# limitation.
def someIntFn(x: Int) -> Int:
  return x + 1

# expected-note @+1 {{function declared here}}
def takesFnParam2[F: def(x: Int) -> Int]():
  pass

def testDefRefNonThinCallParam() raises:
  # expected-error @+2 {{'takesFnParam2' parameter 'F' has 'def(x: Int) -> Int' type}}
  # expected-note @+1 {{a thin function cannot bind to a closure trait; use 'type_of(someIntFn)'}}
  takesFnParam2[someIntFn]()

# expected-error @+1 {{value to 'def(x: Int) -> Int'}}
def testLambdaNonThinDefaultParam[F: def(x: Int) -> Int = lambda (x: Int) {} -> Int: x + 1]():
  pass

# A capturing lambda is not a parameter value at all -- rejected up front in any
# parameter context, including a `thin` one a thin lambda would otherwise satisfy.
def takesThinFnParam[F: def(x: Int) thin -> Int]():
  pass

def testLambdaCapturingCallParam() raises:
  var z = 10
  # expected-error @+1 {{cannot use a capturing lambda in type parameter}}
  takesThinFnParam[lambda (x: Int) {imm z} -> Int: x + z]()

# expected-error @+1 {{cannot use a capturing lambda in default parameter}}
def testLambdaCapturingDefaultParam[F: def(x: Int) thin -> Int = lambda (x: Int) {mut} -> Int: x + 1]():
  pass

# A written capture CONVENTION (`{imm}`/`{mut}`) reifies a closure instance
# even when it captures nothing, so it does not decay -- unlike a written `{}`,
# which is explicitly thin and decays like the elided form.
def testLambdaCapturingNoRuntimeDecay() raises:
  var z = 10
  # expected-error @+1 {{cannot implicitly convert}}
  var f: def(x: Int) thin -> Int = lambda (x: Int) {imm z} -> Int: x + z
  # expected-error @+1 {{cannot implicitly convert}}
  var g: def(x: Int) thin -> Int = lambda (x: Int) {imm} -> Int: x + 1

# A lambda with unbound parameters of its OWN is not a single runtime value;
# it keeps the closure-instance form (which binds parameters at the call), so
# it does not decay either. An enclosing-parameter REFERENCE is different --
# bound at promotion -- and does decay (positive case in exprs/lambda.mojo).
def testLambdaParametricNoRuntimeDecay() raises:
  # expected-error @+1 {{cannot implicitly convert}}
  var f: def(x: Int) thin -> Int = lambda [N: Int](x: Int) {} -> Int: x + N

##===----------------------------------------------------------------------===##
# References and Transfer
##===----------------------------------------------------------------------===##

struct CopyAndInitMemType(ImplicitlyCopyable):
  def __init__(out self): pass
  def __init__(out self, *, copy: Self): pass
  # expected-note @+1 {{function declared here}}
  def __le__(self, other: Self) -> Self: return self
  def __mlir_bool__(self) -> __mlir_type.`!kgen.scalar<bool>`: pass

def compare_mem_result():
  var x = CopyAndInitMemType()
  # https://github.com/modular/mojo/issues/1115
  # expected-error @+1 {{chained comparison operator does not currently support memory-only return types}}
  x <= x <= x

def test_bad_ref(a: Int, b: CopyAndInitMemType):
  var bref = Pointer(to=b) # ok

  # expected-error @+1 {{invalid call to '__le__': value passed to 'other' cannot be converted from 'Pointer[CopyAndInitMemType, origin_of(b)]' to 'CopyAndInitMemType'}}
  _ = b <= bref

def transfer_diags[param: String](borrowed_arg: CopyAndInitMemType, obj: SomeNonTrivRegPassable, *vararg: String):
  var mem3 = CopyAndInitMemType()

  # Test pointless transfers from RValues and trivial values.
  # These should warn and not create IR transfers.

  # First transfer is ok.
  _ = mem3^
  _ = mem3^^ # expected-warning {{transfer from an owned value has no effect and can be removed}}

  # Already an rvalue.
  _ = CopyAndInitMemType()^ # expected-warning {{transfer from an owned value has no effect and can be removed}}

  var someInt = 4
  _ = someInt^ # expected-warning {{transfer from a value of trivial register type 'Int' has no effect and can be removed}}

  var someInt2 = 4
  someInt2 = 4
  _ = someInt2^ # expected-warning {{transfer from a value of trivial register type 'Int' has no effect and can be removed}}

  # MOCO-757: Transfer ^ of read-only arg leads to double free
  # expected-error @+1 {{cannot transfer out of immutable reference}}
  _ = borrowed_arg^

  # expected-error @+1 {{cannot transfer out of immutable reference}}
  _ = obj^

  # expected-error @+1 {{cannot transfer from a parameter expression; did you want to introduce a local 'var'?}}
  _ = param^

  # DLValue.
  # expected-error @+1 {{expression does not designate a value with an origin}}
  _ = vararg[1]^

# Issue #1708: https://github.com/modular/mojo/issues/1708
# Issue #1699: https://github.com/modular/mojo/issues/1699
# Issue #30790: https://github.com/modularml/modular/issues/30790
struct SomeThing(Movable where False):
    def overloaded[a: Int](self, b: Int) -> Int: pass
def testSomeThing(a: SomeThing):
  # expected-error @below {{member method closures are not supported; add '()' to call 'overloaded'}}
   a.overloaded[4] / 1.0

# Test invalid references that cannot bind to potentially-register_passable
# argument values.
# Issue #32603: References to read-only args in generics miscompile when instantiated on regpassable types
def get_ref_to_bad_argument[T: AnyType](a: T, *args: T):
  # These are all fine since they are not returned.
  _ = Pointer(to=a)
  _ = origin_of(a)
  _ = __get_mvalue_as_litref(a)
  # This is okay. The VariadicList has a origin.
  _ = Pointer(to=args)
  _ = Pointer(to=args[0])

struct NonTrivialReg(RegisterPassable):
  pass

def get_ref_to_reg_variadic(*args: NonTrivialReg):
  _ = Pointer(to=args[0])

def variadic_int(*x: Int) -> Bool: pass

# https://github.com/modularml/modular/issues/34675
def invalid_call_variadic_int(a: Int):
    # expected-error @+1 {{cannot use a dynamic value in 'ref' argument}}
    comptime if variadic_int(a, a):
        pass

def test_bad_ref_errors[T: AnyType](a: Pointer[T, _], b: Pointer[T, _]):
  # expected-error @below {{cannot implicitly convert 'T' value to 'Pointer[T, b.origin]'}}
  var x : Pointer[T, b.origin] = a[]

  # expected-error @below {{cannot implicitly convert 'T' value to 'Pointer[T, MutUnsafeAnyOrigin]'}}
  var y : Pointer[T, AnyOrigin[mut=True], address_space=a.address_space] = a[]

def test_subscript_conflict(a: Int):
  # expected-error @below {{keyword parameter 'idx' was already used; remove the duplicate}}
  # expected-note @below {{previously specified here}}
  _ = a[idx=4, idx=7]


struct Addable(Movable where False):
    def __add__(self, other: Self): pass # expected-note {{function declared here}}
def test(a: Pointer[Addable, _], b: Addable):
    # expected-error @+1 {{invalid call to '__add__': value passed to 'other' cannot be converted from 'Pointer[Addable, origin]' to 'Addable'}}
    _ = b+a


# Verify that we can propagate parametric mutability through field accesses.
struct ThingWithFields(Movable where False):
  var field: Int

def field_sensitive_origins(a: ThingWithFields)
    -> Pointer[ThingWithFields, origin_of(a.field)]:

  # expected-error @+1 {{'ThingWithFields' value has no attribute 'field_abc'}}
  _ = origin_of(a.field_abc)
  # expected-error @+1 {{'__mlir_type.index' has no attributes}}
  _ = origin_of(__mlir_type.index.field_abc)

  # expected-error @+1 {{cannot implicitly convert 'ThingWithFields' value to 'Pointer[ThingWithFields, origin_of(a.field)]'}}
  return a


def bad_named_return(out output: String):
   output = "emplaced!"
   # expected-note @below {{remove the expression if the return slot is already initialized}}
   # expected-error @below {{'out' argument cannot be returned by name; remove the return statement entirely or change it to just 'return'}}
   return output


def bad_named_return2(out output: Int):
   output = 42
   # expected-note @below {{remove the expression if the return slot is already initialized}}
   # expected-error @below {{'out' argument cannot be returned by name; remove the return statement entirely or change it to just 'return'}}
   return output

def unbound_function_type():
  # expected-error @below {{function type missing required origin set parameter}}
  var f: def() thin [_] -> None

  # expected-error @below {{cannot use parametric function as a runtime closure}}
  # expected-note @below {{parameter 'p' of type 'Int' is not bound}}
  var g: def(HasIntParam) thin -> None

  # expected-error @below {{'HasIntParamAlias' is not a concrete type, use '[]' to bind missing parameters}}
  # expected-note @below {{'HasIntParamAlias' is aka 'comptime[p: Int] HasIntParam[p]'}}
  var h: HasIntParamAlias

# Various type printing cases.
#
struct HasKWOnlyParam[*, kwplz: Int](Movable where False): pass

# expected-note @+1 {{function declared here}}
def test_kw_only[a: Int](arg: HasKWOnlyParam[kwplz=a]):
  # expected-error @+1 {{invalid call to 'test_kw_only': value passed to 'arg' cannot be converted from 'HasKWOnlyParam[kwplz=a]' to 'HasKWOnlyParam[kwplz=Int(42)]'}}
  test_kw_only[42](arg)

struct HasMultipleOnlyParam[x: Int = 1, y: Int = 4](Movable where False): pass

# expected-note @+1 {{function declared here}}
def test_mixedkw_only[a: Int](arg: HasMultipleOnlyParam[1, a]):
  # expected-error @+1 {{invalid call to 'test_mixedkw_only': value passed to 'arg' cannot be converted from 'HasMultipleOnlyParam[y=a]' to 'HasMultipleOnlyParam[y=Int(42)]'}}
  test_mixedkw_only[42](arg)

def int_fn(arg: Int) -> Int: return arg+1
struct HasDependent[x: Int, y: Int = int_fn(x)](Movable where False): pass

# expected-note @+1 {{function declared here}}
def test_dependent[a: Int](arg: HasDependent[a], arg2: HasDependent[a, 4]):
  # expected-error @+1 {{invalid call to 'test_dependent': value passed to 'arg' cannot be converted from 'HasDependent[a]' to 'HasDependent[Int(42)]'}}
  test_dependent[42](arg, arg2)
  # expected-error @below {{invalid call to 'test_dependent': value passed to 'arg2' cannot be converted from 'HasDependent[a]' to 'HasDependent[a, Int(4)]'}}
  # expected-note @below {{types parameters include unfolded expression at parser time; try rebinding to a consistent type?}}
  test_dependent(arg, arg)

struct HasIntParam[p: Int](Movable where False):
  def __init__(out self): # expected-note {{function declared here}}
     pass

comptime HasIntParamAlias[p: Int] = HasIntParam[p]

# MOCO-846: Poor error message when type conversion fails due to IntLiteral materialization

# expected-note @below {{function declared here}}
def take_dep_args[width: Int, x: IntLiteral](a: HasIntParam[width], b: HasIntParam[width * 4]):
  # expected-error @below {{invalid call to 'take_dep_args': value passed to 'b' cannot be converted from 'HasIntParam[(x.value * 4)]' to 'HasIntParam[(SIMD(x) * Int(4))]'}}
  take_dep_args[x, x](HasIntParam[x](), HasIntParam[x*4]())

def test_signature():
  # expected-error @+1 {{cannot implicitly convert 'def __init__() thin -> HasIntParam[Int(1)]' value to 'def(x: HasIntParam[Int(1)]) thin -> None' in 'var' initializer}}
  var x : def(x: HasIntParam[1]) thin -> None = HasIntParam[1].__init__

  # expected-error @+1 {{use of unknown declaration 'UndefinedStruct'}}
  var y : def(x: UndefinedStruct) thin -> None = HasIntParam[1].__init__

  var z : def(out x: HasIntParam[1]) thin = HasIntParam[1].__init__

  var str : HasIntParam[1]
  # expected-error @+1 {{invalid call to '__init__': unexpected argument}}
  HasIntParam[1].__init__(str)

def bad_union[ao: Origin[mut=True]](ref [ao] a: String, mut b: String) -> ref [a, b] String:
    var c: String
    # expected-error @below {{cannot return reference with incompatible origin: 'origin_of(c)' vs 'origin_of(ao, b)'}}
    return c

# https://github.com/modular/mojo/issues/3829
def apply_in_memory[o: ImmOrigin](f: def(ref[o] x: SomeNonTrivRegPassable) thin -> None, x: SomeNonTrivRegPassable):
# expected-error @below {{value passed to 'x' cannot be converted from 'SomeNonTrivRegPassable' to ref 'SomeNonTrivRegPassable'}}
# expected-note @below {{operand origin 'origin_of(x)' doesn't match expected origin 'origin_of(o)'}}
    f(x)


# Problems binding SRValues to ref arguments.
# https://github.com/modular/mojo/issues/3830

def getSomeNonTrivRegPassable() -> SomeNonTrivRegPassable: pass
# expected-note @below {{function declared here}}
def direct3830(ref a: SomeNonTrivRegPassable, ref[a] b: SomeNonTrivRegPassable): pass
def test3830():
    # expected-error @below {{invalid call to 'direct3830': value passed to 'b' cannot be converted from 'SomeNonTrivRegPassable' to ref 'SomeNonTrivRegPassable'}}
    # expected-note @below {{operand origin 'origin_of(anonymous*)' doesn't match expected origin 'origin_of(anonymous*)'}}
    direct3830(getSomeNonTrivRegPassable(), getSomeNonTrivRegPassable())

def test3830_1[o: ImmOrigin](f: def(ref[o] x: SomeNonTrivRegPassable) thin -> None):
    # expected-error @below {{invalid indirect call: value passed to 'x' cannot be converted from 'SomeNonTrivRegPassable' to ref 'SomeNonTrivRegPassable'}}
    # expected-note @below {{operand origin 'origin_of(anonymous*)' doesn't match expected origin 'origin_of(o)'}}
    f(getSomeNonTrivRegPassable())

def test3830_2[o: ImmOrigin](f: def(ref[o] x: Int) thin -> None, x: Int):
    # expected-error @below {{invalid indirect call: value passed to 'x' cannot be converted from 'Int' to ref 'Int'}}
    # expected-note @below {{operand origin 'origin_of(anonymous*)' doesn't match expected origin 'origin_of(o)'}}
    f(x)

struct Struct3855(Movable where False):
    var l: Int
    def do(self, ref[self.l] e: Int): pass # expected-note {{function declared here}}

def testStruct3855(t: Struct3855):
    # expected-error @+2 {{value passed to 'e' cannot be converted from 'Int' to ref 'Int'}}
    # expected-note @+1 {{operand origin 'origin_of(t.l)' doesn't match expected origin 'origin_of(self.l)'}}
    t.do(t.l)


struct MovableAndExplicitCopyable(Copyable):
    def __init__(out self): pass

struct MovableOnly(Movable):
    def __init__(out self): pass

def test_implicit_copy_errors():
    var a1 = MovableAndExplicitCopyable()
    # expected-error @below {{value of type 'MovableAndExplicitCopyable' cannot be implicitly copied, it does not conform to 'ImplicitlyCopyable'}}
    # expected-note @below {{consider transferring the value with '^'}}
    # expected-note @below {{you can copy it explicitly with '.copy()'}}
    var a2 = a1
    var b1 = MovableOnly()
    # expected-error @below {{value of type 'MovableOnly' cannot be implicitly copied, it does not conform to 'ImplicitlyCopyable'}}
    # expected-note @below {{consider transferring the value with '^'}}
    var b2 = b1

    var x = MemType()
    # expected-error @below {{value of type 'MemType' cannot be implicitly copied, it does not conform to 'ImplicitlyCopyable'}}
    # expected-note @below {{consider transferring the value with '^'}}
    x.consume()

    # expected-error @below {{cannot transfer value into destination, because 'MemType' doesn't conform to 'Movable'}}
    var y = x^

    var l1 = List[Int]()
    # expected-error @below {{value of type 'List[Int]' cannot be implicitly copied, it does not conform to 'ImplicitlyCopyable'}}
    # expected-note @below {{consider transferring the value with '^'}}
    # expected-note @below {{you can copy it explicitly with '.copy()'}}
    var l2 = l1

##===----------------------------------------------------------------------===##
# MergeWith
##===----------------------------------------------------------------------===##

struct TypeA(Movable where False):
    def __merge_with__[other_type: type_of(TypeB)](self) -> TypeB:
        pass
    def __merge_with__[other_type: type_of(TypeC)](self) -> Int:
        pass

struct TypeB(Movable where False):
    def __merge_with__[other_type: type_of(TypeA)](self) -> Int:
        pass

struct TypeC(Movable where False):
    pass


# CHECK-LABEL: lit.fn @"test_mergewith
def test_mergewith(cond: Bool, a: TypeA, b: TypeB, c: TypeC):
  # expected-error @+2 {{value of types 'TypeA' and 'TypeB' have '__merge_with__' methods that disagree on common type}}
  # expected-note @+1 {{one returns 'TypeB' and the other returns 'Int'}}
  _ = a if cond else b

  # expected-error @+2 {{value of types 'TypeA' and 'TypeC' cannot be merged to type 'Int'}}
  # expected-note @+1 {{'TypeC' does not implicitly convert to 'Int'}}
  _ = a if cond else c

def test_mergewith_pointer():
    var a = 1
    var b = 2
    var c = 3

    # FIXME: This really should work, we need to figure out how exclusivity
    # works here.

    # expected-error @below {{aliasing values passed mutably to 'elements' argument and passed mutably to 'elements' argument in 'Array[Pointer[Int, origin_of(a, b)], Int(2)]' initializer call}}
    # expected-note @below {{'origin_of(a)' memory accessed through reference embedded in value of type 'Pointer[Int, origin_of(a, b)]'}}
    # expected-note @below {{'origin_of(b)' memory accessed through reference embedded in value of type 'Pointer[Int, origin_of(a, b)]'}}
    for elt in [Pointer(to=a), Pointer(to=b)]:
        elt[] *= 2


def test_var_decl_error():
  var a = Y # expected-error {{use of unknown declaration 'Y'}}
  var b = a # no secondary error.
  var c = b+1 # no tertiary error.

# MOCO-2094 - String memory leak observed in _get_dylib_function
def test_comptime_materialize():
  # This is ok!
  comptime bad = String("hello").unsafe_ptr()
  # This is ok too.
  comptime byte = bad[]
  # Swimmingly fine.
  var rt_byte = byte

  # expected-error @below {{cannot materialize compile-time value of type 'Pointer[UInt8, ComptimeOrigin]' to a runtime value}}
  # expected-note @below {{the type contains an origin referring to a compile-time value}}
  var use_bad = bad

struct BoolParam[value: Bool](Movable where False):
  pass

def elide_implicit_conversion_in_struct_params[value: __mlir_type.i1](a: BoolParam[value]):
  # expected-error @below {{cannot implicitly convert 'BoolParam[value]' value to 'Int'}}
  var x : Int = a


# MOCO-2332 / https://github.com/modular/modular/issues/5139
struct a_struct(Movable where False):
  comptime an_alias = 1

def a_fn() -> Dict[String, Int]:
  # expected-error @below {{'a_struct' value has no attribute 'an_alias_that_does_not_exist'}}
  return {"an_alias": a_struct.an_alias_that_does_not_exist}


##===----------------------------------------------------------------------===##
# Trait member access errors
##===----------------------------------------------------------------------===##

trait TraitWithMember:
    def member_method(self) -> Int:
        return 42

    comptime MemberAlias: AnyType

def test_trait_member_access_error():
    # expected-error @below {{Direct access of trait members is not supported.}}
    _ = TraitWithMember.member_method

    # expected-error @below {{Direct access of trait members is not supported.}}
    comptime SomeAlias = TraitWithMember.MemberAlias


##===----------------------------------------------------------------------===##
# Deprecated magic functions
##===----------------------------------------------------------------------===##

# expected-error @+1 {{use of unknown declaration '__type_of'; did you mean 'type_of'?}}
def test_type_of_deprecated(x: Int) -> __type_of(x):
    return x

def test_origin_of_deprecated[T: AnyType](a: T):
  # expected-error @+1 {{use of unknown declaration '__origin_of'; did you mean 'origin_of'?}}
  _ = __origin_of(a)

##===----------------------------------------------------------------------===##
# Type sugar processing
##===----------------------------------------------------------------------===##


def some_complex_calculation() -> Int: return 4
comptime ideal_width = some_complex_calculation()*4
comptime IdealSIMD = SIMD[.int32, ideal_width]

def get_data() -> IdealSIMD: return IdealSIMD()


# expected-note @+1 {{function declared here}}
def sugar_test1(x: type_of(HasIntParam[1])): pass

def sugar_test():
    # expected-error @below {{invalid call to 'sugar_test1': value passed to 'x' cannot be converted from 'AnyStruct[HasIntParam[int_fn(Int(0))]]' to 'AnyStruct[HasIntParam[Int(1)]]'}}
    # expected-note @below {{.p of the first value is 'int_fn(Int(0))' but the second value is 'Int(1)'}}
    # expected-note @below {{types parameters include unfolded expression at parser time; try rebinding to a consistent type?}}
    sugar_test1(HasIntParam[int_fn(0)])

    var a = get_data()  # Ok
    var b : SIMD[.int32, 4]

    # expected-error @below {{cannot implicitly convert 'IdealSIMD' value to 'SIMD[.int32, 4]'}}
    # expected-note @below {{'IdealSIMD' is aka 'SIMD[.int32, Int((mul some_complex_calculation(), 4))]'}}
    b = get_data()

    var c = a.join(a) # c has twice the width.

    # expected-error @below {{cannot implicitly convert 'SIMD[.int32, (SIMDLength(Int((mul some_complex_calculation(), 4))) * 2)]' value to 'SIMD[.int32, 4]'}}
    # expected-note @below {{.size of the first value is '(SIMDLength(Int((mul some_complex_calculation(), 4))) * 2)' but the second value is '4'}}
    b = c

    # expected-error @below {{cannot implicitly convert 'IdealSIMD' value to 'SIMD[.int32, 4]'}}
    # expected-note @below {{'IdealSIMD' is aka 'SIMD[.int32, Int((mul some_complex_calculation(), 4))]'}}
    b = a+a


# PR5618 - Compiler crash when should be implicit conversion error
struct MemberAliasSugarCrash(TrivialRegisterPassable):
    comptime ValueType = Int
    var _value: Self.ValueType

    def __init__(out self, v: Self.ValueType):
        self._value = v

    def method(self) -> Self:
      # expected-error @below {{cannot implicitly convert 'MemberAliasSugarCrash.ValueType' value to 'MemberAliasSugarCrash'}}
      # expected-note @below {{'MemberAliasSugarCrash.ValueType' is aka 'Int'}}
        return self._value


##===----------------------------------------------------------------------===##
# comptime expression
##===----------------------------------------------------------------------===##

struct NotRuntimeMaterializable(Movable where False):
    def method(self) -> Int: pass

def test_comptime_expression[nrm: NotRuntimeMaterializable]():
    # expected-error @below {{expression is already evaluated at compile time; remove 'comptime' keyword}}
    test_comptime_expression[comptime(nrm)]()

    # expected-error @below {{cannot materialize comptime value of type 'NotRuntimeMaterializable' to runtime because it is not 'ImplicitlyCopyable'}}
    # expected-note @below {{use 'comptime' to evaluate the entire call at 'comptime' and materialize its result}}
    var a = nrm.method()

    # ok.
    var b = comptime(nrm.method())

    # expected-error @below {{cannot materialize comptime value of type 'NotRuntimeMaterializable' to runtime because it is not 'ImplicitlyCopyable'}}
    # expected-note @below {{use 'comptime' to evaluate the entire call at 'comptime' and materialize its result}}
    var c = comptime(nrm).method()

    # expected-error @below {{expected '(' after 'comptime'}}
    var d = comptime nrm.method()



struct TwoParamsType[a: Int, b: Int](Movable where False):
    pass
comptime TwoParamsTypeAlias[B: Int] = TwoParamsType[B, ...]
# expected-note @+1 {{function declared here}}
def take_anytype[T: AnyType]():
  # expected-error @below {{'take_anytype' parameter 'T' has 'AnyType' type, but value has type '__generator_type[B: Int] AnyStruct[TwoParamsType[B, _]]'}}
    take_anytype[TwoParamsTypeAlias]()

def ternary_missing_else(a: Int, b: Int) -> Int:
  # expected-error @+1 {{expected 'else' clause in ternary; add 'else' and the false branch}}
  return a if a > b


##===----------------------------------------------------------------------===##
# Inferred attribute references (`.member`)
##===----------------------------------------------------------------------===##

struct Color(ImplicitlyCopyable):
  comptime red = Color()
  comptime green = Color()
  comptime blue = Color()
  comptime size: Int = 42

  @staticmethod
  def hsb_to_rgb(h: Int, s: Int, b: Int) -> Color:
    return Color()

  @staticmethod
  def alpha_blended[a: Int](x: Int, y: Int) -> Color:
    return Color()

  def opacity(self, amount: Float64) -> Color:
    return Color()

  def __init__(out self):
    pass

def takes_color(c: Color):  # expected-note {{function declared here}}
  pass

def takes_colors(colors: List[Color]):
  pass

def test_inferred_attribute_ref():
  # Without a contextual type the base cannot be inferred.
  # expected-error @below {{cannot resolve inferred member without a contextual type}}
  _ = .red

  # Call arguments provide a contextual type, so `.green` resolves as
  # `Color.green`.
  takes_color(.green)

  var x: Color = .blue
  takes_color(x)

  # static methods also resolve.
  takes_color(.hsb_to_rgb(120, 100, 50))

  # Parentheses around inferred refs are transparent.
  takes_color((.hsb_to_rgb)(120, 100, 50))

  # Parametric static methods resolve too.
  takes_color(.alpha_blended[42](1, 2))

  # Chained members resolve the leading inferred base, then continue normally.
  takes_color(.red.opacity(0.5))

  # List elements resolve against the element type from List[Color].
  takes_colors([.red, .green, .blue])
  var palette: List[Color] = [.red, .hsb_to_rgb(120, 100, 50)]

  # The member resolves on Color, but its type is Int rather than Color.
  # expected-error @below {{cannot implicitly convert 'Int' value to 'Color'}}
  var wrong: Color = .size

  # expected-error @below {{cannot implicitly convert 'Int' value to 'Color'}}
  # expected-error @below {{invalid call to 'takes_color': cannot resolve inferred attribute reference}}
  takes_color(.size)

def test_dtype_error_message():
    # expected-error @below {{cannot implicitly convert 'SIMD[.float32, 3]' value to 'SIMD[.float32, 4]'}}
    var x: SIMD[DType.float32, 4] = SIMD[DType.float32, 3](1.0)
