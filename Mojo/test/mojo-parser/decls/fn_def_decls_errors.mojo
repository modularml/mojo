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

# RUN: %parse-mojo-isolated %s -verify-diagnostics -o /dev/null

@fieldwise_init
struct MemoryType(ImplicitlyCopyable):
    pass

def test_never_declared_fn():
    # expected-error @+1 {{use of unknown declaration 'never_declared_fn'}}
    never_declared_fn()

def implicit_var_decl(a: Int):
    # expected-error @+1 {{implicit declaration of 'c' is not allowed; add 'var' to declare a new name}}
    c = a

struct BadMethod(Movable where False):
    # expected-error @+1 {{'__add__' requires 2 operands}}
    def __add__(self):
        pass

# expected-error @+1 {{'__sub__' must be a method, not a global function}}
def __sub__(self: Int, a: Int):
     pass

def missing_colon()  # expected-error {{expected ':' in function definition}}
    # Don't get confused by comments or blank lines!

    var x = 1

# Missing colon after def definition complains about function effects
# https://github.com/modularml/modular/issues/23359
# expected-error @+1 {{missing ':' at end of function signature}}
def missing_colon_2()
    test_never_declared_fn()

# expected-error @+1 {{function effect 'thin' must only be used on function types}}
def invalid_thin_effect() thin:
    pass

# expected-error @+1 {{function effect 'thin' must only be used on function types}}
def invalid_thin_before_abi() thin abi("C"):
    pass

# expected-error @+1 {{function effect 'thin' must only be used on function types}}
def invalid_abi_before_thin() abi("C") thin:
    pass

# expected-error @below {{expected argument name}}
def missing_argument_name(*: Int): pass

# expected-error @below {{expected parameter name}}
def missing_parameter_name[: Int](): pass

# expected-error @+1 {{use of unknown declaration 'InvalidType'}}
def test_unknown_arg_type(a: InvalidType) raises:
    _ = a.value  # Should not produce a follow-on error.
    return

# expected-error @+1 {{cannot have two '*' markers in the same argument list}}
def two_stars(a: Int, *, *, b: Int):
    pass

# expected-error @+1 {{cannot have two '/' markers in the same argument list}}
def two_slashes(a: Int, /, /, b: Int):
    pass

# expected-error @+1 {{cannot specify '/' marker after '*' marker}}
def slash_after_start(a: Int, *, /, b: Int):
    pass

# expected-error @+1 {{'/' marker cannot be used at the start of the argument list}}
def leading_slash(/, a: Int):
    pass

# expected-error @+1 {{'*' marker is not allowed at end of argument list}}
def trailing_star(a: Int, *):
    pass

# # expected-error @+1 {{cannot have two '*' markers in the same argument list}}
def two_variadics(*a: Int, *b: Int):
    pass

# expected-error @+1 {{cannot have two '*' markers in the same argument list}}
def two_variadic_packs[*Ts: TrivialRegisterPassable](*a: *Ts, *b: *Ts):
    pass

# expected-error @+2 {{parametric functions must not be used as arguments; pass as a parameter instead}}
# expected-note @+1 {{alternatively, bind its type parameters to create a concrete function}}
def foo(x: def[a: Int] () thin -> None):
    pass

# Member-alias sugar must not bypass the non-singleton parameter check.
struct ParametricFnType:
    comptime type = def[a: Int] () thin -> None


# expected-error @+2 {{parametric functions must not be used as arguments; pass as a parameter instead}}
# expected-note @+1 {{alternatively, bind its type parameters to create a concrete function}}
def parametric_fn_alias_as_arg(func: ParametricFnType.type):
    pass

# Bare `ref` invents a non-singleton Bool mutability parameter; only unbound
# origin / other singleton parameters are allowed on runtime function arguments.
# expected-error @+2 {{parametric functions must not be used as arguments; pass as a parameter instead}}
# expected-note @+1 {{alternatively, bind its type parameters to create a concrete function}}
def unbound_ref_fn_as_arg(func: def(ref arg: Int) thin -> None):
    pass

# expected-error @below {{non-owned variadic keyword arguments are not supported yet}}
def borrowed_kwargs(imm **kwargs: Int):
    pass

# expected-error @below {{variadic keyword arguments only support the 'var' convention; add 'var' before '**'}}
def bare_kwargs(**kwargs: Int):
    pass

def bare_kwargs_fn_type(
    # expected-error @below {{variadic keyword arguments only support the 'var' convention; add 'var' before '**'}}
    f: def(**kwargs: Int) -> None,
):
    pass

# The parser recovers from a missing 'var' as if it were written, so later
# arguments are still checked.
# expected-error @below {{variadic keyword arguments only support the 'var' convention; add 'var' before '**'}}
# expected-error @below {{'**' marker must be at end of argument list}}
def bare_kwargs_not_last(**kwargs: Int, b: Int):
    pass

# expected-error @below {{'//' marker cannot be used at the start of the parameter list}}
def invalid_inferred[//, x: Int]():
    pass

# expected-error @below {{cannot specify '//' marker after '*' marker in parameter list}}
def invalid_inferred_kw_only[*, x: Int, //, y: Int]():
    pass

# expected-error @below {{'//' is reserved for parameter lists; it marks the end of 'infer-only' parameter listings}}
def invalid_inferred_argument(x: Int, //):
    pass


struct NonCopyable(Movable where False):
    def __init__(out self):
       pass

def defTests() raises -> None:
  comptime abc = 1

  # MOCO-83: [mojo][Bug] def methods can't shadow names via assignment
  # expected-error @+1 {{expression must be mutable in assignment}}
  abc = 4

# expected-error @+1 {{value of type 'IntLiteral[4]' doesn't have a memory origin in origin specifier}}
def ref_result_invalid1() -> ref [4] MemoryType:
    pass

def ref_result_invalid2(mut a: MemoryType) -> ref [a] Int:
    # expected-error @+1 {{value of type 'Int' doesn't have a memory origin in return value}}
    return 4

def ref_result_invalid3(mut a: MemoryType, mut b: MemoryType)
     -> ref [a] MemoryType:
    # expected-error @+1 {{cannot return reference with incompatible origin: 'origin_of(b)' vs 'origin_of(a)'}}
    return b

def ref_result_invalid4(mut a: MemoryType, b: Int) -> ref [a] MemoryType:
    # expected-error @+1 {{cannot implicitly convert 'Int' value to 'MemoryType'}}
    return b

# expected-error @+1 {{cannot return 'a's origin, because it might expand to a RegisterPassable type}}
def ref_result_invalid5[T: AnyType](a: T) -> ref [a] T:
    return a

# expected-error @+1 {{cannot return 'b's origin, because it has RegisterPassable type 'Int'}}
def ref_result_invalid6(mut a: MemoryType, mut b: Int) -> ref [b] Int:
    pass

# expected-error @+1 {{'ref' result requires an origin specifier}}
def ref_result_invalid7() -> ref MemoryType:
    pass

# expected-error @+1 {{value of type 'Int' doesn't have a memory origin in origin specifier}}
def ref_result_invalid8(a: Int) -> ref [a] MemoryType:
    pass

# expected-error @+1 {{cannot infer origin for a function result}}
def ref_result_invalid9() -> ref [_] MemoryType:
    pass

def valid_ref_result(x: MemoryType) -> ref [x] MemoryType: return x

def ref_invalid():
    var a = MemoryType()
    # expected-error @+1 {{expression must be mutable in assignment}}
    valid_ref_result(a) = MemoryType()

def return_ref_type_error(a: def (x: MemoryType) thin -> ref [x] MemoryType):
    # expected-error @+1 {{cannot implicitly convert 'def(x: MemoryType) thin -> ref[*[0,0]] MemoryType' value to 'Int'}}
    var b: Int = a

struct SBValue(RegisterPassable):
    pass

# expected-error @below {{TODO: read-only non-trivial register-passable arguments are not yet supported in async functions}}
async def invalid_sb_value(value: SBValue):
    pass

# expected-error @below {{TODO: read-only non-trivial register-passable arguments are not yet supported in async functions}}
async def invalid_sb_value_variadic(*value: SBValue):
    pass

async def borrowed_generic_arg[T: AnyType](value: T):
    pass

def valid_sbvalue_borrow(value: SBValue):
    _ = borrowed_generic_arg(value)

# expected-error @+1 {{address space must be specified once; remove duplicate address space specifications}}
def bad_ref_as[a: AddressSpace](ref [a, a] x: Int):
    pass

struct HasOwnedOverloadedMethod(Movable where False):
    def method(var self) -> Int: pass
    def method(self) -> String: pass

def testOverloadedMethod():
  var x : HasOwnedOverloadedMethod

  _: Int = x.method()    # expected-error {{cannot implicitly convert 'String' value to 'Int'}}
  _: String = x.method()

  _: Int = x^.method()
  _: String = x^.method() # expected-error {{cannot implicitly convert 'Int' value to 'String'}}
