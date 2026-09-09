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

# ===----------------------------------------------------------------------=== #
# Function decorators
# ===----------------------------------------------------------------------=== #

# Issue #14191
# expected-error @+1 {{unexpected tokens after decorator, each need to be on their own line}}
@always_inline wqeqwe
def issue14191() -> Int:
    return 1

def issue1242():
    var decorator: Int

    @decorator # expected-error {{cannot use a dynamic value in decorator}}
    def on_message() capturing: pass

@invalid_dec # expected-error {{use of unknown declaration 'invalid_dec'}}
def unknown_decorator(): pass

def decorator_on_statements() raises:
    @invalid_dec
    var decorated_var: Int  # expected-error {{'var' statement in function body does not support decorators; remove the decorator}}

    @invalid_dec
    comptime decorated_alias = 42  # expected-error {{'comptime' statement in function body does not support decorators; remove the decorator}}

    @invalid_dec
    while True:  # expected-error {{'while' statement does not support decorators; remove the decorator}}
        pass

    @invalid_dec
    _ = 1 + 1  # expected-error {{statement does not support decorators; remove the decorator}}


# expected-error @+1 {{decorators must be on their own line; add a newline after the decorator}}
@always_inline def same_line_decorator(): pass

# comptime if causes confusing indentation error message
# https://github.com/modularml/modular/issues/19163
def some_fn():
    # expected-error @below {{decorators must be on their own line; add a newline after the decorator}}
    @decorator if True:
        pass

def some_fn_2():
        # expected-error @below {{decorator must be followed by a definition on the next line; remove any blank lines between them}}
        @decorator
    # expected-error @+1 {{statement indentation must match the rest of the block; adjust to align}}
    if True: # expected-warning {{'if' condition always evaluates to 'True'; 'else' branch is unreachable}}
        pass

@decorator[]  # expected-error {{invalid expression in decorator}}
def bad_decorator_expression_1():
    pass

@decorator[]()  # expected-error {{invalid expression in decorator}}
def bad_decorator_expression_2():
    pass

@567  # expected-error {{invalid expression in decorator}}
def bad_decorator_expression_3():
    pass

# ===----------------------------------------------------------------------=== #
# @always_inline
# ===----------------------------------------------------------------------=== #

@always_inline("builtin", "nodebug")  # expected-error {{'@always_inline' decorator takes 0 or 1 arguments, found 2}}
def bad_always_inline_1():
    pass

@always_inline(123)  # expected-error {{'@always_inline' operand must be "nodebug" or "builtin"}}
def bad_always_inline_2():
    pass

@always_inline("no_debug")  # expected-error {{'@always_inline' operand must be "nodebug" or "builtin"}}
def bad_always_inline_3():
    pass

# ===----------------------------------------------------------------------=== #
# @staticmethod
# ===----------------------------------------------------------------------=== #

@staticmethod  # expected-error {{only methods on structs may be declared static}}
def not_a_struct_method(): pass

struct HasBadStaticMethod(Movable where False):
    @staticmethod()  # expected-error {{'@staticmethod' cannot have arguments}}
    def bad_static_method_1(): pass

    @staticmethod("abc")  # expected-error {{'@staticmethod' cannot have arguments}}
    def bad_static_method_2(): pass


# ===----------------------------------------------------------------------=== #
# @no_inline
# ===----------------------------------------------------------------------=== #

@no_inline()  # expected-error {{'@no_inline' cannot have arguments}}
def bad_no_inline_1() raises:
    pass

@no_inline("abc")  # expected-error {{'@no_inline' cannot have arguments}}
def bad_no_inline_2():
    pass


# ===----------------------------------------------------------------------=== #
# @implicit
# ===----------------------------------------------------------------------=== #

struct CheckImplicit(Movable where False):
    # expected-error @+1 {{'@implicit' may only be applied to '__init__' methods}}
    @implicit
    def foo(mut self): pass

    # expected-error @+2 {{'@implicit' initializers must accept a single positional argument value}}
    @implicit
    def __init__(out self): pass

    # expected-error @+2 {{'@implicit' initializers must accept a single positional argument value}}
    @implicit
    def __init__(out self, x: Int, y: Int): pass

    # expected-error @+2 {{'@implicit' initializers must accept a single positional argument value}}
    @implicit
    def __init__(out self, *, z: Int): pass

    # expected-error @+1 {{'@implicit' may only be applied to '__init__' methods}}
    @implicit
    def __init__(out self, *, copy: Self): pass

    # expected-error @+1 {{'@implicit' decorator takes 0 or 1 arguments, found 2}}
    @implicit(123, "abc")
    def __init__(out self, a: Int): pass

    # expected-error @+1 {{'@implicit' may only have a keyword argument 'deprecated' with literal boolean value}}
    @implicit(123)
    def __init__(out self, *, a: Int): pass

    # expected-error @+1 {{'@implicit' may only have a keyword argument 'deprecated' with literal boolean value}}
    @implicit(deprecated=123)
    def __init__(out self, b: String): pass

    # expected-error @+1 {{'@implicit' may only have a keyword argument 'deprecated' with literal boolean value}}
    @implicit(foo=True)
    def __init__(out self, c: Bool): pass

struct DeprecatedImplicitConversion(Movable where False):
    # expected-note @+2 {{implicit constructor for 'DeprecatedImplicitConversion' declared here}}
    @implicit(deprecated=True)
    def __init__(out self, value: Int):
        pass

def foo(y: DeprecatedImplicitConversion): pass

def foo(z: String): pass

def deprecated_implicit_conversion():
    # expected-warning @+2 {{deprecated implicit conversion from 'IntLiteral[1]' to 'DeprecatedImplicitConversion'}}
    # expected-note @+1 {{call 'DeprecatedImplicitConversion(...)' explicitly}}
    _: DeprecatedImplicitConversion = 1

    # expected-warning @+2 {{deprecated implicit conversion from 'Int' to 'DeprecatedImplicitConversion'}}
    # expected-note @+1 {{call 'DeprecatedImplicitConversion(...)' explicitly}}
    foo(Int(1))

# ===----------------------------------------------------------------------=== #
# @extern
# ===----------------------------------------------------------------------=== #

@extern  # expected-error {{'@extern' requires 1 argument}}
def bad_extern_1(): ...

@extern()  # expected-error {{'@extern' requires 1 argument}}
def bad_extern_2(): ...

@extern(123)  # expected-error {{'@extern' requires a string literal argument}}
def bad_extern_3(): ...

@extern("bad_extern", "bad_extern_3")  # expected-error {{'@extern' requires 1 argument}}
def bad_extern_4(): ...

# expected-error @+2 {{unexpected function body in extern function declaration, use `...`}}
@extern("add_one")
def my_extern_add_one(x: Int) abi("Mojo") -> Int:
    return x + 1

struct HasExtern(Movable where False):
  # expected-error @+1 {{'@extern' cannot be applied to a method}}
  @extern("add_one_struct")
  def my_extern_struct_add_one(self, x: Int) -> Int:
    ...

# expected-error @+1 {{'@extern' requires an explicit 'abi()' effect on the function}}
@extern("no_abi_extern")
def extern_missing_abi(): ...

# ===----------------------------------------------------------------------=== #
# @__llvm_metadata
# ===----------------------------------------------------------------------=== #

@__llvm_metadata  # expected-error {{'@__llvm_metadata' requires operands}}
def llvm_meta_no_arg_1[x: Int](a: Int, b: Int):
    pass

@__llvm_metadata()  # expected-error {{'@__llvm_metadata' requires operands}}
def llvm_meta_no_arg_2[x: Int](a: Int, b: Int):
    pass


# ===----------------------------------------------------------------------=== #
# @__llvm_arg_metadata
# ===----------------------------------------------------------------------=== #

@__llvm_arg_metadata  # expected-error {{'@__llvm_arg_metadata' requires operands}}
def llvm_arg_meta_no_arg_1[x: Int](a: Int, b: Int):
    pass

@__llvm_arg_metadata()  # expected-error {{'@__llvm_arg_metadata' requires operands}}
def llvm_arg_meta_no_arg_2[x: Int](a: Int, b: Int):
    pass

# expected-error @+1 {{First argument of '@__llvm_arg_metadata' must be an argument name}}
@__llvm_arg_metadata(1 + 1)
def llvm_arg_meta_wrong_type[x: Int](a: Int, b: Int):
    pass

# expected-error @+1 {{Function decorated by '@__llvm_arg_metadata' has no argument named 'c'}}
@__llvm_arg_metadata(c, myMeta)
def llvm_arg_meta_wrong_name[x: Int](a: Int, b: Int):
    pass

# ===----------------------------------------------------------------------=== #
# Closure decorators
# ===----------------------------------------------------------------------=== #

def outer_function():
    @__copy_capture  # expected-error {{'@__copy_capture' must have arguments}}
    @__parameter()  # expected-error {{'@__parameter' cannot have arguments}}
    def copy_capture_no_args_1():
        pass

    @__copy_capture()  # expected-error {{'@__copy_capture' must have arguments}}
    @__parameter("abc")  # expected-error {{'@__parameter' cannot have arguments}}
    def copy_capture_no_args_2():
        pass

    @__move_capture  # expected-error {{'@__move_capture' must have arguments}}
    @__parameter
    def move_capture_no_args_1():
        pass

    @__move_capture()  # expected-error {{'@__move_capture' must have arguments}}
    @__parameter
    def move_capture_no_args_2():
        pass

# ===----------------------------------------------------------------------=== #
# Struct decorators
# ===----------------------------------------------------------------------=== #

@invalid_dec  # expected-error {{use of unknown declaration 'invalid_dec'}}
struct BadStructDecorator(Movable where False): pass


struct DecoratorSameLine(Movable where False):
  # expected-error @+1 {{decorators must be on their own line; add a newline after the decorator}}
  @staticmethod def same_line_decorator(): pass


@fieldwise_init # expected-error {{'FieldwiseInitExample' has an explicitly declared fieldwise initializer}}
struct FieldwiseInitExample[T: Movable & Deinitable](Movable where False):
  var x: Int
  var y: Self.T

  # expected-note @below {{initializer declared here}}
  def __init__(out self, x: Int, y: Self.T):
    pass


# ===----------------------------------------------------------------------=== #
# Trait decorators
# ===----------------------------------------------------------------------=== #

@decorator  # expected-error {{unrecognized body decorators}}
trait NoDecorators:
    pass
