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
# Return
##===----------------------------------------------------------------------===##

def foo() raises:
# expected-error @+1 {{unexpected token in expression}}
  return pass

return 32 # expected-error {{'return' must be inside a function; move this into a function body}}

##===----------------------------------------------------------------------===##
# If / While
##===----------------------------------------------------------------------===##

def elif_parse_error(a: Bool) raises:
  if a:
    pass
 elif a: # expected-error {{unknown tokens at the end of a declaration}}
    pass
 else:
    pass

struct NotBoolConvertible(Movable where False):
  def __init__(out self, *, copy: Self):
    pass

def test_bool_context(a: NotBoolConvertible) raises:
  if a: # expected-error {{NotBoolConvertible' does not implement the '__bool__' method}}
     pass

def test_if_decorator(a: Bool):
  @not_good()
  if a: # expected-error {{'if' statement does not support decorators; remove the decorator}}
    pass

  comptime if 1:
    pass
  elif a:  # expected-error {{cannot use a dynamic value in 'comptime if' condition}}
    pass

def test_comptime_if_dynamic_elif(a: Bool):
  comptime if 1:
    pass
  elif a:  # expected-error {{cannot use a dynamic value in 'comptime if' condition}}
    pass

def test_decorator_with_comptime_if():
  @not_good()
  comptime if True:  # expected-error {{'comptime if' statement does not support decorators; remove the decorator}}
    pass

def test_comptime_elif_not_allowed(a: Bool):
  if a:
    pass
  comptime elif a:  # expected-error {{'comptime' cannot be used with 'elif'}}
    pass

##===----------------------------------------------------------------------===##
# For
##===----------------------------------------------------------------------===##

struct my_iter_no_next(Movable where False):
    def __init__(out self): pass


struct MyList_range_no_next(Movable where False):
    def __init__(out self): pass
    def __iter__(self) -> my_iter_no_next: return my_iter_no_next()


struct MyList_no_iter(Movable where False):
    def __init__(out self): pass

@fieldwise_init
struct MyFloat(Movable where False):
    pass

def test():
    var my_list_no_next = MyList_range_no_next()
    var my_list_no_iter = MyList_no_iter()


    # The failed 'Iterable' conformance makes this target an ordinary
    # assignment, so it declares and needs the 'var'. The two loops below
    # reach no such check: one has no '__iter__', the other a tuple target.
    # expected-error @+2 {{'my_iter_no_next' does not conform to 'Iterable'; add conformance to use in a 'for' loop}}
    # expected-note @+1 {{to conform to 'Iterable', add it to the struct declaration: 'struct Foo(Iterable):'}}
    for var item in my_list_no_next:
        pass

    # expected-error @+1 {{'MyList_no_iter' does not implement the '__iter__' method}}
    for var item in my_list_no_iter:
        pass

    # expected-error @+2 {{'my_iter_no_next' does not conform to 'Iterable'; add conformance to use in a 'for' loop}}
    # expected-note @+1 {{to conform to 'Iterable', add it to the struct declaration: 'struct Foo(Iterable):'}}
    for var key, item in my_list_no_next:
        pass

# Issue #18599
def spurious_for_loop_variable_unknown_decl():
  # expected-error @below {{'FloatLiteral[1]' does not implement the '__iter__' method}}
  for i in 1.0:
    # Note that the bug in issue #18599 is that after the above error, another error
    # will be spuriously raised about i not being bound.  So the real check in
    # this test is that no further error is raised.
    _ = i


struct ListValueInt(Movable where False):
    def __init__(out self): pass
    def __iter__(self) -> ListValueInt: return ListValueInt()
    def __next__(mut self) raises StopIteration -> Int: return 0

struct ListValueStringRef(Movable where False):
    def __init__(out self): pass
    def __iter__(self) -> ListValueStringRef: return ListValueStringRef()
    def __next__(mut self) raises StopIteration -> ref [self] String: pass


def loop_variable_scoped() raises:
  for i in ListValueInt():
     i = i   # expected-error {{expression must be mutable in assignment}}
  _ = i # expected-error {{use of unknown declaration 'i'}}

  for elt in ListValueStringRef():
    elt = "foo" # expected-error {{expression must be mutable in assignment}}

  for ref elt in ListValueStringRef():
    elt = "foo" # ok


##===----------------------------------------------------------------------===##
# With
##===----------------------------------------------------------------------===##

struct ExampleCM(Movable):
  def __enter__(self) -> Int:
    return 42
  def __exit__(self):
    pass # normal
  def __exit__(self, err: Error) -> Bool:
    return True # Raise

def withUsingImmutableVariable(var a: ExampleCM) raises:
  var x = 77
  with a^ as x:
    pass

# External Issue #529 https://github.com/modular/mojo/issues/529
def withWithNoColon(var a: ExampleCM) raises:
  # expected-error @below {{expected ':' or ',' after 'with' expression}}
  with a^ as b

def withNoRaise(var mgr: ExampleCM):
  with mgr^:
    # expected-error @below {{'raise' requires a surrounding 'try' block or the enclosing function to declare 'raises'}}
    raise Error()

  # Allow try-finally, but in a non-raising region.
  try:
    # expected-error @below {{'raise' requires a surrounding 'try' block or the enclosing function to declare 'raises'}}
    raise Error()
  finally:
    pass

# Poor error when with context managers that take ownership in enter
# https://github.com/modularml/modular/issues/23100
struct BadCM(Movable where False): # expected-note {{'BadCM' declared here}}
  def __init__(out self): pass

  def __enter__(var self) -> Int:
    return 42
  def __exit__(self):
    pass # normal
  def __exit__(self, err: Error) -> Bool:
    return True # Raise

def noop(a: Int): pass

def testBadCM():
  # expected-error @+1 {{context manager of type 'BadCM' defines a consuming __enter__ method as well as an __exit__ method; either remove 'var' from its '__enter__' method or remove the '__exit__' method}}
  with BadCM():
    pass


struct MyBool(TrivialRegisterPassable):
    var _mlir_value: __mlir_type.`!kgen.scalar<bool>`

    # expected-note @below {{function declared here}}
    def __mlir_bool__(self) -> __mlir_type.`!kgen.scalar<bool>`:
        return self._mlir_value

def noIndentError():
  for i in ListValueInt():
    # expected-error @+1 {{value passed to 'self' cannot be converted from type value 'MyBool' to an instance of 'MyBool'; did you mean to instantiate 'MyBool'?}}
    if MyBool: # no error 'statements must start at the beginning of a line' should be printed
      pass
