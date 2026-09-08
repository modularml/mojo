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
"""This is a module doc."""

# RUN: %parse-mojo-isolated %s -mojo-diagnose-missing-doc-strings -verify-diagnostics | FileCheck %s --implicit-check-not warning

from docs_package import documented_method_defined_in_init

# CHECK: #[[MODULE_DOC:.*]] = #lit.doc.string<"This is a module doc."
# CHECK: #[[ALIAS_DOC:.*]] = #lit.doc.string<"This is an alias doc."
# CHECK: #[[STRUCT_DOC:.*]] = #lit.doc.string<"This is a struct doc."
# CHECK: #[[STRUCT_FIELD_DOC:.*]] = #lit.doc.string<"This is a struct field doc."
# CHECK: #[[FUNCTION_DOC:.*]] = #lit.doc.string<"This is a function doc."
# CHECK: #[[TRAIT_DOC:.*]] = #lit.doc.string<"This is a trait doc."
# CHECK: #[[TRAIT_FUNCTION_DOC:.*]] = #lit.doc.string<"This is a trait function doc."
# CHECK: #[[PACKAGE_DOC:.*]] = #lit.doc.string<"This is a test package."
# CHECK: #[[IMPORTED_FUNC_DOC:.*]] = #lit.doc.string<"This is an imported method."

# CHECK: lit.file_module @parser_doc{{.*}}docString = #[[MODULE_DOC]]
# CHECK: lit.alias.decl {{.*}}AliasType{{.*}}docString = #[[ALIAS_DOC]]
# CHECK: lit.struct.decl @Struct{{.*}}docString = #[[STRUCT_DOC]]
# CHECK: lit.struct.field value{{.*}}docString = #[[STRUCT_FIELD_DOC]]
# CHECK: lit.fn @"foo()"{{.*}}docString = #[[FUNCTION_DOC]]

# CHECK: lit.package @docs_package{{.*}}docString = #[[PACKAGE_DOC]]
# CHECK: lit.fn @"documented_method_defined_in_init()"{{.*}}docString = #[[IMPORTED_FUNC_DOC]]

comptime AliasType = __mlir_type.`!kgen.non_struct_type`
"""This is an alias doc."""

# This is needed by the compiler to synthesize trivial bit.
struct Bool(Movable where False):
  """This is doc for Bool."""

    @implicit
    def __init__(out self, value: __mlir_type.i1):
      """ Bool init with mlir.i1.

      Args:
        value: Bit in mlir.i1.
      """
      pass

struct Struct(Movable where False):
  """This is a struct doc."""

    var value: __mlir_type.index
    """This is a struct field doc."""

struct Int(Movable where False):
  """A stub for the Int to allow decoupling from the builtins."""
  pass

struct Error(Movable where False):
  """A stub for the Int to allow decoupling from the builtins."""
  pass

def foo():
  """This is a function doc."""
  documented_method_defined_in_init()
  return

trait AnyType:
  """A stub for the AnyType trait to allow decoupling from the builtins."""
  pass

trait Trait:
  """This is a trait doc."""

  def f(self):
    """This is a trait function doc."""
    ...

# MOTO-869: A docstring ending with an example should pass validation.
@fieldwise_init
struct StructWithExamples(Movable where False):
    """A struct with a function in it."""

    def method_with_example(self, zot: Int) raises:
        """A function with an example.

        Args:
            zot: A number.

        Raises:
            An error, obviously.

        Example:

        ```mojo
        method_with_example()
        ```
        """
        pass


# MOTO-1120: Summaries ending with '!' or '?' should pass validation.
struct StructWithExclamationSummary(Movable where False):
    """This struct has an exciting summary!"""

    pass


struct StructWithQuestionSummary(Movable where False):
    """Does this struct have a valid summary?"""

    pass


# MOTO-1120: '!' and '?' should also be valid in section bodies and arg descriptions.
def fn_with_punctuated_sections(arg: Int) raises -> Int:
    """Tests non-period terminators in section bodies.

    Args:
        arg: Pass any integer here!

    Returns:
        Did you really need to ask?

    Raises:
        Basically never!
    """
    pass
