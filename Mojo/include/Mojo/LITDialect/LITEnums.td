//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This defines enums in the LIT dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LITDIALECT_LITENUMS_TD
#define KGEN_LITDIALECT_LITENUMS_TD

include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// ImplicitConversionKind
//===----------------------------------------------------------------------===//

def LIT_ImplicitConversionKind : I32EnumAttr<
  "ImplicitConversionKind", "implicit conversion kind",
[
  I32EnumAttrCase<"None", 0, "none">,
  I32EnumAttrCase<"Implicit", 1, "implicit">,
  I32EnumAttrCase<"Deprecated", 2, "deprecated">
]> {
  let cppNamespace = "::M::KGEN::LIT";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// TypeConvention
//===----------------------------------------------------------------------===//

def LIT_TypeConvention : I32EnumAttr<"TypeConvention",
    "Whether a type is register passable or memory-only", [
  I32EnumAttrCase<"MemoryOnly", 0, "memory_only">,
  I32EnumAttrCase<"RegisterPassable", 1, "register_passable">,
  // TODO(#14086): Trivial should be a separate thing.
  I32EnumAttrCase<"RegisterPassableTrivial", 2, "register_passable_trivial">,
  I32EnumAttrCase<"Unspecified", 3, "unspecified">
]> {
  let cppNamespace = "M::KGEN::LIT";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// VarDeclKind
//===----------------------------------------------------------------------===//

def LIT_VarDeclKind : I32EnumAttr<"VarDeclKind",
    "origin of a VarDeclOp", [
  /// "var x = ...": Value is declared explicitly by the user to be mutable.
  I32EnumAttrCase<"Var", 0, "var">,
  /// "ref x = ...": Explicitly declared reference binding.
  I32EnumAttrCase<"Ref", 1, "ref">,
  /// "for x in ..": Bind to either a immutable value or a reference depending
  /// on the initializer. On assignment, this switches to Ref or Bound
  I32EnumAttrCase<"Bind", 2, "bind">,
  /// "for x in ..": Binding to a value.  This enables the parser to know that
  /// references to "i" in the body are MBValue instead of MLValue.
  I32EnumAttrCase<"Bound", 3, "bound">,

  /// Value is declared implicitly by the user via non-var/let syntax.
  /// For example, implicit declarations in `def` functions, target declarations
  /// in `for` statements.
  I32EnumAttrCase<"Implicit", 4, "imp">,
  /// Value is not declared by the user at all.
  I32EnumAttrCase<"Synthesized", 5, "synth">,
  /// Value is a shadow of a function argument, for arguments that need a
  /// variable location.
  I32EnumAttrCase<"Arg", 6, "arg">,
  /// Value is a shadow of an 'out' argument for a register passable
  /// initializer.
  I32EnumAttrCase<"InitOutArg", 7, "initoutarg">,
]> {
  let cppNamespace = "M::KGEN::LIT";
  let genSpecializedAttr = 0;
}

#endif // KGEN_LITDIALECT_LITENUMS_TD
