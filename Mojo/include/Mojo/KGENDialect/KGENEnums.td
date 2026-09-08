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

#ifndef KGEN_KGENDIALECT_KGENENUMS_TD
#define KGEN_KGENDIALECT_KGENENUMS_TD

include "Mojo/KGENDialect/KGENDialect.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// POCAttr
//===----------------------------------------------------------------------===//

/// Parameter Expression Opcodes: This defines the Parameter Operator Code enum
/// attribute.
let cppNamespace = "M::KGEN" in
def KGEN_POCAttr : I32EnumAttr<"POC", "Parameter Operator Code", [
  /// Fully associative variadic expressions.
  I32EnumAttrCase<"Add", 0, "add">,
  I32EnumAttrCase<"Mul", 1, "mul">,
  I32EnumAttrCase<"MulNoWrap", 2, "mul_no_wrap">,
  I32EnumAttrCase<"And", 3, "and">,
  I32EnumAttrCase<"Or",  4, "or">,
  I32EnumAttrCase<"Xor", 5, "xor">,
  I32EnumAttrCase<"Max", 6, "max">,
  I32EnumAttrCase<"Min", 7, "min">,

  // Binary expressions.
  I32EnumAttrCase<"Shl", 8, "shl">,
  I32EnumAttrCase<"Shr", 9, "shr">,
  I32EnumAttrCase<"Div", 10, "div">, // NOTE division by 0 is undefined behavior
  I32EnumAttrCase<"Mod", 11, "mod">,

  // Compares.
  I32EnumAttrCase<"EQ", 12, "eq">,
  I32EnumAttrCase<"LT", 13, "lt">,
  I32EnumAttrCase<"LE", 14, "le">,

  // Conditional expression.
  I32EnumAttrCase<"Cond", 16, "cond">,

  // Target and build comparisons & utilities.
  I32EnumAttrCase<"CurrentTarget", 17, "current_target">,
  I32EnumAttrCase<"TargetHasFeature", 18, "target_has_feature">,
  I32EnumAttrCase<"TargetGetField", 19, "target_get_field">,
  I32EnumAttrCase<"AcceleratorArch", 20, "accelerator_arch">,
  I32EnumAttrCase<"CrossCompilation", 21, "cross_compilation">,

  // Environment variables.
  I32EnumAttrCase<"GetEnv", 22, "get_env">,

  // Unary type expressions.
  I32EnumAttrCase<"GetSizeOf", 23, "get_sizeof">,
  I32EnumAttrCase<"GetAlignOf", 24, "get_alignof">,

  // Signature processing.
  I32EnumAttrCase<"Apply", 25, "apply">,
  I32EnumAttrCase<"ApplyResultSlot", 26, "apply_result_slot">,
  I32EnumAttrCase<"Rebind", 27, "rebind">,

  // Generics.
  I32EnumAttrCase<"PtrBitcast", 34, "ptr_bitcast">,
  I32EnumAttrCase<"LoadFromMem", 35, "load_from_mem">,

  // Given a list of !param_list<sometype> type, return the same list with each
  // element wrapped in a kgen pointer type: [i32, f32] -> [ptr<i32>, ptr<f32>].
  // The first operand is the list, the second operand is an integer addrspace.
  I32EnumAttrCase<"VariadicPtrMap", 36, "variadic_ptr_map">,
  // Given a list of !param_list<pointer<sometype>> type, return the same list
  // with each element stripped of kgen pointer type:
  //     [ptr<i32>, ptr<f32>] -> [i32, f32].
  // This takes one operand, the list of pointer types.
  I32EnumAttrCase<"VariadicPtrRemoveMap", 37, "variadic_ptrremove_map">,

  /// Convert an arbitrary MLIR attribute into a !kgen.string using
  /// mlir::debugString on the attribute.
  I32EnumAttrCase<"AttrToStr", 39, "attr_to_str">,
  /// Convert a length+pointer representation in interpreter memory into a
  /// !kgen.string, used to convert comp-time String (and related types) to
  /// StringLiteral.
  I32EnumAttrCase<"DataToStr", 40, "data_to_str">,
  /// Get the address of a string.
  I32EnumAttrCase<"StringAddress", 41, "string_address">,
  /// Concatenate two values of !kgen.string type.
  I32EnumAttrCase<"StrConcat", 42, "str_concat">,
  /// Get the arguments from a function signature as a kgen.variadic.
  I32EnumAttrCase<"FunctionGetArgTypes", 43, "function_get_arg_types">,

  /// Division and remainder operations from `mlir.index`.
  I32EnumAttrCase<"DivS", 44, "div_s">,
  I32EnumAttrCase<"DivU", 45, "div_u">,
  I32EnumAttrCase<"CeilDivS", 46, "ceil_div_s">,
  I32EnumAttrCase<"CeilDivU", 47, "ceil_div_u">,
  I32EnumAttrCase<"FloorDivS", 48, "floor_div_s">,
  I32EnumAttrCase<"RemS", 49, "rem_s">,
  I32EnumAttrCase<"RemU", 50, "rem_u">,

  /// HELLO INTREPID COMPILER DEVELOPER!  Are you thinking about adding another
  /// POC opcode for something that isn't a trivial expression like add or sub?
  /// The list above ^^^ is massively out of control, these things are not
  /// "operators" like ParamOperatorAttr was designed for.  MLIR allows us to
  /// define new attributes, and it's really good at it!  Let's stop this
  /// pattern and progressively refactor these away over time so POC can get
  /// back to being simple arithmetic stuff some day.  Thank you, -ChrisL
]>;

//===----------------------------------------------------------------------===//
// CmpPredicate
//===----------------------------------------------------------------------===//

def KGEN_CmpPredicate : I32EnumAttr<
    "CmpPredicate", "cmp comparison predicate", [
      I32EnumAttrCase<"EQ", 0, "eq">,
      I32EnumAttrCase<"NE", 1, "ne">,
      I32EnumAttrCase<"LT", 2, "lt">,
      I32EnumAttrCase<"GT", 3, "gt">,
      I32EnumAttrCase<"LE", 4, "le">,
      I32EnumAttrCase<"GE", 5, "ge">
    ]> {
  let cppNamespace = "::M::KGEN";
  let genSpecializedAttr = 0;
}

def KGEN_CmpPredicateAttr : EnumAttr<KGENDialect, KGEN_CmpPredicate, "cmp_pred">;

//===----------------------------------------------------------------------===//
// InlineLevelAttr
//===----------------------------------------------------------------------===//

/// This defines the Input Generation Kind enum attribute.
def KGEN_InlineLevelAttr : I32EnumAttr<"InlineLevel",
          "Whether and how to inline this function", [
  I32EnumAttrCase<"Automatic", 0, "automatic">,
  I32EnumAttrCase<"Always", 1, "always">,
  I32EnumAttrCase<"AlwaysNoDebug", 2, "always_nodebug">,
  I32EnumAttrCase<"AlwaysBuiltin", 3, "always_builtin">,
  I32EnumAttrCase<"Never", 4, "never">,
]> {
  let cppNamespace = "M::KGEN";
}

//===----------------------------------------------------------------------===//
// ExportKind
//===----------------------------------------------------------------------===//

def KGEN_ExportKind : I32EnumAttr<"ExportKind", "export kind", [
  I32EnumAttrCase<"NotExported", 0, "not_exported">,
  I32EnumAttrCase<"Exported", 1, "exported">
]> {
  let cppNamespace = "::M::KGEN";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// TailKind
//===----------------------------------------------------------------------===//

def KGEN_TailKind : I32EnumAttr<"TailKind", "tail kind", [
  I32EnumAttrCase<"None", 0, "none">,
  I32EnumAttrCase<"MustTail", 1, "musttail">,
  I32EnumAttrCase<"NoTail", 2, "notail">,
  I32EnumAttrCase<"Tail", 3, "tail">,
]> {
  let cppNamespace = "::M::KGEN";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// Conventions
//===----------------------------------------------------------------------===//

/// This describes the encoding of input argument calling conventions, stored
/// in the metadata of a FuncType.
///
/// That these enums have helper functions defined in KGENEnums.h for working
/// with various classes.
def KGEN_ArgConvention : I32EnumAttr<"ArgConvention",
    "Function argument convention", [
  /// This is an immutable borrow of a register-passable value in an SSA
  /// register.  The caller guarantees the value is live for the duration of the
  /// call.
  ///
  /// The MLIR type is any non-memory type that can be lowered to LLVM.
  I32EnumAttrCase<"ImmReg", 0, "imm">,

  /// This is an immutably borrowed memory-only value, which is passed
  /// indirectly.  The caller passes a reference to an initialized memory
  /// slot that may be read but not mutated by the callee.
  ///
  /// The MLIR type is a LIT::RefType which is lowered by LowerLIT to
  /// KGEN::PointerType.
  I32EnumAttrCase<"ImmMem", 1, "imm_mem">,

  /// This is an owned register-passable value an SSA register.  The callee is
  /// responsible for destroying the value before returning.  The Mojo parser
  /// does not use this argument convention but LowerArgConventions produces it.
  ///
  /// The MLIR type is any non-memory type that can be lowered to LLVM.
  I32EnumAttrCase<"OwnedReg", 2, "owned">,

  /// This is an owned 'var' memory value, which is passed indirectly. The
  /// caller initializes the memory slot and the callee destroys it, returning
  /// with it uninitialized.  The callee is responsible for destroying the value
  /// before returning.
  ///
  /// The MLIR type is a LIT::RefType which is lowered by LowerLIT to
  /// KGEN::PointerType.
  I32EnumAttrCase<"OwnedMem", 3, "owned_in_mem">,

  /// This is an 'deinit' memory value, which is passed indirectly. The
  /// caller initializes the memory slot and the callee destroys it (without
  /// running the dtor implicitly), returning with it uninitialized.
  ///
  /// The MLIR type is a LIT::RefType which is lowered by LowerLIT to
  /// KGEN::PointerType.
  I32EnumAttrCase<"DeinitMem", 4, "deinit_mem">,

  /// This is used for mut arguments like `def x(mut a: Int)`.
  ///
  /// The MLIR type is always a LIT::RefType or PointerType, and the underlying
  /// type can be anything.  The value is initialized before entry to the
  /// function and must be initialized when the function returns.
  I32EnumAttrCase<"Mut", 5, "mut">,

  /// This is used for ref arguments like:
  ///   `def x[origin: X](ref [origin, addrspace] T)`
  ///
  /// The MLIR type is always a LIT::RefType or PointerType, and the underlying
  /// type can be anything.  The value is initialized before entry to the
  /// function.
  I32EnumAttrCase<"Ref", 6, "ref">,

  /// This is used for `ref []` arguments that are guaranteed to always be
  /// mutable, and thus are known to be no-alias.
  I32EnumAttrCase<"MutRef", 7, "mutref">,

  /// This is used on the last argument of a function that returns a
  /// memory-only value, requiring it to be returned through memory. The value
  /// is uninitialized on entry to the function and defined on exit from the
  /// function if it returns.  The caller is responsible for destroying the
  /// value.
  ///
  /// The MLIR type is a RefType or PointerType.
  I32EnumAttrCase<"ByRefResult", 8, "byref_result">,

  /// This is used on the second-last argument of a function that may raise,
  /// returning a potentially initialized error value if the function throws.
  I32EnumAttrCase<"ByRefError", 9, "byref_error">,
]> {
  let cppNamespace = "M::KGEN";
}

/// This describes the encoding of the function effects in the metadata of a
/// FuncType.
def KGEN_FnEffectsImpl : I16BitEnumAttr<"FnEffects", "Function effects", [
  /// This function has no effects.
  I16BitEnumAttrCaseNone<"None", "none">,
  /// This function can throw.
  I16BitEnumAttrCaseBit<"Throws", 0, "throws">,
  /// This is an async function.
  I16BitEnumAttrCaseBit<"Async", 1, "async">,
  /// This function captures state.
  I16BitEnumAttrCaseBit<"Capturing", 2, "capturing">,
  /// This function has a auto-derefed result of !lit.ref or !kgen.pointer type.
  I16BitEnumAttrCaseBit<"RefResult", 5, "refresult">,
  /// This function uses the C ABI for argument and return value passing.
  /// Lowering must apply platform C ABI coercion (via CABIInfo) rather than
  /// the default Mojo calling convention.
  I16BitEnumAttrCaseBit<"CABI", 9, "cabi">,
]> {
  let cppNamespace = "M::KGEN::impl";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// ComputeKindAttr
//===----------------------------------------------------------------------===//

def ComputeKind : I32EnumAttr<"ComputeKind", "compute kind", [
  I32EnumAttrCase<"Addition",       0, "addition">,
  I32EnumAttrCase<"Comparison",     1, "comparison">,
  I32EnumAttrCase<"Division",       2, "division">,
  I32EnumAttrCase<"Multiplication", 3, "multiplication">,
  I32EnumAttrCase<"MultiplyAdd",    4, "multiply_add">,
  I32EnumAttrCase<"Other",          5, "other">,
]> {
  let cppNamespace = "M::KGEN";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// EmitAs
//===----------------------------------------------------------------------===//

def EmitAs : I32EnumAttr<"EmitAs", "compile_assembly emission kind", [
  I32EnumAttrCase<"ASM",  0, "asm">,
  I32EnumAttrCase<"LLVM", 1, "llvm">,
  I32EnumAttrCase<"LLVM_OPT", 2, "llvm-opt">,
  I32EnumAttrCase<"OBJECT", 3, "object">,
  I32EnumAttrCase<"LLVM_BITCODE", 4, "llvm-bitcode">,
  I32EnumAttrCase<"LLVM_OPT_BITCODE", 5, "llvm-opt-bitcode">,
]> {
  let cppNamespace = "M::KGEN";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// ClosureMethod
//===----------------------------------------------------------------------===//

def KGEN_ClosureMethod : I32EnumAttr<"ClosureMethod", "closure method", [
  I32EnumAttrCase<"CALL",  0, "call">,
  I32EnumAttrCase<"DEL", 1, "del">,
  I32EnumAttrCase<"MOVE", 2, "move">,
  I32EnumAttrCase<"COPY", 3, "copy">,
  I32EnumAttrCase<"NONE", 4, "none">,
]> {
  let cppNamespace = "M::KGEN";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// ClosureMemoryKind
//===----------------------------------------------------------------------===//

def KGEN_ClosureMemoryKind : I32EnumAttr<"ClosureMemoryKind", "closure memory kind", [
  I32EnumAttrCase<"ESCAPING",  0, "escaping">,
  I32EnumAttrCase<"NONESCAPING", 1, "nonescaping">,
  I32EnumAttrCase<"TRIVIAL", 2, "trivial">,
  I32EnumAttrCase<"REGISTER_PASSABLE", 3, "register_passable">,
]> {
  let cppNamespace = "M::KGEN";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// SugarKind
//===----------------------------------------------------------------------===//

def KGEN_SugarKind : I32EnumAttr<"SugarKind",
    "What kind of sugar a SugarAttr was formed from", [
  // --- Opaque sugar kinds (user do not expect to see through them) ---
  // Sugar kinds with values <= Preserved are opaque and should not produce
  // "aka" notes in diagnostics.

  /// This is an expansion of a call to an @always_inline("builtin") function.
  I32EnumAttrCase<"AlwaysInlineBuiltin", 0, "aibuiltin">,

  /// This is sugar created internally to preserve nested sugar during
  /// transformations.
  I32EnumAttrCase<"Preserved", 1, "preserved">,

  // --- Transparent sugar kinds (user expects to see through them) ---

  /// This is sugar for a member access of the form "T.x".  The "sugared" field
  /// contains the type being accessed and the memberName attr contains the
  /// field name.
  I32EnumAttrCase<"MemberAlias", 2, "member_alias">,

  /// This is an expansion of an alias reference "x" or "A.x".
  I32EnumAttrCase<"Alias", 3, "alias">,
]> {
  let cppNamespace = "M::KGEN";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// VariadicKind
//===----------------------------------------------------------------------===//

def KGEN_VariadicKind : I32EnumAttr<"VariadicKind",
    "Whether an argument is a homogenous variadic or pack", [
  /// Not a variadic argument, e.g. `x: Int`.
  I32EnumAttrCase<"None", 0, "not_vararg">,
  /// A homogeneously typed variadic argument, e.g. `*x` or `*x: Int`.
  I32EnumAttrCase<"PosVarArg", 1, "pos_vararg">,
  /// A heterogeneously typed variadic argument, e.g. `*x: *Ts`.
  I32EnumAttrCase<"PackVarArg", 2, "pack_vararg">,
  /// A variadic keywords argument, e.g. `**x: Int`.
  I32EnumAttrCase<"KwVarArg", 3, "kw_vararg">
]> {
  let cppNamespace = "M::KGEN";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// PassingKind
//===----------------------------------------------------------------------===//

/// Defines how an argument/parameter is allowed (by declaration) to be passed.
def KGEN_PassingKind : I32EnumAttr<"PassingKind",
    "How an argument or parameter is allowed to be passed", [
  I32EnumAttrCase<"PosOrKw", 0, "pos_or_kw">,
  I32EnumAttrCase<"PosOnly", 1, "pos">,
  I32EnumAttrCase<"KwOnly", 2, "kw">,
  // Things like byref results/errors and _Self in Traits.
  I32EnumAttrCase<"Implicit", 3, "implicit">,
  // User-declared inferred-only params like: def x[a: Bool, //, b: Int](...
  I32EnumAttrCase<"Inferred", 4, "inferred">,
]> {
  let cppNamespace = "M::KGEN";
  let genSpecializedAttr = 0;
}

#endif // KGEN_KGENDIALECT_KGENENUMS_TD
