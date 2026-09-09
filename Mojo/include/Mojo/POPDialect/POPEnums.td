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

#ifndef KGEN_POPDIALECT_POPENUMS_TD
#define KGEN_POPDIALECT_POPENUMS_TD

include "Mojo/POPDialect/POPDialect.td"
include "mlir/IR/EnumAttr.td"

class POP_Enum<string name, string summary, list<I32EnumAttrCase> cases> :
  I32EnumAttr<name, summary, cases> {
  let cppNamespace = "::M::KGEN::POP";
  let genSpecializedAttr = 0;
}

//===----------------------------------------------------------------------===//
// FastmathFlags
//===----------------------------------------------------------------------===//


def FASTMATH_NONE            : I32BitEnumAttrCaseNone<"none"      >;
def FASTMATH_NO_NANS         : I32BitEnumAttrCaseBit<"nnan",     0>;
def FASTMATH_NO_INFS         : I32BitEnumAttrCaseBit<"ninf",     1>;
def FASTMATH_NO_SIGNED_ZEROS : I32BitEnumAttrCaseBit<"nsz",      2>;
def FASTMATH_ALLOW_RECIP     : I32BitEnumAttrCaseBit<"arcp",     3>;
def FASTMATH_ALLOW_CONTRACT  : I32BitEnumAttrCaseBit<"contract", 4>;
def FASTMATH_APPROX_FUNC     : I32BitEnumAttrCaseBit<"afn",      5>;
def FASTMATH_REASSOC         : I32BitEnumAttrCaseBit<"reassoc",  6>;
def FASTMATH_FAST            : I32BitEnumAttrCaseGroup<
    "fast",
    [
      FASTMATH_NO_NANS,
      FASTMATH_NO_INFS,
      FASTMATH_NO_SIGNED_ZEROS,
      FASTMATH_ALLOW_RECIP,
      FASTMATH_ALLOW_CONTRACT,
      FASTMATH_APPROX_FUNC,
      FASTMATH_REASSOC
    ]>;

def POP_FastmathFlags : I32BitEnumAttr<"FastmathFlags", "fastmath flags", [
      FASTMATH_NONE,
      FASTMATH_NO_NANS,
      FASTMATH_NO_INFS,
      FASTMATH_NO_SIGNED_ZEROS,
      FASTMATH_ALLOW_RECIP,
      FASTMATH_ALLOW_CONTRACT,
      FASTMATH_APPROX_FUNC,
      FASTMATH_REASSOC,
      FASTMATH_FAST
]> {
  let cppNamespace = "::M::KGEN::POP";
  let genSpecializedAttr = 0;
  let printBitEnumPrimaryGroups = 1;
}

def POP_FastmathFlagsAttr : EnumAttr<POPDialect, POP_FastmathFlags, "fmf">;

//===----------------------------------------------------------------------===//
// GlobalAlloc
//===----------------------------------------------------------------------===//

def POP_GlobalAllocAddressSpace: POP_Enum<
     "GlobalAllocAddressSpace", "pop.global_alloc address space", [
       I32EnumAttrCase<"IGNORE", 0, "ignore">,
       I32EnumAttrCase<"GPU_SHARED", 1, "gpu_shared">
     ]>;

def POP_GlobalAllocAddressSpaceAttr: EnumAttr<POPDialect,
                                               POP_GlobalAllocAddressSpace,
                                               "global_alloc_addr_space">;


//===----------------------------------------------------------------------===//
// AtomicOrdering
//===----------------------------------------------------------------------===//

def POP_AtomicOrdering : POP_Enum<
    "AtomicOrdering", "pop.atomic_cmp_xchg atomic ordering", [
      I32EnumAttrCase<"NOT_ATOMIC", 0, "not_atomic">,
      I32EnumAttrCase<"UNORDERED", 1, "unordered">,
      I32EnumAttrCase<"MONOTONIC", 2, "monotonic">,
      I32EnumAttrCase<"ACQUIRE", 3, "acquire">,
      I32EnumAttrCase<"RELEASE", 4, "release">,
      I32EnumAttrCase<"ACQUIRE_RELEASE", 5, "acq_rel">,
      I32EnumAttrCase<"SEQUENTIALLY_CONSISTENT", 6, "seq_cst">
    ]>;

def POP_AtomicOrderingAttr : EnumAttr<POPDialect,
                                      POP_AtomicOrdering,
                                      "atomic_ordering">;

//===----------------------------------------------------------------------===//
// AtomicBinOp
//===----------------------------------------------------------------------===//

def POP_AtomicBinOp : POP_Enum<
    "AtomicBinOp", "pop.atomic.rmw operator", [
      I32EnumAttrCase<"XCHG", 0, "xchg">,
      I32EnumAttrCase<"ADD", 1, "add">,
      I32EnumAttrCase<"SUB", 2, "sub">,
      I32EnumAttrCase<"AND", 3, "and">,
      I32EnumAttrCase<"NAND", 4, "nand">,
      I32EnumAttrCase<"OR", 5, "or">,
      I32EnumAttrCase<"XOR", 6, "xor">,
      I32EnumAttrCase<"MAX", 7, "max">,
      I32EnumAttrCase<"MIN", 8, "min">
    ]>;

def POP_AtomicBinOpAttr : EnumAttr<POPDialect,
                                   POP_AtomicBinOp,
                                   "bin_op">;

//===----------------------------------------------------------------------===//
// POP_IntLiteralBinKind
//===----------------------------------------------------------------------===//

/// This encodes which specific operation is to be performed.
def POP_IntLiteralBinKind : I32EnumAttr<"IntLiteralBinKind",
  "operation specification for IntLiteral binary operations", [
  I32EnumAttrCase<"Add", 0, "add">,
  I32EnumAttrCase<"Sub", 1, "sub">,
  I32EnumAttrCase<"Mul", 2, "mul">,
  I32EnumAttrCase<"FloorDiv", 3, "floordiv">,
  I32EnumAttrCase<"Mod", 4, "mod">,
  I32EnumAttrCase<"Lshift", 5, "lshift">,
  I32EnumAttrCase<"Rshift", 6, "rshift">,
  I32EnumAttrCase<"And", 7, "and">,
  I32EnumAttrCase<"Or", 8, "or">,
  I32EnumAttrCase<"Xor", 9, "xor">,
  I32EnumAttrCase<"Pow", 10, "pow">,
]> {
  let cppNamespace = "M::KGEN::POP";
  let genSpecializedAttr = 0;
}

def POP_IntLiteralBinKindAttr : EnumAttr<POPDialect,
                                         POP_IntLiteralBinKind,
                                         "int_literal.bin_kind"> {
  // Dotted mnemonic; keep bare `$value` so it embeds
  // cleanly in the IntLiteralBin/FloatLiteralBin wrappers.
  let assemblyFormat = "$value";
}

//===----------------------------------------------------------------------===//
// POP_IntLiteralCmpPred
//===----------------------------------------------------------------------===//

/// This encodes which specific comparison operation is to be performed
/// for `pop.int_literal.cmp` ops.
def POP_IntLiteralCmpPred : I32EnumAttr<"IntLiteralCmpPred",
  "operation specification for IntLiteral comparison", [
  I32EnumAttrCase<"Eq", 0, "eq">,
  I32EnumAttrCase<"Ne", 1, "ne">,
  I32EnumAttrCase<"Lt", 2, "lt">,
  I32EnumAttrCase<"Le", 3, "le">,
  I32EnumAttrCase<"Gt", 4, "gt">,
  I32EnumAttrCase<"Ge", 5, "ge">,
]> {
  let cppNamespace = "M::KGEN::POP";
  let genSpecializedAttr = 0;
}

def POP_IntLiteralCmpPredAttr : EnumAttr<POPDialect, POP_IntLiteralCmpPred,
                                         "int_literal.cmp_pred"> {
  // Dotted mnemonic; keep bare `$value` so it embeds
  // cleanly in the IntLiteralBin/FloatLiteralBin wrappers.
  let assemblyFormat = "$value";
}

//===----------------------------------------------------------------------===//
// POP_FloatLiteralSpecialValues
//===----------------------------------------------------------------------===//

/// This encodes special values for `#pop<float_literal` attributes.
def POP_FloatLiteralSpecialValues : I32EnumAttr<"FloatLiteralSpecialValues",
  "Special value kinds for float literals", [
  I32EnumAttrCase<"Normal", 0, "normal">,
  I32EnumAttrCase<"NegZero", 1, "neg_zero">,
  I32EnumAttrCase<"Inf", 2, "inf">,
  I32EnumAttrCase<"NegInf", 3, "neg_inf">,
  I32EnumAttrCase<"Nan", 4, "nan">,
]> {
  let cppNamespace = "M::KGEN::POP";
  let genSpecializedAttr = 0;
}

def POP_FloatLiteralSpecialValuesAttr :
  EnumAttr<POPDialect, POP_FloatLiteralSpecialValues,
           "float_literal.special_values"> {
  // Dotted mnemonic; keep bare `$value` so it embeds
  // cleanly in the IntLiteralBin/FloatLiteralBin wrappers.
  let assemblyFormat = "$value";
}

/// This encodes which specific operation is to be performed
/// for `pop.float_literal.bin` ops.
def POP_FloatLiteralBinKind : I32EnumAttr<"FloatLiteralBinKind",
  "operation specification for FloatLiteral binary operations", [
  I32EnumAttrCase<"Add", 0, "add">,
  I32EnumAttrCase<"Sub", 1, "sub">,
  I32EnumAttrCase<"Mul", 2, "mul">,
  I32EnumAttrCase<"TrueDiv", 3, "truediv">,
  I32EnumAttrCase<"FloorDiv", 4, "floordiv">
]> {
  let cppNamespace = "M::KGEN::POP";
  let genSpecializedAttr = 0;
}

def POP_FloatLiteralBinKindAttr : EnumAttr<POPDialect,
                                           POP_FloatLiteralBinKind,
                                           "float_literal.bin_kind"> {
  // Dotted mnemonic; keep bare `$value` so it embeds
  // cleanly in the IntLiteralBin/FloatLiteralBin wrappers.
  let assemblyFormat = "$value";
}

/// This encodes which specific comparison operation is to be performed
/// for `pop.float_literal.cmp` ops.
def POP_FloatLiteralCmpPred : I32EnumAttr<"FloatLiteralCmpPred",
  "operation specification for FloatLiteral comparison", [
  I32EnumAttrCase<"Eq", 0, "eq">,
  I32EnumAttrCase<"Ne", 1, "ne">,
  I32EnumAttrCase<"Lt", 2, "lt">,
  I32EnumAttrCase<"Le", 3, "le">,
  I32EnumAttrCase<"Gt", 4, "gt">,
  I32EnumAttrCase<"Ge", 5, "ge">,
]> {
  let cppNamespace = "M::KGEN::POP";
  let genSpecializedAttr = 0;
}

def POP_FloatLiteralCmpPredAttr : EnumAttr<POPDialect,
                                          POP_FloatLiteralCmpPred,
                                          "float_literal.cmp_pred"> {
  // Dotted mnemonic; keep bare `$value` so it embeds
  // cleanly in the IntLiteralBin/FloatLiteralBin wrappers.
  let assemblyFormat = "$value";
}

#endif // KGEN_POPDIALECT_POPENUMS_TD
