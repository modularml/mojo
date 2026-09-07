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
// This defines operations in the POP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_POPDIALECT_POPOPS_TD
#define KGEN_POPDIALECT_POPOPS_TD

include "Mojo/KGENDialect/KGENAttrs.td"
include "Mojo/KGENDialect/KGENInterfaces.td"
include "Mojo/KGENDialect/KGENTypes.td"
include "Mojo/POPDialect/POPAttrs.td"
include "Mojo/POPDialect/POPDialect.td"
include "Mojo/POPDialect/POPInterfaces.td"
include "Mojo/POPDialect/POPTypes.td"
include "Mojo/Interpreter/InterpreterInterface.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for the operation in this dialect.
class POPOp<string mnemonic, list<Trait> traits = []> :
    Op<POPDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// UnaryOp
//===----------------------------------------------------------------------===//

// Base class for unary arithmetic pop operations.
class POP_ArithmeticUnaryOp<string mnemonic, list<Trait> traits = []> :
    POPOp<mnemonic, traits # [Pure, SameOperandsAndResultType]> {
  let arguments = (ins UnboundOrIndexOrArithmeticSIMD:$operand);
  let results = (outs UnboundOrIndexOrArithmeticSIMD:$result);

  let assemblyFormat = "$operand attr-dict `:` qualified(type($result))";

  let hasFolder = 1;
}

// Base class for unary arithmetic or bool pop operations.
class POP_ArithmeticOrBoolUnaryOp<string mnemonic, list<Trait> traits = []> :
    POPOp<mnemonic, traits # [Pure, SameOperandsAndResultType]> {
  let arguments = (ins UnboundOrBoolOrIndexOrArithmeticSIMD:$operand);
  let results = (outs UnboundOrBoolOrIndexOrArithmeticSIMD:$result);

  let assemblyFormat = "$operand attr-dict `:` qualified(type($result))";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// NegOp
//===----------------------------------------------------------------------===//

def POP_NegOp : POP_ArithmeticUnaryOp<"neg"> {
  let summary = "negation";
  let description = [{
    The `pop.neg` operation takes a `!kgen.simd` value and returns its negated
    value. The operand and result types are the same. The operand type must be
    an index, signed integer, or float dtype.

    Example:

    ```mlir
    %b = pop.neg %a : !kgen.scalar<type>
    %b = pop.neg %a : !kgen.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// FloorOp
//===----------------------------------------------------------------------===//

def POP_FloorOp : POP_ArithmeticOrBoolUnaryOp<"floor"> {
  let summary = "floor";
  let description = [{
    The `pop.floor` operation takes a `!kgen.simd` value and returns the element
    wise floor of its values. The operand and result types are the same and
    must be index-like, arithmetic, or boolean dtypes. Integer, index, and
    boolean values are returned unchanged.

    Example:

    ```mlir
    %b = pop.floor %a : !kgen.simd<size, f32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// CeilOp
//===----------------------------------------------------------------------===//

def POP_CeilOp : POP_ArithmeticOrBoolUnaryOp<"ceil"> {
  let summary = "ceil";
  let description = [{
    The `pop.ceil` operation takes a `!kgen.simd` value and returns the element
    wise ceiling of its values. The operand and result types are the same and
    must be index-like, arithmetic, or boolean dtypes. Integer, index, and
    boolean values are returned unchanged.

    Example:

    ```mlir
    %b = pop.ceil %a : !kgen.simd<size, f32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// TruncOp
//===----------------------------------------------------------------------===//

def POP_TruncOp : POP_ArithmeticOrBoolUnaryOp<"trunc"> {
  let summary = "trunc";
  let description = [{
    The `pop.trunc` operation takes a `!kgen.simd` value and returns the element
    wise truncation of its values (rounding toward zero). The operand and result
    types are the same and must be index-like, arithmetic, or boolean dtypes.
    Integer, index, and boolean values are returned unchanged.

    Example:

    ```mlir
    %b = pop.trunc %a : !kgen.simd<size, f32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// RoundOp
//===----------------------------------------------------------------------===//

def POP_RoundOp : POP_ArithmeticOrBoolUnaryOp<"round", [NoMemoryEffect,
                  DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "round";
  let description = [{
    The `pop.round` operation takes a `!kgen.simd` value and returns the element
    wise rounding of its values to the nearest integer, with ties rounding
    toward the nearest even. The operand and result types are the same and must
    be index-like, arithmetic, or boolean dtypes. Integer, index, and boolean
    values are returned unchanged.

    Example:

    ```mlir
    %b = pop.round %a : !kgen.simd<size, f32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// AbsOp
//===----------------------------------------------------------------------===//

def POP_AbsOp : POP_ArithmeticOrBoolUnaryOp<"abs", [NoMemoryEffect,
                UnifiedFolding, InterpreterOpInterface]> {
  let hasFolder = 0;
  let summary = "abs";
  let description = [{
    The `pop.abs` operation takes a `!kgen.simd` value and returns the absolute
    value of it. The operand and result types are the same and must be
    index-like, arithmetic, or boolean dtypes. Unsigned integer, uindex, and
    boolean values are returned unchanged. For signed integers, the absolute
    value of INT_MIN is unchanged.

    Example:

    ```mlir
    %r = pop.abs %x : !kgen.simd<size, f32>
    %r = pop.abs %x : !kgen.simd<size, si32>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// BinaryOp
//===----------------------------------------------------------------------===//

// Base class for binary arithmetic pop operations.
class POP_ArithmeticBinaryOp<string mnemonic, list<Trait> traits = []> :
    POPOp<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins UnboundOrIndexOrArithmeticSIMD:$lhs,
                       UnboundOrIndexOrArithmeticSIMD:$rhs);
  let results = (outs UnboundOrIndexOrArithmeticSIMD:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($result))";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def POP_AddOp : POP_ArithmeticBinaryOp<"add", [Commutative, Pure,
                DeclareOpInterfaceMethods<ComputeOpInterface>,
                POP_FastmathFlagsInterface]>, Addition {
  let summary = "addition";
  let description = [{
    The `pop.add` operation takes two `!kgen.simd` values and returns their sum.
    The operand and result types must be the same. This operation is only valid
    on integer, index, and floating point dtypes.

    Example:

    ```mlir
    %c = pop.add %a, %b : !kgen.scalar<type>
    %c = pop.add %a, %b : !kgen.simd<size, type>
    ```
  }];
  // `contract` by default (like `pop.mul`) so `a + b*c` can fuse into an FMA;
  // the `-fp-mode` pass clears it for `contract=off`.
  let arguments = (ins UnboundOrIndexOrArithmeticSIMD:$lhs,
                       UnboundOrIndexOrArithmeticSIMD:$rhs,
                       DefaultValuedAttr<POP_FastmathFlagsAttr,
                         "FastmathFlags::contract">:$fastmathFlags);
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

def POP_SubOp : POP_ArithmeticBinaryOp<"sub", [Pure,
                DeclareOpInterfaceMethods<ComputeOpInterface>,
                POP_FastmathFlagsInterface]>, Addition {
  let summary = "subtraction";
  let description = [{
    The `pop.sub` operation takes two `!kgen.simd` values and returns their
    difference. The operand and result types must be the same. This operation is
    only valid on integer, index, and floating point dtypes.

    Example:

    ```mlir
    %c = pop.sub %a, %b : !kgen.scalar<type>
    %c = pop.sub %a, %b : !kgen.simd<size, type>
    ```
  }];
  // `contract` by default (like `pop.mul`) so `a*b - c` can fuse into an FMA;
  // the `-fp-mode` pass clears it for `contract=off`.
  let arguments = (ins UnboundOrIndexOrArithmeticSIMD:$lhs,
                       UnboundOrIndexOrArithmeticSIMD:$rhs,
                       DefaultValuedAttr<POP_FastmathFlagsAttr,
                         "FastmathFlags::contract">:$fastmathFlags);
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

def POP_MulOp : POP_ArithmeticBinaryOp<"mul", [Commutative, Pure,
                DeclareOpInterfaceMethods<ComputeOpInterface>,
                POP_FastmathFlagsInterface]>,
                Multiplication {
  let summary = "multiplication";
  let description = [{
    The `pop.mul` operation takes two `!kgen.simd` values and returns their
    product. The operand and result types must be the same. This operation is
    only valid on integer, index, and floating point dtypes.
    Request [fastmath policies](https://llvm.org/docs/LangRef.html#fast-math-flags)
    with the `fastmathFlags` attribute, which defaults to `contract`.
    Default to `contract` because fused multiply add is important for performance, and
    we want to write `a + b*c` instead of `a.fma(b, c)` everywhere.

    Example:

    ```mlir
    %c = pop.mul %a, %b : !kgen.scalar<type>
    %c = pop.mul %a, %b : !kgen.simd<size, type>
    %c = pop.mul %a, %b {fastmathFlags = #pop<fmf fast>} : !kgen.simd<size, type>
    ```
  }];
  let arguments = (ins UnboundOrIndexOrArithmeticSIMD:$lhs,
                       UnboundOrIndexOrArithmeticSIMD:$rhs,
                       // Use `contract` as the default value so that we can
                       // write a + b*c instead of a.fma(b, c).
                       DefaultValuedAttr<POP_FastmathFlagsAttr,
                         "FastmathFlags::contract">:$fastmathFlags);
  let results = (outs UnboundOrIndexOrArithmeticSIMD:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($result))";
}

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

def POP_DivOp : POP_ArithmeticBinaryOp<"div", [NoMemoryEffect,
                UnifiedFolding, InterpreterOpInterface,
                DeclareOpInterfaceMethods<ComputeOpInterface>]>, Division {
  let hasFolder = 0;
  let summary = "divide";
  let description = [{
    The `pop.div` operation takes two `!kgen.simd` values and divides them. The
    operand and result types must be the same. This operation is only valid on
    integer, index, and floating point dtypes. The signedness of integer
    division is determined by the dtype. Index is always treated as signed.
    Division by zero is undefined behaviour.

    Example:

    ```mlir
    %c = pop.div %a, %b : !kgen.scalar<type>
    %c = pop.div %a, %b : !kgen.simd<size, type>
    ```
  }];
  let arguments = (ins UnboundOrIndexOrArithmeticSIMD:$lhs,
                       UnboundOrIndexOrArithmeticSIMD:$rhs,
                       UnitAttr:$isExact);
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// RemOp
//===----------------------------------------------------------------------===//

def POP_RemOp : POP_ArithmeticBinaryOp<"rem", [NoMemoryEffect,
                DeclareOpInterfaceMethods<InterpreterOpInterface>,
                DeclareOpInterfaceMethods<ComputeOpInterface>]>, Division {
  let summary = "remainder";
  let description = [{
    The `pop.rem` operation takes two `!kgen.simd` values and returns their
    remainder. The operand and result types must be the same. This operation is
    only valid on integer, index, and floating point dtypes. The signedness of
    integer remainder is determined by the dtype. Index is always treated as
    signed. Division by zero is undefined behaviour.

    Example:

    ```mlir
    %c = pop.rem %a, %b : !kgen.scalar<type>
    %c = pop.rem %a, %b : !kgen.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// FloorDivOp
//===----------------------------------------------------------------------===//

def POP_FloorDivOp : POP_ArithmeticBinaryOp<"floordiv", [NoMemoryEffect,
                     UnifiedFolding, InterpreterOpInterface,
                     DeclareOpInterfaceMethods<ComputeOpInterface>]>, Division {
  let hasFolder = 0;
  let summary = "floor divide";
  let description = [{
    The `pop.floordiv` operation takes two `!kgen.simd` values and divides them,
    rounding towards negative infinity. The operand and result types must be
    the same. This operation is only valid on integer, index, and floating
    point dtypes. The signedness of integer division is determined by the
    dtype. Index is always treated as signed. Integer division by zero is
    undefined behaviour.

    Example:

    ```mlir
    %c = pop.floordiv %a, %b : !kgen.scalar<type>
    %c = pop.floordiv %a, %b : !kgen.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// MaxOp
//===----------------------------------------------------------------------===//

def POP_MaxOp : POPOp<"max", [Commutative, Pure, SameOperandsAndResultType]> {
  let summary = "max";
  let description = [{
    The `pop.max` operation takes two `!kgen.simd` values and returns their
    elementwise max. The operand and result types must be the same. This
    operation is only valid on integer, index, and floating point dtypes. The
    signedness of the integer maximum is determined by the dtype. Index is
    always treated as signed.

    Example:

    ```mlir
    %c = pop.max %a, %b : !kgen.scalar<type>
    %c = pop.max %a, %b : !kgen.simd<size, type>
    ```
  }];
  let arguments = (ins UnboundOrBoolOrIndexOrArithmeticSIMD:$lhs,
                       UnboundOrBoolOrIndexOrArithmeticSIMD:$rhs);
  let results = (outs UnboundOrBoolOrIndexOrArithmeticSIMD:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($result))";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// MinOp
//===----------------------------------------------------------------------===//

def POP_MinOp : POPOp<"min", [Commutative, Pure, SameOperandsAndResultType]> {
  let summary = "min";
  let description = [{
    The `pop.min` operation takes two `!kgen.simd` values and returns their
    elementwise min. The operand and result types must be the same. This
    operation is only valid on integer, index, and floating point dtypes. The
    signedness of the integer minimum is determined by the dtype. Index is
    always treated as signed.

    Example:

    ```mlir
    %c = pop.min %a, %b : !kgen.scalar<type>
    %c = pop.min %a, %b : !kgen.simd<size, type>
    ```
  }];
  let arguments = (ins UnboundOrBoolOrIndexOrArithmeticSIMD:$lhs,
                       UnboundOrBoolOrIndexOrArithmeticSIMD:$rhs);
  let results = (outs UnboundOrBoolOrIndexOrArithmeticSIMD:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($result))";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ShlOp
//===----------------------------------------------------------------------===//

def POP_ShlOp : POP_ArithmeticBinaryOp<"shl", [Pure,
                UnifiedFolding, InterpreterOpInterface]> {
  let hasFolder = 0;
  let summary = "shift left";
  let description = [{
    The `pop.shl` operation returns the first operand shifted to the left a
    specified number of bits. Both arguments to `pop.shl` must be the same
    integer or index dtype. The RHS value is always treated as unsigned.

    If the RHS value is equal to or larger than the number of bits in LHS, the
    results of this operation is a poison value.

    Example:

    ```mlir
    %0 = pop.shl %val, %shft: !kgen.scalar<type>
    %0 = pop.shl %val, %shft: !kgen.simd<size, type>
    ```
  }];
  let arguments = (ins UnboundOrIndexOrIntegerSIMD:$lhs,
                       UnboundOrIndexOrIntegerSIMD:$rhs);
}

//===----------------------------------------------------------------------===//
// ShrOp
//===----------------------------------------------------------------------===//

def POP_ShrOp : POP_ArithmeticBinaryOp<"shr", [Pure,
                UnifiedFolding, InterpreterOpInterface]> {
  let hasFolder = 0;
  let summary = "shift right";
  let description = [{
    The `pop.shr` operation returns the first operand shifted to the right a
    specified number of bits. Whether the result is sign extended or zero
    extended depends on the result dtype. Both arguments to the `shr` op must be
    the same integer/index or vector of integer/index type. `rhs` is always
    treated as unsigned.

    This operation performs an arithmetic shift right operation if the dtype
    is signed. The most significant bits of the result will be filled with the
    sign bit of `lhs`. This operation performs a logical shift right operation
    if the dtype is unsigned. The most significant bits of the result will be
    filled with zero bits after the shift.

    If `rhs` is equal to or larger than the number of bits in `lhs`, the result
    of this operation is a poison value. If the arguments are vectors, each
    vector element of `lhs` is shifted by the corresponding shift amount in
    `rhs`.

    ```mlir
    %0 = pop.shr %val, %shft: !kgen.scalar<type>
    %0 = pop.shr %val, %shft: !kgen.simd<size, type>
    ```
  }];
  let arguments = (ins UnboundOrIndexOrIntegerSIMD:$lhs,
                       UnboundOrIndexOrIntegerSIMD:$rhs);
}

//===----------------------------------------------------------------------===//
// TernaryOp
//===----------------------------------------------------------------------===//

// Base class for ternary arithmetic pop operations.
class POP_ArithmeticTernaryOp<string mnemonic, list<Trait> traits = []> :
    POPOp<mnemonic, traits # [Pure, SameOperandsAndResultType,
                              POP_FastmathFlagsInterface]> {
  let arguments = (ins UnboundOrIndexOrArithmeticSIMD:$a,
                       UnboundOrIndexOrArithmeticSIMD:$b,
                       UnboundOrIndexOrArithmeticSIMD:$c,
                       DefaultValuedAttr<POP_FastmathFlagsAttr,
                         "FastmathFlags::none">:$fastmathFlags);
  let results = (outs UnboundOrIndexOrArithmeticSIMD:$result);

  let assemblyFormat = [{
    $a `,` $b `,` $c attr-dict `:` qualified(type($result))
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// FMAOp
//===----------------------------------------------------------------------===//

def POP_FMAOp : POP_ArithmeticTernaryOp<"fma",
                [DeclareOpInterfaceMethods<ComputeOpInterface>]>, MultiplyAdd {
  let summary = "fused-multiply add";
  let description = [{
    The `pop.fma` operation takes three `!kgen.simd` values (`a`, `b`, and `c`)
    and computes the fma operation `a * b + c`. The operand and result types
    must be the same. This operation is only valid on integer, index, and
    floating point dtypes.

    Example:

    ```mlir
    %d = pop.fma %a, %b, %c : !kgen.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

def POP_CmpOp : POPOp<"cmp", [Pure, SameTypeOperands, InferTypeOpAdaptor,
    UnifiedFolding, InterpreterOpInterface,
    DeclareOpInterfaceMethods<ComputeOpInterface>]>, Comparison {
  let hasFolder = 0;
  let summary = "comparison";
  let description = [{
    The `pop.cmp` operation takes a comparison predicate and two `!kgen.scalar`
    or `!kgen.simd` types (`lhs` and `rhs`) and computes the resulting comparison
    elementwise.

    The comparison semantics between floating point dtypes is the same as in C.

    Example:

    ```mlir
    %b0 = pop.cmp lt(%a, %b) : !kgen.scalar<type>
    %b1 = pop.cmp ge(%c, %d) : !kgen.simd<?, type>
    ```
  }];

  let arguments = (ins KGEN_CmpPredicateAttr:$pred,
                       UnboundOrBoolOrIndexOrArithmeticSIMD:$lhs,
                       UnboundOrBoolOrIndexOrArithmeticSIMD:$rhs);
  let results = (outs BoolSIMD:$result);

  let assemblyFormat = [{
    enum($pred) `(` $lhs `,` $rhs `)` attr-dict `:` type($lhs)
  }];
}

//===----------------------------------------------------------------------===//
// BitwiseOp
//===----------------------------------------------------------------------===//

// Base class for bitwise arithmetic pop operations.
class POP_BitwiseOp<string mnemonic, list<Trait> traits = []> :
      POPOp<mnemonic, traits # [
        Pure,
        Commutative,
        SameOperandsAndResultType
      ]> {
  // (TODO) MOCO-2374 Relax type restrictions here so that we don't
  // fail verification before types are fully concretized before
  // elaboration. Move the type verification during pop to llvm lowering.
  let arguments = (ins UnboundOrBoolOrIndexOrArithmeticSIMD:$lhs,
                       UnboundOrBoolOrIndexOrArithmeticSIMD:$rhs);
  let results = (outs UnboundOrBoolOrIndexOrArithmeticSIMD:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs)";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// SIMDAndOp
//===----------------------------------------------------------------------===//

def POP_SIMDAndOp : POP_BitwiseOp<"simd.and", [Idempotent]> {
  let summary = "bitwise logical and";
  let description = [{
    The `pop.simd.and` returns the bitwise logical and of its two operands. The dtype
    of the operands must be a boolean, integral, or parametric type with
    underlying boolean or integral dtype.

    Example:

    ```mlir
    %c = pop.simd.and %a, %b : !kgen.scalar<type>
    %c = pop.simd.and %a, %b : !kgen.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// SIMDOrOp
//===----------------------------------------------------------------------===//

def POP_SIMDOrOp : POP_BitwiseOp<"simd.or", [Idempotent]> {
  let summary = "bitwise logical or";
  let description = [{
    The `pop.simd.or` returns the bitwise logical or of its two operands. The dtype
    of the operands must be a boolean, integral, or parametric type with
    underlying boolean or integral dtype.

    Example:

    ```mlir
    %c = pop.simd.or %a, %b : !kgen.scalar<type>
    %c = pop.simd.or %a, %b : !kgen.simd<size, type>
    ```
  }];
}

//===----------------------------------------------------------------------===//
// SIMDXOrOp
//===----------------------------------------------------------------------===//

def POP_SIMDXOrOp : POP_BitwiseOp<"simd.xor"> {
  let summary = "bitwise logical xor";
  let description = [{
    The `pop.simd.xor` returns the bitwise logical exclusive or of its two operands.
    The dtype of the operands must be a boolean, integral, or parametric type
    with underlying boolean or integral dtype.

    Example:

    ```mlir
    %c = pop.simd.xor %a, %b : !kgen.scalar<type>
    %c = pop.simd.xor %a, %b : !kgen.simd<size, type>
    ```

    The XOr is used to implement the `not` operation. For example:

    ```mlir
    %neg_one_si64 = kgen.param.constant: scalar<si64> = <<-1>>
    %neg_one = pop.cast %neg_one_si64 : !kgen.scalar<si64> to !kgen.scalar<type>
    %not = pop.simd.xor %val, %neg_one : !kgen.scalar<type> // returns ~(%val)
    ```
  }];
}

//===----------------------------------------------------------------------===//
// BitCastOp
//===----------------------------------------------------------------------===//

def POP_BitcastOp : POPOp<"bitcast", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "scalar and simd bitcast";
  let description = [{
    Bitcast an integer or floating point value to an integer or floating point
    value of equal bit width. The bitcast operation should not be used to cast
    between pointer types. For conversion between pointer types one should use
    `pop.pointer.bitcast`.

    Example:

    ```mlir
    %0 = pop.bitcast %a : !kgen.scalar<si32> to !kgen.scalar<f32>
    %1 = pop.bitcast %b : !kgen.scalar<ui64> to !kgen.scalar<si64>
    %2 = pop.bitcast %c : !kgen.simd<4, f64> to !kgen.simd<4, si64>
    %3 = pop.bitcast %d : !kgen.simd<2, f64> to !kgen.simd<4, f32>
    ```
  }];

  let arguments = (ins UnboundOrBoolOrArithmeticSIMD:$input);
  let results = (outs UnboundOrBoolOrArithmeticSIMD:$output);
  let assemblyFormat = [{
    $input attr-dict `:` qualified(type($input)) `to` qualified(type($output))
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// PointerBitcastOp
//===----------------------------------------------------------------------===//

def POP_PointerBitcastOp : POPOp<"pointer.bitcast", [Pure,
    DeclareOpInterfaceMethods<CastOpInterface>,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "pointer bitcast";
  let description = [{
    The `pop.pointer.bitcast` operation reinterprets a pointer or function
    pointer as a pointer to a different element or function type.

    Example:

    ```mlir
    // Bitcast an opaque pointer to an f32 pointer.
    %0 = pop.pointer.bitcast %arg0 : !kgen.pointer<scalar<invalid>> to
                                     !kgen.pointer<scalar<f32>>

    // Bitcast a scalar pointer to a SIMD pointer.
    %1 = pop.pointer.bitcast %0 : !kgen.pointer<scalar<f32>> to
                                  !kgen.pointer<simd<4, f32>>

    // Bitcast an opaque pointer to a function pointer.
    %2 = pop.pointer.bitcast %arg0 : !kgen.pointer<scalar<invalid>> to
                                     (i32, i32) -> (f32, f32)
    ```
  }];

  let arguments = (ins AnyTypeOf<[KGEN_ParamType, PointerType,
                                  KGEN_FuncTypeGeneratorType]>:$input);
  let results = (outs AnyTypeOf<[KGEN_ParamType, PointerType,
                                 KGEN_FuncTypeGeneratorType]>:$output);
  let assemblyFormat = [{
    $input attr-dict `:` qualified(type($input)) `to` qualified(type($output))
  }];

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

def POP_CastOp : POPOp<"cast", [Pure,
    DeclareOpInterfaceMethods<CastOpInterface>,
    DeclareOpInterfaceMethods<InterpreterOpInterface>,
    POP_FastmathFlagsInterface]> {
  let summary = "element wise cast";
  let description = [{
    Perform an element-wise cast between SIMD vectors types or between scalar
    types. SIMD vectors must have the same size. The cast semantics are the same
    as C casts. Only bool, integer, and floating point dtypes are supported.

    The `fast` fastmath flag indicates that lowering may generate an efficient
    sequence of instructions/assembly, but result may differ from using LLVM's
    conversion instructions. It also permits collapsing a chain of casts
    `A -> B -> C` into a direct `A -> C`, which may round differently.

    Example:

    ```mlir
    %0 = pop.cast %a : !kgen.scalar<si32> to !kgen.scalar<f32>
    %1 = pop.cast %b : !kgen.scalar<ui64> to !kgen.scalar<si32>
    %2 = pop.cast %c : !kgen.simd<4, f64> to !kgen.simd<4, ui32>
    %3 = pop.cast %c {fastmathFlags = #pop<fmf fast>} :
        !kgen.scalar<f32> to !kgen.scalar<bf16>
    ```
  }];

  let arguments = (ins
    UnboundOrBoolOrIndexOrArithmeticSIMD:$input,
    DefaultValuedAttr<POP_FastmathFlagsAttr,
      "FastmathFlags::none">:$fastmathFlags);
  let results = (outs UnboundOrBoolOrIndexOrArithmeticSIMD:$output);
  let assemblyFormat = [{
    $input attr-dict `:`
    qualified(type($input)) `to`  qualified(type($output))
  }];
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// SIMDExtractElementOp
//===----------------------------------------------------------------------===//

def POP_SIMDExtractElementOp : POPOp<"simd.extractelement", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>,
    SIMDSizeOneOfSameDType<"vector", "result">]> {
  let summary = "extractelement operation";
  let description = [{
    Takes a SIMD vector and an index position and extracts the `!kgen.scalar` at
    that position. The position must be of type index.

    If the position is out-of-bounds, the result is a poison value.

    Example:

    ```mlir
    %0 = pop.simd.extractelement %0[%position]: !kgen.simd<16, f32>
    ```
  }];

  let arguments = (ins KGEN_SIMDType:$vector, Index:$position);
  let results = (outs ScalarType:$result);

  let assemblyFormat = [{
    $vector `[` $position `]` attr-dict `:` qualified(type($vector))
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// SIMDInsertElementOp
//===----------------------------------------------------------------------===//

def POP_SIMDInsertElementOp : POPOp<"simd.insertelement", [Pure,
    SIMDSizeOneOfSameDType<"vector", "value">,
    DeclareOpInterfaceMethods<InterpreterOpInterface>,
    AllTypesMatch<["result", "vector"]>]> {
  let summary = "extractelement operation";
  let description = [{
    Takes a source value, SIMD vector, and an index position and inserts the
    source value into the vector at the proper position. The position must be
    of type index.

    If the position is out-of-bounds, the result is a poison value.

    Example:

    ```mlir
    %0 = pop.simd.extractelement %val, %vec[%position]: !kgen.simd<16, f32>
    ```
  }];

  let arguments = (ins KGEN_SIMDType:$vector, ScalarType:$value,
                       Index:$position);
  let results = (outs KGEN_SIMDType:$result);

  let assemblyFormat = [{
    $value `,` $vector `[` $position `]` attr-dict `:` qualified(type($vector))
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// SIMDShuffle
//===----------------------------------------------------------------------===//

def POP_SIMDShuffleOp : POPOp<"simd.shuffle", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>,
    AllTypesMatch<["lhs", "rhs"]>]> {
  let summary = "shufflevector operation";
  let description = [{
    The `pop.simd.shuffle` operation constructs a permutation of two vectors of
    the same length and returns a vector of the same element type with the same
    length as the provided mask, and thus must be a non-parametric constant.

    Example:

    ```mlir
    %0 = pop.simd.shuffle <2, f32> %lhs, %rhs -> <4, f32> [0, 1, 2, 3]
    ```
  }];

  let arguments = (ins KGEN_SIMDType:$lhs, KGEN_SIMDType:$rhs,
                       AttrOfType<POP_ArrayType>:$mask);
  let results = (outs KGEN_SIMDType:$result);

  let assemblyFormat = [{
    type($lhs) $lhs `,` $rhs `->` type($result)
    custom<ShuffleMask>($mask, ref(type($result))) attr-dict
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// SIMDSelectOp
//===----------------------------------------------------------------------===//

def POP_SIMDSelectOp : POPOp<"simd.select", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>,
    AllTypesMatch<["trueValue", "falseValue", "result"]>,
    TypesMatchWith<"bool of same branch type", "result", "condition",
                   "getBoolOfSameParentType($_self)">]> {
  let summary = "SIMD select";
  let description = [{
    The `pop.simd.select` operation takes a SIMD condition predicate and two
    `!kgen.scalar` or `!kgen.simd` types (`then` and `else`) and chooses one value
    based on a condition, without IR-level branching.

    When the condition is a SIMD vector, then the result is a SIMD vector
    where the value of each element is chosen (aka blended) from the `trueVal`
    and `falseVal` operands based on the condition's element value.

    Example:

    ```mlir
    %a = pop.simd.select %cond, %trueVal, %falseVal : !kgen.scalar<type>
    %b = pop.simd.select %cond, %trueVal, %falseVal : !kgen.simd<size, type>
    ```
  }];

  let arguments = (ins BoolSIMD:$condition, KGEN_SIMDType:$trueValue,
                       KGEN_SIMDType:$falseValue);
  let results = (outs KGEN_SIMDType:$result);

  let assemblyFormat = "operands attr-dict `:` type($result)";
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// SIMDSplatOp
//===----------------------------------------------------------------------===//

def POP_SIMDSplatOp : POPOp<"simd.splat", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>,
    SIMDSizeOneOfSameDType<"result", "scalar">]> {
  let summary = "SIMD splat";
  let description = [{
    The `pop.simd.splat` operation takes a scalar value and replicates it across
    a SIMD vector and returns that value. For example, splatting `%a` into a
    `vector<4xi32>` will produce a vector valued as `[%a, %a, %a, %a]`.

    ```mlir
    %v = pop.simd.splat %arg0 : !kgen.simd<4, f32>

    %u = pop.neg %arg1 : !kgen.simd<1, type>
    %w = pop.simd.splat %u : !kgen.simd<size, type>
    ```
  }];

  let arguments = (ins ScalarType:$scalar);
  let results = (outs KGEN_SIMDType:$result);

  let assemblyFormat = "$scalar attr-dict `:` qualified(type($result))";
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

def POP_LoadOp : POPOp<"load", [
    PointerElementType<"ptr", "result">,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "pointer load operation";
  let description = [{
    Loads the data at the given pointer. Returns a scalar of the element type.
    An optional `alignment` attribute can be provided to specify the alignment of
    the load. An optional `volatile` attribute can be provided to make the load
    volatile. An optional `invariant` attribute can be provided to make the the
    memory location referenced by the load contain the same value at all points
    in the program where the memory location is dereferenceable.

    When using atomic ordering (`atomic` keyword), valid orderings are:
    `unordered`, `monotonic`, `acquire`, and `sequentially_consistent`.
    `release` and `acquire_release` orderings are invalid for load operations.
    `volatile` or `invariant` loads cannot be atomic. An optional `syncscope`
    can be provided for atomic loads.

    Example:

    ```mlir
    %0 = pop.load %ptr0 : !kgen.pointer<scalar<f32>>
    %1 = pop.load %ptr1 : !kgen.pointer<simd<4, type>>
    %2 = pop.load %ptr0 align<8> : !kgen.pointer<scalar<f32>>
    %3 = pop.load volatile<1> %ptr0 align<8> : !kgen.pointer<scalar<f32>>
    %4 = pop.load volatile<1> invariant<1> %ptr0 align<8> : !kgen.pointer<scalar<f32>>
    %5 = pop.load atomic acquire %ptr0 : !kgen.pointer<scalar<f32>>
    %6 = pop.load atomic syncscope("agent") acquire %ptr0 : !kgen.pointer<scalar<f32>>
    ```
  }];

  let arguments = (ins PointerType:$ptr,
                       OptionalAttr<AttrOfType<Index>>:$alignment,
                       AttrOfType<I1>:$isVolatile,
                       AttrOfType<I1>:$isInvariant,
                       AttrOfType<I1>:$isNonTemporal,
                       DefaultValuedAttr<
                        POP_AtomicOrderingAttr, "AtomicOrdering::NOT_ATOMIC">:$ordering,
                       OptionalAttr<AttrOfType<KGEN_StringType>>:$syncscope);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    custom<I1Flag>($isVolatile, "\"volatile\"")
    custom<I1Flag>($isInvariant, "\"invariant\"")
    custom<I1Flag>($isNonTemporal, "\"nontemporal\"")
    (`atomic` (`syncscope` `(` custom<StringParam>($syncscope)^ `)`)? enum($ordering)^)?
    $ptr
    (`align` `<` custom<IndexParamValue>($alignment)^ `>`)? attr-dict `:`
    qualified(type($ptr))
  }];

  let builders = [
    OpBuilder<(ins "Value":$ptr,
                   CArg<"std::optional<unsigned>", "{}">:$alignment,
                   CArg<"bool", "false">:$isVolatile,
                   CArg<"bool", "false">:$isInvariant,
                   CArg<"bool", "false">:$isNonTemporal,
                   CArg<"AtomicOrdering", "AtomicOrdering::NOT_ATOMIC">:$ordering,
                   CArg<"std::optional<StringAttr>", "{}">:$syncscope)>,
  ];

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Return true/false if the volatile attribute is concrete, or nullopt if
    /// still parametric.
    std::optional<bool> isVolatile() {
      if (auto boolAttr = llvm::dyn_cast<BoolAttr>(getIsVolatileAttr()))
        return boolAttr.getValue();
      return std::nullopt;
    }
    /// Return true if this is known volatile or is still parametric.
    bool mightBeVolatile() {
      if (auto boolAttr = llvm::dyn_cast<BoolAttr>(getIsVolatileAttr()))
        return boolAttr.getValue();
      return true;
    }
  }];
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

def POP_StoreOp : POPOp<"store", [PointerElementType<"ptr", "arg">,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "pointer store operation";
  let description = [{
    Store the given value to the given pointer of the element type. An
    optional `alignment` attribute can be provided to specify the alignment of
    the store. An optional `volatile` attribute can be provided to make the
    store volatile.

    When using atomic ordering (`atomic` keyword), valid orderings are:
    `unordered`, `monotonic`, `release`, and `sequentially_consistent`.
    `acquire` and `acquire_release` orderings are invalid for store operations.
    `volatile` stores cannot be atomic. An optional `syncscope` can be
    provided for atomic stores.

    Example:

    ```mlir
    pop.store %0, %ptr0 : !kgen.pointer<scalar<f32>>
    pop.store %1, %ptr1 : !kgen.pointer<simd<4, type>>
    pop.store %1, %ptr1 align 8 : !kgen.pointer<simd<4, type>>
    pop.store volatile<1> %1, %ptr1 align 8: !kgen.pointer<simd<4, type>>
    pop.store atomic release %1, %ptr1 : !kgen.pointer<simd<4, type>>
    pop.store atomic syncscope("agent") release %1, %ptr1 : !kgen.pointer<simd<4, type>>
    ```
  }];

  let arguments = (ins AnyType:$arg,
                       PointerType:$ptr,
                       OptionalAttr<AttrOfType<Index>>:$alignment,
                       AttrOfType<I1>:$isVolatile,
                       AttrOfType<I1>:$isNonTemporal,
                       DefaultValuedAttr<
                        POP_AtomicOrderingAttr, "AtomicOrdering::NOT_ATOMIC">:$ordering,
                       OptionalAttr<AttrOfType<KGEN_StringType>>:$syncscope);
  let assemblyFormat = [{
    custom<I1Flag>($isVolatile, "\"volatile\"")
    custom<I1Flag>($isNonTemporal, "\"nontemporal\"")
    (`atomic` (`syncscope` `(` custom<StringParam>($syncscope)^ `)`)? enum($ordering)^)?
    $arg `,` $ptr
    (`align` `<` custom<IndexParamValue>($alignment)^ `>`)?
    attr-dict `:` qualified(type($ptr))
  }];

  let builders = [
    OpBuilder<(ins "Value":$arg, "Value":$ptr,
                   CArg<"std::optional<unsigned>", "{}">:$alignment,
                   CArg<"bool", "false">:$isVolatile,
                   CArg<"bool", "false">:$isNonTemporal,
                   CArg<"AtomicOrdering", "AtomicOrdering::NOT_ATOMIC">:$ordering,
                   CArg<"std::optional<StringAttr>", "{}">:$syncscope)>
  ];

  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Return true/false if the volatile attribute is concrete, or nullopt if
    /// still parametric.
    std::optional<bool> isVolatile() {
      if (auto boolAttr = llvm::dyn_cast<BoolAttr>(getIsVolatileAttr()))
        return boolAttr.getValue();
      return std::nullopt;
    }
    /// Return true if this is known volatile or is still parametric.
    bool mightBeVolatile() {
      if (auto boolAttr = llvm::dyn_cast<BoolAttr>(getIsVolatileAttr()))
        return boolAttr.getValue();
      return true;
    }
  }];
}

//===----------------------------------------------------------------------===//
// MemcpyOp
//===----------------------------------------------------------------------===//

def POP_MemcpyOp : POPOp<"memcpy", [
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "memcpy operation";
  let description = [{
    Performs an in-memory copy operation between the source pointer and
    destination pointer. The length is given in number of bytes to copy. The
    two pointers may have differing address space qualifiers.

    Example:

    ```mlir
    pop.memcpy %dst, %src, 4 : !kgen.pointer<scalar<f32>> -> !kgen.pointer<scalar<f32>>
    ```
  }];

  let arguments = (ins Arg<PointerType, "", [MemWrite]>:$dst,
                       Arg<PointerType, "", [MemRead]>:$src,
                       Index:$len,
                       AttrOfType<I1>:$isVolatile);
  let assemblyFormat = [{
    custom<I1Flag>($isVolatile, "\"volatile\"")
    $dst `,` $src `,` $len
    attr-dict `:` qualified(type($src)) `->` qualified(type($dst))
  }];

  let builders = [
    OpBuilder<(ins "Value":$dst, "Value":$src, "Value":$len,
                   CArg<"bool", "false">:$isVolatile)>,
  ];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Return true/false if the volatile attribute is concrete, or nullopt if
    /// still parametric.
    std::optional<bool> isVolatile() {
      if (auto boolAttr = llvm::dyn_cast<BoolAttr>(getIsVolatileAttr()))
        return boolAttr.getValue();
      return std::nullopt;
    }
    /// Return true if this is known volatile or is still parametric.
    bool mightBeVolatile() {
      if (auto boolAttr = llvm::dyn_cast<BoolAttr>(getIsVolatileAttr()))
        return boolAttr.getValue();
      return true;
    }
  }];
}

//===----------------------------------------------------------------------===//
// OffsetOp
//===----------------------------------------------------------------------===//

def POP_OffsetOp : POPOp<"offset", [Pure, AllTypesMatch<["result", "ptr"]>,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "pointer offset operation";
  let description = [{
    The `pop.offset` operation takes a pointer and increments or decrements its
    position by a certain amount. The pointer cannot have a `?` element type.

    Example:

    ```mlir
    %addr0 = pop.offset %ptr0[%idx] : !kgen.pointer<scalar<f32>>
    %addr1 = pop.offset %ptr1[%idx] : !kgen.pointer<simd<4, type>>
    ```
  }];

  let arguments = (ins PointerType:$ptr, Index:$index);
  let results = (outs PointerType:$result);
  let assemblyFormat = [{
    $ptr `[` $index `]` attr-dict `:` qualified(type($ptr))
  }];

  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

def POP_SelectOp : POPOp<"select", [Pure,
    AllTypesMatch<["trueValue", "falseValue", "result"]>]> {
  let summary = "conditional select";
  let description = [{
    The `pop.select` operation is used to choose one value based on a condition,
    without IR-level branching.

    Example:

    ```mlir
    %0 = pop.select %cond, %arg0, %arg1 : !kgen.struct<(f32)>
    ```
  }];

  let arguments = (ins BoolScalar:$condition, AnyType:$trueValue,
                   AnyType:$falseValue);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $condition `,` $trueValue `,` $falseValue attr-dict `:` type($result)
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// BoolOp
//===----------------------------------------------------------------------===//

// Base class for bitwise arithmetic pop operations.
class POP_BoolOp<string mnemonic, list<Trait> traits = []> : POPOp<mnemonic,
      traits # [Pure, Commutative, SameOperandsAndResultType]> {

  let arguments = (ins I1:$lhs, I1:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AndOp
//===----------------------------------------------------------------------===//

def POP_AndOp : POP_BoolOp<"and", [Idempotent]> {
  let summary = "bitwise logical and";
  let description = [{
    The `pop.and` returns the logical and of its two operands. The type
    of the operands must be a boolean.

    Example:

    ```mlir
    %c = pop.and %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// OrOp
//===----------------------------------------------------------------------===//

def POP_OrOp : POP_BoolOp<"or", [Idempotent]> {
  let summary = "bitwise logical or";
  let description = [{
    The `pop.or` returns the logical or of its two operands. The type
    of the operands must be a boolean.

    Example:

    ```mlir
    %c = pop.or %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// XOrOp
//===----------------------------------------------------------------------===//

def POP_XOrOp : POP_BoolOp<"xor"> {
  let summary = "bitwise logical xor";
  let description = [{
    The `pop.xor` returns the logical exclusive or of its two operands. The
    type of the operands must be a boolean.

    Example:

    ```mlir
    %c = pop.xor %a, %b
    ```
  }];
}

//===----------------------------------------------------------------------===//
// ArrayCreateOp
//===----------------------------------------------------------------------===//

def POP_ArrayCreateOp : POPOp<"array.create", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>,
    ArrayElementType<"result", "operands">]> {
  let summary = "create an array";
  let description = [{
    The `pop.array.create` operation creates an `!pop.array` populated with the
    given SSA values. The resulting array must have a fixed size equal to the
    number of operands.

    Example:

    ```mlir
    // Create an array of scalar floats.
    %0 = pop.array.create [%arg0, %arg1] : !pop.array<2, scalar<f32>>

    // Create an array of structs.
    %1 = pop.array.create [%arg2, %arg3] : !pop.array<2, struct<()>>

    // Create an array of parametric types.
    %2 = pop.array.create [%arg4] : !pop.array<1, T>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs ArrayKnownSize:$result);
  let assemblyFormat = [{
    ` ` `[` $operands `]` attr-dict `:` qualified(type($result))
  }];

  let builders = [
    OpBuilder<(ins "ValueRange":$elements)>
  ];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ArrayRepeatOp
//===----------------------------------------------------------------------===//

def POP_ArrayRepeatOp : POPOp<"array.repeat", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>,
    ArrayElementType<"result", "operands">]> {
  let summary = "create an array";
  let description = [{
    The `pop.array.repeat` operation facilitates the creation of splat or
    parametric arrays. The operation repeats the given operands into the
    resulting array in the provided order until the array is full. Extra
    elements are truncated.

    It is only valid to have no operands if the resulting array has no elements.

    Example:

    ```mlir
    // Create a splat array.
    %0 = pop.array.repeat [%arg0] : !pop.array<N, scalar<f32>>

    // Repeat the pair to fill the array. The last element is dropped.
    %1 = pop.array.repeat [%arg2, %arg3] : !pop.array<9, T>

    // Repeat the triplet to populate the array.
    %2 = pop.array.create [%a, %b, %c] : !pop.array<M, T>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs POP_ArrayType:$result);
  let assemblyFormat = [{
    ` ` `[` $operands `]` attr-dict `:` qualified(type($result))
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ArrayGetOp
//===----------------------------------------------------------------------===//

def POP_ArrayGetOp : POPOp<"array.get", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>,
    ArrayElementType<"array", "result">]> {
  let summary = "array get element";
  let description = [{
    The `pop.array.get` operation extracts the element of the given array at the
    specified index. An out-of-bounds index is invalid.

    Example:

    ```mlir
    // Get the third element.
    %0 = pop.array.get %arr0[2] : !pop.array<4, T>

    // Get the first element.
    %1 = pop.array.get %arr1[0] : !pop.array<N, f32>
    ```
  }];

  let arguments = (ins POP_ArrayType:$array, AttrOfType<Index>:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $array `[` custom<IndexParamValue>($index) `]` attr-dict `:`
    qualified(type($array))
  }];

  let builders = [OpBuilder<(ins "Value":$array, "int64_t":$index)>];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ArrayReplaceOp
//===----------------------------------------------------------------------===//

def POP_ArrayReplaceOp : POPOp<"array.replace", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>,
    AllTypesMatch<["array", "result"]>, ArrayElementType<"result", "value">]> {
  let summary = "array replace element";
  let description = [{
    The `pop.array.replace` operation inserts the given into the array at the
    given index. It returns a new array with the inserted value. The original
    array is not modified. An out-of-bounds index is invalid.

    Example:

    ```mlir
    // Replace the second element.
    %0 = pop.array.replace %a, %arr0[1] : !pop.array<4, T>

    // Replace the first element.
    %1 = pop.array.replace %b, %arr1[0] : !pop.array<N, i32>
    ```
  }];

  let arguments = (ins AnyType:$value, POP_ArrayType:$array,
                       AttrOfType<Index>:$index);
  let results = (outs POP_ArrayType:$result);
  let assemblyFormat = [{
    $value `,` $array `[` custom<IndexParamValue>($index) `]` attr-dict `:`
    qualified(type($result))
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizeMethod = true;
}

//===----------------------------------------------------------------------===//
// ArrayGEPOp
//===----------------------------------------------------------------------===//

def POP_ArrayGEPOp : POPOp<"array.gep", [Pure,
    TypesMatchWith<"pointer to array element type", "array", "result",
                   "getPointerToArrayElementType($_self)">,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "array get element pointer";
  let description = [{
    The `pop.array.gep` operation takes a pointer to a `!pop.array` and an index
    and returns a pointer to the array element at that index. If the index is
    out-of-bounds, the result of the operation is a poison value.

    Example:

    ```mlir
    %arr = pop.stack_allocation 1 : !pop.array<4, i32>
    %two = index.constant 2
    %v2 = pop.array.gep %arr[%two] : <array<4, i32>>
    ```
  }];

  let arguments = (ins PointerOf<POP_ArrayType>:$array, Index:$index);
  let results = (outs PointerType:$result);
  let assemblyFormat = "$array `[` $index `]` attr-dict `:` type($array)";
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// StackAllocationOp
//===----------------------------------------------------------------------===//

def POP_StackAllocationOp : POPOp<"stack_allocation", [BytecodeOpInterface]> {
  let summary = "scoped stack allocation operation";
  let description = [{
    The `pop.stack_allocation` operation allocates statically-sized scoped
    memory on the stack. The operation returns a pointer to the allocated memory
    of count elements aligned to the element type.

    The user can specify the alignment of the allocation via the optional
    `align` attribute. Similarly, the user can provide the address space
    of the allocation via the `address_space` parameter.

    Example:

    ```mlir
    %0 = pop.stack_allocation count x !kgen.simd<4, f32>
    %1 = pop.stack_allocation count x !kgen.scalar<ui64>
    %2 = pop.stack_allocation count x !kgen.scalar<ui64> align 8
    %3 = pop.stack_allocation count x !kgen.scalar<ui64> align 8 address_space 5
    ```

    The allocated memory is automatically released when control transfers back
    from the region of its closest surrounding operation.

    Example:

    ```mlir
    kgen.generator @stack_memory<count, T: type>() {
      // This memory is allocated at the start of the function call and released
      // at the end.
      %0 = pop.stack_allocation count x T

      scf.if %cond {
        // This memory is allocated when control flow branches into the 'then'
        // region and released when control flow transfer back.
        %1 = pop.stack_allocation count x T
      }

      scf.for %iv = %lb to %ub step %step {
        // This memory is allocated each time the loop iterates and released
        // at the end of each iteration.
        %2 = pop.stack_allocation count x T
      }
    }
    ```

    If stack memory escapes the scope of its enclosing region, interacting with
    the memory outside its origin is undefined behaviour.

    Example:

    ```mlir
    // Scoped to the enclosing region.
    %0 = pop.stack_allocation count x type

    %mem = scf.if %cond -> !kgen.pointer<type> {
      // OK: The memory is scoped to the enclosing region.
      scf.yield %0 : !kgen.pointer<type>
    } else {
      // Scoped to the 'else' region.
      %1 = pop.stack_allocation count x type

      // The memory is escaping its enclosed region.
      scf.yield %1 : !kgen.pointer<type>
    }

    // NOT OK: Loading pointer to invalid memory.
    %v = pop.load %mem : !kgen.pointer<type>
    ```
  }];

  let arguments = (ins
    AttrOfType<Index>:$count,
    OptionalAttr<AttrOfType<Index>>:$alignment,
    UnitAttr:$markedLifetimes
  );
  let results = (outs Res<PointerType, "", [MemAlloc]>:$result);

  let assemblyFormat = [{
    custom<IndexParamValue>($count) `x` custom<PointerOf>(type($result))
    (`align` custom<IndexParamValue>($alignment)^)?
    (`marked` $markedLifetimes^)? attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Type":$result, CArg<"int64_t", "1">:$count,
                   CArg<"TypedAttr", "TypedAttr()">:$alignment,
                   CArg<"bool", "false">:$markedLifetimes)>,
    OpBuilder<(ins "bool":$markedLifetimes, "Type":$result)>,
  ];

  let extraClassDeclaration = [{
    struct Payload { int32_t size, align; };
    ErrorOrSuccess compile(Payload &payload, TargetInfoAttr target);
    ErrorOrSuccess parametric_compile(Payload &payload,
      TargetInfoAttr target,
      ArrayRef<Attribute> operands,
      ParametricInterpreterState& state);
    ErrorTreeOrSuccess interpret(ArrayRef<Attribute> operands,
      const Payload &payload, InterpreterState &state);
    ErrorTreeOrSuccess parametric_interpret(ArrayRef<Attribute> operands,
      const Payload &payload, ParametricInterpreterState &state);
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// StackAllocLifetimeStartOp
//===----------------------------------------------------------------------===//

def POP_StackAllocLifetimeStartOp : POPOp<"stack_alloc.lifetime.start", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "Marks the start of stack allocated values lifetimes";
  let description = [{
    The `pop.stack_alloc.lifetime.start` operation marks the beginning of
    the origin of a stack allocated value.

    Example:

    ```mlir
    %2 = pop.stack_allocation 1 x index
    pop.stack_alloc.lifetime.start(%2) : !kgen.pointer<index>
    ```
  }];

  let arguments = (ins Variadic<PointerType>:$values);
  let assemblyFormat = [{
    attr-dict `(` $values `)` (`:` type($values)^)?
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// StackAllocLifetimeEndOp
//===----------------------------------------------------------------------===//

def POP_StackAllocLifetimeEndOp : POPOp<"stack_alloc.lifetime.end", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "Marks the end of stack allocated values lifetimes";
  let description = [{
    The `pop.stack_alloc.lifetime.end` operation marks the end of the
    origin of a stack allocated value.

    Example:

    ```mlir
    %2 = pop.stack_allocation 1 x index
    pop.stack_alloc.lifetime.end(%2) : !kgen.pointer<index>
    ```
  }];

  let arguments = (ins Variadic<PointerType>:$values);
  let assemblyFormat = [{
    attr-dict `(` $values `)` (`:` type($values)^)?
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AlignedAllocOp
//===----------------------------------------------------------------------===//

def POP_AlignedAllocOp : POPOp<"aligned_alloc", [
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "aligned heap allocation";
  let description = [{
    The `pop.aligned_alloc` operation performs an aligned allocation of memory
    on the heap with a dynamic alignment. The resulting pointer is guaranteed to
    have the specified alignment. A value of `-1` may be passed as alignment to
    indicate system alignment.

    Example:

    ```mlir
    %0 = pop.aligned_alloc %align, %size : <struct<(f32, f64)>>
    ```
  }];

  let arguments = (ins Index:$align, Index:$size);
  let results = (outs Res<PointerType, "", [MemAlloc]>:$ptr);

  let assemblyFormat = "$align `,` $size attr-dict `:` type($ptr)";
}

//===----------------------------------------------------------------------===//
// AlignedFreeOp
//===----------------------------------------------------------------------===//

def POP_AlignedFreeOp : POPOp<"aligned_free", [
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "free aligned heap memory";
  let description = [{
    The `pop.aligned_free` operation frees heap memory allocated by
    `pop.aligned_alloc`. Passing any other kind of pointer to the operation is
    undefined behaviour.

    Example:

    ```mlir
    pop.aligned_free %ptr : <struct<(f32, i64)>>
    ```
  }];

  let arguments = (ins Arg<PointerType, "", [MemFree]>:$ptr);

  let assemblyFormat = "$ptr attr-dict `:` type($ptr)";
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// NoAliasPointerCast
//===----------------------------------------------------------------------===//

def POP_NoAliasPointerCastOp : POPOp<"noalias_pointer_cast", [Pure,
    AllTypesMatch<["in", "out"]>]> {
  let summary = "declare a pointer value that locally does not alias";
  let description = [{
    The `pop.noalias_pointer_cast` operation takes a pointer and casts it to a
    noalias pointer. This tells the optimizer that the resultant point is known
    not to alias any other pointer within the current scope. The "current scope"
    refers to the span of instructions that comprise the function in which the
    operation is contained. This property carries transitively through inlining.

    Example:

    ```mlir
    %0 = pop.noalias_pointer_cast %ptr : <simd<4, i32>>
    ```
  }];

  let arguments = (ins PointerType:$in);
  let results = (outs PointerType:$out);

  let assemblyFormat = "$in attr-dict `:` type($in)";
}

//===----------------------------------------------------------------------===//
// ExternalCallOp
//===----------------------------------------------------------------------===//

def POP_ExternalCallOp : POPOp<"external_call", [
    DeclareOpInterfaceMethods<InterpreterOpInterface>,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
  ]> {
  let summary = "call an external function";
  let description = [{
    The `pop.external_call` operation invokes an externally-defined function,
    e.g. from libc, using the provided arguments. The function signature is
    inferred from the operand and result types.

    Example:

    ```mlir
    %0 = pop.external_call @sqrt(%arg) : (!kgen.scalar<f64>) -> !kgen.scalar<f64>
    ```

    A C variadic callee is indicated by the `numFixedArgs` attribute, which
    gives how many of the first operands are fixed arguments of the callee. The
    operands after those are passed as variadic arguments, which the C ABI may
    place differently than fixed arguments. The presence of the attribute, not
    its value, is what marks the callee variadic, so a count of zero is a callee
    whose every operand is variadic.

    ```mlir
    pop.external_call @printf(%fmt, %v0, %v1)
      attributes {numFixedArgs = 1 : index}
      : (!kgen.pointer<scalar<si8>>, !kgen.scalar<f32>, !kgen.scalar<ui32>) -> ()
    ```

    The `funcAttrs` attribute specifies LLVM function attributes on the external
    callee, such as `noinline` or `noreturn`. LLVM argument and result
    attributes can be attached to the operation through `argAttrs` and
    `resAttrs` as well, where `resAttrs` is a `DictionaryAttr` as only one
    result can be returned.
  }];

  let arguments = (ins
    AttrOfType<KGEN_StringType>:$func, Variadic<AnyType>:$operands,
    OptionalAttr<AttrOfType<Index>>:$numFixedArgs,
    OptionalAttr<ArrayAttr>:$funcAttrs, OptionalAttr<DictArrayAttr>:$argAttrs,
    OptionalAttr<DictionaryAttr>:$resAttrs, OptionalAttr<AnyAttr>:$memory);
  let results = (outs Optional<AnyType>:$result);

  let assemblyFormat = [{
    custom<ExternalCallee>($func) `(` operands `)`
    attr-dict-with-keyword `:`
    functional-type(operands, results)
  }];

  let extraClassDeclaration = [{
    // Get the name of the concrete callee.
    StringAttr getCallee() { return cast<StringAttr>(getFunc()); }
  }];

  let builders = [
    OpBuilder<(ins "StringRef":$func, "ValueRange":$operands)>,
    OpBuilder<(ins "Type":$result, "StringRef":$func,
                   "ValueRange":$operands)>,
    OpBuilder<(ins "Type":$result, "StringRef":$func,
                   "ValueRange":$operands, "int64_t":$numFixedArgs)>
  ];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ExternPointerSymbolOp
//===----------------------------------------------------------------------===//

def POP_ExternPointerSymbolOp : POPOp<"extern_ptr_symbol"> {
  let summary = "Gets a pointer to an external symbol with particular alignment";

  let description = [{
    The `pop.external_ptr_symbol` returns a pointer to an opaque external symbol

    This operation is useful to model runtime allocation for backends that models
    this allocation as externalized symbols e.g NVPTX.

    Example:

    ```mlir
    %1 = pop.extern_ptr_symbol "name" alignment <4> : !kgen.pointer<scalar<f32>, 3>
    ```
  }];

  let arguments = (ins AttrOfType<KGEN_StringType>:$name,
                       AttrOfType<Index>:$alignment);
  let results = (outs Res<PointerType, "", [MemAlloc]>:$res_symbol);

  let assemblyFormat = [{
    custom<ParamValue>($name, "StringType::get($_ctxt)")
    `alignment` `<` custom<IndexParamValue>($alignment) `>`
    attr-dict `:` type($res_symbol)
  }];

}

//===----------------------------------------------------------------------===//
// GlobalAllocOp
//===----------------------------------------------------------------------===//

def POP_GlobalAllocOp : POPOp<"global_alloc", [Pure,
    BytecodeOpInterface]> {
  let summary = "allocate a variable in global memory";
  let description = [{
    The `pop.global_alloc` operation allocates a variable as a global. What
    this means is the returned pointer refers to some global value. The
    operation is not inherently thread-safe or re-entrant, but that depends on
    the particular interpretation of the address space.

    An optional `initializer` attribute places the global in `.data` with a
    compile-time value pre-set, instead of the default BSS (zero-initialized)
    behavior. When an initializer is present, `count` must be 1.

    Example:

    ```mlir
    %ptr = pop.global_alloc "name" 2 x !kgen.scalar<si32> align 32 address_space 3
    %init = pop.global_alloc "val" 1 x !kgen.scalar<si32> = <42>
    ```
  }];

  let arguments = (ins AttrOfType<KGEN_StringType>:$name,
                       AttrOfType<Index>:$count,
                       DefaultValuedAttr<
                        POP_GlobalAllocAddressSpaceAttr,
                        "GlobalAllocAddressSpace::IGNORE">:$memoryType,
                       OptionalAttr<AttrOfType<Index>>:$alignment,
                       OptionalAttr<TypedAttrInterface>:$initializer);
  let results = (outs Res<PointerType, "", [MemAlloc]>:$result);

  let assemblyFormat = [{
    custom<ParamValue>($name, "StringType::get($_ctxt)")
    custom<IndexParamValue>($count) `x` custom<PointerOf>(type($result))
    (`align` custom<IndexParamValue>($alignment)^)?
    custom<OptionalGlobalAllocInitializer>($initializer, ref(type($result)))
    attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    struct Payload { int32_t size, align, addressSpace; };
    ErrorOrSuccess compile(Payload &payload, TargetInfoAttr target);
    ErrorOrSuccess parametric_compile(Payload &payload,
      TargetInfoAttr target,
      ArrayRef<Attribute> operands,
      ParametricInterpreterState& state);
    ErrorTreeOrSuccess interpret(ArrayRef<Attribute> operands,
      const Payload &payload, InterpreterState &state);
    ErrorTreeOrSuccess parametric_interpret(ArrayRef<Attribute> operands,
      const Payload &payload, ParametricInterpreterState &state);
  }];
}

//===----------------------------------------------------------------------===//
// GlobalConstantOp
//===----------------------------------------------------------------------===//

def POP_GlobalConstantOp : POPOp<"global_constant",
   [Pure, PointerElementType<"result", "value">, InferTypeOpAdaptor,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "create a global constant";
  let description = [{
    The `pop.global_constant` operation creates and returns a pointer to a
    global constant value. The returned pointer is read-only. Writing the to
    pointer is undefined behaviour.

    Example:

    ```mlir
    // SIMD global constant.
    %0 = pop.global_constant: <simd<2, ui32>> = <<2, 5>>
    // Parametric global constant.
    %1 = pop.global_constant: <type> =
        <apply(:(f32) -> !kgen.param<type>, 12.0)>
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value,
                        OptionalAttr<AttrOfType<Index>>:$alignment);
  let results = (outs PointerType:$result);

  let assemblyFormat = [{
    `` custom<GlobalConstantOpValue>($value, type($result))
    (`align` custom<IndexParamValue>($alignment)^)? attr-dict
  }];

  let builders = [
    OpBuilder<(ins "TypedAttr":$value)>
  ];
}

//===----------------------------------------------------------------------===//
// CompilerGlobalLoadOp
//===----------------------------------------------------------------------===//

def POP_CompilerGlobalLoadOp : POPOp<"compiler.global_load", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "load the value of a global variable";
  let description = [{
    The `pop.compiler.global_load` operation loads a global variable by
    name so it can be used by SSA value. This op is a compile time only
    operation and expects to be folded away.

    Example:

    ```mlir
    %0 = pop.compiler.global_load "foo" : !kgen.scalar<ui32>
    ```
  }];

  let arguments = (ins StrAttr:$name);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$name attr-dict `:` qualified(type($result))";
}

//===----------------------------------------------------------------------===//
// CompilerGlobalStoreOp
//===----------------------------------------------------------------------===//

def POP_CompilerGlobalStoreOp : POPOp<"compiler.global_store", [
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "store a value into a global variable";
  let description = [{
    The `pop.compiler.global_store` operation stores a value into a global
    variable by symbol name. It is a compile error if this op exists by the
    time POP is lowered to LLVM.

    Example:

    ```mlir
    %0 = pop.constant(0) : !kgen.scalar<ui32>
    pop.compiler.global_store "foo", %0 : !kgen.scalar<ui32>
    ```
  }];

  let arguments = (ins StrAttr:$name, AnyType:$value);

  let assemblyFormat = "$name `,` $value `:` type($value) attr-dict";
}


//===----------------------------------------------------------------------===//
// PointerToIndexOp
//===----------------------------------------------------------------------===//

def POP_PointerToIndexOp : POPOp<"pointer_to_index", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>,
    DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "gets the address of the pointer as an index type.";
  let description = [{
    The `pop.pointer_to_index` instruction takes a pointer or a simd value of
    address dtype to cast and gets its address as a simd of index dtype.

    Example:

    ```mlir
    %addr0 = pop.pointer_to_index %ptr : !kgen.pointer<scalar<f32>> to index
    %addr1 = pop.pointer_to_index %idx : !kgen.pointer<simd<4, type>> to index
    ```
  }];

  let arguments = (ins PointerType:$value);
  let results = (outs Index:$result);
  let assemblyFormat = "$value attr-dict `:` type($value)";
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// CastToBuiltinOp
//===----------------------------------------------------------------------===//

def POP_CastToBuiltinOp : POPOp<"cast_to_builtin", [Pure]> {
  let summary = "Unmaterialized type lowering conversion.";
  let description = [{
    The `pop.cast_to_builtin` operation casts a POP type to an MLIR builtin
    type. If the types have the same underlying implementation, then this
    operation is a no-op adaptor between two type systems. It can, for example,
    cast a `!kgen.scalar` to an MLIR builtin integer or float type, or a
    `pop.simd` to a `vector.

    Example:

    ```mlir
    kgen.func @llvm_ops(%arg0: !kgen.scalar<f32>) -> !kgen.scalar<f32> {
      %0 = pop.cast_to_builtin %arg0 : !kgen.scalar<f32> to f32
      %1 = llvm.fadd %0, %0 : f32
      %2 = pop.cast_from_builtin %1 : f32 to !kgen.scalar<f32>
      kgen.return %2 : !kgen.scalar<f32>
    }
    ```
  }];

  let arguments = (ins KGEN_SIMDType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    $input attr-dict `:` qualified(type($input)) `to` type($output)
  }];

  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CastFromBuiltinOp
//===----------------------------------------------------------------------===//

def POP_CastFromBuiltinOp : POPOp<"cast_from_builtin", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "Unmaterialized type raising conversion.";
  let description = [{
    The `pop.cast_from_builtin` operation casts an MLIR builtin type to a POP
    type. If the types have the same underlying implementation, then this
    operation is a no-op adaptor between two type systems. It can, for example,
    cast an MLIR builtin integer or float type to a `!kgen.scalar`, or a 1-D
    vector to `!kgen.simd`.

    Example:

    ```mlir
    kgen.func @llvm_ops(%arg0: !kgen.scalar<f32>) -> !kgen.scalar<f32> {
      %0 = pop.cast_to_builtin %arg0 : !kgen.scalar<f32> to f32
      %1 = llvm.fadd %0, %0 : f32
      %2 = pop.cast_from_builtin %1 : f32 to !kgen.scalar<f32>
      kgen.return %2 : !kgen.scalar<f32>
    }
    ```

    Importantly, this operation can adapt parametric constants for use in
    parametric operations.

    Example:

    ```mlir
    kgen.generator @addN<N: si32>(%arg0: !kgen.scalar<si32>)
        -> !kgen.scalar<si32> {
      %N = kgen.param.constant: si32 = <N>
      %rhs = pop.cast_from_builtin %N : si32 to !kgen.scalar<si32>
      %result = pop.add %arg0, %rhs : !kgen.scalar<f32>
      kgen.return %result : !kgen.scalar<f32>
    }
    ```
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs KGEN_SIMDType:$output);
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `to` qualified(type($output))
  }];

  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// CallLLVMIntrinsicOp
//===----------------------------------------------------------------------===//

def POP_CallLLVMIntrinsicOp : POPOp<"call_llvm_intrinsic",[
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<ConditionallySpeculatable>,
    POP_FastmathFlagsInterface,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "Call to an LLVM intrinsic function.";
  let description = [{
    The `pop.call_llvm_intrinsic` operation calls an llvm intrinsic using the
    `llvm.call_intrinsic` op.

    Example:

    ```mlir
      %0 = pop.call_llvm_intrinsic "llvm.round", (%arg0) :
        !kgen.scalar<f32> -> !kgen.scalar<f32>
    ```
  }];
  let arguments = (ins
    AttrOfType<KGEN_StringType>:$intrin, Variadic<AnyType>:$args,
    DefaultValuedAttr<POP_FastmathFlagsAttr,
                      "FastmathFlags::none">:$fastmathFlags,
    AttrOfType<I1>:$hasSideEffects);

  let results = (outs Optional<AnyType>:$results);
  let assemblyFormat = [{
    custom<I1Flag>($hasSideEffects, "\"side_effecting\"")
    custom<ParamValue>($intrin, "StringType::get($_ctxt)") `,` `(` $args `)`
    attr-dict `:` functional-type($args, $results)
  }];
}

//===----------------------------------------------------------------------===//
// InlineAsmOp
//===----------------------------------------------------------------------===//

def POP_InlineAsmOp : POPOp<"inline_asm"> {
  let description = [{
    The `pop.inline_asm` mirrors the underlying LLVM semantics with a notable
    exception: the embedded `asm_string` is not allowed to define or reference
    any symbol or any global variable: only the operands of the op may be read,
    written, or referenced.
    Attempting to define or reference any symbol or any global behavior is
    considered undefined behavior at this time.
  }];
  let arguments = (ins
    Variadic<AnyType>:$operands,
    AttrOfType<KGEN_StringType>:$assembly,
    AttrOfType<KGEN_StringType>:$constraints,
    AttrOfType<I1>:$hasSideEffects,
    AttrOfType<I1>:$isStackAligned,
    OptionalAttr<ArrayAttr>:$operandAttrs
  );

  let results = (outs Variadic<AnyType>:$res);

  let assemblyFormat = [{
    custom<I1Flag>($hasSideEffects, "\"side_effecting\"")
    custom<I1Flag>($isStackAligned, "\"stack_aligned\"")
    (`operand_attrs` `=` $operandAttrs^)?
    attr-dict
    custom<ParamValue>($assembly, "StringType::get($_ctxt)") `,`
    custom<ParamValue>($constraints, "StringType::get($_ctxt)") `,`
    `(` operands `)` `:` functional-type(operands, results)
   }];
}

//===----------------------------------------------------------------------===//
// AtomicCmpXchgOp
//===----------------------------------------------------------------------===//

def POP_AtomicCmpXchgResultType : Type<And<[
  KGEN_StructType.predicate,
  // Ensure the struct is resolved before accessing element types.
  CPred<"llvm::cast<StructType>($_self).isResolved()">,
  CPred<"*llvm::cast<StructType>($_self).getNumElements() == 2">,
  SubstLeaves<"$_self",
              "(*llvm::cast<StructType>($_self).getElementTypes())[0]",
              UnboundOrIndexOrInt.predicate>,
  SubstLeaves<"$_self",
              "(*llvm::cast<StructType>($_self).getElementTypes())[1]",
              BoolScalar.predicate>]>,
 "a resolved KGEN struct type with any integer or pointer followed by a boolean">;

def POP_AtomicCmpXchgOp : POPOp<"atomic.cmpxchg", [
    PointerElementType<"ptr", "cmp">,
    PointerElementType<"ptr", "val">,
    AllTypesMatch<["cmp", "val"]>,
    TypesMatchWith<
      "an KGEN struct type with any integer or pointer followed by a boolean",
      "ptr", "result", "getCmpXChgResultType($_self)">]> {
  let summary = "comparison";
  let description = [{
    The `pop.atomic.cmpxchg` operation is used to atomically modify memory. It
    loads a value in memory and compares it to a given value. If they are equal,
    it tries to store a new value into the memory.

    There are three arguments to the `atomic.cmpxchg` instruction are:

      - an address to operate on
      - a value to compare to the value currently be at that address
      - a new value to place at that address if the compared values are equal.

    The type of `cmp` must be an integer or pointer type whose bit width is a
    power of two greater than or equal to eight and less than or equal to a
    target-specific size limit. `cmp` and `val` must have the same type, and
    the type of `pointer` must be a pointer to that type.

    The success and failure ordering arguments specify how this cmpxchg
    synchronizes with other atomic operations. Both ordering parameters must be
    at least monotonic, the failure ordering cannot be either release or acq_rel.

    The `weak` argument allows the operation to fail spuriously, that is, act as
    if `%ptr` != `%cmp` even if they are equal. This is sometimes desirable if using
    cmpxchg in a loop as it may yield better performance.

    The return type is a KGEN struct type with any integer or pointer followed
    by a boolean value.

    Example:

    ```mlir
    %b0 = pop.atomic.cmpxchg %ptr, %cmp, %val monotonic monotonic :
        !kgen.pointer<scalar<index>>

    %b1 = pop.atomic.cmpxchg %ptr, %cmp, %val scyncscope("singlethread")
        monotonic monotonic : !kgen.pointer<scalar<index>>

    %b2 = pop.atomic.cmpxchg weak %ptr, %cmp, %val acq_rel monotonic :
        !kgen.pointer<scalar<index>>
    ```
  }];

  let arguments = (ins PointerOf<UnboundOrIndexOrInt>:$ptr,
                       UnboundOrIndexOrInt:$cmp,
                       UnboundOrIndexOrInt:$val,
                       POP_AtomicOrderingAttr:$success_ordering,
                       POP_AtomicOrderingAttr:$failure_ordering,
                       OptionalAttr<AttrOfType<KGEN_StringType>>:$syncscope,
                       OptionalAttr<AttrOfType<Index>>:$alignment,
                       OptionalAttr<UnitAttr>:$weak);
  let results = (outs POP_AtomicCmpXchgResultType:$result);

  let assemblyFormat = [{
    (`weak` $weak^)?
    $ptr `,` $cmp `,` $val
    (`syncscope` `(` custom<StringParam>($syncscope)^ `)`)? enum($success_ordering)
    enum($failure_ordering) (`align` custom<IndexParamValue>($alignment)^ )?
    attr-dict `:` qualified(type($ptr))
  }];
}

//===----------------------------------------------------------------------===//
// AtomicRMWOp
//===----------------------------------------------------------------------===//

def POP_AtomicRMWOp : POPOp<"atomic.rmw", [
    PointerElementType<"ptr", "val">,
    PointerElementType<"ptr", "result">,
    AllTypesMatch<["val", "result"]>]> {
  let summary = "used to atomically modify memory using a specified binop";
  let description = [{
    The `pop.atomic.rmw` op is used to atomically modify memory.

    There are three arguments to the `pop.atomic.rmw` op are: an operation to
    apply, an address whose value to modify, an argument to the operation.

    The contents of memory at the location specified by the `ptr` operand are
    atomically read, modified, and written back. The original value at the
    location is returned. The modification is specified by the operation
    argument:

    - xchg: `*ptr = val`
    - add: `*ptr = *ptr + val`
    - sub: `*ptr = *ptr - val`
    - and: `*ptr = *ptr & val`
    - nand: `*ptr = ~(*ptr & val)`
    - or: `*ptr = *ptr | val`
    - xor: `*ptr = *ptr ^ val`
    - max: `*ptr = *ptr > val ? *ptr : val`
    - min: `*ptr = *ptr < val ? *ptr : val`

    Example:

    ```mlir
    %0 = pop.atomic.rmw add(%ptr, %val) monotonic : !kgen.pointer<scalar<index>>
    %0 = pop.atomic.rmw add(%ptr, %val) syncscope(singlethread") monotonic :
                                        !kgen.pointer<scalar<index>>
    ```
  }];

  let arguments = (ins POP_AtomicBinOpAttr:$bin_op,
                       PointerOf<UnboundOrArithmeticScalar>:$ptr,
                       UnboundOrArithmeticScalar:$val,
                       POP_AtomicOrderingAttr:$ordering,
                       OptionalAttr<AttrOfType<KGEN_StringType>>:$syncscope);
  let results = (outs UnboundOrArithmeticScalar:$result);

  let assemblyFormat = [{
    enum($bin_op) `(` $ptr `,` $val `)`
    (`syncscope` `(` custom<StringParam>($syncscope)^ `)`)? enum($ordering)
    attr-dict `:` qualified(type($ptr))
  }];
}

//===----------------------------------------------------------------------===//
// FenceOp
//===----------------------------------------------------------------------===//

def POP_FenceOp : POPOp<"fence"> {
  let summary = "used to introduce happens-before edges between operations";
  let description = [{
    The `pop.fence` op is used to introduce happens-before edges between
    operations.

    The `pop.fence` op take an ordering argument which defines what
    synchronizes-with edges they add. They can only be given acquire, release,
    acq_rel, and seq_cst orderings.

    Example:

    ```mlir
    pop.fence seq_cst
    pop.fence syncscope("singlethread") acq_rel
    ```
  }];

  let arguments = (ins POP_AtomicOrderingAttr:$ordering,
                       OptionalAttr<StrAttr>:$syncscope);

  let assemblyFormat = [{
    (`syncscope` `(` $syncscope^ `)`)? enum($ordering) attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// StringAddressOp
//===----------------------------------------------------------------------===//

def POP_StringAddressOp : POPOp<"string.address", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "Get the address of a string.";
  let description = [{
    The `pop.string.address` operation gets the pointer of the first byte of a
    string.

    Example:

    ```mlir
    %0 = kgen.param.constant: string = <"hello">
    %1 = pop.string.address %0
    ```
  }];
  let arguments = (ins KGEN_StringType:$str);
  let results = (outs PointerOf<SI8Scalar>:$result);
  let assemblyFormat = "$str attr-dict";
}

//===----------------------------------------------------------------------===//
// StringSizeOp
//===----------------------------------------------------------------------===//

def POP_StringSizeOp : POPOp<"string.size", [Pure]> {
  let summary = "Get the size of a string.";
  let description = [{
    The `pop.string.size` operation gets the size of string.

    Example:

    ```mlir
    %0 = kgen.param.constant: string = <"hello">
    %1 = pop.string.size %0  // size = 5
    ```
  }];

  let arguments = (ins KGEN_StringType:$str);
  let results = (outs Index:$result);
  let assemblyFormat = "$str attr-dict";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// DTypeToUI8
//===----------------------------------------------------------------------===//

def POP_DTypeToUI8 : POPOp<"dtype.to_ui8", [Pure]> {
  let summary = "Converts a !kgen.dtype to a ui8 value.";
  let description = [{
    The `pop.dtype.to_ui8` operation gets the `ui8` representation of the
    `!kgen.dtype`.

    Example:

    ```mlir
    %0 = kgen.param.constant: dtype = <bool>
    %1 = pop.dtype.to_ui8 %0  // result = 1
    ```
  }];

  let arguments = (ins KGEN_DTypeType:$dType);
  let results = (outs UI8:$result);
  let assemblyFormat = "$dType attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// DTypeFromUI8
//===----------------------------------------------------------------------===//

def POP_DTypeFromUI8 : POPOp<"dtype.from_ui8", [Pure]> {
  let summary = "Converts a ui8 value to a !kgen.dtype.";
  let description = [{
    The `pop.dtype.to_ui8` operation gets the `!kgen.dtype` type from the `ui8`
    value.

    Example:

    ```mlir
    %0 = kgen.param.constant: ui8 = <1>
    %1 = pop.dtype.from_ui8 1  // result = bool
    ```
  }];

  let arguments = (ins UI8:$value);
  let results = (outs KGEN_DTypeType:$result);
  let assemblyFormat = "$value attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// VariantBitcastOp
//===----------------------------------------------------------------------===//

def POP_VariantBitcastOp : POPOp<"variant.bitcast", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "Bitcast a pointer to a variant to one of its element types.";
  let description = [{
    The `pop.variant.bitcast` operation takes a pointer to a variant type and
    bitcasts it to a pointer of one of its element types. This operation should
    be used to access element types of a variant over other pointer bitcasts
    because it provides stronger invariants over the target element type.

    The result type is still required because the variant type might have a
    parametric type list and the index may be a parameter expression. If both
    are concrete, then the result element type must match the variant type at
    the specified index.

    Example:

    ```mlir
    %i32ptr = pop.variant.bitcast %variant, <0> : <variant<i32, i64>> as <i32>
    ```
  }];

  let arguments = (ins PointerOf<KGEN_VariantType>:$variant,
                       AttrOfType<Index>:$index);
  let results = (outs PointerType:$result);
  let assemblyFormat = [{
    $variant `,` `<` custom<IndexParamValue>($index) `>` attr-dict `:`
    type($variant) `as` type($result)
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// VariantDiscrGEPOp
//===----------------------------------------------------------------------===//

def POP_VariantDiscrGEPOp : POPOp<"variant.discr_gep", [Pure,
  BytecodeOpInterface]> {
  let summary = "Return a pointer to the discriminant field of a variant.";
  let description = [{
    The `pop.variant.discr_gep` operation takes a pointer to a variant type and
    returns a pointer to the discriminant field inside the variant. This
    operation should be used to modify the discriminant of a variant. The
    operation returns a pointer to an unsigned integer `!kgen.scalar` whose dtype
    is just large enough to represent the number of variant types.

    The reason `!kgen.variant` combines both a union and a discriminant is
    because `!pop.union` is ultimately an opaque array of bytes. Before
    elaboration, the compiler doesn't have a target model with which to
    understand the layout of `!pop.union`, and thus it can neither constant
    fold nor interpreter union operations. `!kgen.variant` includes the
    discriminant in its representation, which is sufficient for the compiler to
    abstractly interpret it without a target model.

    Example:

    ```mlir
    %0 = pop.variant.discr_gep %arg0 : <variant<i32, i64>> as <scalar<ui8>>
    ```
  }];

  let arguments = (ins PointerOf<KGEN_VariantType>:$variant);
  let results = (outs PointerOf<ScalarOfDType<UnboundOr<UIntDType>,
                                              "uint scalar">>:$discr);
  let assemblyFormat = [{
    $variant attr-dict `:` type($variant) `as` type($discr)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    struct Payload { int32_t offset; };
    ErrorOrSuccess compile(Payload &payload, TargetInfoAttr target);
    ErrorOrSuccess parametric_compile(Payload &payload,
      TargetInfoAttr target,
      ArrayRef<Attribute> operands,
      ParametricInterpreterState& state);
    ErrorTreeOrSuccess interpret(ArrayRef<Attribute> operands,
      const Payload &payload, InterpreterState &state);
    ErrorTreeOrSuccess parametric_interpret(ArrayRef<Attribute> operands,
      const Payload &payload, ParametricInterpreterState &state);
  }];
}

//===----------------------------------------------------------------------===//
// UnionBitcastOp
//===----------------------------------------------------------------------===//

def POP_UnionBitcastOp : POPOp<"union.bitcast", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "bitcast a union pointer to an element type";
  let description = [{
    The `pop.union.bitcast` operation casts a pointer of a union to a pointer of
    one of its element types. This operation guarantees reading and writing of
    the resultant pointer to be inbounds for the underlying union value.

    Example:

    ```mlir
    %0 = pop.union.bitcast %arg0 : <union<i32, i64>> to <i64>
    ```
  }];

  let arguments = (ins PointerOf<POP_UnionType>:$value);
  let results = (outs PointerType:$result);
  let assemblyFormat = "$value attr-dict `:` type($value) `as` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// UnionWrapOp
//===----------------------------------------------------------------------===//

def POP_UnionWrapOp : POPOp<"union.wrap", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "wrap a value as a union value";
  let description = [{
    The `pop.union.wrap` operation wraps a value into a union type where the
    value type is one of the union element types.

    Example:

    ```mlir
    %0 = pop.union.wrap %arg0 : i32 as !pop.union<i32, i64>
    ```
  }];

  let arguments = (ins AnyType:$value);
  let results = (outs POP_UnionType:$result);
  let assemblyFormat = "$value attr-dict `:` type($value) `as` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// UnionUnwrapOp
//===----------------------------------------------------------------------===//

def POP_UnionUnwrapOp : POPOp<"union.unwrap", [Pure,
    DeclareOpInterfaceMethods<InterpreterOpInterface>]> {
  let summary = "unwrap a union as one of its element types";
  let description = [{
    The `pop.union.unwrap` operation takes a union and converts it to a value of
    one of its element types.

    Example:

    ```mlir
    %0 = pop.union.unwrap %arg0 : !pop.union<i32, i64> as i64
    ```
  }];

  let arguments = (ins POP_UnionType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$value attr-dict `:` type($value) `as` type($result)";
  let hasVerifier = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// SIMDReduceOrOp
//===----------------------------------------------------------------------===//

class POP_SIMDIntReduceOp<string name> : POPOp<"simd.reduce_" # !tolower(name),
      [Pure, SIMDSizeOneOfSameDType<"vector", "result">]> {
  let summary = "SIMD '" # name # "' Reduction";
  let description = !strconcat(
    "The `pop.simd.reduce_", !tolower(name), "` op performs a SIMD '", name,
    "' reduction over the vector.\n",
    "The dtype must be a boolean or integral, or parametric type with \n",
    "underlying boolean or integral dtype.\n",
    "\n",
    "```mlir\n",
    "%arg = ... : !kgen.simd<4, i32>\n",
    "%x = pop.simd.reduce_" # !tolower(name) # " %arg : !kgen.scalar<i32>\n",
    "```"
  );

  let arguments = (ins KGEN_SIMDType:$vector);
  let results = (outs ScalarType:$result);

  let assemblyFormat = "$vector attr-dict `:` qualified(type($vector))";
  let hasFolder = 1;
}

def POP_SIMDReduceOrOp : POP_SIMDIntReduceOp<"Or">;
def POP_SIMDReduceAndOp : POP_SIMDIntReduceOp<"And">;

#endif // KGEN_POPDIALECT_POPOPS_TD
