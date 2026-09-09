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

#ifndef KGEN_POPDIALECT_POPATTRS_TD
#define KGEN_POPDIALECT_POPATTRS_TD

include "Mojo/KGENDialect/KGENAttrInterfaces.td"
include "Mojo/POPDialect/POPDialect.td"
include "Mojo/POPDialect/POPEnums.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"

class POP_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<POPDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

/// The base class of all parameter expression attributes. Expression attributes
/// are all typed attributes.
class POP_ExprAttr<string attrName, string attrMnemonic,
                    list<Trait> traits = []>
    : POP_Attr<attrName, attrMnemonic,
               [TypedAttrInterface,
                DeclareAttrInterfaceMethods<ParameterAttrInterface>] # traits>;

//===----------------------------------------------------------------------===//
// UnionAttr
//===----------------------------------------------------------------------===//

def POP_UnionAttr : POP_Attr<"Union", "union", [TypedAttrInterface]> {
  let summary = "a union with a specific typed value";
  let description = [{
    The `#pop.union` attribute represents a union with a value of one of the
    union's element types. The attribute knows which union element the value is
    and contains an attribute of said value type.

    Example:

    ```mlir
    #pop.union<0> : !pop.union<index, struct<()>>
    #pop.union<:i64 42> : !pop.union<i64, none>
    ```
  }];

  let parameters = (ins "TypedAttr":$value,
                         AttributeSelfTypeParameter<"", "UnionType">:$type);
  let assemblyFormat = "`<` custom<ColonTypeParamValue>($value) `>`";
  let genVerifyDecl = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins "TypedAttr":$value, "UnionType":$type), [{
      return $_get(type.getContext(), value, type);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// ArrayAttr
//===----------------------------------------------------------------------===//

def POP_ArrayAttr : POP_ExprAttr<"Array", "array"> {
  let summary = "An array attribute.";
  let description = [{
    The `#pop.array` attribute contains a homogeneous list of elements of an
    array type. It is used to represent array constant values.

    Example:

    ```mlir
    #pop.array<1, 2> : !pop.array<2, index>
    ```
  }];

  let parameters = (ins OptionalArrayRefParameter<"TypedAttr">:$values,
                        AttributeSelfTypeParameter<"", "ArrayType">:$type);

  let assemblyFormat = [{
    `<` (`>`) : (custom<SequenceElements>($values, ref($type))^ `>`)?
  }];

  let builders = [
    AttrBuilderWithInferredContext<(ins "ArrayRef<TypedAttr>":$values,
                                        "ArrayType":$type), [{
      return get(type.getContext(), values, type);
    }]>
  ];


  let extraClassDeclaration = [{
    /// Construct an ArrayAttr where the elements have the specified type. This
    /// is a convenience for having to set up the ArrayType.
    static ArrayAttr getWithElementType(ArrayRef<TypedAttr> values,
                                        Type elementType) {
      return get(values, ArrayType::get(values.size(), elementType));
    }
  }];

  let genVerifyDecl = 1;
}


//===----------------------------------------------------------------------===//
// IntLiteralAttr
//===----------------------------------------------------------------------===//

def POP_IntLiteralAttr : POP_ExprAttr<"IntLiteral", "int_literal"> {
  let summary = "Attribute for holding infinite precision integer literals.";
  let description = [{
    The `#pop.int_literal` represents an integer literal, storing its
    data in an IPInt.
  }];

  let parameters = (ins "IPInt":$value);

  let assemblyFormat = "`<` custom<IntLiteral>($value) `>`";

  let extraClassDeclaration = [{
    Type getType() const;
  }];

  let builders = [
    AttrBuilder<(ins "int64_t":$value), [{
      return $_get($_ctxt, IPInt(value));
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// IntLiteralConvertAttr
//===----------------------------------------------------------------------===//

def POP_IntLiteralConvertAttr : POP_ExprAttr<"IntLiteralConvert",
                                             "int_literal_convert", [
    DeclareAttrInterfaceMethods<ParameterAttrInterface,
                                ["validateForElaborator"]>]> {
  let summary = "Convert an integer literal to a different type.";
  let description = [{
    The `#pop.int_literal_convert` attribute is used to convert an integer
    literal to a different SIMD type.
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"", "Type">:$type, "TypedAttr":$input
  );

  let assemblyFormat = [{
    `<` custom<ParamValue>($input, "IntLiteralType::get($_ctxt)") `>`
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    // Override default builder so we can return a folded attribute.
    AttrBuilder<(ins "Type":$type, "TypedAttr":$input), "", "TypedAttr">,
  ];
}

//===----------------------------------------------------------------------===//
// IntLiteralBinAttr
//===----------------------------------------------------------------------===//

def POP_IntLiteralBinAttr : POP_ExprAttr<"IntLiteralBin",
                                         "int_literal_bin"> {
  let summary = "Convert an integer literal to a different type.";
  let description = [{
    The `#pop.int_literal_binop` attr does standard arithmetic binary
    operations on `!pop.int_literal` values that return a `!pop.int_literal`.
  }];

  let parameters = (ins
    "TypedAttr":$lhs, "TypedAttr":$rhs,
    "IntLiteralBinKindAttr":$oper
  );

  let assemblyFormat = "`<` $oper $lhs `,` $rhs `>`";

  let skipDefaultBuilders = 1;
  let builders = [
    // Override default builder so we can return a folded attribute.
    AttrBuilder<(ins "TypedAttr":$lhs, "TypedAttr":$rhs,
                     "IntLiteralBinKindAttr":$oper), "", "TypedAttr">,
  ];

  let extraClassDeclaration = [{ Type getType() const; }];
}

//===----------------------------------------------------------------------===//
// IntLiteralCmpAttr
//===----------------------------------------------------------------------===//

def POP_IntLiteralCmpAttr : POP_ExprAttr<"IntLiteralCmp", "int_literal_cmp"> {
  let summary = "Comparison operator for int_literals.";
  let description = [{
    The `#pop.int_literal_cmp` attr does comparison testing
    operations on `!pop.int_literal` values that return an `i1`.
  }];

  let parameters = (ins
    "IntLiteralCmpPredAttr":$pred,
    "TypedAttr":$lhs, "TypedAttr":$rhs
  );

  let assemblyFormat = [{
    `<` $pred custom<ParamValue>($lhs, "IntLiteralType::get($_ctxt)") `,`
       custom<ParamValue>($rhs, "IntLiteralType::get($_ctxt)") `>`
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return an IntLiteralConvertAttr.
    AttrBuilder<(ins "IntLiteralCmpPredAttr":$pred, "TypedAttr":$lhs,
                     "TypedAttr":$rhs), "", "TypedAttr">,
  ];

  let extraClassDeclaration = [{ Type getType() const; }];
}

//===----------------------------------------------------------------------===//
// FloatLiteralAttr
//===----------------------------------------------------------------------===//

def POP_FloatLiteralAttr : POP_ExprAttr<"FloatLiteral", "float_literal"> {
  let summary = "Attribute for holding infinite precision float literals.";
  let description = [{
    The `#pop<float_literal` represents an float literal, though it
    is actually stored as a rational number IPRational.

    The `special` field tells whether it is a normal number or one of
    the special values (infinities, negative zero, or NAN).  When it
    is a non-normal number, it does not include a rational number
    value.

    TODO(#23387): MLIR's AsmParser doesn't have `parseSlash` or a more
    generic way to parse literal strings/characters, so we use
    the pipe "|" character instead for rationals.

    Example:

    ```mlir
    #pop.float_literal<normal(5|3)>
    #pop.float_literal<neg_zero>
    #pop.float_literal<inf>
    #pop.float_literal<neg_inf>
    #pop.float_literal<nan>
    ```
  }];

  let parameters = (ins
    POP_FloatLiteralSpecialValuesAttr:$special,
    "std::optional<IPRational>":$rational
  );

  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;
  let builders = [
    // Override default builder so we can canonicalize.
    AttrBuilder<(ins "FloatLiteralSpecialValuesAttr":$input,
                     "std::optional<IPRational>":$rational)>,
    // Always a normal value, set the SpecialValue to Normal.
    AttrBuilder<(ins "IPRational":$value)>,
  ];

  let extraClassDeclaration = [{
    Type getType() const;
  }];
}

//===----------------------------------------------------------------------===//
// FloatLiteralIsaAttr
//===----------------------------------------------------------------------===//

def POP_FloatLiteralIsaAttr : POP_ExprAttr<"FloatLiteralIsa",
                                           "float_literal_isa"> {
  let summary = "Check if the float_literal is of the given special value.";
  let description = [{
    The `#pop<float_literal_isa` attribute is used to check if the float
    literal is of the given special value.
  }];

  let parameters = (ins
    "FloatLiteralSpecialValuesAttr":$kind,
    "TypedAttr":$input
  );

  let assemblyFormat = [{
    `<` $kind custom<ParamValue>($input, "FloatLiteralType::get($_ctxt)") `>`
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    // Override default builder so we can return a folded attribute.
    AttrBuilder<(ins "FloatLiteralSpecialValuesAttr":$kind,
                     "TypedAttr":$input), "", "TypedAttr">,
  ];

  let extraClassDeclaration = [{
    Type getType() const;
  }];
}

//===----------------------------------------------------------------------===//
// FloatLiteralConvertAttr
//===----------------------------------------------------------------------===//

def POP_FloatLiteralConvertAttr : POP_ExprAttr<"FloatLiteralConvert",
                                               "float_literal_convert", [
    DeclareAttrInterfaceMethods<ParameterAttrInterface,
                                ["validateForElaborator"]>]> {
  let summary = "Convert an float literal to a different type.";
  let description = [{
    The `#pop<float_literal_convert` attribute is used to convert an float
    literal to a different type.
  }];

  let parameters = (ins
    AttributeSelfTypeParameter<"", "Type">:$type,
    "TypedAttr":$input
  );

  let assemblyFormat = [{
    `<` custom<ParamValue>($input, "FloatLiteralType::get($_ctxt)") `>`
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    // Override default builder so we can return a folded attribute.
    AttrBuilder<(ins "Type":$type, "TypedAttr":$input), "", "TypedAttr">,
  ];
}

//===----------------------------------------------------------------------===//
// IntToFloatLiteralAttr
//===----------------------------------------------------------------------===//

def POP_IntToFloatLiteralAttr : POP_ExprAttr<"IntToFloatLiteral",
                                             "int_to_float_literal"> {
  let summary = "Convert an int literal to an float literal";
  let description = [{
    The `#pop.int_to_float_literal` attr converts an int literal to an
    float literal.
  }];

  let parameters = (ins "TypedAttr":$input);

  let assemblyFormat = [{
    `<` custom<ParamValue>($input, "IntLiteralType::get($_ctxt)") `>`
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    // Override default builder so we can return a folded attribute.
    AttrBuilder<(ins "TypedAttr":$input), "", "TypedAttr">,
  ];
  let extraClassDeclaration = [{ Type getType() const; }];
}

//===----------------------------------------------------------------------===//
// FloatToIntLiteralAttr
//===----------------------------------------------------------------------===//

def POP_FloatToIntLiteralAttr : POP_ExprAttr<"FloatToIntLiteral",
                                             "float_to_int_literal"> {
  let summary = "Convert an float literal to an int literal.";
  let description = [{
    The `#pop<float_to_int_literal` attr converts an float literal to an
    int literal.
  }];

  let parameters = (ins
    "TypedAttr":$input
  );

  let assemblyFormat = [{
    `<` custom<ParamValue>($input, "FloatLiteralType::get($_ctxt)") `>`
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return an FloatToIntLiteralAttr.
    AttrBuilder<(ins "TypedAttr":$input), "", "TypedAttr">,
  ];

  let extraClassDeclaration = [{ Type getType() const; }];
}

//===----------------------------------------------------------------------===//
// FloatLiteralBinAttr
//===----------------------------------------------------------------------===//

def POP_FloatLiteralBinAttr : POP_ExprAttr<"FloatLiteralBin",
                                           "float_literal_bin"> {
  let summary = "Convert an float literal to a different type.";
  let description = [{
    The `#pop<float_literal_bin` attr does standard arithmetic binary
    operations on `!pop.float_literal` values that return a `!pop.float_literal`.
  }];

  let parameters = (ins
    "TypedAttr":$lhs, "TypedAttr":$rhs,
    "FloatLiteralBinKindAttr":$oper
  );

  let assemblyFormat = [{
    `<` $oper custom<ParamValue>($lhs, "FloatLiteralType::get($_ctxt)") `,`
       custom<ParamValue>($rhs, "FloatLiteralType::get($_ctxt)") `>`
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return an IntLiteralConvertAttr.
    AttrBuilder<(ins "TypedAttr":$lhs, "TypedAttr":$rhs,
                     "FloatLiteralBinKindAttr":$oper), "", "TypedAttr">,
  ];

  let extraClassDeclaration = [{ Type getType() const; }];
}

//===----------------------------------------------------------------------===//
// FloatLiteralCmpAttr
//===----------------------------------------------------------------------===//

def POP_FloatLiteralCmpAttr : POP_ExprAttr<"FloatLiteralCmp",
                                           "float_literal_cmp"> {
  let summary = "Comparison operator for float literals.";
  let description = [{
    The `#pop<float_literal_cmp` attr does comparison testing
    operations on `!pop.float_literal` values that return an `i1`.
  }];

  let parameters = (ins
    "FloatLiteralCmpPredAttr":$pred,
    "TypedAttr":$lhs, "TypedAttr":$rhs
  );

  let assemblyFormat = [{
    `<` $pred custom<ParamValue>($lhs, "FloatLiteralType::get($_ctxt)") `,`
       custom<ParamValue>($rhs, "FloatLiteralType::get($_ctxt)") `>`
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return an IntLiteralConvertAttr.
    AttrBuilder<(ins "FloatLiteralCmpPredAttr":$pred, "TypedAttr":$lhs,
                     "TypedAttr":$rhs), "", "TypedAttr">,
  ];

  let extraClassDeclaration = [{ Type getType() const; }];
}

//===----------------------------------------------------------------------===//
// StringSizeAttr
//===----------------------------------------------------------------------===//

def POP_StringSizeAttr : POP_ExprAttr<"StringSize", "string_size"> {
  let summary = "Get the size of a string.";
  let description = [{
    The `#pop.string_size` attribute gets the size of a !kgen.string value.
  }];

  let parameters = (ins "TypedAttr":$str);

  let assemblyFormat = [{
    `<` custom<ParamValue>($str, "StringType::get($_ctxt)") `>`
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return a StringSizeAttr.
    AttrBuilder<(ins "TypedAttr":$str), "", "TypedAttr">,
  ];

  let extraClassDeclaration = [{ Type getType() const; }];
}

//===----------------------------------------------------------------------===//
// StringConcatAttr
//===----------------------------------------------------------------------===//

def POP_StringConcatAttr : POP_ExprAttr<"StringConcat", "string_concat"> {
  let summary = "Concatenate two strings.";
  let description = [{
    The `#pop.string_concat` attribute concatenates two !kgen.string values.
  }];

  let parameters = (ins "TypedAttr":$lhs, "TypedAttr":$rhs);

  let assemblyFormat = [{
    `<` custom<ParamValue>($lhs, "StringType::get($_ctxt)") `,`
       custom<ParamValue>($rhs, "StringType::get($_ctxt)") `>`
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return an StringConcatAttr.
    AttrBuilder<(ins "TypedAttr":$lhs, "TypedAttr":$rhs), "", "TypedAttr">,
  ];

  let extraClassDeclaration = [{ Type getType() const; }];
}

//===----------------------------------------------------------------------===//
// Cast
//===----------------------------------------------------------------------===//

def POP_CastAttr : POP_ExprAttr<"Cast", "cast",
    [ContextuallyEvaluatedAttrInterface]> {
  let summary = "Convert between POP types.";
  let description = [{
    The `#pop.cast` attribute converts a POP type to another POP type.
  }];

  let parameters = (ins "TypedAttr":$arg,
                        AttributeSelfTypeParameter<"", "Type">:$type);

  let assemblyFormat = " `<` $arg `>` ";

  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return a CastAttr.
    AttrBuilder<(ins "TypedAttr":$arg, "Type":$type), "", "TypedAttr">,

    AttrBuilderWithInferredContext<(ins "TypedAttr":$arg, "Type":$type), [{
      return get(arg.getContext(), arg, type);
    }], "TypedAttr">
  ];
}

//===----------------------------------------------------------------------===//
// DTypeToUI8Attr
//===----------------------------------------------------------------------===//

def POP_DTypeToUI8Attr : POP_ExprAttr<"DTypeToUI8", "dtype_to_ui8"> {
  let summary = "Convert a DType to UInt8.";
  let description = [{
    The `#pop.dtype_to_ui8` attribute converts a !kgen.dtype to a ui8 type.
  }];

  let parameters = (ins "TypedAttr":$value);

  let assemblyFormat = "`<` custom<DTypeParamValue>($value) `>`";

  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return an DTypeToUI8Attr.
    AttrBuilder<(ins "TypedAttr":$value), "", "TypedAttr">,

    AttrBuilderWithInferredContext<(ins "TypedAttr":$value), [{
      return get(value.getContext(), value);
    }], "TypedAttr">
  ];

  let extraClassDeclaration = [{ Type getType() const; }];
}

//===----------------------------------------------------------------------===//
// DTypeFromUI8Attr
//===----------------------------------------------------------------------===//

def POP_DTypeFromUI8Attr : POP_ExprAttr<"DTypeFromUI8", "dtype_from_ui8"> {
  let summary = "Convert a UInt8 to DType.";
  let description = [{
    The `#pop.dtype_from_ui8` attribute converts a ui8 type to a !kgen.dtype.
  }];

  let parameters = (ins "TypedAttr":$value);

  let assemblyFormat = "`<` $value `>`";

  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return an DTypeFromUI8Attr.
    AttrBuilder<(ins "TypedAttr":$value), "", "TypedAttr">,

    AttrBuilderWithInferredContext<(ins "TypedAttr":$value), [{
      return get(value.getContext(), value);
    }], "TypedAttr">
  ];

  let extraClassDeclaration = [{ Type getType() const; }];
}


//===----------------------------------------------------------------------===//
// SIMD Unary Operation Attributes
//===----------------------------------------------------------------------===//

class POP_SIMDUnaryAttr<string name, list<Trait> extraTraits = []>
      : POP_ExprAttr<"SIMD" # name, "simd_" # !tolower(name), extraTraits> {
  let summary = !strconcat("SIMD ", name, ".");

  let parameters = (ins "TypedAttr":$operand);

  let assemblyFormat = " `<` $operand `>` ";

  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return the same attribute type due to folding on construction.
    AttrBuilder<(ins "TypedAttr":$operand), "", "TypedAttr">,

    AttrBuilderWithInferredContext<(ins "TypedAttr":$operand), [{
      return get(operand.getContext(), operand);
    }], "TypedAttr">
  ];

  let extraClassDeclaration = [{
    Type getType() const { return getOperand().getType(); }
  }];

  let extraClassDefinition = [{
    bool $cppClass::isConstant() const { return false; }

    LogicalResult $cppClass::verify(function_ref<InFlightDiagnostic()> emitError,
                                    TypedAttr operand) {
      if (!isa<SIMDType>(operand.getType()))
        return emitError() << "requires a SIMD-typed operand";
      return success();
    }

    TypedAttr $cppClass::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                    MLIRContext *context, TypedAttr operand) {
      if (failed(verify(emitError, operand)))
        return {};
      return $cppClass::get(context, operand);
    }
  }];
}

def POP_SIMDNegAttr : POP_SIMDUnaryAttr<"Neg"> {
  let description = [{
    The `#pop.simd_neg` attribute takes a scalar value and returns the negation
    of it.
  }];
}

def POP_SIMDFloorAttr : POP_SIMDUnaryAttr<"Floor"> {
  let description = [{
    The `#pop.simd_floor` attribute takes a scalar value and returns the floor
    of it.
  }];
}

def POP_SIMDCeilAttr : POP_SIMDUnaryAttr<"Ceil"> {
  let description = [{
    The `#pop.simd_ceil` attribute takes a scalar value and returns the ceiling
    of it.
  }];
}

def POP_SIMDTruncAttr : POP_SIMDUnaryAttr<"Trunc"> {
  let description = [{
    The `#pop.simd_trunc` attribute takes a scalar value and returns the
    truncation of it (rounding toward zero).
  }];
}

def POP_SIMDAbsAttr : POP_SIMDUnaryAttr<"Abs",
    [DeclareAttrInterfaceMethods<ContextuallyEvaluatedAttrInterface,
                                 ["evaluateWithContext"]>]> {
  let description = [{
    The `#pop.simd_abs` attribute takes a SIMD value and returns the
    absolute value of it.
  }];
}

def POP_SIMDRoundAttr : POP_SIMDUnaryAttr<"Round"> {
  let description = [{
    The `#pop.simd_round` attribute takes a SIMD value and returns it rounded
    to the nearest integer, with ties rounding to the nearest even.
  }];
}

//===----------------------------------------------------------------------===//
// SIMD Binary Operation Attributes
//===----------------------------------------------------------------------===//

class POP_SIMDBinaryAttr<string name, list<Trait> traits = []>
      : POP_ExprAttr<"SIMD" # name, "simd_" # !tolower(name), traits> {
  let summary = !strconcat("SIMD ", name, ".");
  let description = !strconcat(
    "The `#pop.simd_", !tolower(name), "` attribute performs a SIMD ", name,
    " operation between two SIMD vectors.");

  let parameters = (ins "TypedAttr":$lhs, "TypedAttr":$rhs);

  let assemblyFormat = " `<` $lhs `,` $rhs `>` ";

  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return the same attribute type due to folding on construction.
    AttrBuilder<(ins "TypedAttr":$lhs, "TypedAttr":$rhs), "", "TypedAttr">,

    AttrBuilderWithInferredContext<(ins "TypedAttr":$lhs, "TypedAttr":$rhs), [{
      return get(lhs.getContext(), lhs, rhs);
    }], "TypedAttr">
  ];

  let extraClassDeclaration = [{
    Type getType() const { return getLhs().getType(); };
  }];

  let extraClassDefinition = [{
    bool $cppClass::isConstant() const { return false; }

    LogicalResult $cppClass::verify(function_ref<InFlightDiagnostic()> emitError,
                                    TypedAttr lhs, TypedAttr rhs) {
      if (lhs.getType() != rhs.getType() || !isa<SIMDType>(lhs.getType()))
        return emitError() << "requires two equally-typed SIMD operands";
      return success();
    }

    TypedAttr $cppClass::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                    MLIRContext *context, TypedAttr lhs,
                                    TypedAttr rhs) {
      if (failed(verify(emitError, lhs, rhs)))
        return {};
      return $cppClass::get(context, lhs, rhs);
    }
  }];
}

// Add, Mul, Div, FloorDiv, And, Or, Xor and the comparisons are all created
// directly as `#kgen.param.expr` (see `ParamOperatorAttr`), so only the
// operators without an equivalent `POC` opcode need a dedicated attribute.
def POP_SIMDSubAttr : POP_SIMDBinaryAttr<"Sub">;

//===----------------------------------------------------------------------===//
// SIMD Shift Operation Attributes
//===----------------------------------------------------------------------===//

class POP_SIMDShiftAttr<string name>
      : POP_ExprAttr<"SIMD" # name, "simd_" # !tolower(name),
          [DeclareAttrInterfaceMethods<ContextuallyEvaluatedAttrInterface,
                                       ["evaluateWithContext"]>]> {
  let parameters = (ins "TypedAttr":$val, "TypedAttr":$shft);

  let assemblyFormat = " `<` $val `,` $shft `>` ";

  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return the same attribute type due to folding on construction.
    AttrBuilder<(ins "TypedAttr":$val, "TypedAttr":$shft), "", "TypedAttr">,

    AttrBuilderWithInferredContext<(ins "TypedAttr":$val, "TypedAttr":$shft), [{
      return get(val.getContext(), val, shft);
    }], "TypedAttr">
  ];

  let extraClassDeclaration = [{
    Type getType() const { return getVal().getType(); }
  }];

  let extraClassDefinition = [{
    bool $cppClass::isConstant() const { return false; }

    TypedAttr $cppClass::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                    MLIRContext *context, TypedAttr value,
                                    TypedAttr shft) {
      if (failed(verify(emitError, value, shft)))
        return {};
      return $cppClass::get(context, value, shft);
    }

    LogicalResult $cppClass::verify(function_ref<InFlightDiagnostic()> emitError,
                                    TypedAttr value, TypedAttr shft) {
      if (!isa<SIMDType>(value.getType()))
        return emitError() << "requires a SIMD-type operand";

      return success();
    }
  }];
}

def POP_SIMDShlAttr : POP_SIMDShiftAttr<"Shl"> {
  let summary = "SIMD Left Shift.";
  let description = [{
    The `pop.simd_shl` attribute returns the first operand shifted to the left
    a specified number of bits. Both arguments to `pop.shl` must be the same
    integer or index dtype. The shift amount is always treated as unsigned.

    If the shift amount is equal to or larger than the number of bits in the
    value, the results of this operation is a poison value.
  }];
}

def POP_SIMDShrAttr : POP_SIMDShiftAttr<"Shr"> {
  let summary = "SIMD Right Shift.";
  let description = [{
    The `pop.shr` attribute returns the first operand shifted to the right a
    specified number of bits. Whether the result is sign extended or zero
    extended depends on the result dtype. Both arguments to the `shr` op must be
    the same integer/index or vector of integer/index type. The shift amount is
    always treated as unsigned.

    This operation performs an arithmetic shift right operation if the dtype is
    signed. The most significant bits of the result will be filled with the
    sign bit of the value to shift. This operation performs a logical shift
    right operation if the dtype is unsigned. The most significant bits of the
    result will be filled with zero bits after the shift.

    If the shift amount is equal to or larger than the number of bits in the
    shift value, the result of this operation is a poison value. If the
    arguments are vectors, each vector element of the shifted value is shifted
    by the corresponding shift amount in the shift amount.
  }];
}

class POP_SIMDReduceAttr<string name>
      : POP_ExprAttr<"SIMDReduce" # name, "simd_reduce_" # !tolower(name)> {
  let summary = "SIMD '" # name # "' reduction.";
  let description = !strconcat(
    "The `#pop.simd_reduce_" , !tolower(name), "` attribute performs a ",
    "bitwise '", name, "' reduction over a SIMD vector."
  );

  let parameters = (ins "TypedAttr":$vector,
                        AttributeSelfTypeParameter<"", "SIMDType">:$type);

  let assemblyFormat = " `<` $vector `>` ";

  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    // Override the default builder used by the generated parser because this
    // may not return an identically-typed attribute.
    AttrBuilder<(ins "TypedAttr":$vector, "SIMDType":$type), "", "TypedAttr">,

    AttrBuilderWithInferredContext<(ins "TypedAttr":$vector, "SIMDType":$type),
    [{ return get(vector.getContext(), vector, type); }], "TypedAttr">
  ];

  let extraClassDeclaration = [{
    static OpFoldResult fold(TypedAttr vector, SIMDType outType);
  }];

  let extraClassDefinition = [{
    bool $cppClass::isConstant() const { return false; }

    TypedAttr $cppClass::get(MLIRContext *ctx, TypedAttr vector,
                             SIMDType outType) {
      // Fold if possible
      if (auto foldRes = fold(vector, cast<SIMDType>(vector.getType()))) {
        if (auto ret = dyn_cast<TypedAttr>(cast<Attribute>(foldRes)))
          return ret;
      }
      return Base::get(ctx, vector, outType);
    }

    TypedAttr $cppClass::getChecked(function_ref<InFlightDiagnostic()> emitError,
                                    MLIRContext *context, TypedAttr vector,
                                    SIMDType outType) {
      if (failed(verify(emitError, vector, outType)))
        return {};
      return $cppClass::get(context, vector, outType);
    }

    LogicalResult $cppClass::verify(function_ref<InFlightDiagnostic()> emitError,
                                    TypedAttr vector, SIMDType outType) {
      if (!isa<SIMDType>(vector.getType()))
        return emitError() << "requires a SIMD-type operand";

      return success();
    }
  }];
}

def POP_SIMDReduceOrAttr : POP_SIMDReduceAttr<"Or">;
def POP_SIMDReduceAndAttr : POP_SIMDReduceAttr<"And">;

//===----------------------------------------------------------------------===//
// VariadicToArrayAttr
//===----------------------------------------------------------------------===//

def POP_VariadicToArrayAttr : POP_ExprAttr<"VariadicToArray", "variadic_to_array"> {
  let summary = "Convert a variadic to an array.";
  let description = [{
    The `#pop.variadic_to_array` attribute converts a `!kgen.param_list` into a
    `!pop.array` whose length equals the variadic size and whose elements are
    the variadic's elements.

    Example:

    ```mlir
    #pop.variadic_to_array<:!kgen.param_list<index> #kgen.param_list<1, 2, 3>>
      : !pop.array<3, index>
    ```
  }];

  let parameters = (ins "TypedAttr":$variadic);

  let assemblyFormat = "`<`  custom<ColonTypeParamValue>($variadic) `>`";
  let skipDefaultBuilders = 1;
  let genVerifyDecl = 1;
  let builders = [
    AttrBuilderWithInferredContext<(ins "TypedAttr":$variadic), "",
                                   "TypedAttr">,
    AttrBuilder<(ins "TypedAttr":$variadic), [{
      return get(variadic);
    }], "TypedAttr">,
  ];

  let extraClassDeclaration = [{
    POP::ArrayType getType() const;
    Type getElementType() const;
  }];
}

#endif // KGEN_POPDIALECT_POPATTRS_TD
