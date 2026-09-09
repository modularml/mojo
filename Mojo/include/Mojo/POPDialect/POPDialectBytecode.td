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

#ifndef KGEN_POPDIALECT_POPDIALECTBYTECODE_TD
#define KGEN_POPDIALECT_POPDIALECTBYTECODE_TD

include "Support/Compiler/Bytecode.td"

//===----------------------------------------------------------------------===//
// POPDialect Types
//===----------------------------------------------------------------------===//

def ArrayType : DialectType<(type
  TypedAttr:$size,
  Type:$elementType
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def UnionType : DialectType<(type
  TypedAttr:$variadic
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def IntLiteralType : DialectType<(type)>;

def FloatLiteralType : DialectType<(type)>;

def POPDialectTypes : DialectTypes<"pop"> {
  let elems = [
    ArrayType,
    UnionType,
    IntLiteralType,
    FloatLiteralType
  ];
}

//===----------------------------------------------------------------------===//
// POPDialect Attributes
//===----------------------------------------------------------------------===//

def AtomicOrderingAttr : DialectAttribute<(attr
  EnumAttr<"AtomicOrdering">:$value
)>;

def AtomicBinOpAttr : DialectAttribute<(attr
  EnumAttr<"AtomicBinOp">:$value
)>;

def FastmathFlagsAttr : DialectAttribute<(attr
  EnumAttr<"FastmathFlags">:$value
)>;

def UnionAttr : DialectAttribute<(attr
  TypedAttr:$value,
  ConcreteType<"UnionType">:$type
)>;

def ArrayAttr : DialectAttribute<(attr
  TypedAttrArray:$values,
  ConcreteType<"ArrayType">:$type
)>;

def IntLiteralAttr : DialectAttribute<(attr
  Custom<"IPInt">:$value
)>;

def IntLiteralConvertAttr : DialectAttribute<(attr
  Type:$type,
  TypedAttr:$input
)>;

def IntLiteralBinAttr : DialectAttribute<(attr
  TypedAttr:$lhs, TypedAttr:$rhs,
  ConcreteAttribute<"IntLiteralBinKindAttr">:$oper
)>;

def IntLiteralCmpAttr : DialectAttribute<(attr
  ConcreteAttribute<"IntLiteralCmpPredAttr">:$pred,
  TypedAttr:$lhs, TypedAttr:$rhs
)>;

def IntLiteralCmpPredAttr : DialectAttribute<(attr
  EnumAttr<"IntLiteralCmpPred">:$value
)>;

def IntLiteralBinKindAttr : DialectAttribute<(attr
  EnumAttr<"IntLiteralBinKind">:$value
)>;

def FloatLiteralAttr : DialectAttribute<(attr
  ConcreteAttribute<"FloatLiteralSpecialValuesAttr">:$special,
  Custom<"OptionalIPRational">:$rational
)>;

def FloatLiteralSpecialValuesAttr : DialectAttribute<(attr
  EnumAttr<"FloatLiteralSpecialValues">:$value
)>;

def FloatLiteralBinKindAttr : DialectAttribute<(attr
  EnumAttr<"FloatLiteralBinKind">:$value
)>;

def FloatLiteralCmpPredAttr : DialectAttribute<(attr
  EnumAttr<"FloatLiteralCmpPred">:$value
)>;

def FloatLiteralIsaAttr : DialectAttribute<(attr
  ConcreteAttribute<"FloatLiteralSpecialValuesAttr">:$kind,
  TypedAttr:$input
)>;

def FloatLiteralConvertAttr : DialectAttribute<(attr
  Type:$type,
  TypedAttr:$input
)>;

def IntToFloatLiteralAttr : DialectAttribute<(attr TypedAttr:$input)>;

def FloatToIntLiteralAttr : DialectAttribute<(attr TypedAttr:$input)>;

def FloatLiteralBinAttr : DialectAttribute<(attr
  TypedAttr:$lhs, TypedAttr:$rhs,
  ConcreteAttribute<"FloatLiteralBinKindAttr">:$oper
)>;

def FloatLiteralCmpAttr : DialectAttribute<(attr
  ConcreteAttribute<"FloatLiteralCmpPredAttr">:$pred,
  TypedAttr:$lhs, TypedAttr:$rhs
)>;

def StringSizeAttr : DialectAttribute<(attr TypedAttr:$str)>;

def StringConcatAttr : DialectAttribute<(attr
  TypedAttr:$lhs, TypedAttr:$rhs
)>;

def GlobalAllocAddressSpaceAttr : DialectAttribute<(attr
  EnumAttr<"GlobalAllocAddressSpace">:$value
)>;

def SIMDSubAttr : DialectAttribute<(attr TypedAttr:$lhs, TypedAttr:$rhs)>;

def SIMDShlAttr : DialectAttribute<(attr TypedAttr:$val, TypedAttr:$shft)>;
def SIMDShrAttr : DialectAttribute<(attr TypedAttr:$val, TypedAttr:$shft)>;

def CastAttr : DialectAttribute<(attr
  TypedAttr:$arg, Type:$type
)>;

def DTypeToUI8Attr : DialectAttribute<(attr TypedAttr:$value)>;
def DTypeFromUI8Attr : DialectAttribute<(attr TypedAttr:$value)>;

def SIMDNegAttr : DialectAttribute<(attr TypedAttr:$operand)>;
def SIMDCeilAttr : DialectAttribute<(attr TypedAttr:$operand)>;
def SIMDFloorAttr : DialectAttribute<(attr TypedAttr:$operand)>;
def SIMDTruncAttr : DialectAttribute<(attr TypedAttr:$operand)>;
def SIMDAbsAttr : DialectAttribute<(attr TypedAttr:$operand)>;
def SIMDRoundAttr : DialectAttribute<(attr TypedAttr:$operand)>;

def SIMDReduceOrAttr : DialectAttribute<(attr
  TypedAttr:$vector, ConcreteType<"SIMDType">:$type
)>;
def SIMDReduceAndAttr : DialectAttribute<(attr
  TypedAttr:$vector, ConcreteType<"SIMDType">:$type
)>;

def VariadicToArrayAttr : DialectAttribute<(attr
  TypedAttr:$variadic
)>;

def POPDialectAttributes : DialectAttributes<"pop"> {
  let elems = [
    AtomicOrderingAttr,
    AtomicBinOpAttr,
    FastmathFlagsAttr,
    UnionAttr,
    ArrayAttr,

    IntLiteralCmpPredAttr,
    IntLiteralBinKindAttr,
    FloatLiteralSpecialValuesAttr,
    FloatLiteralCmpPredAttr,
    FloatLiteralBinKindAttr,

    IntLiteralAttr,
    FloatLiteralAttr,

    SIMDSubAttr,

    SIMDShlAttr,
    SIMDShrAttr,

    CastAttr,
    DTypeToUI8Attr,
    DTypeFromUI8Attr,

    SIMDNegAttr,
    SIMDCeilAttr,
    SIMDFloorAttr,
    SIMDTruncAttr,
    SIMDAbsAttr,
    SIMDRoundAttr,

    SIMDReduceOrAttr,
    SIMDReduceAndAttr,
    VariadicToArrayAttr,

    IntLiteralConvertAttr,
    IntLiteralBinAttr,
    IntLiteralCmpAttr,
    FloatLiteralIsaAttr,
    FloatLiteralConvertAttr,
    IntToFloatLiteralAttr,
    FloatToIntLiteralAttr,
    FloatLiteralBinAttr,
    FloatLiteralCmpAttr,
    StringSizeAttr,
    StringConcatAttr,
    GlobalAllocAddressSpaceAttr,
  ];
}

#endif // KGEN_POPDIALECT_POPDIALECTBYTECODE_TD
