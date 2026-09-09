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

#ifndef LITDIALECT_BYTECODE
#define LITDIALECT_BYTECODE

include "Support/Compiler/Bytecode.td"

//===----------------------------------------------------------------------===//
// LITDialect Types
//===----------------------------------------------------------------------===//

def TypeCheckErrorType : DialectType<(type)>;

def UnresolvedType : DialectType<(type)>;

def EllipsisType : DialectType<(type)>;

def ModuleType : DialectType<(type
  ConcreteAttribute<"SymbolRefAttr">:$symbol
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def TypeSignatureType : DialectType<(type
  TypeArray:$paramTypes,
  ConcreteAttribute<"::M::KGEN::PogListAttr">:$paramListAttrs
)>;

def StructType : DialectType<(type
  ConcreteAttribute<"SymbolRefAttr">:$symbol,
  TypedAttrArray:$paramValues,
  ConcreteType<"TypeSignatureType">:$signature
)>;

def TraitType : DialectType<(type
  ConcreteAttrArray<"::M::KGEN::TraitSymbolAttr">:$symbols,
  ConcreteAttrArray<"ConstraintAttr">:$constraints
)>;

def MetaType : DialectType<(type
  Type:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def OriginType : DialectType<(type
  TypedAttr:$isMutable
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def OriginSetType : DialectType<(type)>;

def RefType : DialectType<(type
  Type:$elementType,
  TypedAttr:$origin,
  TypedAttr:$addressSpace
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def RefPackType : DialectType<(type
  TypedAttr:$variadic,
  TypedAttr:$origin,
  TypedAttr:$addressSpace
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def REPLResultRefType : DialectType<(type
  Type:$elementType
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def MagicMLIRAttrType : DialectType<(type)>;

def MagicMLIROpType : DialectType<(type)>;

def MagicMLIRTypeType : DialectType<(type)>;

def MagicMLIRDeferredAttrType : DialectType<(type)>;

def NameLookupArgWildcardType : DialectType<(type)>;

def LITDialectTypes : DialectTypes<"lit"> {
  let elems = [
    TypeCheckErrorType,
    UnresolvedType,
    EllipsisType,
    TypeSignatureType,
    StructType,
    TraitType,
    MetaType,
    OriginType,
    OriginSetType,
    RefType,
    RefPackType,
    REPLResultRefType,
    MagicMLIRAttrType,
    MagicMLIROpType,
    MagicMLIRTypeType,
    MagicMLIRDeferredAttrType,
    NameLookupArgWildcardType
  ];
}

//===----------------------------------------------------------------------===//
// LITDialect Attributes
//===----------------------------------------------------------------------===//

def DenseResourceElementsArrayAttr : DialectAttribute<(attr
  ConcreteAttrArray<"DenseResourceElementsAttr">:$value
)>;

def SymbolAttr : DialectAttribute<(attr
  ConcreteAttribute<"SymbolRefAttr">:$value
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def FnMetaOriginDataAttr : DialectAttribute<(attr
  VarInt:$numImplicitOriginDecls,
  TypedAttr:$captureOrigins,
  Bool:$isNestedOriginsReadOnly,
  Bool:$definesInteriorOrigins
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def DeprecationInfoAttr : DialectAttribute<(attr
  StringAttr:$warning,
  OptionalAttribute<"StringAttr">:$replacement
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def UnavailableInfoAttr : DialectAttribute<(attr
  StringAttr:$reason,
  OptionalAttribute<"StringAttr">:$replacement
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def ImportPathAttr : DialectAttribute<(attr
  VarInt:$relativeLevel,
  ConcreteAttrArray<"StringAttr">:$components
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def DocStringAttr : DialectAttribute<(attr
  StringAttr:$string,
  OptionalAttribute<"FileLineColLoc">:$location
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def EllipsisAttr : DialectAttribute<(attr)>;

def UnboundMLIROperationAttr : DialectAttribute<(attr
  StringAttr:$name,
  ConcreteAttribute<"DictionaryAttr">:$attrs
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def UnpackedAttr : DialectAttribute<(attr
  TypedAttr:$value,
  Bool:$kwOnly,
  Type:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def ModuleAttr : DialectAttribute<(attr
  ConcreteType<"ModuleType">:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def AnyOriginAttr : DialectAttribute<(attr
  ConcreteType<"OriginType">:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def ComptimeOriginAttr : DialectAttribute<(attr
  ConcreteType<"OriginType">:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def StaticOriginAttr : DialectAttribute<(attr
  ConcreteType<"OriginType">:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def WrappedOriginUnionAttr : DialectAttribute<(attr
  TypedAttrArray:$operands,
  ConcreteType<"OriginType">:$type
)>;

def OriginEqAttr : DialectAttribute<(attr
  TypedAttr:$lhs,
  TypedAttr:$rhs
)>;

def WrappedOriginMutCastAttr : DialectAttribute<(attr
  TypedAttr:$operand,
  ConcreteType<"OriginType">:$type
)>;

def WrappedOriginFieldAttr : DialectAttribute<(attr
  TypedAttr:$base,
  StringAttr:$field,
  ConcreteType<"OriginType">:$type
)>;

def WrappedInteriorOriginAttr : DialectAttribute<(attr
  TypedAttr:$base,
  TypedAttr:$userName,
  ConcreteType<"OriginType">:$type
)>;

def WrappedOriginSubtreeAttr : DialectAttribute<(attr
  TypedAttr:$base,
  ConcreteType<"OriginType">:$type
)>;

def ImplicitOriginRefAttr : DialectAttribute<(attr
  VarInt:$depth,
  VarInt:$index,
  ConcreteType<"OriginType">:$type
)>;

def WrappedOriginSetAttr : DialectAttribute<(attr
  TypedAttrArray:$operands,
  ConcreteType<"OriginSetType">:$type
)>;

def WrappedOriginSetUnionAttr : DialectAttribute<(attr
  TypedAttr:$value,
  ConcreteType<"OriginType">:$type
)>;

def UnresolvedAliasValueAttr : DialectAttribute<(attr
  Type:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def LITStructAttr : DialectAttribute<(attr
  WithParser<"succeeded(readStructValues($_reader, $_var))",
  WithPrinter<"writeStructValues($_writer, $_getter)",
  WithType<"SmallVector<std::tuple<StringAttr, TypedAttr>>">>>:$values,
  ConcreteType<"StructType">:$type
)>;

def WrappedStructExtractAttr : DialectAttribute<(attr
  TypedAttr:$structValue,
  StringAttr:$field,
  Type:$type
)>;

def TypeConventionAttr : DialectAttribute<(attr
  EnumAttr<"TypeConvention">:$value
)>;

def VarDeclKindAttr : DialectAttribute<(attr
  EnumAttr<"VarDeclKind">:$value
)>;

def ImplicitConversionKindAttr : DialectAttribute<(attr
  EnumAttr<"ImplicitConversionKind">:$value
)>;

def RefPackAttr : DialectAttribute<(attr
  TypedAttrArray:$values,
  ConcreteType<"RefPackType">:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def LITDialectAttributes : DialectAttributes<"lit"> {
  let elems = [
    DenseResourceElementsArrayAttr,
    SymbolAttr,
    EllipsisAttr,
    FnMetaOriginDataAttr,
    DocStringAttr,
    UnboundMLIROperationAttr,
    UnpackedAttr,
    ModuleAttr,
    AnyOriginAttr,
    OriginEqAttr,
    WrappedOriginUnionAttr,
    WrappedOriginMutCastAttr,
    WrappedOriginFieldAttr,
    ImplicitOriginRefAttr,
    WrappedOriginSetAttr,
    WrappedOriginSetUnionAttr,
    UnresolvedAliasValueAttr,
    LITStructAttr,
    WrappedStructExtractAttr,
    RefPackAttr,
    TypeConventionAttr,
    VarDeclKindAttr,
    DeprecationInfoAttr,
    UnavailableInfoAttr,
    ComptimeOriginAttr,
    StaticOriginAttr,
    WrappedInteriorOriginAttr,
    WrappedOriginSubtreeAttr,
    ImplicitConversionKindAttr,
    ImportPathAttr,
  ];
}

#endif // LITDIALECT_BYTECODE
