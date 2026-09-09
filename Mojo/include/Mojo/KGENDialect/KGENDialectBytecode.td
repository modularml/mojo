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

#ifndef KGENDIALECT_BYTECODE
#define KGENDIALECT_BYTECODE

include "Support/Compiler/Bytecode.td"

//===----------------------------------------------------------------------===//
// KGENDialect Types
//===----------------------------------------------------------------------===//

def WrappedParamRefType : DialectType<(type
  TypedAttr:$param
)>;

def WrappedTypeValueType : DialectType<(type
  TypedAttr:$typeValue
)>;

def NonStructTypeType : DialectType<(type)>;

def TypeType : DialectType<(type)>;

def DTypeType : DialectType<(type)>;

def StringType : DialectType<(type)>;

def GeneratorType : DialectType<(type
  TypeArray:$inputParamTypes,
  Type:$body,
  ConcreteOptionalAttribute<"PogListAttr">:$paramListAttrs
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def FuncType : DialectType<(type
  ConcreteType<"FunctionType">:$values,
  ConcreteAttribute<"FnMetadataAttr">:$metadataAttr,
  ConcreteAttribute<"PogListAttr">:$argListAttrs
)>;

def FuncLiteralType : DialectType<(type
  TypedAttr:$funcLiteral
)>;

def PointerType : DialectType<(type
  Type:$elementType,
  TypedAttr:$addressSpace,
  Bool:$isNonNull
)>;

def ParamListType : DialectType<(type
  Type:$elementType
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def NoneType : DialectType<(type)>;

def TargetType : DialectType<(type)>;

def BuildInfoType : DialectType<(type)>;

def StructType : DialectType<(type
  TypedAttr:$elementTypesVariadic,
  TypedAttr:$isMemoryOnly,
  TypedAttr:$minAlignment,
  Bool:$isParamPack
)>;

def StructInstanceType : DialectType<(type
  StringAttr:$name,
  ConcreteAttrArray<"StringAttr">:$paramNames,
  TypedAttrArray:$paramValues,
  ConcreteAttrArray<"StructDefFieldAttr">:$fields,
  TypedAttr:$isMemoryOnly
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def WrappedVariantType : DialectType<(type
  TypedAttr:$variadic
)>;

def DeferredType : DialectType<(type)>;

def MLIRDeferredType : DialectType<(type
  Attribute:$attr
)>;

def NeverType : DialectType<(type)>;

def SIMDType : DialectType<(type
  TypedAttr:$size,
  TypedAttr:$dType
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def KGENDialectTypes : DialectTypes<"kgen"> {
  let elems = [
    WrappedParamRefType,
    WrappedTypeValueType,
    NonStructTypeType,
    TypeType,
    DTypeType,
    StringType,
    GeneratorType,
    FuncType,
    FuncLiteralType,
    PointerType,
    ParamListType,
    NoneType,
    TargetType,
    BuildInfoType,
    StructType,
    StructInstanceType,
    WrappedVariantType,
    DeferredType,
    MLIRDeferredType,
    NeverType,
    SIMDType,
  ];
}

//===----------------------------------------------------------------------===//
// KGENDialect Attributes
//===----------------------------------------------------------------------===//

def ParamDeclAttr : DialectAttribute<(attr
  StringAttr:$name,
  Type:$type
)>;

def ParamDeclArrayAttr : DialectAttribute<(attr
  ConcreteAttrArray<"ParamDeclAttr">:$value
)>;

def PogMetadataAttr : DialectAttribute<(attr
  StringAttr:$name,
  EnumAttr<"::M::KGEN::PassingKind">:$passingKind,
  EnumAttr<"::M::KGEN::VariadicKind">:$variadic,
  OptionalAttribute<"TypedAttr">:$defaultValue
)>;

def PogListAttr : DialectAttribute<(attr
  ConcreteAttrArray<"PogMetadataAttr">:$pogs,
  ConcreteAttrArray<"ConstraintAttr">:$bodyConstraints,
  EnumAttr<"ArgConvention">:$origVariadicConvention
)>;

def FnMetadataAttr : DialectAttribute<(attr
  EnumAttrArray<"ArgConvention">:$argConventions,
  Custom<"FnEffects">:$fnEffects,
  ConcreteOptionalAttribute<"FnMetadataAttrInterface">:$metadata
)>;

def ParameterExprArrayAttr : DialectAttribute<(attr
  TypedAttrArray:$value
)>;

def ParamListAttr : DialectAttribute<(attr
  TypedAttrArray:$values,
  ConcreteType<"ParamListType">:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def ParamListReduceAttr : DialectAttribute<(attr
    Type:$type,
    TypedAttr:$base,
    TypedAttr:$paramList,
    TypedAttr:$generator
)>;

def ParamListTabulateAttr : DialectAttribute<(attr
  ConcreteType<"ParamListType">:$type,
  TypedAttr:$count,
  TypedAttr:$generator
)>;

def ParamListSizeAttr : DialectAttribute<(attr
  ConcreteType<"IndexType">:$type,
  TypedAttr:$paramList
)>;

def ParamListGetAttr : DialectAttribute<(attr
  Type:$type,
  TypedAttr:$paramList,
  TypedAttr:$index
)>;

def ParamListConcatAttr : DialectAttribute<(attr
  ConcreteType<"ParamListType">:$type,
  TypedAttr:$paramLists
)>;

def UnknownAttr : DialectAttribute<(attr
  Type:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def SingletonAttr : DialectAttribute<(attr
  Type:$type
)>;

def UnboundAttr : DialectAttribute<(attr
  Type:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def NoneAttr : DialectAttribute<(attr)>;

def ParamDeclRefAttr : DialectAttribute<(attr
  StringAttr:$name,
  Type:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def ParamIndexRefAttr : DialectAttribute<(attr
  VarInt:$depth,
  VarInt:$index,
  Type:$type
)>;

def WrappedTypeParamAttr : DialectAttribute<(attr
  Type:$typeValue,
  Type:$mlirType,
  Type:$type
)>;

def BindParamsAttr : DialectAttribute<(attr
  ConcreteAttribute<"TypedAttr">:$generator,
  TypedAttrArray:$paramValues,
  ConcreteOptionalAttribute<"DenseBoolArrayAttr">:$discharged
)> {
  let cBuilder = "BindParamsAttr::getFromBytecode($_args)";
}

def TypeGeneratorRefAttr : DialectAttribute<(attr
  ConcreteAttribute<"SymbolRefAttr">:$symbol,
  TypedAttrArray:$paramValues,
  Type:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def TypeInstanceRefAttr : DialectAttribute<(attr
  ConcreteAttribute<"SymbolRefAttr">:$symbol,
  Type:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def DTypeConstantAttr : DialectAttribute<(attr
  Custom<"KGENDType">:$dType
)>;

def SymbolConstantAttr : DialectAttribute<(attr
  ConcreteAttribute<"SymbolRefAttr">:$symbol,
  TypedAttrArray:$paramValues,
  ConcreteType<"GeneratorType">:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def FuncSymbolAttr : DialectAttribute<(attr
  ConcreteAttribute<"SymbolRefAttr">:$symbol,
  TypedAttrArray:$paramValues,
  ConcreteType<"FuncType">:$type
)>;

def FuncPtrBitcastAttr : DialectAttribute<(attr
  ConcreteAttribute<"SymbolConstantAttr">:$callee,
  ConcreteType<"FuncTypeGeneratorType">:$type
)>;

def GeneratorAttr : DialectAttribute<(attr
  TypeArray:$inputParamTypes,
  ConcreteAttribute<"TypedAttr">:$body,
  ConcreteOptionalAttribute<"PogListAttr">:$metadata
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def TargetParamAttr : DialectAttribute<(attr
  ConcreteAttribute<"TargetInfoAttr">:$target
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def EnvAttr : DialectAttribute<(attr
  ConcreteAttribute<"DictionaryAttr">:$values
)>;

def WrappedParamOperatorAttr : DialectAttribute<(attr
  EnumAttr<"POC">:$opcode,
  TypedAttrArray:$operands,
  Type:$type
)>;

def ParamIdenticalAttr : DialectAttribute<(attr
  TypedAttrArray:$operands
)>;

def CmpPredicateAttr : DialectAttribute<(attr
  EnumAttr<"CmpPredicate">:$value
)>;

def MLIROpAttr : DialectAttribute<(attr
  StringAttr:$name,
  ConcreteAttribute<"DictionaryAttr">:$attrs,
  ConcreteType<"FuncTypeGeneratorType">:$type
)>;

def DecoratorsAttr : DialectAttribute<(attr
  TypedAttrArray:$value
)>;

def ExportKindAttr : DialectAttribute<(attr
  EnumAttr<"ExportKind">:$value
)>;

def StructDefFieldAttr : DialectAttribute<(attr
  StringAttr:$name,
  TypedAttr:$typeValue
)>;

def StructAttr : DialectAttribute<(attr
  TypedAttrArray:$values,
  ConcreteType<"StructType">:$type
)>;

def WrappedStructExtractAttr : DialectAttribute<(attr
  TypedAttr:$structValue,
  TypedAttr:$fieldNo,
  Type:$type
)>;

def UpcastAttr : DialectAttribute<(attr
  Type:$type,
  TypedAttr:$inputTypeValue
)>;

def TraitSymbolAttr : DialectAttribute<(attr
  ConcreteAttribute<"SymbolRefAttr">:$symbol
)>;

def TraitSymbolArrayAttr : DialectAttribute<(attr
  ConcreteAttrArray<"TraitSymbolAttr">:$value
)>;

def GetWitnessAttr : DialectAttribute<(attr
  TypedAttr:$typeValue,
  ConcreteAttribute<"TraitSymbolAttr">:$traitSymbol,
  StringAttr:$witnessName,
  Type:$type
)>;

def VariantAttr : DialectAttribute<(attr
  TypedAttr:$value,
  VarInt:$index,
  ConcreteType<"VariantType">:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def LinkDependencyArrayAttr : DialectAttribute<(attr
  ConcreteAttrArray<"FlatSymbolRefAttr">:$value
)>;

def PreservedAttr : DialectAttribute<(attr
  Attribute:$value
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def TailKindAttr : DialectAttribute<(attr
  EnumAttr<"TailKind">:$value
)>;

def ClosureMethodAttr : DialectAttribute<(attr
  EnumAttr<"ClosureMethod">:$value
)>;

def ConstraintAttr : DialectAttribute<(attr
  TypedAttr:$proposition,
  ConcreteAttribute<"LocationAttr">:$loc,
  ConcreteOptionalAttribute<"StringAttr">:$message
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def LLVMBitcodeLibAttr : DialectAttribute<(attr
  ConcreteAttribute<"BoolAttr">:$used,
  Attribute:$library
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def LLVMBitcodeLibArrayAttr : DialectAttribute<(attr
  ConcreteAttrArray<"LLVMBitcodeLibAttr">:$value
)>;

def SugarAttr : DialectAttribute<(attr
  EnumAttr<"SugarKind">:$kind,
  ConcreteOptionalAttribute<"StringAttr">:$memberName,
  TypedAttr:$sugared,
  TypedAttr:$expanded
)>;

def ClosureSymbolAttr : DialectAttribute<(attr
  ConcreteAttribute<"SymbolRefAttr">:$parentSymbol,
  StringAttr:$nestedFuncName,
  ConcreteAttribute<"ClosureMethodAttr">:$method,
  TypedAttrArray:$paramValues,
  ConcreteType<"GeneratorType">:$type
)>;

def LinkageNameAttr : DialectAttribute<(attr
  TypedAttr:$name,
  ConcreteAttribute<"BoolAttr">:$mangle
)>;

def TypeConformsToTraitAttr : DialectAttribute<(attr
  TypedAttr:$typeValue,
  ConcreteAttribute<"TypedAttr">:$traitType
)>;

def DowncastAttr : DialectAttribute<(attr
  Type:$type,
  TypedAttr:$inputTypeValue
)>;

def ExtensionAttr : DialectAttribute<(attr
  Type:$type,
  TypedAttr:$anchor,
  TypedAttrArray:$extensions
)>;

def CompileOffloadClosureAttr : DialectAttribute<(attr
  TypedAttr:$target,
  TypedAttr:$func,
  Type:$type
)>;

def GetLinkageNameAttr : DialectAttribute<(attr
  TypedAttr:$target,
  TypedAttr:$func,
  Type:$type
)>;

def GetSourceNameAttr : DialectAttribute<(attr
  TypedAttr:$func,
  Type:$type
)>;

def GetFunctionParameterCountAttr : DialectAttribute<(attr
  TypedAttr:$func,
  ConcreteType<"IndexType">:$type
)>;

def GetFunctionParameterNamesAttr : DialectAttribute<(attr
  TypedAttr:$func,
  ConcreteType<"ParamListType">:$type
)>;

def GetFunctionIsRaisingAttr : DialectAttribute<(attr
  TypedAttr:$func,
  ConcreteType<"IntegerType">:$type
)>;

def CompileAssemblyAttr : DialectAttribute<(attr
  TypedAttr:$target,
  TypedAttr:$emissionKind,
  TypedAttr:$emissionOptions,
  ConcreteAttribute<"BoolAttr">:$propagateError,
  TypedAttr:$func,
  Type:$type
)>;

def GetTypeNameAttr : DialectAttribute<(attr
  TypedAttr:$typeValue,
  TypedAttr:$qualifiedBuiltins,
  Type:$type
)>;

def StructFieldTypesAttr : DialectAttribute<(attr
  TypedAttr:$typeValue,
  ConcreteType<"ParamListType">:$type
)>;

def StructFieldNamesAttr : DialectAttribute<(attr
  TypedAttr:$typeValue,
  ConcreteType<"ParamListType">:$type
)>;

def StructFieldIndexByNameAttr : DialectAttribute<(attr
  TypedAttr:$typeValue,
  TypedAttr:$fieldName,
  ConcreteType<"IndexType">:$type
)>;

def StructFieldTypeByNameAttr : DialectAttribute<(attr
  TypedAttr:$typeValue,
  TypedAttr:$fieldName,
  Type:$type
)>;

def StructFieldOffsetByIndexAttr : DialectAttribute<(attr
  TypedAttr:$typeValue,
  TypedAttr:$fieldIndex,
  TypedAttr:$target
)>;

def StructFieldOffsetByNameAttr : DialectAttribute<(attr
  TypedAttr:$typeValue,
  TypedAttr:$fieldName,
  TypedAttr:$target
)>;

def IsStructTypeAttr : DialectAttribute<(attr
  TypedAttr:$typeValue,
  ConcreteType<"IntegerType">:$type
)>;

def FnTypeIsCABIAttr : DialectAttribute<(attr
  TypedAttr:$typeValue,
  ConcreteType<"IntegerType">:$type
)>;

def GetBaseTypeNameAttr : DialectAttribute<(attr
  TypedAttr:$typeValue
)>;

def DeferredAttr : DialectAttribute<(attr
  Attribute:$attr
)>;

def AttrCtorDeferredAttr : DialectAttribute<(attr
  TypedAttrArray:$strings
)>;

def ToStringDeferredAttr : DialectAttribute<(attr
  Attribute:$attr,
  ConcreteOptionalAttribute<"UnitAttr">:$needElideType
)>;

def CastFromBuiltinAttr : DialectAttribute<(attr
  TypedAttr:$arg, ConcreteType<"SIMDType">:$type
)>;
def CastToBuiltinAttr : DialectAttribute<(attr
  TypedAttr:$arg, Type:$type
)>;

def DTypeValues :
  WithParser<"succeeded(readDTypeValues($_reader, $_var))",
  WithPrinter<"writeDTypeValues($_writer, $_getter)",
  WithType<"SmallVector<DTypeValue>">>>;

def SIMDAttr : DialectAttribute<(attr
  DTypeValues:$values,
  ConcreteType<"SIMDType">:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def SIMDSplatAttr : DialectAttribute<(attr
  TypedAttr:$arg, ConcreteType<"SIMDType">:$type
)>;

def KGENDialectAttributes : DialectAttributes<"kgen"> {
  let elems = [
    ConstraintAttr,
    ParameterExprArrayAttr,
    ParamDeclAttr,
    ParamDeclArrayAttr,
    ParamListAttr,
    ParamListReduceAttr,
    ParamListTabulateAttr,
    ParamListSizeAttr,
    ParamListGetAttr,
    ParamListConcatAttr,
    UnknownAttr,
    UnboundAttr,
    NoneAttr,
    ParamDeclRefAttr,
    ParamIndexRefAttr,
    WrappedTypeParamAttr,
    BindParamsAttr,
    TypeGeneratorRefAttr,
    TypeInstanceRefAttr,
    DTypeConstantAttr,
    SymbolConstantAttr,
    FuncSymbolAttr,
    GeneratorAttr,
    TargetParamAttr,
    StructDefFieldAttr,
    StructAttr,
    WrappedStructExtractAttr,
    UpcastAttr,
    GetWitnessAttr,
    VariantAttr,
    EnvAttr,
    WrappedParamOperatorAttr,
    CmpPredicateAttr,
    MLIROpAttr,
    DecoratorsAttr,
    PreservedAttr,
    ExportKindAttr,
    LinkDependencyArrayAttr,
    TailKindAttr,
    LLVMBitcodeLibAttr,
    LLVMBitcodeLibArrayAttr,
    SugarAttr,
    ClosureSymbolAttr,
    ClosureMethodAttr,
    LinkageNameAttr,
    TypeConformsToTraitAttr,
    DowncastAttr,
    ExtensionAttr,
    CompileOffloadClosureAttr,
    GetLinkageNameAttr,
    GetSourceNameAttr,
    GetFunctionParameterCountAttr,
    GetFunctionParameterNamesAttr,
    GetFunctionIsRaisingAttr,
    CompileAssemblyAttr,
    GetTypeNameAttr,
    StructFieldTypesAttr,
    StructFieldNamesAttr,
    StructFieldIndexByNameAttr,
    StructFieldTypeByNameAttr,
    StructFieldOffsetByIndexAttr,
    StructFieldOffsetByNameAttr,
    IsStructTypeAttr,
    FnTypeIsCABIAttr,
    GetBaseTypeNameAttr,
    DeferredAttr,
    AttrCtorDeferredAttr,
    ToStringDeferredAttr,
    CastFromBuiltinAttr,
    CastToBuiltinAttr,
    SIMDAttr,
    SIMDSplatAttr,
    PogMetadataAttr,
    PogListAttr,
    FuncPtrBitcastAttr,
    ParamIdenticalAttr,
    SingletonAttr,
    FnMetadataAttr,
    TraitSymbolAttr,
    TraitSymbolArrayAttr
  ];
}

#endif // KGENDIALECT_BYTECODE
