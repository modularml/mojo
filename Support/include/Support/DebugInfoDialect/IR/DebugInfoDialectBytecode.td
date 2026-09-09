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

#ifndef DEBUGINFODIALECT_BYTECODE
#define DEBUGINFODIALECT_BYTECODE

include "Support/Compiler/Bytecode.td"

//===----------------------------------------------------------------------===//
// DebugInfoDialect Types
//===----------------------------------------------------------------------===//

def DIArrayType : DialectType<(type
  ConcreteType<"DIType">:$elementType,
  VarInt:$elementCount
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def DIBasicType : DialectType<(type
  StringAttr:$name,
  VarInt:$sizeInBits,
  VarInt:$alignInBits,
  VarInt:$encoding
)>;

def DIMemberType : DialectType<(type
  StringAttr:$name,
  ConcreteType<"DIType">:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def DIPointerType : DialectType<(type
  ConcreteType<"DIType">:$elementType,
  VarInt:$sizeInBits,
  VarInt:$alignInBits,
  Custom<"OptionalUnsigned">:$addressSpace
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def DIStructType : DialectType<(type
  StringAttr:$name,
  ConcreteTypeArray<"DIMemberType">:$members
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def DISubroutineType : DialectType<(type
  VarInt:$callingConvention,
  ConcreteTypeArray<"DIType">:$argumentTypes,
  ConcreteTypeArray<"DIType">:$resultTypes
)>;

def DIUnresolvedMLIRType : DialectType<(type
  Type:$type
)>;

def DIUnspecifiedType : DialectType<(type
  StringAttr:$name
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def DIVectorType : DialectType<(type
  ConcreteType<"DIType">:$elementType,
  VarInt:$elementCount
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def DebugInfoDialectTypes : DialectTypes<"debuginfo"> {
  let elems = [
    DIArrayType,
    DIBasicType,
    DIMemberType,
    DIPointerType,
    DIStructType,
    DISubroutineType,
    DIUnresolvedMLIRType,
    DIUnspecifiedType,
    DIVectorType
  ];
}

//===----------------------------------------------------------------------===//
// DebugInfoDialect Attributes
//===----------------------------------------------------------------------===//

def SourceNameAttr : DialectAttribute<(attr
  StringAttr:$name,
  ConcreteAttrArray<"SourceNameAttr">:$paramTypes,
  ConcreteAttrArray<"SourceNameAttr">:$argTypes,
  StringAttrArray:$paramValues,
  ConcreteOptionalAttribute<"SourceNameAttr">:$parent,
  EnumAttr<"SourceNameKind">:$kind,
  ConcreteAttrArray<"SourceNameAttr">:$decorators
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def DICompileUnitAttr : DialectAttribute<(attr
  VarInt:$sourceLanguage,
  ConcreteAttribute<"DIFileAttr">:$file,
  StringAttr:$producer,
  Bool:$isOptimized,
  EnumAttr<"EmissionKind">:$emissionKind,
  EnumAttr<"NameTableKind">:$nameTableKind
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def DIFileAttr : DialectAttribute<(attr
  StringAttr:$name,
  StringAttr:$directory
)>;

def DILexicalBlockAttr : DialectAttribute<(attr
  ConcreteAttribute<"DILocalScopeAttr">:$scope,
  ConcreteAttribute<"DIFileAttr">:$file,
  VarInt:$line,
  VarInt:$column
)>;

def DILocalVariableAttr : DialectAttribute<(attr
  ConcreteAttribute<"DILocalScopeAttr">:$scope,
  StringAttr:$name,
  ConcreteOptionalAttribute<"DIFileAttr">:$file,
  VarInt:$line,
  VarInt:$arg,
  VarInt:$alignInBits,
  ConcreteType<"DIType">:$type,
  EnumAttr<"DIFlags">:$flags
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def DISubprogramAttr : DialectAttribute<(attr
  ConcreteOptionalAttribute<"DICompileUnitAttr">:$compileUnit,
  ConcreteOptionalAttribute<"DIScopeAttr">:$scope,
  ConcreteAttribute<"SourceNameAttr">:$sourceName,
  ConcreteOptionalAttribute<"StringAttr">:$linkageName,
  ConcreteOptionalAttribute<"DIFileAttr">:$file,
  VarInt:$line,
  VarInt:$scopeLine,
  EnumAttr<"SubprogramFlags">:$subprogramFlags,
  ConcreteType<"DISubroutineType">:$type
)> {
  let cBuilder = "get<$_resultType>(context, $_args)";
}

def DebugInfoDialectAttributes : DialectAttributes<"debuginfo"> {
  let elems = [
    SourceNameAttr,
    DICompileUnitAttr,
    DIFileAttr,
    DILexicalBlockAttr,
    DILocalVariableAttr,
    DISubprogramAttr
  ];
}

#endif // DEBUGINFODIALECT_BYTECODE
