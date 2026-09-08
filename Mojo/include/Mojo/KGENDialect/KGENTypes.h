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
// This file declares types for the KGEN dialect.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_KGENTYPES_H
#define KGEN_KGENDIALECT_KGENTYPES_H

#include "Mojo/Interpreter/InterpreterInterface.h"
#include "Mojo/KGENDialect/KGENDType.h"
#include "Mojo/KGENDialect/KGENDialect.h"
#include "Mojo/KGENDialect/KGENEnums.h"
#include "Mojo/KGENDialect/KGENTypeInterfaces.h"
#include "Support/ForwardDecls.h"
#include "Support/MDialect/MTypeInterfaces.h"

namespace M::KGEN {
class ParameterEvaluationContext;
class FnMetadataAttrInterface;
class FuncInterface;
class ParamDeclAttr;
class ParamDeclArrayAttr;
struct PartiallySpecializedInputParams;
class FuncSymbolAttr;
class FuncTypeGeneratorType;
class FuncLiteralTypeGeneratorType;
class ConstraintAttr;
class FnMetadataAttr;
class PogListAttr;
class StructDefFieldAttr;
class SymbolConstantAttr;
class ParamListType;
class ParamListAttr;
} // namespace M::KGEN

#define GET_TYPEDEF_CLASSES
#include "Mojo/KGENDialect/KGENTypes.h.inc"

namespace M::KGEN {
//===----------------------------------------------------------------------===//
// FuncTypeGeneratorType
//===----------------------------------------------------------------------===//
class FuncTypeGeneratorType : public GeneratorType {
public:
  using GeneratorType::GeneratorType;
  FuncTypeGeneratorType(GeneratorType sig);

  static FuncTypeGeneratorType
  get(ArrayRef<Type> inputParamTypes, FunctionType values,
      ArrayRef<ArgConvention> argConvs = {}, FnEffects effects = {},
      Attribute fnMetadata = {}, Attribute genMetadata = {},
      Attribute argListAttrs = {});

  static FuncTypeGeneratorType get(ArrayRef<Type> inputParamTypes, FuncType sig,
                                   Attribute genMetadata);

  /// Get this GeneratorType with some parameters bound.
  FuncTypeGeneratorType
  getSpecializedGenerator(ArrayRef<TypedAttr> paramBindings,
                          ParameterEvaluationContext *evaluationContext,
                          function_ref<InFlightDiagnostic()> emitErrorFn = {});
  FuncTypeGeneratorType
  getSpecializedGenerator(ArrayRef<TypedAttr> paramBindings,
                          ParameterEvaluationContext *evaluationContext,
                          Location location);

  /// Construct a signature from named parameter declarations, a function
  /// type, and metadata. This helper is used to convert between a named
  /// signature structure to a nameless `FuncTypeGeneratorType`
  /// representation.
  static FuncTypeGeneratorType remapToFuncTypeGenerator(
      ArrayRef<ParamDeclAttr> inputParams, FunctionType functionType,
      ArrayRef<ArgConvention> argConventions = {}, FnEffects effects = {},
      Attribute fnMetadata = {}, Attribute genMetadata = {},
      function_ref<InFlightDiagnostic()> emitError = {},
      Attribute argListAttrs = {});

  FuncType getBody();
  FuncType getInstantiatedBody();

  /// A FuncTypeGeneratorType is a GeneratorType containing a FuncType.
  static bool classof(GeneratorType type);
  static bool classof(Type type);
};

//===----------------------------------------------------------------------===//
// FuncLiteralTypeGeneratorType
//===----------------------------------------------------------------------===//

class FuncLiteralTypeGeneratorType : public GeneratorType {
public:
  using GeneratorType::GeneratorType;
  FuncLiteralTypeGeneratorType(GeneratorType gen);

  static FuncLiteralTypeGeneratorType get(ArrayRef<Type> inputParamTypes,
                                          TypedAttr funcLiteral,
                                          Attribute genMetadata = {});

  /// Get this GeneratorType with some parameters bound.
  FuncLiteralTypeGeneratorType
  getSpecializedGenerator(ArrayRef<TypedAttr> paramBindings,
                          ParameterEvaluationContext *evaluationContext,
                          function_ref<InFlightDiagnostic()> emitErrorFn = {});
  FuncLiteralTypeGeneratorType
  getSpecializedGenerator(ArrayRef<TypedAttr> paramBindings,
                          ParameterEvaluationContext *evaluationContext,
                          Location location);

  // Unwrap the target literal from the literal type.
  SymbolConstantAttr getSymbolConstantAttr();

  FuncLiteralType getBody();
  FuncLiteralType getInstantiatedBody();

  /// A FuncLiteralTypeGeneratorType is a GeneratorType containing a
  /// FuncLiteralType.
  static bool classof(GeneratorType type);
  static bool classof(Type type);
};
} // namespace M::KGEN

#endif // KGEN_KGENDIALECT_KGENTYPES_H
