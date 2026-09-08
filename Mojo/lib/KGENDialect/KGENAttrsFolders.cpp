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
// This file contains evaluation/folding implementations for KGEN attributes.
// These methods implement
// ContextuallyEvaluatedAttrInterface::evaluateWithContext.
//
//===----------------------------------------------------------------------===//

#include "Mojo/KGENDialect/FoldUtils.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Support/MDialect/MTypeInterfaces.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseSet.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// ParamListReduceAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> ParamListReduceAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  auto paramList = sugarDynCast<ParamListAttr>(getParamList());
  auto reducer = sugarDynCast<GeneratorAttr>(getGenerator());

  if (!paramList || !reducer)
    return failure();

  // We have a concrete value for both the generator/variadic, then fold
  unsigned eltCnt = paramList.getValues().size();
  TypedAttr reducedVal = sugarCast<TypedAttr>(getBase());
  for (unsigned i = 0; i < eltCnt; ++i) {
    IntegerAttr vaIdx =
        IntegerAttr::get(IndexType::get(paramList.getContext()), i);
    GeneratorAttr spGen = reducer.getSpecializedGenerator(
        {reducedVal, paramList, vaIdx}, &context);
    if (!spGen)
      return TypedAttr();
    // This should never happen, we should have verified VariadicMapAttr.
    assert(spGen.isFullyBound() && "invalid form of variadic map");
    reducedVal = spGen.getInstantiatedValue();
  }

  return {reducedVal};
}

//===----------------------------------------------------------------------===//
// ParamListTabulateAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> ParamListTabulateAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  auto cntAttr = sugarDynCast<IntegerAttr>(getCount());
  auto genAttr = sugarDynCast<GeneratorAttr>(getGenerator());
  if (!cntAttr || !genAttr)
    return failure();

  int64_t n = cntAttr.getInt();
  if (n < 0)
    return failure();

  SmallVector<TypedAttr> values;
  values.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    IntegerAttr idxAttr = IntegerAttr::get(IndexType::get(getContext()), i);
    GeneratorAttr spGen = genAttr.getSpecializedGenerator({idxAttr}, &context);
    if (!spGen)
      return TypedAttr();
    if (!spGen.isFullyBound())
      return failure();
    values.push_back(sugarCast<TypedAttr>(spGen.getInstantiatedValue()));
  }
  return {ParamListAttr::get(values, getType())};
}

//===----------------------------------------------------------------------===//
// GetWitnessAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr>
GetWitnessAttr::evaluateWithContext(ParameterEvaluationContext &context) const {
  // Returns nullopt when `handle` has no matching conformance.
  auto simplifyFrom =
      [&](ResolvedStructHandle handle) -> std::optional<FailureOr<TypedAttr>> {
    Operation *conformanceOp =
        context.resolveConformanceForStruct(handle, getTraitSymbol());
    if (!conformanceOp)
      return std::nullopt;

    FailureOr<TypedAttr> result = failure();
    context.withEvaluator(handle.decl.getInputParams(), handle.paramValues,
                          [&](ParameterEvaluator &evaluator) {
                            result = simplify(
                                cast<ConformanceOp>(conformanceOp), &evaluator);
                          });
    if (failed(result))
      context.emitMaterializationError(
          "failed to locate witness entry '" + getWitnessName().getValue() +
          "' for trait '" + getTraitSymbol().getFlattenedName().getValue() +
          "'");
    return result;
  };

  // First, look for the requested conformance directly on the anchor type.
  FailureOr<ResolvedStructHandle> resolvedOr =
      context.resolveStructOp(getTypeValue(), /*acceptAsync=*/false);
  if (succeeded(resolvedOr)) {
    if (auto result = simplifyFrom(*resolvedOr))
      return *result;
  }

  // The anchor type does not itself carry the requested conformance, or is not
  // yet resolvable. Check the extension conformance table.
  if (auto extension =
          sugarDynCast<ExtensionAttr>(SugarAttr::strip(getTypeValue()))) {
    for (TypedAttr ext : extension.getExtensions()) {
      FailureOr<ResolvedStructHandle> extResolvedOr =
          context.resolveStructOp(ext, /*acceptAsync=*/false);
      if (failed(extResolvedOr))
        return failure();
      if (auto result = simplifyFrom(*extResolvedOr))
        return *result;
    }
  }

  // Neither the anchor nor any extension provided the conformance. If the
  // anchor was resolvable and we still could not evaluate this is an error.
  if (succeeded(resolvedOr))
    context.emitMaterializationError(
        "struct '" +
        SymbolTable::getSymbolName(resolvedOr->decl.getOperation()).getValue() +
        "' does not have witness table for trait '" +
        getTraitSymbol().getFlattenedName().getValue() + "'");
  return failure();
}

//===----------------------------------------------------------------------===//
// ParamOperatorAttr
//===----------------------------------------------------------------------===//

// FIXME(MOCO-4110): The reason why we need to extend
// `ParamOperatorAttr::evaluateWithContext` here is because
// `ContextuallyEvaluatedAttrInterface` is **NOT** always created with a
// context. For things like `TypeConformsToTraitAttr` that canonicalizes to a
// `ParamOperatorAttr` without a context, it then needs to be further evaluated
// here.
//
// If we can make `EvalContext` mandatory for creating any
// `ContextuallyEvaluatedAttrInterface`, we could then delete the code below. At
// that time, every `ContextuallyEvaluatedAttrInterface` should be in
// canonicalized + fully-evaluated form UPON CONSTRUCTION, which eliminates the
// need for `evaluator.getReboundXXX`, and the attr evaluation can be
// implemented in a much more local way.
FailureOr<TypedAttr> ParamOperatorAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  bool changed = false;
  SmallVector<TypedAttr> operands(getOperands());
  for (auto [i, cur] : llvm::enumerate(operands)) {
    if (auto ctxEval = sugarDynCast<ContextuallyEvaluatedAttrInterface>(cur)) {
      FailureOr<TypedAttr> result = context.evaluateExpression(ctxEval);
      if (failed(result)) {
        operands[i] = cur;
        continue;
      }

      TypedAttr evaluated = *result;
      // Defer the entire evaluation
      if (!evaluated)
        return TypedAttr();

      changed |= (operands[i] != evaluated);
      operands[i] =
          ParamOperatorAttr::getRebind(evaluated, operands[i].getType());
    }
  }
  if (changed)
    return ParamOperatorAttr::get(getOpcode(), operands, getType());

  // Can not be further folded.
  return failure();
}

//===----------------------------------------------------------------------===//
// ParamIdenticalAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> ParamIdenticalAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  // `get()` decided everything that does not need a target, so the only thing
  // left to contribute here is the index bit width.
  TargetInfoAttr target = context.getTargetInfo();
  if (!target)
    return failure();

  // Given the index bit width, an operand's index-like leaves re-expressed at
  // that width are canonical for the value, so two operands denote the same
  // value exactly when that key is the same uniqued attribute and one pass
  // answers for the whole class.
  //
  // Only an operand that is fully evaluated and free of unknowns has a value
  // for a key to stand for; the rest sit out the partition. Two others can
  // still settle the class `false` around one, which is why this counts
  // distinct keys rather than requiring them all to match.
  DenseSet<Attribute> keys;
  bool sawResidual = false;
  for (TypedAttr operand : getOperands()) {
    TypedAttr canonical = getCanonicalAttr(operand);
    PreparedConstant prepared(canonical, target);
    if (!ParameterAttr::isSimpleConstant(canonical) || prepared.hasUnknown()) {
      sawResidual = true;
      continue;
    }
    // One pair that cannot denote the same value settles the whole class, so
    // stop before keying the rest.
    keys.insert(prepared.getKey());
    if (keys.size() > 1)
      return TypedAttr(SIMDAttr::getScalarBool(getContext(), false));
  }

  // An operand with no key never collapses into another, so it leaves the class
  // residual however the rest of them compare.
  if (sawResidual)
    return failure();

  assert(!keys.empty() && "an identity class holds at least two operands");
  return TypedAttr(SIMDAttr::getScalarBool(getContext(), true));
}

//===----------------------------------------------------------------------===//
// TypeConformsToTraitAttr
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> TypeConformsToTraitAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  FailureOr<ResolvedStructHandle> resolvedOr =
      context.resolveStructOp(getTypeValue(), /*acceptAsync=*/false);
  if (failed(resolvedOr)) {
    if (context.isMaterializationContext()) {
      if (auto typeParam = dyn_cast<TypeParamAttr>(getTypeValue());
          typeParam && !isa<TypeValueType>(typeParam.getTypeValue())) {
        // This is a non-struct type.
        //
        // FIXME: non-struct type is TRP, maybe we should resolve it to
        // `__MLIRType` instead?
        return TypedAttr(SIMDAttr::getScalarBool(getContext(), false));
      }
    }
    return failure();
  }

  ResolvedStructHandle resolved = *resolvedOr;
  FailureOr<TypedAttr> result = failure();
  context.withEvaluator(resolved.decl.getInputParams(), resolved.paramValues,
                        [&](ParameterEvaluator &evaluator) {
                          result = simplify(resolved.decl, evaluator);
                        });
  return result;
}

//===----------------------------------------------------------------------===//
// Struct Field Attr evaluateWithContext implementations
//===----------------------------------------------------------------------===//

/// Compute the byte offset of a field within a struct given its field types.
/// Returns failure if the field index is out of bounds or if size/alignment
/// cannot be determined for any field type.
static FailureOr<int64_t>
computeStructFieldOffset(ArrayRef<Type> fieldTypes, int64_t fieldIndex,
                         TargetInfoAttr target,
                         llvm::function_ref<void(const Twine &)> emitError) {
  if (fieldIndex < 0 || fieldIndex >= static_cast<int64_t>(fieldTypes.size())) {
    emitError("field index " + std::to_string(fieldIndex) +
              " is out of bounds for struct with " +
              std::to_string(fieldTypes.size()) + " fields");
    return failure();
  }

  int64_t offset = 0;
  for (int64_t i = 0; i < fieldIndex; ++i) {
    std::optional<int64_t> curFieldAlign =
        DataLayoutInterface::getTypeABIAlign(target, fieldTypes[i]);
    std::optional<int64_t> curFieldSize =
        DataLayoutInterface::getTypeAllocSize(target, fieldTypes[i]);
    if (!curFieldAlign || !curFieldSize) {
      emitError("could not determine size or alignment for field type");
      return failure();
    }
    offset = llvm::alignTo(offset, *curFieldAlign) + *curFieldSize;
  }

  // Align to the target field's alignment.
  std::optional<int64_t> fieldAlign =
      DataLayoutInterface::getTypeABIAlign(target, fieldTypes[fieldIndex]);
  if (!fieldAlign) {
    emitError("could not determine alignment for field type");
    return failure();
  }
  return llvm::alignTo(offset, *fieldAlign);
}

/// Extract field types from a struct, wrapping each in ParamType.
static SmallVector<Type>
getFieldTypesFromStruct(StructInstanceType structType) {
  SmallVector<Type> fieldTypes;
  for (StructDefFieldAttr field : structType.getFields())
    fieldTypes.push_back(ParamType::get(field.getTypeValue()));
  return fieldTypes;
}

/// Extract and rebind field types from a struct declaration using an evaluator.
/// Returns an empty vector for empty structs.
///
/// Returns std::nullopt when a field type is not ready yet. A null result from
/// `getReboundAttribute` is the async retry signal, so callers must forward it
/// as a null success rather than collapsing it into failure().
static std::optional<SmallVector<Type>>
rebindFieldTypes(StructDeclInterface decl, ParameterEvaluator &evaluator) {
  SmallVector<TypedAttr> fieldTypeAttrs;
  // MetaType does not really matter here, they will be striped later by
  // `ParamType::get(rebound)` anyway.
  decl.getFieldTypes(fieldTypeAttrs, TypeType::get(decl.getContext()));

  SmallVector<Type> fieldTypes;
  for (TypedAttr typeAttr : fieldTypeAttrs) {
    TypedAttr rebound = evaluator.getReboundAttribute(typeAttr);
    if (!rebound)
      return std::nullopt;
    fieldTypes.push_back(ParamType::get(rebound));
  }
  return fieldTypes;
}

FailureOr<TypedAttr> StructFieldTypesAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  if (auto typeParam = dyn_cast<TypeParamAttr>(getTypeValue())) {
    if (auto structType =
            sugarDynCast<KGEN::StructType>(typeParam.getTypeValue())) {
      auto elementTypes = structType.getElementTypes();
      if (!elementTypes)
        return failure();
      SmallVector<TypedAttr> resultAttrs;
      resultAttrs.reserve(elementTypes->size());
      Type resultElemType = getType().getElementType();
      for (Type fieldType : *elementTypes)
        resultAttrs.push_back(TypeParamAttr::get(fieldType, resultElemType));
      return cast<TypedAttr>(ParamListAttr::get(resultAttrs, getType()));
    }
  }

  FailureOr<ResolvedStructHandle> resolvedOr =
      context.resolveStructOp(getTypeValue(), /*acceptAsync=*/true);
  if (failed(resolvedOr)) {
    context.emitMaterializationError(
        "struct_field_types requires a struct type");
    return failure();
  }
  ResolvedStructHandle resolved = *resolvedOr;

  // If concrete instance is available, use its already-substituted field types.
  if (resolved.instance) {
    auto structType =
        cast<StructInstanceType>(resolved.instance.getValueDomainType());
    SmallVector<TypedAttr> resultAttrs;
    for (StructDefFieldAttr field : structType.getFields())
      resultAttrs.push_back(field.getTypeValue());
    return cast<TypedAttr>(ParamListAttr::get(resultAttrs, getType()));
  }

  // If the decl is null, we are in an async context and the struct instance is
  // not yet ready.
  if (!resolved.decl)
    return TypedAttr();

  // Otherwise, use generator types and rebind with param values.
  SmallVector<TypedAttr> fieldTypes;
  resolved.decl.getFieldTypes(fieldTypes, getType().getElementType());

  FailureOr<TypedAttr> result = failure();
  context.withEvaluator(
      resolved.decl.getInputParams(), resolved.paramValues,
      [&](ParameterEvaluator &evaluator) {
        SmallVector<TypedAttr> resultAttrs;
        for (TypedAttr fieldType : fieldTypes) {
          TypedAttr rebound = evaluator.getReboundAttribute(fieldType);
          if (!rebound) {
            result = TypedAttr();
            return;
          }
          resultAttrs.push_back(rebound);
        }
        result = cast<TypedAttr>(ParamListAttr::get(resultAttrs, getType()));
      });
  return result;
}

FailureOr<TypedAttr> StructFieldNamesAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  FailureOr<ResolvedStructHandle> resolvedOr =
      context.resolveStructOp(getTypeValue(), /*acceptAsync=*/false);
  if (failed(resolvedOr)) {
    context.emitMaterializationError(
        "struct_field_names requires a struct type");
    return failure();
  }
  ResolvedStructHandle resolved = *resolvedOr;
  SmallVector<StringAttr> fieldNames;
  resolved.decl.getFieldNames(fieldNames);

  SmallVector<TypedAttr> resultAttrs;
  MLIRContext *ctx = getContext();
  for (StringAttr name : fieldNames)
    resultAttrs.push_back(
        StringAttr::get(name.getValue(), StringType::get(ctx)));

  return cast<TypedAttr>(ParamListAttr::get(resultAttrs, getType()));
}

//===----------------------------------------------------------------------===//
// Function Reflection Attrs
//===----------------------------------------------------------------------===//

namespace {
/// Resolve a function-valued `TypedAttr` to its defining op via the
/// evaluation context. Returns null if the value is not a direct function
/// reference (`#kgen.symbol.constant<@...>`).
///
/// The returned `FuncInterface` op is `lit.fn` when resolved through the
/// parser or LIT symbol-table contexts, or `kgen.generator` when resolved
/// through the KGEN symbol-table or IR evaluator contexts. Reflection
/// attrs are evaluated during parsing or during elaboration, so the
/// post-elaboration `kgen.func` form is never the resolution target. Both
/// reachable ops also implement `DeclInterface`, so callers needing the
/// param list cast accordingly.
FuncInterface resolveFuncDecl(TypedAttr funcValue,
                              ParameterEvaluationContext &context) {
  // Mojo function values reach reflection as `#kgen.symbol.constant<@func>`.
  // Closure literals are not yet supported.
  auto symbol = dyn_cast<SymbolConstantAttr>(funcValue);
  if (!symbol)
    return nullptr;
  return context.resolveFunctionDecl(symbol.getSymbol());
}
} // namespace

FailureOr<TypedAttr> GetFunctionParameterCountAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  FuncInterface func = resolveFuncDecl(getFunc(), context);
  if (!func) {
    context.emitMaterializationError(
        "get_function_parameter_count requires a concrete function value");
    return failure();
  }
  // Prefer the source-declared parameter list snapshot on `kgen.generator`
  // when available so reflection counts remain stable across transforms that
  // rewrite the live `inputParams` (e.g. `RemoveUnusedParams`). Falls back to
  // the live input params for `lit.fn` (pre-LowerLIT reflection) and for
  // generators that don't carry a snapshot.
  size_t count;
  if (auto gen = dyn_cast<GeneratorOp>(func.getOperation())) {
    if (PogListAttr snapshot = gen.getSourceParamListAttr()) {
      count = snapshot.size();
    } else {
      count = gen.getInputParams().size();
    }
  } else {
    count = cast<DeclInterface>(func.getOperation()).getInputParams().size();
  }
  return cast<TypedAttr>(IntegerAttr::get(IndexType::get(getContext()), count));
}

FailureOr<TypedAttr> GetFunctionParameterNamesAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  FuncInterface func = resolveFuncDecl(getFunc(), context);
  if (!func) {
    context.emitMaterializationError(
        "get_function_parameter_names requires a concrete function value");
    return failure();
  }
  MLIRContext *ctx = getContext();
  SmallVector<TypedAttr> resultAttrs;

  auto appendName = [&](StringAttr name) {
    resultAttrs.push_back(
        StringAttr::get(name.getValue(), StringType::get(ctx)));
  };

  // Prefer the source-declared parameter list snapshot on `kgen.generator`
  // when available; see the comment in `GetFunctionParameterCountAttr`.
  if (auto gen = dyn_cast<GeneratorOp>(func.getOperation())) {
    if (PogListAttr snapshot = gen.getSourceParamListAttr()) {
      resultAttrs.reserve(snapshot.size());
      for (PogMetadataAttr pog : snapshot.getPogs())
        appendName(pog.getName());
      return cast<TypedAttr>(ParamListAttr::get(resultAttrs, getType()));
    }
  }

  ArrayRef<ParamDeclAttr> params =
      cast<DeclInterface>(func.getOperation()).getInputParams();
  resultAttrs.reserve(params.size());
  for (ParamDeclAttr param : params)
    appendName(param.getName());
  return cast<TypedAttr>(ParamListAttr::get(resultAttrs, getType()));
}

FailureOr<TypedAttr> GetFunctionIsRaisingAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  FuncInterface func = resolveFuncDecl(getFunc(), context);
  if (!func) {
    context.emitMaterializationError(
        "get_function_is_raising requires a concrete function value");
    return failure();
  }
  return cast<TypedAttr>(BoolAttr::get(getContext(), func.isThrows()));
}

FailureOr<TypedAttr> StructFieldIndexByNameAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  auto fieldNameAttr = dyn_cast<StringAttr>(getFieldName());
  if (!fieldNameAttr)
    return failure();

  FailureOr<ResolvedStructHandle> resolvedOr =
      context.resolveStructOp(getTypeValue(), /*acceptAsync=*/false);
  if (failed(resolvedOr)) {
    context.emitMaterializationError(
        "struct_field_index_by_name requires a struct type");
    return failure();
  }
  ResolvedStructHandle resolved = *resolvedOr;
  auto index = resolved.decl.findFieldIndex(fieldNameAttr.getValue());
  if (!index) {
    context.emitMaterializationError(
        "struct '" +
        SymbolTable::getSymbolName(resolved.decl.getOperation()).getValue() +
        "' has no field named '" + fieldNameAttr.getValue() + "'");
    return failure();
  }
  return cast<TypedAttr>(Builder(getType().getContext()).getIndexAttr(*index));
}

FailureOr<TypedAttr> StructFieldTypeByNameAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  auto fieldNameAttr = dyn_cast<StringAttr>(getFieldName());
  if (!fieldNameAttr)
    return failure();

  FailureOr<ResolvedStructHandle> resolvedOr =
      context.resolveStructOp(getTypeValue(), /*acceptAsync=*/true);
  if (failed(resolvedOr)) {
    context.emitMaterializationError(
        "struct_field_type_by_name requires a struct type");
    return failure();
  }
  ResolvedStructHandle resolved = *resolvedOr;
  StringRef fieldName = fieldNameAttr.getValue();

  // If concrete instance is available, search its fields directly.
  if (resolved.instance) {
    auto structType =
        cast<StructInstanceType>(resolved.instance.getValueDomainType());
    for (StructDefFieldAttr field : structType.getFields())
      if (field.getName().getValue() == fieldName)
        return field.getTypeValue();
    context.emitMaterializationError(
        "struct '" +
        SymbolTable::getSymbolName(resolved.decl.getOperation()).getValue() +
        "' has no field named '" + fieldName + "'");
    return failure();
  }

  // If the decl is null, we are in an async context and the struct instance is
  // not yet ready.
  if (!resolved.decl)
    return TypedAttr();

  // Otherwise, use generator's field type and rebind.
  TypedAttr fieldType = resolved.decl.getFieldType(fieldName, getType());
  if (!fieldType) {
    context.emitMaterializationError(
        "struct '" +
        SymbolTable::getSymbolName(resolved.decl.getOperation()).getValue() +
        "' has no field named '" + fieldName + "'");
    return failure();
  }

  FailureOr<TypedAttr> result = failure();
  context.withEvaluator(resolved.decl.getInputParams(), resolved.paramValues,
                        [&](ParameterEvaluator &evaluator) {
                          result = evaluator.getReboundAttribute(fieldType);
                        });
  return result;
}

FailureOr<TypedAttr> StructFieldOffsetByIndexAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  // Return failure() without an error if parameters aren't resolved to
  // constants yet. The evaluation framework will retry later when more
  // information is available.
  auto fieldIndexAttr = dyn_cast<IntegerAttr>(getFieldIndex());
  if (!fieldIndexAttr)
    return failure();

  auto targetAttr = sugarDynCast<TargetParamAttr>(getTarget());
  if (!targetAttr)
    return failure();

  FailureOr<ResolvedStructHandle> resolvedOr =
      context.resolveStructOp(getTypeValue(), /*acceptAsync=*/true);
  if (failed(resolvedOr)) {
    context.emitMaterializationError(
        "struct_field_offset_by_index requires a struct type");
    return failure();
  }
  ResolvedStructHandle resolved = *resolvedOr;
  int64_t fieldIndex = fieldIndexAttr.getInt();
  TargetInfoAttr target = targetAttr.getTarget();
  MLIRContext *ctx = getType().getContext();

  auto emitError = [&context](const Twine &msg) {
    context.emitMaterializationError(msg);
  };

  // If concrete instance is available, use its field types directly.
  if (resolved.instance) {
    assert(resolved.decl && "instance requires valid decl");
    auto structType =
        cast<StructInstanceType>(resolved.instance.getValueDomainType());
    SmallVector<Type> fieldTypes = getFieldTypesFromStruct(structType);

    FailureOr<int64_t> offsetOr =
        computeStructFieldOffset(fieldTypes, fieldIndex, target, emitError);
    if (failed(offsetOr))
      return failure();
    return cast<TypedAttr>(Builder(ctx).getIndexAttr(*offsetOr));
  }

  // If the decl is null, we are in an async context and the struct instance is
  // not yet ready.
  if (!resolved.decl)
    return TypedAttr();

  // Otherwise, use generator's field types with rebinding.
  FailureOr<TypedAttr> result = failure();
  context.withEvaluator(
      resolved.decl.getInputParams(), resolved.paramValues,
      [&](ParameterEvaluator &evaluator) {
        std::optional<SmallVector<Type>> fieldTypesOpt =
            rebindFieldTypes(resolved.decl, evaluator);
        if (!fieldTypesOpt) {
          result = TypedAttr();
          return;
        }

        FailureOr<int64_t> offsetOr = computeStructFieldOffset(
            *fieldTypesOpt, fieldIndex, target, emitError);
        if (failed(offsetOr))
          return;
        result = cast<TypedAttr>(Builder(ctx).getIndexAttr(*offsetOr));
      });
  return result;
}

FailureOr<TypedAttr> StructFieldOffsetByNameAttr::evaluateWithContext(
    ParameterEvaluationContext &context) const {
  // Return failure() without an error if parameters aren't resolved to
  // constants yet.
  auto fieldNameAttr = dyn_cast<StringAttr>(getFieldName());
  if (!fieldNameAttr)
    return failure();

  auto targetAttr = sugarDynCast<TargetParamAttr>(getTarget());
  if (!targetAttr)
    return failure();

  FailureOr<ResolvedStructHandle> resolvedOr =
      context.resolveStructOp(getTypeValue(), /*acceptAsync=*/true);
  if (failed(resolvedOr)) {
    context.emitMaterializationError(
        "struct_field_offset_by_name requires a struct type");
    return failure();
  }
  ResolvedStructHandle resolved = *resolvedOr;
  StringRef fieldName = fieldNameAttr.getValue();
  TargetInfoAttr target = targetAttr.getTarget();
  MLIRContext *ctx = getType().getContext();

  auto emitError = [&context](const Twine &msg) {
    context.emitMaterializationError(msg);
  };

  // Helper to find field index by name and emit error if not found.
  auto findFieldIndexOrError =
      [&](auto fields, StringRef structName) -> std::optional<int64_t> {
    int64_t idx = 0;
    for (auto field : fields) {
      if (field.getName().getValue() == fieldName)
        return idx;
      ++idx;
    }
    context.emitMaterializationError(
        "struct '" + structName + "' has no field named '" + fieldName + "'");
    return std::nullopt;
  };

  // If concrete instance is available, use its fields directly.
  if (resolved.instance) {
    assert(resolved.decl && "instance requires valid decl");
    auto structType =
        cast<StructInstanceType>(resolved.instance.getValueDomainType());
    auto fields = structType.getFields();
    StringRef structName =
        SymbolTable::getSymbolName(resolved.decl.getOperation()).getValue();

    std::optional<int64_t> fieldIndexOpt =
        findFieldIndexOrError(fields, structName);
    if (!fieldIndexOpt)
      return failure();

    SmallVector<Type> fieldTypes = getFieldTypesFromStruct(structType);
    FailureOr<int64_t> offsetOr =
        computeStructFieldOffset(fieldTypes, *fieldIndexOpt, target, emitError);
    if (failed(offsetOr))
      return failure();
    return cast<TypedAttr>(Builder(ctx).getIndexAttr(*offsetOr));
  }

  // If the decl is null, we are in an async context and the struct instance is
  // not yet ready.
  if (!resolved.decl)
    return TypedAttr();

  // Find field index using the decl.
  std::optional<uint64_t> fieldIndexOpt =
      resolved.decl.findFieldIndex(fieldName);
  if (!fieldIndexOpt) {
    context.emitMaterializationError(
        "struct '" +
        SymbolTable::getSymbolName(resolved.decl.getOperation()).getValue() +
        "' has no field named '" + fieldName + "'");
    return failure();
  }
  int64_t fieldIndex = static_cast<int64_t>(*fieldIndexOpt);

  // Use generator's field types with rebinding.
  FailureOr<TypedAttr> result = failure();
  context.withEvaluator(
      resolved.decl.getInputParams(), resolved.paramValues,
      [&](ParameterEvaluator &evaluator) {
        std::optional<SmallVector<Type>> fieldTypesOpt =
            rebindFieldTypes(resolved.decl, evaluator);
        if (!fieldTypesOpt) {
          result = TypedAttr();
          return;
        }

        FailureOr<int64_t> offsetOr = computeStructFieldOffset(
            *fieldTypesOpt, fieldIndex, target, emitError);
        if (failed(offsetOr))
          return;
        result = cast<TypedAttr>(Builder(ctx).getIndexAttr(*offsetOr));
      });
  return result;
}
