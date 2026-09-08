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

#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/KGENDialect/KGENInterfaces.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Support/ErrorOr.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

using namespace M;
using namespace M::KGEN;

/// If `type` is a `!kgen.deferred_type`, attempt to concretize it by building
/// the resolved type string from the wrapped `AttrCtorDeferredAttr` and
/// parsing it as an MLIR type. Returns the concrete type on success, or
/// `type` unchanged if parameters are still unresolved.
static Type
tryConcretizeDeferredType(Type type,
                          ParameterEvaluationContext *evaluationContext) {
  auto deferredType = dyn_cast<MLIRDeferredType>(type);
  if (!deferredType)
    return type;
  auto attrCtor = dyn_cast<AttrCtorDeferredAttr>(deferredType.getAttr());
  if (!attrCtor)
    return type;

  // An escaping parameter reference (a param of an enclosing `def(...)`
  // signature) stays abstract until the callee is instantiated, so the type
  // can't be built yet -- leave it deferred and skip the doomed parse.
  if (EscapingReferenceFinder::check(type))
    return type;

  std::string typeStr;
  llvm::raw_string_ostream os(typeStr);
  for (Attribute str : attrCtor.getStrings()) {
    if (auto strAttr = dyn_cast<StringAttr>(str)) {
      os << strAttr.str();
      continue;
    }
    auto toStrAttr = dyn_cast<ToStringDeferredAttr>(str);
    if (!toStrAttr)
      llvm_unreachable("unexpected attribute type in AttrCtorDeferredAttr");

    // Strip sugar wrapping before printing to get the canonical value.
    Attribute val = SugarAttr::strip(toStrAttr.getAttr());
    bool elideType = toStrAttr.getNeedElideType() != nullptr;
    // Evaluate !kgen.string attrs (e.g. data_to_str) before printing.
    bool evaluatedToString = false;
    if (evaluationContext) {
      if (auto evalAttr = dyn_cast<ContextuallyEvaluatedAttrInterface>(val)) {
        FailureOr<TypedAttr> evaluated =
            evaluationContext->evaluateExpression(evalAttr);
        // A null success means the operand is not ready yet; print it
        // unevaluated so the parse fails and the type stays deferred.
        if (succeeded(evaluated) && *evaluated) {
          val = *evaluated;
          evaluatedToString = isa<StringAttr>(val);
        }
      }
    }
    if (evaluatedToString)
      os << cast<StringAttr>(val).str();
    else if (auto strAttr = dyn_cast<StringAttr>(val); strAttr && elideType)
      os << strAttr.str();
    else
      val.print(os, elideType);
  }

  MLIRContext *ctx = type.getContext();
  Type parsedType;
  {
    mlir::ScopedDiagnosticHandler suppress(ctx, [](Diagnostic &) {});
    parsedType = mlir::parseType(typeStr, ctx);
  }
  if (parsedType)
    return parsedType;
  // Parsing failed: parameters may still be unresolved. Report an error only
  // in a materialization context where all parameters must be concrete.
  if (evaluationContext && evaluationContext->isMaterializationContext())
    evaluationContext->emitMaterializationError(
        "invalid MLIR type in deferred_type: " + typeStr);
  return type;
}

//===----------------------------------------------------------------------===//
// Helper methods for inspecting possibly-parameterized attributes and types.
//===----------------------------------------------------------------------===//

/// Given a parameter expression, walk it and return any references to named
/// parameters.  This fails if an unknown parameter expression exists.
void KGEN::collectParameterReferences(
    Attribute attr, SmallVectorImpl<ParamDeclRefAttr> &results,
    bool &hasCtxEvalExpr, size_t &requiredSignatureDepth) {
  ParameterCollector::Analysis cache;
  ParameterCollector c(cache);
  c.collectUsesFromAttr(attr, results, hasCtxEvalExpr, requiredSignatureDepth);
}

/// Given a potentially-parameterized MLIR type, walk it and return any
/// references to named parameters.
void KGEN::collectParameterReferences(
    Type type, SmallVectorImpl<ParamDeclRefAttr> &results, bool &hasCtxEvalExpr,
    size_t &requiredSignatureDepth) {
  ParameterCollector::Analysis cache;
  ParameterCollector c(cache);
  c.collectUsesFromType(type, results, hasCtxEvalExpr, requiredSignatureDepth);
}

/// Return true if the specified type contains parameter references, e.g.
/// `!kgen.scalar<dt>` returns true, but `!kgen.scalar<f32>` returns false.
///
/// TODO: This isn't an efficient method, it walks the entire type graph without
/// caching.
bool KGEN::isParameterizedType(Type type) {
  SmallVector<ParamDeclRefAttr> paramDecls;
  bool hasCtxEvalExpr = false;
  size_t requiredSignatureDepth = 0;
  collectParameterReferences(type, paramDecls, hasCtxEvalExpr,
                             requiredSignatureDepth);
  return !paramDecls.empty() || hasCtxEvalExpr || requiredSignatureDepth != 0;
}

//===----------------------------------------------------------------------===//
// ParameterEvaluationContext
//===----------------------------------------------------------------------===//

ParameterEvaluationContext::~ParameterEvaluationContext() {}

FailureOr<TypedAttr> ParameterEvaluationContext::evaluateContextSpecific(
    ContextuallyEvaluatedAttrInterface /*attr*/) {
  // Default implementation - no context-specific handling.
  return failure();
}

void ParameterEvaluationContext::emitMaterializationError(
    const Twine & /*message*/) {
  // Base class does nothing - derived classes can override to emit diagnostics.
  // This is only meaningful in materialization contexts where the expression
  // will persist and be used. In non-materialization contexts (e.g.,
  // speculative partial-evaluation during lowering), ill-formed expressions
  // simply persist unevaluated.
}

bool ParameterEvaluationContext::isMaterializationContext() const {
  return false;
}

Operation *ParameterEvaluationContext::resolveConformanceForStruct(
    ResolvedStructHandle resolved, TraitSymbolAttr traitSymbol) {
  Operation *ret = nullptr;
  withEvaluator(resolved.decl.getInputParams(), resolved.paramValues,
                [&](ParameterEvaluator &evaluator) {
                  ret = resolved.decl.lookupConformance(evaluator, traitSymbol);
                });
  return ret;
}

FailureOr<TypedAttr> ParameterEvaluationContext::evaluateExpression(
    ContextuallyEvaluatedAttrInterface attr) {
  // First try context-specific evaluation (handles attrs that need special
  // treatment in specific contexts like parser vs elaborator).
  FailureOr<TypedAttr> contextResult = evaluateContextSpecific(attr);
  if (succeeded(contextResult))
    return contextResult;

  // Fall back to interface-based evaluation.
  return attr.evaluateWithContext(*this);
}

//===----------------------------------------------------------------------===//
// SymTabEvaluationContext
//===----------------------------------------------------------------------===//

FailureOr<ResolvedStructHandle>
SymTabEvaluationContext::resolveStructOp(TypedAttr typeValue,
                                         bool /*acceptAsync*/) {
  // SymTabEvaluationContext does not support async concretization, so
  // acceptAsync is ignored - we always return the generator.
  auto genRef = sugarDynCastIfPresent<TypeGeneratorRefAttr>(
      getTypeRefForTypeValueIfResolved(typeValue));
  if (!genRef)
    return failure();

  auto structDecl =
      symtab.lookupSymbolIn<StructGeneratorOp>(module, genRef.getSymbol());
  if (!structDecl)
    return failure();

  // Return the generator. instance is null since SymTabEvaluationContext
  // doesn't support async concretization.
  return ResolvedStructHandle{
      cast<StructDeclInterface>(structDecl.getOperation()),
      genRef.getParamValues(), nullptr,
      /*instance=*/nullptr};
}

void SymTabEvaluationContext::withEvaluator(
    ArrayRef<ParamDeclAttr> paramDecls, ArrayRef<TypedAttr> paramValues,
    llvm::function_ref<void(ParameterEvaluator &)> callback) {
  ParameterEvaluator evaluator(paramDecls, paramValues);
  evaluator.setEvaluationContext(this);
  callback(evaluator);
}

FuncInterface
SymTabEvaluationContext::resolveFunctionDecl(SymbolRefAttr symbol) {
  return symtab.lookupSymbolIn<GeneratorOp>(module, symbol);
}

FailureOr<TypedAttr> SymTabEvaluationContext::evaluateContextSpecific(
    ContextuallyEvaluatedAttrInterface attr) {
  TypedAttr typedAttr = dyn_cast<TypedAttr>((Attribute)attr);

  // Handle inlined apply operations.
  if (auto pocAttr = sugarDynCast<ParamOperatorAttr>(typedAttr))
    return inlineApply(pocAttr);

  return failure();
}

FailureOr<TypedAttr>
SymTabEvaluationContext::inlineApply(ParamOperatorAttr apply) {
  // if there is any generator that is marked to have an inlined form, we inline
  // it to reach the canonicalized form.
  if (apply.getOpcode() != POC::Apply &&
      apply.getOpcode() != POC::ApplyResultSlot)
    return failure();

  auto cst = dyn_cast<SymbolConstantAttr>(apply.getOperand(0));
  if (!cst)
    return failure();

  Operation *op =
      symtab.lookupSymbolIn(module, cst.getSymbol().getLeafReference());

  // TODO(MOCO-2656): At the moment, only generator will be annotated by
  //`ApplyInliner`, this can be generalized to handle indirect calls to
  //"always_inline("builtin")" (e.g., via trait method) function.
  if (auto func = dyn_cast_or_null<GeneratorOp>(op);
      func && func.getInlinedFormAttr()) {
    TypedAttr inlinedExpr = func.getInlinedFormAttr();
    // Drop the symbol to get the parameter binding;
    ArrayRef<TypedAttr> paramBinding = apply.getOperands().drop_front();

    // Rebind first
    ParameterEvaluator evaluator(func.getInputParams(), cst.getParamValues());
    evaluator.setEvaluationContext(this);
    inlinedExpr = cast<TypedAttr>(evaluator.getReboundAttribute(inlinedExpr));
    if (!paramBinding.empty()) {
      // If generator takes input, we need to further evaluate the inlined
      // expression with the parameter binding provided by the apply.
      inlinedExpr = cast<GeneratorAttr>(inlinedExpr)
                        .getSpecializedGenerator(paramBinding, this)
                        .getInstantiatedValue();
    }

    return inlinedExpr;
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// ParameterEvaluator core implementation.
//===----------------------------------------------------------------------===//

ParameterEvaluator::ParameterEvaluator(ArrayRef<ParamDeclAttr> paramDecls,
                                       ArrayRef<TypedAttr> declBindings) {
  for (auto [decl, value] : llvm::zip(paramDecls, declBindings))
    setDeclBinding(decl, value);
}

ParameterEvaluator::ParameterEvaluator(ArrayRef<TypedAttr> declBindings) {
  for (TypedAttr param : declBindings)
    appendIndexBinding(param);
}

std::pair<IntegerAttr, bool>
ParameterEvaluator::narrowCondOp(Attribute attr, size_t rootDepth) {
  if (auto op = dyn_cast<ParamOperatorAttr>(attr);
      op && op.getOpcode() == POC::Cond) {
    TypedAttr typedCond = op.getOperands().front();
    assert(isScalarOf<KGENDType::kBool>(typedCond.getType()));
    Attribute cond = replaceImpl(typedCond, rootDepth);
    if (!cond)
      return {nullptr, true};
    return {
        dyn_cast<IntegerAttr>(CastToBuiltinAttr::get(cast<TypedAttr>(cond))),
        false};
  }
  return {nullptr, false};
}

Attribute ParameterEvaluator::doReplace(Attribute attr, size_t rootDepth) {
  if (isa<ParameterScopeAttrInterface>(attr))
    ++rootDepth;

  // If a parameter got rebound to an index reference, we need to increase its
  // depth based on the current signature, per STCHDDDOS.
  // FIXME: Is there a better way around this? This previously manifested as
  // unintentional name shadowing problems, but walking here is inefficient.
  auto upbindValue = [&](TypedAttr value) -> TypedAttr {
    if (rootDepth + inputDepth == 0)
      return value;
    IndexDepthAdjuster adjuster(/*adjustDepth=*/rootDepth + inputDepth);
    return adjuster.replace(value);
  };

  // If this is a foldable parameter expression, do it.
  Attribute result = attr;
  if (auto declRef = dyn_cast<ParamDeclRefAttr>(attr)) {
    // If the referenced parameter is not bound, forward the reference.
    auto declRefType = doReplace(declRef.getType(), rootDepth);
    // Forward nullptr from elaborator to signal a skipped node to avoid race
    // condition.
    if (!declRefType)
      return nullptr;
    if (auto it = declBindings.find(declRef.getName());
        it != declBindings.end()) {
      auto resultV = upbindValue(it->second);
      // If we are mapping between a sugared and non-sugared version of the
      // parameter, make sure to keep a consistent type.  This enables us to
      // substitute values into parameter expressions that have sugared and
      // canonical forms.
      if (resultV.getType() != declRefType &&
          isEqualCanon(resultV.getType(), declRefType))
        resultV = ParamOperatorAttr::getRebind(resultV, declRefType);
      result = resultV;
    } else {
      result = ParamDeclRefAttr::get(declRef.getName(), declRefType);
    }
  } else if (auto indexRef = dyn_cast<ParamIndexRefAttr>(attr);
             indexRef && indexRef.getDepth() == rootDepth) {
    assert(indexRef.getIndex() < indexBindings.size() &&
           "parameter index out of range");
    auto indexRefType = doReplace(indexRef.getType(), rootDepth);
    // Forward nullptr from elaborator to signal a skipped node to avoid race
    // condition.
    if (!indexRefType)
      return nullptr;
    TypedAttr resultV = indexBindings[indexRef.getIndex()];
    if (resultV) {
      resultV = cast<TypedAttr>(upbindValue(resultV));
      // If we are mapping between a sugared and non-sugared version of the
      // parameter, make sure to keep a consistent type.  This enables us to
      // substitute values into parameter expressions that have sugared and
      // canonical forms.
      if (resultV.getType() != indexRefType &&
          isEqualCanon(resultV.getType(), indexRefType))
        resultV = ParamOperatorAttr::getRebind(resultV, indexRefType);
    } else if (indexRefType == indexRef.getType()) {
      resultV = indexRef; // Reuse the IndexRef if the type matches.
    } else {              // Otherwise rebuild it.
      resultV = ParamIndexRefAttr::get(indexRef.getDepth(), indexRef.getIndex(),
                                       indexRefType);
    }
    result = resultV;
  } else if (isa<MLIROpAttr>(attr)) {
    // Expression functions and MLIR operation expressions are isolated from
    // above, so don't collect from them.
  } else if (auto [condVal, skip] = narrowCondOp(attr, rootDepth);
             condVal || skip) {
    if (skip)
      return nullptr;
    // If condition is a constant rebind only one of the clauses.
    auto op = cast<ParamOperatorAttr>(attr);
    if (condVal.getValue().isZero())
      result = replaceImpl(op.getOperands()[2], rootDepth);
    else
      result = replaceImpl(op.getOperands()[1], rootDepth);
    if (!result)
      return nullptr;
  } else if (auto bindParams = dyn_cast<BindParamsAttr>(attr)) {
    bool changed = false;
    // BindParamsAttr must always be re-created using an Evaluation Context.
    SmallVector<TypedAttr> newParamValues;
    for (TypedAttr param : bindParams.getParamValues()) {
      auto newParam = replaceImpl(param, rootDepth);
      if (!newParam)
        return nullptr;
      changed |= newParam != param;
      newParamValues.push_back(cast<TypedAttr>(newParam));
    }

    Attribute newGenerator = replaceImpl(bindParams.getGenerator(), rootDepth);
    if (!newGenerator)
      return nullptr;
    changed |= newGenerator != bindParams.getGenerator();

    Type newType = replaceImpl(bindParams.getType(), rootDepth);
    if (!newType)
      return nullptr;
    changed |= newType != bindParams.getType();

    if (changed) {
      TypedAttr result = BindParamsAttr::get(
          bindParams.getContext(), cast<TypedAttr>(newGenerator),
          newParamValues, bindParams.getDischarged(), getEvaluationContext());
      if (!result)
        return nullptr;
      assert(isEqualCanon(result.getType(), newType) &&
             "inferred bind_params type must match rebound type");
      return result;
    }
    return bindParams;
  } else {
    SmallVector<Attribute, 16> newAttrs;
    SmallVector<Type, 16> newTypes;
    // Stop walking and propagate failures when they occur.
    bool changed = false;
    bool failed = false;
    attr.walkImmediateSubElements(
        [&](Attribute attr) {
          if (failed)
            return;
          Attribute newAttr = replaceImpl(attr, rootDepth);
          if (!newAttr)
            failed = true;
          changed |= newAttr != attr;
          newAttrs.push_back(newAttr);
        },
        [&](Type type) {
          if (failed)
            return;
          Type newType = replaceImpl(type, rootDepth);
          if (!newType)
            failed = true;
          changed |= newType != type;
          newTypes.push_back(newType);
        });
    if (failed)
      return nullptr;
    if (changed)
      result = attr.replaceImmediateSubElements(newAttrs, newTypes);
  }

  // If an evaluatable parameter persisted, try to simplify it with additional
  // context.
  if (evaluationContext) {
    if (auto attr = dyn_cast<ContextuallyEvaluatedAttrInterface>(result)) {
      FailureOr<TypedAttr> expr = evaluationContext->evaluateExpression(attr);
      if (succeeded(expr))
        result = *expr;
    }
  }
  return result;
}

Type ParameterEvaluator::doReplace(Type type, size_t rootDepth) {
  Type result = type;

  if (isa<ParameterScopeTypeInterface>(type))
    ++rootDepth;

  // Rebind types in aggregates that implement SubElementTypeInterface.
  SmallVector<Attribute, 16> newAttrs;
  SmallVector<Type, 16> newTypes;
  bool changed = false;
  // Stop walking and propagate failures when they occur.
  bool failed = false;
  type.walkImmediateSubElements(
      [&](Attribute attr) {
        if (failed)
          return;
        Attribute newAttr = replaceImpl(attr, rootDepth);
        if (!newAttr)
          failed = true;
        changed |= newAttr != attr;
        newAttrs.push_back(newAttr);
      },
      [&](Type type) {
        if (failed)
          return;
        Type newType = replaceImpl(type, rootDepth);
        if (!newType)
          failed = true;
        changed |= newType != type;
        newTypes.push_back(newType);
      });
  if (failed)
    return nullptr;
  if (changed)
    result = type.replaceImmediateSubElements(newAttrs, newTypes);

  result = tryConcretizeDeferredType(result, evaluationContext);

  // If an evaluatable type persists, try to simplify it with additional
  // context.
  if (evaluationContext) {
    if (auto evalType = dyn_cast<ContextuallyEvaluatedTypeInterface>(result)) {
      FailureOr<Type> foldedType =
          evalType.evaluateWithContext(*evaluationContext);
      if (succeeded(foldedType))
        result = *foldedType;
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// ParameterEvaluator debugging support.
//===----------------------------------------------------------------------===/r

// Note: this dumps out in non-stable hash table order, only use for debugging
// purposes!
void ParameterEvaluator::dump() const {
  auto &os = llvm::errs();
  os << "ParameterEvaluator: \n";
  for (auto [name, value] : declBindings)
    os << "  " << name << " = " << value << "\n";
  for (auto [idx, value] : llvm::enumerate(indexBindings))
    os << "  *(0," << idx << ") = " << value << "\n";
}

//===----------------------------------------------------------------------===//
// Helper methods involving parameter evaluation.
//===----------------------------------------------------------------------===//

std::optional<PartiallySpecializedInputParams>
PartiallySpecializedInputParams::from(
    ArrayRef<Type> paramTypes, ArrayRef<TypedAttr> paramBindings,
    ParameterEvaluationContext *evaluationContext,
    function_ref<InFlightDiagnostic()> emitErrorFn,
    const llvm::BitVector &dischargedBodyConstraints) {
  // Verify the number of input parameters.
  if (paramBindings.size() != paramTypes.size()) {
    assert(emitErrorFn && "unexpected invalid bindings");
    emitErrorFn() << "generator type expects " << paramTypes.size()
                  << " parameters but got bindings for "
                  << paramBindings.size();
    return std::nullopt;
  }

  PartiallySpecializedInputParams result;
  result.dischargedBodyConstraints = dischargedBodyConstraints;
  ParameterEvaluator &evaluator = result.evaluator;
  SmallVector<Type, 16> &unboundParamTypes = result.unboundParamTypes;
  llvm::BitVector &boundParams = result.boundParams;
  boundParams.resize(paramTypes.size());

  evaluator.setEvaluationContext(evaluationContext);
  evaluator.setInputDepth(1);
  IndexDepthAdjuster minusOneAdjuster(/*adjustDepth=*/-1);

  auto remapType = [&](Type type) -> Type {
    return evaluator.getReboundType(type);
  };

  for (auto [paramNo, valueX, type] :
       llvm::enumerate(paramBindings, paramTypes)) {
    auto value = valueX;
    // Bound parameters are allowed to refine the type of subsequent
    // parameters, e.g. in `<ty: type, fn: () -> !kgen.param<ty>>`, the
    // expected type of the second parameter will be refined when the first
    // parameter is bound.
    auto remappedDeclType = remapType(type);

    // Even if we're skipping a binding site, we still need to remap the decl.
    // TODO: Disallow UnboundAttr for skipping bindings.
    if (::isa<UnboundAttr>(value)) {
      // Set the binding to a declref of the thing itself - that will keep it
      // from becoming #kgen.unbound.  This #param.index.ref will have a level
      // of -1, and we adjust the level of its type by -1 so it balances out
      // correctly when referenced.
      auto adjustedParamType = minusOneAdjuster.replace(remappedDeclType);
      auto value = ParamIndexRefAttr::get(
          /*depth=*/-1, unboundParamTypes.size(), adjustedParamType);
      unboundParamTypes.push_back(remappedDeclType);
      evaluator.appendIndexBinding(value);
    } else {
      // We must remap the value type being provided as well, because it may
      // be referring to outer-context indexed parameters, whose depth will be
      // increased when substituted into this signature, per STCHDDDOS.
      auto valueType = value.getType();
      remappedDeclType = minusOneAdjuster.replace(remappedDeclType);
      if (valueType != remappedDeclType &&
          !isEqualCanon(valueType, remappedDeclType)) {
        if (!emitErrorFn)
          return {};
        emitErrorFn() << "caller input parameter #" << paramNo << " has type "
                      << valueType << " but callee expected type "
                      << remappedDeclType;
        return {};
      }

      // Realign sugar if necessary.
      if (valueType != remappedDeclType)
        value = ParamOperatorAttr::getRebind(value, remappedDeclType);

      evaluator.appendIndexBinding(value);
      boundParams.set(paramNo);
    }
  }

  return result;
}
