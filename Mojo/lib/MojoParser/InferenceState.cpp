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

#include "InferenceState.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/ASTType.h"
#include "Mojo/MojoParser/SharedState.h"
#include "Mojo/lib/MojoParser/ExprNodes.h"
#include "Mojo/lib/MojoParser/IREmitter.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

OptionalDiag::OptionalDiag(SharedState &shared, SMLoc defaultLoc,
                           bool discardError)
    : discardError(discardError), diag(std::nullopt) {
  getDiagClosure = [=, &shared,
                    this](std::optional<SMLoc> loc) -> MojoInflightDiag & {
    this->diag = shared.emitError(loc ? *loc : defaultLoc);
    return *this->diag;
  };
}

llvm::function_ref<MojoInflightDiag &(std::optional<SMLoc>)>
OptionalDiag::getDiag() {
  return getDiagClosure;
}

InferenceState::InferenceState(ASTDecl &declScope,
                               ArrayRef<Type> declaredParamTypes,
                               PogListAttr declaredParamPogs, SMLoc defaultLoc,
                               bool discardError,
                               DeferredTypingContext *deferredTypingContext,
                               ArrayRef<ConstraintAttr> additionalAssumptions)
    : deferredTypingContext(deferredTypingContext),
      additionalAssumptions(additionalAssumptions), declScope(declScope),
      shared(declScope.getShared()), evaluator(shared.getParameterEvaluator()),
      declaredParamTypes(declaredParamTypes),
      declaredParamPogs(declaredParamPogs),
      diag(shared, defaultLoc, discardError) {
  for (size_t i = 0; i != declaredParamTypes.size(); ++i)
    evaluator.appendIndexBinding(TypedAttr());
}

void InferenceState::setInferredValue(size_t paramIdx, TypedAttr paramVal,
                                      bool isDefaulted) {
  paramVal = evaluator.getReboundAttribute(paramVal);
  ASTType targetType = evaluator.getReboundType(declaredParamTypes[paramIdx]);

  // If the parameter being inferred is a type, and if the source value is
  // non-mterializable, infer to the materialized type. This ensures that things
  // like `def foo[T: AnyType](a: T):` infer `foo(1)` to T=Int instead of
  // IntLiteral.
  if (!isDefaulted && LIT::isTypeExpr(paramVal)) {
    IREmitter emitter(declScope, EC_TypeParamValue);
    if (auto nmTarget = ASTType(paramVal).getNonmaterializableTarget(shared)) {
      TypedAttr nmTargetAttr = PValue(nmTarget).get();
      FailureOr<TriBool> typeUpCastable = IREmitter::canMetaTypeUpCastTo(
          shared, declScope.getLoc(), nmTargetAttr.getType(), targetType,
          &declScope);
      // If the nonmaterializable type can be upcast to the target type, then
      // make sure we infer to the nonmaterializable type:
      //    def example[T: TrivialRegisterPassable](a: T): ...
      //    example(1) # T should be Int, not IntLiteral.
      if (succeeded(typeUpCastable) && typeUpCastable->isTrue()) {
        SyntheticNode expr(declScope.getLoc());
        paramVal = emitter.emitPValue({nmTargetAttr, &expr}, EC_TypeParamValue,
                                      targetType);
        assert(paramVal && "must be convertible");
        ++numImplicitConversions;
      }
    }
  }

  // Type must be equal
  assert(targetType.isEqualCanon(paramVal.getType()));

  // now align sugar
  if (paramVal.getType() != targetType)
    paramVal = ParamOperatorAttr::getRebind(paramVal, targetType);

  evaluator.overwriteIndexBinding(paramIdx, paramVal);
}

namespace {
/// Walks an attribute/type tree and reports whether it still refers to
/// parameters that are not yet bound. Substituting the bound parameters leaves
/// an unbound one's reference in place, so a surviving `ParamIndexRefAttr` at
/// the current depth is exactly such a parameter.
///
/// The walker mirrors the depth bookkeeping in `ParamIndexRefAttrFinder`
/// (incrementing depth on entering nested parameter scopes) so index refs into
/// inner scopes are correctly ignored. Results are memoized on (depth,
/// opaque-pointer) so deeply shared sub-trees in parameter expressions are not
/// re-walked.
class ConcretenessChecker {
public:
  bool isConcrete(Attribute attr) { return !hasUnresolvedImpl(attr, 0); }

private:
  DenseMap<std::pair<size_t, const void *>, bool> cache;

  template <typename T>
  bool hasUnresolvedImpl(T value, size_t depth) {
    if (!value)
      return false;

    std::pair<size_t, const void *> cacheKey(depth, value.getAsOpaquePointer());
    if (auto it = cache.find(cacheKey); it != cache.end())
      return it->second;

    // Entering a nested parameter scope reframes "our" depth-0 refs as
    // depth-1 from the inner perspective, so bump depth before checking.
    if constexpr (std::is_base_of_v<Type, T>)
      if (isa<ParameterScopeTypeInterface>(value))
        ++depth;
    if constexpr (std::is_base_of_v<Attribute, T>)
      if (isa<ParameterScopeAttrInterface>(value))
        ++depth;

    bool unresolved = false;
    if constexpr (std::is_base_of_v<Attribute, T>) {
      if (auto indexRef = dyn_cast<ParamIndexRefAttr>(value))
        unresolved = indexRef.getDepth() == depth;
    }

    if (!unresolved) {
      value.walkImmediateSubElements(
          [&](Attribute subAttr) {
            unresolved = unresolved || hasUnresolvedImpl(subAttr, depth);
          },
          [&](Type subType) {
            unresolved = unresolved || hasUnresolvedImpl(subType, depth);
          });
    }

    cache[cacheKey] = unresolved;
    return unresolved;
  }
};
} // namespace

LogicalResult InferenceState::checkBodyConstraints(
    ArrayRef<ConstraintAttr> extraConstraints) {
  // Substitute the parameters that are bound, but leave the unbound ones as
  // references rather than inlining their `UnboundAttr` marker.
  ParameterEvaluator partialEvaluator;
  partialEvaluator.setEvaluationContext(evaluator.getEvaluationContext());
  partialEvaluator.setInputDepth(evaluator.getInputDepth());
  for (TypedAttr binding : evaluator.getIndexBindings())
    partialEvaluator.appendIndexBinding(
        binding && sugarIsa<UnboundAttr>(binding) ? TypedAttr() : binding);

  // Check the extra constraints first.
  if (LIT::canDischargeConstraintsInScope(
          declScope, declaredParamPogs, extraConstraints,
          /*origConstraints=*/{}, diag.getDiag(), &bodyUnprovableConstraints,
          &partialEvaluator, additionalAssumptions)
          .isFalse())
    return failure();

  ArrayRef<ConstraintAttr> bodyConstraints =
      declaredParamPogs.getBodyConstraints();
  dischargedBodyConstraints.resize(bodyConstraints.size());

  if (bodyConstraints.empty())
    return success();

  // Filter to the body constraints that are fully concrete once the bound
  // parameters are substituted. Being unproven is only tolerable while a
  // constraint is not fully concrete.
  ConcretenessChecker concreteness;
  SmallVector<ConstraintAttr> concreteConstraints;
  SmallVector<size_t> concreteConstraintIndices;
  concreteConstraints.reserve(bodyConstraints.size());
  concreteConstraintIndices.reserve(bodyConstraints.size());
  for (auto [idx, constraint] : llvm::enumerate(bodyConstraints)) {
    if (concreteness.isConcrete(partialEvaluator.getReboundAttribute(
            constraint.getProposition()))) {
      concreteConstraints.push_back(constraint);
      concreteConstraintIndices.push_back(idx);
    }
  }

  if (concreteConstraints.empty())
    return success();

  // Verify that no concrete constraints are violated. Any unprovable concrete
  // constraints are recorded in `bodyUnprovableConstraints`. The caller is
  // responsible for surfacing any errors if appropriate.
  llvm::BitVector provenConstraints;
  if (LIT::canDischargeConstraintsInScope(
          declScope, declaredParamPogs, concreteConstraints,
          /*origConstraints=*/{}, diag.getDiag(), &bodyUnprovableConstraints,
          &partialEvaluator, additionalAssumptions, &provenConstraints)
          .isFalse())
    return failure();

  for (auto [i, idx] : llvm::enumerate(concreteConstraintIndices))
    if (provenConstraints[i])
      dischargedBodyConstraints.set(idx);

  return success();
}

TypedAttr
VerifiedParamBindings::specializeStructType(StructDeclOp structOp) const {
  assert(*this && "specializing with failed inference bindings");
  return PValue(structOp.bindReference(getValues()));
}

TypedAttr
VerifiedParamBindings::specializeGenerator(TypedAttr generator) const {
  assert(*this && "specializing with failed inference bindings");
  // Special case for structs types: If the generator is a struct meta type,
  // bind the corresponding struct type directly.
  if (auto structMetaType = sugarDynCast<StructMetaType>(generator.getType())) {
    LIT::StructType boundType =
        structMetaType.getType().bindUnbound(getValues());
    return TypeParamAttr::get(boundType, StructMetaType::get(boundType));
  }

  // Otherwise, the type of the generator must be a GeneratorType.
  assert(sugarIsa<GeneratorType>(generator.getType()) &&
         "generator type expected");
  Type specializedType =
      specializeGeneratorType(sugarCast<GeneratorType>(generator.getType()));
  DenseBoolArrayAttr discharged = KGEN::getDenseBoolArrayAttr(
      generator.getContext(), dischargedBodyConstraints);
  TypedAttr result =
      BindParamsAttr::get(generator.getContext(), generator, getValues(),
                          discharged, evaluationContext);
  assert(isEqualCanon(result.getType(), specializedType) &&
         "inferred bind_params type must match verified specialization");
  return result;
}

Type VerifiedParamBindings::specializeGeneratorType(
    GeneratorType genType) const {
  assert(*this && "specializing with failed inference bindings");
  std::optional<PartiallySpecializedInputParams> specializationOpt =
      PartiallySpecializedInputParams::from(
          genType.getInputParamTypes(), getValues(), evaluationContext,
          /*emitErrorFn=*/{}, dischargedBodyConstraints);
  assert(specializationOpt &&
         "verified bindings should specialize this generator");

  PartiallySpecializedInputParams &specialization = *specializationOpt;
  GeneratorType specialized =
      genType.getSpecializedGenerator(specialization, /*emitErrorFn=*/{});
  // Perform generator instantiation only if all parameters are bound and all
  // body constraints are discharged. In addition, for back-compat, we do not
  // eagerly instantiate function generators.
  Type reboundBody = specialized.getBody();
  bool canEagerInstantiate = specialization.unboundParamTypes.empty() &&
                             dischargedBodyConstraints.all() &&
                             !sugarIsa<FuncType>(reboundBody) &&
                             !sugarIsa<FuncLiteralType>(reboundBody);
  if (canEagerInstantiate)
    return reboundBody;
  assert(specialized && "parser generator specialization should succeed");
  return specialized;
}

void InferenceState::dump() const {
  auto &os = llvm::errs() << "ParamInf:\n";
  for (auto [idx, value] : llvm::enumerate(evaluator.getIndexBindings())) {
    os << "  *(0," << idx << ") = ";
    if (value)
      os << value;
    else
      os << "<not yet set> : "
         << const_cast<InferenceState *>(this)->evaluator.getReboundType(
                declaredParamTypes[idx]);
    os << "\n";
  }
}
