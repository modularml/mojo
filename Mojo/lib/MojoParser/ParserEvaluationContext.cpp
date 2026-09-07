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

#include "ParserEvaluationContext.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/MojoParser/SharedState.h"
#include "Traits.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// Struct Reflection Helpers
//===----------------------------------------------------------------------===//

FailureOr<ResolvedStructHandle>
ParserEvaluationContext::resolveStructOp(TypedAttr typeValue,
                                         bool /*acceptAsync*/) {
  // Parser doesn't support async concretization, so acceptAsync is ignored -
  // we always return the generator.
  auto typeParam = sugarDynCast<TypeParamAttr>(typeValue);
  if (!typeParam)
    return failure();

  // Typically, this is a LIT struct type.
  if (auto resolvedType =
          sugarDynCast<LIT::StructType>(typeParam.getTypeValue())) {
    ASTDecl &astDecl =
        shared.declResolver->getDeclForTypeSymbol(resolvedType.getSymbol());
    auto structDeclOp = cast<StructDeclOp>(astDecl.getIfOperation());

    if (failed(shared.declResolver->resolveBody(astDecl, astDecl.getLoc())))
      return failure();

    // Return the decl. instance is null since this is not an IREvaluator
    // context.
    return ResolvedStructHandle{
        cast<StructDeclInterface>(structDeclOp.getOperation()),
        resolvedType.getParamValues(), &astDecl,
        /*instance=*/nullptr};
  }
  return failure();
}

FuncInterface
ParserEvaluationContext::resolveFunctionDecl(SymbolRefAttr symbol) {
  ASTDecl *decl = shared.declResolver->getDeclForFuncSymbol(symbol);
  if (!decl)
    return nullptr;
  return dyn_cast_or_null<FuncInterface>(decl->getIfOperation());
}

Operation *ParserEvaluationContext::resolveConformanceForStruct(
    ResolvedStructHandle resolved, TraitSymbolAttr traitSymbol) {
  auto *astDecl = static_cast<ASTDecl *>(resolved.handle);

  // Typically, this is a StructDeclOp created by the parser.
  if (isa<StructDeclOp>(astDecl->getIfOperation())) {
    auto conformanceDecls =
        astDecl->lookupInCurrentScope(traitSymbol.getFlattenedName());
    if (conformanceDecls.empty())
      return nullptr;
    // TODO: we might need to allow multiple conformance ops for parametric
    // traits, and we need to filter it here.
    assert(conformanceDecls.size() == 1 && "expected exactly one conformance");
    ASTDecl &conformDecl = *conformanceDecls.front();
    if (failed(shared.declResolver->resolveBody(conformDecl,
                                                conformDecl.getLoc())))
      return nullptr;

    return conformDecl.getIfOperation();
  }

  return nullptr;
}

void ParserEvaluationContext::withEvaluator(
    ArrayRef<ParamDeclAttr> paramDecls, ArrayRef<TypedAttr> paramValues,
    llvm::function_ref<void(ParameterEvaluator &)> callback) {
  ParameterEvaluator evaluator(paramDecls, paramValues);
  evaluator.setEvaluationContext(this);
  callback(evaluator);
}

FailureOr<TypedAttr> ParserEvaluationContext::evaluateContextSpecific(
    ContextuallyEvaluatedAttrInterface attr) {
  TypedAttr typedAttr = dyn_cast<TypedAttr>((Attribute)attr);

  // Handle TypeConformsToTraitAttr.
  if (auto conformsTo =
          sugarDynCastIfPresent<TypeConformsToTraitAttr>(typedAttr)) {
    auto traitDeclResolver = [&](SymbolRefAttr symbol) -> TraitDeclOp {
      ASTDecl &decl = shared.declResolver->getDeclForTypeSymbol(symbol);
      return cast<TraitDeclOp>(decl.getIfOperation());
    };

    // Try LIT-specific trait type folding first, then fall back to the attr
    // folder for struct resolution.
    FailureOr<TypedAttr> result =
        simplifyConformsToAgainstTypeValue(conformsTo, traitDeclResolver);
    if (succeeded(result))
      return result;

    return conformsTo.evaluateWithContext(*this);
  }

  // For now, only fold IsSubTraitAttr in parser context, un-foldable parametric
  // trait will be erased after lowerLIT.
  // TODO: preserve trait symbols in KGEN.
  if (auto isRefinedTrait =
          sugarDynCastIfPresent<IsRefinedTypeAttr>(typedAttr)) {
    auto sourceType = ASTType(isRefinedTrait.getSourceType());
    if (auto targetTrait =
            sugarDynCast<TraitType>(ASTType(isRefinedTrait.getTargetType()))) {
      TriBool foldResult = sourceType.doesConformTo(targetTrait, shared,
                                                    /*callerAssumptions=*/{});
      if (foldResult.isUnknown())
        return failure(); // un-foldable

      return TypedAttr(SIMDAttr::getScalarBool(isRefinedTrait.getContext(),
                                               foldResult.isTrue()));
    }
  }

  // Handle DowncastAttr.
  if (auto downcast = sugarDynCastIfPresent<DowncastAttr>(typedAttr)) {
    if (TypedAttr folded = LIT::foldDowncastToStructType(downcast))
      return folded;
    // If we are downcasting a more-refined trait to a less-refined trait, use
    // the more refined trait.
    if (TraitType toTrait = sugarDynCast<TraitType>(downcast.getType())) {
      auto fromType = ASTType(downcast.getInputTypeValue());
      bool fromImpliesTo = fromType.doesConformTo(toTrait, shared, {}).isTrue();
      if (fromImpliesTo)
        return UpcastAttr::get(downcast.getType(),
                               downcast.getInputTypeValue());
      // canonicalize downcast<:ft T, tt> into
      // upcast<:tt, downcast<:ft T, tt & ft>
      if (auto from = sugarDynCast<TraitType>(fromType.extractMetaType())) {
        SmallVector<TraitSymbolAttr> symbols(from.getSymbols());
        llvm::append_range(symbols, toTrait.getSymbols());
        sortAndDeduplicateTraitSymbols(symbols);

        auto allTraits = TraitType::get(from.getContext(), symbols, {});

        auto ret = UpcastAttr::get(
            downcast.getType(),
            DowncastAttr::get(allTraits, downcast.getInputTypeValue()));

        return ret;
      }
    }
  }

  // Otherwise, this is not something we can evaluate, which is ok, because
  // the parser won't be able to evaluate everything. The user is expected to
  // use rebind in these cases.
  return failure();
}
