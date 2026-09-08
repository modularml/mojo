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
// This pass lowers a variety of high level Mojo types in the 'lit' dialect
// to lower level KGEN abstractions.  Notably, this eliminates symbol based
// struct references (in favor of `!kgen.struct`), `!lit.ref` => `!kgen.pointer`
// etc.  This runs immediately after the LowerLIT pass.
//
//===----------------------------------------------------------------------===//

#include "LowerLITTypes.h"

#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Support/Compiler/DomainAwareReplacer.h"
#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "Config/Version.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

namespace {
/// A DomainAwareReplacer that distinguishes the two roles of a mojo type:
/// As a value or as a type itself. The different roles require different Type
/// representations in KGEN.
class LowerLITReplacer : public DomainAwareReplacer {
public:
  enum TypeDomain : DomainId {
    AsType,  // Types are used as types.
    AsValue, // Types are used as values.
  };

  LowerLITReplacer() {
    // Parameters should always use the AsType domain so that their types are
    // lowered as types.
    addNonRecursiveReplacement(
        [&](TypedAttr attr) -> FailureOr<Attribute> {
          return replace(attr, TypeDomain::AsType);
        },
        TypeDomain::AsValue);
  }

  /// Add a replacement that skips recursing down the replaced result.
  /// The replacement callback itself must handle any further replacing by
  /// calling back into this DomainAwareReplacer. This way the exact replacer
  /// domain can be controlled at each replacement step.
  template <typename FnT,
            typename T = typename llvm::function_traits<
                std::decay_t<FnT>>::template arg_t<0>,
            typename BaseT = std::conditional_t<std::is_base_of_v<Attribute, T>,
                                                Attribute, Type>,
            typename ResultT = std::invoke_result_t<FnT, T>>
  std::enable_if_t<std::is_convertible_v<ResultT, FailureOr<BaseT>>>
  addNonRecursiveReplacement(FnT &&callback, DomainId domain) {
    addReplacement(mlir::AttrTypeReplacer::ReplaceFn<BaseT>(
                       [f = std::forward<FnT>(callback)](BaseT base)
                           -> mlir::AttrTypeReplacer::ReplaceFnResult<BaseT> {
                         if constexpr (std::is_same_v<T, BaseT>) {
                           FailureOr<BaseT> ret = f(base);
                           if (succeeded(ret))
                             return {{*ret, WalkResult::skip()}};
                           else
                             return {{nullptr, WalkResult::interrupt()}};
                         }
                         if (auto derived = dyn_cast<T>(base)) {
                           FailureOr<BaseT> ret = f(derived);
                           if (succeeded(ret))
                             return {{*ret, WalkResult::skip()}};
                           else
                             return {{nullptr, WalkResult::interrupt()}};
                         }
                         return {};
                       }),
                   domain);
  }

  /// Add a domain-agnostic replacement function.
  /// Since TypedAttr replacements only need to happen in the type domain, any
  /// replacement functions for TypedAttrs are only registered in one replacer.
  /// Other replacers must be registered in both.
  template <typename FnT,
            typename T = typename llvm::function_traits<
                std::decay_t<FnT>>::template arg_t<0>,
            typename BaseT = std::conditional_t<std::is_base_of_v<Attribute, T>,
                                                Attribute, Type>,
            typename ResultT = std::invoke_result_t<FnT, T>>
  std::enable_if_t<std::is_convertible_v<ResultT, FailureOr<BaseT>>>
  addInferredDomainNonRecursiveReplacement(FnT &&callback) {
    addNonRecursiveReplacement(std::forward<FnT>(callback), TypeDomain::AsType);
    if constexpr (!std::is_same_v<BaseT, TypedAttr>)
      addNonRecursiveReplacement(std::forward<FnT>(callback),
                                 TypeDomain::AsValue);
  }

  /// Convenience helper for replacing parameters and returning parameters.
  FailureOr<TypedAttr> replaceParameter(TypedAttr attr) {
    FailureOr<Attribute> attrOr = replace(attr, TypeDomain::AsType);
    if (failed(attrOr))
      return failure();

    return cast<TypedAttr>(*attrOr);
  }

  /// Table of "erased" struct layouts keyed by the concrete LIT struct
  /// type (parameter values included). Pointer and function-generator fields
  /// are erased to pointer-sized indirections.
  llvm::DenseMap<LIT::StructType, Type> erasedStructs;
};

using TypeDomain = LowerLITReplacer::TypeDomain;

} // namespace

//===----------------------------------------------------------------------===//
// ParameterEvaluationContext
//===----------------------------------------------------------------------===//

namespace {
/// Evaluation context for LowerLIT that maps LIT struct types to KGEN struct
/// generators via the StructDecls mapping.
class LowerLITEvaluationContext : public SymTabEvaluationContext {
public:
  LowerLITEvaluationContext(ModuleOp module,
                            mlir::LockedSymbolTableCollection &symtab,
                            StructDecls &decls)
      : SymTabEvaluationContext(module, symtab), decls(decls) {}

protected:
  /// Resolve LIT struct types to KGEN struct generators using the decls
  /// mapping.
  FailureOr<ResolvedStructHandle> resolveStructOp(TypedAttr typeValue,
                                                  bool acceptAsync) override;

  /// Handle DowncastAttr so that conforms_to expressions with
  /// trait-constrained type parameters can resolve during LowerLIT.
  FailureOr<TypedAttr>
  evaluateContextSpecific(ContextuallyEvaluatedAttrInterface attr) override;

private:
  StructDecls &decls;
};
} // namespace

static FailureOr<Type>
lowerStructType(StructDecls &decls, LowerLITReplacer &replacer,
                ParameterEvaluationContext &evalContext, MLIRContext *ctx,
                Type noneType, LIT::StructType ref, bool eraseIndirections) {
  StructDecl &decl = decls.get(ref.getName());
  // Substitute the given parameters in.
  ParameterEvaluator evaluator(decl.decls, ref.getParamValues());
  evaluator.setEvaluationContext(&evalContext);

  // For the outer (non-erased) lowering, precompute and cache the erased layout
  // so the AsType cycle-breaker can substitute it when the struct re-enters
  // itself through an indirection.
  if (!eraseIndirections && decl.needsErasure &&
      !replacer.erasedStructs.contains(ref)) {
    FailureOr<Type> erasedOr =
        lowerStructType(decls, replacer, evalContext, ctx, noneType, ref,
                        /*eraseIndirections=*/true);
    if (failed(erasedOr))
      return failure();
    replacer.erasedStructs[ref] = *erasedOr;
  }

  SmallVector<Type> fieldTypes;
  for (Type type : llvm::make_second_range(decl.fields)) {
    if (auto ptrType = dyn_cast<PointerType>(type)) {
      fieldTypes.push_back(PointerType::get(
          noneType, evaluator.getReboundAttribute(ptrType.getAddressSpace()),
          ptrType.getIsNonNull()));
      continue;
    }

    Type reboundType = evaluator.getReboundType(type);
    if (!reboundType)
      return failure();
    if (eraseIndirections &&
        isa<LIT::StructType, FuncTypeGeneratorType>(reboundType)) {
      fieldTypes.push_back(PointerType::get(noneType));
      continue;
    }

    fieldTypes.push_back(reboundType);
  }
  if (decl.isSingleElement()) {
    return replacer.replace(fieldTypes.front(), LowerLITReplacer::AsType);
  }
  // Replace each field type individually, then create the struct.
  SmallVector<Type> replacedTypes;
  replacedTypes.reserve(fieldTypes.size());
  for (Type t : fieldTypes) {
    auto replaced = replacer.replace(t, LowerLITReplacer::AsType);
    if (failed(replaced) || !*replaced)
      return failure();
    replacedTypes.push_back(*replaced);
  }
  TypedAttr reboundAlignment = evaluator.getReboundAttribute(decl.minAlignment);
  FailureOr<TypedAttr> loweredAlignmentOr =
      replacer.replaceParameter(reboundAlignment);
  if (failed(loweredAlignmentOr))
    return failure();
  // Resolve the parametric isMemoryOnly through the evaluator.
  TypedAttr reboundIsMemoryOnly =
      evaluator.getReboundAttribute(decl.isMemoryOnlyAttr);
  FailureOr<TypedAttr> loweredIsMemoryOnlyOr =
      replacer.replaceParameter(reboundIsMemoryOnly);
  if (failed(loweredIsMemoryOnlyOr))
    return failure();
  return KGEN::StructType::get(ctx, replacedTypes, *loweredIsMemoryOnlyOr,
                               *loweredAlignmentOr);
}

FailureOr<ResolvedStructHandle>
LowerLITEvaluationContext::resolveStructOp(TypedAttr typeValue,
                                           bool /*acceptAsync*/) {
  // LowerLITEvaluationContext does not support async concretization, so
  // acceptAsync is ignored - we always return the generator.

  // We can only resolve if the type reference is a resolved LIT struct type.
  auto typeParam = sugarDynCast<TypeParamAttr>(typeValue);
  if (!typeParam)
    return failure();

  auto structType = sugarDynCast<LIT::StructType>(typeParam.getTypeValue());
  if (!structType)
    return SymTabEvaluationContext::resolveStructOp(typeValue, false);

  SymbolRefAttr structDeclRef = structType.getSymbol();
  StringAttr leafName = structDeclRef.getLeafReference();

  auto it = decls.structDecls.find(leafName);
  if (it != decls.structDecls.end()) {
    auto structDecl =
        symtab.lookupSymbolIn<StructGeneratorOp>(module, it->second.symRef);
    if (!structDecl)
      return failure();
    return ResolvedStructHandle{
        cast<StructDeclInterface>(structDecl.getOperation()),
        structType.getParamValues(), nullptr,
        /*instance=*/nullptr};
  }

  return SymTabEvaluationContext::resolveStructOp(typeValue, false);
}

FailureOr<TypedAttr> LowerLITEvaluationContext::evaluateContextSpecific(
    ContextuallyEvaluatedAttrInterface attr) {
  TypedAttr typedAttr = dyn_cast<TypedAttr>((Attribute)attr);

  // Fold DowncastAttr when the input is a concrete struct type value. Unwrap
  // and expose the concrete type value, which can be used to further simplify
  // things like conforms_to.
  if (auto downcast = sugarDynCastIfPresent<DowncastAttr>(typedAttr))
    if (TypedAttr folded = LIT::foldDowncastToStructType(downcast))
      return folded;

  return SymTabEvaluationContext::evaluateContextSpecific(attr);
}

//===----------------------------------------------------------------------===//
// Type Lowering
//===----------------------------------------------------------------------===//

/// Populate `replacer` with the lowering patterns for attributes and types
/// from the computed lowerings for each struct decl.
static void populateReplacer(StructDecls &decls, LowerLITReplacer &replacer,
                             ParameterEvaluationContext &evalContext,
                             MLIRContext *ctx) {
  auto typeType = TypeType::get(ctx);
  auto emptyStructType = KGEN::StructType::get(ctx, ArrayRef<Type>{});
  auto emptyStruct = StructAttr::get({}, emptyStructType);
  auto noneType = KGEN::NoneType::get(ctx);

  replacer.addInferredDomainNonRecursiveReplacement(
      [&replacer, evalCtxPtr = &evalContext](
          BindParamsAttr bindParams) -> FailureOr<Attribute> {
        // We always simplify BindParamsAttr against a evaluation context.
        SmallVector<TypedAttr> loweredParams;
        for (TypedAttr param : bindParams.getParamValues()) {
          auto replaced = replacer.replace(param, TypeDomain::AsType);
          if (failed(replaced))
            return failure();
          loweredParams.push_back(cast<TypedAttr>(*replaced));
        }
        auto generatorOr =
            replacer.replace(bindParams.getGenerator(), TypeDomain::AsType);
        if (failed(generatorOr))
          return failure();

        // BindParamsAttr has to be constructed with an evaluation context to
        // fold properly.
        TypedAttr evaluated = BindParamsAttr::get(
            bindParams.getContext(), cast<TypedAttr>(*generatorOr),
            loweredParams, bindParams.getDischarged(), evalCtxPtr);
        return evaluated;
      });

  // TypeParamAttr dispatches replacing to different domains.
  replacer.addInferredDomainNonRecursiveReplacement(
      [&replacer](TypeParamAttr typeValue) -> FailureOr<Attribute> {
        auto typeValueOr =
            replacer.replace(typeValue.getTypeValue(), TypeDomain::AsValue);
        auto mlirTypeOr =
            replacer.replace(typeValue.getMlirType(), TypeDomain::AsType);
        auto typeOr = replacer.replace(typeValue.getType(), TypeDomain::AsType);

        if (failed(typeValueOr) || failed(mlirTypeOr) || failed(typeOr))
          return failure();

        return TypeParamAttr::get(*typeValueOr, *mlirTypeOr, *typeOr);
      });

  // NOTE: Downcast becomes an no-op after lower-lit. However, we should
  // probably keep the attr till elaboration time after we preserve traits
  // properly in KGEN for a better error message.
  // We simply strip all downcast at the moment otherwise all downcasts will be
  // in same (useless) form of `#downcast<T> : !kgen.type` anyway.
  //
  // TODO: preserve trait symbol in KGEN for downcast/conforms_to/is_sub_trait.
  replacer.addInferredDomainNonRecursiveReplacement(
      [&replacer](DowncastAttr downcast) -> FailureOr<Attribute> {
        auto typeOr = replacer.replace(downcast.getType(), TypeDomain::AsType);
        if (failed(typeOr))
          return failure();
        auto downcastValOr =
            replacer.replaceParameter(downcast.getInputTypeValue());
        if (failed(downcastValOr))
          return failure();
        // Since we are erasing the trait target type, the downcast becomes
        // essentially an upcast to type.type
        return UpcastAttr::get(*typeOr, *downcastValOr);
      });
  replacer.addInferredDomainNonRecursiveReplacement(
      [](IsRefinedTypeAttr isRefinedTrait) -> FailureOr<Attribute> {
        return SIMDAttr::getScalarBool(isRefinedTrait.getContext(), true);
      });

  // ParamRefTypes should be TypeValueType if in the value domain.
  replacer.addNonRecursiveReplacement(
      [&replacer](ParamType paramRef) -> FailureOr<Type> {
        auto paramRefOr = replacer.replaceParameter(paramRef.getParam());
        if (failed(paramRefOr))
          return failure();
        return TypeValueType::get(*paramRefOr);
      },
      TypeDomain::AsValue);

  // The param types of a GeneratorType are always types, not values. Only the
  // body is lowered in the enclosing domain, so a value-domain generator has
  // value-domain argument/result types (its body) while its parameter decl
  // types stay in the type domain. Keeping param decl types in the type domain
  // matches how `ParamIndexRefAttr` types are lowered (always as types), so
  // `verify-parameters` sees index references whose types agree with the
  // parameter declarations they point to.
  for (TypeDomain domain : {TypeDomain::AsType, TypeDomain::AsValue}) {
    // Simply report the error after cycle detected.
    replacer.addCycleBreaker(
        [&decls](Type t) -> std::optional<Type> {
          auto structTp = dyn_cast<LIT::StructType>(t);
          if (structTp) {
            // Simply return a nullptr to signal a error has occurs.
            mlir::emitError(decls.get(structTp.getName()).loc,
                            "struct has recursive reference to itself");
            return Type();
          }
          // Should be unreachable? must be a aggregated type in order to have
          // recursive reference.
          return std::nullopt;
        },
        domain);

    auto replaceAsType = [&replacer](Type type) {
      return replacer.replace(type, TypeDomain::AsType);
    };
    replacer.addNonRecursiveReplacement(
        [domain, replaceAsType,
         &replacer](GeneratorType gen) -> FailureOr<Type> {
          SmallVector<FailureOr<Type>> inputParamTypesOr(
              map_range(gen.getInputParamTypes(), replaceAsType));
          if (llvm::any_of(inputParamTypesOr, failed))
            return failure();

          SmallVector<Type> inputParamTypes = llvm::map_to_vector(
              inputParamTypesOr, [](FailureOr<Type> t) { return *t; });
          Attribute metadata = gen.getParamListAttrs();
          if (metadata) {
            auto metadataOr = replacer.replace(metadata, domain);
            if (failed(metadataOr))
              return failure();
            metadata = *metadataOr;
          }
          auto bodyOr = replacer.replace(gen.getBody(), domain);
          if (failed(bodyOr))
            return failure();

          return GeneratorType::get(inputParamTypes, *bodyOr, metadata);
        },
        domain);
  }

  // All metatypes lower to `!kgen.type`.
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](StructMetaType) { return typeType; });
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](StructMetaMetaType) { return typeType; });
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](AnyTraitType) { return typeType; });
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](FnLiteralTypeGeneratorMetaType) { return typeType; });
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](FnLiteralTypeGeneratorMetaMetaType) { return typeType; });
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](NonStructTypeType) { return typeType; });

  // #lit.ref.pack => #kgen.struct
  replacer.addInferredDomainNonRecursiveReplacement(
      [&replacer](RefPackAttr refPack) -> FailureOr<Attribute> {
        SmallVector<TypedAttr> loweredElts;
        loweredElts.reserve(refPack.getValues().size());
        for (TypedAttr elt : refPack.getValues()) {
          auto eltOr = replacer.replaceParameter(elt);
          if (failed(eltOr))
            return failure();
          loweredElts.push_back(*eltOr);
        }
        FailureOr<Type> typeOr =
            replacer.replace(refPack.getType(), TypeDomain::AsType);
        if (failed(typeOr))
          return failure();
        return StructAttr::get(loweredElts, cast<KGEN::StructType>(*typeOr));
      });

  // !lit.ref.pack<:param_list<!kgen.type> types, owned_in_mem, mut origin, 42>
  // => !kgen.struct<variadic_ptr_map(types), 42>
  replacer.addInferredDomainNonRecursiveReplacement(
      [&replacer](RefPackType ref) -> FailureOr<Type> {
        auto variadicOr = replacer.replaceParameter(ref.getVariadic());
        auto addrSpaceOr = replacer.replaceParameter(ref.getAddressSpace());
        if (failed(variadicOr) || failed(addrSpaceOr))
          return failure();
        auto mapped = ParamOperatorAttr::get(POC::VariadicPtrMap, *variadicOr,
                                             *addrSpaceOr);
        return KGEN::StructType::get(ref.getContext(), mapped,
                                     /*memOnly=*/false, /*minAlign*/ {},
                                     /*isParamPack=*/true);
      });

  // !lit.ref -> !kgen.pointer
  for (TypeDomain domain : {TypeDomain::AsType, TypeDomain::AsValue})
    replacer.addNonRecursiveReplacement(
        [domain, &replacer](RefType ref) -> FailureOr<Type> {
          auto elemTpOr = replacer.replace(ref.getElementType(), domain);
          auto addrSpaceOr = replacer.replaceParameter(ref.getAddressSpace());
          if (failed(elemTpOr) || failed(addrSpaceOr))
            return failure();
          return PointerType::get(*elemTpOr, *addrSpaceOr);
        },
        domain);

  // Replace all origin attributes with empty structs. These attributes are
  // all terminal.
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](AnyOriginAttr) { return emptyStruct; });
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](StaticOriginAttr) { return emptyStruct; });
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](ComptimeOriginAttr) { return emptyStruct; });
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](OriginUnionAttr) { return emptyStruct; });
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](OriginMutCastAttr) { return emptyStruct; });
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](ImplicitOriginRefAttr) { return emptyStruct; });
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](OriginSetAttr) { return emptyStruct; });
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](EllipsisAttr) { return emptyStruct; });
  replacer.addInferredDomainNonRecursiveReplacement(
      [](OriginEqAttr) -> FailureOr<Attribute> {
        llvm_unreachable("OriginEqAttr should be replaced by now");
      });

  // !lit.origin -> !kgen.struct<()>
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](OriginType) { return emptyStructType; });
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](EllipsisType) { return emptyStructType; });
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](OriginSetType) { return emptyStructType; });

  // When a function-typed field refers back to its containing struct, keep the
  // outer function type intact for CTFE symbol storage, but lower the recursive
  // struct occurrence to an erased, pointer-sized layout.
  replacer.addCycleBreaker(
      [&decls, &replacer](Type t) -> std::optional<Type> {
        auto structTp = dyn_cast<LIT::StructType>(t);
        if (!structTp)
          return std::nullopt;
        auto it = replacer.erasedStructs.find(structTp);
        if (it == replacer.erasedStructs.end()) {
          mlir::emitError(decls.get(structTp.getName()).loc,
                          "struct has recursive reference to itself");
          return Type();
        }
        return it->second;
      },
      TypeDomain::AsType);

  // #lit.struct -> #kgen.struct
  replacer.addInferredDomainNonRecursiveReplacement(
      [&, noneType](LITStructAttr attr) -> FailureOr<Attribute> {
        LIT::StructType ref = attr.getType();
        StructDecl &decl = decls.get(ref.getName());

        SmallVector<TypedAttr> values;
        values.reserve(attr.getValues().size());
        for (auto [entry, type] : llvm::zip(attr.getValues(), decl.fields)) {
          FailureOr<TypedAttr> valueOr =
              replacer.replaceParameter(std::get<1>(entry));
          if (failed(valueOr))
            return failure();

          TypedAttr value = *valueOr;
          // We to check if this is a value for a struct field that is known to
          // be a pointer type, in which case we erase the element type
          // but preserve other pointer attributes like nonnull.
          if (isa<PointerType>(type.second)) {
            auto type = cast<PointerType>(value.getType());
            auto ptrType = PointerType::get(noneType, type.getAddressSpace(),
                                            type.getIsNonNull());
            value = ParamOperatorAttr::get(POC::PtrBitcast, value, ptrType);
          }
          values.push_back(value);
        }

        if (decl.isSingleElement())
          return values.front();

        auto refOr = replacer.replace(ref, TypeDomain::AsType);
        if (failed(refOr))
          return failure();
        if (auto type = cast_or_null<KGEN::StructType>(*refOr))
          return StructAttr::get(values, type);
        return failure();
      });

  // #lit.struct.extract -> #kgen.struct.extract
  replacer.addInferredDomainNonRecursiveReplacement(
      [&](LIT::StructExtractAttr attr) -> FailureOr<Attribute> {
        auto ref = cast<LIT::StructType>(attr.getStructValue().getType());
        int idx = decls.fieldIndices.at({ref.getName(), attr.getField()});
        auto valueOr = replacer.replaceParameter(attr.getStructValue());
        if (failed(valueOr))
          return failure();
        if (decls.get(ref.getName()).isSingleElement())
          return *valueOr;
        return KGEN::StructExtractAttr::get(*valueOr, idx);
      });

  // Sugar attr is turned into canonical form.
  replacer.addInferredDomainNonRecursiveReplacement([&](SugarAttr sugar) {
    llvm_unreachable("sugar should be replaced by now");
    return Attribute();
  });

  // lower TraitType to `!kgen.type` in type domain and
  //                 to `!kgen.typevalue<@trait>` in value domain.
  replacer.addNonRecursiveReplacement([=](TraitType) { return typeType; },
                                      TypeDomain::AsType);
  replacer.addNonRecursiveReplacement(
      [=](TraitType traitType) {
        return TypeValueType::get(
            TraitInstanceRefAttr::get(ctx, traitType.getSymbols(), typeType));
      },
      TypeDomain::AsValue);

  replacer.addInferredDomainNonRecursiveReplacement(
      [&](TraitSymbolAttr attr) -> FailureOr<Attribute> {
        if (attr.getSymbol().getLeafReference().strref().ends_with(
                UNI_CLOSURE_TRAIT_NAME)) {
          assert(attr.getParamValues().size() == 6);
          SmallVector<TypedAttr> params;
          for (auto param : attr.getParamValues()) {
            if (isa<PogListAttr>(param)) // irrelevant after lower-lit
              continue;
            auto replaced = replacer.replaceParameter(param);
            if (failed(replaced))
              return failure();

            // strip the origin metadata too.
            if (auto meta = dyn_cast<FnMetadataAttr>(*replaced))
              params.push_back(meta.getWithMetadata(nullptr));
            else
              params.push_back(*replaced);
          }
          return TraitSymbolAttr::get(attr.getSymbol(), params);
        }

        // It has to be a non-parametric trait for non-closures.
        assert(attr.getParamValues().empty());
        return attr;
      });

  // Since lowerings have been generated for all struct types, we just need to
  // lookup the lowered type and substitute the parameters.
  // - For the AsType type domain, convert into a StructType.
  // - For the AsValue type domain, convert into a symbol reference to the
  //   pre-created symbol generator op.
  replacer.addNonRecursiveReplacement(
      [&, ctx, noneType](LIT::StructType ref) -> FailureOr<Type> {
        return lowerStructType(decls, replacer, evalContext, ctx, noneType, ref,
                               /*eraseIndirections=*/false);
      },
      TypeDomain::AsType);

  replacer.addNonRecursiveReplacement(
      [&](LIT::StructType ref) -> FailureOr<Type> {
        StringAttr leafName = ref.getValue().getValue().getLeafReference();
        auto structDeclIter = decls.structDecls.find(leafName);
        StructDecl &decl = structDeclIter->second;
        SmallVector<FailureOr<TypedAttr>> loweredParamValuesOr(
            map_range(ref.getParamValues(), [&](TypedAttr value) {
              return replacer.replaceParameter(value);
            }));
        if (llvm::any_of(loweredParamValuesOr, failed))
          return failure();

        SmallVector<TypedAttr> loweredParamValues(
            map_range(loweredParamValuesOr,
                      [&](FailureOr<TypedAttr> value) { return *value; }));
        auto structMetaOr = replacer.replace(
            StructMetaType::get(LIT::StructType::get(
                decl.symRef, loweredParamValues, ref.getSignature())),
            TypeDomain::AsType);
        if (failed(structMetaOr))
          return failure();

        auto concreteSymRef = TypeGeneratorRefAttr::get(
            decl.symRef, loweredParamValues, *structMetaOr);
        return TypeValueType::get(concreteSymRef);
      },
      TypeDomain::AsValue);
}

// Check if there exists an illegal recursion among struct decls.
static LogicalResult detectIllegalStructDeclsRecursion(StructDecls &decls) {
  struct RecursionFrame {
    Type funcType;
    StringAttr structName;

    bool isFuncType() const { return static_cast<bool>(funcType); }
  };
  enum class StructRecursionStatus { NoMatch, FunctionBoundary, Recursive };

  SmallVector<RecursionFrame> recursionStack;
  auto containsFuncType = [&](FuncTypeGeneratorType type) {
    return llvm::any_of(recursionStack, [&](const RecursionFrame &frame) {
      return frame.isFuncType() && frame.funcType == Type(type);
    });
  };
  auto getStructRecursionStatus = [&](StringAttr name) {
    bool crossedFuncType = false;
    for (const RecursionFrame &frame : llvm::reverse(recursionStack)) {
      if (frame.isFuncType())
        crossedFuncType = true;
      if (frame.structName == name)
        return crossedFuncType ? StructRecursionStatus::FunctionBoundary
                               : StructRecursionStatus::Recursive;
    }
    return StructRecursionStatus::NoMatch;
  };

  // DFS through the parametric types to see if there is recursion.
  mlir::AttrTypeReplacer dfs;
  auto walkFuncType =
      [&](FuncTypeGeneratorType type) -> std::pair<Type, WalkResult> {
    if (containsFuncType(type))
      return std::make_pair(Type(type), WalkResult::skip());

    recursionStack.push_back({Type(type), StringAttr()});
    for (Type inputParamType : type.getInputParamTypes()) {
      if (!dfs.replace(inputParamType)) {
        recursionStack.pop_back();
        return std::make_pair(Type(), WalkResult::interrupt());
      }
    }
    if (!dfs.replace(type.getBody())) {
      recursionStack.pop_back();
      return std::make_pair(Type(), WalkResult::interrupt());
    }
    if (PogListAttr metadata = type.getParamListAttrs()) {
      if (!dfs.replace(metadata)) {
        recursionStack.pop_back();
        return std::make_pair(Type(), WalkResult::interrupt());
      }
    }
    recursionStack.pop_back();
    return std::make_pair(Type(type), WalkResult::skip());
  };
  dfs.addReplacement(
      [&](FuncTypeGeneratorType type) { return walkFuncType(type); });
  dfs.addReplacement([](PointerType type) {
    return std::make_pair(Type(type), WalkResult::skip());
  });

  std::function<LogicalResult(StringAttr)> computeLoweredType =
      [&](StringAttr name) -> LogicalResult {
    StructDecl &decl = decls.get(name);
    if (decl.done)
      return success();

    // If the struct is already in the active path, then there is recursion.
    if (getStructRecursionStatus(name) == StructRecursionStatus::Recursive) {
      // TODO: Improve the error message. We could show the recursive path.
      mlir::emitError(decl.loc, "struct has recursive reference to itself");
      return failure();
    }
    if (getStructRecursionStatus(name) ==
        StructRecursionStatus::FunctionBoundary)
      return success();

    recursionStack.push_back({Type(), name});

    // Now recurse on the field types.
    for (Type type : llvm::make_second_range(decl.fields)) {
      if (!dfs.replace(type)) {
        recursionStack.pop_back();
        return failure();
      }
    }
    // We know the type can be lowered.
    recursionStack.pop_back();
    decl.done = true;
    return success();
  };

  dfs.addReplacement([&](LIT::StructType ref) -> std::pair<Type, WalkResult> {
    StringAttr name = ref.getName();
    // Break cycles by checking whether the reference points back into the
    // active struct-layout path before crossing a function-typed field.
    StructRecursionStatus status = getStructRecursionStatus(name);
    if (status == StructRecursionStatus::Recursive) {
      mlir::emitError(decls.get(name).loc,
                      "struct has recursive reference to itself");
      return {{}, WalkResult::interrupt()};
    }
    if (status == StructRecursionStatus::FunctionBoundary) {
      decls.get(name).needsErasure = true;
      return {ref, WalkResult::skip()};
    }

    // Recurse into a the definition of a struct.
    if (failed(computeLoweredType(name)))
      return {{}, WalkResult::interrupt()};
    return {ref, WalkResult::skip()};
  });

  dfs.addReplacement([&](SugarAttr sugar) -> std::pair<Attribute, WalkResult> {
    // Only look at the canonical value, not the sugar.
    return {dfs.replace(sugar.getCanonical()), WalkResult::skip()};
  });

  // Start from any struct and make sure our DFS terminates.
  for (auto &[name, decl] : decls.structDecls) {
    (void)decl;
    if (failed(computeLoweredType(name)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Type Lowering
//===----------------------------------------------------------------------===//

namespace {
/// Struct operations need to refer to the struct declaration symbol.
struct LITTypeLowerer : public IRRewriter, LowerLITReplacer {
  explicit LITTypeLowerer(ModuleOp module, StructDecls &structDecls,
                          mlir::LockedSymbolTableCollection &symtab);

  /// Get the index of the struct field.
  int getField(StringAttr name, LIT::StructType ref) {
    return structDecls.fieldIndices.lookup({ref.getName(), name});
  }
  /// Return true if the struct is single element.
  bool isSingleElement(LIT::StructType ref) {
    return structDecls.get(ref.getName()).isSingleElement();
  }
  Value getCastedToType(Location loc, Value value, Type type);

  /// Materialize destination conversions.
  template <typename OpT>
  LogicalResult materializeLowering(OpT op);

  /// Evaluation context used for simplifying parameters.
  LowerLITEvaluationContext evalContext;
  /// The struct decl map.
  StructDecls &structDecls;
  /// Converter for debuginfo.
  DebugInfo::DebugInfoNonCyclicTypeConverter debugTypeConverter;
  /// Unrealized casts to resolve at the end of type lowering.
  SmallVector<mlir::UnrealizedConversionCastOp> unrealizedCasts;
};
} // namespace

static DebugInfo::DIType buildDebugInfoForStructRef(
    LIT::StructType ref, StructDecls &structDecls,
    DebugInfo::DebugInfoNonCyclicTypeConverter &converter,
    ParameterEvaluationContext &evalContext) {
  // Substitute parameters into the field types.
  StructDecl &decl = structDecls.get(ref.getName());
  ParameterEvaluator evaluator(decl.decls, ref.getParamValues());
  evaluator.setEvaluationContext(&evalContext);

  auto getDebugInfoType = [&](const std::pair<StringAttr, Type> &nameAndType) {
    auto [name, type] = nameAndType;
    auto reboundType = evaluator.getReboundType(type);
    DebugInfo::DIType fieldDIType = converter.convertDebugType(reboundType);
    if (!fieldDIType) {
      fieldDIType = converter.convertDebugType(
          PointerType::get(KGEN::NoneType::get(type.getContext())));
    }
    return DebugInfo::DIMemberType::get(name, fieldDIType);
  };

  // Flatten register-passable, single-element structs.
  // TODO(#23914): Track this optimization with DWARF expressions.
  if (decl.fields.size() == 1 && decl.isRegisterPassable())
    return getDebugInfoType(decl.fields.front()).getType();

  SmallVector<DebugInfo::DIMemberType> elementTypes =
      llvm::map_to_vector(decl.fields, getDebugInfoType);

  // Parameterize the raw source name.
  DebugInfo::SourceNameAttr sourceName = decl.sourceName;
  // TODO: Make StructDeclOp's sourceName a DefaultValuedAttr once properties
  // play nicely with it.
  if (!sourceName) {
    std::string name;
    llvm::raw_string_ostream os(name);
    printNestedSymbolReference(os, ref.getSymbol());
    sourceName =
        DebugInfo::SourceNameAttr::get(StringAttr::get(ref.getContext(), name),
                                       DebugInfo::SourceNameKind::Struct);
  }

  SmallVector<StringAttr> paramValues;
  for (TypedAttr value : ref.getParamValues())
    paramValues.push_back(getParamTypeAsString(value));
  sourceName = DebugInfo::SourceNameAttr::get(
      sourceName.getName(), sourceName.getParamTypes(),
      sourceName.getArgTypes(), paramValues, sourceName.getParent(),
      sourceName.getKind(), sourceName.getDecorators());

  return DebugInfo::DIStructType::get(sourceName.encode(), elementTypes);
}

LITTypeLowerer::LITTypeLowerer(ModuleOp module, StructDecls &structDecls,
                               mlir::LockedSymbolTableCollection &symtab)
    : IRRewriter(module.getContext()), evalContext(module, symtab, structDecls),
      structDecls(structDecls) {
  populateReplacer(structDecls, *this, evalContext, module.getContext());

  // Build a converter to handle updating converted types within debug info
  // constructs.
  debugTypeConverter.addConversion([&](Type type) -> std::optional<Type> {
    FailureOr<Type> newTypeOr = replace(type, TypeDomain::AsType);
    if (succeeded(newTypeOr) && *newTypeOr != type)
      return debugTypeConverter.convertDebugType(*newTypeOr);
    return std::nullopt;
  });
  debugTypeConverter.addConversion(
      [&](LIT::StructType type) -> DebugInfo::DIType {
        return buildDebugInfoForStructRef(type, structDecls, debugTypeConverter,
                                          evalContext);
      });
  debugTypeConverter.addConversion([&](PointerType type) -> DebugInfo::DIType {
    DebugInfo::DIType elementType =
        debugTypeConverter.convertDebugType(type.getElementType());
    if (!elementType) {
      // If the type that we point to can't be converted into a
      // debuginfo type, make a None pointer debuginfo type.
      elementType = debugTypeConverter.convertDebugType(
          KGEN::NoneType::get(type.getContext()));
    }
    return DebugInfo::DITargetIndependentPointerType::get(elementType);
  });
  debugTypeConverter.addConversion([&](RefType type) -> DebugInfo::DIType {
    return debugTypeConverter.convertDebugType(type.getAsPointerType());
  });

  addInferredDomainNonRecursiveReplacement([&](DebugInfo::DIType type) {
    return debugTypeConverter.convertDebugType(type);
  });
}

static Value lowerOp(StructInsertOp op, StructInsertOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  LIT::StructType ref = op.getContainer().getType();
  if (b.isSingleElement(ref))
    return adaptor.getValue();

  int index = b.getField(op.getFieldAttr(), ref);
  return StructReplaceOp::create(b, op.getLoc(), adaptor.getValue(),
                                 adaptor.getContainer(), index);
}

static Value lowerOp(LIT::StructExtractOp op,
                     LIT::StructExtractOpAdaptor adaptor, LITTypeLowerer &b) {
  LIT::StructType ref = op.getContainer().getType();
  if (b.isSingleElement(ref))
    return adaptor.getContainer();

  int index = b.getField(op.getFieldAttr(), ref);
  return KGEN::StructExtractOp::create(b, op.getLoc(), adaptor.getContainer(),
                                       b.getIndexAttr(index));
}

static TypedAttr getAlignmentFromType(Type type, LITTypeLowerer &b) {
  auto structType = dyn_cast<LIT::StructType>(type);
  if (!structType)
    return {};

  StructDecl &decl = b.structDecls.get(structType.getName());
  if (!decl.minAlignment)
    return {};

  // Substitute the alignment with struct parameters.
  ParameterEvaluator evaluator(decl.decls, structType.getParamValues());
  evaluator.setEvaluationContext(&b.evalContext);
  return evaluator.getReboundAttribute(decl.minAlignment);
}

static Value lowerOp(VarDeclOp op, VarDeclOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  // Lower a lit.var.decl to pop.stack_allocation.
  // Check if the element type is a struct with explicit alignment.
  TypedAttr alignment = getAlignmentFromType(op.getType().getElementType(), b);
  return POP::StackAllocationOp::create(
      b, op.getLoc(), op.getType().getAsPointerType(),
      /*count=*/1, alignment, /*markedLifetimes=*/true);
}

static Value lowerOp(VarLifetimeStartOp op, VarLifetimeStartOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  b.replaceOpWithNewOp<POP::StackAllocLifetimeStartOp>(
      op, op.getArg().getDefiningOp()->getOperand(0));
  return {};
}

static Value lowerOp(VarLifetimeEndOp op, VarLifetimeEndOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  b.replaceOpWithNewOp<POP::StackAllocLifetimeEndOp>(
      op, op.getArg().getDefiningOp()->getOperand(0));
  return {};
}

static Value lowerOp(RefImmutOp op, RefImmutOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return adaptor.getRef();
}

static Value lowerOp(RefUpcastOp op, RefUpcastOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return adaptor.getRef();
}

static Value lowerOp(RefToPointerOp op, RefToPointerOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return adaptor.getRef();
}

static Value lowerOp(RefFromPointerOp op, RefFromPointerOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return adaptor.getPtr();
}

static Value lowerOp(RefFromPointerREPLOp op,
                     RefFromPointerREPLOpAdaptor adaptor, LITTypeLowerer &b) {
  return adaptor.getPtr();
}

static Value lowerOp(RefToKgenPtrOp op, RefToKgenPtrOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return adaptor.getRef();
}

static Value lowerOp(RefFromKgenPtrOp op, RefFromKgenPtrOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return adaptor.getPointer();
}

static Value lowerOp(RefLoadOp op, RefLoadOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return POP::LoadOp::create(b, op.getLoc(), adaptor.getRef());
}

static Value lowerOp(MaterializeIntoOp op, MaterializeIntoOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  Value dynVal =
      ParamMaterializeOp::create(b, op->getLoc(), adaptor.getValue());
  b.replaceOpWithNewOp<POP::StoreOp>(op, dynVal, adaptor.getDest());
  return {};
}

static Value lowerOp(RefStoreOp op, RefStoreOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  b.replaceOpWithNewOp<POP::StoreOp>(op, adaptor.getValue(), adaptor.getDest());
  return {};
}

static Value lowerOp(MemcpyOp op, MemcpyOpAdaptor adaptor, LITTypeLowerer &b) {
  TypedAttr target =
      ParamOperatorAttr::get(POC::CurrentTarget, {}, b.getType<TargetType>());
  Value dst = adaptor.getDst();
  auto elementTypeAttr =
      TypeParamAttr::get(cast<PointerType>(dst.getType()).getElementType(),
                         TypeType::get(dst.getContext()));
  Value sizeOfElt = ParamConstantOp::create(
      b, op.getLoc(),
      ParamOperatorAttr::get(POC::GetSizeOf, {elementTypeAttr, target}));
  // Note - swap the source and destination operands around.
  b.replaceOpWithNewOp<POP::MemcpyOp>(op, dst, adaptor.getSrc(), sizeOfElt);
  return {};
}

static Value lowerOp(RefStructGEROp op, RefStructGEROpAdaptor adaptor,
                     LITTypeLowerer &b) {
  if (op.usesFieldAccess()) {
    // Field name access: lower to kgen.struct.gep with computed index
    auto ref =
        cast<LIT::StructType>(op.getContainer().getType().getElementType());
    if (b.isSingleElement(ref))
      return adaptor.getContainer();

    int index = b.getField(op.getFieldAttr(), ref);
    return StructGEPOp::create(b, op.getLoc(), adaptor.getContainer(), index);
  } else {
    // Index access: lower to kgen.struct.gep with parametric index

    // Check if this is a single-element struct, similar to field access.
    // For single-element structs, the container IS the element, so just
    // return it directly instead of creating a GEP.
    auto elementType = op.getContainer().getType().getElementType();
    if (auto ref = dyn_cast<LIT::StructType>(elementType)) {
      if (b.isSingleElement(ref))
        return adaptor.getContainer();
    }

    auto resultTypeOr = b.replace(op.getType(), TypeDomain::AsType);
    if (failed(resultTypeOr))
      return nullptr;
    auto resultType = cast<PointerType>(*resultTypeOr);

    // Handle single-element struct flattening for parametric types.
    // When a single-element trivial struct is accessed through parametric
    // types (e.g., trait Self type), the struct may be flattened during
    // lowering. In either case, there's no struct to GEP into, so return
    // the container directly. We must check for ParamType to avoid
    // prematurely short-circuiting parametric cases that will resolve to
    // multi-element structs.
    auto containerPtrType = cast<PointerType>(adaptor.getContainer().getType());
    Type containerElemType = containerPtrType.getElementType();

    // Detection method 1: Types match after lowering (identity operation).
    // This catches cases where parametric types have resolved identically.
    bool isIdentity = adaptor.getContainer().getType() == resultType;

    auto isNonPackStruct =
        isa<KGEN::StructType>(containerElemType) &&
        !cast<KGEN::StructType>(containerElemType).getIsParamPack();

    // Detection method 2: Container already flattened to a concrete scalar.
    // This catches cases where the struct was flattened before this point.
    bool isFlattenedNonStruct =
        !isNonPackStruct && !isa<KGEN::ParamType>(containerElemType);

    if (isIdentity || isFlattenedNonStruct)
      return adaptor.getContainer();

    return StructGEPOp::create(b, op.getLoc(), resultType,
                               adaptor.getContainer(), *op.getIndex());
  }
}

/// Squash noop rebinds exposed by ref -> ptr lowering.
static Value lowerOp(RebindOp op, RebindOpAdaptor adaptor, LITTypeLowerer &b) {
  // If this is a noop after lowering, squish it
  if (adaptor.getInput().getType() ==
      b.replace(op.getType(), TypeDomain::AsType))
    return adaptor.getInput();
  // Otherwise just leave it and type replacement will form a valid rebind
  // in the new type domain.
  return op.getResult();
}

// lit.ref.pack.create => kgen.struct.create
static Value lowerOp(RefPackCreateOp op, RefPackCreateOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  auto typeOr = b.replace(op.getType(), TypeDomain::AsType);
  if (failed(typeOr))
    return nullptr;
  return StructCreateOp::create(b, op.getLoc(), *typeOr, adaptor.getOperands());
}

// lit.ref.pack.extract => kgen.struct.extract
static Value lowerOp(RefPackExtractOp op, RefPackExtractOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  Value value = KGEN::StructExtractOp::create(b, op.getLoc(), adaptor.getPack(),
                                              adaptor.getIndex());
  // If the result didn't fold to a pointer type, we need to emit a rebind.
  FailureOr<Type> expectedOr = b.replace(op.getType(), TypeDomain::AsType);
  if (failed(expectedOr))
    return nullptr;
  if (value.getType() != *expectedOr)
    value = RebindOp::create(b, op.getLoc(), *expectedOr, value);
  return value;
}

static Value lowerOp(RefPackFromPointerPackOp op,
                     RefPackFromPointerPackOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return adaptor.getPack();
}

static Value lowerVersionOp(Operation *op, int64_t number, LITTypeLowerer &b) {
  return ParamConstantOp::create(
      b, op->getLoc(),
      KGEN::SIMDAttr::get(
          KGEN::DTypeValue(number, KGENDType::index),
          SIMDType::get(
              /*size=*/1,
              DTypeConstantAttr::get(op->getContext(), KGENDType::index))));
}

// lit.mojo.version.major => kgen.param.constant : scalar<index>
static Value lowerOp(MojoVersionMajorOp op, MojoVersionMajorOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return lowerVersionOp(op, M::getMojoVersion().major, b);
}

// lit.mojo.version.minor => kgen.param.constant : scalar<index>
static Value lowerOp(MojoVersionMinorOp op, MojoVersionMinorOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return lowerVersionOp(op, M::getMojoVersion().minor, b);
}

// lit.mojo.version.patch => kgen.param.constant : scalar<index>
static Value lowerOp(MojoVersionPatchOp op, MojoVersionPatchOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return lowerVersionOp(op, M::getMojoVersion().patch, b);
}

Value LITTypeLowerer::getCastedToType(Location loc, Value value, Type type) {
  // If already casted, done.
  if (value.getType() == type)
    return value;

  // If coming from a cast, use input.
  if (auto castOp = value.getDefiningOp<mlir::UnrealizedConversionCastOp>())
    return getCastedToType(loc, castOp.getOperand(0), type);

  // Otherwise create a new cast.
  auto cast = mlir::UnrealizedConversionCastOp::create(*this, loc, type, value);
  unrealizedCasts.push_back(cast);
  return cast.getResult(0);
}

template <typename OpT>
LogicalResult LITTypeLowerer::materializeLowering(OpT op) {
  setInsertionPoint(op);
  SmallVector<Value> castedOperands;
  castedOperands.reserve(op->getNumOperands());
  // Get type adjusted values into the adaptor to simplify clients.
  for (OpOperand &operand : op->getOpOperands()) {
    Value value = operand.get();

    auto newTypeOr = replace(value.getType(), TypeDomain::AsType);
    if (failed(newTypeOr))
      return failure();
    // When value is a function argument, location info's function scope is
    // different from the operations in the function body. Use op->getLoc()
    // for new cast op's location instead of using value.loc().
    castedOperands.push_back(getCastedToType(op->getLoc(), value, *newTypeOr));
  }

  typename OpT::Adaptor adaptor(castedOperands, op->getAttrDictionary(),
                                op.getProperties());
  if (op->getNumResults() == 1) {
    auto resultType = op->getResult(0).getType();
    Value result = lowerOp(op, adaptor, *this);
    if (!result)
      return failure();
    if (result.getType() != resultType)
      result = getCastedToType(result.getLoc(), result, resultType);

    if (op->getResult(0) != result)
      replaceOp(op, {result});
  } else {
    assert(op->getNumResults() == 0);
    [[maybe_unused]] Value result = lowerOp(op, adaptor, *this);
    assert(!result && "nullary lowering shouldn't produce an op");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Entrypoint.
//===----------------------------------------------------------------------===//

LogicalResult LIT::lowerLITTypes(ModuleOp module, StructDecls &state,
                                 mlir::LockedSymbolTableCollection &symtab) {
  // Do a simple recursive type detection, this does not guarantees completeness
  // as it does not take parameter into account. Additional cycle detection will
  // be performed during lowering.
  if (failed(detectIllegalStructDeclsRecursion(state)))
    return failure();
  LITTypeLowerer b(module, state, symtab);

  // Lower operations first.
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    return llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<MaterializeIntoOp, StructInsertOp, StructExtractOp, RefImmutOp,
              RefUpcastOp, RefToPointerOp, RefFromPointerOp,
              RefFromPointerREPLOp, RefToKgenPtrOp, RefFromKgenPtrOp,
              RefStructGEROp, RefLoadOp, RefStoreOp, MemcpyOp, RebindOp,
              RefPackCreateOp, RefPackExtractOp, RefPackFromPointerPackOp,
              VarDeclOp, VarLifetimeStartOp, VarLifetimeEndOp,
              MojoVersionMajorOp, MojoVersionMinorOp, MojoVersionPatchOp>(
            [&](auto op) { return b.materializeLowering(op); })
        .Default([&](auto op) { return success(); });
  });
  if (result.wasInterrupted())
    return failure();

  // FIXME(MOCO-4167): Duplicate a kgen-lowered witness entry, during lower-lit,
  // we might have ordering issue during lit->kgen conversion, depending on
  // whether `get_witness_attr` is evaluated before/after the referenced struct
  // generator is lowered. It might or might not be folded correctly.
  //
  // Simply postpone the struct generator lowering to the last step (as we are
  // already doing) won't help either, as the witness_op might also have a
  // witness_attr inside for complicated cases.
  for (StructGeneratorOp structGen : module.getOps<StructGeneratorOp>()) {
    for (auto conformsOp : structGen.getOps<ConformanceOp>()) {
      for (WitnessOp witnessOp : conformsOp.getOps<WitnessOp>()) {
        b.setInsertionPoint(witnessOp);
        Operation *kgenWitnessOp = b.clone(*witnessOp);
        witnessOp.setSymName(std::string(witnessOp.getSymName()) + ".#lit#");
        LogicalResult res =
            b.replaceElementsIn(kgenWitnessOp, TypeDomain::AsType,
                                /*replaceAttrs=*/true,
                                /*replaceLocs=*/true,
                                /*replaceTypes=*/true);
        if (failed(res))
          return failure();
      }
    }
  }

  result = module.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Skip StructGeneratorOps and lower them the last.
    if (auto structGen = dyn_cast<StructGeneratorOp>(op))
      return WalkResult::skip();

    LogicalResult res =
        b.replaceElementsIn(op, TypeDomain::AsType, /*replaceAttrs=*/true,
                            /*replaceLocs=*/true,
                            /*replaceTypes=*/true);

    if (failed(res))
      return WalkResult::interrupt();

    if (auto cast = dyn_cast<mlir::UnrealizedConversionCastOp>(op)) {
      b.setInsertionPoint(cast);
      Type inType = cast.getOperand(0).getType();
      Type outType = cast.getResult(0).getType();
      if (inType == outType) {
        b.replaceOp(cast, cast.getOperand(0));
        return WalkResult::skip();
      } else if (isa<PointerType>(inType) && isa<PointerType>(outType)) {
        b.replaceOpWithNewOp<POP::PointerBitcastOp>(cast, outType,
                                                    cast.getOperand(0));
        return WalkResult::skip();
      }
    }
    return WalkResult::advance();
  });

  // Lower types in StructGeneratorOps last because we need signature and
  // witness entries to keep using LIT types in order for ParameterEvaluator to
  // work smoothly.
  for (StructGeneratorOp structGen : module.getOps<StructGeneratorOp>()) {
    // Make sure valueDomainType is translated in the value domain.
    auto valueDomainTypeOr =
        b.replace(structGen.getValueDomainType(), TypeDomain::AsValue);
    if (failed(valueDomainTypeOr))
      return failure();

    LogicalResult res = b.replaceElementsIn(structGen, TypeDomain::AsType,
                                            /*replaceAttrs=*/true,
                                            /*replaceLocs=*/true,
                                            /*replaceTypes=*/true);
    if (failed(res))
      return failure();
    structGen.setValueDomainType(*valueDomainTypeOr);

    // Then lower the body.
    structGen.getBody().walk([&](Operation *op) {
      LogicalResult res =
          b.replaceElementsIn(op, TypeDomain::AsType, /*replaceAttrs=*/true,
                              /*replaceLocs=*/true,
                              /*replaceTypes=*/true);
      if (failed(res))
        return WalkResult::interrupt();

      return WalkResult::advance();
    });
  }

  // FIXME(MOCO-4167): Erase the duplicated witness entries at the end after
  // everything is lowered properly.
  for (StructGeneratorOp structGen : module.getOps<StructGeneratorOp>())
    for (auto conformsOp : structGen.getOps<ConformanceOp>())
      for (auto witnessOp :
           llvm::make_early_inc_range(conformsOp.getOps<WitnessOp>()))
        if (witnessOp.getSymName().ends_with(".#lit#"))
          witnessOp.erase();

  if (result.wasInterrupted())
    return failure();

  return success();
}
