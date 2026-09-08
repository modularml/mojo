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
// This file provides the implementation of the ClosureEmitter class.
//
//===----------------------------------------------------------------------===//

#include "ClosureEmitter.h"
#include "IREmitter.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/ASTType.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "MojoUtils.h"
#include "OverloadSet.h"
#include "ParamBindings.h"
#include "ParserEvaluationContext.h"
#include "Signatures.h"
#include "SpecializeInf.h"
#include "Traits.h"
#include "mlir/IR/IRMapping.h"

#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/Interpreter/InterpreterAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENPogUtils.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/POPDialect/POPAttrs.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/Support/NameMangling.h"
#include "Support/Compiler/OperationUtils.h"

#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

// File-local
namespace {
static constexpr char kToDeviceType[] = "_to_device_type";
static constexpr char kIsDeviceTypeConvertible[] =
    "_is_convertible_to_device_type";
static constexpr char kIsImplicitlyEncodableTo[] =
    "_is_implicitly_encodable_to";
static constexpr char kDeviceType[] = "device_type";

static bool usesClosurePipeline(FnOp fn) {
  return fn->getParentOfType<FnOp>() && !fn.isOptionalSymbol() &&
         !fn.getFuncTypeGenerator().isCapturing();
}
} // namespace

static FnOp getFnOpNamed(TraitDeclOp traitDecl, StringRef name) {
  for (FnOp candidate : traitDecl.getFields().getOps<FnOp>()) {
    StringRef sourceName = *candidate.getSourceName();
    if (sourceName.contains(name))
      return candidate;
  }
  return {};
}

// Instantiate the storage struct.
static VarDeclOp emitInitializerCall(ASTDecl &declScope,
                                     ImplicitLocOpBuilder &builder,
                                     Location location, StructDeclOp structDecl,
                                     ArrayRef<TypedAttr> paramArgs,
                                     ArrayRef<CValue> args, StringRef name) {
  LIT::StructType boundType = structDecl.bindReference(paramArgs);
  VarDeclOp var =
      VarDeclOp::create(builder, location, boundType, name,
                        declScope.mangleParamName(name), VarDeclKind::Var);

  IREmitter emitter(declScope, builder);
  SyntheticNode node(declScope.getLoc());
  ExprDest dest(MLValue(var), EC_ReturnValue);
  CallOperands operands(CallSyntax::kTypeCall, &node, std::move(dest));
  for (CValue arg : args)
    operands.add({arg, &node});
  emitter.emitConstructorCall(ASTType(boundType), std::move(operands));
  return var;
}

static LogicalResult emitForwardingCall(ImplicitLocOpBuilder &builder,
                                        ASTDecl &declScope, TypedAttr callee,
                                        FnTypeGeneratorType calleeSig,
                                        Type resultType,
                                        ArrayRef<Value> arguments) {
  IREmitter emitter(declScope, builder);
  // We are forwarding the call in a synthetic function, pushing the debug
  // scope with the synthetic function scope.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (declScope.getShared().diBuilder) {
    auto fnOp = cast<FnOp>(builder.getInsertionBlock()->getParentOp());
    diScopeGuard =
        declScope.getShared().diBuilder->pushScopeGuard(fnOp.getLocScope());
  }

  ExprDest dest(EC_ReturnValue);
  if (!calleeSig.isAsync() && calleeSig.hasMemoryOnlyResult())
    dest = ExprDest(MLValue(arguments.back()), EC_ReturnValue);

  SyntheticNode syntheticExpr(declScope.getLoc());
  CallOperands callOperands(CallSyntax::kMethodCall, &syntheticExpr,
                            std::move(dest));
  for (auto [bbArg, convention, pog] :
       llvm::zip_equal(arguments, calleeSig.getArgConventions(),
                       calleeSig.getArgListAttrs().getPogs())) {
    if (convention == ArgConvention::ByRefResult ||
        convention == ArgConvention::ByRefError)
      continue;

    AnyValue argValue = [&]() -> AnyValue {
      if (convention == ArgConvention::ImmReg)
        return SRValue(bbArg);
      // Forward the moved argument.
      if (convention == ArgConvention::OwnedMem ||
          convention == ArgConvention::DeinitMem)
        return MRValue(bbArg);
      return CValue::getMValueForRef(bbArg);
    }();

    // Check the variadic kinds before the passing kind: a `**kwargs` argument
    // is keyword-only AND keyword-variadic, and must forward as a `**` splat.
    if (pog.isKwVarArg())
      callOperands.add({argValue, &syntheticExpr}, ArgUnpackStyle::kStarStar);
    else if (pog.isPosVarArg() || pog.isPack())
      callOperands.add({argValue, &syntheticExpr}, ArgUnpackStyle::kStar);
    else if (pog.getPassingKind() == PassingKind::KwOnly)
      callOperands.add(pog.getName(), {argValue, &syntheticExpr},
                       ArgUnpackStyle::kKeyword);
    else
      callOperands.add({argValue, &syntheticExpr}, ArgUnpackStyle::kPositional);
  }

  CValue callResult =
      emitter.emitCallUnchecked(callee, std::move(callOperands));
  // Forwarding reuses the normal call machinery; an unhandleable signature
  // surfaces as a failed call with the matcher's diagnostic -- propagate.
  if (!callResult)
    return failure();
  if (!calleeSig.isAsync()) {
    auto regRet = callResult.getIfSRValue();
    if (regRet && resultType != regRet.getType())
      regRet = RebindOp::create(builder, resultType, regRet);

    IREmitter::emitNormalReturn(builder, regRet);
    return success();
  }

  // Handle async calls.
  ExprDest awaitDest(MLValue(arguments.back()), EC_SynthesizedMethod);
  if (!emitter.emitNamedMethodCall(
          "__await__",
          CallOperands(CallSyntax::kMethodCallSynthetic, &syntheticExpr,
                       std::move(awaitDest), {{callResult, &syntheticExpr}})))
    return failure();

  IREmitter::emitNormalReturn(builder);
  return success();
}

static void
addConformanceTable(ASTDecl &structDecl,
                    const ClosureEmitter::ClosureParent &closureParent,
                    ArrayRef<std::pair<StringRef, TypedAttr>> witnesses) {
  // Insert the new witness into the conformance table.
  MLIRContext *ctx = structDecl.getContext();
  StructDeclOp structDeclOp = cast<StructDeclOp>(structDecl.getIfOperation());
  ImplicitLocOpBuilder b(structDeclOp->getLoc(), structDeclOp.getContext());
  b.setInsertionPointToEnd(&structDeclOp.getBodyRegion().front());
  TraitDeclOp traitDeclOp = closureParent.getTrait(structDecl.getShared());
  TraitSymbolArrayAttr immediateParents = traitDeclOp.getImmediateParentsAttr();
  TraitSymbolAttr traitSymbol = closureParent.getSymbol();
  StringAttr parentName = closureParent.getFlattenedName();
  ConformanceOp witnessTable =
      ConformanceOp::create(b, traitSymbol, immediateParents);
  Block &block = witnessTable.getBody().emplaceBlock();
  b.setInsertionPointToStart(&block);
  for (auto [name, newWitness] : witnesses)
    WitnessOp::create(b, StringAttr::get(ctx, name), /*sym_visibility=*/nullptr,
                      newWitness);

  // Register the conformance with the ASTDecl so lookupInCurrentScope can find
  // it during constraint checking.
  ASTDecl &conformDecl = structDecl.getShared().getDeclResolver().addDecl(
      witnessTable, structDecl.getLoc(), parentName, &structDecl, {}, {}, -1);
  conformDecl.resolvedness = DeclResolvedness::signature;

  // Update the types of the struct wrapper.
  TraitType oldTraitType = structDeclOp.getCanonicalTrait();
  if (llvm::is_contained(oldTraitType.getSymbols(), traitSymbol))
    return;
  SmallVector<TraitSymbolAttr> symbols;
  llvm::append_range(symbols, oldTraitType.getSymbols());
  symbols.push_back(traitSymbol);
  canonicalizeTraitCompositionSymbols(structDecl.getShared(), symbols);

  TraitType traitType = TraitType::get(ctx, symbols);
  structDeclOp.setCanonicalTrait(traitType);
}

ClosureEmitter::ClosureParent
ClosureEmitter::getBuiltinParent(StringRef traitName, StringRef traitFnName,
                                 ClosureMethod closureMethod) {
  ASTDecl *traitDecl = shared.lookupBuiltinTrait(traitName, SMLoc());
  assert(traitDecl && "missing builtin closure parent trait");
  return ClosureParent(
      shared, cast<TraitDeclOp>(traitDecl->getIfOperation()).bindReference({}),
      traitFnName, closureMethod);
}

ClosureEmitter::ClosureEmitter(SharedState &shared)
    : FunctionEmitter(shared), ctx(shared.getContext()),
      selfName(StringAttr::get(ctx, "self")),
      copyName(StringAttr::get(ctx, "copy")) {}

TraitDeclOp ClosureEmitter::ClosureParent::getTrait(SharedState &shared) const {
  assert(symbol && "closure parent must name a trait");
  // This is a cached lookup.
  ASTDecl &traitDecl =
      shared.declResolver->getDeclForTypeSymbol(symbol.getSymbol());
  return cast<TraitDeclOp>(traitDecl.getIfOperation());
}

ClosureEmitter::ClosureParent::ClosureParent(SharedState &shared,
                                             TraitSymbolAttr symbol,
                                             StringRef traitFnName,
                                             ClosureMethod closureMethod)
    : symbol(symbol), closureMethod(closureMethod) {
  if (traitFnName.empty())
    return;

  ASTDecl &traitDecl =
      shared.declResolver->getDeclForTypeSymbol(symbol.getSymbol());
  if (traitDecl.resolvedness < DeclResolvedness::body) {
    [[maybe_unused]] bool outcome = succeeded(
        shared.declResolver->resolveBody(traitDecl, traitDecl.getLoc()));
    assert(outcome && "closure parent trait should not fail body resolution");
  }

  // A trait member carries no symbol name until its signature is resolved, and
  // that name is what identifies it here and keys its witness.
  for (auto [_, members] : traitDecl.getDeclsInScope()) {
    for (ASTDecl *member : members) {
      [[maybe_unused]] bool outcome = succeeded(
          shared.declResolver->resolveSignature(*member, member->getLoc()));
      assert(outcome && "closure parent trait members should not fail "
                        "signature resolution");
    }
  }

  FnOp definingFn =
      getFnOpNamed(cast<TraitDeclOp>(traitDecl.getIfOperation()), traitFnName);
  assert(definingFn && "missing function in closure parent trait");
  witnessName = definingFn.getSymNameAttr();
  signature = definingFn.getFullSignature();
  inlineLevel = definingFn.getInlineLevel();
}

static StructFieldOp addFieldOpAndDecl(StringAttr name, Type type,
                                       StructDeclOp structOp,
                                       ASTDecl &structDecl, OpBuilder &b,
                                       DeclResolver &declResolver) {
  auto field = StructFieldOp::create(b, structOp.getLoc(), name, type);
  declResolver.addFullyResolvedDecl(&*field, field.getNameAttr(),
                                    structDecl.getLoc(), &structDecl);
  return field;
}

static std::pair<ASTDecl &, StructDeclOp>
createStruct(SharedState &shared, ASTDecl &moduleDecl, StringAttr name,
             ArrayRef<ParamDeclAttr> params, SMLoc loc,
             ArrayRef<PassingKind> passingKinds) {
  auto module = cast_or_null<FileModuleOp>(moduleDecl.getIfOperation());
  assert(module && "extension/wrapper structs require a FileModuleOp parent");
  assert(passingKinds.size() == params.size() &&
         "passing kind per struct parameter");
  OpBuilder b(module.getRegion());
  SmallVector<StringAttr> paramNames;
#ifndef NDEBUG // Only used for assertion checks below.
  SmallPtrSet<StringAttr, 16> paramNamesSet;
#endif
  for (ParamDeclAttr param : params) {
    // The parameter for a synthesized closure are captured variable name, do
    // not demangle the capture parameter name here, as they can never be
    // referenced by user.
    paramNames.push_back(param.getName());
    assert(paramNamesSet.insert(param.getName()).second &&
           "duplicate parameter name");
  }
  // TODO: The type may contain decl references that need to be remapped.
  auto paramListAttr =
      PogListAttr::get(b.getContext(), paramNames, passingKinds);

  StructDeclOp declOp =
      StructDeclOp::create(b, shared.diags.translateLocation(loc), name);
  declOp.setSynthetic(true);

  // Set attributes in bulk.
  NamedAttrList attrs = declOp->getAttrDictionary();
  attrs.set(declOp.getParamsAttrName(), b.getAttr<ParamDeclArrayAttr>(params));
  auto sig = TypeSignatureType::remapToSignature(
      [&]() -> InFlightDiagnostic {
        llvm_unreachable("unexpected invalid signature");
      },
      ParamDeclArrayAttr::get(b.getContext(), params), paramListAttr);
  attrs.set(declOp.getSignatureAttrName(), TypeAttr::get(sig));
  declOp->setAttrs(attrs.getDictionary(module.getContext()));

  ASTDecl &structDecl = shared.declResolver->addFullyResolvedDecl(
      &*declOp, name, loc, &moduleDecl);

  structDecl.setTypeDeclSelf(ASTDecl::computeSelfTypeForStruct(declOp));
  return {structDecl, declOp};
}

static bool isByReferenceCapture(CaptureConvention c) {
  switch (c) {
  case CaptureConvention::kConventionUnspecified:
  case CaptureConvention::kConventionMut:
  case CaptureConvention::kConventionRead:
  case CaptureConvention::kConventionRef:
    return true;
  case CaptureConvention::kConventionTrivialCopy:
  case CaptureConvention::kConventionCopy:
  case CaptureConvention::kConventionMove:
    return false;
  }
  return false;
}

static FailureOr<ASTType> getDeviceType(ASTType hostType, ASTDecl &scope,
                                        SharedState &shared);

static FailureOr<Type>
getReboundCaptureDeviceFieldType(ASTType captureStorageHostType,
                                 ASTDecl &scopeDecl, SharedState &shared) {
  FailureOr<ASTType> deviceCaptureType =
      getDeviceType(captureStorageHostType, scopeDecl, shared);
  if (failed(deviceCaptureType))
    return failure();

  ArrayRef<TypedAttr> captureBindings =
      captureStorageHostType.getParamBindings();
  if (captureBindings.empty())
    return getCanonicalType(*deviceCaptureType);
  ASTDecl *captureTypeDecl = captureStorageHostType.getDecl(shared);
  assert(captureTypeDecl && "expected declared type for parametric capture");
  auto structOp =
      dyn_cast_or_null<StructDeclOp>(captureTypeDecl->getIfOperation());
  assert(structOp && !structOp.getInputParams().empty() &&
         "expected parametric struct for rebound capture device field type");
  ParameterEvaluator evaluator =
      shared.getParameterEvaluator(structOp.getInputParams(), captureBindings);
  return getCanonicalType(evaluator.getReboundType(*deviceCaptureType));
}

static FailureOr<LIT::StructType>
createDeviceTypeStruct(SharedState &shared, ASTDecl &moduleDecl,
                       ASTDecl &storageStructDecl,
                       ArrayRef<Type> deviceFieldTypes) {
  MLIRContext *ctx = shared.getContext();
  auto storageStruct = cast<StructDeclOp>(storageStructDecl.getIfOperation());
  ArrayRef<ParamDeclAttr> structParams = storageStruct.getInputParams();
  StringAttr deviceStructName = StringAttr::get(
      ctx, Twine(storageStruct.getSymName()).concat("::__device_type"));
  auto [deviceStructDecl, deviceStructOp] = createStruct(
      shared, moduleDecl, deviceStructName, structParams,
      storageStructDecl.getLoc(),
      SmallVector<PassingKind>(structParams.size(), PassingKind::Inferred));
  OpBuilder b(deviceStructOp.getRegion());
  b.setInsertionPointToStart(&deviceStructOp.getFields().front());
  for (auto [field, image] :
       llvm::zip(storageStruct.getFieldDecls(), deviceFieldTypes))
    addFieldOpAndDecl(field.getNameAttr(), image, deviceStructOp,
                      deviceStructDecl, b, *shared.declResolver);
  SmallVector<TypedAttr> structBindings =
      llvm::map_to_vector(structParams, [](ParamDeclAttr param) -> TypedAttr {
        return ParamDeclRefAttr::get(param);
      });
  return deviceStructOp.bindReference(structBindings);
}

/// Given a signature of a function, create a FuncType by inserting a closure
/// argument at index 0 with the given convention.
static FnTypeGeneratorType
addClosureSelfArgToFunctionSignature(Type closureType, ArgConvention convention,
                                     FnTypeGeneratorType sig) {
  MLIRContext *ctx = sig.getContext();

  unsigned newArgCount = sig.getNumArguments() + 1;
  SmallVector<Type> signatureInputs;
  signatureInputs.reserve(newArgCount);
  SmallVector<ArgConvention> argConventions;
  argConventions.reserve(newArgCount);
  SmallVector<PogMetadataAttr> argPogs;
  argPogs.reserve(newArgCount);

  // Add self.
  signatureInputs.push_back(closureType);
  argConventions.push_back(convention);
  argPogs.emplace_back(
      PogMetadataAttr::get(StringAttr::get(ctx), PassingKind::PosOnly));
  // Add the rest of the arguments.
  FnMetaOriginDataAttr oldFnMetadata = sig.getFnMetaOriginData();
  PogListAttr argListAttr = sig.getArgListAttrs();
  llvm::append_range(signatureInputs, sig.getArguments());
  llvm::append_range(argConventions, sig.getArgConventions());
  // For a fully-populated source `argListAttr`, append its pogs to keep
  // `argPogs.size() == argConventions.size()`. For an empty source (a 0-arg
  // closure with no source-level metadata), the prepended `self` pog is the
  // only pog the closure trait method has — fill the rest with anonymous
  // positional-only pogs so the synthetic trait method is fully shaped.
  llvm::append_range(argPogs, argListAttr.getPogs());
  while (argPogs.size() < argConventions.size())
    argPogs.emplace_back(
        PogMetadataAttr::get(StringAttr::get(ctx), PassingKind::PosOnly));
  assert(argPogs.size() == argConventions.size());

  // Closure storage is carried by the inserted self argument, not by FnEffects.
  auto newArgListAttr = argListAttr.cloneWith(argPogs);
  auto metadata =
      FnMetaOriginDataAttr::get(ctx, oldFnMetadata.getNumImplicitOriginDecls(),
                                oldFnMetadata.getCaptureOrigins(),
                                oldFnMetadata.getIsNestedOriginsReadOnly(),
                                oldFnMetadata.getDefinesInteriorOrigins());
  return FuncTypeGeneratorType::get(
      sig.getInputParamTypes(),
      FunctionType::get(ctx, signatureInputs, sig.getResults()), argConventions,
      sig.getFnEffects(), metadata, sig.getParamListAttrs(), newArgListAttr);
}

std::pair<TraitDeclOp, ASTDecl *> ClosureEmitter::createTraitOp(
    StringAttr name, SmallVector<ClosureParent> &closureParents,
    SMLoc nestedFunctionOrTypeLocation,
    llvm::function_ref<
        void(ASTDecl &traitDecl,
             DenseSet<std::pair<StringAttr, StringAttr>> &functions)>
        populateTrait) {
  OpBuilder b(shared.getTopLevelDecl().getIfOperation());
  b.setInsertionPointToStart(
      &cast<ModuleOp>(shared.getTopLevelDecl().getIfOperation())
           .getBodyRegion()
           .front());
  MLIRContext *ctx = b.getContext();
  Location location =
      shared.diags.translateLocation(nestedFunctionOrTypeLocation);
  StringRef originalName = name.getValue();
  auto closureTrait =
      TraitDeclOp::create(b, location, StringAttr::get(ctx, originalName));
  ASTDecl &traitDecl = shared.declResolver->addFullyResolvedDecl(
      &*closureTrait, name, nestedFunctionOrTypeLocation,
      &shared.getTopLevelDecl());

  closureTrait.setDefinesClosure(true);
  // Populate the trait with parent and self methods.
  SmallVector<TraitSymbolAttr> parents;
  DenseSet<TraitSymbolAttr> immediateParents;
  for (const ClosureParent &p : closureParents) {
    immediateParents.insert(p.getSymbol());
    parents.push_back(p.getSymbol());
  }
  (void)shared.declResolver->addSelfTypeToTrait(closureTrait, traitDecl,
                                                parents, immediateParents);
  DenseSet<std::pair<StringAttr, StringAttr>> existingFns;
  populateTrait(traitDecl, existingFns);
  return std::pair<TraitDeclOp, ASTDecl *>(closureTrait, &traitDecl);
}

/// Converts function type generator parameters to ParamDeclAttr instances.
///
/// The function type generator stores parameters as (name, metadata) pairs and
/// types, where types can reference earlier parameters by index. This function
/// converts these to ParamDeclAttr instances with canonical types that use
/// named references.
///
/// @param sig The function type generator type
/// @return Vector of ParamDeclAttr instances with canonical types
static SmallVector<ParamDeclAttr>
populateParametersFromFnGeneratorType(FnTypeGeneratorType sig) {
  auto pogAttrs = sig.getParamListAttrs().getPogs();
  SmallVector<StringAttr> pogNames = llvm::map_to_vector(
      pogAttrs, [&](PogMetadataAttr pog) { return pog.getName(); });
  ParamRefRemapper replacer(pogNames);
  SmallVector<ParamDeclAttr> parameters;
  parameters.reserve(pogAttrs.size());
  assert(pogAttrs.size() == sig.getInputParamTypes().size());
  for (auto [name, type] : llvm::zip(pogNames, sig.getInputParamTypes())) {
    Type canonicalType = replacer.replace(type);
    parameters.push_back(ParamDeclAttr::get(name, canonicalType));
  }

  return parameters;
}

static TraitType
getTraitType(SmallVector<ClosureEmitter::ClosureParent> &closureParents,
             ASTDecl &moduleDecl) {
  SmallVector<TraitSymbolAttr> symbols = llvm::map_to_vector(
      closureParents, [](const ClosureEmitter::ClosureParent &parent) {
        return parent.getSymbol();
      });
  canonicalizeTraitCompositionSymbols(moduleDecl.getShared(), symbols);
  return TraitType::get(moduleDecl.getContext(), symbols);
}

/// If a parameter is captured in a signature it
/// becomes an inferred parameter on the struct. Collect such parameters.
static SmallVector<TypedAttr> getCaptureBindings(StructDeclOp structDeclOp) {
  ArrayRef<ParamDeclAttr> params = structDeclOp.getInputParams();
  ArrayRef<PogMetadataAttr> pogs =
      structDeclOp.getSignature().getParamListAttrs().getPogs();
  assert(params.size() == pogs.size() &&
         "struct params and POGs must agree in arity");
  SmallVector<TypedAttr> bindings;
  for (auto [param, pog] : llvm::zip_equal(params, pogs)) {
    if (pog.getPassingKind() == PassingKind::Inferred)
      bindings.push_back(ParamDeclRefAttr::get(param));
  }
  return bindings;
}

/// Replace GetWitnessAttr lookups on a specific type with lookups on the impl
/// parameter. Used to redirect trait Self lookups to the wrapper struct's impl.
static FnTypeGeneratorType replaceTraitWitnessLookupsWithParamWitnessLookups(
    FnTypeGeneratorType sig, Type replaceMeType, ParamDeclAttr implType) {
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](GetWitnessAttr getWitness) -> TypedAttr {
    if (getWitness.getTypeValue().getType() != replaceMeType)
      return getWitness;
    return GetWitnessAttr::get(
        ParamDeclRefAttr::get(implType), getWitness.getTraitSymbol(),
        getWitness.getWitnessName(), getWitness.getType());
  });
  return cast<FnTypeGeneratorType>(replacer.replace(sig));
}

static std::pair<TypedAttr, SmallVector<TypedAttr>>
selfContainedSymbolAndCaptures(PValue fnPValue,
                               FnTypeGeneratorType wrapperImplType,
                               SharedState &shared, Location loc);

static SymbolConstantAttr
buildSymbol(FnOp impl, ArrayRef<ParamDeclAttr> structLevelParams) {
  MLIRContext *ctx = impl.getContext();
  SymbolRefAttr implSymbol = getFullyResolvedSymbolRef(
      cast<mlir::SymbolOpInterface>(impl.getOperation()));
  // Build symbol by binding struct level parameters and explicit parameters.
  FuncTypeGeneratorType baseSigGen = impl.getFuncTypeGenerator();
  SmallVector<TypedAttr> params;
  llvm::append_range(
      params, llvm::map_range(structLevelParams, [](ParamDeclAttr param) {
        return ParamDeclRefAttr::get(param);
      }));
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](ParamDeclRefAttr reference) -> TypedAttr {
    return UnboundAttr::get(reference.getType());
  });
  for (auto param : impl.getInputParams().drop_back(
           impl.getFuncTypeGenerator().getNumImplicitOriginDecls()))
    params.push_back(
        cast<TypedAttr>(replacer.replace(ParamDeclRefAttr::get(param))));
  SymbolConstantAttr symbolConstant =
      SymbolConstantAttr::get(ctx, implSymbol, params, baseSigGen);
  return symbolConstant;
}

static SymbolConstantAttr
buildSymbolWithBindings(FnOp impl, ArrayRef<ParamDeclAttr> structLevelParams,
                        ArrayRef<TypedAttr> fnLevelBindings) {
  MLIRContext *ctx = impl.getContext();
  SymbolRefAttr implSymbol = getFullyResolvedSymbolRef(
      cast<mlir::SymbolOpInterface>(impl.getOperation()));

  // Build symbol by binding struct level parameters.
  SmallVector<TypedAttr> params;
  llvm::append_range(
      params, llvm::map_range(structLevelParams, [](ParamDeclAttr param) {
        return ParamDeclRefAttr::get(param);
      }));
  llvm::append_range(params, fnLevelBindings);
  FuncTypeGeneratorType baseSigGen = impl.getFuncTypeGenerator();
  FuncTypeGeneratorType specializedSigGen = baseSigGen.getSpecializedGenerator(
      fnLevelBindings, /*evaluationContext=*/nullptr, impl.getLoc());

  SymbolConstantAttr symbolConstant =
      SymbolConstantAttr::get(ctx, implSymbol, params, specializedSigGen);
  return symbolConstant;
}

static size_t explicitParamCount(FnOp fn) {
  return fn.getInputParams().size() -
         fn.getFuncTypeGenerator().getNumImplicitOriginDecls();
}

/// Count leading `PassingKind::Inferred` params on `fn`.
static size_t leadingInferredParamCount(FnOp fn) {
  size_t explicitCount = explicitParamCount(fn);
  ArrayRef<PogMetadataAttr> pogs =
      fn.getFuncTypeGenerator().getParamListAttrs().getPogs();
  size_t count = 0;
  for (size_t i = 0; i < explicitCount && i < pogs.size(); ++i) {
    if (pogs[i].getPassingKind() != PassingKind::Inferred)
      break;
    ++count;
  }
  return count;
}

/// Populate `wrapperFn` with a forwarding call to `callee`, rebinding each
/// block argument to the corresponding `expectedOperandTypes` entry when
/// needed.
static LogicalResult
emitCallForwarderBody(SharedState &shared, FnOp wrapperFn, ASTDecl &wrapperDecl,
                      TypedAttr callee, FnTypeGeneratorType calleeSig,
                      Type resultType, ArrayRef<Type> expectedOperandTypes) {
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(wrapperFn.getLocScope());
  ImplicitLocOpBuilder bodyBuilder = ImplicitLocOpBuilder::atBlockBegin(
      wrapperFn.getLoc(), wrapperFn.getBody());

  Block &block = wrapperFn.getBodyRegion().front();
  assert(block.getNumArguments() == expectedOperandTypes.size() &&
         "forwarder arity must match expected operand types");
  SmallVector<Value> operands;
  operands.reserve(block.getNumArguments());
  for (auto [arg, ty] :
       llvm::zip_equal(block.getArguments(), expectedOperandTypes))
    operands.push_back(arg.getType() != ty
                           ? Value(RebindOp::create(bodyBuilder, ty, arg))
                           : Value(arg));

  return emitForwardingCall(bodyBuilder, wrapperDecl, callee, calleeSig,
                            resultType, operands);
}

/// Synthesize the trait-shaped always-inline `__call__$trait` forwarder and
/// publish it as storage `__call__`.
static FnOp emitStorageCallWitness(
    ASTDecl &structDecl, StructDeclOp structOp, FnOp promotedCall, SMLoc smLoc,
    llvm::function_ref<std::tuple<FnOp, ArrayRef<ParamDeclAttr>, Type>(
        ASTDecl &, bool, StringAttr)>
        pushBackTraitFn) {
  SharedState &shared = structDecl.getShared();
  MLIRContext *ctx = shared.getContext();

  ImplicitLocOpBuilder b(structOp.getLoc(), structOp);
  b.setInsertionPointToEnd(&structOp.getFields().front());
  StringAttr witnessName = StringAttr::get(ctx, "__call__$trait");
  auto [callWitness, callParameters, callResult] =
      pushBackTraitFn(structDecl, /*synthetic=*/true, witnessName);
  ASTDecl *callWitnessDecl = shared.declResolver->getDeclForFuncSymbol(
      getFullyResolvedSymbolRef(callWitness));
  callWitnessDecl->resolvedness = DeclResolvedness::body;
  callWitness.setInlineLevel(InlineLevel::Always);

  const size_t promotedParams = explicitParamCount(promotedCall);
  assert(
      callParameters.size() >= promotedParams &&
      "trait-shaped witness cannot have fewer params than the promoted body");
  const size_t extraAux = callParameters.size() - promotedParams;

  // Map trait auxiliary parameters to the capture bindings of the storage
  // struct.
  SmallVector<TypedAttr> captureBindings = getCaptureBindings(structOp);
  assert(extraAux <= captureBindings.size() &&
         "trait aux must not exceed storage capture bindings");
  DenseMap<StringRef, TypedAttr> paramToAliasValue;
  for (auto [param, binding] :
       llvm::zip_equal(callParameters.take_front(extraAux),
                       ArrayRef(captureBindings).take_front(extraAux)))
    paramToAliasValue.insert({param.getName().getValue(), binding});

  mlir::AttrTypeReplacer aliasReplacer;
  aliasReplacer.addReplacement([&](ParamDeclRefAttr paramRef) -> TypedAttr {
    auto it = paramToAliasValue.find(paramRef.getName().getValue());
    if (it != paramToAliasValue.end())
      return it->second;
    return paramRef;
  });

  TypedAttr calleeSymbol = buildSymbol(promotedCall, structOp.getInputParams());
  SmallVector<TypedAttr> paramArgs;
  for (ParamDeclAttr param : callParameters.drop_front(extraAux)) {
    Type paramType = cast<Type>(aliasReplacer.replace(param.getType()));
    paramArgs.push_back(
        ParamOperatorAttr::getRebind(ParamDeclRefAttr::get(param), paramType));
  }
  if (!paramArgs.empty())
    calleeSymbol = BindParamsAttr::get(ctx, calleeSymbol, paramArgs,
                                       &shared.getEvaluationContext());

  // Linkage / compile-offload see the promoted body through this thunk.
  callWitness->setAttr(kTransparentThunkCalleeExprAttr, calleeSymbol);
  auto calleeSig = cast<FnTypeGeneratorType>(calleeSymbol.getType());
  if (failed(emitCallForwarderBody(shared, callWitness, *callWitnessDecl,
                                   calleeSymbol, calleeSig, callResult,
                                   calleeSig.getArguments()))) {
    shared.emitError(smLoc, "failed to emit trait-shaped __call__ forwarder");
    return {};
  }

  return callWitness;
}

/// Get a name for trait method parameter at idx, we just want a placeholder
/// here, the scheme that we use here does not matter (the decl/ref mapping is
/// what matters). Using a illegal user-space name to avoid collision.
inline static std::string getTraitMethodParamName(size_t idx) {
  return "Closure_Syn#" + llvm::utostr(idx);
}

std::tuple<FnOp, ArrayRef<ParamDeclAttr>, Type>
ClosureEmitter::pushBackTraitFunctionImpl(
    FnTypeGeneratorType traitFnSignature, ASTDecl &structDecl, bool synthetic,
    StringAttr fnName, SpecialFunctionKind specialFnID, InlineLevel inlineLevel,
    bool redirectWitnessToImplParam, ASTType selfTypeOverride) {
  StructDeclOp structDeclOp = cast<StructDeclOp>(structDecl.getIfOperation());
  ImplicitLocOpBuilder b(structDeclOp.getLoc(), structDeclOp);
  b.setInsertionPointToEnd(&structDeclOp.getFields().front());
  SharedState &shared = structDecl.getShared();
  // Wrapper signature is the signature of the method on the wrapper struct.
  // We create it by specializing the trait method by binding the struct type
  // to the self parameter.
  ASTType selfType =
      selfTypeOverride ? selfTypeOverride : structDecl.getTypeDeclSelf();
  FnTypeGeneratorType wrapperSignature =
      specializeSignature(traitFnSignature, selfType, *shared.declResolver);

  if (redirectWitnessToImplParam) {
    wrapperSignature = replaceTraitWitnessLookupsWithParamWitnessLookups(
        wrapperSignature, selfType.extractMetaType(),
        structDeclOp.getParams().front());
  }

  // Calculate the argument types and result types in terms of the named
  // parameters.
  ParamRefRemapper replacer;
  SmallVector<ParamDeclAttr> parameters;
  for (auto [idx, paramType] :
       llvm::enumerate(wrapperSignature.getInputParamTypes())) {
    parameters.push_back(ParamDeclAttr::get(getTraitMethodParamName(idx),
                                            replacer.replace(paramType)));
    replacer.appendParamDecl(parameters.back());
  }

  SmallVector<Type> argumentTypes;
  llvm::append_range(
      argumentTypes,
      llvm::map_range(wrapperSignature.getArguments(), [&](Type original) {
        return replacer.replace(original);
      }));
  Type result = replacer.replace(wrapperSignature.getResults().front());
  auto [op, decl] = synthesizeFunction(
      structDecl, fnName, parameters, wrapperSignature.getParamListAttrs(),
      argumentTypes, wrapperSignature.getArgConventions(),
      wrapperSignature.getArgListAttrs(), result, specialFnID,
      structDecl.getLoc(), b, wrapperSignature.getFnEffects(), "", synthetic,
      inlineLevel);
  size_t synthesizedOrigins =
      op.getFuncTypeGenerator().getNumImplicitOriginDecls();
  return {op, op.getInputParams().drop_back(synthesizedOrigins), result};
}

static SymbolConstantAttr getSymbolNoParamValues(StructDeclOp declOp,
                                                 FnOp impl) {
  SymbolRefAttr implSymbol = getFullyResolvedSymbolRef(
      cast<mlir::SymbolOpInterface>(impl.getOperation()));
  FnTypeGeneratorType baseSigGen = impl.getFuncTypeGenerator();
  baseSigGen = FuncTypeGeneratorType::remapToFuncTypeGenerator(
      declOp.getInputParams(),
      FunctionType::get(baseSigGen.getContext(),
                        baseSigGen.getBody().getArguments(),
                        baseSigGen.getResultType()),
      baseSigGen.getArgConventions(), baseSigGen.getFnEffects(),
      baseSigGen.getFnMetaOriginData(), {});
  return SymbolConstantAttr::get(implSymbol, baseSigGen, {});
}

static ConformanceOp lookupConformanceTable(StructDeclOp op,
                                            SymbolRefAttr traitSymbol) {
  for (auto conformance : op.getFields().getOps<ConformanceOp>()) {
    if (conformance.getTraitSymbolAttr().getSymbol() == traitSymbol) {
      return conformance;
    }
  }

  assert(false && "conformance table should be present");
  return {};
}

static void
generateIsTrivialSpecialAlias(StringRef name, bool value, SharedState &shared,
                              ASTDecl &structDecl,
                              const ClosureEmitter::ClosureParent &parent) {
  auto ctx = shared.getContext();
  auto declOp = dyn_cast<StructDeclOp>(structDecl.getIfOperation());
  auto conformanceOp = lookupConformanceTable(declOp, parent.getSymbolRef());

  ImplicitLocOpBuilder b = ImplicitLocOpBuilder::atBlockEnd(
      declOp->getLoc(), &declOp.getBodyRegion().front());
  IREmitter emitter(structDecl, EC_AliasValue);
  SyntheticNode node(structDecl.getLoc());
  TypedAttr valueAttr =
      emitter
          .emitBool({BoolAttr::get(ctx, value), &node}, EC_OperatorOperandValue)
          .getIfPValue();

  ParamDeclAttr paramAttr =
      ParamDeclAttr::get(ctx, StringAttr::get(ctx, name), valueAttr.getType());
  AliasDeclOp aliasOp = LIT::AliasDeclOp::create(
      b, declOp.getBodyRegion().getLoc(), paramAttr, valueAttr);
  aliasOp.setInheritedFromAttr(parent.getSymbol());
  shared.declResolver->addFullyResolvedDecl(aliasOp, StringAttr::get(ctx, name),
                                            structDecl.getLoc(), &structDecl);

  b.setInsertionPointToEnd(&conformanceOp.getBody().front());
  WitnessOp::create(b, StringAttr::get(ctx, name), /*sym_visibility=*/nullptr,
                    valueAttr);
}

//===----------------------------------------------------------------------===//
// Closure Parameter Type Constraint Collection
//===----------------------------------------------------------------------===//

void ClosureEmitter::processClosureTraits(
    TraitType traitType, std::function<void(TraitDeclOp)> const &process) {
  for (TraitSymbolAttr traitSymbol : traitType.getSymbols()) {
    ASTDecl *traitDecl = shared.getDeclResolver().getDeclForTypeSymbolIfExists(
        traitSymbol.getSymbol());
    if (!traitDecl)
      continue;
    auto closureTrait = dyn_cast<TraitDeclOp>(traitDecl->getIfOperation());
    if (!closureTrait || !closureTrait.getDefinesClosure())
      continue;
    process(closureTrait);
  }
}

std::optional<TraitDeclOp> ClosureEmitter::getClosureDecl(SharedState &shared,
                                                          Type type) {
  auto closureTrait = [&](TraitType traitType) -> std::optional<TraitDeclOp> {
    for (auto sym : traitType.getSymbols()) {
      ASTDecl &decl =
          shared.getDeclResolver().getDeclForTypeSymbol(sym.getSymbol());
      if (auto traitOp =
              dyn_cast_if_present<TraitDeclOp>(decl.getIfOperation())) {
        if (traitOp.getDefinesClosure())
          return traitOp;
      }
    }
    return std::nullopt;
  };
  if (auto traitType = dyn_cast<TraitType>(type))
    return closureTrait(traitType);
  if (auto structType = dyn_cast<LIT::StructType>(type)) {
    ASTDecl &structDecl =
        shared.getDeclResolver().getDeclForTypeSymbol(structType.getSymbol());
    auto structDeclOp =
        dyn_cast_if_present<LIT::StructDeclOp>(structDecl.getIfOperation());
    if (!structDeclOp)
      return std::nullopt;
    return closureTrait(structDeclOp.getCanonicalTrait());
  }
  if (auto refType = dyn_cast<RefType>(type))
    return getClosureDecl(shared, refType.getElementType());
  // An opaque parameter constrained by a closure trait (e.g. `G: def() -> T`)
  // appears as a `ParamType` wrapping a value whose metatype is that trait.
  if (auto paramType = dyn_cast<ParamType>(type))
    return getClosureDecl(shared, paramType.getParam().getType());
  return std::nullopt;
}

bool ClosureEmitter::isClosureType(SharedState &shared, Type type) {
  return getClosureDecl(shared, type).has_value();
}

void ClosureEmitter::collectClosureExternalRefs(
    ParamDeclAttr closureParam, SmallVectorImpl<ClosureExternalRef> &refs) {

  auto traitType = sugarDynCast<TraitType>(closureParam.getType());
  if (!traitType)
    return;

  // Collect alias ops - these represent external parameter references.
  auto collectAliases = [&](TraitDeclOp closureTrait) {
    // The externalized reference names are exactly the trait's non-inherited
    // aliases; only witness references to those names should be rewritten.
    DenseSet<StringRef> aliasNames;
    for (AliasDeclOp aliasOp : closureTrait.getOps<AliasDeclOp>())
      aliasNames.insert(aliasOp.getName());

    // The dependent capture types are internalized to a get_witness attr;
    // extern it back to a reference of the original parameter declaration.
    mlir::AttrTypeReplacer externCapture;
    externCapture.addReplacement([&](GetWitnessAttr witness) -> TypedAttr {
      if (aliasNames.contains(witness.getWitnessName().strref()))
        return ParamDeclRefAttr::get(witness.getWitnessName(),
                                     witness.getType());
      return witness;
    });
    for (AliasDeclOp aliasOp : closureTrait.getOps<AliasDeclOp>()) {
      Type externalType = externCapture.replace(aliasOp.getType());
      refs.push_back({closureParam, aliasOp.getName(), externalType});
    }
  };
  processClosureTraits(traitType, collectAliases);
}

/// Format a closure signature for diagnostics, omitting argument names.
/// E.g. "def(Int) -> Int".
static std::string formatClosureSignature(FnTypeGeneratorType sig,
                                          SharedState &shared,
                                          unsigned numPrependedCaptures = 0) {
  SmallVector<ParamDeclAttr> parameters =
      populateParametersFromFnGeneratorType(sig);
  ParamRefRemapper replacer(parameters);
  auto remapped = replacer.replace(sig);

  // A closure trait/wrapper is named from its signature, but it is a closure
  // interface, not a thin function pointer, so its name must not carry the
  // `thin` keyword (unlike a genuine thin function type or its `_PtrWrapper`).
  ASTTypePrinterContext ctx{&shared};
  ctx.suppressThin = true;
  std::string result = ASTType(remapped).getAsString(ctx);
  if (numPrependedCaptures)
    result += (Twine("{") + Twine(numPrependedCaptures) + "}").str();
  return result;
}

ASTDecl *
ClosureEmitter::createFnStructWrapper(ASTDecl &moduleDecl, ASTDecl &traitDecl,
                                      FnTypeGeneratorType rawSignatureType,
                                      SMLoc smLocation) {
  FnTypeGeneratorType signatureType =
      cast<FnTypeGeneratorType>(getCanonicalType(rawSignatureType));
  auto [capturedRefs, selfContainedSignature] =
      DeclResolver::createSelfContainedSignature(signatureType);
  selfContainedSignature =
      cast<FnTypeGeneratorType>(getCanonicalType(selfContainedSignature));

  // The struct we're trying to create looks like this:
  // struct FnClosureWrapper[Impl: def() -> Int](`def() -> Int`):
  //   def __init__(self):
  //     pass
  //   def __call__(self) -> Int:
  //     return Impl()

  // The wrapper relies only on the function signature. Use that as the struct
  // name.
  SmallString<128> name(ASTType(selfContainedSignature).getAsString({&shared}));
  name += "_PtrWrapper";
  TraitDeclOp trait = cast<TraitDeclOp>(traitDecl.getIfOperation());
  if (auto decls = moduleDecl.lookupInCurrentScope(name); !decls.empty()) {
    ASTDecl *existing = decls.front();
    // Two closure traits that share a canonical signature but
    // differ in implicit parameter name suffixes will hit the same cached
    // wrapper. If the traits are different then emit conformance.
    [[maybe_unused]] auto outcome = augmentWitnessTablesToConformTo(
        existing->getTypeDeclSelf(), &traitDecl);
    assert(succeeded(outcome) && "unexpected failure in lazy conformance");
    return existing;
  }

  StringRef implName = "Impl";

  auto module = cast<FileModuleOp>(moduleDecl.getIfOperation());
  Location location = shared.diags.translateLocation(smLocation);
  ImplicitLocOpBuilder b =
      ImplicitLocOpBuilder::atBlockBegin(location, module->getBlock());
  b.setInsertionPointAfter(trait);
  MLIRContext *ctx = b.getContext();

  // Give the struct a parameter "Impl" of the def pointer type.
  SmallVector<ParamDeclAttr> implParameters;
  SmallVector<ParamDeclAttr> captureParams;
  llvm::MapVector<StringAttr, std::pair<Type, TypedAttr>> aliases;
  {
    size_t aliasCount = 0;
    for (auto alias : trait.getFields().getOps<AliasDeclOp>()) {
      aliasCount++;
      StringAttr aliasName = alias.getParamDecl().getName();
      StringAttr captureName =
          b.getStringAttr("__capture_" + aliasName.getValue());
      ParamDeclAttr captureParam =
          ParamDeclAttr::get(ctx, captureName, alias.getType());
      captureParams.push_back(captureParam);
      TypedAttr captureRef = ParamDeclRefAttr::get(captureParam);
      aliases.insert({aliasName, {alias.getType(), captureRef}});
    }
    assert(aliasCount == capturedRefs.size() &&
           "expected top-level wrapper captures to mirror trait aliases");
  }
  llvm::append_range(implParameters, captureParams);
  ParamDeclAttr implType = ParamDeclAttr::get(implName, selfContainedSignature);
  implParameters.push_back(implType);

  SmallVector<PassingKind> wrapperPassingKinds(captureParams.size(),
                                               PassingKind::Inferred);
  wrapperPassingKinds.push_back(PassingKind::PosOnly);

  // Create a zero-size struct with the Impl parameter.
  std::pair<ASTDecl &, StructDeclOp> pair =
      createStruct(shared, moduleDecl, StringAttr::get(b.getContext(), name),
                   implParameters, smLocation, wrapperPassingKinds);
  ASTDecl &structDecl = pair.first;
  StructDeclOp declOp = pair.second;
  declOp.setDefinesClosure(true);
  declOp.setConvention(TypeConvention::RegisterPassableTrivial);

  ClosureParent callParent{shared, trait.bindReference({}), "__call__",
                           ClosureMethod::CALL};
  ClosureParent movable = getMoveParent();
  ClosureParent copyable = getCopyParent();
  ClosureParent deinitable = getDeinitableParent();
  SmallVector<ClosureParent> parents{callParent,
                                     getAnyParent(),
                                     movable,
                                     copyable,
                                     getImplicitlyCopyableParent(),
                                     deinitable,
                                     getTrivialRegisterTypeParent(),
                                     getRegisterPassableParent()};
  TraitType traitType = getTraitType(parents, moduleDecl);
  declOp.setCanonicalTrait(traitType);
  b.setInsertionPointToEnd(&declOp.getFields().front());
  for (auto [aliasName, value] : aliases)
    AliasDeclOp::create(b, ParamDeclAttr::get(aliasName, value.first),
                        value.second);

  // Emit conformance tables
  auto addWitnessEntry = [&](const ClosureParent &parent, FnOp impl) {
    auto traitParent = parent.getTrait(shared);
    b.setInsertionPointToEnd(&declOp.getBodyRegion().front());
    TraitSymbolArrayAttr immediateParents =
        traitParent.getImmediateParentsAttr();
    TraitSymbolAttr parentTrait = parent.getSymbol();

    ConformanceOp witnessTable =
        ConformanceOp::create(b, parentTrait, immediateParents);
    ASTDecl &witnessDecl = shared.declResolver->addDecl(
        witnessTable, structDecl.getLoc(), parentTrait.getFlattenedName(),
        &structDecl, {}, {}, -1);
    witnessDecl.resolvedness = DeclResolvedness::body;
    Block &block = witnessTable.getBody().emplaceBlock();
    b.setInsertionPointToStart(&block);
    SymbolConstantAttr symbolConstant = buildSymbol(impl, implParameters);
    WitnessOp::create(b, parent.getWitnessName(), /*sym_visibility=*/nullptr,
                      symbolConstant);
    if (parent.getClosureMethod() == ClosureMethod::CALL) {
      for (auto [aliasName, value] : aliases)
        WitnessOp::create(b, aliasName, /*sym_visibility=*/nullptr,
                          value.second);
    }

    return witnessTable;
  };

  // The constructor is a no-op.
  auto initName = StringAttr::get(ctx, "__init__");
  SmallVector<Type> initArgumentTypes;
  SmallVector<ArgConvention> argConventions;

  RefType refSelfType = ASTType(structDecl.getTypeDeclSelf())
                            .getRefForArgument(selfName.getValue(), true);
  argConventions.push_back(ArgConvention::ByRefResult);
  initArgumentTypes.push_back(refSelfType);
  b.setInsertionPointToEnd(&declOp.getFields().front());
  auto [initFnOp, initDecl] = synthesizeFunction(
      structDecl, initName, {}, PogListAttr::get(ctx), initArgumentTypes,
      argConventions,
      PogListAttr::get(ctx, {selfName}, {PassingKind::Implicit}),
      NoneType::get(ctx), SpecialFunctionKind::kInit, smLocation, b,
      /*fnEffects=*/{}, /*suffix=*/"", /*synthetic=*/true, InlineLevel::Always);
  b.setInsertionPointToStart(&initFnOp.getBodyRegion().front());
  IREmitter::emitNormalReturn(b);
  initDecl->resolvedness = DeclResolvedness::body;

  StructEmitter structEmitter(structDecl);

  // Empty __del__
  auto delFnOp = structEmitter.synthesizeEmptyDtor();
  addWitnessEntry(deinitable, delFnOp);

  // Empty move ctor.
  auto moveFnOp = structEmitter.synthesizeEmptyMoveOrCopyInit(true);
  declOp.setMoveInitAttr(getSymbolNoParamValues(declOp, moveFnOp));
  addWitnessEntry(movable, moveFnOp);

  // Empty copy ctor
  auto copyFnOp = structEmitter.synthesizeEmptyMoveOrCopyInit(false);
  declOp.setCopyInitAttr(getSymbolNoParamValues(declOp, copyFnOp));
  addWitnessEntry(copyable, copyFnOp);

  // All of these operations are trivial in all cases; the struct has no fields.
  generateIsTrivialSpecialAlias("__del__is_trivial", true, shared, structDecl,
                                deinitable);
  generateIsTrivialSpecialAlias("__move_ctor_is_trivial", true, shared,
                                structDecl, movable);
  generateIsTrivialSpecialAlias("__copy_ctor_is_trivial", true, shared,
                                structDecl, copyable);

  // Generate the __call__ method based on the function signature.
  // The __call__ method is effectively the in-source body of the function.
  // Mark it as *not* synthetic so that debugging will step into the body.
  auto [callMethod, parameters, result] = pushBackTraitFunctionImpl(
      callParent.getSignature(), structDecl, /*synthetic=*/false,
      StringAttr::get(shared.getContext(), "__call__"),
      SpecialFunctionKind::kNormal, callParent.getInlineLevel());
  callMethod.setInlineLevel(InlineLevel::Always);
  addWitnessEntry(callParent, callMethod);

  // Marker parents (AnyType, ImplicitlyCopyable, RegisterPassable,
  // TrivialRegisterPassable) declare no requirements of their own; their
  // inherited requirements (e.g. Copyable's copy init for ImplicitlyCopyable)
  // are witnessed in the declaring parent's ConformanceOp above. An empty
  // ConformanceOp per claimed trait is still required for
  // TypeConformsToTraitAttr::simplify() to verify conformance on concrete
  // closure types.
  for (const ClosureParent &parent : parents)
    if (parent.isEmpty())
      addConformanceTable(structDecl, parent, {});

  // Populate the body of ClosureWrapper::__call__.
  {
    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (shared.diBuilder)
      diScopeGuard = shared.diBuilder->pushScopeGuard(callMethod.getLocScope());
    ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockBegin(
        callMethod.getLoc(), callMethod.getBody());

    TypedAttr callee = ParamDeclRefAttr::get(implType);
    SmallVector<TypedAttr> paramArgs;
    ArrayRef<ParamDeclAttr> callParams = parameters;
    ArrayRef<ParamDeclAttr> auxiliaryParams;
    if (!captureParams.empty()) {
      assert(parameters.size() >= captureParams.size() &&
             "wrapper auxiliary parameters must correspond to captures");
      auxiliaryParams = parameters.take_front(captureParams.size());
      callParams = parameters.drop_front(captureParams.size());
    }
    llvm::append_range(
        paramArgs, llvm::map_range(auxiliaryParams, [](ParamDeclAttr param) {
          return TypedAttr(ParamDeclRefAttr::get(param));
        }));
    llvm::append_range(
        paramArgs,
        llvm::map_range(callParams, [](ParamDeclAttr p) -> TypedAttr {
          return ParamDeclRefAttr::get(p);
        }));
    if (!paramArgs.empty()) {
      callee = BindParamsAttr::get(callee.getContext(), callee, paramArgs,
                                   &shared.getEvaluationContext());
    }

    // Mark `__call__` as a transparent thunk so its identity delegates to the
    // wrapped function pointer's underlying generator.
    callMethod->setAttr(kTransparentThunkCalleeExprAttr, callee);

    SmallVector<Value> arguments;
    // Ignore the self field and pass the other arguments as-is.
    llvm::append_range(arguments,
                       callMethod.getBody()->getArguments().drop_front());
    auto calleeSig = cast<FnTypeGeneratorType>(callee.getType());
    if (failed(emitForwardingCall(builder, structDecl, callee, calleeSig,
                                  result, arguments)))
      return {};
  }

  return &structDecl;
}

Type ClosureEmitter::getConcreteClosureWrapperTypeForFnSymbol(
    ASTDecl &declScope, SMLoc loc, PValue fnPValue) {
  auto fnSig = cast<FnTypeGeneratorType>(fnPValue.getType());
  ASTDecl &moduleDecl = *declScope.getNearestDeclOfType<FileModuleOp>();
  auto rvClosureTrait = shared.getOrCreateClosureTrait(loc, moduleDecl, fnSig);
  ASTDecl *wrapper =
      createFnStructWrapper(moduleDecl, *rvClosureTrait, fnSig, loc);
  if (!wrapper)
    return {};
  auto structDeclOp = cast<StructDeclOp>(wrapper->getIfOperation());

  auto [fnVal, captureBindings] = selfContainedSymbolAndCaptures(
      fnPValue,
      cast<FnTypeGeneratorType>(structDeclOp.getInputParams().back().getType()),
      shared, shared.diags.translateLocation(loc));
  SmallVector<TypedAttr> wrapperBindings;
  llvm::append_range(wrapperBindings, captureBindings);
  wrapperBindings.push_back(fnVal);
  return structDeclOp.bindReference(wrapperBindings);
}

ASTDecl *ClosureEmitter::getOrCreateClosureTrait(
    FnTypeGeneratorType key, llvm::function_ref<ASTDecl *()> creation) {
  auto ptr = closureTraitCache.find(key);
  if (ptr != closureTraitCache.end())
    return ptr->getSecond();
  ASTDecl *traitDecl = creation();
  // Only cache successful creations. A null return (e.g. stub trait with no
  // methods from bytecode) leaves the key absent so a later package with the
  // full definition can fill the slot.
  if (traitDecl)
    closureTraitCache.insert({key, traitDecl});
  return traitDecl;
}

static std::pair<TypedAttr, SmallVector<TypedAttr>>
selfContainedSymbolAndCaptures(PValue fnPValue,
                               FnTypeGeneratorType wrapperImplType,
                               SharedState &shared, Location loc) {
  // Rebuild the symbol with captures materialized as leading parameters.
  auto fnSig = cast<FnTypeGeneratorType>(fnPValue.getType());
  auto [captures, selfContainedSig] =
      DeclResolver::createSelfContainedSignature(fnSig);
  selfContainedSig =
      cast<FnTypeGeneratorType>(getCanonicalType(selfContainedSig));
  // Remove captures in signature from symbol.
  auto symbol = cast<SymbolConstantAttr>(fnPValue.get());
  IndexRefRemapper toIdx(captures);
  SmallVector<TypedAttr> params;

  // The parameter index that the current unbound parameter will be wired to,
  // starting right after the captures since captures are prepended in
  // `selfContainedSig`.
  size_t unboundIdx = captures.size();
  for (TypedAttr binding : symbol.getParamValues()) {
    TypedAttr replaced = toIdx.replace(binding);
    // If this is a `_`, wire it to the corresponding generator input parameter
    // that we are going to create later.
    if (isa<UnboundAttr>(replaced)) {
      params.push_back(ParamIndexRefAttr::get(
          unboundIdx, selfContainedSig.getInputParamTypes()[unboundIdx]));
      unboundIdx++;
      continue;
    }
    // Any index reference must be a *direct* reference to a capture.
    if (auto idxRef = dyn_cast<ParamIndexRefAttr>(replaced);
        idxRef && idxRef.getDepth() == 0) {
      auto bindingRef = cast<ParamDeclRefAttr>(binding);
      assert(llvm::any_of(captures, [&](ParamDeclRefAttr capture) {
        return capture.getName() == bindingRef.getName() &&
               isEqualCanon(capture.getType(), bindingRef.getType());
      }));
    }
    params.push_back(replaced);
  };

  FuncSymbolAttr fnSymbol = FuncSymbolAttr::get(
      symbol.getSymbol(), selfContainedSig.getBody(), params);
  TypedAttr fnVal =
      GeneratorAttr::get(selfContainedSig.getInputParamTypes(), fnSymbol,
                         selfContainedSig.getParamListAttrs());

  assert(
      ClosureEmitter::isTypeRebindableTo(
          cast<FuncTypeGeneratorType>(fnVal.getType()), wrapperImplType) &&
      "self-contained promoted signature must match wrapper Impl canonically");
  if (fnVal.getType() != wrapperImplType)
    fnVal = ParamOperatorAttr::getRebind(fnVal, wrapperImplType);

  ParameterEvaluator evaluator(populateParametersFromFnGeneratorType(fnSig),
                               symbol.getParamValues());
  SmallVector<TypedAttr> captureBindings;
  captureBindings.reserve(captures.size());
  for (ParamDeclRefAttr capture : captures)
    captureBindings.push_back(
        cast<TypedAttr>(evaluator.getReboundAttribute(capture)));
  return {fnVal, captureBindings};
}

// Find all extern parameter references in the sig. For each reference, create
// an alias. Replace the original extern parameter reference by calling the
// custom replacer
static std::pair<FnTypeGeneratorType, llvm::MapVector<StringRef, Type>>
extractParameterReferencesIntoAliasRef(
    FnTypeGeneratorType dependentSignatureType, StringRef selfName,
    llvm::function_ref<TypedAttr(StringRef, Type)> externParameterRefReplacer) {
  DenseSet<StringRef> callParams;
  for (PogMetadataAttr pog :
       dependentSignatureType.getParamListAttrs().getPogs())
    callParams.insert(pog.getName());
  callParams.insert(selfName);
  llvm::MapVector<StringRef, Type> aliasMembers;
  mlir::AttrTypeReplacer externRefReplacer;
  FnTypeGeneratorType canonicalType =
      cast<FnTypeGeneratorType>(getCanonicalType(dependentSignatureType));
  externRefReplacer.addReplacement(
      [&](ParamDeclRefAttr reference) -> TypedAttr {
        if (!callParams.contains(reference.getName().getValue())) {
          auto ptr = aliasMembers.find(reference.getName());
          if (ptr == aliasMembers.end()) {
            Type replacedType = externRefReplacer.replace(reference.getType());
            aliasMembers.insert({reference.getName(), replacedType});
            return externParameterRefReplacer(reference.getName(),
                                              replacedType);
          } else {
            return externParameterRefReplacer(ptr->first, ptr->second);
          }
        }
        return reference;
      });
  auto newSignature =
      cast<FnTypeGeneratorType>(externRefReplacer.replace(canonicalType));
  return {newSignature, aliasMembers};
}

static std::pair<FnTypeGeneratorType, llvm::MapVector<StringRef, Type>>
extractParameterReferencesIntoAliasRef(
    ASTDecl &decl, FnTypeGeneratorType dependentSignatureType) {
  TraitDeclOp closureTrait = cast<TraitDeclOp>(decl.getIfOperation());
  SharedState &shared = decl.getShared();
  MLIRContext *ctx = shared.getContext();
  ASTType selfType = decl.getTypeDeclSelf();
  auto declRef = dyn_cast<ParamType>(selfType.mlirType);
  auto ref = dyn_cast_if_present<ParamDeclRefAttr>(declRef.getParam());
  assert(ref && "expected the self type of a trait to be a parameter");
  auto traitSymbol =
      TraitSymbolAttr::get(getFullyResolvedSymbolRef(closureTrait));
  StringRef selfName = ref.getName().getValue();
  auto externParamReplacer = [&](StringRef witnessName,
                                 Type witnessType) -> TypedAttr {
    return GetWitnessAttr::get(PValue(selfType), traitSymbol,
                               StringAttr::get(ctx, witnessName), witnessType);
  };
  return extractParameterReferencesIntoAliasRef(dependentSignatureType,
                                                selfName, externParamReplacer);
}

std::pair<FnTypeGeneratorType, unsigned>
ClosureEmitter::getClosureTraitKey(FnTypeGeneratorType rawSignature) {
  auto [capturedRefs, selfContainedSig] =
      DeclResolver::createSelfContainedSignature(rawSignature);
  auto canonicalSig =
      cast<FnTypeGeneratorType>(getCanonicalType(selfContainedSig));

  // Normalize auto parameters.
  PogListAttr meta = canonicalSig.getParamListAttrs();
  ArrayRef<PogMetadataAttr> pogs = meta.getPogs();
  SmallVector<PogMetadataAttr> normalizedPogs(pogs.begin(), pogs.end());
  bool changed = false;
  for (size_t i = 0; i < normalizedPogs.size(); ++i) {
    PogMetadataAttr pog = normalizedPogs[i];
    PassingKind pk = pog.getPassingKind();
    StringRef name = pog.getName().getValue();
    if (!isHiddenGeneratorParam(pk, name))
      continue;
    StringRef newName;
    SmallString<32> positional;
    if (name.contains("._mlir_origin")) {
      newName = demangleParameterName(name);
    } else {
      positional = ("." + Twine(i)).str();
      newName = positional;
    }
    if (newName == name)
      continue;
    normalizedPogs[i] = PogMetadataAttr::get(
        StringAttr::get(canonicalSig.getContext(), newName), pk,
        pog.getVariadic(), pog.getDefaultValue());
    changed = true;
  }
  PogListAttr normalizedMeta = changed ? meta.cloneWith(normalizedPogs) : meta;

  FnTypeGeneratorType key = FnTypeGeneratorType::get(
      canonicalSig.getInputParamTypes(), canonicalSig.getValues(),
      canonicalSig.getArgConventions(),
      canonicalSig.getFnEffects().setCapturing(false),
      canonicalSig.getFnMetaOriginData(), normalizedMeta,
      canonicalSig.getArgListAttrs());
  return {key, capturedRefs.size()};
}

ASTDecl *ClosureEmitter::createClosureTrait(
    ASTDecl &moduleDecl, FnTypeGeneratorType dependentSignatureType,
    FnTypeGeneratorType key, unsigned numPrependedCaptures,
    SMLoc nestedFunctionOrTypeLocation) {
  // Generate the movable, destructable closure trait, populating the trait
  // definition with the single characteristic "__call__" method.
  SmallVector<ClosureParent> parents{getMoveParent(), getDeinitableParent()};
  auto populate = [&](ASTDecl &decl,
                      DenseSet<std::pair<StringAttr, StringAttr>> &functions) {
    TraitDeclOp closureTrait = cast<TraitDeclOp>(decl.getIfOperation());
    auto [signatureNoSelf, aliasMembers] =
        extractParameterReferencesIntoAliasRef(decl, dependentSignatureType);
    ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockEnd(
        closureTrait.getLoc(), &closureTrait.getFields().front());
    for (auto [aliasName, aliasType] : aliasMembers) {
      shared.declResolver->addFullyResolvedDecl(
          AliasDeclOp::create(
              builder, ParamDeclAttr::get(ctx, builder.getStringAttr(aliasName),
                                          aliasType)),
          aliasName, decl.getLoc(), &decl);
    }

    RefType refType = decl.getTypeDeclSelf().getRefForArgument("self", true);
    FnTypeGeneratorType sig = addClosureSelfArgToFunctionSignature(
        refType, ArgConvention::ImmMem, signatureNoSelf);
    // Augment the call function with auxiliary parameters. These auxiliary
    // parameters enable rebinding argument types in terms of external
    // parameters (e.g. "T") in terms of the alias members of closure type C
    SmallVector<ParamDeclAttr> sigParams(
        populateParametersFromFnGeneratorType(sig));
    SmallVector<PogMetadataAttr> extendedPogs;
    DenseMap<StringRef, ParamDeclAttr> aliasNameToParam;
    SmallVector<ParamDeclAttr> parameters;
    for (auto [aliasName, aliasType] : aliasMembers) {
      StringAttr nameAttr = builder.getStringAttr("_" + Twine(aliasName));
      ParamDeclAttr param = ParamDeclAttr::get(ctx, nameAttr, aliasType);
      parameters.push_back(param);
      aliasNameToParam[aliasName] = param;
      extendedPogs.push_back(
          PogMetadataAttr::get(nameAttr, PassingKind::Inferred));
    }
    llvm::append_range(extendedPogs, sig.getParamListAttrs().getPogs());
    PogListAttr extendedParamListAttrs = PogListAttr::get(
        ctx, extendedPogs, sig.getParamListAttrs().getBodyConstraints(),
        sig.getParamListAttrs().getOrigVariadicConvention());
    auto callName = StringAttr::get(ctx, "__call__");
    // Calculate the argument types and result types in terms of the named
    // parameters. Also replace GetWitnessAttr references to aliases with
    // references to auxiliary parameters.
    ParamRefRemapper replacer(sigParams);
    mlir::AttrTypeReplacer aliasReplacer;
    aliasReplacer.addReplacement([&](GetWitnessAttr getWitness) -> TypedAttr {
      StringRef witnessName = getWitness.getWitnessName().getValue();
      auto it = aliasNameToParam.find(witnessName);
      if (it != aliasNameToParam.end())
        return ParamDeclRefAttr::get(it->second);
      return getWitness;
    });
    // Internalize all get_witness attr references to the corresponding
    // parameter declaration.
    llvm::append_range(parameters, sigParams);
    for (ParamDeclAttr &p : parameters)
      p = cast<ParamDeclAttr>(aliasReplacer.replace(replacer.replace(p)));

    SmallVector<Type> argumentTypes =
        llvm::map_to_vector(sig.getArguments(), [&](Type original) -> Type {
          return aliasReplacer.replace(replacer.replace(original));
        });
    Type result = cast<Type>(
        aliasReplacer.replace(replacer.replace(sig.getResultType())));
    // TODO: remove capturing when legacy closures are removed.
    auto [fnOp, fnDecl] = synthesizeFunction(
        decl, callName, parameters, extendedParamListAttrs, argumentTypes,
        sig.getArgConventions(), sig.getArgListAttrs(), result,
        SpecialFunctionKind::kNormal, nestedFunctionOrTypeLocation, builder,
        sig.getFnEffects().setCapturing(true), "", true, InlineLevel::Always);
    builder.setInsertionPointToEnd(&fnOp.getBodyRegion().front());
    UnreachableOp::create(builder);
    functions.insert({callName, fnOp.getSymNameAttr()});
  };
  StringAttr name = StringAttr::get(
      shared.getContext(),
      formatClosureSignature(key, shared, numPrependedCaptures));
  auto createTraitFn = [&]() -> ASTDecl * {
    auto [closureTrait, traitDecl] =
        createTraitOp(name, parents, nestedFunctionOrTypeLocation, populate);
    closureTrait.setClosureSignature(key);
    std::string prettyName =
        formatClosureSignature(dependentSignatureType, shared);
    closureTrait.setSourceNameAttr(DebugInfo::SourceNameAttr::get(
        StringAttr::get(shared.getContext(), prettyName)));
    return traitDecl;
  };
  return getOrCreateClosureTrait(key, createTraitFn);
}

TraitType
ClosureEmitter::getSpecializedClosureTrait(GeneratorType aliasGenerator,
                                           ArrayRef<TypedAttr> paramValues,
                                           ASTDecl &moduleDecl, SMLoc loc) {
  // Locate the closure defining trait.
  auto anyTrait = sugarDynCastIfPresent<AnyTraitType>(aliasGenerator.getBody());
  if (!anyTrait)
    return {};
  TraitType origTraitType = anyTrait.getTraitType();
  TraitDeclOp closureTrait;
  TraitSymbolAttr closureSymbol;
  for (TraitSymbolAttr symbol : origTraitType.getSymbols()) {
    ASTDecl &decl =
        shared.getDeclResolver().getDeclForTypeSymbol(symbol.getSymbol());
    if (auto candidate =
            dyn_cast_if_present<TraitDeclOp>(decl.getIfOperation());
        candidate && candidate.getDefinesClosure()) {
      closureTrait = candidate;
      closureSymbol = symbol;
      break;
    }
  }
  if (!closureTrait)
    return {};
  std::optional<FnTypeGeneratorType> keyOr = closureTrait.getClosureSignature();
  if (!keyOr)
    return {};
  FnTypeGeneratorType key = *keyOr;

  // Map each alias parameter name to its bound value
  ArrayRef<PogMetadataAttr> genPogs =
      aliasGenerator.getParamListAttrs().getPogs();
  llvm::DenseMap<StringAttr, TypedAttr> valueByName;
  for (auto [pog, value] : llvm::zip(genPogs, paramValues))
    valueByName[pog.getName()] = value;

  // "Bind" param to alias by replacing pog.
  ArrayRef<PogMetadataAttr> keyPogs = key.getParamListAttrs().getPogs();
  ArrayRef<Type> keyParamTypes = key.getInputParamTypes();
  SmallVector<TypedAttr> keyBindings;
  keyBindings.reserve(keyPogs.size());
  bool boundAny = false;
  for (auto [pog, paramType] : llvm::zip(keyPogs, keyParamTypes)) {
    auto it = valueByName.find(pog.getName());
    if (it != valueByName.end()) {
      keyBindings.push_back(it->second);
      boundAny = true;
    } else {
      keyBindings.push_back(UnboundAttr::get(paramType));
    }
  }
  if (!boundAny)
    return {};

  // Substitute the arguments into the closure signature and (re)create the
  // trait.
  auto reboundSig = dyn_cast_if_present<FnTypeGeneratorType>(
      key.getSpecializedGenerator(keyBindings, /*evaluationContext=*/nullptr,
                                  shared.translateLocation(loc)));
  if (!reboundSig)
    return {};

  ASTDecl *newTraitDecl =
      shared.getOrCreateClosureTrait(loc, moduleDecl, reboundSig);
  if (!newTraitDecl)
    return {};
  auto newClosureSymbol = TraitSymbolAttr::get(getFullyResolvedSymbolRef(
      cast<mlir::SymbolOpInterface>(newTraitDecl->getIfOperation())));

  // Preserve the non-closure conjuncts
  SmallVector<TraitSymbolAttr> symbols;
  for (TraitSymbolAttr symbol : origTraitType.getSymbols())
    symbols.push_back(symbol == closureSymbol ? newClosureSymbol : symbol);
  return TraitType::canonicalizeAndGet(shared.getContext(), symbols, {});
}

static bool hasCapturingParameterType(SharedState &shared,
                                      ArrayRef<ParamDeclAttr> params) {
  mlir::AttrTypeWalker walker;
  walker.addWalk([](FuncType sig) {
    if (sig.isCapturing())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  walker.addWalk([&](SymbolRefAttr symbol) {
    ASTDecl *traitDecl =
        shared.getDeclResolver().getDeclForTypeSymbolIfExists(symbol);
    if (!traitDecl)
      return WalkResult::advance();
    auto traitDeclOp =
        dyn_cast_if_present<TraitDeclOp>(traitDecl->getIfOperation());
    if (traitDeclOp && traitDeclOp.getDefinesClosure())
      return WalkResult::interrupt();
    return WalkResult::advance();
  });

  return llvm::any_of(params, [&](ParamDeclAttr param) {
    return walker.walk(param).wasInterrupted();
  });
}

/// Prepend a new implicit origin at index 0 in `sig`, shifting all existing
/// depth-local ImplicitOriginRefAttrs up by one and incrementing the
/// FnMetadata origin count. This is the implicit-origin analogue of
/// FnTypeGeneratorType::prependParams for explicit parameters.
static FnTypeGeneratorType prependImplicitOriginDecl(FnTypeGeneratorType sig) {
  struct OriginIndexShifter
      : public IndexParameterReplacer<OriginIndexShifter> {
    Type tryReplace(Type, size_t) { return {}; }
    Attribute tryReplace(Attribute attr, size_t depth) {
      // Only shift refs that are scoped to this function (depth + 1 == depth
      // after the replacer increments depth for each nesting level).
      if (depth == 0)
        return {};
      if (auto ref = dyn_cast<ImplicitOriginRefAttr>(attr);
          ref && ref.getDepth() + 1 == depth)
        return ImplicitOriginRefAttr::get(ref.getDepth(), ref.getIndex() + 1,
                                          ref.getType());
      return {};
    }
  } shifter;
  sig = cast<FnTypeGeneratorType>(shifter.replace(sig));
  FnMetaOriginDataAttr oldMeta = sig.getFnMetaOriginData();
  FnMetaOriginDataAttr newMeta = FnMetaOriginDataAttr::get(
      sig.getContext(), oldMeta.getNumImplicitOriginDecls() + 1,
      oldMeta.getCaptureOrigins(), oldMeta.getIsNestedOriginsReadOnly(),
      oldMeta.getDefinesInteriorOrigins());
  return FnTypeGeneratorType::get(sig.getInputParamTypes(), sig.getValues(),
                                  sig.getArgConventions(), sig.getFnEffects(),
                                  newMeta, sig.getParamListAttrs(),
                                  sig.getArgListAttrs());
}

struct PromotedSignature {
  FnTypeGeneratorType signature;
  FunctionType functionType;
  Type selfRuntimeArgType;
  SmallVector<ParamDeclAttr> newParams;
};

/// Build the promoted signature for a closure being lifted to file scope.
static PromotedSignature buildPromotedSignature(
    SharedState &shared, FnTypeGeneratorType sig,
    ArrayRef<ParamDeclAttr> params, ArrayRef<ParamDeclAttr> prependedParams,
    std::optional<ClosureEmitter::PromotedClosureSelfArg> selfArg,
    std::optional<bool> capturingOverride = std::nullopt) {
  MLIRContext *ctx = shared.getContext();
  size_t oldNumImplicitOrigins =
      sig.getFnMetaOriginData().getNumImplicitOriginDecls();
  assert(oldNumImplicitOrigins <= params.size());

  // Step 1: prepend a new origin slot for self and collect origin decls.
  SmallVector<ParamDeclAttr> implicitOriginDecls;
  ParamDeclAttr selfImplicitOriginDecl;
  if (selfArg) {
    selfImplicitOriginDecl = ParamDeclAttr::get(
        StringAttr::get(ctx, "__self_origin"), OriginType::get(ctx, true));
    implicitOriginDecls.push_back(selfImplicitOriginDecl);
    sig = prependImplicitOriginDecl(sig);
  }
  llvm::append_range(implicitOriginDecls,
                     params.take_back(oldNumImplicitOrigins));

  // Step 2: prepend explicit params for each captured parameter.
  SmallVector<ParamDeclAttr> explicitPrependedParams(prependedParams.begin(),
                                                     prependedParams.end());
  std::optional<IndexRefRemapper> prependParamRefRemapper;
  if (!explicitPrependedParams.empty()) {
    prependParamRefRemapper =
        std::make_optional<IndexRefRemapper>(explicitPrependedParams);
    sig = FnTypeGeneratorType::prependParams(sig, explicitPrependedParams);
  }

  // Step 3: prepend the self runtime argument so it becomes arg[0].
  if (selfArg) {
    Type selfArgType = selfArg->type;
    if (prependParamRefRemapper)
      selfArgType = prependParamRefRemapper->replace(selfArgType);

    Type selfRefType = RefType::get(
        selfArgType,
        ImplicitOriginRefAttr::get(0, 0, selfImplicitOriginDecl.getType()));
    sig = addClosureSelfArgToFunctionSignature(selfRefType, selfArg->convention,
                                               sig);
  }

  // Step 4: resolve depth-local index refs to named param/origin references.
  SmallVector<ParamDeclAttr> explicitParamDecls;
  llvm::append_range(explicitParamDecls, explicitPrependedParams);
  llvm::append_range(explicitParamDecls,
                     params.drop_back(oldNumImplicitOrigins));
  assert(explicitParamDecls.size() == sig.getInputParamTypes().size());
  FunctionType promotedFunctionType = replaceIndexRefsWithNamedRefs(
      sig.getValues(), explicitParamDecls, implicitOriginDecls);

  bool shouldBeCapturing;
  if (capturingOverride)
    shouldBeCapturing = *capturingOverride;
  else
    shouldBeCapturing =
        sig.isCapturing() ||
        hasCapturingParameterType(shared, explicitPrependedParams);
  FnTypeGeneratorType promotedSignature = FnTypeGeneratorType::get(
      sig.getInputParamTypes(), sig.getValues(), sig.getArgConventions(),
      sig.getFnEffects().setCapturing(shouldBeCapturing),
      sig.getFnMetaOriginData(), sig.getParamListAttrs(),
      sig.getArgListAttrs());

  Type selfRuntimeArgType;
  if (selfArg)
    selfRuntimeArgType = promotedFunctionType.getInputs().front();

  // Assemble the full param list when something changed: new capture params
  // were prepended or a self origin was inserted.
  SmallVector<ParamDeclAttr> newParams;
  if (!explicitPrependedParams.empty() || selfArg) {
    ArrayRef<ParamDeclAttr> oldExplicit =
        params.drop_back(oldNumImplicitOrigins);
    newParams.reserve(explicitPrependedParams.size() + oldExplicit.size() +
                      implicitOriginDecls.size());
    llvm::append_range(newParams, explicitPrependedParams);
    llvm::append_range(newParams, oldExplicit);
    llvm::append_range(newParams, implicitOriginDecls);
  }

  return {promotedSignature, promotedFunctionType, selfRuntimeArgType,
          std::move(newParams)};
}

ASTDecl *ClosureEmitter::promoteClosure(
    ASTDecl &nestedFnDecl, ArrayRef<ParamDeclAttr> prependedParams,
    std::optional<PromotedClosureSelfArg> selfArg,
    std::optional<bool> capturingOverride, ASTDecl *targetParent) {
  assert(nestedFnDecl.resolvedness == DeclResolvedness::body &&
         "nested decl must be fully resolved to promote");
  // Mark dead unparsed code as resolved to prevent resolution dependent on
  // the old parent scope, which is about to change below.
  for (auto &[_, children] : nestedFnDecl.getDeclsInScope()) {
    for (ASTDecl *child : children) {
      if (child->getParentDecl() == &nestedFnDecl &&
          child->resolvedness == DeclResolvedness::unparsed)
        child->resolvedness = DeclResolvedness::body;
    }
  }
  MLIRContext *ctx = shared.getContext();
  FnOp function = cast<FnOp>(nestedFnDecl.getIfOperation());
  SMLoc loc = nestedFnDecl.getLoc();
  if (!targetParent)
    targetParent = nestedFnDecl.getNearestDeclOfType<FileModuleOp>();
  assert(targetParent && "expected a target parent for promotion");

  auto [promotedSignature, promotedFunctionType, selfRuntimeArgType,
        newParams] =
      buildPromotedSignature(shared, function.getFuncTypeGenerator(),
                             function.getParams(), prependedParams, selfArg,
                             capturingOverride);

  OpBuilder builder = targetParent->getDeclEndBuilder();
  function->moveBefore(builder.getInsertionBlock(),
                       builder.getInsertionPoint());
  function.setSymName(
      targetParent->mangleParamName(function.getSymName()->str()));

  // Update function attributes and body to reflect self argument addition.
  if (selfArg) {
    auto &entryBlock = function.getBodyRegion().front();
    Location selfArgLoc = function.getLoc();
    if (FileLineColLoc sourceLoc = DebugInfo::extractSourceLoc(selfArgLoc))
      selfArgLoc = sourceLoc;
    entryBlock.insertArgument(static_cast<unsigned>(0), selfRuntimeArgType,
                              selfArgLoc);

    if (ArrayAttr argMeta = function.getLLVMArgMetadataArray();
        argMeta && !argMeta.empty()) {
      SmallVector<Attribute> newMeta;
      newMeta.push_back(ArrayAttr::get(ctx, {}));
      llvm::append_range(newMeta, argMeta);
      function.setLLVMArgMetadataArrayAttr(ArrayAttr::get(ctx, newMeta));
    }

    // Augment DISubroutineType with self argument.
    if (DebugInfo::DISubprogramAttr oldScope = function.getSubprogramScope()) {
      auto subroutineType =
          cast<DebugInfo::DISubroutineType>(oldScope.getType());
      SmallVector<DebugInfo::DIType> updatedArgTypes;
      updatedArgTypes.push_back(DebugInfo::DIUnspecifiedType::get(ctx, "self"));
      llvm::append_range(updatedArgTypes, subroutineType.getArgumentTypes());
      auto newSubroutineType = DebugInfo::DISubroutineType::get(
          ctx, subroutineType.getCallingConvention(), updatedArgTypes,
          subroutineType.getResultTypes());
      auto newScope = DebugInfo::DISubprogramAttr::get(
          oldScope.getCompileUnit(), oldScope.getScope(),
          oldScope.getSourceName(), oldScope.getLinkageName(),
          oldScope.getFile(), oldScope.getLine(), oldScope.getScopeLine(),
          oldScope.getSubprogramFlags(),
          cast<DebugInfo::DISubroutineType>(newSubroutineType));
      mlir::AttrTypeReplacer replacer;
      replacer.addReplacement(
          [&](DebugInfo::DISubprogramAttr sp) -> DebugInfo::DISubprogramAttr {
            if (sp == oldScope)
              return newScope;
            return sp;
          });
      replacer.recursivelyReplaceElementsIn(function, /*replaceAttrs=*/true,
                                            /*replaceLocs=*/true);
    }
  }
  function.setFuncTypeGenerator(promotedSignature);
  function.setFunctionType(promotedFunctionType);
  if (!newParams.empty())
    function.setParamsAttr(ParamDeclArrayAttr::get(ctx, newParams));
  // Transfer the linkage name to the promoted op: the mangled sym_name
  // above overwrites the original name, so preserve it so it survives
  // into elaboration.
  if (auto linkageName = function.getLinkageNameAttr())
    function.setLinkageNameAttr(linkageName);
  function.setNoDocRequired(true);
  function.setSynthetic(true);
  auto &decl = shared.declResolver->addFullyResolvedDecl(
      function, /*name=*/StringAttr(), loc, targetParent);
  // Transfer child decls from the original to the promoted decl. Since the op
  // was moved (not cloned), all mlir::Value pointers are still valid.
  decl.takeDecls(nestedFnDecl);
  // Register the lifted function to the symbol table.
  [[maybe_unused]] Operation *existing =
      shared.declResolver->finalizeFuncSignature(function, decl);
  assert(!existing && "unexpected redefinition of promoted closure");
  if (prependedParams.empty()) {
    nestedFnDecl.setIRValue(function);
    return &decl;
  }

  ArrayRef<ParamDeclAttr> promotedFnParams = function.getParams();
  ArrayRef<ParamDeclAttr> captureParams =
      promotedFnParams.take_front(prependedParams.size());
  SmallVector<TypedAttr> bindings;
  bindings.reserve(promotedFnParams.size());
  size_t captureIndex = 0;
  for (auto [paramIndex, param] : llvm::enumerate(promotedFnParams)) {
    if (paramIndex < captureParams.size()) {
      bindings.push_back(ParamDeclRefAttr::get(captureParams[captureIndex++]));
      continue;
    }
    bindings.push_back(UnboundAttr::get(ctx, param.getType()));
  }
  assert(captureIndex == captureParams.size() &&
         "all capture params must be rebound");
  nestedFnDecl.setIRValue(PValue(function.getFuncLiteralGenerator(
      shared.getEvaluationContext(),
      ParameterExprArrayAttr::get(ctx, bindings))));
  return &decl;
}

ASTDecl *ClosureEmitter::promoteClosure(
    ASTDecl &nestedFnDecl, ArrayRef<ParamDeclRefAttr> prependedParamRefs,
    std::optional<PromotedClosureSelfArg> selfArg,
    std::optional<bool> capturingOverride, ASTDecl *targetParent) {
  SmallVector<ParamDeclAttr> prependedParams =
      llvm::map_to_vector(prependedParamRefs, [](ParamDeclRefAttr paramRef) {
        return ParamDeclAttr::get(paramRef);
      });
  return promoteClosure(nestedFnDecl, prependedParams, selfArg,
                        capturingOverride, targetParent);
}

template <typename T>
static SymbolRefAttr getFullyResolvedSymbolRefUpTo(mlir::SymbolOpInterface op) {
  SmallVector<FlatSymbolRefAttr> symbols;
  Operation *current = op;
  while (current && !isa<T>(current)) {
    if (mlir::SymbolOpInterface next =
            dyn_cast<mlir::SymbolOpInterface>(current))
      symbols.push_back(FlatSymbolRefAttr::get(next.getNameAttr()));
    current = current->getParentOp();
  }
  if (symbols.size() == 1)
    return symbols.front();
  std::reverse(symbols.begin(), symbols.end());
  return SymbolRefAttr::get(symbols[0].getAttr(),
                            ArrayRef(symbols).drop_front());
}

static void meetOriginMutability(DenseMap<StringAttr, bool> &originMutability,
                                 StringAttr name, bool isKnownImmutable) {
  auto [it, isNew] = originMutability.try_emplace(name, /*mutable=*/false);
  if (!isKnownImmutable)
    it->second = true;
}

static bool isOutlinedInteriorOrigin(
    TypedAttr typed,
    const llvm::MapVector<TypedAttr, ParamDeclRefAttr> &interiorOrigins) {
  if (!typed || !sugarIsa<OriginType>(typed.getType()))
    return false;
  TypedAttr stripped = OriginType::stripMutCastAndRebind(typed);
  if (!sugarIsa<InteriorOriginAttr, OriginSubtreeAttr>(stripped))
    return false;
  return interiorOrigins.contains(cast<TypedAttr>(getCanonicalAttr(stripped)));
}

static bool hasOutlinedInteriorOrigin(
    Type type,
    const llvm::MapVector<TypedAttr, ParamDeclRefAttr> &interiorOrigins) {
  WalkResult result = type.walk([&](Attribute attr) {
    auto typed = dyn_cast<TypedAttr>(attr);
    return isOutlinedInteriorOrigin(typed, interiorOrigins)
               ? WalkResult::interrupt()
               : WalkResult::advance();
  });
  return result.wasInterrupted();
}

// Given an attribute, update origin mutability information and register any
// interior origins for outlining. Returns false to stop recursion for origin
// typed nodes so cast operands and base origins of outlined interior origins
// aren't counted separately.
static bool checkOriginAndOutline(
    MLIRContext *ctx, DenseMap<StringAttr, bool> &originMutability,
    llvm::MapVector<TypedAttr, ParamDeclRefAttr> &interiorOrigins,
    Attribute attr) {
  auto typed = dyn_cast<TypedAttr>(attr);
  if (!typed || !sugarIsa<OriginType>(typed.getType()))
    return true;

  // Strip mutcasts before classifying so an immutable use of a mutable
  // interior origin (mutcast(interior)) is not counted as a mutable use of
  // the outlined parameter. Mutability is still taken from `typed`.
  TypedAttr stripped = OriginType::stripMutCastAndRebind(typed);

  if (sugarIsa<InteriorOriginAttr, OriginSubtreeAttr>(stripped)) {
    TypedAttr canon = cast<TypedAttr>(getCanonicalAttr(stripped));
    auto it = interiorOrigins.find(canon);
    if (it == interiorOrigins.end()) {
      StringAttr name = StringAttr::get(ctx, Twine("?__interior_origin_") +
                                                 Twine(interiorOrigins.size()));
      ParamDeclAttr param = ParamDeclAttr::get(name, canon.getType());
      it = interiorOrigins.insert({canon, ParamDeclRefAttr::get(param)}).first;
    }
    meetOriginMutability(originMutability, it->second.getName(),
                         OriginType::isMutableKnown(typed, false));
    return false;
  }

  if (auto ref = dyn_cast<ParamDeclRefAttr>(stripped)) {
    meetOriginMutability(originMutability, ref.getName(),
                         OriginType::isMutableKnown(typed, false));
    return false;
  }

  // Otherwise this is a derived origin such as a field projection,
  // which holds the origins it is derived from in its sub-elements. Keep
  // walking so those are counted at their own mutability; stopping here would
  // leave a mutably-used origin unrecorded and let it be wrongly promoted.
  return true;
}

template <typename AttrOrType>
static void checkOriginAndOutlineImpl(
    MLIRContext *ctx, DenseMap<StringAttr, bool> &originMutability,
    llvm::MapVector<TypedAttr, ParamDeclRefAttr> &interiorOrigins,
    AttrOrType attrOrType) {
  if (!attrOrType)
    return;
  if constexpr (std::is_convertible_v<AttrOrType, Attribute>) {
    if (!checkOriginAndOutline(ctx, originMutability, interiorOrigins,
                               attrOrType))
      return;
  }
  attrOrType.walkImmediateSubElements(
      [&](Attribute attribute) {
        checkOriginAndOutlineImpl(ctx, originMutability, interiorOrigins,
                                  attribute);
      },
      [&](Type type) {
        checkOriginAndOutlineImpl(ctx, originMutability, interiorOrigins, type);
      });
}

template <typename AttrOrType>
static void checkOriginAndOutline(
    MLIRContext *ctx, DenseMap<StringAttr, bool> &originMutability,
    llvm::MapVector<TypedAttr, ParamDeclRefAttr> &interiorOrigins,
    AttrOrType attrOrType) {
  if constexpr (std::is_convertible_v<AttrOrType, Attribute>)
    attrOrType = cast<AttrOrType>(getCanonicalAttr(attrOrType));
  else
    attrOrType = cast<AttrOrType>(getCanonicalType(attrOrType));
  checkOriginAndOutlineImpl(ctx, originMutability, interiorOrigins, attrOrType);
}

static SmallPtrSet<StringAttr, 8>
collectPromotedOrigins(MLIRContext *ctx,
                       const DenseMap<StringAttr, bool> &originMutability,
                       SmallVectorImpl<ParamDeclAttr> &structParams,
                       SmallVectorImpl<TypedAttr> &structBindings) {
  SmallPtrSet<StringAttr, 8> promotedOriginNames;
  for (auto [index, param] : llvm::enumerate(structParams)) {
    StringAttr name = param.getName();
    auto originType = sugarDynCast<OriginType>(param.getType());
    // If an origin is already immutable, no need to promote.
    if (!originType || originType.isMutableKnown(false))
      continue;
    if (auto it = originMutability.find(name);
        it != originMutability.end() && it->second)
      continue;
    structParams[index] = ParamDeclAttr::get(name, OriginType::get(ctx, false));
    structBindings[index] =
        OriginMutCastAttr::get(structBindings[index], false);
    promotedOriginNames.insert(name);
  }
  return promotedOriginNames;
}

static TypedAttr
getOutlinedOriginRef(MLIRContext *ctx, ParamDeclRefAttr paramRef,
                     const SmallPtrSetImpl<StringAttr> &promotedOriginNames) {
  StringAttr name = paramRef.getName();
  Type originType = promotedOriginNames.contains(name)
                        ? OriginType::get(ctx, false)
                        : paramRef.getType();
  return ParamDeclRefAttr::get(name, originType);
}

static std::optional<TypedAttr>
getPromotedOriginRef(MLIRContext *ctx, TypedAttr origin,
                     const SmallPtrSetImpl<StringAttr> &promotedOriginNames) {
  auto originRef = dyn_cast<ParamDeclRefAttr>(origin);
  if (!originRef || !promotedOriginNames.contains(originRef.getName()))
    return std::nullopt;
  return ParamDeclRefAttr::get(originRef.getName(),
                               OriginType::get(ctx, false));
}

static LogicalResult outlineAndPromoteOrigins(
    SharedState &shared, SmallVectorImpl<StructDefFieldAttr> &fieldDecls,
    SmallVectorImpl<ParamDeclAttr> &allStructParams,
    SmallVectorImpl<TypedAttr> &structParamBindings, FnOp nestedFn,
    SmallVectorImpl<Type> &deviceCaptureFieldTypes,
    llvm::MapVector<StringRef, Type> &aliases, Location closureLoc) {
  MLIRContext *ctx = shared.getContext();
  assert(allStructParams.size() == structParamBindings.size() &&
         "expected parallel struct parameters and bindings");

  DenseMap<StringAttr, bool> originMutability;
  llvm::MapVector<TypedAttr, ParamDeclRefAttr> interiorOrigins;

  for (StructDefFieldAttr fieldDecl : fieldDecls)
    checkOriginAndOutline(ctx, originMutability, interiorOrigins,
                          fieldDecl.getTypeValue());
  for (ParamDeclAttr param : allStructParams)
    checkOriginAndOutline(ctx, originMutability, interiorOrigins,
                          param.getType());

  // Append fresh parameter declarations and their parent-scope bindings.
  for (auto &[interiorAttr, paramRef] : interiorOrigins) {
    allStructParams.push_back(
        ParamDeclAttr::get(paramRef.getName(), paramRef.getType()));
    structParamBindings.push_back(interiorAttr);
  }

  // Prune origin parameters in allStructParams that were not referenced outside
  // of interior origins.
  // If it were referenced outside the interior origin it would be registered in
  // the origin mutability table so if its not there its safe to assume its
  // unused and can be dropped.
  assert(allStructParams.size() == structParamBindings.size() &&
         "expected parallel struct parameters and bindings before pruning");
  SmallVector<ParamDeclAttr> prunedParams;
  SmallVector<TypedAttr> prunedBindings;
  for (auto [param, binding] :
       llvm::zip_equal(allStructParams, structParamBindings)) {
    if (sugarIsa<OriginType>(param.getType())) {
      if (!originMutability.contains(param.getName()))
        continue;
    }
    prunedParams.push_back(param);
    prunedBindings.push_back(binding);
  }
  allStructParams = std::move(prunedParams);
  structParamBindings = std::move(prunedBindings);

  // Promote origin parameters that are only read-only into immutable origins.
  SmallPtrSet<StringAttr, 8> promotedOriginNames = collectPromotedOrigins(
      ctx, originMutability, allStructParams, structParamBindings);

  if (interiorOrigins.empty() && promotedOriginNames.empty())
    return success();

  Location rewriteLoc = closureLoc;
  bool hadConflict = false;

  // Replace all interior origin occurrences and promoted origin references in a
  // single walk.
  mlir::AttrTypeReplacer originReplacer;
  originReplacer.addReplacement([&](SymbolConstantAttr sym)
                                    -> std::pair<Attribute, WalkResult> {
    bool bindsOutlinedInterior = false;
    bool changed = false;
    DenseMap<TypedAttr, TypedAttr> paramReplacements;
    SmallVector<TypedAttr> newParamValues;
    newParamValues.reserve(sym.getParamValues().size());
    for (TypedAttr pv : sym.getParamValues()) {
      bindsOutlinedInterior |= isOutlinedInteriorOrigin(pv, interiorOrigins);
      auto newPv = cast<TypedAttr>(originReplacer.replace(pv));
      if (newPv != pv) {
        changed = true;
        paramReplacements[cast<TypedAttr>(getCanonicalAttr(pv))] = newPv;
      }
      newParamValues.push_back(newPv);
    }

    // An interior origin in the signature that the call does not bind as a
    // parameter value is derived from the callee's own parameters, so rewriting
    // it would misstate the callee's signature. Keep it as is and reject if
    // that same interior origin was outlined (interior origins are spelled
    // inline, so preserved occurrences are indistinguishable from the ones that
    // need to be rewritten). Reject this case (for now).
    if (!bindsOutlinedInterior &&
        hasOutlinedInteriorOrigin(sym.getType(), interiorOrigins)) {
      hadConflict = true;
      shared.emitError(
          rewriteLoc,
          "cannot derive an interior origin from a captured container while "
          "an interior reference to the container is also captured");
      return {sym, WalkResult::skip()};
    }

    if (!changed)
      return {sym, WalkResult::skip()};

    mlir::AttrTypeReplacer symTypeReplacer;
    symTypeReplacer.addReplacement(
        [&](TypedAttr attr) -> std::optional<TypedAttr> {
          if (!sugarIsa<OriginType>(attr.getType()))
            return std::nullopt;
          TypedAttr canon = cast<TypedAttr>(getCanonicalAttr(attr));
          auto it = paramReplacements.find(canon);
          if (it != paramReplacements.end())
            return it->second;
          return getPromotedOriginRef(ctx,
                                      OriginType::stripMutCastAndRebind(attr),
                                      promotedOriginNames);
        });
    auto newType =
        cast<FuncTypeGeneratorType>(symTypeReplacer.replace(sym.getType()));
    return {SymbolConstantAttr::get(sym.getSymbol(), newType, newParamValues),
            WalkResult::skip()};
  });
  originReplacer.addReplacement(
      [&](TypedAttr attr) -> std::optional<TypedAttr> {
        if (!sugarIsa<OriginType>(attr.getType()))
          return std::nullopt;

        TypedAttr stripped = OriginType::stripMutCastAndRebind(attr);
        if (sugarIsa<InteriorOriginAttr, OriginSubtreeAttr>(stripped)) {
          auto it =
              interiorOrigins.find(cast<TypedAttr>(getCanonicalAttr(stripped)));
          if (it == interiorOrigins.end())
            return std::nullopt;
          if (attr != stripped)
            return std::nullopt;
          return getOutlinedOriginRef(ctx, it->second, promotedOriginNames);
        }

        return getPromotedOriginRef(ctx, stripped, promotedOriginNames);
      });

  // Rewrite the body before the storage struct so a conflict is diagnosed at
  // the operation that hit it, and so a rejected closure leaves the struct
  // alone.
  if (nestedFn) {
    nestedFn.walk([&](Operation *op) {
      rewriteLoc = op->getLoc();
      originReplacer.replaceElementsIn(op, /*replaceAttrs=*/true,
                                       /*replaceLocs=*/true,
                                       /*replaceTypes=*/true);
      return hadConflict ? WalkResult::interrupt() : WalkResult::advance();
    });
    if (hadConflict)
      return failure();
    rewriteLoc = closureLoc;
  }

  for (StructDefFieldAttr &fieldDecl : fieldDecls) {
    auto newTypeValue =
        cast<TypedAttr>(originReplacer.replace(fieldDecl.getTypeValue()));
    fieldDecl = StructDefFieldAttr::get(fieldDecl.getName(), newTypeValue);
  }
  for (Type &fieldType : deviceCaptureFieldTypes)
    fieldType = cast<Type>(originReplacer.replace(fieldType));
  for (auto &[name, type] : aliases) {
    if (sugarIsa<OriginType>(type)) {
      if (promotedOriginNames.contains(StringAttr::get(ctx, name)))
        type = OriginType::get(ctx, false);
    } else {
      type = cast<Type>(originReplacer.replace(type));
    }
  }
  return failure(hadConflict);
}

static SmallVector<Type>
getConcreteStructFieldTypes(StructInstanceType structInstType,
                            ArrayRef<ParamDeclAttr> structParams,
                            ArrayRef<TypedAttr> structBindings) {
  ParameterEvaluator structEvaluator(structParams, structBindings);
  return llvm::map_to_vector(
      structInstType.getFields(), [&](StructDefFieldAttr field) -> Type {
        TypedAttr fieldType =
            structEvaluator.getReboundAttribute(field.getTypeValue());
        return ASTType(fieldType);
      });
}

static KGEN::StructType getMlirType(MLIRContext *ctx,
                                    StructInstanceType structInstType,
                                    ArrayRef<ParamDeclAttr> structParams,
                                    ArrayRef<TypedAttr> structBindings,
                                    TypeConvention convention) {
  SmallVector<Type> mlirFieldTypes =
      getConcreteStructFieldTypes(structInstType, structParams, structBindings);
  bool isMemOnly = convention == TypeConvention::MemoryOnly;
  return KGEN::StructType::get(ctx, mlirFieldTypes, isMemOnly);
}

bool ClosureEmitter::provenConformsToTrait(
    ASTType type, ASTDecl *traitDecl, SharedState &shared,
    ArrayRef<ConstraintAttr> callerAssumptions) {
  assert(traitDecl && "expected a trait declaration");
  auto trait = cast<TraitDeclOp>(traitDecl->getIfOperation());
  return type
      .doesConformTo(TraitType::get(getFullyResolvedSymbolRef(trait)), shared,
                     callerAssumptions)
      .isTrue();
}

static FailureOr<ASTType> getDeviceType(ASTType hostType, ASTDecl &scope,
                                        SharedState &shared) {
  ASTDecl *devicePassableDecl =
      shared.getBuiltinDevicePassableTrait(scope.getLoc());
  assert(devicePassableDecl && "could not find device passable dependency");

  SmallVector<ConstraintAttr> assumptions =
      ASTDecl::getAssumptionsFromScope(&scope);

  // A capture is device-encodable only if it conforms to DevicePassable;
  // resolve its device_type via GetWitness + fold attempt.
  if (!ClosureEmitter::provenConformsToTrait(hostType, devicePassableDecl,
                                             shared, assumptions))
    return failure();

  if (failed(shared.declResolver->resolveBody(*devicePassableDecl,
                                              scope.getLoc())))
    return failure();

  for (auto [_, decls] : devicePassableDecl->getDeclsInScope()) {
    for (ASTDecl *decl : decls) {
      if (failed(shared.declResolver->resolveSignature(*decl, scope.getLoc())))
        return failure();
    }
  }

  ArrayRef<ASTDecl *> aliasDecls = devicePassableDecl->lookupInCurrentScope(
      StringAttr::get(shared.getContext(), kDeviceType));
  if (aliasDecls.empty())
    return failure();

  auto aliasOp =
      dyn_cast_or_null<AliasDeclOp>(aliasDecls.front()->getIfOperation());
  if (!aliasOp || !aliasOp.getType())
    return failure();

  MLIRContext *ctx = shared.getContext();
  auto traitSymbol = TraitSymbolAttr::get(devicePassableDecl->getSymbolRef());
  TypedAttr deviceTypeWitness =
      shared.getEvaluationContext().getAndFold<GetWitnessAttr>(
          PValue(hostType), traitSymbol, StringAttr::get(ctx, kDeviceType),
          aliasOp.getType());

  if (!deviceTypeWitness || !LIT::isTypeExpr(deviceTypeWitness))
    return failure();
  return ASTType(deviceTypeWitness);
}

static TypedAttr getRefLikeOrigin(Type type) {
  if (auto refType = dyn_cast<RefType>(type))
    return refType.getOrigin();
  if (auto refPackType = dyn_cast<RefPackType>(type))
    return refPackType.getOrigin();
  return nullptr;
}

static void
addOriginReplacements(mlir::AttrTypeReplacer &originReplacer,
                      const DenseMap<TypedAttr, TypedAttr> &originMap) {
  originReplacer.addReplacement(
      [&](TypedAttr attr) -> std::optional<TypedAttr> {
        auto it = originMap.find(attr);
        if (it == originMap.end())
          return std::nullopt;
        return it->second;
      });
}

ASTDecl *ClosureEmitter::liftClosureIntoMethod(
    ASTDecl &nestedFnDecl, ASTDecl &storageStructDecl,
    PromotedClosureSelfArg selfArg,
    ArrayRef<StructDefFieldAttr> concreteFieldDecls,
    ArrayRef<Value> concreteFieldCaptures,
    ArrayRef<CaptureConvention> captureConventions,
    ArrayRef<Type> selfBoundFieldTypes, Location location) {
  MLIRContext *ctx = shared.getContext();
  // Nest under the storage struct as a method. Captured parameters already
  // live on the storage struct, so do not prepend them to the method.
  ASTDecl *promotedDecl = promoteClosure(
      nestedFnDecl, ArrayRef<ParamDeclAttr>{}, /*selfArg=*/selfArg,
      /*capturingOverride=*/true, /*targetParent=*/&storageStructDecl);
  FnOp promotedCallFunction = cast<FnOp>(promotedDecl->getIfOperation());
  assert(concreteFieldDecls.size() == concreteFieldCaptures.size() &&
         "expected one capture value per closure field");
  assert(concreteFieldDecls.size() == captureConventions.size() &&
         "expected one capture convention per closure field");

  // Wire the captures in the promoted function body to the struct fields
  // accessed via the self argument.
  Block &callBody = promotedCallFunction.getBodyRegion().front();
  Value selfArgValue = callBody.getArgument(0);
  OpBuilder bodyBuilder = OpBuilder::atBlockBegin(&callBody);
  Location promotedBodyLoc = DebugInfo::extractSourceLoc(location);
  if (DebugInfo::DISubprogramAttr promotedScope =
          promotedCallFunction.getSubprogramScope())
    promotedBodyLoc = FusedLoc::get(ctx, promotedBodyLoc, promotedScope);

  DenseMap<Value, Value> captureReplacements;
  captureReplacements.reserve(concreteFieldCaptures.size());
  DenseMap<TypedAttr, TypedAttr> originMap;
  bool hadOriginConflict = false;
  for (auto [index, captureAndConvention] :
       llvm::enumerate(llvm::zip(concreteFieldCaptures, captureConventions))) {
    auto [capture, convention] = captureAndConvention;
    StringAttr fieldName = concreteFieldDecls[index].getName();
    auto selfRefType = cast<RefType>(selfArgValue.getType());
    Type fieldElementType = selfBoundFieldTypes[index];
    Type resultRefType = RefStructGEROp::getReboundFieldType(
        selfRefType, fieldName, fieldElementType);
    Value extractedRef =
        RefStructGEROp::create(bodyBuilder, promotedBodyLoc, resultRefType,
                               fieldName, selfArgValue)
            ->getResults()
            .front();

    Value replacement = extractedRef;
    if (!sugarIsa<RefType>(capture.getType()) ||
        isByReferenceCapture(convention))
      replacement =
          RefLoadOp::create(bodyBuilder, promotedBodyLoc, extractedRef);
    captureReplacements[capture] = replacement;

    TypedAttr oldOrigin = getRefLikeOrigin(capture.getType());
    TypedAttr newOrigin = getRefLikeOrigin(replacement.getType());
    if (oldOrigin && newOrigin && (newOrigin != oldOrigin)) {
      auto [it, inserted] = originMap.try_emplace(oldOrigin, newOrigin);
      if (!inserted && it->second != newOrigin)
        hadOriginConflict = true;
    }
  }
  if (hadOriginConflict)
    llvm::report_fatal_error(
        "conflicting capture origins while lifting closure into method");
  mlir::AttrTypeReplacer originReplacer;
  addOriginReplacements(originReplacer, originMap);
  promotedCallFunction.getBodyRegion().walk([&](Operation *op) {
    for (OpOperand &operand : op->getOpOperands()) {
      auto it = captureReplacements.find(operand.get());
      if (it == captureReplacements.end())
        continue;
      Value replacement = it->second;
      Type expectedType = operand.get().getType();
      if (expectedType != replacement.getType() &&
          isEqualCanon(expectedType, replacement.getType())) {
        OpBuilder rebindBuilder(op);
        replacement = RebindOp::create(rebindBuilder, op->getLoc(),
                                       expectedType, replacement);
      }
      operand.set(replacement);
    }
    if (!originMap.empty())
      originReplacer.recursivelyReplaceElementsIn(op,
                                                  /*replaceAttrs=*/true,
                                                  /*replaceLocs=*/true,
                                                  /*replaceTypes=*/true);
  });

  return promotedDecl;
}

ClosureEmitter::Closure ClosureEmitter::liftClosure(
    ASTDecl &moduleDecl, SMLoc smLoc,
    SmallVector<ClosureParent> &closureParents, SymbolRefAttr parentSymbolRef,
    llvm::MapVector<StringRef, Type> const &aliases,
    SmallVector<StructDefFieldAttr> &&concreteFieldDecls,
    SmallVector<Value> &&concreteFieldCaptures,
    SmallVector<CaptureConvention> &&concreteFieldCaptureConventions,
    SmallVector<ParamDeclAttr> &&concreteParams,
    SmallVector<TypedAttr> &&concreteStructBindings, StringAttr name,
    TypeConvention convention, SmallVector<Type> &&deviceCaptureFieldTypes,
    bool capturesEncodable, ASTDecl &nestedFnDecl) {
  Location location = shared.translateLocation(smLoc);
  MLIRContext *ctx = shared.getContext();

  SmallVector<Type> promotedDeviceCaptureFieldTypes =
      std::move(deviceCaptureFieldTypes);

  SmallVector<TypedAttr> selfRefParamValues = llvm::map_to_vector(
      concreteParams, [](ParamDeclAttr declAttr) -> TypedAttr {
        return ParamDeclRefAttr::get(declAttr);
      });
  SmallVector<StringAttr> paramNames = llvm::map_to_vector(
      concreteParams,
      [](ParamDeclAttr declAttr) -> StringAttr { return declAttr.getName(); });
  StructInstanceType structInstType = StructInstanceType::get(
      StringAttr::get(ctx, Twine(getFlattenedSymbolName(parentSymbolRef))
                               .concat("::")
                               .concat(name.getValue())),
      paramNames, selfRefParamValues, concreteFieldDecls,
      BoolAttr::get(ctx, convention == TypeConvention::MemoryOnly));
  KGEN::StructType kgenStructType = getMlirType(
      ctx, structInstType, concreteParams, concreteStructBindings, convention);
  SmallVector<Type> selfBoundFieldTypes = getConcreteStructFieldTypes(
      structInstType, concreteParams, selfRefParamValues);

  // Create a StructType to serve as the self. The __call__ method will become a
  // method on the struct
  StringAttr structName =
      StringAttr::get(ctx, Twine(getFlattenedSymbolName(parentSymbolRef))
                               .concat("::")
                               .concat(name.getValue())
                               .concat("::__storage"));
  auto [structDecl, structOp] = createStruct(
      shared, moduleDecl, structName, concreteParams, smLoc,
      SmallVector<PassingKind>(concreteParams.size(), PassingKind::Inferred));
  structOp.setConvention(convention);
  structOp.setDefinesClosure(true);
  TraitType traitType = getTraitType(closureParents, moduleDecl);
  structOp.setCanonicalTrait(traitType);
  OpBuilder structBuilder(structOp.getRegion());
  structBuilder.setInsertionPointToStart(&structOp.getFields().front());
  for (auto [index, fieldDecl] : llvm::enumerate(concreteFieldDecls))
    addFieldOpAndDecl(fieldDecl.getName(), selfBoundFieldTypes[index], structOp,
                      structDecl, structBuilder, *shared.declResolver);
  LIT::StructType closureStructType =
      structOp.bindReference(selfRefParamValues);
  assert(isa<FnOp>(nestedFnDecl.getIfOperation()) &&
         "expected nested closure declaration to be a function");
  PromotedClosureSelfArg selfArg{closureStructType, ArgConvention::ImmMem};
  ASTDecl *promotedCallDecl = liftClosureIntoMethod(
      nestedFnDecl, structDecl, selfArg, concreteFieldDecls,
      concreteFieldCaptures, concreteFieldCaptureConventions,
      selfBoundFieldTypes, location);
  FnOp promotedCallFunction = cast<FnOp>(promotedCallDecl->getIfOperation());
  StructEmitter structEmitter(structDecl);
  DenseMap<ClosureMethod, FnOp> methodImpls;
  methodImpls[ClosureMethod::CALL] = promotedCallFunction;
  auto synthesizeValueMethodBody = [&](bool isMove) -> FnOp {
    FnOp fn = structEmitter.synthesizeEmptyMoveOrCopyInit(/*isMove=*/isMove);
    ASTDecl *decl = shared.declResolver->getDeclForFuncSymbol(
        getFullyResolvedSymbolRef(fn));
    assert(decl && "synthesized value method must be registered");
    (void)structEmitter.populateMoveCopy(*decl, isMove);
    return fn;
  };
  for (const ClosureParent &closureParent : closureParents) {
    switch (closureParent.getClosureMethod()) {
    case ClosureMethod::DEL:
      methodImpls[ClosureMethod::DEL] = structEmitter.synthesizeEmptyDtor();
      break;
    case ClosureMethod::MOVE:
      methodImpls[ClosureMethod::MOVE] =
          synthesizeValueMethodBody(/*isMove=*/true);
      break;
    case ClosureMethod::COPY:
      methodImpls[ClosureMethod::COPY] =
          synthesizeValueMethodBody(/*isMove=*/false);
      break;
    default:
      break;
    }
  }

  ImplicitLocOpBuilder builder(location, ctx);

  // Synthesize the storage struct's initializer.
  auto initName = StringAttr::get(ctx, "__init__");
  SmallVector<Type> initArgumentTypes;
  SmallVector<StringAttr> argNames;
  SmallVector<PassingKind> argPassingKinds;
  SmallVector<ArgConvention> argConventions;

  // Each captured value becomes a positional-only constructor argument.
  assert(concreteFieldDecls.size() == selfBoundFieldTypes.size() &&
         "expected one bound field type per closure field");
  assert(concreteFieldDecls.size() == concreteFieldCaptureConventions.size() &&
         "expected one capture convention per closure field");
  size_t argCount = concreteFieldDecls.size() + 1;
  initArgumentTypes.reserve(argCount);
  argNames.reserve(argCount);
  argPassingKinds.reserve(argCount);
  argConventions.reserve(argCount);
  for (auto [index, fieldDecl] : llvm::enumerate(concreteFieldDecls)) {
    StringAttr fieldName = fieldDecl.getName();
    Type fieldType = selfBoundFieldTypes[index];
    CaptureConvention captureConvention =
        concreteFieldCaptureConventions[index];

    // `__init__`'s byref-result is always `self`; a capture of the same
    // spelling would alias both origins. Keep the field name; rename the arg.
    StringAttr initArgName = fieldName;
    if (fieldName.getValue() == "self")
      initArgName = StringAttr::get(ctx, "__capture_self");

    Type argType;
    ArgConvention argConvention;
    switch (captureConvention) {
    case CaptureConvention::kConventionRead:
    case CaptureConvention::kConventionMut:
    case CaptureConvention::kConventionUnspecified:
    case CaptureConvention::kConventionRef:
      if (sugarIsa<RefType>(fieldType)) {
        argType = fieldType;
        argConvention = ArgConvention::Ref;
        break;
      }
      [[fallthrough]];
    case CaptureConvention::kConventionTrivialCopy:
    case CaptureConvention::kConventionCopy:
      argType = ASTType(fieldType).getRefForArgument(initArgName.getValue(),
                                                     /*isMut=*/false);
      argConvention = ArgConvention::ImmMem;
      break;
    case CaptureConvention::kConventionMove: {
      TypeConvention passability = ASTType(fieldType).getRegisterPassability(
          nestedFnDecl.getLoc(), shared);
      if (passability == TypeConvention::RegisterPassableTrivial) {
        argType = fieldType;
        argConvention = ArgConvention::OwnedReg;
      } else {
        argType = ASTType(fieldType).getRefForArgument(initArgName.getValue(),
                                                       /*isMut=*/true);
        argConvention = ArgConvention::OwnedMem;
      }
      break;
    }
    }

    initArgumentTypes.push_back(argType);
    argConventions.push_back(argConvention);
    argNames.push_back(initArgName);
    argPassingKinds.push_back(PassingKind::PosOnly);
  }

  // The trailing implicit `self` argument is the result slot the constructor
  // initializes.
  ASTType selfType = structDecl.getTypeDeclSelf();
  initArgumentTypes.push_back(
      selfType.getRefForArgument("self", /*isMut=*/true));
  argConventions.push_back(ArgConvention::ByRefResult);
  argNames.push_back(StringAttr::get(ctx, "self"));
  argPassingKinds.push_back(PassingKind::Implicit);

  builder.setInsertionPointToEnd(&structOp.getFields().front());
  auto [initFnOp, initDecl] = synthesizeFunction(
      structDecl, initName, {}, PogListAttr::get(ctx), initArgumentTypes,
      argConventions, PogListAttr::get(ctx, argNames, argPassingKinds),
      NoneType::get(ctx), SpecialFunctionKind::kInit, smLoc, builder,
      /*fnEffects=*/{}, /*suffix=*/"", /*synthetic=*/true,
      InlineLevel::Automatic);

  // Generate the constructor body.
  if (initFnOp) {
    Block *body = initFnOp.getBody();
    ImplicitLocOpBuilder bodyBuilder =
        ImplicitLocOpBuilder::atBlockEnd(initFnOp.getLoc(), body);
    bodyBuilder.setInsertionPointToStart(body);
    IREmitter emitter(*initDecl, bodyBuilder);

    DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
    if (shared.diBuilder)
      diScopeGuard = shared.diBuilder->pushScopeGuard(initFnOp.getLocScope());

    // The trailing implicit argument is the `self` result slot to initialize
    Value selfValue = body->getArgument(body->getNumArguments() - 1);
    SmallVector<StructFieldOp> fieldOps =
        llvm::to_vector(structOp.getFieldDecls());
    assert(fieldOps.size() == concreteFieldCaptureConventions.size() &&
           "expected one struct field per capture");

    for (auto [index, fieldOp] : llvm::enumerate(fieldOps)) {
      Value arg = body->getArgument(index);
      Value fieldRef = RefStructGEROp::create(bodyBuilder, selfValue, fieldOp)
                           ->getResults()
                           .front();
      Type fieldType = selfBoundFieldTypes[index];

      // Reference captures
      if (isByReferenceCapture(concreteFieldCaptureConventions[index]) &&
          sugarIsa<RefType>(fieldType)) {
        RefStoreOp::create(bodyBuilder, arg, fieldRef);
        continue;
      }

      // Value captures
      CValue argValue;
      switch (argConventions[index]) {
      case ArgConvention::ImmReg:
        argValue = SRValue(arg);
        break;
      case ArgConvention::ImmMem:
        argValue = MBValue(arg);
        break;
      case ArgConvention::OwnedMem:
        argValue = MRValue(arg);
        break;
      case ArgConvention::OwnedReg:
        argValue = SRValue(arg);
        break;
      default:
        llvm_unreachable("unexpected argument convention for a value capture");
      }
      SyntheticNode node(structDecl.getLoc());
      emitter.emitStoreToLValue({argValue, &node}, MLValue(fieldRef),
                                EC_AttributeRefBase);
    }

    emitter.emitNormalReturn(initFnOp.getLoc(), /*returnVal=*/Value());
  }

  auto callParentIt = llvm::find_if(closureParents, [](const ClosureParent &p) {
    return p.getClosureMethod() == ClosureMethod::CALL;
  });
  assert(callParentIt != closureParents.end() &&
         "closure parents must include the call trait");
  const ClosureParent &callParent = *callParentIt;
  FnOp callWitness = emitStorageCallWitness(
      structDecl, structOp, promotedCallFunction, smLoc,
      [&](ASTDecl &decl, bool synthetic, StringAttr name) {
        // Storage has no `impl` param — do not redirect Self witnesses to it.
        return pushBackTraitFunctionImpl(
            callParent.getSignature(), decl, synthetic, name,
            SpecialFunctionKind::kNormal, callParent.getInlineLevel(),
            /*redirectWitnessToImplParam=*/false);
      });
  if (!callWitness)
    return {};
  methodImpls[ClosureMethod::CALL] = callWitness;

  StringAttr callName = StringAttr::get(ctx, "__call__");

  ASTDecl *callWitnessDecl = shared.declResolver->getDeclForFuncSymbol(
      getFullyResolvedSymbolRef(callWitness));
  assert(callWitnessDecl && "call witness must be registered");
  shared.declResolver->attachDeclToParentNameTable(callWitnessDecl, callName);
  callWitness.setSourceNameAttr(callName);

  // Give storage a pretty closure-signature name.
  {
    TraitDeclOp callTrait = callParent.getTrait(shared);
    if (auto keyOr = callTrait.getClosureSignature()) {
      std::string prettyName = formatClosureSignature(*keyOr, shared);
      structOp.setSourceNameAttr(
          DebugInfo::SourceNameAttr::get(StringAttr::get(ctx, prettyName)));
    }
  }

  // Emit the conformance ops into the storage struct by finding the closure
  // method and FnOp associated with each parent trait.
  auto addWitnessTable = [&](const ClosureParent &closureParent) {
    TraitDeclOp traitParent = closureParent.getTrait(shared);
    builder.setInsertionPointToEnd(&structOp.getFields().front());
    TraitSymbolArrayAttr immediateParents =
        traitParent.getImmediateParentsAttr();
    StringAttr parentName = closureParent.getFlattenedName();
    ConformanceOp witnessTable = ConformanceOp::create(
        builder, closureParent.getSymbol(), immediateParents);
    Block &block = witnessTable.getBody().emplaceBlock();

    ASTDecl &conformDecl = shared.declResolver->addDecl(
        witnessTable, structDecl.getLoc(), parentName, &structDecl, {}, {}, -1);
    conformDecl.resolvedness = DeclResolvedness::signature;

    // Marker traits like AnyType have no methods -- empty ConformanceOp is
    // sufficient for TypeConformsToTraitAttr::simplify().
    if (closureParent.isEmpty())
      return;

    builder.setInsertionPointToStart(&block);
    ClosureMethod method = closureParent.getClosureMethod();
    auto it = methodImpls.find(method);
    assert(it != methodImpls.end() &&
           "non-marker closure method missing an implementation");

    TypedAttr symbol = buildSymbol(it->second, structOp.getInputParams());
    WitnessOp::create(builder, closureParent.getWitnessName(),
                      /*sym_visibility=*/nullptr, symbol);

    // add the alias entries if this is not a parametric trait: Non-parametric
    // trait always has a `_Self` parameter.
    if (closureParent.getClosureMethod() == ClosureMethod::CALL &&
        traitParent.getInputParams().size() == 1) {
      SmallVector<AliasDeclOp> traitAliases;
      for (AliasDeclOp alias : traitParent.getFields().getOps<AliasDeclOp>())
        traitAliases.push_back(alias);
      assert(traitAliases.size() == aliases.size() &&
             "trait capture aliases must mirror closure captures");
      SmallVector<TypedAttr> captureBindings = getCaptureBindings(structOp);
      assert(traitAliases.size() <= captureBindings.size() &&
             "storage must publish a binding per CALL-trait capture alias");
      for (auto [alias, witnessValue] : llvm::zip_equal(
               traitAliases,
               ArrayRef(captureBindings).take_front(traitAliases.size())))
        WitnessOp::create(builder, alias.getParamDecl().getName(),
                          /*sym_visibility=*/nullptr, witnessValue);
    }
  };

  for (const ClosureParent &closureParent : closureParents)
    addWitnessTable(closureParent);

  bool isTrivial = convention == TypeConvention::RegisterPassableTrivial;
  generateIsTrivialSpecialAlias("__del__is_trivial", isTrivial, shared,
                                structDecl, getDeinitableParent());
  generateIsTrivialSpecialAlias("__move_ctor_is_trivial", isTrivial, shared,
                                structDecl, getMoveParent());
  if (methodImpls.contains(ClosureMethod::COPY))
    generateIsTrivialSpecialAlias("__copy_ctor_is_trivial", isTrivial, shared,
                                  structDecl, getCopyParent());
  LIT::StructType boundClosureStructType =
      structOp.bindReference(concreteStructBindings);
  TypedAttr typeParamAttr =
      TypeParamAttr::get(boundClosureStructType, kgenStructType, traitType);
  if (capturesEncodable) {
    unsigned numStorageFields = std::distance(structOp.getFieldDecls().begin(),
                                              structOp.getFieldDecls().end());
    assert(promotedDeviceCaptureFieldTypes.size() == numStorageFields &&
           "device field types must match storage struct fields");
    addStorageConformanceToDevicePassable(
        structDecl, promotedDeviceCaptureFieldTypes, name.getValue());
  }
  return Closure{&structDecl, promotedCallDecl, typeParamAttr};
}

static unsigned conventionRank(TypeConvention convention) {
  if (convention == TypeConvention::Unspecified)
    return 0;
  return static_cast<unsigned>(convention);
}

static TypeConvention meetCaptureConvention(TypeConvention lhs,
                                            TypeConvention rhs) {
  return conventionRank(lhs) <= conventionRank(rhs) ? lhs : rhs;
}

static TypeConvention typeConventionOf(SharedState &shared,
                                       LIT::StructType structType) {
  ASTDecl &structDecl =
      shared.declResolver->getDeclForTypeSymbol(structType.getSymbol());
  StructDeclOp structDeclOp = cast<StructDeclOp>(structDecl.getIfOperation());
  return structDeclOp.isRegisterPassableTrivial()
             ? TypeConvention::RegisterPassableTrivial
         : structDeclOp.isRegisterPassable() ? TypeConvention::RegisterPassable
                                             : TypeConvention::MemoryOnly;
}

static TypeConvention typeConventionOf(SharedState &shared, ParamType paramType,
                                       const Capture &capture,
                                       ASTDecl &nestedFnDecl) {
  // The captured value's type may have been refined in the capturing scope,
  // wrapping the parameter reference in a `DowncastAttr` that carries the
  // additional trait bounds (see the by-copy refinement in addCaptureValue).
  // Strip it to recover the underlying parameter reference.
  auto paramRef =
      dyn_cast<ParamDeclRefAttr>(DowncastAttr::strip(paramType.getParam()));
  if (!paramRef) {
    shared.emitError(nestedFnDecl.getLoc(),
                     "cannot capture " + capture.getSpelling() +
                         " because its type is not a parameter reference.");
    return TypeConvention::Unspecified;
  }

  if (!sugarIsa<TraitType>(paramRef.getType())) {
    shared.emitError(nestedFnDecl.getLoc(),
                     "cannot capture " + capture.getSpelling() +
                         " because its type constraint is not a trait.");
    return TypeConvention::Unspecified;
  }

  return ASTType(paramType).getRegisterPassability(nestedFnDecl.getLoc(),
                                                   shared);
}

Value ClosureEmitter::emitClosure(ASTDecl &moduleDecl, ASTDecl &nestedFnDecl,
                                  ArrayRef<Capture> captures,
                                  ASTDecl &traitDecl, Location location,
                                  bool isCopyable,
                                  FnTypeGeneratorType closureSig,
                                  ArrayRef<ParamDeclRefAttr> paramCaptures) {
  TraitDeclOp trait = cast<TraitDeclOp>(traitDecl.getIfOperation());

  // (1) Lift the nested function into a storage struct and instantiate it.
  FnOp nestedFn = cast<FnOp>(nestedFnDecl.getIfOperation());
  FnOp parent = nestedFn->getParentOfType<FnOp>();
  assert(parent && "expected the function to be a nested function");
  Block *closureInsertBlock = nestedFn->getBlock();
  Operation *closureInsertBefore = nestedFn->getNextNode();
  ImplicitLocOpBuilder builder(location, shared.getContext());
  builder.setInsertionPoint(nestedFn);
  MLIRContext *ctx = builder.getContext();
  StringAttr fnName = nestedFn.getSourceNameAttr();

  SmallVector<Value> captureValues;
  SmallVector<CValue> constructorArgs;
  SmallVector<CaptureConvention> captureConventions;

  TraitType anyType =
      shared.lookupBuiltinTraitType("AnyType", nestedFnDecl.getLoc());
  IREmitter emitter(*nestedFnDecl.getParentDecl(), builder);

  TypeConvention highestCaptureConvention =
      TypeConvention::RegisterPassableTrivial;
  SmallVector<StructDefFieldAttr> fieldDecls;
  SmallVector<ParamDeclAttr> allStructParams;
  SmallVector<TypedAttr> structParamBindings;
  SmallVector<Type> deviceCaptureFieldTypes;

  SmallPtrSet<StringAttr, 8> byValueCapturedOriginParamNames;
  auto updateCaptureConvention = [&](TypeConvention captureConventionMet) {
    highestCaptureConvention =
        meetCaptureConvention(highestCaptureConvention, captureConventionMet);
  };
  bool allCapturesEncodable = true;
  for (const Capture &capture : captures) {
    Value value = capture.getValue().getMlirValue();
    captureValues.push_back(value);
    if (capture.getCaptureConvention() == CaptureConvention::kConventionMove &&
        sugarIsa<RefType>(value.getType()) &&
        sugarCast<RefType>(value.getType()).isMutableKnown(true))
      constructorArgs.push_back(MRValue(value));
    else
      constructorArgs.push_back(capture.getValue());
    captureConventions.push_back(capture.getCaptureConvention());

    SyntheticNode synthNode(nestedFnDecl.getLoc());
    ExprDest dest(anyType, EC_Type);
    PValue captureTypeValue =
        emitter
            .emitImplicitConversionToType({value.getType(), &synthNode},
                                          anyType, dest)
            .getIfPValue();
    auto captureTypeAttr = cast<TypedAttr>(captureTypeValue.get());
    auto captureName = StringAttr::get(ctx, capture.getSpelling());
    auto captureConvention = capture.getCaptureConvention();
    Type mlirType = value.getType();
    if (auto refType = sugarDynCast<LIT::RefType>(mlirType))
      mlirType = refType.getElementType();
    switch (captureConvention) {
    case CaptureConvention::kConventionUnspecified:
    case CaptureConvention::kConventionMut:
    case CaptureConvention::kConventionRead:
    case CaptureConvention::kConventionRef: {
      // Mutability casts should have been emitted during parse time.
      // TODO: Pointers are register passable, so this demotion
      // should become unnecessary once downstream passes are fixed.
      TypeConvention captureConventionMet =
          (sugarIsa<LIT::RefType>(value.getType())
               ? ASTType(
                     sugarCast<LIT::RefType>(value.getType()).getElementType())
               : ASTType(value.getType()))
              .getRegisterPassability(nestedFnDecl.getLoc(), shared);
      updateCaptureConvention(captureConventionMet);
      break;
    }
    case CaptureConvention::kConventionTrivialCopy:
      break;
    case CaptureConvention::kConventionCopy:
    case CaptureConvention::kConventionMove: {
      if (auto refType = sugarDynCast<LIT::RefType>(value.getType())) {
        if (auto captureOriginParam = dyn_cast<ParamDeclRefAttr>(
                OriginType::stripMutCastAndRebind(refType.getOrigin())))
          byValueCapturedOriginParamNames.insert(captureOriginParam.getName());
      }
      // Copy/move captures materialize storage for the captured value itself,
      // not for a reference wrapper. Use the pointee as the field type.
      if (sugarIsa<LIT::RefType>(value.getType()))
        captureTypeAttr = TypeParamAttr::get(mlirType, anyType);

      if (auto structType = sugarDynCast<StructType>(mlirType)) {
        updateCaptureConvention(typeConventionOf(shared, structType));
      } else if (sugarIsa<TraitType>(mlirType)) {
        shared.emitError(nestedFnDecl.getLoc(),
                         "cannot capture a value of trait type yet because "
                         "existentials are not implemented.");
        return {};
      } else if (auto paramType = sugarDynCast<ParamType>(mlirType)) {
        updateCaptureConvention(
            typeConventionOf(shared, paramType, capture, nestedFnDecl));
      }
      break;
    }
    }
    fieldDecls.push_back(StructDefFieldAttr::get(captureName, captureTypeAttr));
    if (allCapturesEncodable) {
      // A by-reference capture stores a host pointer (LIT::RefType) as its
      // storage field, while its device field type is computed from the
      // pointee. The two disagree in `encode_fields` (ref != pointee device
      // type), so a reference is not device-encodable.
      if (isByReferenceCapture(captureConvention)) {
        allCapturesEncodable = false;
      } else {
        FailureOr<Type> deviceFieldType = getReboundCaptureDeviceFieldType(
            ASTType(mlirType), nestedFnDecl, shared);
        if (failed(deviceFieldType))
          allCapturesEncodable = false;
        else
          deviceCaptureFieldTypes.push_back(*deviceFieldType);
      }
    }
  }
  // TODO(MOCO-4045): DevicePassable conformance currently requires a
  // register-passable storage struct.
  if (allCapturesEncodable &&
      highestCaptureConvention == TypeConvention::MemoryOnly)
    allCapturesEncodable = false;

  SmallVector<ClosureParent> closureParents;
  if (trait.getInputParams().size() > 1) {
    TraitSymbolAttr boundSymbol =
        emitter.bindParamsToClosureTraitFromSig(closureSig);
    ParameterEvaluator evaluator =
        *populateTraitBindingEvaluator(boundSymbol, shared);
    auto callAlias = cast<AliasDeclOp>(
        traitDecl.lookupInCurrentScope("__call__").front()->getIfOperation());
    assert(callAlias.getDeclName() == "__call__");
    closureParents.emplace_back(
        boundSymbol,
        cast<FnTypeGeneratorType>(evaluator.replace(callAlias.getType())),
        callAlias.getDeclName(), ClosureMethod::CALL);
  } else {
    closureParents.emplace_back(shared, trait.bindReference({}), "__call__",
                                ClosureMethod::CALL);
  }
  closureParents.append(
      {getMoveParent(), getDeinitableParent(), getAnyParent()});

  if (isCopyable) {
    closureParents.push_back(getCopyParent());
    closureParents.push_back(getImplicitlyCopyableParent());
  }
  if (highestCaptureConvention == TypeConvention::RegisterPassableTrivial) {
    closureParents.push_back(getTrivialRegisterTypeParent());
    closureParents.push_back(getRegisterPassableParent());
  } else if (highestCaptureConvention == TypeConvention::RegisterPassable)
    closureParents.push_back(getRegisterPassableParent());

  FnTypeGeneratorType original = nestedFn.getFuncTypeGenerator();
  // TODO: Remove capturing when legacy closures are removed
  FnTypeGeneratorType closureBodySignature = FnTypeGeneratorType::get(
      original.getInputParamTypes(), original.getValues(),
      original.getArgConventions(), original.getFnEffects().setCapturing(true),
      original.getFnMetaOriginData(), original.getParamListAttrs(),
      original.getArgListAttrs());
  auto [capturedRefs, _] =
      DeclResolver::createSelfContainedSignature(closureBodySignature);
  llvm::MapVector<StringRef, Type> aliases;
  for (ParamDeclRefAttr reference : capturedRefs) {
    auto [_, inserted] =
        aliases.insert({reference.getName().getValue(), reference.getType()});
    (void)inserted;
  }

  SmallPtrSet<StringAttr, 8> seen;
  for (ParamDeclAttr param : allStructParams)
    seen.insert(param.getName());
  for (auto capturedParam : paramCaptures) {
    if (byValueCapturedOriginParamNames.contains(capturedParam.getName()))
      continue;
    if (!seen.insert(capturedParam.getName()).second)
      continue;
    allStructParams.push_back(
        ParamDeclAttr::get(capturedParam.getName(), capturedParam.getType()));
    structParamBindings.push_back(capturedParam);
  }

  // (1) Outline interior origins and promote origin parameters in a unified
  // pass.
  if (!allCapturesEncodable)
    deviceCaptureFieldTypes.clear();
  if (failed(outlineAndPromoteOrigins(
          shared, fieldDecls, allStructParams, structParamBindings, nestedFn,
          deviceCaptureFieldTypes, aliases, location)))
    return {};

  // Storage bindings passed to initializer call.
  SmallVector<TypedAttr> storageParamBindings = structParamBindings;

  Closure liftedClosure = liftClosure(
      moduleDecl, nestedFnDecl.getLoc(), closureParents,
      SymbolRefAttr::get(
          ctx,
          getFlattenedSymbolName(getFullyResolvedSymbolRefUpTo<FileModuleOp>(
              cast<mlir::SymbolOpInterface>(parent.getOperation())))),
      aliases, std::move(fieldDecls), std::move(captureValues),
      std::move(captureConventions), std::move(allStructParams),
      std::move(structParamBindings), fnName, highestCaptureConvention,
      std::move(deviceCaptureFieldTypes), allCapturesEncodable, nestedFnDecl);
  if (!liftedClosure.structDecl)
    return {};

  // The nested closure function is moved into the storage struct as a method.
  // Emit closure materialization ops back in the original parent function body.
  if (closureInsertBefore &&
      closureInsertBefore->getBlock() == closureInsertBlock)
    builder.setInsertionPoint(closureInsertBefore);
  else
    builder.setInsertionPointToEnd(closureInsertBlock);

  // Instantiate the storage struct directly through its synthesized `__init__`,
  // passing the captured values as the positional arguments.
  StructDeclOp storageStructOp =
      cast<StructDeclOp>(liftedClosure.structDecl->getIfOperation());
  VarDeclOp storageVar = emitInitializerCall(
      *nestedFnDecl.getParentDecl(), builder, location, storageStructOp,
      storageParamBindings, /*args=*/constructorArgs, fnName.getValue());

  return MLValue(storageVar);
}

static CValue ASTDeclToCValue(ASTDecl *decl, OpBuilder &builder, Location loc) {
  if (!decl)
    return {};
  if (auto cv = decl->getIfIRValue()) {
    return cv;
  } else if (auto var = dyn_cast_or_null<VarDeclOp>(decl->getIfOperation())) {
    if (!sugarIsa<RefType>(var.getType()))
      return SRValue(var);
    Value value = var;
    if (var.getKind() == VarDeclKind::Ref)
      value = RefLoadOp::create(builder, loc, var);
    return CValue::getMValueForRef(value);
  }
  return {};
}

ASTDecl *ClosureEmitter::addCaptureValue(SharedState &shared, ASTDecl &closure,
                                         StringRef name, SMLoc location) {
  CaptureConvention capture = shared.defaultCaptureConventionInScope(closure);
  FnOp funcOp = cast<FnOp>(closure.getIfOperation());
  IREmitter emitter(*closure.getParentDecl(), OpBuilder(funcOp));
  return ClosureEmitter::addCaptureValue(closure, location, name, capture,
                                         emitter);
}

// Lookup the decl in the named decls that have been collected thus far. This
// may be an incomplete list because we have not finished resolving the scope.
static FailureOr<ASTDecl *> partialLookup(StringAttr name, ASTDecl &scope,
                                          llvm::SMLoc loc) {
  for (auto [declName, list] : scope.getDeclsInScope()) {
    if (name == declName) {
      if (list.size() != 1) {
        scope.getShared().emitError(loc, "ambiguous captured value: ") << name;
        return failure();
      }
      return list.front();
    }
  }
  return nullptr;
}

// Search the scope and its parents for a decl with the name without resolving
// anything. If `upperBound` is set, the walk includes that decl and then stops.
static FailureOr<ASTDecl *> findCapture(SharedState &shared, StringRef name,
                                        llvm::SMLoc loc, ASTDecl &scope,
                                        ASTDecl *upperBound = nullptr) {
  auto nameAttr = StringAttr::get(shared.getContext(), name);
  ASTDecl *current = &scope;
  do {
    FailureOr<ASTDecl *> result = partialLookup(nameAttr, *current, loc);
    if (failed(result))
      return failure();
    if (result.value()) {
      // Error already diagnosed.
      if (result.value()->isErroneous())
        return failure();
      return result.value();
    }
    if (current == upperBound)
      break;
  } while ((current = current->getParentDecl()));
  return nullptr;
}

ASTDecl *ClosureEmitter::addCaptureValue(ASTDecl &closure, SMLoc location,
                                         StringRef name,
                                         CaptureConvention parsedConvention,
                                         IREmitter &emitter,
                                         ASTDecl *signatureDecl) {
  // Check if already emitted.
  SharedState &shared = emitter.shared;
  if (shared.captureInstanceExistsInScope(closure, name)) {
    auto nameAttr = StringAttr::get(shared.getContext(), name);
    ArrayRef<ASTDecl *> existing = closure.lookupInCurrentScope(nameAttr);
    assert(existing.size() == 1 &&
           "if the capture instance exists in the scope then it should have "
           "been registered in the scope");
    return existing.front();
  }
  FnOp funcOp = cast<FnOp>(closure.getIfOperation());
  ASTDecl *fnParentDecl = closure.getParentDecl()->getNearestDeclOfType<FnOp>();
  auto parentFn = cast<FnOp>(fnParentDecl->getIfOperation());
  ASTDecl *result = nullptr;
  if (usesClosurePipeline(parentFn)) {
    auto localMaybe = findCapture(shared, name, location,
                                  *closure.getParentDecl(), fnParentDecl);
    if (failed(localMaybe))
      return nullptr;
    result = localMaybe.value();
    if (!result) {
      result = addCaptureValue(shared, *fnParentDecl, name, location);
      if (!result)
        return nullptr;
    }
  } else {
    auto hitMaybe = partialLookup(StringAttr::get(shared.getContext(), name),
                                  closure, location);
    if (failed(hitMaybe))
      return nullptr;
    // No need to emit a capture instance since this closure defines the
    // value.
    if (hitMaybe.value())
      return hitMaybe.value();

    // otherwise, this is a capture. Find the def.
    auto maybeResult =
        findCapture(shared, name, location, *closure.getParentDecl());
    if (failed(maybeResult))
      return nullptr;
    result = maybeResult.value();
    if (!result) {
      shared.emitError(location, "reference to an unknown value: ") << name;
      return nullptr;
    }
    if (auto pval = result->getIfIRValue().getIfPValue()) {
      shared.emitError(location, "value ")
          << name << " is a parameter and does not need a capture convention";
      return nullptr;
    }
  }

  // Capture materialization is inserted into the enclosing function. Stamp
  // those ops with that function's debug scope, using the inner closure's
  // file/line only. Keeping the inner subprogram on the loc would lower as
  // an inlined location and fail LLVM's dbg-value verifier.
  emitter.builder->setInsertionPoint(closure.getIfOperation());
  DebugInfo::DIBuilder::ScopeGuard diGuard;
  Location captureLoc = funcOp.getLoc();
  if (auto fileLoc = captureLoc->findInstanceOf<FileLineColLoc>())
    captureLoc = fileLoc;
  if (shared.diBuilder) {
    diGuard = shared.diBuilder->pushScopeGuard(parentFn.getLocScope());
    captureLoc = shared.diBuilder->createScopedLoc(captureLoc);
  }

  CValue valueInParent = ASTDeclToCValue(result, *emitter.builder, captureLoc);
  if (!valueInParent) {
    shared.emitError(location, "'")
        << name << "' does not name a capturable value";
    return nullptr;
  }

  CaptureConvention convention;
  /// The captureValue is a map of the valueInParent. For example, the
  /// valueInParent may be an immutable borrowed value. If this value is
  /// captured by copy the capturedValue in the body of the closure is a
  /// mutable owned value. Since the captured value does not exist until
  /// later, we have to create a temporary value to represent the change in
  /// the properties of the value in the body of the closure.
  CValue captureValue;

  auto captureByRef = [&](CValue value,
                          std::optional<bool> mutability) -> CValue {
    // Ensure we are not capturing an immutable reference by mutable
    // reference.
    if (auto refType = sugarDynCast<RefType>(value.getType().mlirType)) {
      // If the mutability is not specified or the reference type match the
      // specified mutability, return the original value.
      OriginType originType = refType.getOriginType();
      if (!mutability.has_value() || originType.isMutableKnown(*mutability))
        return value;

      if (originType.isMutableKnown(false)) {
        // mutable capture of an immutable reference, error.
        shared.emitError(location, "Cannot capture ")
            << name << " by mut because it could be immutable";
        return {};
      }

      if (originType.isMutableKnown(true)) {
        // convert a mut ref to immut ref
        auto refImmutOp = LIT::RefImmutOp::create(*emitter.builder, captureLoc,
                                                  valueInParent.getMlirValue());
        return MBValue(refImmutOp->getResult(0));
      }
    }

    // Not a reference capture, then it must be a read effect.
    if (mutability.has_value() && *mutability == false)
      return value;

    shared.emitError(location, "register passible value '")
        << name << "' can not be captured by "
        << (mutability.has_value() ? "'mut'" : "'ref'")
        << ". Do you mean 'imm'?";
    return {};
  };

  // Apply scope-based type refinement before by-value capture checks. A
  // parameter may gain extra `conforms_to` constraints in the capturing scope;
  // without rebinding to that refined type, move/copy validation would still
  // see the original generic bound.
  if (parsedConvention == CaptureConvention::kConventionMove ||
      parsedConvention == CaptureConvention::kConventionCopy) {
    SyntheticNode refineNode(location);
    valueInParent =
        maybeEmitRefinementRebind({valueInParent, &refineNode}, emitter);
  }

  switch (parsedConvention) {
  case CaptureConvention::kConventionMove: {
    Type type = valueInParent.getType().mlirType;
    if (auto ref = sugarDynCast<RefType>(valueInParent.getType().mlirType))
      type = ref.getElementType();
    if (!ASTType(type).isMovable(closure.getLoc(), shared, *fnParentDecl)) {
      shared.emitError(location, "Cannot capture ")
          << name << " by move because the type is not movable";
      return nullptr;
    }
    if (valueInParent.getIfBValue()) {
      shared.emitError(location, "Cannot capture")
          << name << " by move because the value is read only";
      return nullptr;
    }
    // If it was captured by move then there was a transfer operation.
    convention = parsedConvention;
    if (sugarIsa<RefType>(valueInParent.getType().mlirType))
      captureValue = CValue::getMValueForRef(valueInParent.getMlirValue());
    else
      captureValue = MRValue(valueInParent.getMlirValue());
    break;
  }
  case CaptureConvention::kConventionCopy: {
    ASTType originalType = valueInParent.getRValueType();
    if (originalType.isTrivial(closure.getLoc(), shared)) {
      // Remap to trivial copy convention to avoid storing symbols.
      convention = CaptureConvention::kConventionTrivialCopy;
      // if we are capturing by mutable copy and its trivial do not capture
      // the reference.
      if (sugarIsa<RefType>(valueInParent.getType())) {
        SyntheticNode node(result->getLoc());
        ExprDest dest(EC_Capture);
        captureValue = emitter.emitRValue(
            {CValue::getMValueForRef(valueInParent.getMlirValue()), &node},
            dest);
      } else {
        captureValue = valueInParent;
      }
    } else {
      convention = parsedConvention;
      if (auto refType =
              sugarDynCast<RefType>(valueInParent.getType().mlirType)) {
        OriginType originType = refType.getOriginType();
        if (originType.isMutableKnown(false)) {
          auto refImmutOp = LIT::RefImmutOp::create(
              *emitter.builder, captureLoc, valueInParent.getMlirValue());
          captureValue = MBValue(refImmutOp->getResult(0));
        }
      }
      ExprDest dest(EC_Capture);
      SyntheticNode node(result->getLoc());
      ASTExprAnd<CValue> valueInParentExpr{valueInParent, &node};
      LValue copiedOrMovedValue =
          dest.getLValueForResult(valueInParentExpr.expr->getLoc(),
                                  valueInParentExpr.ir.getRValueType(),
                                  /*allowIncompatibleTypes=*/false,
                                  /*requireMLValue=*/false, emitter);
      emitter.emitStoreToLValue(valueInParentExpr, copiedOrMovedValue,
                                dest.getContext());
      // Diagnose an uncopyable capture.
      if (!originalType.isCopyable(closure.getLoc(), shared,
                                   /*isImplicit=*/false, *fnParentDecl)) {
        shared.emitError(location, "cannot capture ")
            << name << " by copy because it is not copyable.";
        return nullptr;
      }
      captureValue = copiedOrMovedValue;
    }
    break;
  }
  case CaptureConvention::kConventionMut:
  case CaptureConvention::kConventionRead:
  case CaptureConvention::kConventionRef: {
    convention = parsedConvention;
    auto mutability = [convention]() -> std::optional<bool> {
      if (convention == CaptureConvention::kConventionRef)
        return std::nullopt;
      return convention == CaptureConvention::kConventionMut;
    }();
    captureValue = captureByRef(valueInParent, mutability);
    if (!captureValue)
      return nullptr;
    break;
  }
  case CaptureConvention::kConventionTrivialCopy:
    llvm_unreachable("trivial copy is derived from by-copy, not parsed");
  case CaptureConvention::kConventionUnspecified:
    shared.emitError(
        location, "Could not infer capture convention of the captured value ")
        << name;
    return nullptr;
  }
  assert(captureValue && "must set capture value");
  // Ensure the capture value we created is used when parsing the body of the
  // closure.
  ASTDecl &captureValueDecl = shared.getDeclResolver().addFullyResolvedDecl(
      captureValue, name, closure.getLoc(),
      signatureDecl ? signatureDecl : &closure);
  shared.addCaptureToScope(closure, result,
                           Capture(captureValue, convention, name));
  return &captureValueDecl;
}

/// If an alias is already bound, verify the new
/// value is consistent with the existing binding. Returns false if
/// inconsistent.
static bool tryRecordSubstitution(AliasSubstitutions &substitutions,
                                  StringAttr aliasName, TypedAttr newValue) {
  if (!newValue)
    return false;
  auto it = substitutions.find(aliasName);
  if (it != substitutions.end()) {
    Type existingType = it->second.getType();
    Type newType = newValue.getType();
    if (auto existingParam = dyn_cast<TypeParamAttr>(it->second))
      existingType = existingParam.getTypeValue();
    if (auto newParam = dyn_cast<TypeParamAttr>(newValue))
      newType = newParam.getTypeValue();
    return isEqualCanon(existingType, newType);
  }
  substitutions[aliasName] = newValue;
  return true;
}

namespace M::KGEN::LIT {

struct AuxiliaryParameters {
  size_t startingIndex;
  size_t numStructAuxiliaryParams;
  SmallVector<ParamDeclAttr> traitAuxiliaryParameters;
  SmallVector<TypedAttr> structAliases;
  SmallVector<StringAttr> traitAliases;

  TypedAttr getAliasRef(size_t index) {
    return get<TypedAttr>(index, structAliases);
  }

private:
  template <typename Result>
  Result get(size_t index, SmallVector<Result> &container) {
    if (index < startingIndex)
      return Result{};
    size_t auxIdx = index - startingIndex;
    if (auxIdx >= container.size())
      return Result{};
    return container[auxIdx];
  }
};

} // namespace M::KGEN::LIT

namespace {

static TypedAttr getUnderlyingParamRef(TypedAttr attr) {
  if (auto upcast = dyn_cast<UpcastAttr>(attr))
    return getUnderlyingParamRef(upcast.getInputTypeValue());

  if (auto typeParam = dyn_cast<TypeParamAttr>(attr)) {
    if (auto paramType = dyn_cast<ParamType>(typeParam.getTypeValue()))
      return getUnderlyingParamRef(paramType.getParam());
  }

  if (isa<ParamDeclRefAttr, ParamIndexRefAttr>(attr))
    return attr;

  return {};
}

// Given a type parameter that wraps a strong type than its type, convert it to
// an upcast, which explicitly communicates the relationship between the
// underlying parameter and the expected type.
static TypedAttr makeExplicitUpcastBinding(TypedAttr binding) {
  if (auto typeParam = dyn_cast<TypeParamAttr>(binding)) {
    if (auto paramType = dyn_cast<ParamType>(typeParam.getTypeValue())) {
      if (auto paramRef = dyn_cast<ParamDeclRefAttr>(paramType.getParam())) {
        if (!isEqualCanon(paramRef.getType(), typeParam.getType()))
          return UpcastAttr::get(typeParam.getType(), paramRef);
      }
    }
  }
  return binding;
}
struct ConformanceTableEntryMapper {
  struct Result {
    // the conformance table entry
    TypedAttr binding;
    // The name of the parameter that violates no escaping parameter rule. Used
    // for error messages.
    StringAttr escapedParamName;
  };

  // Maps bindings inferred against the actual struct method into the trait
  // conformance table's auxiliary parameter space.
  //
  // For example, suppose we have:
  //
  //   struct X:
  //     alias A: Coord
  //     def __call__[_A: Coord](self, x: Cartesian, z: _A):
  //         pass
  //
  // and:
  //
  //   trait Y:
  //     alias T: Coord
  //     alias R: Coord
  //     def __call__[_T: Coord, _R: Coord](self, y: _T, z: _R):
  //         ...
  //
  // Specialization inference produces bindings in the actual method's
  // parameter space, here {Cartesian, _A}. To populate the conformance table,
  // those bindings must be rewritten into the trait/struct alias space,
  // yielding {Cartesian, Self.A} for {T, R}.
  ConformanceTableEntryMapper(FnOp actualFn, AuxiliaryParameters &ctx) {
    StructDeclOp structDeclOp = actualFn->getParentOfType<StructDeclOp>();
    for (ParamDeclAttr structParam : structDeclOp.getInputParams())
      allowedConformanceScopeParams.insert(structParam.getName());

    FnTypeGeneratorType actualSig = actualFn.getFuncTypeGenerator();
    ArrayRef<ParamDeclAttr> actualParams = actualFn.getInputParams().drop_back(
        actualSig.getNumImplicitOriginDecls());
    assert(actualParams.size() >= ctx.numStructAuxiliaryParams &&
           "struct auxiliary params should be present in function signature");
    ArrayRef<ParamDeclAttr> actualAuxiliaryParams =
        actualParams.take_front(ctx.numStructAuxiliaryParams);
    for (auto [offset, auxiliaryParam] :
         llvm::enumerate(actualAuxiliaryParams)) {
      TypedAttr aliasValue = ctx.getAliasRef(ctx.startingIndex + offset);
      if (aliasValue)
        auxiliaryBindingsByName[auxiliaryParam.getName()] = aliasValue;
    }

    walker.addReplacement([&](ParamDeclRefAttr paramRef) -> TypedAttr {
      if (allowedConformanceScopeParams.contains(paramRef.getName()))
        return paramRef;

      auto it = auxiliaryBindingsByName.find(paramRef.getName());
      if (it == auxiliaryBindingsByName.end()) {
        pendingEscapedParamName = paramRef.getName();
        return paramRef;
      }
      return it->second;
    });
  }

  Result map(TypedAttr binding) {
    pendingEscapedParamName = {};
    TypedAttr mappedBinding = cast<TypedAttr>(walker.replace(binding));
    return {mappedBinding, pendingEscapedParamName};
  }
  // The only parameter references allowed to remain in a conformance table
  // entry are parameters from the enclosing closure struct itself (storage
  // capture params / fn-pointer `Impl`).
  bool isAllowedConformanceScopeRef(TypedAttr attr) const {
    TypedAttr paramRef = getUnderlyingParamRef(attr);
    auto declRef = dyn_cast_if_present<ParamDeclRefAttr>(paramRef);
    return declRef && allowedConformanceScopeParams.contains(declRef.getName());
  }

private:
  DenseMap<StringAttr, TypedAttr> auxiliaryBindingsByName;
  DenseSet<StringAttr> allowedConformanceScopeParams;
  mlir::AttrTypeReplacer walker;
  StringAttr pendingEscapedParamName;
};

static bool canFunctionSignatureMatchTraitParamInf(FnOp actualFn,
                                                   FnTypeGeneratorType target,
                                                   AuxiliaryParameters &ctx,
                                                   SharedState &shared,
                                                   AdapteeParts &adapteeParts) {
  FnTypeGeneratorType actualSig = actualFn.getFuncTypeGenerator();
  if (!actualSig.hasMemoryOnlyResult() && target.hasMemoryOnlyResult())
    adapteeParts.needsResultConversion = true;
  else if (actualSig.hasMemoryOnlyResult() != target.hasMemoryOnlyResult())
    return false;
  if (actualSig.getFnEffects() != target.getFnEffects())
    return false;

  ArrayRef<Type> actualExplicitParams =
      actualSig.getInputParamTypes().drop_front(ctx.numStructAuxiliaryParams);
  ArrayRef<Type> targetExplicitParams = target.getInputParamTypes().drop_front(
      ctx.traitAuxiliaryParameters.size());
  SMLoc loc = shared.getTopLevelDecl().getLoc();
  SyntheticNode syntheticExpr(loc);

  SpecializeInf inference(shared.getTopLevelDecl(), &syntheticExpr,
                          target.getInputParamTypes(),
                          target.getParamListAttrs(), loc,
                          /*discardError=*/true);
  if (actualExplicitParams.size() != targetExplicitParams.size())
    return false;
  ParamRefRemapper remapper(actualFn.getInputParams());
  size_t actualAuxCount = ctx.numStructAuxiliaryParams;
  size_t targetAuxCount = ctx.traitAuxiliaryParameters.size();
  for (auto [index, actualParamType, targetParam] :
       llvm::enumerate(actualExplicitParams, targetExplicitParams)) {
    Type actualParam = remapper.replace(actualParamType);
    if (!isEqualCanon(actualParam, targetParam))
      return false;

    StringAttr actualParamName =
        actualFn.getInputParams()[index + actualAuxCount].getName();
    inference.setInitialInferredValue(
        index + targetAuxCount,
        ParamDeclRefAttr::get(actualParamName, actualParam));
  }
  // Bind leading aux (`_A`) to storage aliases before matching args so the
  // trait-shaped `__call__` compares like the residual promoted layout.
  FnTypeGeneratorType actualSigForMatch = actualSig;
  if (actualAuxCount != 0) {
    mlir::AttrTypeReplacer auxToAlias;
    ArrayRef<ParamDeclAttr> actualParams = actualFn.getInputParams().drop_back(
        actualSig.getNumImplicitOriginDecls());
    for (auto [offset, auxParam] :
         llvm::enumerate(actualParams.take_front(actualAuxCount))) {
      TypedAttr aliasValue = ctx.getAliasRef(ctx.startingIndex + offset);
      if (!aliasValue)
        return false;
      StringAttr auxName = auxParam.getName();
      auxToAlias.addReplacement([=](ParamDeclRefAttr ref) -> TypedAttr {
        if (ref.getName() == auxName)
          return aliasValue;
        return ref;
      });
    }
    actualSigForMatch =
        cast<FnTypeGeneratorType>(auxToAlias.replace(actualSig));
  }
  FailureOr<SmallVector<TypedAttr>> specialization =
      inference.inferSpecialization(target, actualSigForMatch,
                                    actualFn.getInputParams());
  if (failed(specialization))
    return false;

  // Walk each target trait aux specialization. The inference produces these
  // bindings in the *wrapper's __call__* parameter space (e.g. _a + _b);
  // we need them in the struct space to call from the adaptee.
  ConformanceTableEntryMapper createConformanceTableEntry(actualFn, ctx);
  unsigned targetAuxStart = ctx.startingIndex;
  for (auto [offset, aliasAndParam] : llvm::enumerate(
           llvm::zip(ctx.traitAliases, ctx.traitAuxiliaryParameters))) {
    auto [aliasName, auxiliaryParameter] = aliasAndParam;
    TypedAttr rawBinding = (*specialization)[targetAuxStart + offset];
    if (!rawBinding || isa<UnboundAttr>(rawBinding))
      return false;

    rawBinding = makeExplicitUpcastBinding(rawBinding);
    auto mappedBinding = createConformanceTableEntry.map(rawBinding);
    if (mappedBinding.escapedParamName) {
      auto &error = inference.getMojoDiag(loc);
      error << "closure conformance alias '" << aliasName
            << "' cannot reference parameter "
            << mappedBinding.escapedParamName;

      inference.diag.release();
      return false;
    }

    if (!tryRecordSubstitution(adapteeParts.aliasSubstitutions, aliasName,
                               mappedBinding.binding))
      return false;

    // The adaptor's block-argument types reference target trait aux params.
    // Rewrite them into struct-level expressions so that, after symbol
    // binding (which substitutes the wrapper's __call__ aux with the same
    // struct-level expressions), the operand types match the callee's
    // expected types.
    adapteeParts.adapteeTypeMap[StringAttr::get(
        auxiliaryParameter.getContext(), getTraitMethodParamName(offset))] =
        mappedBinding.binding;
  }

  // Leading adaptee aux bind to storage params/aliases.
  adapteeParts.fnLevelBindings.reserve(ctx.numStructAuxiliaryParams +
                                       targetExplicitParams.size());
  for (size_t offset = 0; offset < ctx.numStructAuxiliaryParams; ++offset) {
    TypedAttr aliasValue = ctx.getAliasRef(ctx.startingIndex + offset);
    if (!aliasValue)
      return false;
    adapteeParts.fnLevelBindings.push_back(aliasValue);
  }
  for (auto [index, explicitParamType] : llvm::enumerate(targetExplicitParams))
    adapteeParts.fnLevelBindings.push_back(ParamDeclRefAttr::get(
        getTraitMethodParamName(index + targetAuxCount), explicitParamType));

  return true;
}

static SmallVector<AliasDeclOp> collectClosureAliases(TraitDeclOp trait) {
  SmallVector<AliasDeclOp> aliases;
  for (AliasDeclOp alias : trait.getFields().getOps<AliasDeclOp>())
    aliases.push_back(alias);
  return aliases;
}

/// Compute associated types of targetTrait in terms of sourceTrait
static LogicalResult
inferClosureTraitExtension(SharedState &shared, TraitDeclOp sourceTrait,
                           TraitDeclOp targetTrait, TypedAttr anchor,
                           AdapteeParts &parts, ASTDecl &declScope) {
  FnOp sourceCall = getFnOpNamed(sourceTrait, "__call__");
  FnOp targetCall = getFnOpNamed(targetTrait, "__call__");
  if (!sourceCall || !targetCall)
    return failure();

  SmallVector<AliasDeclOp> sourceAliases = collectClosureAliases(sourceTrait);
  SmallVector<AliasDeclOp> targetAliases = collectClosureAliases(targetTrait);
  size_t sourceAuxCount = sourceAliases.size();
  size_t targetAuxCount = targetAliases.size();
  ArrayRef<ParamDeclAttr> sourceParams = sourceCall.getInputParams();
  ArrayRef<ParamDeclAttr> targetParams = targetCall.getInputParams();
  // A closure trait's associated types are emitted as the leading auxiliary
  // parameters of its `__call__`
  assert(sourceParams.size() >= sourceAuxCount &&
         targetParams.size() >= targetAuxCount &&
         "closure __call__ must carry an auxiliary param per associated type");

  FnTypeGeneratorType sourceSignature = sourceCall.getFuncTypeGenerator();
  FnTypeGeneratorType targetSignature = specializeSignature(
      targetCall.getFullSignature(),
      ASTDecl::computeSelfTypeForTrait(sourceTrait), shared.getDeclResolver());

  if (!sourceSignature.hasMemoryOnlyResult() &&
      targetSignature.hasMemoryOnlyResult())
    parts.needsResultConversion = true;
  else if (sourceSignature.hasMemoryOnlyResult() !=
           targetSignature.hasMemoryOnlyResult())
    return failure();
  if (sourceSignature.getFnEffects() != targetSignature.getFnEffects())
    return failure();

  ArrayRef<Type> sourceExplicitParams =
      sourceSignature.getInputParamTypes().drop_front(sourceAuxCount);
  ArrayRef<Type> targetExplicitParams =
      targetSignature.getInputParamTypes().drop_front(targetAuxCount);
  if (sourceExplicitParams.size() != targetExplicitParams.size())
    return failure();

  SMLoc loc = declScope.getLoc();
  SyntheticNode syntheticExpr(loc);
  SpecializeInf inference(declScope, &syntheticExpr,
                          targetSignature.getInputParamTypes(),
                          targetSignature.getParamListAttrs(), loc,
                          /*discardError=*/true);
  ParamRefRemapper remapper(sourceParams);
  for (auto [index, sourceParamType, targetParamType] :
       llvm::enumerate(sourceExplicitParams, targetExplicitParams)) {
    Type sourceParam = remapper.replace(sourceParamType);
    if (!isEqualCanon(sourceParam, targetParamType))
      return failure();
    inference.setInitialInferredValue(
        index + targetAuxCount,
        ParamDeclRefAttr::get(sourceParams[index + sourceAuxCount].getName(),
                              sourceParam));
  }
  FailureOr<SmallVector<TypedAttr>> specialization =
      inference.inferSpecialization(targetSignature, sourceCall);
  if (failed(specialization))
    return failure();

  auto sourceTraitSymbol = TraitSymbolAttr::get(getFullyResolvedSymbolRef(
      cast<mlir::SymbolOpInterface>(sourceTrait.getOperation())));

  // An adaptor calls the anchor's own `__call__`
  DenseMap<StringAttr, TypedAttr> sourceAuxiliaryWitnesses;
  for (auto [alias, auxiliaryParam] :
       llvm::zip(sourceAliases, sourceParams.take_front(sourceAuxCount))) {
    TypedAttr witness =
        GetWitnessAttr::get(anchor, sourceTraitSymbol,
                            alias.getParamDecl().getName(), alias.getType());
    sourceAuxiliaryWitnesses[auxiliaryParam.getName()] = witness;
    parts.fnLevelBindings.push_back(witness);
  }
  for (auto [index, targetParamType] : llvm::enumerate(targetExplicitParams))
    parts.fnLevelBindings.push_back(ParamDeclRefAttr::get(
        getTraitMethodParamName(index + targetAuxCount), targetParamType));

  // Remap references to the source aliases.
  auto anchorRef =
      dyn_cast_if_present<ParamDeclRefAttr>(getUnderlyingParamRef(anchor));
  auto rewriteExtensionBinding =
      [&](TypedAttr binding) -> FailureOr<TypedAttr> {
    bool escapes = false;
    mlir::AttrTypeReplacer walker;
    walker.addReplacement([&](ParamDeclRefAttr paramRef) -> TypedAttr {
      if (anchorRef && paramRef.getName() == anchorRef.getName())
        return paramRef;
      auto it = sourceAuxiliaryWitnesses.find(paramRef.getName());
      if (it == sourceAuxiliaryWitnesses.end()) {
        escapes = true;
        return paramRef;
      }
      return it->second;
    });
    TypedAttr mapped =
        cast<TypedAttr>(walker.replace(makeExplicitUpcastBinding(binding)));
    if (escapes)
      return failure();
    return mapped;
  };

  // Map the target trait alias to the source binding.
  for (auto [offset, alias] : llvm::enumerate(targetAliases)) {
    TypedAttr binding = (*specialization)[offset];
    if (!binding || isa<UnboundAttr>(binding))
      return failure();
    FailureOr<TypedAttr> rewritten = rewriteExtensionBinding(binding);
    if (failed(rewritten))
      return failure();
    binding = *rewritten;
    if (!tryRecordSubstitution(parts.aliasSubstitutions,
                               alias.getParamDecl().getName(), binding))
      return failure();
    parts.adapteeTypeMap[StringAttr::get(
        shared.getContext(), getTraitMethodParamName(offset))] = binding;
  }
  return success();
}

} // namespace

void ClosureEmitter::buildCallAdaptorAndAddWitness(
    StructDeclOp structDeclOp, ASTDecl &structDecl, TraitDeclOp traitDeclOp,
    FnOp traitCallFn, TypedAttr callee, const AdapteeParts &adapteeParts,
    ASTType selfTypeOverride) {
  SharedState &shared = structDecl.getShared();
  MLIRContext *ctx = shared.getContext();
  ArrayRef<ParamDeclAttr> structParams = structDeclOp.getInputParams();
  bool redirectWitnessToImplParam =
      !structParams.empty() && structParams.front().getName() == "impl";

  SymbolRefAttr traitSymbol = getFullyResolvedSymbolRef(
      cast<mlir::SymbolOpInterface>(traitDeclOp.getOperation()));
  StringAttr adaptorNameAttr =
      StringAttr::get(ctx, "__call__$" + getFlattenedSymbolName(traitSymbol));
  auto [adaptorFnOp, _, adaptorResult] = pushBackTraitFunctionImpl(
      traitCallFn.getFullSignature(), structDecl, true, adaptorNameAttr,
      traitCallFn.getSpecialFunctionKind(), traitCallFn.getInlineLevel(),
      redirectWitnessToImplParam, selfTypeOverride);
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](ParamDeclRefAttr ref) -> TypedAttr {
    auto ptr = adapteeParts.adapteeTypeMap.find(ref.getName());
    if (ptr == adapteeParts.adapteeTypeMap.end())
      return ref;
    return ptr->second;
  });
  // Populate the adaptor body: rebind arguments and call original
  ImplicitLocOpBuilder b(adaptorFnOp.getLoc(), adaptorFnOp);
  b.setInsertionPointToEnd(&adaptorFnOp.getBodyRegion().front());
  SmallVector<Value> callOperands;
  SmallVector<TypedAttr> origins;
  Block &adaptorBlock = adaptorFnOp.getBodyRegion().front();
  SmallVector<Type> expectedTypes;
  expectedTypes.reserve(adaptorBlock.getNumArguments());
  for (BlockArgument arg : adaptorBlock.getArguments())
    expectedTypes.push_back(replacer.replace(arg.getType()));
  auto calleeSigGen = cast<FnTypeGeneratorType>(callee.getType());
  auto calleeConventions = calleeSigGen.getArgConventions();

  for (auto [arg, targetType, conv] : llvm::zip(
           adaptorBlock.getArguments(), expectedTypes, calleeConventions)) {
    Value operand = arg;
    if (targetType != arg.getType())
      operand = RebindOp::create(b, targetType, operand);

    // Handle convention mismatches between the adaptor (trait signature) and
    // the callee (storage/wrapper `__call__`). Generic trait parameters use
    // ImmMem (ref), but concrete RegisterPassable types use ImmReg (value).
    if (!hasImplicitOrigin(conv) && isa<RefType>(operand.getType()))
      operand = RefLoadOp::create(b, operand);

    callOperands.push_back(operand);
    if (hasImplicitOrigin(conv))
      origins.push_back(cast<RefType>(operand.getType()).getOrigin());
  }
  auto callOp = LIT::CallOp::create(b, calleeSigGen.getResultType(), callee,
                                    origins, callOperands);
  Value result = callOp.getResult(0);

  if (adapteeParts.needsResultConversion) {
    // The callee returns in-register but the adaptor expects a memory-only
    // result. Store the register value into the ByRefResult slot.
    Value resultSlot = adaptorBlock.getArguments().back();
    Type concreteSlotType = replacer.replace(resultSlot.getType());
    if (concreteSlotType != resultSlot.getType())
      resultSlot = RebindOp::create(b, concreteSlotType, resultSlot);
    RefStoreOp::create(b, result, resultSlot);
    IREmitter::emitNormalReturn(b);
  } else {
    Type resultType = calleeSigGen.getResultType();
    if (resultType != adaptorResult)
      result = RebindOp::create(b, adaptorResult, result);
    IREmitter::emitNormalReturn(b, result);
  }

  // Build the witness using the adaptor function
  SymbolConstantAttr adaptorSymbol = buildSymbol(adaptorFnOp, structParams);
  SmallVector<std::pair<StringRef, TypedAttr>> witnesses;
  witnesses.emplace_back(traitCallFn.getSymNameAttr(), adaptorSymbol);
  for (auto &[aliasName, aliasValue] : adapteeParts.aliasSubstitutions)
    witnesses.emplace_back(aliasName.getValue(), aliasValue);

  addConformanceTable(
      structDecl,
      ClosureEmitter::ClosureParent(shared, traitDeclOp.bindReference({}),
                                    "__call__", ClosureMethod::CALL),
      witnesses);
}

LogicalResult ClosureEmitter::checkStructCompatibility(ASTType structType,
                                                       ASTDecl *traitDecl,
                                                       bool rebind) {
  // Ensure that we have a valid closure trait and a struct metatype.
  TraitDeclOp traitDeclOp =
      llvm::dyn_cast_if_present<TraitDeclOp>(traitDecl->getIfOperation());
  if (!traitDeclOp)
    return failure();
  if (!traitDeclOp.getDefinesClosure())
    return failure();
  ASTDecl &structDecl = *structType.getDecl(shared);
  if (!structDecl.getIfOperation())
    return failure();

  StructDeclOp structDeclOp =
      dyn_cast<StructDeclOp>(structDecl.getIfOperation());
  if (!structDeclOp)
    return failure();

  // does the struct already conform to the trait?
  auto target = TraitSymbolAttr::get(getFullyResolvedSymbolRef(
      cast<mlir::SymbolOpInterface>(traitDeclOp.getOperation())));
  for (TraitSymbolAttr currentTrait :
       structDeclOp.getCanonicalTrait().getSymbols()) {
    if (target == currentTrait) {
      return success();
    }
  }
  // Require explicit trait conformance for user-written structs. Only
  // compiler-synthesized structs (the closure wrappers) may implicitly conform
  // and fall through.
  if (!structDeclOp.isSynthetic())
    return failure();

  // This trait defines a closure which means it has a single call function.
  if (structDecl.resolvedness < DeclResolvedness::body) {
    if (failed(
            shared.declResolver->resolveBody(structDecl, structDecl.getLoc())))
      return failure();
  }
  StringRef name = "__call__";
  auto callDecls = structDecl.lookupInCurrentScope(name);
  // Resolve signatures for all call declarations before creating the
  // OverloadSet, which requires DeclResolvedness::signature.
  for (ASTDecl *callDecl : callDecls) {
    if (failed(shared.declResolver->resolveSignature(*callDecl,
                                                     structDecl.getLoc())))
      return failure();
  }
  FnOp callFunction = getFnOpNamed(traitDeclOp, "__call__");
  // get the call function in terms of the struct wrapper
  SyntheticNode syntheticNode(structDecl.getLoc());
  ASTType structSelfType = structDecl.getTypeDeclSelf();
  IREmitter emitter(structDecl, EC_Trait);
  FnTypeGeneratorType traitSignature =
      specializeSignature(callFunction.getFullSignature(),
                          structSelfType.mlirType, *shared.declResolver);

  auto bindings = ParamBindings::getForDeclaredType(
      emitter.getDeclScope(), structSelfType, &syntheticNode);
  // This could be a parametric function, we don't need to bind the parameter on
  // the function to test the compatibility.
  bindings.relaxBindingKindTo(ParamBindings::kWithEllipsis);
  OverloadSet ov(name, callDecls, std::move(bindings),
                 CallSyntax::kMethodCallSynthetic);
  /// Perform rebind on method that implements the trait function but with
  /// different argument names.
  auto newWitnessResult =
      ov.filterOverloadSetForValueType(traitSignature, nullptr);
  PValue newWitness =
      newWitnessResult.isYes() ? newWitnessResult.getYes().callee : PValue();
  if (newWitness) {
    SmallVector<StringRef> traitAliasNames;
    for (AliasDeclOp traitAlias :
         traitDeclOp.getFields().getOps<AliasDeclOp>()) {
      traitAliasNames.push_back(traitAlias.getParamDecl().getName().getValue());
    }
    SmallVector<TypedAttr> captureBindings = getCaptureBindings(structDeclOp);
    bool aliasesOk = traitAliasNames.size() <= captureBindings.size();
    if (aliasesOk) {
      if (rebind) {
        SmallVector<std::pair<StringRef, TypedAttr>> witnesses;
        witnesses.emplace_back(callFunction.getSymNameAttr(), newWitness.get());
        for (auto [aliasName, aliasValue] : llvm::zip_equal(
                 traitAliasNames,
                 ArrayRef(captureBindings).take_front(traitAliasNames.size())))
          witnesses.emplace_back(aliasName, aliasValue);
        addConformanceTable(
            structDecl,
            ClosureEmitter::ClosureParent(shared, traitDeclOp.bindReference({}),
                                          "__call__", ClosureMethod::CALL),
            witnesses);
      }
      return success();
    }
    // Fall through to param-inf when capture arity does not line up.
  }

  // Exact Matching Failed. Check if we can conform to a trait by declaring
  // alias members. This requires conformance checked substitution.
  if (callDecls.empty())
    return failure();

  // Collect closure-specific alias names. Inherited AliasDeclOps (e.g.
  // `__del__is_trivial`) are cloned into the trait's fields by lazy body
  // resolution and are marked with `inheritedFrom`; skip them.
  SmallVector<StringAttr> traitAliasOps;
  for (AliasDeclOp aliasOp : traitDeclOp.getFields().getOps<AliasDeclOp>())
    traitAliasOps.push_back(aliasOp.getParamDecl().getName());

  size_t traitAliasCount = traitAliasOps.size();
  SmallVector<ParamDeclAttr> auxiliaryParams;
  for (ParamDeclAttr auxiliaryParam :
       callFunction.getInputParams().take_front(traitAliasCount))
    auxiliaryParams.push_back(auxiliaryParam);
  // Storage publishes captured types as inferred struct params.
  size_t targetPayloadParams =
      explicitParamCount(callFunction) - traitAliasCount;

  for (ASTDecl *callDecl : callDecls) {
    auto structCallFn = dyn_cast_or_null<FnOp>(callDecl->getIfOperation());
    if (!structCallFn)
      continue;
    if (failed(shared.declResolver->resolveSignature(*callDecl,
                                                     structDecl.getLoc())))
      continue;

    size_t actualExplicit = explicitParamCount(structCallFn);
    if (actualExplicit < targetPayloadParams)
      continue;
    size_t inferredPrefix = leadingInferredParamCount(structCallFn);
    size_t actualAuxCount = actualExplicit - targetPayloadParams;
    if (actualAuxCount > inferredPrefix)
      continue;

    SmallVector<TypedAttr> captureBindings = getCaptureBindings(structDeclOp);
    if (actualAuxCount > captureBindings.size())
      continue;
    SmallVector<TypedAttr> actualAuxBindings(
        captureBindings.begin(), captureBindings.begin() + actualAuxCount);

    AuxiliaryParameters auxCtx{/*startingIndex=*/0, actualAuxCount,
                               auxiliaryParams, actualAuxBindings,
                               traitAliasOps};
    AdapteeParts adapteeParts;
    if (canFunctionSignatureMatchTraitParamInf(structCallFn, traitSignature,
                                               auxCtx, shared, adapteeParts)) {
      if (rebind)
        buildCallAdaptorAndAddWitness(
            structDeclOp, structDecl, traitDeclOp, callFunction,
            buildSymbolWithBindings(structCallFn, structDeclOp.getInputParams(),
                                    adapteeParts.fnLevelBindings),
            adapteeParts);

      return success();
    }
  }

  return failure();
}

LogicalResult
ClosureEmitter::augmentWitnessTablesToConformTo(ASTType structType,
                                                ASTDecl *traitDecl) {
  return checkStructCompatibility(structType, traitDecl, true);
}

LogicalResult ClosureEmitter::isCompatibleWith(ASTType structType,
                                               ASTDecl *traitDecl) {
  return checkStructCompatibility(structType, traitDecl, false);
}

LogicalResult ClosureEmitter::isTraitCompatibleWith(ASTType sourceTraitType,
                                                    TraitDeclOp targetTrait,
                                                    ASTDecl *declScope) {
  if (!targetTrait || !targetTrait.getDefinesClosure())
    return failure();

  // FIXME: it does not handle a trait composition with multiple
  // closure traits correctly.
  std::optional<TraitDeclOp> sourceOp =
      getClosureDecl(shared, sourceTraitType.mlirType);
  if (!sourceOp || !sourceOp->getDefinesClosure())
    return failure();

  // Compatibility is a property of the two signatures, not of the type value
  // that will eventually anchor the extension, so probe with the source's
  // `Self`.
  ASTDecl &scope = declScope ? *declScope : shared.getTopLevelDecl();
  AdapteeParts extension;
  return inferClosureTraitExtension(
      shared, *sourceOp, targetTrait,
      cast<TypedAttr>(
          PValue(ASTDecl::computeSelfTypeForTrait(*sourceOp)).get()),
      extension, scope);
}

ASTDecl *ClosureEmitter::createExtensionStruct(ASTDecl &moduleDecl,
                                               TraitDeclOp sourceTrait,
                                               TraitDeclOp targetTrait,
                                               ASTType sourceMetaType,
                                               SMLoc location) {
  MLIRContext *ctx = shared.getContext();

  FnOp sourceCall = getFnOpNamed(sourceTrait, "__call__");
  FnOp targetCall = getFnOpNamed(targetTrait, "__call__");
  if (!sourceCall || !targetCall) {
    shared.diags.emitError(location,
                           "internal error: closure trait missing __call__");
    return nullptr;
  }

  // The extension has a single type parameter `Anchor`, constrained by the
  // source closure trait. Use the TraitType itself (same representation as
  // trait `_Self`), not `!lit.meta<!lit.trait<...>>` — binders pass the
  // source parameter / type value at the trait level.
  Type anchorConstraint = sourceMetaType.mlirType;
  if (auto anyTrait = sugarDynCast<AnyTraitType>(sourceMetaType))
    anchorConstraint = anyTrait.getTraitType();
  StringAttr anchorName = StringAttr::get(ctx, "Anchor");
  ParamDeclAttr paramAnchor = ParamDeclAttr::get(anchorName, anchorConstraint);
  TypedAttr anchorRef = ParamDeclRefAttr::get(paramAnchor);

  // The two traits need not expose matching associated types, so the target's
  // are solved against the source's rather than paired positionally.
  AdapteeParts extension;
  if (failed(inferClosureTraitExtension(shared, sourceTrait, targetTrait,
                                        anchorRef, extension, moduleDecl))) {
    shared.diags.emitError(
        location, "internal error: incompatible closure traits extended");
    return nullptr;
  }

  SmallString<128> name(targetTrait.getSymName());
  name += "$extension$";
  name += sourceTrait.getSymName();
  auto [structDecl, declOp] = createStruct(
      shared, moduleDecl, StringAttr::get(ctx, name), {paramAnchor}, location,
      /*passingKinds=*/{PassingKind::PosOnly});

  // A stateless extension carries no storage
  declOp.setConvention(TypeConvention::RegisterPassable);

  // The source trait, used as the trait key for the `get_witness` lookups
  // against `Anchor`.
  SymbolRefAttr sourceSymbol = getFullyResolvedSymbolRef(
      cast<mlir::SymbolOpInterface>(sourceTrait.getOperation()));
  auto sourceTraitSymbol = TraitSymbolAttr::get(sourceSymbol);

  // The anchor's own `__call__`, viewed through the extension's type parameter.
  ASTType anchorType(ParamType::get(anchorRef));
  FnTypeGeneratorType callWitnessType = specializeSignature(
      sourceCall.getFullSignature(), anchorType, *shared.declResolver);
  TypedAttr callWitness =
      GetWitnessAttr::get(ctx, anchorRef, sourceTraitSymbol,
                          sourceCall.getSymNameAttr(), callWitnessType);

  // If the target trait is more abstract than the struct an adaptor method is
  // needed.
  FuncTypeGeneratorType sourceClosureSignature =
      sourceTrait.getClosureSignature().value_or(nullptr);
  FuncTypeGeneratorType targetClosureSignature =
      targetTrait.getClosureSignature().value_or(nullptr);
  if (!sourceClosureSignature || !targetClosureSignature ||
      !isTypeRebindableTo(sourceClosureSignature, targetClosureSignature)) {
    buildCallAdaptorAndAddWitness(
        declOp, structDecl, targetTrait, targetCall,
        BindParamsAttr::get(ctx, callWitness, extension.fnLevelBindings,
                            &shared.getEvaluationContext()),
        extension, anchorType);
    return &structDecl;
  }

  // Build the single conformance table (for the target trait).
  SmallVector<std::pair<StringRef, TypedAttr>> witnesses;
  witnesses.emplace_back(targetCall.getSymNameAttr().getValue(), callWitness);

  // Associated-type aliases, as solved above against the source's.
  for (auto &[aliasName, aliasWitness] : extension.aliasSubstitutions)
    witnesses.emplace_back(aliasName.getValue(), aliasWitness);

  addConformanceTable(
      structDecl,
      ClosureEmitter::ClosureParent(shared, targetTrait.bindReference({}),
                                    "__call__", ClosureMethod::CALL),
      witnesses);
  return &structDecl;
}

CValue ClosureEmitter::createExtensionType(ASTDecl &fileModule,
                                           CValue sourceValue,
                                           Type targetMetaType,
                                           TraitDeclOp targetTrait) {
  assert(isa_and_nonnull<FileModuleOp>(fileModule.getIfOperation()) &&
         "extension structs must be emitted into a FileModuleOp");

  ASTType sourceType = sourceValue.getRValueType();
  std::optional<TraitDeclOp> sourceTraitOp =
      getClosureDecl(shared, sourceType.mlirType);
  if (!sourceTraitOp)
    return {};
  if (failed(isTraitCompatibleWith(sourceType, targetTrait, &fileModule)))
    return {};

  SMLoc loc = shared.diags.convertLocToSMLoc(targetTrait.getLoc());
  MLIRContext *ctx = shared.getContext();

  // The physical type value of the source closure.
  PValue sourcePValue = sourceValue.getIfPValue();
  TypedAttr anchor;
  if (sourcePValue)
    anchor = sourcePValue.get();
  else if (auto paramTy = dyn_cast<ParamType>(sourceType.mlirType))
    anchor = paramTy.getParam();
  else
    anchor = TypeParamAttr::get(sourceType.mlirType, sourceType.mlirType,
                                sourceType.extractMetaType());

  // The stateless extension struct that supplies the target-trait conformance
  // by forwarding to the anchor's own source-trait witnesses. Pass the source
  // trait type (not its metatype) so `Anchor`'s ParamDeclAttr matches the
  // trait-typed value bound below.
  ASTDecl *extensionDecl = shared.getOrCreateExtension(
      loc, *sourceTraitOp, targetTrait,
      ASTType(sourceTraitOp->getCanonicalTrait()), &fileModule);
  if (!extensionDecl)
    return {};
  auto extensionOp = cast<StructDeclOp>(extensionDecl->getIfOperation());

  // Bind the extension's anchor parameter `A` to the source closure.
  auto extensionStructType = extensionOp.bindReference({anchor});
  TypedAttr extensionTypeValue =
      TypeParamAttr::get(extensionStructType, extensionStructType,
                         StructMetaType::get(extensionStructType));

  // The augmented type: the anchor's physical identity, viewed as the target
  // trait, with the extension struct supplying the bridging conformance.
  TypedAttr extension =
      ExtensionAttr::get(ctx, targetMetaType, anchor, {extensionTypeValue});

  // The extension only fires in the trait-metatype domain.
  assert(!sourceValue.getMlirValue() &&
         "closure trait extension cannot see a runtime value");
  return PValue(extension);
}

static void populateDevicePassableTypeName(FnOp implementation,
                                           ASTDecl &structDecl,
                                           TypedAttr closureName) {
  MLIRContext *ctx = structDecl.getContext();
  Block &block = implementation.getBodyRegion().front();
  ImplicitLocOpBuilder b(implementation.getLoc(), implementation);
  b.setInsertionPointToStart(&block);
  IREmitter emitter(structDecl, b);
  SyntheticNode loc(structDecl.getLoc());

  ASTType strLitType = structDecl.getShared().lookupBuiltinType(
      "StringLiteral", structDecl, structDecl.getLoc());
  auto strLitDecl = cast<StructDeclOp>(
      strLitType.getDecl(structDecl.getShared())->getIfOperation());
  Type boundStrLitType = strLitDecl.bindReference({closureName});
  CValue literalValue = emitter.emitConstructorCall(
      ASTType(boundStrLitType),
      CallOperands(CallSyntax::kTypeCall, &loc, EC_CallArgValue));

  ASTType stringType = structDecl.getShared().lookupBuiltinType(
      "String", structDecl, structDecl.getLoc());
  ExprDest resultDest(MLValue(block.getArguments().back()), EC_ReturnValue);
  CallOperands ctorOperands(CallSyntax::kTypeCall, &loc, std::move(resultDest));
  ctorOperands.add(ASTExprAnd<CValue>{literalValue, &loc});
  emitter.emitConstructorCall(stringType, std::move(ctorOperands));
  auto noneAttr = KGEN::ParamConstantOp::create(b, KGEN::NoneAttr::get(ctx));
  IREmitter::emitNormalReturn(b, noneAttr);
}

static void emitIsConvertibleToDeviceTypeBody(
    FnOp implementation, ArrayRef<ParamDeclAttr> parameters,
    ImplicitLocOpBuilder &b, TypedAttr deviceTypeAttr) {
  b.setInsertionPointToStart(&implementation.getBodyRegion().front());
  assert(!parameters.empty() &&
         "expected _is_convertible_to_device_type to have type parameter");
  TypedAttr targetType = ParamDeclRefAttr::get(parameters.front());
  TypedAttr isConvertible = ParamIdenticalAttr::get(targetType, deviceTypeAttr);
  auto isConvertibleValue = KGEN::ParamConstantOp::create(b, isConvertible);
  IREmitter::emitNormalReturn(b, isConvertibleValue);
}

// TODO: replace clone with witness entry that binds self parameter once self
// parameter of traits becomes function level.
static void cloneTraitDefaultBody(FnOp implementation, FnOp traitFn,
                                  ASTDecl &structDecl) {
  implementation.getBodyRegion().getBlocks().clear();
  IRMapping mapping;
  traitFn.getBodyRegion().cloneInto(&implementation.getBodyRegion(), mapping);

  // Map `traitFn`'s parameters to `implementation` parameter.
  DenseMap<StringAttr, StringAttr> paramNames;
  for (auto [traitParam, implParam] :
       llvm::zip(traitFn.getInputParams(), implementation.getInputParams()))
    paramNames.insert({traitParam.getName(), implParam.getName()});

  ASTType selfType = structDecl.getTypeDeclSelf();
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](ParamDeclRefAttr paramRef) -> TypedAttr {
    if (auto it = paramNames.find(paramRef.getName()); it != paramNames.end())
      return ParamDeclRefAttr::get(it->second, paramRef.getType());
    return paramRef;
  });
  replacer.addReplacement([&](GetWitnessAttr getWitness) -> TypedAttr {
    SmallString<64> buf;
    llvm::raw_svector_ostream os(buf);
    getWitness.getTypeValue().print(os);
    if (!StringRef(buf).contains("_Self"))
      return getWitness;
    return GetWitnessAttr::get(PValue(selfType), getWitness.getTraitSymbol(),
                               getWitness.getWitnessName(),
                               getWitness.getType());
  });
  implementation.walk([&](Operation *op) {
    replacer.replaceElementsIn(op, /*replaceAttrs=*/true,
                               /*replaceLocs=*/false, /*replaceUses=*/false);
  });
}

static AliasDeclOp getDeviceTypeAlias(SharedState &shared, llvm::SMLoc loc) {
  ASTDecl *devicePassableTrait = shared.getBuiltinDevicePassableTrait(loc);
  assert(devicePassableTrait && "DevicePassable trait should be present");
  ArrayRef<ASTDecl *> aliasDecls = devicePassableTrait->lookupInCurrentScope(
      StringAttr::get(shared.getContext(), kDeviceType));
  assert(aliasDecls.size() == 1 &&
         "DevicePassable trait should define one device_type alias");
  return cast<AliasDeclOp>(aliasDecls.front()->getIfOperation());
}

void ClosureEmitter::addConformanceToDevicePassable(
    ASTDecl &structDecl, const DevicePassablePopulators &populators) {
  ASTDecl *devicePassableTrait =
      shared.getBuiltinDevicePassableTrait(structDecl.getLoc());
  if (!devicePassableTrait)
    return;
  if (failed(shared.declResolver->resolveBody(*devicePassableTrait,
                                              devicePassableTrait->getLoc())))
    return;
  TraitDeclOp trait = cast<TraitDeclOp>(devicePassableTrait->getIfOperation());
  for (auto &nameGroup : devicePassableTrait->getDeclsInScope()) {
    for (ASTDecl *funcFieldOrAlias : nameGroup.second) {
      if (failed(shared.declResolver->resolveBody(*funcFieldOrAlias,
                                                  funcFieldOrAlias->getLoc())))
        return;
    }
  }

  SmallVector<std::pair<StringRef, TypedAttr>> devicePassableWitnesses;
  TypedAttr deviceTypeWitness = populators.deviceType();

  for (Operation &member : trait.getFields().getOps()) {
    if (auto function = dyn_cast<FnOp>(member)) {
      FailureOr<SymbolConstantAttr> witness =
          [&]() -> FailureOr<SymbolConstantAttr> {
        if (function.getSourceName() == kIsDeviceTypeConvertible)
          return populators.isConvertible(function);
        if (function.getSourceName() == kIsImplicitlyEncodableTo)
          return populators.isEncodable(function);
        if (function.getSourceName() == kToDeviceType)
          return populators.toDeviceType(function);
        if (function.getIsStatic() &&
            function.getUserResultType() ==
                shared.lookupBuiltinType("String", structDecl,
                                         structDecl.getLoc()))
          return populators.typeName(function);
        llvm_unreachable("unexpected function in DevicePassable trait");
      }();
      if (failed(witness))
        return;
      devicePassableWitnesses.push_back(
          {*function.getSymName(), std::move(*witness)});
      continue;
    }

    if (auto alias = dyn_cast<AliasDeclOp>(member)) {
      assert(alias.getDeclName().getValue() == kDeviceType &&
             "unexpected alias in DevicePassable trait");
      devicePassableWitnesses.push_back({kDeviceType, deviceTypeWitness});
      continue;
    }
    llvm_unreachable(("unexpected member type '" +
                      member.getName().getStringRef().str() +
                      "' encountered in DevicePassable trait")
                         .c_str());
  }
  ClosureParent devicePassableParent(shared, trait.bindReference({}), "",
                                     ClosureMethod::NONE);
  addConformanceTable(structDecl, devicePassableParent,
                      devicePassableWitnesses);
}

void ClosureEmitter::addStorageConformanceToDevicePassable(
    ASTDecl &structDecl, ArrayRef<Type> deviceCaptureFieldTypes,
    StringRef name) {
  ASTDecl &fileModule = *structDecl.getNearestDeclOfType<FileModuleOp>();
  MLIRContext *ctx = structDecl.getContext();
  StructDeclOp structDeclOp = cast<StructDeclOp>(structDecl.getIfOperation());
  ImplicitLocOpBuilder b(structDeclOp->getLoc(), structDeclOp);
  FailureOr<LIT::StructType> deviceType = createDeviceTypeStruct(
      shared, fileModule, structDecl, deviceCaptureFieldTypes);
  if (failed(deviceType))
    return;

  TypedAttr deviceTypeValue;
  auto populateIsConvertible =
      [&](FnOp function) -> FailureOr<SymbolConstantAttr> {
    auto [implementation, parameters, result] = pushBackTraitFunctionImpl(
        function.getFullSignature(), structDecl, /*synthetic=*/true,
        function.getSourceNameAttr(), function.getSpecialFunctionKind(),
        function.getInlineLevel(),
        /*redirectWitnessToImplParam=*/false);
    emitIsConvertibleToDeviceTypeBody(implementation, parameters, b,
                                      deviceTypeValue);
    return buildSymbol(implementation, structDeclOp.getInputParams());
  };
  auto populateIsEncodable =
      [&](FnOp function) -> FailureOr<SymbolConstantAttr> {
    auto [implementation, parameters, result] = pushBackTraitFunctionImpl(
        function.getFullSignature(), structDecl,
        /*synthetic=*/true, function.getSymNameAttr(),
        function.getSpecialFunctionKind(), function.getInlineLevel(),
        /*redirectWitnessToImplParam=*/false);
    cloneTraitDefaultBody(implementation, function, structDecl);
    return buildSymbol(implementation, structDeclOp.getInputParams());
  };
  auto populateToDeviceType =
      [&](FnOp function) -> FailureOr<SymbolConstantAttr> {
    auto [toDevice, params, result] = pushBackTraitFunctionImpl(
        function.getFullSignature(), structDecl, /*synthetic=*/true,
        function.getSourceNameAttr(), function.getSpecialFunctionKind(),
        function.getInlineLevel(),
        /*redirectWitnessToImplParam=*/false);
    b.setInsertionPointToStart(&toDevice.getBodyRegion().front());
    assert(toDevice.getBodyRegion().getNumArguments() == 3);

    Value selfArgument = toDevice.getBodyRegion().front().getArgument(0);
    Value encoderRef = toDevice.getBodyRegion().front().getArgument(1);
    Value targetArgument = toDevice.getBodyRegion().front().getArgument(2);

    IREmitter emitter(structDecl, b);
    SyntheticNode syntheticNode(structDecl.getLoc());
    ExprDest dest(EC_ReturnValue);
    CallOperands callOperands(CallSyntax::kMethodCall, &syntheticNode,
                              std::move(dest));
    CValue encoderValue = CValue::getMValueForRef(encoderRef);
    callOperands.add({encoderValue, &syntheticNode});
    callOperands.add({CValue::getMValueForRef(selfArgument), &syntheticNode});
    callOperands.add({SRValue(targetArgument), &syntheticNode});
    OverloadSet overloads = OverloadSet::lookup(
        structDecl, encoderValue.getRValueType(), "encode_closure_state",
        &syntheticNode, CallSyntax::kMethodCall);
    overloads.paramBindings.add(&syntheticNode, PValue(deviceTypeValue),
                                StringAttr::get(ctx, "DeviceStructType"));
    auto calleeResult = overloads.filterOverloadSet(
        callOperands, /*emitDiagnosticOnFailure=*/true, emitter);
    if (!calleeResult.isYes())
      return failure();
    CValue callResult = emitter.emitIndirectCall(calleeResult.getYes(),
                                                 std::move(callOperands));
    if (!callResult)
      return failure();
    auto noneAttr =
        KGEN::ParamConstantOp::create(b, KGEN::NoneAttr::get(b.getContext()));
    IREmitter::emitNormalReturn(b, noneAttr);

    return buildSymbol(toDevice, structDeclOp.getInputParams());
  };
  auto populateTypeName = [&](FnOp function) -> FailureOr<SymbolConstantAttr> {
    auto [implementation, _, result] = pushBackTraitFunctionImpl(
        function.getFullSignature(), structDecl, /*synthetic=*/true,
        function.getSourceNameAttr(), function.getSpecialFunctionKind(),
        function.getInlineLevel(),
        /*redirectWitnessToImplParam=*/false);
    auto closureName = StringAttr::get(name, StringType::get(ctx));
    populateDevicePassableTypeName(implementation, structDecl, closureName);
    return buildSymbol(implementation, structDeclOp.getInputParams());
  };
  auto populateDeviceType = [&]() {
    deviceTypeValue = TypeParamAttr::get(
        *deviceType, getDeviceTypeAlias(shared, structDecl.getLoc()).getType());
    return deviceTypeValue;
  };
  DevicePassablePopulators populators{populateIsConvertible,
                                      populateIsEncodable, populateToDeviceType,
                                      populateTypeName, populateDeviceType};
  addConformanceToDevicePassable(structDecl, populators);
}

bool ClosureEmitter::isTypeRebindableTo(FuncTypeGeneratorType from,
                                        FuncTypeGeneratorType to) {
  if (from == to)
    return true;
  if (from.getInputParamTypes() != to.getInputParamTypes())
    return false;
  if (from.getBody() != to.getBody())
    return false;

  // Enforce parameter-name equality for every passing kind except
  // `Inferred`. Inferred parameters appear before the `+` separator in the
  // pog list and are not user-bindable, so their names are arbitrary
  // disambiguators that may legitimately differ between alpha-equivalent
  // generator types.
  PogListAttr fromPogs = from.getParamListAttrs();
  PogListAttr toPogs = to.getParamListAttrs();
  if (!fromPogs || !toPogs)
    return false;
  if (fromPogs == toPogs)
    return true;
  if (fromPogs.getOrigVariadicConvention() !=
      toPogs.getOrigVariadicConvention())
    return false;
  ArrayRef<PogMetadataAttr> a = fromPogs.getPogs();
  ArrayRef<PogMetadataAttr> b = toPogs.getPogs();
  assert(a.size() == b.size() &&
         "PogListAttr size invariant: tied to input-param-types count");
  for (auto [pa, pb] : llvm::zip(a, b)) {
    if (pa.getPassingKind() != pb.getPassingKind() ||
        pa.getVariadic() != pb.getVariadic() ||
        pa.getDefaultValue() != pb.getDefaultValue() ||
        (pa.getPassingKind() != PassingKind::Inferred &&
         pa.getName() != pb.getName()))
      return false;
  }
  return true;
}
