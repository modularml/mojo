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
// This file provides the implementation of the StructEmitter class.
//
//===----------------------------------------------------------------------===//

#include "StructEmitter.h"
#include "ExprNodes.h"
#include "IREmitter.h"
#include "MojoUtils.h"
#include "OverloadSet.h"
#include "ParserBase.h"
#include "ParserEvaluationContext.h"
#include "Traits.h"

#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Support/Compiler/OperationUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/StringExtras.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

/// Check if a parameterized field type conditionally conforms to a builtin
/// trait based on the function's where-clause constraints.  Constructs a
/// conforms_to proposition and delegates to isImplicationProven.
///
/// Returns the resolved TraitDeclOp on success (for reuse in code-gen),
/// or a null TraitDeclOp if no conditional conformance was found.
static TraitDeclOp
fieldConditionallyConformsToBuiltin(ASTType fieldType, StringRef traitName,
                                    SharedState &shared, ASTDecl &structDecl,
                                    ArrayRef<ConstraintAttr> bodyConstraints) {
  auto paramType = dyn_cast<ParamType>(fieldType);
  if (!paramType)
    return {};

  ASTDecl *requiredTraitDecl =
      shared.lookupBuiltinTrait(traitName, structDecl.getLoc());
  if (!requiredTraitDecl)
    return {};

  auto traitOp = dyn_cast<TraitDeclOp>(requiredTraitDecl->getIfOperation());
  if (!traitOp)
    return {};

  TypedAttr fieldParam = getCanonicalAttr(paramType.getParam());
  SmallVector<TraitSymbolAttr> traitSymbols = {
      TraitSymbolAttr::get(requiredTraitDecl->getSymbolRef())};
  auto traitType = TraitType::get(fieldParam.getContext(), traitSymbols);
  auto conformsTo =
      TypeConformsToTraitAttr::get(fieldParam, traitType.getPValue());

  for (ConstraintAttr constraint : bodyConstraints)
    if (isImplicationProven(conformsTo, constraint.getProposition()))
      return traitOp;
  return {};
}

//===----------------------------------------------------------------------===//
// FunctionEmitter
//===----------------------------------------------------------------------===//

static FnOp
createFunction(ASTDecl &parent, StringRef name, ArrayRef<ParamDeclAttr> params,
               PogListAttr paramListAttrs, ArrayRef<Type> argTypes,
               ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
               Type resultType, SpecialFunctionKind specialFnID, SMLoc loc,
               ImplicitLocOpBuilder &builder, FnEffects fnEffects,
               StringRef suffix, bool synthetic, InlineLevel inlineLevel) {
  MLIRContext *ctx = parent.getContext();
  SharedState &shared = parent.getShared();

  // Replace all `ImplicitOriginRefAttr` with `ParamRefDeclAttr`s that point to
  // explicitly *named* parameter-decls.
  ImplicitOriginToNameRefAttrReplacer indexReplacer(ctx);

  // The caller specifies all the input types, which means that all the input
  // reference types that carry implicit origins will already have them
  // specified with names, so dig those out and use them as parameters.
  // If the caller provided indexed inputs, rewrite them to named inputs as our
  // body will expect.
  SmallVector<Type> adjustedArgTypes;
  for (auto [argNo, argTypeMaybeWithIndices, argConv] :
       llvm::enumerate(argTypes, argConventions)) {
    auto argTypeNoIndices = indexReplacer.replace(argTypeMaybeWithIndices);
    adjustedArgTypes.push_back(argTypeNoIndices);
    if (!hasImplicitOrigin(argConv))
      continue;

    // Dig out the origin decl.
    auto refArgType = cast<RefType>(argTypeNoIndices);
    auto originAttr = refArgType.getOrigin();
    ParamDeclAttr decl;
    // If this is a reference to a named one already, just reuse the name.
    if (auto originRef =
            dyn_cast<ParamDeclRefAttr>(OriginMutCastAttr::strip(originAttr))) {
      assert(isa<OriginType>(originRef.getType()) &&
             "origins should have OriginType");
      // Look through a cast to get the name, but use the expected mutability of
      // the origin type.
      decl = ParamDeclAttr::get(originRef.getName(), originAttr.getType());
    }
    // Only add a param decl if it doesn't already exist (and we don't already
    // plan to add it).
    bool foundExisting = false;
    for (auto &existingParam : params) {
      if (existingParam.getName() == decl.getName() &&
          existingParam.getType() == decl.getType()) {
        foundExisting = true;
      }
    }
    for (auto &existingParam : indexReplacer.getNewOriginParamDecls()) {
      if (existingParam.getName() == decl.getName() &&
          existingParam.getType() == decl.getType()) {
        foundExisting = true;
      }
    }
    if (!foundExisting)
      indexReplacer.getNewOriginParamDecls().push_back(decl);
  }
  size_t numImplicitOriginDecls = indexReplacer.getNewOriginParamDecls().size();

  auto metadata = FnMetaOriginDataAttr::get(
      argListAttrs.getContext(), numImplicitOriginDecls,
      getOriginsAccessibleByParams(paramListAttrs, params, shared,
                                   /*captureOrigins=*/nullptr),
      /*isNestedOriginsReadOnly=*/false, /*definesInteriorOrigins=*/false);
  FunctionType functionType =
      builder.getFunctionType(adjustedArgTypes, {resultType});
  Location location = shared.translateLocation(loc);
  FnTypeGeneratorType sigGen = FuncTypeGeneratorType::remapToFuncTypeGenerator(
      params, functionType, argConventions, fnEffects, metadata, paramListAttrs,
      [&] { return mlir::emitError(location); }, argListAttrs);
  // Strip off the named origin decl references and replace them with indices.
  // We keep the named parameters in the ParamDeclAttr list on the FnOp and
  // in the BBArgs.
  sigGen = sigGen.replaceImplicitOriginsWithIndexes(
      indexReplacer.getNewOriginParamDecls());

  StringAttr sourceName = builder.getStringAttr(name);
  StringAttr mangledName = builder.getStringAttr(
      DeclResolver::getMangledName(sourceName, parent, sigGen).getValue() +
      suffix);

  // If a function with this signature already exists in the struct, don't
  // create a new one. Return null to indicate that there was an existing
  // method.
  if (shared.lookupSymbolIn(&parent, mangledName))
    return nullptr;

  FnOp fnOp = FnOp::create(builder, mangledName, sourceName, sigGen);

  // Set the attributes on the FnOp in bulk.
  NamedAttrList attrs = fnOp->getAttrDictionary();

  if (SpecialFunctionInfo::get(specialFnID).isImplicitlyStatic())
    attrs.set(fnOp.getIsStaticAttrName(), UnitAttr::get(ctx)); // True.

  // Figure out the full set of parameter declarations, this is the explicit
  // parameter declarations + implicit origins.
  SmallVector<ParamDeclAttr> fullParams;
  llvm::append_range(fullParams, params);
  llvm::append_range(fullParams, indexReplacer.getNewOriginParamDecls());
  if (!fullParams.empty()) {
    attrs.set(fnOp.getParamsAttrName(),
              builder.getAttr<ParamDeclArrayAttr>(fullParams));
  }

  attrs.set(fnOp.getSpecialFnKindAttrName(),
            builder.getI8IntegerAttr(uint8_t(specialFnID)));
  attrs.set(fnOp.getSyntheticAttrName(), UnitAttr::get(ctx)); // True.
  attrs.set(fnOp.getFunctionTypeAttrName(), TypeAttr::get(functionType));
  attrs.set(fnOp.getInlineLevelAttrName(),
            InlineLevelAttr::get(ctx, inlineLevel));
  fnOp->setAttrs(attrs.getDictionary(ctx));

  // Generate a debug subprogram for this function.
  shared.setLocationDebugScope(fnOp);
  if (!fnOp.getBody())
    fnOp.getBodyRegion().push_back(new Block());
  for (Type argType : adjustedArgTypes)
    fnOp.getBody()->addArgument(argType, fnOp.getLoc());
  return fnOp;
}

std::pair<FnOp, ASTDecl *> FunctionEmitter::synthesizeFunction(
    ASTDecl &parent, StringRef name, ArrayRef<ParamDeclAttr> params,
    PogListAttr paramListAttrs, ArrayRef<Type> argTypes,
    ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
    Type resultType, SpecialFunctionKind specialFnID, SMLoc loc,
    ImplicitLocOpBuilder &builder, FnEffects fnEffects, StringRef suffix,
    bool synthetic, InlineLevel inlineLevel) {
  FnOp funcOp =
      createFunction(parent, name, params, paramListAttrs, argTypes,
                     argConventions, argListAttrs, resultType, specialFnID, loc,
                     builder, fnEffects, suffix, synthetic, inlineLevel);

  // Return null if the function already exists with the same signature.
  if (!funcOp)
    return {nullptr, nullptr};

  // Register the method in the struct.
  ASTDecl &funcDecl = shared.declResolver->addFullyResolvedDecl(
      funcOp.getOperation(), StringAttr::get(shared.getContext(), name), loc,
      &parent);

  // Synthesized functions skip DeclResolution's insertKnownAssumptions, so
  // inject the fn's where-clause constraints as known assumptions here,
  // when the ASTDecl is first created.
  funcDecl.insertKnownAssumptions(paramListAttrs.getBodyConstraints());

  // Set the symbol and notice if we are redeclaring something.
  [[maybe_unused]] Operation *existing =
      shared.declResolver->finalizeFuncSignature(funcOp, funcDecl);
  assert(!existing && "unexpected redefinition of synthesized method");

  return {funcOp, &funcDecl};
}

FnOp StructEmitter::synthesizeDefaultTraitMethodWrapper(
    ASTDecl &existingDecl, StringRef name, FnTypeGeneratorType wrapperSignature,
    FnOp traitFn, ASTDecl *traitFnDecl, ImplicitLocOpBuilder &builder,
    StringRef suffix, ConstraintAttr conformanceConstraint) {

  assert(existingDecl.resolvedness <= DeclResolvedness::signature &&
         "synthesizeMethodInStruct is only valid on non-body resolved Fn "
         "ASTDecls");

  // Extract signature components from the high-level types
  FuncType fnType = wrapperSignature.getBody();
  ArrayRef<Type> inputTypes = fnType.getArguments();

  PogListAttr traitArgListAttrs =
      traitFn.getFuncTypeGenerator().getArgListAttrs();

  ArrayRef<ParamDeclAttr> params = traitFn.getParams().drop_back(
      wrapperSignature.getNumImplicitOriginDecls());

  SmallVector<ParamDeclAttr> mangledParams;
  for (ParamDeclAttr param : params) {
    // Mangle the param name if a conflict exists -- this is needed for cases
    // where the struct we're creating the wrapper function in has a param with
    // the same name as one defined by the default trait method, for example:
    //
    // trait Foo:
    //   def foo[x: Int](): ...
    //
    // struct Bar[x: Int](Foo): ...
    StringAttr mangledName =
        structDecl.mangleUserDefinedParamName(param.getName());
    ParamDeclAttr newParamDecl =
        ParamDeclAttr::get(mangledName, param.getType());
    mangledParams.push_back(newParamDecl);
  }

  // 'wrapperSignature' is generated by 'getTraitFunctionSignature' in
  // Traits.cpp and remaps param decl ref attrs to index ref attrs (for the sake
  // of comparing struct methods to trait methods to check conformances).
  //
  // Since we're also using it in this function to help synthesize a
  // default trait method wrapper function we must map ParamIndexRefAttrs back
  // to ParamDeclRefAttrs. (Otherwise we'd have a mismatch in expected types in
  // the lit.call op we later materialize in populateDefaultedTraitFunction).
  //
  // See DCRTODS in arcana/Generics.md for more details.
  SmallVector<Type> argTypes;
  SmallVector<ArgConvention> argConventions;
  for (auto [idx, argType] : llvm::enumerate(inputTypes)) {
    argTypes.push_back(replaceIndexRefsWithNamedRefs(argType, mangledParams));
    argConventions.push_back(fnType.getArgConvention(idx));
  }

  // Make sure we remap any IndexRefAttrs in the result type as well
  Type resultType = replaceIndexRefsWithNamedRefs(
      wrapperSignature.getResultType(), mangledParams);

  InlineLevel inlineLevel = InlineLevel::Automatic;
  if (structDeclOp.getConvention() == TypeConvention::RegisterPassableTrivial)
    inlineLevel = InlineLevel::AlwaysNoDebug;

  SmallVector<ConstraintAttr> bodyConstraints;
  if (conformanceConstraint &&
      !isTriviallyTrueConstraint(conformanceConstraint))
    bodyConstraints.push_back(conformanceConstraint);

  PogListAttr wrapperParamListAttrs = wrapperSignature.getParamListAttrs();
  PogListAttr paramListAttrs = PogListAttr::get(
      wrapperParamListAttrs.getContext(), wrapperParamListAttrs.getPogs(),
      bodyConstraints, wrapperParamListAttrs.getOrigVariadicConvention());
  FnOp funcOp =
      createFunction(structDecl, name, mangledParams, paramListAttrs, argTypes,
                     argConventions, traitArgListAttrs, resultType,
                     SpecialFunctionKind::kNormal, structDecl.getLoc(), builder,
                     traitFn.getFuncTypeGenerator().getFnEffects(), suffix,
                     /*synthetic=*/true, inlineLevel);

  if (!funcOp)
    return nullptr;

  // createFunction first calls FuncTypeGeneratorType::remapToFuncTypeGenerator
  // on the passed in arg/result types and constructs the FnOp with that. Part
  // of what that function does is remap ParamDeclRefAttrs to IndexRefAttrs of
  // arg/result types. Set the proper FunctionType here.
  //
  // TODO: Should this logic just go right into createFunction?
  funcOp.setFunctionType(FunctionType::get(
      funcOp.getContext(), funcOp.getFunctionType().getInputs(), {resultType}));

  // Attach the new operation to the provided declaration.
  existingDecl.setIRValue(funcOp.getOperation());
  existingDecl.resolvedness = DeclResolvedness::signature;

  [[maybe_unused]] Operation *existing =
      shared.declResolver->finalizeFuncSignature(funcOp, existingDecl);
  assert(!existing &&
         "unexpected redefinition when synthesizing method into existing decl");

  existingDecl.insertKnownAssumptions(paramListAttrs.getBodyConstraints());

  assert(funcOp && "Couldn't synthesize default trait wrapper in body");

  funcOp.setInheritedFromAttr(
      TraitSymbolAttr::get(traitFnDecl->getParentDecl()->getSymbolRef()));
  // Annotate with metadata linking back to trait default implementation. The
  // trait names the conformance this wrapper satisfies; the function symbol is
  // what `populateDefaultedTraitFunction` forwards the call to.
  funcOp.setDefaultFnRefAttr(traitFnDecl->getSymbolRef());

  // Right now there's not really a great way to re-apply the decorators that
  // were on the defaulted trait method to the struct's wrapper lit.fn op, but
  // fortunately all the behavior for our current set of decorators is limited
  // changing the def op's signature or attribute values.
  if (traitFn.getIsStatic())
    funcOp.setIsStatic(true);

  if (traitFn.isImplicitConversion())
    funcOp.setImplicitConversion(traitFn.getImplicitConversion());

  if (traitFn.isExternal())
    funcOp.setExternal(true);

  funcOp.setExportKind(traitFn.getExportKind());

  if (!traitFn.getLLVMMetadataArray().empty())
    funcOp.setLLVMMetadataArrayAttr(traitFn.getLLVMMetadataArrayAttr());

  if (!traitFn.getLLVMArgMetadataArray().empty())
    funcOp.setLLVMArgMetadataArrayAttr(traitFn.getLLVMArgMetadataArrayAttr());

  funcOp.setInlineLevel(KGEN::InlineLevel::AlwaysNoDebug);

  // When we're in the LSP we may not fully body resolve the wrapper
  // functions. Add a EndFnOp with unresolved=True so we can still verify
  // cleanly in passes run by the check LIT pipeline.
  auto atBlockEndBuilder = OpBuilder::atBlockEnd(funcOp.getBody());
  EndFnOp::create(atBlockEndBuilder, funcOp.getLoc(), /*unresolved=*/true);

  return funcOp;
}

/// Populates a struct's default trait method wrapper with the IR to actually
/// call the the trait method its wrapping. Takes the stub function that was
/// created during synthesizeDefaultTraitMethodWrapper and forwards all the
/// arguments of the FnOp created there to the call op on the actual defaulted
/// trait method.
LogicalResult StructEmitter::populateDefaultedTraitFunction(ASTDecl &fnDecl) {
  auto fn = cast<FnOp>(fnDecl.getIfOperation());
  ASTDecl &structDecl = *fnDecl.getParentDecl();
  ASTType structSelfType = structDecl.getTypeDeclSelf();

  IREmitter emitter(fnDecl, EC_Trait);
  fn.getBody()->clear(); // Remove the lit.end_fn

  emitter.builder = OpBuilder::atBlockBegin(fn.getBody());

  auto defaultFnRefAttr = fn.getDefaultFnRef();
  assert(defaultFnRefAttr &&
         "defaultFnRef attribute should always be present on a"
         " default-method stub");

  // Look up the trait's default implementation function
  ASTDecl *traitDefaultMethodDecl =
      shared.declResolver->getDeclForFuncSymbol(*defaultFnRefAttr);
  assert(traitDefaultMethodDecl &&
         "Could not find trait default method implementation");

  ASTDecl *parentTraitDecl = traitDefaultMethodDecl->getParentDecl();
  [[maybe_unused]] LogicalResult result =
      fnDecl.getShared().getDeclResolver().resolveSignature(*parentTraitDecl,
                                                            fnDecl.getLoc());
  assert(result.succeeded() && "failed to resolve signature");

  TraitType parentTrait =
      cast<TraitDeclOp>(parentTraitDecl->getIfOperation()).getCanonicalTrait();
  SyntheticNode synthNode(structDecl.getLoc());
  CValue selfTypeCValue(structSelfType.mlirType);
  PValue selfAsTrait = emitter.emitMetaTypeToTraitConversion(
      {selfTypeCValue, &synthNode}, parentTrait);

  // emitMetaTypeToTraitConversion can fail if the struct holding the defaulted
  // trait function wrapper didn't conform to the trait due to an unimplemented
  // function.
  // Simply bail early without worrying about the body of the lit.fn op we're
  // currently working on as compilation will fail anyways.
  if (selfAsTrait.isNull()) {
    fnDecl.setErroneous();
    return failure();
  }

  FnOp traitDefaultMethodOp =
      cast<FnOp>(traitDefaultMethodDecl->getIfOperation());
  auto fnTypeGen = traitDefaultMethodOp.getFullSignature();

  auto &builder = *emitter.builder;

  // Collect the bindings needed to call the trait method in this.
  SmallVector<TypedAttr> callParamBindings;

  callParamBindings.push_back(selfAsTrait.get());

  auto fnParams =
      fn.getParams().drop_back(fnTypeGen.getNumImplicitOriginDecls());

  for (ParamDeclAttr param : fnParams)
    callParamBindings.push_back(KGEN::ParamDeclRefAttr::get(param));

  // create a specialized generator from the fnTypeGen
  FuncTypeGeneratorType specializedGenerator =
      fnTypeGen.getSpecializedGenerator(
          callParamBindings, &fnDecl.getShared().getEvaluationContext(),
          fn.getLoc());

  // Bail if specialization failed (already diagnosed)
  if (!specializedGenerator) {
    fnDecl.setErroneous();
    return failure();
  }

  SymbolRefAttr calleeSym =
      LIT::getFullyResolvedSymbolRef(traitDefaultMethodOp);
  TypedAttr typedSymbol = KGEN::SymbolConstantAttr::get(
      calleeSym, specializedGenerator, callParamBindings);

  SmallVector<Value> operands(fn.getArguments().begin(),
                              fn.getArguments().end());

  // Determine the values for all of the implicit origins that need to be passed
  // to the calls.  These are the origins of any reference argument.
  SmallVector<TypedAttr> implicitOrigins;
  auto argConvs = fnTypeGen.getArgConventions();
  for (auto [idx, val, conv] : llvm::enumerate(operands, argConvs)) {
    if (hasImplicitOrigin(conv))
      implicitOrigins.push_back(cast<LIT::RefType>(val.getType()).getOrigin());
  }

  ArrayRef<Type> resultTypes = specializedGenerator.getBody().getResults();
  auto callOp = LIT::CallOp::create(builder, fn.getLoc(), resultTypes,
                                    typedSymbol, implicitOrigins, operands);

  emitter.emitNormalReturn(fn.getLoc(), callOp->getResult(0));

  fnDecl.resolvedness = DeclResolvedness::body;
  return success();
}

//===----------------------------------------------------------------------===//
// StructEmitter
//===----------------------------------------------------------------------===//

StructEmitter::StructEmitter(ASTDecl &structDecl)
    : FunctionEmitter(structDecl.getShared()), structDecl(structDecl) {
  structDeclOp = cast<StructDeclOp>(*structDecl.getIfOperation());
}

std::pair<FnOp, ASTDecl *> StructEmitter::synthesizeMethodInStruct(
    StringRef name, ArrayRef<Type> argTypes,
    ArrayRef<ArgConvention> argConventions, PogListAttr argListAttrs,
    Type resultType, SpecialFunctionKind specialFnID, FnEffects fnEffects,
    StringRef suffix, bool synthetic) {
  return synthesizeMethodInStruct(
      name, /*params=*/{}, /*paramListAttrs=*/PogListAttr::get(getContext()),
      argTypes, argConventions, argListAttrs, resultType, specialFnID,
      fnEffects, suffix, synthetic);
}

std::pair<FnOp, ASTDecl *> StructEmitter::synthesizeMethodInStruct(
    StringRef name, ArrayRef<ParamDeclAttr> params, PogListAttr paramListAttrs,
    ArrayRef<Type> argTypes, ArrayRef<ArgConvention> argConventions,
    PogListAttr argListAttrs, Type resultType, SpecialFunctionKind specialFnID,
    FnEffects fnEffects, StringRef suffix, bool synthetic) {
  ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockEnd(
      structDeclOp.getLoc(), &structDeclOp.getFields().front());
  InlineLevel inlineLevel = InlineLevel::Automatic;
  // If the struct is TrivialRegisterPassable, make this
  // @always_inline("nodebug").
  if (structDeclOp.getConvention() == TypeConvention::RegisterPassableTrivial)
    inlineLevel = InlineLevel::AlwaysNoDebug;
  return synthesizeFunction(structDecl, name, params, paramListAttrs, argTypes,
                            argConventions, argListAttrs, resultType,
                            specialFnID, structDecl.getLoc(), builder,
                            fnEffects, suffix, synthetic, inlineLevel);
}

/// Given a struct and a trait declaration, make the struct inherit from the
/// trait if it does not already.
static void addTraitParent(StructDeclOp structOp, ASTDecl *traitDecl) {
  // Pull in the entire ancestor chain of the new symbol.
  SmallVector<TraitSymbolAttr> newSymbols = {
      TraitSymbolAttr::get(traitDecl->getSymbolRef())};
  canonicalizeTraitCompositionSymbols(traitDecl->getShared(), newSymbols);
  // Merge the new canonical symbols with the existing canonical trait symbols.
  TraitType trait = structOp.getCanonicalTrait();
  llvm::append_range(newSymbols, trait.getSymbols());
  // No need to pull in any ancestors now. Just sort and deduplicate.
  sortAndDeduplicateTraitSymbols(newSymbols);
  structOp.setCanonicalTrait(TraitType::get(structOp.getContext(), newSymbols));
}

/// Add a attribute initializer method for this struct with a body.
FnOp StructEmitter::synthesizeFieldwiseInit() {
  ASTType selfType = structDecl.getTypeDeclSelf();

  SmallVector<Type> argTypes;
  SmallVector<ArgConvention> argConventions;
  SmallVector<StringAttr> argNames;
  SmallVector<PassingKind> argPassingKinds;

  // We declare all of the operands to the init constructor as owned.  This
  // enables it to work with move-only fields, and, for copyable types, forces
  // the copy into the caller, which can then be elided with a consume or
  // RValue.
  for (auto fieldOp : structDeclOp.getFieldDecls()) {
    ASTType fieldType = fieldOp.getType();
    ArgConvention conv;
    switch (fieldType.getRegisterPassability(structDecl.getLoc(), shared)) {
    case TypeConvention::MemoryOnly:
    case TypeConvention::Unspecified:
    case TypeConvention::RegisterPassable:
      fieldType = fieldType.getRefForArgument(fieldOp.getName().str(),
                                              /*isMut=*/true);
      conv = ArgConvention::OwnedMem;
      break;
    case TypeConvention::RegisterPassableTrivial:
      conv = ArgConvention::ImmReg;
      break;
    }
    argTypes.push_back(fieldType);
    argConventions.push_back(conv);
    argNames.push_back(fieldOp.getNameAttr());
    argPassingKinds.push_back(PassingKind::PosOrKw);
  }

  // Add the 'out self' argument if memory-only.
  Type litResultType = selfType;
  if (!selfType.isRegisterPassable(structDecl.getLoc(), shared)) {
    litResultType = shared.getNoneType();
    argTypes.push_back(selfType.getRefForArgument("self", /*isMut=*/true));
    argConventions.push_back(ArgConvention::ByRefResult);
    argNames.push_back(StringAttr::get(shared.getContext(), "self"));
    argPassingKinds.push_back(PassingKind::Implicit);
  }

  return synthesizeFieldwiseInit(
      argTypes, argConventions,
      PogListAttr::get(getContext(), argNames, argPassingKinds), litResultType);
}

FnOp StructEmitter::synthesizeFieldwiseInit(
    ArrayRef<Type> argTypes, ArrayRef<ArgConvention> argConventions,
    PogListAttr argListAttrs,
    // None or Self if register passable.
    ASTType litReturnType) {

  // Create the FnOp and ASTDecl for the method.
  auto [funcOp, _] = synthesizeMethodInStruct(
      "__init__", argTypes, argConventions, argListAttrs, litReturnType,
      SpecialFunctionKind::kInit);
  assert(funcOp && "couldn't synthesize method or had a conflict?");
  funcOp.setInlineLevel(InlineLevel::AlwaysNoDebug);

  // Set up the body.
  ImplicitLocOpBuilder builder =
      ImplicitLocOpBuilder::atBlockEnd(funcOp.getLoc(), funcOp.getBody());
  Block *body = funcOp.getBody();
  builder.setInsertionPointToStart(body);
  builder.setLoc(funcOp->getLoc());
  ASTDecl *funcDecl = shared.declResolver->getDeclForFuncSymbol(
      getFullyResolvedSymbolRef(funcOp));
  IREmitter emitter(*funcDecl, builder);

  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(funcOp.getLocScope());

  Value selfValue;
  bool hasResultTemp = false;
  if (!argConventions.empty() &&
      argConventions.back() == ArgConvention::ByRefResult) {
    selfValue = body->getArgument(body->getNumArguments() - 1);
  } else {
    // Register result needs a temporary.
    hasResultTemp = true;
    selfValue = emitter.emitVarDecl("self", litReturnType, funcOp.getLoc(),
                                    VarDeclKind::InitOutArg);
  }

  // Emit a bunch of stores to fields indexing our 'out self' result.
  for (auto [idx, fieldOp] : llvm::enumerate(structDeclOp.getFieldDecls())) {
    ASTType fieldType = fieldOp.getType();

    // TODO: Add a nicer accessor.
    auto fieldEntries = structDecl.lookupInCurrentScope(fieldOp.getNameAttr());
    assert(fieldEntries.size() == 1 && "field decls cannot be overloaded");
    ASTDecl &fieldASTDecl = *fieldEntries[0];

    if (!fieldType.isImplicitlyCopyable(fieldASTDecl.getLoc(), shared,
                                        *funcDecl) &&
        !fieldType.isMovable(fieldASTDecl.getLoc(), shared, *funcDecl)) {
      auto diag = emitError(fieldASTDecl.getLoc())
                  << "cannot synthesize fieldwise init because field '"
                  << fieldOp.getName()
                  << "' has non-copyable and non-movable type " << fieldType;
      return {};
    }

    // Add the block argument, get it as an RValue since it is owned. Skip the
    // self argument.
    BlockArgument arg = body->getArgument(idx);
    CValue argVal;
    switch (argConventions[idx]) {
    default:
      llvm_unreachable("unknown convention");
    case ArgConvention::ImmReg:
      argVal = SRValue(arg);
      break;
    case ArgConvention::OwnedMem:
      argVal = MRValue(arg);
      break;
    case ArgConvention::ImmMem:
      argVal = MBValue(arg);
      break;
    }

    // Project self to the right field and store the RValue.
    auto fieldRef = RefStructGEROp::create(builder, selfValue, fieldOp);
    SyntheticNode node(structDecl.getLoc());
    emitter.emitStoreToLValue({argVal, &node}, MLValue(fieldRef),
                              EC_AttributeRefBase);
  }

  // For a register-passable result, load the result from the temporary.
  Value returnVal;
  if (hasResultTemp) {
    SyntheticNode exprTmp(funcDecl->getLoc());
    returnVal =
        emitter.emitSRValue({MRValue(selfValue), &exprTmp}, EC_ReturnValue);
  }

  // Finish off the function with a return + lit.endfunc.
  emitter.emitNormalReturn(funcOp.getLoc(), returnVal);
  return funcOp;
}

FnOp StructEmitter::synthesizeEmptyDtor(ConstraintAttr conformanceConstraint) {
  auto builder = ImplicitLocOpBuilder::atBlockEnd(
      structDeclOp.getLoc(), &structDeclOp.getFields().front());

  // Figure out the type of the 'self' argument, which is always indirect since
  // it is owned.
  ASTType selfType = structDecl.getTypeDeclSelf();
  if (!selfType)
    return {};

  selfType = selfType.getRefForArgument("self", /*isMut*/ true);
  StringAttr selfName = builder.getStringAttr("self");

  SmallVector<ConstraintAttr> constraints;
  if (conformanceConstraint &&
      !isTriviallyTrueConstraint(conformanceConstraint))
    constraints.push_back(conformanceConstraint);

  // Create the FnOp and ASTDecl for the method.
  auto [funcOp, funcDecl] = synthesizeMethodInStruct(
      "__deinit__", /*params=*/{},
      /*paramListAttrs=*/
      PogListAttr::get(getContext(), /*numPogs=*/0, constraints),
      /*argTypes=*/{selfType.mlirType},
      /*argConvs=*/{ArgConvention::DeinitMem},
      /*argListAttrs=*/
      PogListAttr::get(getContext(), selfName, PassingKind::PosOnly),
      shared.getNoneType(), SpecialFunctionKind::kDeinit);
  if (!funcOp)
    return {};
  funcOp.setInlineLevel(InlineLevel::AlwaysNoDebug);

  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(funcOp.getLocScope());

  // Finish off the function with a return + lit.endfunc.
  builder = ImplicitLocOpBuilder::atBlockEnd(funcOp.getLoc(), funcOp.getBody());
  IREmitter::emitNormalReturn(builder);

  // Perform semantic analysis to make sure all the fields are implicitly
  // deletable. We would rather error in the parser than in check lifetimes.
  // We do this after creating the function so we don't emit a redundant error
  // complaining that a missing __deinit__ means the struct doesn't conform to
  // Deinitable.
  SmallVector<ConstraintAttr> assumptions;
  structDecl.getKnownAssumptionsIncludingParents(assumptions);
  assumptions.append(constraints);
  for (StructFieldOp fieldOp : structDeclOp.getFieldDecls()) {
    ASTType fieldType = fieldOp.getType();
    TriBool confResult =
        fieldType
            .conformsToBuiltinTrait("Deinitable", structDecl.getLoc(), shared,
                                    assumptions)
            .first;
    if (!confResult.isTrue() &&
        !fieldConditionallyConformsToBuiltin(fieldType, "Deinitable", shared,
                                             structDecl, assumptions) &&
        !fieldType.isTrivialRegisterType(structDecl.getLoc(), shared)) {
      emitError(fieldOp.getLoc())
          << "field '" << fieldOp.getName() << "' has non-'Deinitable' type "
          << fieldType;
      funcDecl->setErroneous();
      break;
    }
  }

  return funcOp;
}

FnOp StructEmitter::synthesizeEmptyMoveOrCopyInit(
    bool isMove, ConstraintAttr conformanceConstraint) {
  ASTType selfType = structDecl.getTypeDeclSelf();
  MLIRContext *ctx = shared.getContext();
  Builder b(ctx);
  StringAttr srcName = b.getStringAttr(isMove ? "move" : "copy");

  // If the type is register passable trivial, the 'src' value will be
  // passed as a register, otherwise a reference.
  Type srcArgType = selfType.getRefForArgument(srcName.strref(), isMove);
  ArgConvention srcConv =
      isMove ? ArgConvention::DeinitMem : ArgConvention::ImmMem;

  SmallVector<ConstraintAttr> constraints;
  if (conformanceConstraint &&
      !isTriviallyTrueConstraint(conformanceConstraint))
    constraints.push_back(conformanceConstraint);

  Type selfArgType = selfType.getRefForArgument("self", /*isMut=*/true);
  auto argListAttrs =
      PogListAttr::get(ctx, {srcName, b.getStringAttr("self")},
                       {PassingKind::KwOnly, PassingKind::Implicit});
  auto [resultFn, resultDecl] = synthesizeMethodInStruct(
      "__init__", /*params=*/{},
      /*paramListAttrs=*/PogListAttr::get(ctx, /*numPogs=*/0, constraints),
      /*argTypes*/ {srcArgType, selfArgType},
      /*argConvs*/ {srcConv, ArgConvention::ByRefResult}, argListAttrs,
      shared.getNoneType(),
      isMove ? SpecialFunctionKind::kMoveCtor : SpecialFunctionKind::kCopyCtor);
  if (!resultFn)
    return {};
  resultDecl->resolvedness = DeclResolvedness::signature;

  // Add a unresolved EndFnOp to the end of the function. This makes the
  // function able to verify clean, even if we don't body or signature resolve
  // it.
  auto resultAtBlockEndBuilder = OpBuilder::atBlockEnd(resultFn.getBody());
  EndFnOp::create(resultAtBlockEndBuilder, resultFn.getLoc(),
                  /*unresolved=*/true);

  // TODO: Should only do this if the type is RP or small?
  resultFn.setInlineLevel(InlineLevel::AlwaysNoDebug);
  return resultFn;
}

/// Rebind a field reference to access it through a trait downcast.
static Value rebindRefForTrait(Value fieldRef, Type fieldMLIRType,
                               TraitDeclOp traitOp, ImplicitLocOpBuilder &b) {
  assert(isa<ParamType>(fieldMLIRType) &&
         "rebindRefForTrait requires a parameterized field type");
  auto paramType = cast<ParamType>(fieldMLIRType);
  TypedAttr downcast =
      DowncastAttr::get(traitOp.getCanonicalTrait(), paramType.getParam());
  Type reboundElementType = ParamType::get(downcast);
  RefType fieldRefType = cast<RefType>(fieldRef.getType());
  return RebindOp::create(b, fieldRefType.getWithElement(reboundElementType),
                          fieldRef);
}

/// Given a function of the form
///    def __init__(out self: MyStruct, *, copy: MyStruct)
/// populate the method with the following:
///   %targetField0Ptr = lit.ref.struct.ger %self[field0]
///   %sourceField0Ptr = lit.ref.struct.ger %copy[field0]
///   copyinit_of_type_of_field0(%targetField0, %field)
LogicalResult StructEmitter::populateMoveCopy(ASTDecl &fnDecl, bool isMove) {
  // This method body resolves the decl.
  // TODO: This is because clients are directly calling this instead of having
  // declresolution do it.
  fnDecl.resolvedness = DeclResolvedness::body;

  auto fn = cast<FnOp>(fnDecl.getIfOperation());

  // We want to populate a move but the move/copy should be a method!
  SMLoc location = fnDecl.getLoc();
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(fn.getLocScope());

  // Start by emitting the return at the end of the function.  Closure emission
  // may have emitted stuff into the body of one of these functions and the
  // return needs to come at the end.
  auto endFn = cast<EndFnOp>(fn.getBody()->getTerminator());
  endFn.setUnresolved(false); // Body is resolved now.
  ImplicitLocOpBuilder b(fn.getLoc(), endFn);
  IREmitter::emitNormalReturn(b, Value(), /*emitEndFunc=*/false);

  // Generate the copy/moves of all of the elements, emit this at the start of
  // the function so it is ahead of whatever closure emission might generate.
  b = ImplicitLocOpBuilder::atBlockBegin(fn.getLoc(), fn.getBody());
  // Use fnDecl (not structDecl) so emitConstructorCall resolves overloads in
  // the fn's scope, where where-clause constraints are visible as assumptions.
  IREmitter emitter(fnDecl, b);

  assert(fn.getNumArguments() == 2 &&
         "copy and move functions should have two arguments");
  Value existingArg = fn.getBody()->getArgument(0);
  Value selfArg = fn.getBody()->getArgument(1);

  // If the value is RP trivial then we can just load and store the whole thing
  // in one shot instead of breaking it down into fields because we know all the
  // underlying copy/move operations are trivial.
  // TODO: Use memcpy for memory trivial types when we have them.
  if (structDeclOp.isRegisterPassableTrivial()) {
    Value value = LIT::RefLoadOp::create(b, existingArg);
    RefStoreOp::create(b, value, selfArg);
    return success();
  }

  // Otherwise, invoke the copy/move ctors fieldwise as appropriate.
  bool isImplicitlyCopyableStruct =
      structDecl.getTypeDeclSelf().isImplicitlyCopyable(structDecl.getLoc(),
                                                        shared, fnDecl);
  ArrayRef<ConstraintAttr> bodyConstraints =
      fn.getFuncTypeGenerator().getParamListAttrs().getBodyConstraints();

  for (StructFieldOp fieldOp : structDeclOp.getFieldDecls()) {
    ASTType fieldType = fieldOp.getType();

    // TODO: Add a nicer accessor.
    auto fieldEntries = structDecl.lookupInCurrentScope(fieldOp.getNameAttr());
    assert(fieldEntries.size() == 1 && "field decls cannot be overloaded");
    ASTDecl &fieldASTDecl = *fieldEntries[0];
    if (failed(getDeclResolver().resolveSignature(fieldASTDecl,
                                                  fieldASTDecl.getLoc())))
      return failure();

    auto targetFieldOp = RefStructGEROp::create(b, selfArg, fieldOp);
    Value srcFieldOp = RefStructGEROp::create(b, existingArg, fieldOp);

    // Set to the resolved trait when the field requires a conditional
    // conformance (i.e. a where-clause constraint proves the field's type
    // parameter conforms to the required trait).  Null otherwise.
    TraitDeclOp conditionalTraitOp;

    if (isMove) {
      // The move constructor can work with movable (preferably) or implicitly
      // copyable (as a fallback) types.
      if (!fieldType.isMovable(fieldASTDecl.getLoc(), shared, fnDecl) &&
          !fieldType.isImplicitlyCopyable(fieldASTDecl.getLoc(), shared,
                                          fnDecl)) {
        conditionalTraitOp = fieldConditionallyConformsToBuiltin(
            fieldType, "Movable", shared, structDecl, bodyConstraints);
        if (!conditionalTraitOp)
          return emitError(fieldASTDecl.getLoc())
                 << "cannot synthesize move constructor because field '"
                 << fieldOp.getName()
                 << "' has non-movable and non-implicitly-copyable type "
                 << fieldType;
      }
    } else {
      // We only synthesize copy ctor for `ImplicitlyCopyable` object iff all
      // its fields are `ImplicitlyCopyable`. That is, we won't synthesize for
      // the following struct:
      // ```
      // struct T(ImplicitlyCopyable):
      //   var f: some Copyable
      // ```
      if (!fieldType.isCopyable(fieldASTDecl.getLoc(), shared,
                                isImplicitlyCopyableStruct, fnDecl)) {
        conditionalTraitOp = fieldConditionallyConformsToBuiltin(
            fieldType,
            isImplicitlyCopyableStruct ? "ImplicitlyCopyable" : "Copyable",
            shared, structDecl, bodyConstraints);
        if (!conditionalTraitOp) {
          return emitError(fieldASTDecl.getLoc())
                 << "cannot synthesize "
                 << (isImplicitlyCopyableStruct ? "implicit " : "")
                 << "copy constructor because field '" << fieldOp.getName()
                 << "' has non-"
                 << (isImplicitlyCopyableStruct ? "implicitly-" : "")
                 << "copyable type " << fieldType;
        }
      }
    }

    if (conditionalTraitOp) {
      Value reboundTarget = rebindRefForTrait(targetFieldOp, fieldType.mlirType,
                                              conditionalTraitOp, b);
      Value reboundSrc = rebindRefForTrait(srcFieldOp, fieldType.mlirType,
                                           conditionalTraitOp, b);

      SyntheticNode expr(location);
      if (isMove) {
        CValue reboundMoveSrc = CValue(MRValue(reboundSrc));
        emitter.emitStoreToLValue({reboundMoveSrc, &expr},
                                  MLValue(reboundTarget), EC_SynthesizedMethod);
      } else {
        CValue reboundCopySrc = CValue(MBValue(reboundSrc));
        ExprDest dest(MLValue(reboundTarget), EC_SynthesizedMethod);
        ASTType refinedFieldType(
            cast<RefType>(reboundTarget.getType()).getElementType());
        CallOperands operands(CallSyntax::kImplicitCopyCtor, &expr,
                              std::move(dest));
        operands.add(StringAttr::get(shared.getContext(), "copy"),
                     {reboundCopySrc, &expr}, ArgUnpackStyle::kKeyword);
        emitter.emitConstructorCall(refinedFieldType, std::move(operands));
      }
      continue;
    }

    CValue src =
        isMove ? CValue(MRValue(srcFieldOp)) : CValue(MBValue(srcFieldOp));

    SyntheticNode expr(location);
    if (!isMove) {
      // If this a copy constructor and the field is only `Copyable` but not
      // implicitly copyable, generate the explicit call to copy ctor so
      // the rest of the compiler doesn't have to know about explicit copying.
      // We only do this when not-implicitly copyable so we don't have to deal
      // with the MLIR types.
      if (!isImplicitlyCopyableStruct &&
          !fieldType.isImplicitlyCopyable(fieldASTDecl.getLoc(), shared,
                                          fnDecl)) {
        // Invoke `T(*, copy: Self)`.
        ExprDest dest(MLValue(targetFieldOp), EC_SynthesizedMethod);
        CallOperands operands(CallSyntax::kImplicitCopyCtor, &expr,
                              std::move(dest));
        operands.add(StringAttr::get(shared.getContext(), "copy"), {src, &expr},
                     ArgUnpackStyle::kKeyword);
        (void)emitter.emitConstructorCall(fieldType, std::move(operands));
        continue;
      }
    }
    emitter.emitStoreToLValue({src, &expr}, MLValue(targetFieldOp),
                              EC_SynthesizedMethod);
  }

  SymbolConstantAttr ref = fn.getBoundSymbolRef(shared.getEvaluationContext());
  if (isMove)
    structDeclOp.setMoveInitAttr(ref);
  else
    structDeclOp.setCopyInitAttr(ref);
  return success();
}

std::optional<ValueInfo> ValueInfo::lookupExisting(ASTDecl &structDecl) {
  auto &shared = structDecl.getShared();
  ValueInfo result;
  auto find = [&](StringRef name, SpecialFunctionKind kind,
                  FnOp &member) -> LogicalResult {
    LookupResult lookupResult =
        shared.lookupAndResolveDecl(name, structDecl.getLoc(), structDecl,
                                    /*searchParentScopes=*/false);
    if (!lookupResult.isSuccess())
      return success();
    for (ASTDecl *decl : lookupResult.getIfSuccess()) {
      if (auto func = dyn_cast_or_null<FnOp>(decl->getIfOperation()))
        if (SpecialFunctionKind(func.getSpecialFnKind()) == kind)
          member = func;
    }

    return success();
  };
  if (failed(find("__deinit__", SpecialFunctionKind::kDeinit, result.del)))
    return {};
  if (failed(
          find("__init__", SpecialFunctionKind::kCopyCtor, result.copyctor)) ||
      failed(find("__init__", SpecialFunctionKind::kMoveCtor, result.movector)))
    return {};

  return result;
}

std::optional<ValueInfo> StructEmitter::addMissingValueMemberStubsToStruct(
    bool forceGenerateDestructor) {
  std::optional<ValueInfo> valueInfo = ValueInfo::lookupExisting(structDecl);
  if (!valueInfo)
    return {};

  if (!valueInfo->del && forceGenerateDestructor)
    valueInfo->del = synthesizeEmptyDtor();

  auto addCopyOrMoveBuiltinTrait = [&](StringRef traitName) {
    ASTDecl *traitDecl =
        shared.lookupBuiltinTrait(traitName, structDecl.getLoc());
    if (traitDecl) // Don't crash if the builtin trait is not found.
      addTraitParent(structDeclOp, traitDecl);
  };

  if (!valueInfo->copyctor && !structDeclOp.isRegisterPassableTrivial())
    valueInfo->copyctor = synthesizeEmptyMoveOrCopyInit(/*isMove=*/false);
  addCopyOrMoveBuiltinTrait("ImplicitlyCopyable");

  if (!valueInfo->movector && !structDeclOp.isRegisterPassable())
    valueInfo->movector = synthesizeEmptyMoveOrCopyInit(/*isMove=*/true);
  addCopyOrMoveBuiltinTrait("Movable");
  return valueInfo;
}

/// Synthesize an unresolved alias into the struct with the specified name .
ASTDecl *StructEmitter::synthesizeUnresolvedAlias(StringRef name) {
  auto paramDecl =
      ParamDeclAttr::get(name, LIT::UnresolvedType::get(getContext()));

  auto builder = ImplicitLocOpBuilder::atBlockEnd(
      structDeclOp.getLoc(), &structDeclOp.getFields().front());
  auto declOp = AliasDeclOp::create(builder, paramDecl);

  // Create an ASTDecl so it can be resolved with name lookup.
  ASTDecl &aliasDecl = getDeclResolver().addDecl(
      declOp, structDecl.getLoc(), StringAttr::get(getContext(), name),
      &structDecl, LexerCursor(), LexerCursor(), /*indentation=*/0);
  aliasDecl.resolvedness = DeclResolvedness::unparsed;
  return &aliasDecl;
}

TypedAttr StructEmitter::populateSpecialFnIsTrivial(SpecialFunctionKind kind) {
  StringRef baseName;
  StringRef traitName;
  switch (kind) {
  case SpecialFunctionKind::kDeinit:
    baseName = "__del__";
    traitName = "Deinitable";
    break;
  case SpecialFunctionKind::kCopyCtor:
    baseName = "__copy_ctor_";
    traitName = "Copyable";
    break;
  case SpecialFunctionKind::kMoveCtor:
    baseName = "__move_ctor_";
    traitName = "Movable";
    break;
  default:
    llvm_unreachable("unknown synthesized alias");
  }

  IREmitter emitter(structDecl, EC_AliasValue);
  // NOTE: we have to first synthesize the bit to `i1` (instead of `Bool`) to
  // avoid signature resolving `Bool::__init__`s, the implicit conversion will
  // be taken care of when body resolve conformanceOp.
  auto emitBoolAttr = [&](BoolAttr v) -> TypedAttr {
    SyntheticNode node(structDecl.getLoc());
    return emitter.emitBool({v, &node}, EC_OperatorOperandValue).getIfPValue();
  };

  StringRef spName =
      kind == SpecialFunctionKind::kDeinit ? "__deinit__" : "__init__";
  LookupResult spDecls =
      shared.lookupAndResolveDecl(spName, structDecl.getLoc(), structDecl,
                                  /*searchParentScope=*/false);
  if (spDecls.isErroneous())
    return nullptr;
  for (ASTDecl *decl : spDecls.getIfSuccess()) {
    // Skip disabled ASTDecls which have null operations.
    if (decl->isDisabled())
      continue;
    auto fnOp = dyn_cast<FnOp>(decl->getIfOperation());
    if (!fnOp)
      continue;
    if (fnOp.getSpecialFunctionKind() == kind) {
      // If has a user provided implementation, consider them as non-trivial.
      if (!decl->getCursor().isInvalid())
        return emitBoolAttr(BoolAttr::get(emitter.getContext(), false));
    }
  }

  // When forming a&b&a we can just treat subsequent uses of 'a' as true.
  SmallPtrSet<Attribute, 4> seenExprs;
  auto getBoolConstant = [&](CValue value) -> std::optional<bool> {
    SyntheticNode node(structDecl.getLoc());
    PValue i1 = emitter.emitScalarBool({value, &node}, EC_OperatorOperandValue)
                    .getIfPValue();
    if (SIMDAttr asIntAttr = sugarDynCastIfPresent<SIMDAttr>(i1.get()))
      return asIntAttr.getAsBool();
    // No need to double check the same value. This crushes sugar bloat.
    if (!seenExprs.insert(i1.get()).second)
      return true;
    return {};
  };

  // This emits an "and" as a PValue expression, maintaining the type of lhs/rhs
  // (which are Bool) instead of turning them into i1.
  auto emitAnd = [&](CValue lhs, CValue rhs) -> CValue {
    SyntheticNode node(structDecl.getLoc());
    // Short circuit obvious cases to avoid piling up sugar.
    if (std::optional<bool> lhsI1 = getBoolConstant(lhs)) {
      if (*lhsI1)
        return rhs;
      return lhs;
    }
    if (std::optional<bool> rhsI1 = getBoolConstant(rhs)) {
      if (*rhsI1)
        return lhs;
      return rhs;
    }

    ExprDest dest(EC_OperatorOperandValue);
    return emitter.emitNamedMethodCall(
        "__and__", CallOperands(CallSyntax::kOperator, &node, std::move(dest),
                                {{lhs, &node}, {rhs, &node}}));
  };

  ASTDecl *traitDecl =
      shared.lookupBuiltinTrait(traitName, structDecl.getLoc());
  auto witnessName =
      StringAttr::get(getContext(), Twine(baseName) + "is_trivial");
  auto traitSymbol = TraitSymbolAttr::get(traitDecl->getSymbolRef());

  CValue ret = emitBoolAttr(BoolAttr::get(emitter.getContext(), true));
  if (!ret.getIfPValue())
    return nullptr;
  for (StructFieldOp fieldOp : structDeclOp.getFieldDecls()) {
    // TODO: Add a nicer accessor.
    auto fieldEntries = structDecl.lookupInCurrentScope(fieldOp.getNameAttr());
    assert(fieldEntries.size() == 1 && "field decls cannot be overloaded");
    ASTDecl &fieldASTDecl = *fieldEntries[0];
    if (failed(getDeclResolver().resolveSignature(fieldASTDecl,
                                                  fieldASTDecl.getLoc())))
      return nullptr;

    if (ASTType(fieldOp.getType())
            .isTrivialRegisterType(fieldASTDecl.getLoc(), shared))
      continue; // skip MLIR types and TrivialRegisterPassable types.

    TypedAttr fieldIsTrivial =
        shared.getEvaluationContext().getAndFold<GetWitnessAttr>(
            PValue(fieldOp.getType()), traitSymbol, witnessName, ret.getType());

    ret = emitAnd(ret, fieldIsTrivial);
  }

  return ret.getIfPValue();
}
