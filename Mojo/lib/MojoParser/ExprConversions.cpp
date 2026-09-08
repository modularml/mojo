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
// This file contains implementation details of IREmitter that are related to
// value conversions.
//
//===----------------------------------------------------------------------===//

#include "ClosureEmitter.h"
#include "ExprNodes.h"
#include "IREmitter.h"
#include "InferenceState.h"

#include "OverloadSet.h"

#include "Mojo/MojoParser/Constraints.h"
#include "MojoUtils.h"
#include "ParamMatcher.h"
#include "ParserEvaluationContext.h"
#include "SpecializeInf.h"
#include "StructEmitter.h"
#include "Traits.h"

#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/LITDialect/LITAttrs.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/MojoParser/ASTDecl.h"
#include "Mojo/MojoParser/ASTType.h"
#include "Mojo/MojoParser/DeclResolver.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Support/Compiler/OperationUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/Support/xxhash.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// Function Conversions
//===----------------------------------------------------------------------===//

// Strips references from the expected and actual types, reconciling allowed
// differences and extracting the pointee types to compare.
bool checkConventionsConvertible(ArgConvention expectedConv,
                                 ArgConvention actualConv) {
  // DeinitMem is the same as OwnedMem, so we can convert between them.
  if (expectedConv == ArgConvention::DeinitMem)
    expectedConv = ArgConvention::OwnedMem;
  if (actualConv == ArgConvention::DeinitMem)
    actualConv = ArgConvention::OwnedMem;

  // Check the argument convention, reconciling allowed differences and
  // extracting the actual type to compare. This also doesn't check for
  // passing convention, since those are trivially convertible.
  switch (expectedConv) {
  case ArgConvention::OwnedReg:
    llvm_unreachable("not used by the mojo parser");
  case ArgConvention::ByRefError:
    // We checked that the function effects line up, so if we see
    // `byref_error`, then the other function must have it as well.
    assert(actualConv == ArgConvention::ByRefError &&
           "both functions must be throwing");
    [[fallthrough]];
  case ArgConvention::OwnedMem:
  case ArgConvention::MutRef:
  case ArgConvention::Ref:
  case ArgConvention::Mut:
    if (actualConv == ArgConvention::ImmMem) {
      // If the actual function accepts a read reference, and we have an
      // owned/mutref/ref/mut, we can make a thunk to convert those nicely.
    } else if (actualConv == ArgConvention::ImmReg) {
      // If the actual function accepts a register-passable read, and we have
      // an owned/mutref/ref/mut, we can make a thunk to convert that nicely.
    } else if (actualConv == expectedConv) {
      // Exactly equal, so can convert easily.
    } else {
      return false; // Otherwise, we can't convert.
    }
    break;

  case ArgConvention::ImmMem:
  case ArgConvention::ImmReg:
    if (!llvm::is_contained({ArgConvention::ImmMem, ArgConvention::ImmReg},
                            actualConv))
      return false;
    break;

  case ArgConvention::DeinitMem:
  case ArgConvention::ByRefResult:
    llvm_unreachable("`byref_result` was already handled");
  }

  return true;
}

// TODO: Return more than a boolean, so we can have better error messages.
static bool canConvertFunctionTypes(FnTypeGeneratorType actualGen,
                                    FnTypeGeneratorType expectedGen,
                                    const ExprNode *expr, ASTDecl &declScope) {
  ParamBindings bindings(declScope, expr);
  SpecializeInf paramInf(declScope, expr, /*no params to infer*/ {},
                         PogListAttr::get(declScope.getContext(), {}),
                         expr->getLoc(), /*discardError=*/true);

  ParamMatcher matcher(expr, paramInf, /*allowImplicitConversions=*/true);
  return succeeded(matcher.matchFunctionTypes(actualGen, expectedGen));
}

//===----------------------------------------------------------------------===//
// Generator body-constraint discharge
//===----------------------------------------------------------------------===//

/// Prove `from`'s generator body constraints under `declScope`, treating
/// `to`'s body constraints and `additionalAssumptions` as extra facts. It
/// only checks the constraints, not the body types of the generators. When
/// `unprovenConstraints` is non-null an `unknown` verdict fills it with the
/// constraints that lacked evidence.
static TriBool
canProveBodyConstraints(GeneratorType from, GeneratorType to,
                        ASTDecl &declScope,
                        ArrayRef<ConstraintAttr> additionalAssumptions,
                        SmallVectorImpl<ConstraintAttr> *unprovenConstraints) {
  ArrayRef<ConstraintAttr> fromConstraints = from.getBodyConstraints();
  if (fromConstraints.empty())
    return TriBool::yes();

  // `additionalAssumptions` lets callers supply facts that hold in their
  // context but are not in `declScope`'s known assumptions; `to`'s constraints
  // join them because a value of that type is only ever reached where they
  // hold.
  SmallVector<ConstraintAttr> assumptions(additionalAssumptions);
  llvm::append_range(assumptions, to.getBodyConstraints());

  auto fromParamList = cast<PogListAttr>(from.getParamListAttrs());
  OptionalDiag diag(declScope.getShared(), declScope.getLoc(),
                    /*discardError=*/true);
  return canDischargeConstraintsInScope(declScope, fromParamList,
                                        fromConstraints, fromConstraints,
                                        diag.getDiag(), unprovenConstraints,
                                        /*evaluator=*/nullptr, assumptions);
}

static TriBool
canConvertGeneratorTypes(ASTExprAnd<CValue> valueExpr, GeneratorType actual,
                         GeneratorType expected, ASTDecl &declScope,
                         ArrayRef<ConstraintAttr> additionalAssumptions = {}) {
  // Tentatively disallow converting from constrained generators when bodies
  // are different. When bodies are the same, `canZeroCostConvert` will have
  // already allowed it.
  if (!actual.getBodyConstraints().empty())
    return TriBool::no();

  // Handle function conversions.
  if (auto actualFnType = sugarDynCast<FnTypeGeneratorType>(actual))
    if (auto expectedFnType = sugarDynCast<FnTypeGeneratorType>(expected)) {
      return TriBool::fromBool(canConvertFunctionTypes(
          actualFnType, expectedFnType, valueExpr.expr, declScope));
    }

  if (auto actualType = sugarDynCast<FnLiteralTypeGeneratorType>(actual)) {
    if (auto expectedType = sugarDynCast<FnTypeGeneratorType>(expected)) {
      // See if the literal itself has a compatible type.
      return TriBool::fromBool(
          canConvertFunctionTypes(actualType.getSymbolConstantAttr().getType(),
                                  expectedType, valueExpr.expr, declScope));
    }
  }

  // Generators with different parameterization cannot be converted between each
  // other. If the types are equal but the passing conventions are different,
  // then the conversion is allowed.
  // TODO: Consider default parameter values and enable parameter inference to
  // reconcile differences.
  if (actual.getInputParamTypes() != expected.getInputParamTypes())
    return TriBool::no();

  // We are pulling out the body of the generator to test type convertibility.
  // To do it correctly, we need to replace index ref to name refs. Otherwise,
  // it confuses parameter inference (as index refs are to be inferred).
  ParamRefRemapper remapper;
  for (size_t i = 0, e = actual.getInputParamTypes().size(); i != e; ++i) {
    remapper.parameters.push_back(
        StringAttr::get(actual.getContext(), "Ctx#" + Twine(i)));
  }

  // Otherwise, the bodies must be convertible. This is possible if we can get
  // the body, meaning the value must be a GeneratorAttr.
  auto genAttr =
      sugarDynCastIfPresent<GeneratorAttr>(valueExpr.ir.getIfPValue().get());
  if (!genAttr)
    return TriBool::no();

  return IREmitter::classifyImplicitConversion(
      {remapper.replace(genAttr.getBody()), valueExpr.expr},
      ASTType(remapper.replace(expected.getBody())), declScope,
      additionalAssumptions);
}

/// Return `metadata` with its capture origin set cleared. Capture origins
/// describe which storage a callable's captures point into, which does not
/// affect its representation, so signature comparisons that only care about
/// representation compare the metadata with them cleared.
static FnMetadataAttr withoutCaptureOrigins(FnMetadataAttr metadata) {
  auto originData = cast_or_null<FnMetaOriginDataAttr>(metadata.getMetadata());
  if (!originData)
    return metadata;
  return metadata.getWithMetadata(FnMetaOriginDataAttr::get(
      metadata.getContext(), originData.getNumImplicitOriginDecls(),
      /*captureOrigins=*/{}, originData.getIsNestedOriginsReadOnly(),
      originData.getDefinesInteriorOrigins()));
}

// Strip out irrelevant details of a function that can be rebound away to make
// convertibility checking easier.
static FuncType getReducedFnType(FuncType sig) {
  MLIRContext *ctx = sig.getContext();

  auto origPogListAttr = sig.getArgListAttrs();

  SmallVector<PassingKind> passingKinds;
  SmallVector<StringAttr> names;
  SmallVector<VariadicKind> variadics;
  SmallVector<TypedAttr> defaults(sig.getNumArguments(), {});
  for (size_t i = 0, e = sig.getNumArguments(); i != e; ++i) {
    passingKinds.push_back(origPogListAttr.getPassingKind(i));
    names.push_back(origPogListAttr.getName(i));
    variadics.push_back(origPogListAttr.getVariadicKind(i));
  }

  auto newPogListAttr =
      PogListAttr::get(ctx, names, passingKinds, variadics, defaults,
                       origPogListAttr.getOrigVariadicConvention(),
                       origPogListAttr.getBodyConstraints());

  FnMetadataAttr metadata = withoutCaptureOrigins(sig.getMetadataAttr());
  return FuncType::get(ctx, sig.getValues(), metadata, newPogListAttr);
}

static GeneratorType getReducedGeneratorType(GeneratorType gen) {
  // If the body is a function, we can further reduce it.
  Type bodyType = gen.getBody();
  if (auto fnType = sugarDynCast<FuncType>(bodyType))
    bodyType = getReducedFnType(fnType);

  ArrayRef<ConstraintAttr> bodyConstraints;
  if (PogListAttr pogs = gen.getParamListAttrs())
    bodyConstraints = pogs.getBodyConstraints();
  auto metadata = PogListAttr::get(
      gen.getContext(), gen.getInputParamTypes().size(), bodyConstraints);
  return GeneratorType::get(gen.getInputParamTypes(), bodyType, metadata);
}

static std::string generateThunkName(Type expected, Type actual) {
  std::string name;
  llvm::raw_string_ostream os(name);
  ASTType(expected).print(os, /*diags=*/{});
  os << '|';
  ASTType(actual).print(os, /*diags=*/{});

  // Mix in the full signatures to disambiguate.
  std::string sigHash;
  llvm::raw_string_ostream sigHashOs(sigHash);
  expected.print(sigHashOs);
  actual.print(sigHashOs);
  os << '|';
  os << llvm::utohexstr(llvm::xxh3_64bits(sigHash),
                        /*LowerCase=*/true, /*Width=*/16);
  return name;
}

static FnOp generateConversionThunk(Attribute key, ASTDecl &moduleDecl,
                                    SMLoc useLoc) {
  auto &shared = moduleDecl.getShared();
  // Don't generate any debuginfo for the thunk. Push a null scope.
  DebugInfo::DIBuilder::ScopeGuard diScopeGuard;
  if (shared.diBuilder)
    diScopeGuard = shared.diBuilder->pushScopeGuard(/*scope=*/nullptr);

  auto keyValues = cast<ArrayAttr>(key);
  // The actual signature may be wrapped in a GeneratorType that provides the
  // scope for clarifying parameter index references. Unwrap if needed.
  Type keyActualType = cast<TypeAttr>(keyValues[0]).getValue();
  auto actualSignature = dyn_cast<FnTypeGeneratorType>(keyActualType);
  if (!actualSignature)
    actualSignature =
        cast<FnTypeGeneratorType>(cast<GeneratorType>(keyActualType).getBody());
  auto thunkSignature =
      cast<FnTypeGeneratorType>(cast<TypeAttr>(keyValues[1]).getValue());

  MLIRContext *ctx = shared.getContext();
  Location mlirLoc = shared.translateLocation(moduleDecl.getLoc());

  // Declare a function with expected function type. Add the parameters from the
  // expected signature. This contains the types of the clarifying parameters
  // (see TAPCPTTT) and the actual function's input parameters.
  SmallVector<ParamDeclAttr> paramDecls;
  SmallVector<TypedAttr> paramValues;
  ParameterEvaluator evaluator = shared.getParameterEvaluator();
  ImplicitLocOpBuilder b(mlirLoc, ctx);
  for (auto [idx, type] :
       llvm::enumerate(thunkSignature.getInputParamTypes())) {
    // The parameter names are derived from the decl name.
    paramDecls.push_back(
        ParamDeclAttr::get(moduleDecl.mangleUserDefinedParamName(
                               b.getStringAttr("_" + Twine(idx))),
                           evaluator.getReboundType(type)));
    paramValues.push_back(ParamDeclRefAttr::get(paramDecls.back()));
    evaluator.appendIndexBinding(paramValues.back());
  }
  // Rebind the argument and result types into the scope of the body.
  FunctionType functionType =
      thunkSignature
          .getSpecializedGenerator(paramValues, &shared.getEvaluationContext())
          .getBody()
          .getValues();

  // Add an additional parameter, representing the actual callee. Rebind the
  // actual function type into the scope of the body.
  auto calleeDecl = ParamDeclAttr::get(
      moduleDecl.mangleUserDefinedParamName(b.getStringAttr("callee")),
      evaluator.getReboundType(actualSignature));
  paramDecls.push_back(calleeDecl);

  // Generate a mangled name.
  std::string name = generateThunkName(thunkSignature, actualSignature);

  // Extract the callee's where-clause constraints from the rebound callee
  // type. The evaluator has already remapped index-based parameter references
  // to named references using the thunk's parameter declarations, so these
  // constraints can be used as known assumptions in the thunk's scope.
  // This is needed for TrivialRegisterPassable types with conditional
  // conformance: the witness entry uses a conversion thunk to bridge calling
  // conventions, and the callee (struct method) may carry a where clause.
  auto reboundCalleeType =
      sugarCast<FnTypeGeneratorType>(evaluator.getReboundType(actualSignature));
  ArrayRef<ConstraintAttr> remappedConstraints =
      reboundCalleeType.getParamListAttrs().getBodyConstraints();

  // Declare the function at the bottom of the decl.
  b = ImplicitLocOpBuilder(mlirLoc, moduleDecl.getDeclEndBuilder());
  FunctionEmitter structEmitter(shared);
  auto paramListAttrs = PogListAttr::get(
      ctx, thunkSignature.getInputParamTypes().size() + 1, remappedConstraints);
  auto [thunk, thunkDecl] = structEmitter.synthesizeFunction(
      moduleDecl, name, paramDecls, paramListAttrs, functionType.getInputs(),
      thunkSignature.getArgConventions(),
      PogListAttr::get(ctx, thunkSignature.getNumArguments()),
      functionType.getResults().front(), SpecialFunctionKind::kNormal,
      moduleDecl.getLoc(), b, thunkSignature.getFnEffects(),
      /*suffix=*/"", /*synthetic=*/true, InlineLevel::Automatic);

  // Annotate the function as a thunk by adding the conversion types.
  NamedAttrList attrs = thunk->getAttrDictionary();
  attrs.set(thunk.getThunkKeyAttrName(), key);

  // Always inline the thunk. The calling convention conversion overhead is
  // guaranteed to be optimized away.
  attrs.set(thunk.getInlineLevelAttrName(),
            InlineLevelAttr::get(ctx, InlineLevel::AlwaysNoDebug));

  // Set the attributes.
  thunk->setAttrs(attrs.getDictionary(ctx));

  // Now prepare to emit the call.
  b = ImplicitLocOpBuilder::atBlockBegin(mlirLoc, thunk.getBody());
  IREmitter emitter(*thunkDecl, b);

  // Construct the call operands from the function block arguments.
  SyntheticNode node(useLoc);
  CallOperands operands(CallSyntax::kMethodCall, &node, EC_ConversionThunk);

  std::optional<size_t> thunkVariadicArgIndexOpt =
      thunkSignature.findPackVarArgIndex();
  std::optional<size_t> actualVariadicArgIndexOpt =
      actualSignature.findPackVarArgIndex();

  ArrayRef<Type> actualArgTypes = actualSignature.getArguments().drop_back(
      actualSignature.hasMemoryOnlyResult() + actualSignature.isThrows());
  ArrayRef<Type> thunkArgTypes = thunkSignature.getArguments().drop_back(
      thunkSignature.hasMemoryOnlyResult() + thunkSignature.isThrows());
  std::optional<size_t> actualKwVarArgIndex;
  std::optional<size_t> thunkKwVarArgIndex;
  if (!actualArgTypes.empty() &&
      actualSignature.isKwVarArg(actualArgTypes.size() - 1))
    actualKwVarArgIndex = actualArgTypes.size() - 1;
  if (!thunkArgTypes.empty() &&
      thunkSignature.isKwVarArg(thunkArgTypes.size() - 1))
    thunkKwVarArgIndex = thunkArgTypes.size() - 1;
  assert(actualKwVarArgIndex.has_value() == thunkKwVarArgIndex.has_value() &&
         "function conversion must preserve kwargs");

  for (size_t actualArgIndex = 0; actualArgIndex < actualArgTypes.size();
       actualArgIndex++) {
    bool actualArgIsKwVarArg =
        actualKwVarArgIndex && actualArgIndex == *actualKwVarArgIndex;
    bool actualArgIsVariadicPack =
        actualVariadicArgIndexOpt && thunkVariadicArgIndexOpt &&
        actualVariadicArgIndexOpt == thunkVariadicArgIndexOpt &&
        actualArgIndex == *actualVariadicArgIndexOpt;
    bool actualArgIsForVariadic =
        !actualArgIsKwVarArg && thunkVariadicArgIndexOpt.has_value() &&
        actualArgIndex >= thunkVariadicArgIndexOpt.value();

    Value argForActual;
    KGEN::ArgConvention convForActual;
    if (actualArgIsKwVarArg) {
      argForActual = thunk.getArgument(*thunkKwVarArgIndex);
      convForActual = thunkSignature.getArgConvention(*thunkKwVarArgIndex);
    } else if (actualArgIsVariadicPack) {
      argForActual = thunk.getArgument(*thunkVariadicArgIndexOpt);
      convForActual =
          thunkSignature.getArgConvention(*thunkVariadicArgIndexOpt);
    } else if (actualArgIsForVariadic) {
      size_t thunkVariadicArgIndex = thunkVariadicArgIndexOpt.value();
      size_t indexInVariadic = actualArgIndex - thunkVariadicArgIndex;

      MBValue packRefMBValue =
          MBValue(thunk.getArgument(thunkVariadicArgIndex));

      // Emit: the_pack[index]
      auto indexAttr = IntegerAttr::get(IndexType::get(ctx), indexInVariadic);
      CValue indexCValue = emitter.emitInt(
          ASTExprAnd<PValue>{PValue(indexAttr), &node}, EC_ConversionThunk);
      SyntheticNode indexSynthNode(useLoc, indexCValue);
      SyntheticNode packSynthNode(useLoc, packRefMBValue);
      Operand subscriptOperand(&indexSynthNode, useLoc,
                               ArgUnpackStyle::kKeyword,
                               StringAttr::get(ctx, "index"));
      SubscriptNode packSubscriptNode(&packSynthNode, useLoc, subscriptOperand,
                                      useLoc);
      CValue getItemResult =
          emitter.emitExprCValue(&packSubscriptNode, EC_ConversionThunk);
      if (!getItemResult)
        return {};
      argForActual = getItemResult.getMlirValue();
      convForActual = ArgConvention::ImmMem;
    } else {
      argForActual = thunk.getArgument(actualArgIndex);
      convForActual = thunkSignature.getArgConvention(actualArgIndex);
    }

    AnyValue value;
    switch (convForActual) {
    case ArgConvention::OwnedReg:
      llvm_unreachable("not used by the mojo parser");
    case ArgConvention::ByRefResult:
    case ArgConvention::ByRefError:
      continue; // Ignore this, it will be assigned to later.

    case ArgConvention::Mut:
    case ArgConvention::MutRef:
      value = MLValue(argForActual);
      break;
    case ArgConvention::OwnedMem:
    case ArgConvention::DeinitMem:
      value = MRValue(argForActual);
      break;
    case ArgConvention::ImmReg:
      value = SRValue(argForActual);
      break;
    case ArgConvention::ImmMem:
      value = MBValue(argForActual);
      break;
    case ArgConvention::Ref:
      value = MBPValue(argForActual);
      break;
    }

    if (actualArgIsKwVarArg) {
      operands.add({value, &node}, ArgUnpackStyle::kStarStar);
      continue;
    }
    if (actualArgIsVariadicPack) {
      operands.add({value, &node}, ArgUnpackStyle::kStar);
      continue;
    }

    // Pass any required-keyword args with a name.
    StringAttr name;
    if (!actualArgIsForVariadic &&
        thunkSignature.getArgListAttrs().getPassingKind(actualArgIndex) ==
            PassingKind::KwOnly)
      name = thunkSignature.getArgName(actualArgIndex);
    operands.add(name, {value, &node},
                 name ? ArgUnpackStyle::kKeyword : ArgUnpackStyle::kPositional);
  }

  // Allocate the value dest for the call. Set the value dest to the result
  // slot, if there is one, otherwise provide the expected rvalue type.
  ExprDest dest(EC_ConversionThunk);
  bool hasRegisterResult = false;
  if (thunkSignature.isAsync()) {
    // An async call returns a coroutine we have to await.
  } else if (thunkSignature.hasMemoryOnlyResult()) {
    dest = ExprDest(MLValue(thunk.getArguments().back()), EC_ConversionThunk);
  } else {
    hasRegisterResult = true;
  }

  // Bind the function parameters declared on the thunk to the callee. This does
  // NOT include the clarifying parameters -- the callee has already been
  // rebound to them when it was declared on the parameter list.
  //
  // In this example (from TAAMCE):
  //
  //     def ship_func_thunk[
  //         Z: int,
  //         Y: Bool,
  //         callee: def[Y: Bool](read Ship[Z])->None
  //     ](mut s: Ship[Z, Y]):
  //         callee[Y](s) # implicit cast to imm
  //
  // notice how we're calling `callee[Y](s)` and the clarifying parameter Z
  // doesn't appear on that call line.
  TypedAttr calleeGenerator = ParamDeclRefAttr::get(calleeDecl);
  ArrayRef<TypedAttr> calleeParamValues =
      ArrayRef(paramValues)
          .take_back(actualSignature.getInputParamTypes().size());
  TypedAttr calleeParam =
      BindParamsAttr::get(calleeGenerator.getContext(), calleeGenerator,
                          calleeParamValues, &shared.getEvaluationContext());
  assert(sugarCast<FnTypeGeneratorType>(calleeParam.getType())
             .getInputParamTypes()
             .size() == 0);

  // Stash the fully-bound-when-thunk-is-bound callee expression on the thunk so
  // we can later recover the wrapped function's SymbolConstantAttr with all its
  // compile-time params filled in (sourced from the thunk's bound paramValues).
  thunk->setAttr(kTransparentThunkCalleeExprAttr, calleeParam);

  // EXPLICIT-COPY-REF-RETURN: If the callee has a ref result and we expect a
  // value result, then we need to copy out of the ref into the value result. As
  // a (very sad) hack, we need to allow explicitly copyable types (not just
  // implicitly) for __next__ in iterators to work.
  // TODO: Eliminate this when __next__ can return references and we have
  // stronger ref result and corresponding iterator traits.  This isn't
  // something we want to support in general.
  bool needsExplicitCopyOut = false;
  ExprDest explicitCopyOutDest(dest.getContext());
  if (actualSignature.isRefResult() && !thunkSignature.isRefResult()) {
    explicitCopyOutDest = std::move(dest);
    needsExplicitCopyOut = true;
  }

  operands.dest = std::move(dest);
  CValue callResult =
      emitter.emitIndirectCall(PValue(calleeParam), std::move(operands));
  // If we need an explicit copy out, emit a call to T(copy=) on the result into
  // the ultimate dest.
  if (needsExplicitCopyOut) {
    CallOperands operands(CallSyntax::kImplicitCopyCtor, &node,
                          std::move(explicitCopyOutDest));
    operands.add(StringAttr::get(shared.getContext(), "copy"),
                 {callResult, &node}, ArgUnpackStyle::kKeyword);
    callResult = emitter.emitConstructorCall(callResult.getRValueType(),
                                             std::move(operands));
  }

  // If the callee is async, we got a coroutine. Now await it into the result.
  if (thunkSignature.isAsync()) {
    ExprDest dest(MLValue(thunk.getArguments().back()), EC_ConversionThunk);
    if (!emitter.emitNamedMethodCall(
            "__await__", CallOperands(CallSyntax::kMethodCall, &node,
                                      std::move(dest), {{callResult, &node}})))
      return {};
  }

  // Emit the function return. It's just a none return if the function has a
  // result slot.
  Value retVal;
  if (hasRegisterResult) {
    // If we're returning an SSA value, it could be an RValue or could be a
    // ref-result.
    if (thunkSignature.isRefResult()) {
      retVal = emitter.emitRefValue({callResult, &node}, EC_ConversionThunk);
      // Implicitly convert to the right result type if needed, e.g. widening
      // the origin to a union or discarding mutability.
      retVal = emitter.emitSRValue({SRValue(retVal), &node}, EC_ConversionThunk,
                                   thunk.getUserResultType());
    } else {
      retVal = emitter.emitSRValue({callResult, &node}, EC_ConversionThunk);
    }
    if (!retVal)
      return {};
  }

  emitter.emitNormalReturn(mlirLoc, retVal);
  return thunk;
}

static CValue convertFunctionGeneratorValue(CValue value, const ExprNode *expr,
                                            FnTypeGeneratorType expected,
                                            IREmitter &emitter,
                                            ExprDest &dest) {
  PValue callee = value.getIfPValue();
  if (!callee) {
    emitter.emitError(
        expr->getLoc(),
        "TODO: function type conversions between closures not supported yet")
        << expr->getRange();
    dest.resetForError(emitter);
    return {};
  }

  if (auto funcLiteralType =
          sugarDynCast<FuncLiteralTypeGeneratorType>(callee.getType())) {
    // Simply convert the literal itself, call the top-most conversion API,
    // because this could be a zero cost conversion without needing to generate
    // a thunk.
    return emitter.emitImplicitConversionToType(
        {PValue(funcLiteralType.getSymbolConstantAttr()), expr}, expected,
        dest);
  }

  // Strip all sugar so we don't bind parameters wrong.
  // TODO: We could improve this to maintain sugar better.
  callee = getCanonicalAttr(callee.get());
  expected = cast<FnTypeGeneratorType>(getCanonicalType(expected));

  MLIRContext *ctx = expected.getContext();
  auto actual = sugarCast<FnTypeGeneratorType>(callee.getType());

  // Canonicalize the function types. This strips away unnecessary metadata that
  // does not affect the conversion semantics. In other words, a function type
  // and its reduced type can be trivially converted with a rebind.
  auto reducedActual =
      sugarCast<FnTypeGeneratorType>(getReducedGeneratorType(actual));
  auto reducedExpected =
      sugarCast<FnTypeGeneratorType>(getReducedGeneratorType(expected));

  // We need to specially handle when `actual` mentions any parameters in its
  // scope, like how `= read_ship[Z]` mentions the `Z` parameter here:
  //
  //     struct Ship[X: int, Y: Bool]:
  //         pass
  //
  //     def read_ship[X: int, Y: Bool](read s: Ship[X, Y]):
  //         pass
  //
  //     def foo():
  //         alias Z: int = 42
  //         alias my_func_alias: def[Y: Bool](mut Ship[Z, Y]) -> None =
  //             read_ship[Z]
  //
  // `read_ship[Z]`s type is `def(read Ship[Y: Bool][Z])`. However, when our
  // thunk accepts that type as an input parameter, the thunk is malformed
  // because it has no idea what `ZC` is (see TAPRCT for more).
  //
  // So, we prepend a "clarifying" parameter to the thunk's input parameters,
  // like the `Z` here:
  //
  //     def ship_func_thunk[
  //         Z: int,
  //         Y: Bool,
  //         callee: def[Y: Bool](read Ship[Z])->None
  //     ](mut s: Ship[Z, Y]):
  //         callee[Y](s) # implicit cast to imm
  //
  // See TAPCPTTT for more.

  SmallVector<Type> thunkParamTypes;
  // `mentionedParamRefs` contains all of `actual`'s mentions of parameters from
  // the containing scope, like the `Z` in the above `read_ship[Z]`.
  // This *only* refers to parameters declared in/by `foo`.
  llvm::SmallSetVector<ParamDeclRefAttr, 4> mentionedParamRefs;
  // NOTE: The walk here to determine the parameter mentions only works if the
  // walk visits types in the same order as lexical parsing. This is because the
  // mentioned parameters can depend on each other, so the list has to have them
  // in an order that keeps the dependencies valid.
  // I *think* we don't need to walk `expected` too... I could be wrong though.
  getCanonicalType(actual).walk(
      [&](ParamDeclRefAttr ref) { mentionedParamRefs.insert(ref); });
  // This replacer will help us figure out the thunk's param types, so the thunk
  // signature has a correct:
  //     mut s: Ship[ship_func_thunk's Z]
  // instead of an incorrect:
  //     mut s: Ship[foo's Z]
  // It also helps us generate some more general signatures for the thunk keys.
  ParameterEvaluator paramRefsReplacer = emitter.shared.getParameterEvaluator();
  for (auto [i, ref] : llvm::enumerate(mentionedParamRefs)) {
    // Add these mentioned param refs as "clarifying" parameters to the thunk,
    // see TAPCPTTT.
    thunkParamTypes.push_back(paramRefsReplacer.getReboundType(ref.getType()));
    paramRefsReplacer.setDeclBinding(
        ref.getName(), ParamIndexRefAttr::get(i, thunkParamTypes.back()));
  }
  auto reparamActualForThunkKey = sugarCast<FnTypeGeneratorType>(
      paramRefsReplacer.getReboundType(reducedActual));
  // Above, clarifying parameters were at the beginning (and were replaced with
  // `*(0,i) where i < N`).
  //
  // Now, we need to add `expected`'s input params, like the `[Y: Bool]` in:
  //
  //     alias my_func_alias: def[Y: Bool](mut Ship[Z, Y]) -> None = ...
  //
  // Note that `expected` contains param refs to parameters declared in/by foo.
  // `expected` does NOT contain paramrefs referring to the callee's function
  // definition's parameters.
  for (auto [i, type] : llvm::enumerate(expected.getInputParamTypes())) {
    // Note that `type` might contain UnboundAttr at this point, that's fine.
    thunkParamTypes.push_back(paramRefsReplacer.getReboundType(type));
    paramRefsReplacer.appendIndexBinding(ParamIndexRefAttr::get(
        i + mentionedParamRefs.size(), thunkParamTypes.back()));
  }
  // The thunk metadata and function type will mostly look like `expected`,
  // except for the thunk param types (which also includes clarifying
  // parameters, see TAPCPTTT).
  auto thunkMetadata = FnMetaOriginDataAttr::get(
      ctx, reducedExpected.getNumImplicitOriginDecls(),
      reducedExpected.getCaptureOrigins(),
      reducedExpected.getIsNestedOriginsReadOnly(),
      reducedExpected.getDefinesInteriorOrigins());
  auto thunkFuncType = sugarCast<FunctionType>(
      paramRefsReplacer.getReboundType(reducedExpected.getValues()));
  SmallVector<ConstraintAttr> thunkBodyConstraints;
  for (ConstraintAttr constraint :
       reducedExpected.getParamListAttrs().getBodyConstraints()) {
    thunkBodyConstraints.push_back(cast<ConstraintAttr>(
        paramRefsReplacer.getReboundAttribute(constraint)));
  }
  auto thunkSignature = FuncTypeGeneratorType::get(
      /*inputParamTypes=*/thunkParamTypes,
      /*values=*/thunkFuncType,
      /*argConvs=*/reducedExpected.getArgConventions(),
      /*effects=*/reducedExpected.getFnEffects(),
      /*fnMetadata=*/thunkMetadata,
      /*genMetadata=*/
      PogListAttr::get(ctx, thunkParamTypes.size(), thunkBodyConstraints),
      /*argListAttrs=*/reducedExpected.getArgListAttrs());

  // There shouldn't be any ParamDeclRefAttr in the thunk signature, because
  // there's no parent scope param-decls for them to refer to.
#ifndef NDEBUG
  getCanonicalType(thunkSignature).walk([&](ParamDeclRefAttr ref) {
    assert(false);
  });
#endif

  // We can attempt to generate the thunk now.
  // When there are clarifying parameters, `reparamActualForThunkKey` contains
  // depth-1 index references that refer to those parameters. Wrap it in a
  // GeneratorType whose inputParamTypes are the clarifying types so that the
  // depth-1 refs have a valid enclosing scope and don't escape.
  Type keyActualType = reparamActualForThunkKey;
  if (!mentionedParamRefs.empty()) {
    keyActualType = GeneratorType::get(
        ArrayRef(thunkParamTypes).take_front(mentionedParamRefs.size()),
        reparamActualForThunkKey);
  }
  Attribute key = ArrayAttr::get(
      ctx, {TypeAttr::get(keyActualType), TypeAttr::get(thunkSignature)});
  FnOp thunk = emitter.shared.getOrCreateFunctionThunk(
      key, generateConversionThunk, expr->getLoc());
  if (!thunk) {
    dest.resetForError(emitter);
    return {};
  }

  // Now that we have the thunk defined somewhere, we're going to reference it.
  // In the above `foo` example, in this `alias` line:
  //
  //     alias my_func_alias: def(mut Ship[ZC]) -> None =
  //         ship_func_thunk[ZC, read_ship[ZC]]
  //
  // ...we'll now produce the `ship_func_thunk[ZC, read_ship[ZC]]`.

  // First, cast the callee to the reduced actual type.
  auto calleeParam = ParamOperatorAttr::getRebind(callee.get(), reducedActual);

  // Assemble the parameters (`ZC, read_ship[ZC]`) that we'll bind to the thunk.
  ParameterEvaluator evaluator = emitter.shared.getParameterEvaluator();
  for (ParamDeclRefAttr ref : mentionedParamRefs) {
    // Bind the clarifying parameter (see TAPCPTTT).
    evaluator.appendIndexBinding(ref);
  }
  for (Type type :
       ArrayRef(thunkParamTypes).drop_front(mentionedParamRefs.size())) {
    // If there are "remaining input parameters", like in:
    //
    //     alias my_func_alias: def[Y: Bool]() -> None = ...
    //
    // then we leave them unbound (see TARIPNBITM).
    evaluator.appendIndexBinding(
        UnboundAttr::get(evaluator.getReboundType(type)));
  }
  // Bind the user kernel as the conversion thunk's `callee` input parameter.
  // Both the position (last param) and the explicit
  // `kgen.transparent_thunk_callee_param` marker (set above) are part of how
  // we recover the wrapped function downstream — see
  // `IREvaluatorContext::resolveTransparentThunkCallee` in
  // `IREvaluatorContext.{h,cpp}`.
  evaluator.appendIndexBinding(calleeParam);

  SymbolConstantAttr symbol = thunk.getBoundSymbolRef(
      emitter.shared.getEvaluationContext(),
      ParameterExprArrayAttr::get(ctx, evaluator.getIndexBindings()));

  // Finally, cast the result back to the expected type.
  return emitter.emitCResult(ParamOperatorAttr::getRebind(symbol, expected),
                             expr, dest);
}

//===----------------------------------------------------------------------===//
// Generator value conversion (emit side)
//===----------------------------------------------------------------------===//

/// Build the value that discharges *all* of `generatorValue`'s body constraints
/// (which must already be proven in scope).
static TypedAttr emitFullConstraintDischarge(PValue generatorValue,
                                             SharedState &shared) {
  auto generator = sugarCast<GeneratorType>(generatorValue.getType());
  ArrayRef<ConstraintAttr> bodyConstraints = generator.getBodyConstraints();
  DenseBoolArrayAttr discharged;
  if (!bodyConstraints.empty()) {
    llvm::BitVector indicesToDischarge(bodyConstraints.size());
    indicesToDischarge.set();
    discharged = KGEN::getDenseBoolArrayAttr(generatorValue.get().getContext(),
                                             indicesToDischarge);
  }
  return BindParamsAttr::get(
      generatorValue.get().getContext(), generatorValue.get(),
      /*paramValues=*/{}, discharged, &shared.getEvaluationContext());
}

static CValue
convertGeneratorValue(CValue value, const ExprNode *expr,
                      GeneratorType expected, IREmitter &emitter,
                      ExprDest &dest,
                      ArrayRef<ConstraintAttr> additionalAssumptions = {}) {
  // Constraint discharge (shedding `value`'s body constraints to reach
  // `expected`) is handled up front by `emitZeroCostConvert`.

  // If this is a function generator value, defer to function conversion.
  if (auto expectedFnType = sugarDynCast<FnTypeGeneratorType>(expected)) {
    return convertFunctionGeneratorValue(value, expr, expectedFnType, emitter,
                                         dest);
  }

  // We do not have dynamic generators at all.
  PValue genAttr = value.getIfPValue();
  if (!genAttr) {
    emitter.emitError(expr->getLoc(),
                      "TODO: dynamic generator conversions not supported yet")
        << expr->getRange();
    dest.resetForError(emitter);
    return {};
  }

  // This must be a concrete generator attr, and it should have been ensured by
  // `canConvertGeneratorTypes`
  auto concreteGenAttr = sugarCast<GeneratorAttr>(genAttr.get());
  ExprDest tmpDest(dest.getContext());
  CValue convBody = emitter.emitImplicitConversionToType(
      {concreteGenAttr.getBody(), expr}, expected.getBody(), tmpDest,
      additionalAssumptions);

  assert(convBody && convBody.getIfPValue());
  auto convGen = GeneratorAttr::get(expected.getInputParamTypes(),
                                    convBody.getIfPValue().get(),
                                    expected.getParamListAttrs());
  return emitter.emitCResult(convGen, expr, dest);
}

static CValue convertEmptyGeneratorToBody(ASTExprAnd<CValue> valueExpr,
                                          ASTType requiredType,
                                          IREmitter &emitter, ExprDest &dest) {
  PValue value = valueExpr.ir.getIfPValue();
  assert(value && "canConvertEmptyGeneratorToBody should require a PValue");
  return emitter.emitCResult(emitFullConstraintDischarge(value, emitter.shared),
                             valueExpr.expr, dest);
}

//===----------------------------------------------------------------------===//
// Generator convertibility classification (can side)
//===----------------------------------------------------------------------===//

namespace {

struct ConversionResult {
  /// Whether a convertibility result depends on scope-level assumptions, which
  /// determines if it may be cached. The convertibility cache is keyed only on
  /// the (value, required) type pair, so scope-dependent results (e.g. dropping
  /// provable generator body constraints) must never be cached.
  enum class Sensitivity {
    /// The converter does not apply to this value/type pair. The caller should
    /// keep trying other converters, which may or may not cache their results.
    NotApplicable,
    /// The converter applies and its result does not depend on scope-level
    /// assumptions, so the result is safe to cache.
    ScopeIndependent,
    /// The converter applies but its result depends on scope-level assumptions
    /// (e.g. provable generator body constraints), so the result must be
    /// returned without caching.
    ScopeDependent,
  };

  Sensitivity sensitivity;
  TriBool isConvertible; // TODO: Change into ConstraintResult.

  static ConversionResult notApplicable() {
    return {Sensitivity::NotApplicable, TriBool::no()};
  }
  static ConversionResult scopeIndependent(TriBool isConvertible) {
    return {Sensitivity::ScopeIndependent, isConvertible};
  }
  static ConversionResult scopeDependent(TriBool isConvertible) {
    return {Sensitivity::ScopeDependent, isConvertible};
  }

  bool applies() const { return sensitivity != Sensitivity::NotApplicable; }
  bool isCacheable() const {
    return sensitivity == Sensitivity::ScopeIndependent;
  }
};
} // namespace

static ConversionResult
classifyEmptyGeneratorToBody(ASTExprAnd<CValue> valueExpr, ASTType requiredType,
                             ASTDecl &declScope,
                             ArrayRef<ConstraintAttr> additionalAssumptions) {
  PValue value = valueExpr.ir.getIfPValue();
  if (!value)
    return ConversionResult::notApplicable();

  auto generator = sugarDynCast<GeneratorType>(value.getType());
  if (!generator || !generator.getInputParamTypes().empty())
    return ConversionResult::notApplicable();

  if (!ASTType(generator.getBody()).isEqualCanon(requiredType))
    return ConversionResult::notApplicable();

  ArrayRef<ConstraintAttr> bodyConstraints = generator.getBodyConstraints();
  if (bodyConstraints.empty())
    return ConversionResult::scopeIndependent(TriBool::yes());

  // Whether the body constraints are dischargeable depends on this scope's
  // assumptions (and any caller-supplied assumptions), so the result is
  // scope-dependent and must not be cached.
  auto paramList = cast<PogListAttr>(generator.getParamListAttrs());
  OptionalDiag diag(declScope.getShared(), valueExpr.expr->getLoc(),
                    /*discardError=*/true);
  TriBool satisfied = canDischargeConstraintsInScope(
      declScope, paramList, bodyConstraints,
      /*origConstraints=*/{}, diag.getDiag(),
      /*unprovableConstraints=*/nullptr,
      /*evaluator=*/nullptr, additionalAssumptions);
  return ConversionResult::scopeDependent(satisfied);
}

static bool
canConvertEmptyGeneratorToBody(ASTExprAnd<CValue> valueExpr,
                               ASTType requiredType, ASTDecl &declScope,
                               ArrayRef<ConstraintAttr> additionalAssumptions) {
  ConversionResult conv = classifyEmptyGeneratorToBody(
      valueExpr, requiredType, declScope, additionalAssumptions);
  return conv.applies() && conv.isConvertible.isTrue();
}

static ConversionResult
classifyGeneratorToGenerator(ASTExprAnd<CValue> valueExpr,
                             GeneratorType requiredGenerator, ASTType rvType,
                             ASTDecl &declScope,
                             ArrayRef<ConstraintAttr> additionalAssumptions) {
  auto rvGeneratorType = sugarDynCast<GeneratorType>(rvType);
  if (!rvGeneratorType)
    return ConversionResult::notApplicable();

  // We don't support shedding constraints yet for non-zero-cost conversions.
  if (!rvGeneratorType.getBodyConstraints().empty())
    return ConversionResult::notApplicable();

  TriBool result =
      canConvertGeneratorTypes(valueExpr, rvGeneratorType, requiredGenerator,
                               declScope, additionalAssumptions);
  // A result that could have depended on `additionalAssumptions` is not safe to
  // cache.
  if (additionalAssumptions.empty())
    return ConversionResult::scopeIndependent(result);
  return ConversionResult::scopeDependent(result);
}

//===----------------------------------------------------------------------===//
// Zero Cost Conversions
//===----------------------------------------------------------------------===//

static TypedAttr stripTypeValueCasts(TypedAttr typeValue) {
  typeValue = KGEN::stripIdentityWrappers(typeValue);
  if (auto typeParam = sugarDynCast<TypeParamAttr>(typeValue))
    if (auto paramType = sugarDynCast<ParamType>(typeParam.getTypeValue()))
      typeValue = KGEN::stripIdentityWrappers(paramType.getParam());
  return typeValue;
}

namespace {
class CastRemover : public ParameterReplacer<CastRemover> {
  template <typename T>
  std::conditional_t<std::is_base_of_v<Type, T>, Type, Attribute>
  doReplace(T value, size_t depth) {
    if constexpr (std::is_base_of_v<Attribute, T>) {
      if (auto downcast = dyn_cast<DowncastAttr>(value))
        value = TypeParamAttr::get(ParamType::get(downcast.getInputTypeValue()),
                                   downcast.getType());
      if (auto upcast = dyn_cast<UpcastAttr>(value))
        value = TypeParamAttr::get(ParamType::get(upcast.getInputTypeValue()),
                                   upcast.getType());
    }

    SmallVector<Attribute, 16> newAttrs;
    SmallVector<Type, 16> newTypes;
    bool changed = false;
    auto walkFn = [&](auto value, SmallVectorImpl<decltype(value)> &values) {
      auto newValue = this->replaceImpl(value, depth);
      changed |= newValue != value;
      values.push_back(newValue);
    };
    value.walkImmediateSubElements(
        [&](Attribute attr) { walkFn(attr, newAttrs); },
        [&](Type type) { walkFn(type, newTypes); });
    if (!changed)
      return value;
    return value.replaceImmediateSubElements(newAttrs, newTypes);
  }

  friend class ParameterReplacer<CastRemover>;
};
} // namespace

// Returns true if `fromParam` and `toParam` denote the same parameter value up
// to `downcast`/`upcast` wrappers. A downcasted (or upcasted) type-value is
// semantically equivalent to its underlying type-value for rebind purposes. The
// wrappers only add (or relax) compile-time trait constraints.
static bool zeroCostConvertibleTypeValues(TypedAttr fromParam,
                                          TypedAttr toParam) {
  if (isEqualCanon(fromParam, toParam))
    return true;

  // This also covers combined upcast/downcast cases, e.g. when a downcasted
  // type from struct_field_types is compared against an upcasted metatype,
  // which can occur in generic serialization code that uses reflection to
  // iterate over fields while also using trait-based dispatch.
  return stripTypeValueCasts(fromParam) == stripTypeValueCasts(toParam);
}

static bool canZeroCostConvertParamTypes(ParamType fromParamType,
                                         ParamType toParamType,
                                         SharedState &shared) {
  // If the source & target types are both get_witness on the same type-values,
  // we can zero-cost convert.
  if (auto fromGetWitness =
          sugarDynCast<GetWitnessAttr>(fromParamType.getParam())) {
    if (auto toGetWitness =
            sugarDynCast<GetWitnessAttr>(toParamType.getParam())) {
      if (fromGetWitness.getWitnessName() != toGetWitness.getWitnessName())
        return false;

      auto fromTypeValue = stripTypeValueCasts(fromGetWitness.getTypeValue());
      auto toTypeValue = stripTypeValueCasts(toGetWitness.getTypeValue());
      if (fromTypeValue != toTypeValue)
        return false;

      return true;
    }
  }

  // Handle downcast<X> -> X conversions.
  return zeroCostConvertibleTypeValues(fromParamType.getParam(),
                                       toParamType.getParam());
}

static FailureOr<bool>
isValidUpCastToTypeType(SharedState &shared, ASTType fromType, ASTType toType) {
  // Trait metatypes/struct MetaMetaType are allowed to upcast to trivial types.
  if (sugarIsa<TypeType>(toType)) {
    // Allowing casting from any metatype to type of all types.
    return sugarIsa<StructMetaType, StructMetaMetaType, TraitType, AnyTraitType,
                    TypeType, NonStructTypeType, FnLiteralTypeGeneratorMetaType,
                    FnLiteralTypeGeneratorMetaMetaType>(fromType);
  }

  // Not applicable.
  return failure();
}

/// Returns true if two function-type generators have the same representation
/// post-elaboration: they differ only in argument names, parameter names,
/// passing kinds, or capture origins. Body constraints are deliberately not
/// considered here (they are handled by `canProveBodyConstraints`), and this
/// match is intentionally looser than `getWithoutBodyConstraints()` equality.
static bool canZeroCostConvertFnTypes(FnTypeGeneratorType from,
                                      FnTypeGeneratorType to) {
  // If the fn meta data mismatches (different arg conventions, effects, or
  // origin metadata), return false. Capture origins are excluded from the
  // comparison (it will be deleted anyway in the near future).
  if (withoutCaptureOrigins(from.getFnMetadata()) !=
      withoutCaptureOrigins(to.getFnMetadata()))
    return false;

  // Allow signature types to be converted for free if they differ only in
  // argument names, parameter names, passing kinds.
  if (from.getNumArguments() != to.getNumArguments())
    return false;

  // Result types must match exactly.
  if (from.getResults() != to.getResults())
    return false;

  for (auto [idx, fromTy, toTy, conv] : llvm::enumerate(
           from.getArguments(), to.getArguments(), from.getArgConventions())) {
    Type fromCmpTy = RefType::stripRefConvention(fromTy, conv);
    Type toCmpTy = RefType::stripRefConvention(toTy, conv);
    if (!ASTType(fromCmpTy).isEqualCanon(toCmpTy))
      return false;

    // If the argument has a required keyword, then the two must match names.
    if (from.getArgListAttrs().getPassingKind(idx) == PassingKind::KwOnly ||
        to.getArgListAttrs().getPassingKind(idx) == PassingKind::KwOnly) {
      if (from.getArgName(idx) != to.getArgName(idx))
        return false;
    }
  }

  return true;
}

/// Returns if a value of the specified type can be coerced to the other type
/// with a zero-cost conversion like a rebind.  This means that values of the
/// two types have exactly the same representation post-elaboration.
/// Shared body of `IREmitter::canZeroCostConvert` and the one caller in this
/// file that wants the constraints behind an `unknown` verdict, which
/// `unprovenConstraints` receives when non-null.
static TriBool
canZeroCostConvertImpl(ASTType sugaredFromType, ASTType sugaredToType,
                       SharedState &shared, ASTDecl &declScope,
                       ArrayRef<ConstraintAttr> additionalAssumptions,
                       SmallVectorImpl<ConstraintAttr> *unprovenConstraints) {
  if (sugaredFromType.isEqualCanon(sugaredToType))
    return TriBool::yes(); // No rebind needed!
  ASTType toType = getCanonicalType(sugaredToType);
  ASTType fromType = getCanonicalType(sugaredFromType);

  FailureOr<bool> upCastable =
      isValidUpCastToTypeType(shared, fromType, toType);
  if (succeeded(upCastable))
    return TriBool::fromBool(upCastable.value());

  // fn type is non-struct type (but should it?)
  if (sugarIsa<FnLiteralTypeGeneratorMetaType>(fromType) &&
      sugarIsa<NonStructTypeType>(toType))
    return TriBool::yes();

  // Check for param type conversions.
  if (auto fromParamType = sugarDynCast<ParamType>(fromType))
    if (auto toParamType = sugarDynCast<ParamType>(toType))
      return TriBool::fromBool(
          canZeroCostConvertParamTypes(fromParamType, toParamType, shared));

  // Check for closure structs and dig out their underlying signature types to
  // check whether the conversion can occur.
  auto fromDecl = fromType.getDecl(shared);
  auto toDecl = toType.getDecl(shared);
  if (fromDecl && toDecl) {
    auto fromDeclOp =
        dyn_cast_or_null<StructDeclOp>(fromDecl->getIfOperation());
    auto toDeclOp = dyn_cast_or_null<StructDeclOp>(toDecl->getIfOperation());
    if (fromDeclOp && toDeclOp) {
      FuncTypeGeneratorType fromSig =
          fromDeclOp.getClosureSignature().value_or(nullptr);
      FuncTypeGeneratorType toSig =
          toDeclOp.getClosureSignature().value_or(nullptr);
      if (fromSig && toSig) {
        // Compare the specialized signatures.
        fromSig = fromSig.getSpecializedGenerator(
            fromType.getParamBindings(), &shared.getEvaluationContext());
        toSig = toSig.getSpecializedGenerator(toType.getParamBindings(),
                                              &shared.getEvaluationContext());
        return canZeroCostConvertImpl(fromSig, toSig, shared, declScope,
                                      additionalAssumptions,
                                      unprovenConstraints);
      }

      // Otherwise, if both types reference the same struct declaration (e.g.
      // `Iter[X]` vs `Iter[Y]`), the conversion is zero-cost as long as each
      // pair of parameter bindings is zero-cost convertible. We conservatively
      // require both sides to be `StructType` here, though in principle this
      // just needs both sides to be some metatype of struct types at the same
      // type-universe level.
      if (fromDecl == toDecl && sugarIsa<LIT::StructType>(fromType) &&
          sugarIsa<LIT::StructType>(toType)) {
        CastRemover remover;
        if (remover.replace(fromType.mlirType) ==
            remover.replace(toType.mlirType))
          return TriBool::yes();
      }
      return TriBool::no();
    }
  }

  // Check origin downcasting.  The safe conversions are:
  //   Origins with identical mutability will be uniqued and already handled.
  //   Conversion from any mutability to KNOWN immutable is fine.
  //   Conversion from KNOWN mutable to any mutability is fine.
  //   Conversion from with mutability "X" to "X&Y" is known to be fine.
  // We allow KGEN to fold the true and false cases for us.
  if (auto fromOrigin = sugarDynCast<OriginType>(fromType))
    if (auto toOrigin = sugarDynCast<OriginType>(toType)) {
      auto toMut = toOrigin.getIsMutable();
      auto result =
          ParamOperatorAttr::get(POC::And, toMut, fromOrigin.getIsMutable());
      if (result == toMut)
        return TriBool::yes();
    }

  // Check reference downcasting.  The only thing allowed to disagree is the
  // origin set / mutability.
  if (auto fromRef = sugarDynCast<RefType>(fromType)) {
    if (auto toRef = sugarDynCast<RefType>(toType)) {
      // Element types and address space have to be exactly equal.
      if (fromRef.getAddressSpace() != toRef.getAddressSpace() ||
          !ASTType(fromRef.getElementType())
               .isEqualCanon(toRef.getElementType()))
        return TriBool::no();

      // Verify compatible OriginType(mutability).  This is checking the type
      // of the origin, which contains its mutability specifier.
      auto toOriginType = toRef.getOriginType();
      if (fromRef.getOriginType() != toOriginType &&
          !canZeroCostConvertImpl(fromRef.getOriginType(), toOriginType, shared,
                                  declScope, additionalAssumptions,
                                  /*unprovenConstraints=*/nullptr)
               .isTrue())
        return TriBool::no();

      // We allow converting an "any" origin to anything concrete.
      // NOTE: This is not memory safe; we should make this an explicit
      // operation someday.
      if (sugarIsa<AnyOriginAttr>(fromRef.getOrigin()))
        return TriBool::yes();

      // FIXME: People are using things StaticString to refer to comptime
      // strings, even though StaticString is a runtime concept :-/.
      if (sugarIsa<ComptimeOriginAttr>(fromRef.getOrigin())) {
        if (auto originField =
                sugarDynCast<OriginFieldAttr>(toRef.getOrigin())) {
          if (isa<StaticOriginAttr>(originField.getBase()) &&
              originField.getField().str() == "__constants__" &&
              originField.getType().isMutableKnown(false)) {
            return TriBool::yes();
          }
        }
      }

      // We can convert origin subset to a origins superset.
      auto toOrigin = toRef.getOrigin();
      auto originUnion = OriginUnionAttr::get(
          {toOrigin, OriginMutCastAttr::get(fromRef.getOrigin(), toOriginType)},
          toOriginType);
      return TriBool::fromBool(toOrigin == originUnion);
    }
  }

  if (auto actual = sugarDynCast<FnLiteralTypeGeneratorType>(sugaredFromType))
    if (sugarIsa<FnTypeGeneratorType>(sugaredToType))
      return canZeroCostConvertImpl(actual.getSymbolConstantAttr().getType(),
                                    sugaredToType, shared, declScope,
                                    additionalAssumptions, unprovenConstraints);

  // Otherwise handle generator conversions. Both sides must be generators.
  auto fromGen = sugarDynCast<GeneratorType>(sugaredFromType);
  auto toGen = sugarDynCast<GeneratorType>(sugaredToType);
  if (!fromGen || !toGen)
    return TriBool::no();

  // Input parameter types must match exactly.
  if (fromGen.getInputParamTypes() != toGen.getInputParamTypes())
    return TriBool::no();

  // The representations must agree. Function types have their own looser match
  // (argument/parameter names, passing kinds, and capture origins may differ);
  // every other generator requires stricter structural equality once body
  // constraints are stripped.
  if (auto fromFn = sugarDynCast<FnTypeGeneratorType>(fromGen)) {
    auto toFn = sugarDynCast<FnTypeGeneratorType>(toGen);
    if (!toFn || !canZeroCostConvertFnTypes(fromFn, toFn))
      return TriBool::no();
  } else if (!ASTType(fromGen.getWithoutBodyConstraints())
                  .isEqualCanon(toGen.getWithoutBodyConstraints())) {
    return TriBool::no();
  }

  // The representations agree, so all that is left is the body constraints.
  // Gaining a constraint is free; shedding one requires this scope to prove it.
  return canProveBodyConstraints(fromGen, toGen, declScope,
                                 additionalAssumptions, unprovenConstraints);
}

TriBool
IREmitter::canZeroCostConvert(ASTType sugaredFromType, ASTType sugaredToType,
                              SharedState &shared, ASTDecl &declScope,
                              ArrayRef<ConstraintAttr> additionalAssumptions) {
  return canZeroCostConvertImpl(sugaredFromType, sugaredToType, shared,
                                declScope, additionalAssumptions,
                                /*unprovenConstraints=*/nullptr);
}

/// If there is a common type shared between the two reference types, return
/// it. Otherwise return null.
RefType IREmitter::getCommonRefType(RefType ref1, RefType ref2) {
  if (ref1 == ref2)
    return ref1;
  // Element types and addr spaces have to be exactly equal.
  auto eltType = ref1.getElementType();
  if (!ASTType(eltType).isEqualCanon(ref2.getElementType()) ||
      ref1.getAddressSpace() != ref2.getAddressSpace())
    return {};

  // If so, we can form a common type with a subset of their mutability and
  // a union of their origins.
  auto isMutableAttr =
      ParamOperatorAttr::get(POC::And, ref1.isMutable(), ref2.isMutable());

  auto l1 = OriginMutCastAttr::get(ref1.getOrigin(), isMutableAttr);
  auto l2 = OriginMutCastAttr::get(ref2.getOrigin(), isMutableAttr);
  auto origin =
      OriginUnionAttr::get({l1, l2}, sugarCast<OriginType>(l1.getType()));
  return RefType::get(eltType, origin, ref1.getAddressSpace());
}

/// If there is a shared supertype for the two specified types, return it in
/// 'result' and return success.
///
/// For example, we may have two derived classes that have the same base class
/// even if neither is convertible to the other.
///
/// This function uses `__merge_with__` if available, otherwise it uses
/// implicit conversions to find a common match.  If a `__merge_with__` is
/// involved, the PValue for the function to invoke is returned.
enum CommonTypeResult {
  CTR_Success,
  CTR_Ambiguous,
  CTR_NoCommonType,
  CTR_MergeWithConflict,
  CTR_MergeWithConvertFail, // One __merge_with__ exists, but other doesn't work
};

static std::tuple<CommonTypeResult, PValue, PValue>
findCommonType(ASTExprAnd<CValue> val1, ASTExprAnd<CValue> val2,
               ASTType &result, IREmitter &emitter, ASTType contextualType) {

  // If the types already match, then we're done.
  ASTType type1 = val1.ir.getRValueType();
  ASTType type2 = val2.ir.getRValueType();

  auto succeed =
      [&](ASTType type, PValue lhsMWPV = {},
          PValue rhsMWPV = {}) -> std::tuple<CommonTypeResult, PValue, PValue> {
    result = type;
    return {CTR_Success, lhsMWPV, rhsMWPV};
  };

  if (type1.isEqualCanon(type2))
    return succeed(type1);

  if (LIT::isFirstLevelTypeExpr(val1.ir.getIfPValue()) &&
      LIT::isFirstLevelTypeExpr(val2.ir.getIfPValue())) {
    result = LIT::mergeTwoMetaTypeBounds(emitter.shared, type1, type2);
    return {CTR_Success, PValue(), PValue()};
  }

  // Ok, they are different types.  If either type has a __merge_with__ member,
  // then we use that in preference to anything else.

  // This checks to see if 'src' has a __merge_with__ member that unambiguously
  // takes 'other' as an parameter. If so it returns the PValue for the method
  // and the result type of calling the method.
  auto lookupMergeWith = [&](ASTExprAnd<CValue> srcValue, ASTType srcType,
                             ASTType otherType) -> std::pair<PValue, ASTType> {
    // Look up __merge_with__ and bind other_type.
    OverloadSet os =
        OverloadSet::lookup(emitter.declScope, srcType, "__merge_with__",
                            srcValue.expr, CallSyntax::kMethodCall);
    os.paramBindings.add(srcValue.expr, PValue(otherType),
                         StringAttr::get(emitter.getContext(), "other_type"));
    CallOperands operands(CallSyntax::kMethodCall, srcValue.expr, EC_MergeWith,
                          {srcValue});
    auto res = os.filterOverloadSet(
        operands, /*emitDiagnosticsOnFailure=*/false, emitter);
    if (!res.isYes())
      return {{}, {}};
    return {res.getYes(), res.getYes().getType().getSignatureUserResultType()};
  };

  auto [lhsMWPV, lhsMPType] = lookupMergeWith(val1, type1, type2);
  auto [rhsMWPV, rhsMPType] = lookupMergeWith(val2, type2, type1);

  // Handle two __merge_with__ methods.
  if (lhsMWPV && rhsMWPV) {
    if (!lhsMPType.isEqualCanon(rhsMPType))
      return {CTR_MergeWithConflict, lhsMWPV, rhsMWPV};
    // If both convert to the same type, then we're good.
    return succeed(lhsMPType, lhsMWPV, rhsMWPV);
  }
  // If there is one __merge_with__ method, then we use that if the other type
  // converts to the result value.
  if (lhsMWPV) {
    if (IREmitter::canImplicitlyConvertToType(val2, lhsMPType,
                                              emitter.declScope))
      return succeed(lhsMPType, lhsMWPV, PValue());
    result = lhsMPType;
    return {CTR_MergeWithConvertFail, lhsMWPV, PValue()};
  }
  if (rhsMWPV) {
    if (IREmitter::canImplicitlyConvertToType(val1, rhsMPType,
                                              emitter.declScope))
      return succeed(rhsMPType, PValue(), rhsMWPV);
    result = rhsMPType;
    return {CTR_MergeWithConvertFail, PValue(), rhsMWPV};
  }

  // Otherwise, we have no __merge_with__ method, see if there is a contextual
  // type.  If so, convert to that.
  if (contextualType) {
    if (IREmitter::canImplicitlyConvertToType(val1, contextualType,
                                              emitter.declScope) &&
        IREmitter::canImplicitlyConvertToType(val2, contextualType,
                                              emitter.declScope))
      return succeed(contextualType);
  }

  // Otherwise, check out implicit conversions from one value to the other.

  // If one type implicit converts to the other, then the other is a common
  // type.  Don't do this if both convert to each other, this would be
  // ambiguous.
  bool isConvertibleToType2 =
      IREmitter::canImplicitlyConvertToType(val1, type2, emitter.declScope);
  bool isConvertibleToType1 =
      IREmitter::canImplicitlyConvertToType(val2, type1, emitter.declScope);
  if (isConvertibleToType2 && !isConvertibleToType1)
    return succeed(type2);
  if (isConvertibleToType1 && !isConvertibleToType2)
    return succeed(type1);
  if (isConvertibleToType1 && isConvertibleToType2)
    return {CTR_Ambiguous, PValue(), PValue()};

  // If one or the other type is nonmaterializable, the conversion is free,
  // so check to see if there is an unambiguous common type.
  bool type2ConvertsToType1Nonmat = false;
  bool type1ConvertsToType2Nonmat = false;
  auto type1Nonmat = type1.getNonmaterializableTarget(emitter.shared);
  auto type2Nonmat = type2.getNonmaterializableTarget(emitter.shared);
  if (type1Nonmat)
    type2ConvertsToType1Nonmat = IREmitter::canImplicitlyConvertToType(
        val2, type1Nonmat, emitter.declScope);
  if (type2Nonmat)
    type1ConvertsToType2Nonmat = IREmitter::canImplicitlyConvertToType(
        val1, type2Nonmat, emitter.declScope);

  if (type2ConvertsToType1Nonmat && !type1ConvertsToType2Nonmat)
    return succeed(type1Nonmat);
  if (type1ConvertsToType2Nonmat && !type2ConvertsToType1Nonmat)
    return succeed(type2Nonmat);
  if (type1ConvertsToType2Nonmat && type2ConvertsToType1Nonmat) {
    if (type1Nonmat.isEqualCanon(type2Nonmat))
      return succeed(type1Nonmat);
    return {CTR_Ambiguous, PValue(), PValue()};
  }

  // No common type found.
  return {CTR_NoCommonType, PValue(), PValue()};
}

/// Given two values that need to match, try to coerce one to the other if they
/// disagree on type.  This emits an error (when loc is non-null) and returns
/// failure if the request is ambiguous or impossible.
///
/// The 'configEmitter' function is called to set the insertion point of the
/// emitter for the true/false branches of the conditional.
ParseResult IREmitter::coerceTypesToEachOther(
    SMLoc loc, CValue &lhs, const ExprNode *lhsExpr, CValue &rhs,
    const ExprNode *rhsExpr, std::function<void(bool isLHS)> configEmitter,
    ASTType contextualType) {
  if (!configEmitter)
    configEmitter = [&](bool isLHS) {};

  if (!lhs || !rhs)
    return failure();

  // If they are the same or if there is a common type between these, convert
  // them to it.
  ASTType commonType;
  auto [commonTypeResult, lhsMWPV, rhsMWPV] = findCommonType(
      {lhs, lhsExpr}, {rhs, rhsExpr}, commonType, *this, contextualType);

  // If we failed and have no source location, we just return failure without
  // returning an error.
  if (commonTypeResult != CTR_Success && !loc.isValid())
    return failure();

  ASTType lhsType = lhs.getRValueType(), rhsType = rhs.getRValueType();
  switch (commonTypeResult) {
  case CTR_Success:
    break;
  case CTR_NoCommonType:
    emitError(loc, "value of type ")
        << lhsType << " is not compatible with value of type " << rhsType
        << lhsExpr->getRange() << rhsExpr->getRange();
    return failure();
  case CTR_Ambiguous: {
    auto diag = emitError(loc, "ambiguous merge: left value has type ")
                << lhsType << " and right value has type " << rhsType
                << ", and both convert to each other" << lhsExpr->getRange()
                << rhsExpr->getRange();
    diag.attachNote(loc)
        << "you could disambiguate by casting the left value to " << rhsType
        << lhsExpr->getRange();
    diag.attachNote(loc) << "or cast the right value to " << lhsType
                         << rhsExpr->getRange();
    return failure();
  }
  case CTR_MergeWithConflict: {
    auto diag = emitError(loc, "value of types ")
                << lhsType << " and " << rhsType
                << " have '__merge_with__' methods that disagree on common type"
                << lhsExpr->getRange() << rhsExpr->getRange();
    auto lhsDest = lhsMWPV.getType().getSignatureUserResultType();
    auto rhsDest = rhsMWPV.getType().getSignatureUserResultType();
    diag.attachNote(loc) << "one returns " << lhsDest
                         << " and the other returns " << rhsDest;
    return failure();
  }
  case CTR_MergeWithConvertFail: {
    auto diag = emitError(loc, "value of types ")
                << lhsType << " and " << rhsType << " cannot be merged to type "
                << commonType << lhsExpr->getRange() << rhsExpr->getRange();
    // One of lhsMWPV/rhsMWPV will be nonnull, indicating which mergewith.
    diag.attachNote(loc) << (lhsMWPV ? rhsType : lhsType)
                         << " does not implicitly convert to " << commonType;
    return failure();
  }
  }

  // Okay we found a successful conversion path.  See if we need to apply any
  // __merge_with__ methods first.
  if (lhsMWPV) {
    configEmitter(/*isLHS*/ true);
    lhs =
        emitIndirectCall(lhsMWPV, CallOperands(CallSyntax::kMethodCall, lhsExpr,
                                               EC_MergeWith, {{lhs, lhsExpr}}));
  }
  if (rhsMWPV) {
    configEmitter(/*isLHS*/ false);
    rhs =
        emitIndirectCall(rhsMWPV, CallOperands(CallSyntax::kMethodCall, rhsExpr,
                                               EC_MergeWith, {{rhs, rhsExpr}}));
  }

  // Next apply any implicit conversions that may be needed.
  if (!lhsType.isEqualCanon(commonType)) {
    configEmitter(/*isLHS*/ true);
    lhs = emitCValue({lhs, lhsExpr}, EC_OperatorOperandValue, commonType);
  }
  if (!rhsType.isEqualCanon(commonType)) {
    configEmitter(/*isLHS*/ false);
    rhs = emitCValue({rhs, rhsExpr}, EC_OperatorOperandValue, commonType);
  }

  // If we are in a dynamic context and the result is nonmaterializable, then
  // we need to emit the conversion in the parameter domain before the
  // conditional and decide what the result type should be based on that.
  if (builder) {
    if (auto mat = commonType.getNonmaterializableTarget(shared)) {
      configEmitter(/*isLHS*/ true);
      lhs = emitCValue({lhs, lhsExpr}, EC_CondExpr, mat);
      configEmitter(/*isLHS*/ false);
      rhs = emitCValue({rhs, rhsExpr}, EC_CondExpr, mat);
    }
  }

  // Ensure sugar types agree.
  if (lhs && rhs &&
      lhs.getRValueType().mlirType != rhs.getRValueType().mlirType) {
    configEmitter(/*isLHS*/ false);
    Type destType;
    // LHS and RHS may differ in MValue'ness.  The LHS might be an SRValue and
    // the RHS may be an MLValue for example.
    if (rhs.isMValue())
      destType = rhs.getMValueType().getWithElement(lhs.getRValueType());
    else
      destType = lhs.getRValueType();
    rhs = rebindValue({rhs, rhsExpr}, destType);
  }

  return success(lhs && rhs);
}

/// Given a value of a type that can be zero cost converted to another type,
/// emit a rebind or other operation to get it in the right type.
PValue IREmitter::emitZeroCostConvert(PValue value, ASTType toType,
                                      SharedState &shared) {
  assert(toType.mlirType != value.getType() && "Already the same");

  // PValues of origin type have a special conversion.
  if (sugarIsa<OriginType>(toType) && sugarIsa<OriginType>(value.getType()))
    value = OriginMutCastAttr::get(value, toType);

  if (sugarIsa<TypeType>(toType) &&
      sugarIsa<TraitType, AnyTraitType>(value.getType()))
    return TypeParamAttr::get(ASTType(value), toType);

  if (sugarIsa<FnLiteralTypeGeneratorMetaType>(value.getType()) &&
      sugarIsa<NonStructTypeType>(toType))
    return TypeParamAttr::get(ASTType(value), toType);

  if (auto actual = sugarDynCast<FnLiteralTypeGeneratorType>(value.getType()))
    if (auto expected = sugarDynCast<FnTypeGeneratorType>(toType))
      return ParamOperatorAttr::getRebind(actual.getSymbolConstantAttr(),
                                          expected);

  // Shedding a generator body constraint requires BindParams discharge so the
  // value's own type drops the obligation. Instead of figuring out which subset
  // of constraints to discharge, discharge all of `from`'s constraints when it
  // has any. Gaining `to`'s constraints back is a free rebind.
  if (auto fromGen = sugarDynCast<GeneratorType>(value.getType())) {
    if (sugarIsa<GeneratorType>(toType) &&
        !fromGen.getBodyConstraints().empty()) {
      value = PValue(emitFullConstraintDischarge(value, shared));
    }
  }

  return ParamOperatorAttr::getRebind(value.get(), toType);
}

CValue IREmitter::emitZeroCostConvert(ASTExprAnd<CValue> value,
                                      ASTType toType) {
  assert(toType.mlirType != value.ir.getType() && "Already the same");

  // PValue handling has a helper.
  if (auto pv = value.ir.getIfPValue())
    return emitZeroCostConvert(pv, toType, shared);

  // The RValue types need to be rebound, but MValues have a level of
  // reference around them that we want to maintain.
  if (value.ir.isMValue())
    toType = value.ir.getMValueType().getWithElement(toType);

  // Rebind the value if we can.
  return rebindValue(value, toType);
}

//===----------------------------------------------------------------------===//
// Trait conversions
//===----------------------------------------------------------------------===//

/// Return true if the MLIR type can implicitly conform to the trait.
static bool checkMLIRTypeConformance(SharedState &shared, SMLoc loc,
                                     TraitType trait) {
  // Use a special wrapper decl in the builtins as stubs.
  ASTType wrapperType = shared.getBuiltinStubsMLIRType(loc);
  return wrapperType
      .doesConformTo(
          trait, shared,
          ASTDecl::getAssumptionsFromScope(wrapperType.getDecl(shared)))
      .isTrue();
}

/// Emit a conversion from an MLIR type to a trait type by materializing stubs
/// for the type's witness table.
PValue IREmitter::bindNonStructTypeToTrait(ASTExprAnd<CValue> value,
                                           TraitType trait) {
  // Only parameter-domain type-values are supported right now.
  PValue typeValue = value.ir.getIfPValue();
  if (!typeValue) {
    shared.emitError(value.expr->getLoc(),
                     "existentials are not supported yet!");
    return {};
  }

  // If the function generator type is upcastable to a non-struct type (but
  // should it?, esp. for parametric type. We can not easily disable the
  // conversion at the moment since many existing code relies on it).
  if (sugarIsa<FnLiteralTypeGeneratorMetaType>(typeValue.getType()))
    typeValue =
        UpcastAttr::get(NonStructTypeType::get(getContext()), typeValue.get());

  ASTType mlirType = typeValue.getIfTypeValue();
  SMLoc loc = value.expr->getLoc();

  // Use a special wrapper decl in the builtins as stubs.
  ASTType wrapperType = shared.getBuiltinStubsMLIRType(loc);
  ASTDecl *wrapperDecl = wrapperType.getDecl(shared);
  if (!wrapperDecl ||
      !isa_and_nonnull<StructDeclOp>(wrapperDecl->getIfOperation())) {
    shared.emitError(loc, "malformed builtin._stubs.__MLIRType");
    return {};
  }

  // Explicitly check that the wrapper conforms to the trait so that
  // conformances & special functions may be generated.  __MLIRType has only
  // unconditional conformances, so no caller scope is needed.
  if (!wrapperType.doesConformTo(trait, shared, {}).isTrue()) {
    MojoInflightDiag diag =
        shared.emitError(value.expr->getLoc(), "cannot bind MLIR type ")
        << mlirType << " to trait " << ASTType(trait);
    return {};
  }

  // If the type is a param type, then we just need to upcast it to the trait.
  if (auto paramType = sugarDynCast<ParamType>(mlirType)) {
    return UpcastAttr::get(trait, PValue(paramType.getParam()));
  }

  // Otherwise, create a new type value whose witness table is provided by the
  // wrapper stub.
  ASTType boundWrapper = cast<StructDeclOp>(wrapperDecl->getIfOperation())
                             .bindReference({typeValue});
  return TypeParamAttr::get(boundWrapper, mlirType, trait);
}

//===----------------------------------------------------------------------===//
// Generalized Implicit Conversions
//===----------------------------------------------------------------------===//

static ASTDecl *getClosureTraitDecl(SharedState &shared,
                                    const TraitType &traitTy) {
  for (const auto &symbol : traitTy.getSymbols()) {
    auto &symbolDecl =
        shared.declResolver->getDeclForTypeSymbol(symbol.getSymbol());
    if (symbolDecl.isErroneous())
      continue;

    if (auto traitDeclOp =
            dyn_cast_if_present<TraitDeclOp>(symbolDecl.getIfOperation());
        traitDeclOp && traitDeclOp.getDefinesClosure())
      return &symbolDecl;
  }

  return nullptr;
}

// Returns the upcastability verdict (`yes`/`no`/`unknown`) for converting a
// type value to a trait. Returns failure for non-applicable cases (i.e.,
// `fromType` is not a typetype and/or `toType` is not a trait type).
//
// Shared body of the two public entry points. `unsatConstraints`, when
// non-null, is filled with the constraints behind a non-`yes` verdict; when
// passed null, the conformance query short-circuits.
static FailureOr<TriBool>
canMetaTypeUpCastToImpl(SharedState &shared, SMLoc loc, ASTType fromType,
                        ASTType toType, ASTDecl *declScope,
                        bool *scopeDependent,
                        SmallVectorImpl<ConstraintAttr> *unsatConstraints) {
  auto conformanceVerdict = [&](ASTType type, TraitType trait) {
    // Owned, not an ArrayRef: `getAssumptionsFromScope` returns by value.
    SmallVector<ConstraintAttr> assumptions =
        ASTDecl::getAssumptionsFromScope(declScope);
    if (scopeDependent && !assumptions.empty())
      *scopeDependent = true;
    if (!unsatConstraints)
      return type.doesConformTo(trait, shared, assumptions);

    ConstraintResult result =
        type.doesConformToWithDetails(trait, shared, assumptions);
    if (result.isYes())
      return TriBool::yes();
    if (result.isNo()) {
      *unsatConstraints = std::move(result).getNo();
      return TriBool::no();
    }
    *unsatConstraints = std::move(result).getUnknown();
    return TriBool::unknown();
  };

  // By default the verdict depends only on the (fromType, toType) pair. The
  // branches below that consult `declScope`'s assumptions set this, since an
  // assumption-derived verdict is scope-dependent and must not be memoized.
  if (scopeDependent)
    *scopeDependent = false;

  if (isEqualCanon(fromType, toType))
    return TriBool::yes();

  // Trait metatypes/struct MetaMetaType are allowed to upcast to trivial
  // types.
  FailureOr<bool> upCastable =
      isValidUpCastToTypeType(shared, fromType, toType);
  if (succeeded(upCastable))
    return TriBool::fromBool(*upCastable);

  auto canFnLiteralUpCastToTrait = [&](TypedAttr fnPValue,
                                       AnyTraitType anyTrait) {
    TraitType closureTrait = anyTrait.getTraitType();
    if (auto traitDecl = getClosureTraitDecl(shared, closureTrait)) {
      Type concreteWrapperType =
          shared.getClosureEmitter().getConcreteClosureWrapperTypeForFnSymbol(
              *declScope, loc, fnPValue);
      if (!concreteWrapperType)
        return false;
      return succeeded(shared.getClosureEmitter().isCompatibleWith(
          concreteWrapperType, traitDecl));
    }
    // Maintain convertibility as a MLIR type ...
    return checkMLIRTypeConformance(shared, loc, closureTrait);
  };

  // Values of known {struct/trait/mlir} type can convert to any trait type
  // they implement.
  if (auto anyTrait = sugarDynCast<AnyTraitType>(toType.extractMetaType())) {
    TraitType trait = anyTrait.getTraitType();
    bool result = false;

    if (sugarIsa<NonStructTypeType>(fromType)) {
      // MLIR types can conform to traits that have limited requirements.
      // AnyTraitType (the type of all traits) conforms to traits with only a
      // destructor (e.g. AnyType) since all traits have that.
      result = checkMLIRTypeConformance(shared, loc, trait);
    } else if (sugarIsa<StructMetaMetaType, AnyTraitType>(
                   fromType.extractMetaType())) {
      if (ASTType(fromType).getDecl(shared)) {
        SmallVector<TraitSymbolAttr> toCheck;
        // Check for closure rebindability.
        for (const auto &symbol : trait.getSymbols()) {
          auto &symbolDecl =
              shared.declResolver->getDeclForTypeSymbol(symbol.getSymbol());
          auto traitDeclOp = cast<TraitDeclOp>(symbolDecl.getIfOperation());
          if (traitDeclOp.getDefinesClosure()) {
            // If this is a struct, check whether we can do lazy conformance.
            if (sugarIsa<StructMetaType>(fromType) &&
                failed(shared.closureEmitter->isCompatibleWith(fromType,
                                                               &symbolDecl)))
              return TriBool::no();

            if (sugarIsa<TraitType>(fromType) &&
                failed(shared.closureEmitter->isTraitCompatibleWith(
                    fromType, traitDeclOp, declScope)))
              toCheck.push_back(symbol); // maybe don't need extension.
          } else {
            // Non closure traits are checked separately.
            toCheck.push_back(symbol);
          }
        }

        // Test only traits which are not closure traits
        trait = TraitType::get(shared.getContext(), toCheck);

        // Assumptions needed: e.g. `where AllWritable[*Ts]` proves
        // Tuple[*Ts]: Writable when binding to a Writable parameter.
        // Assumptions needed: implicit conversion of e.g. Tuple[*Ts] to
        // Writable
        // inside a fn with `where AllWritable[*Ts]`.
        return conformanceVerdict(fromType, trait);
      }
    } else if (auto fnGen =
                   sugarDynCastIfPresent<FnLiteralTypeGeneratorMetaType>(
                       fromType)) {
      return TriBool::fromBool(canFnLiteralUpCastToTrait(
          fnGen.getType().getSymbolConstantAttr(), anyTrait));
    } else {
      // This isn't relevant, e.g. in function pointer to closure case.
      return failure();
    }
    return TriBool::fromBool(result);
  }

  if (auto anyTrait = sugarDynCast<AnyTraitType>(toType)) {
    // Upcasts to a closure trait by inflating the function literal to its
    // closure-wrapper struct, then checking that the wrapper conforms. This
    // mirrors the corresponding FnLiteralTypeGeneratorMetaType branch above but
    // one metatype level up.
    if (auto fnGen = sugarDynCastIfPresent<FnLiteralTypeGeneratorMetaMetaType>(
            fromType)) {
      return TriBool::fromBool(canFnLiteralUpCastToTrait(
          fnGen.getType().getType().getSymbolConstantAttr(), anyTrait));
    }

    ASTType concreteType;
    // 2 cases, e.g,:
    // 1st, AnyTraitType[Copyable] to AnyTraitType[AnyType].
    // 2nd, Meta[Meta[Int]] to AnyTraitType[Copyable]
    if (auto rvAnyTrait = sugarDynCast<AnyTraitType>(fromType)) {
      concreteType = ASTType(rvAnyTrait.getTraitType());
    } else if (auto mmType = sugarDynCast<StructMetaMetaType>(fromType)) {
      concreteType = ASTType(mmType.getType());
    }

    if (concreteType) {
      // Check for closure rebindability, mirroring the AnyTraitType-metatype
      // branch above.
      for (const auto &symbol : anyTrait.getTraitType().getSymbols()) {
        auto &symbolDecl =
            shared.declResolver->getDeclForTypeSymbol(symbol.getSymbol());
        if (auto traitDeclOp =
                dyn_cast_if_present<TraitDeclOp>(symbolDecl.getIfOperation());
            traitDeclOp && traitDeclOp.getDefinesClosure()) {
          if (succeeded(shared.closureEmitter->isCompatibleWith(concreteType,
                                                                &symbolDecl)) ||
              succeeded(shared.closureEmitter->isTraitCompatibleWith(
                  concreteType, traitDeclOp, declScope)))
            return TriBool::yes();
        }
      }

      // Assumptions needed: e.g. AnyTraitType[Copyable] → AnyTraitType[Movable]
      // upcast when the Copyable conformance depends on caller assumptions.
      return conformanceVerdict(concreteType, anyTrait.getTraitType());
    }
  }

  // Not applicable.
  return failure();
}

FailureOr<TriBool> IREmitter::canMetaTypeUpCastTo(SharedState &shared,
                                                  SMLoc loc, ASTType fromType,
                                                  ASTType toType,
                                                  ASTDecl *declScope,
                                                  bool *scopeDependent) {
  return canMetaTypeUpCastToImpl(shared, loc, fromType, toType, declScope,
                                 scopeDependent, /*details=*/nullptr);
}

FailureOr<ConstraintResult> IREmitter::canMetaTypeUpCastToWithDetails(
    SharedState &shared, SMLoc loc, ASTType fromType, ASTType toType,
    ASTDecl *declScope, bool *scopeDependent) {
  SmallVector<ConstraintAttr, 2> unsatConstraints;
  FailureOr<TriBool> verdict =
      canMetaTypeUpCastToImpl(shared, loc, fromType, toType, declScope,
                              scopeDependent, &unsatConstraints);
  if (failed(verdict))
    return failure();
  return makeConstraintResult(*verdict, std::move(unsatConstraints));
}

//===----------------------------------------------------------------------===//
// Generalized Implicit Conversions
//===----------------------------------------------------------------------===//

static bool isClosureWrapperStruct(SharedState &shared, PValue value,
                                   LIT::StructType structTy) {
  if (!value)
    return false;
  auto fnSig = dyn_cast<FnTypeGeneratorType>(value.getType());
  if (!fnSig)
    return false;
  ASTDecl &decl =
      shared.declResolver->getDeclForTypeSymbol(structTy.getSymbolRef());
  if (StructDeclOp structOp = dyn_cast<StructDeclOp>(decl.getIfOperation())) {
    auto [capturedRefs, selfContainedSig] =
        DeclResolver::createSelfContainedSignature(fnSig);
    selfContainedSig =
        cast<FnTypeGeneratorType>(getCanonicalType(selfContainedSig));
    if (!structOp.getDefinesClosure() ||
        structOp.getInputParams().size() != 1 + capturedRefs.size())
      return false;
    auto wrapperImplType = dyn_cast<FuncTypeGeneratorType>(
        structOp.getInputParams().back().getType());
    if (!wrapperImplType ||
        !ClosureEmitter::isTypeRebindableTo(selfContainedSig, wrapperImplType))
      return false;
    return llvm::all_of(
        llvm::zip(capturedRefs,
                  structOp.getInputParams().take_front(capturedRefs.size())),
        [](auto it) {
          auto [capture, param] = it;
          return isEqualCanon(capture.getType(), param.getType());
        });
  }

  return false;
}

void ConversionFailure::addExplanation(MojoInflightDiag &diag) && {
  if (auto *refuted = std::get_if<RefutedConstraints>(&reason))
    attachConstraintNotes(diag, refuted->constraints, "failed");
}

namespace {
/// A struct for collecting the reasons for a non-`yes` answer from
/// `classifyImplicitConversionImpl`.
struct ConversionNonYesReason {
  ConversionFailure rejection;
  SmallVector<ConstraintAttr, 2> unproven;

  void recordUnprovenIfEmpty(SmallVector<ConstraintAttr, 2> constraints) {
    if (unproven.empty())
      unproven = std::move(constraints);
  }
};
} // namespace

/// Shared body of the two public `classifyImplicitConversion` entry points:
/// `reason` non-null means the caller asked why the answer was not `yes`.
///
/// CAUTION: This must line up with `IREmitter::emitImplicitConversionToType`!!!
static TriBool classifyImplicitConversionImpl(
    ASTExprAnd<CValue> value, ASTType requiredType, ASTDecl &declScope,
    ArrayRef<ConstraintAttr> additionalAssumptions,
    DeferredTypingContext *deferralCtx, ConversionNonYesReason *reason) {
  auto &shared = declScope.getShared();
  assert(value.ir && "Should only query valid values");
  ASTType rvType = value.ir.getRValueType();
  // Clear so an early return leaves no stale reason behind.
  if (reason)
    *reason = {};

  // If it already matches, then we're done.
  if (rvType.isEqualCanon(requiredType))
    return TriBool::yes();

  // If the types have the same representation after elaboration then they are
  // implicitly convertible.
  SmallVector<ConstraintAttr, 2> zeroCostUnproven;
  TriBool zeroCost = canZeroCostConvertImpl(
      rvType, requiredType, shared, declScope, additionalAssumptions,
      reason ? &zeroCostUnproven : nullptr);
  if (zeroCost.isTrue())
    return TriBool::yes();
  if (reason && zeroCost.isUnknown())
    reason->recordUnprovenIfEmpty(std::move(zeroCostUnproven));
  // An undecided zero-cost verdict does not stop the other strategies from
  // finding a definitive answer, but if none of them does, undecided is the
  // honest result rather than a flat no.
  bool sawUnknown = zeroCost.isUnknown();

  // Origin values can convert into an OriginSet by becoming a member of the
  // set.  OriginSet is a singleton type, the value carries the origins.
  if (sugarIsa<OriginType>(rvType) && sugarIsa<OriginSetType>(requiredType))
    return TriBool::yes();

  // Check to see if we already cached this convertibility check. If the caller
  // asked why, we use the cached value only if the verdict was true.
  std::optional<bool> cache =
      shared.getCachedImplicitConvertibility(rvType, requiredType);
  if (cache.has_value() && (!reason || cache.value()))
    return TriBool::fromBool(cache.value());

  // Cache and return a convertibility verdict. When `scopeDependent` is true
  // the verdict was derived from this scope's assumptions rather than being a
  // stable function of the (from, to) pair, so it is returned without caching:
  // the cache is keyed only on the type pair and would otherwise poison queries
  // from scopes with a different assumption set. `reason` is recorded on a
  // false verdict; `None` (the default) is a no-op.
  auto cacheAndReturnVal =
      [&shared, reason](ASTType from, ASTType to, bool isConvertible,
                        bool scopeDependent = false,
                        ConversionFailure::Reason rejection = {}) -> TriBool {
    if (!scopeDependent)
      shared.cacheImplicitConvertibility(from, to, isConvertible);
    if (!isConvertible && reason)
      reason->rejection.recordIfEmpty(std::move(rejection));
    return TriBool::fromBool(isConvertible);
  };

  // Handle turning a ConversionResult into a TriBool for returning from this
  // function, or for falling through to the next conversion strategy. Also
  // handles caching & recording the rejection reason if needed.
  auto resolveConversionResult =
      [&](ConversionResult conv) -> std::optional<TriBool> {
    if (!conv.applies())
      return std::nullopt;
    // An undecided verdict is scope-dependent by nature and never cached.
    if (conv.isConvertible.isUnknown())
      return TriBool::unknown();
    if (conv.isCacheable())
      return cacheAndReturnVal(rvType, requiredType,
                               conv.isConvertible.isTrue());
    return conv.isConvertible;
  };

  // Handle turning a ConstraintResult into a TriBool for returning from this
  // function. Also handles caching & recording the rejection reason if needed.
  auto resolveConstraintResult = [&](ConstraintResult result,
                                     bool scopeDependent) -> TriBool {
    if (result.isYes())
      return cacheAndReturnVal(rvType, requiredType, true, scopeDependent);
    if (result.isUnknown()) {
      // An undecided result is never cached. Record and return directly.
      if (reason)
        reason->recordUnprovenIfEmpty(std::move(result.getUnknown()));
      return TriBool::unknown();
    }
    // A failure result may need a rejection reason.
    ConversionFailure::Reason rejection;
    if (reason)
      rejection =
          ConversionFailure::RefutedConstraints{std::move(result.getNo())};
    return cacheAndReturnVal(rvType, requiredType, false, scopeDependent,
                             std::move(rejection));
  };

  // Empty generators are zero-cost convertible to their body type when the
  // generator's body constraints are satisfied by this scope's assumptions.
  // A conversion that "applies" but fails the convertibility check returns
  // false immediately intentionally.
  if (std::optional<TriBool> resolved =
          resolveConversionResult(classifyEmptyGeneratorToBody(
              value, requiredType, declScope, additionalAssumptions)))
    return *resolved;

  bool upCastScopeDependent = false;
  FailureOr<ConstraintResult> canUpCast =
      IREmitter::canMetaTypeUpCastToWithDetails(
          shared, value.expr->getLoc(), rvType, requiredType, &declScope,
          &upCastScopeDependent);
  if (succeeded(canUpCast))
    return resolveConstraintResult(*canUpCast, upCastScopeDependent);

  if (sugarIsa<ParamListType>(rvType) &&
      sugarIsa<ParamListType>(requiredType)) {
    // If the element types of the variadic is meta type (AnyStruct/AnyTrait),
    // we allow them to be implicitly converted.
    //
    // That is, we allow `VariadicOf[Copyable]       -> VariadicOf[AnyType]`
    // and,              `VariadicOf[AnyStruct[xxx]] -> VariadicOf[AnyType]`
    //
    // Notably, this does NOT support implicit conversion between from
    // `Variadic[Int]` to `Variadic[UInt]`
    ASTType toEltTp = sugarCast<ParamListType>(requiredType).getElementType();
    ASTType fromEltTp = sugarCast<ParamListType>(rvType).getElementType();
    // Reuse assumptions from above for variadic element upcast.
    bool eltUpCastScopeDependent = false;
    FailureOr<ConstraintResult> canEltUpCast =
        IREmitter::canMetaTypeUpCastToWithDetails(
            shared, value.expr->getLoc(), fromEltTp, toEltTp, &declScope,
            &eltUpCastScopeDependent);
    if (succeeded(canEltUpCast))
      return resolveConstraintResult(*canEltUpCast, eltUpCastScopeDependent);
  }

  // Support implicit conversions of generator types, including dropping
  // generator body constraints proven by this scope.  This is kept in the same
  // relative order as the generator-conversion path in
  // `emitImplicitConversionToType` so the two stay in lockstep.
  if (auto requiredGenerator = sugarDynCast<GeneratorType>(requiredType)) {
    if (std::optional<TriBool> resolved = resolveConversionResult(
            classifyGeneratorToGenerator(value, requiredGenerator, rvType,
                                         declScope, additionalAssumptions)))
      return *resolved;
  }

  // Functions can implicitly convert to their corresponding closure wrapper.
  // This is distinct from converting to a closure trait.
  if (sugarIsa<FnTypeGeneratorType, FnLiteralTypeGeneratorType>(rvType)) {
    if (auto structMeta =
            sugarDynCast<StructMetaType>(requiredType.extractMetaType())) {
      LIT::StructType structTy = structMeta.getType();
      PValue target = value.ir.getIfPValue();
      if (auto fnLiteral = sugarDynCast<FnLiteralTypeGeneratorType>(rvType))
        target = PValue(fnLiteral.getSymbolConstantAttr());
      if (isClosureWrapperStruct(shared, target, structTy))
        return cacheAndReturnVal(rvType, requiredType, true);
    }
  }

  // We can implicitly convert to the specified type if we can construct it with
  // the value as an implicit conversion.
  //
  // TODO: can we make `canConstructType` working without passing in the ir
  // here? This is the only reason that prevent us from turning
  // `ASTExprAnd<CValue> value` into a `ASTType actualType` in the signature
  // (such that we can ensure type conversion not looking at the value itself
  // for future changes to guarantee referential transparency).
  CallOperands operands(CallSyntax::kImplicitConvert, value.expr,
                        EC_OverloadResolution, {value});
  FailureOr<CalleeResult> result =
      OverloadSet::canConstructType(requiredType, operands, declScope);
  bool isConvertible = succeeded(result) && result->isYes();

  // A negative answer is only definitive if nothing along the way came back
  // undecided.
  if (!isConvertible &&
      (sawUnknown || (succeeded(result) && result->isUnknown())))
    return TriBool::unknown();
  // Must cache the overall value type, not just its stripped down rvType.
  shared.cacheImplicitConvertibility(value.ir.getType(), requiredType,
                                     isConvertible);
  return TriBool::fromBool(isConvertible);
}

TriBool IREmitter::classifyImplicitConversion(
    ASTExprAnd<CValue> value, ASTType requiredType, ASTDecl &declScope,
    ArrayRef<ConstraintAttr> additionalAssumptions,
    DeferredTypingContext *deferralCtx) {
  return classifyImplicitConversionImpl(value, requiredType, declScope,
                                        additionalAssumptions, deferralCtx,
                                        /*reason=*/nullptr);
}

IREmitter::ImplicitConversionResult
IREmitter::classifyImplicitConversionWithDetails(
    ASTExprAnd<CValue> value, ASTType requiredType, ASTDecl &declScope,
    ArrayRef<ConstraintAttr> additionalAssumptions,
    DeferredTypingContext *deferralCtx) {
  ConversionNonYesReason reason;
  TriBool verdict = classifyImplicitConversionImpl(
      value, requiredType, declScope, additionalAssumptions, deferralCtx,
      &reason);
  if (verdict.isTrue())
    return ImplicitConversionResult::yes();
  if (verdict.isFalse())
    return ImplicitConversionResult::no(std::move(reason.rejection));
  return ImplicitConversionResult::unknown(std::move(reason.unproven));
}

FailureOr<PValue>
IREmitter::emitTypeValueUpCastToTrait(ASTExprAnd<CValue> valueExpr,
                                      ASTType toType) {
  ASTType fromType = valueExpr.ir.getRValueType();
  auto emitFnLiteralUpCastToTrait = [&](TypedAttr fnPValue,
                                        AnyTraitType anyTrait) {
    TraitType closureTrait = anyTrait.getTraitType();
    if (auto traitDecl = getClosureTraitDecl(shared, closureTrait)) {
      ASTType structWrapper =
          shared.getClosureEmitter().getConcreteClosureWrapperTypeForFnSymbol(
              declScope, valueExpr.expr->getLoc(), fnPValue);
      if (!structWrapper)
        return PValue();
      (void)shared.getClosureEmitter().augmentWitnessTablesToConformTo(
          structWrapper, traitDecl);
      return emitMetaTypeToTraitConversion(
          {PValue(structWrapper), valueExpr.expr}, closureTrait);
    }
    // FnTypeGeneratorType is still a non-struct type...
    return bindNonStructTypeToTrait(valueExpr, anyTrait.getTraitType());
  };
  // Emit metatype conversions to trait types if the metatype implements
  // the specified trait.
  if (auto anyTrait = sugarDynCast<AnyTraitType>(toType.extractMetaType())) {
    TraitType trait = anyTrait.getTraitType();
    if (sugarIsa<NonStructTypeType>(fromType)) {
      // Conversions from MLIR types.
      return bindNonStructTypeToTrait(valueExpr, trait);
    }

    if (sugarIsa<StructMetaMetaType, AnyTraitType>(
            fromType.extractMetaType())) {
      // Augment the witness table of closure wrapper with rebind if
      // necessary. We do this for every closure trait in the type.
      for (const auto &symbol : trait.getSymbols()) {
        auto &symbolDecl =
            shared.declResolver->getDeclForTypeSymbol(symbol.getSymbol());
        if (auto traitDeclOp =
                dyn_cast_if_present<TraitDeclOp>(symbolDecl.getIfOperation());
            traitDeclOp && traitDeclOp.getDefinesClosure()) {
          if (failed(shared.getClosureEmitter().augmentWitnessTablesToConformTo(
                  fromType, &symbolDecl))) {
            // Augmentation failed. Only treat this as fatal for a *genuine*
            // trait-to-trait extension: a trait-view source that does not
            // already expose this (structurally compatible but distinct) trait
            // symbol. Failing here lets the caller's extension branch
            // synthesize the `#kgen.extension`. When the source already exposes
            // the trait (subset conversion, e.g. `A & B` -> `B`) or is a
            // struct, fall through to the normal conversion below (baseline
            // behavior).
            if (auto fromTrait =
                    sugarDynCast<AnyTraitType>(fromType.extractMetaType());
                fromTrait && !llvm::is_contained(
                                 fromTrait.getTraitType().getSymbols(), symbol))
              return failure();
          }
        }
      }
      // Conversions from structs or traits.
      return emitMetaTypeToTraitConversion(valueExpr, trait);
    }

    if (auto fnGen =
            sugarDynCastIfPresent<FnLiteralTypeGeneratorMetaType>(fromType)) {
      return emitFnLiteralUpCastToTrait(fnGen.getType().getSymbolConstantAttr(),
                                        anyTrait);
    }
  }

  // We can convert from AnyTraitType[Derived] to AnyTraitType[Base].
  // This is a conversion of things like "the Movable type" (which has
  // type "AnyTraitType[Movable]") to "AnyTraitType[AnyType]".
  if (auto anyTrait = sugarDynCast<AnyTraitType>(toType)) {
    if (auto fnGen = sugarDynCastIfPresent<FnLiteralTypeGeneratorMetaMetaType>(
            fromType)) {
      auto closureMetaType = emitFnLiteralUpCastToTrait(
          fnGen.getType().getType().getSymbolConstantAttr(), anyTrait);
      if (!closureMetaType)
        return failure();
      return PValue(TypeParamAttr::get(closureMetaType.getType(), anyTrait));
    }

    PValue typePValue = valueExpr.ir.getIfPValue();
    if (!typePValue) {
      emitError(valueExpr.expr->getLoc(),
                "existentials are not supported yet!");
      return PValue();
    }

    ASTType concreteType;
    if (auto rvAnyTrait = sugarDynCast<AnyTraitType>(fromType)) {
      concreteType = ASTType(rvAnyTrait.getTraitType());
    } else if (auto mmType = sugarDynCast<StructMetaMetaType>(fromType)) {
      concreteType = ASTType(mmType.getType());
    }

    if (concreteType) {
      // Augment the witness table of a closure wrapper with a rebind if
      // necessary, mirroring the AnyTraitType-metatype branch above.
      for (const auto &symbol : anyTrait.getTraitType().getSymbols()) {
        auto &symbolDecl =
            shared.declResolver->getDeclForTypeSymbol(symbol.getSymbol());
        if (auto traitDeclOp =
                dyn_cast_if_present<TraitDeclOp>(symbolDecl.getIfOperation());
            traitDeclOp && traitDeclOp.getDefinesClosure()) {
          (void)shared.getClosureEmitter().augmentWitnessTablesToConformTo(
              concreteType, &symbolDecl);
        }
      }

      if (concreteType
              .doesConformTo(anyTrait.getTraitType(), shared,
                             ASTDecl::getAssumptionsFromScope(&declScope))
              .isTrue()) {
        // This is just the trait itself, not a conformance, just upcast.
        return PValue(TypeParamAttr::get(ASTType(typePValue), anyTrait));
      }
    }
  }

  // Not applicable
  return failure();
}

static ASTDecl *getFileModuleForValue(SharedState &shared, ASTDecl &declScope,
                                      const CValue &value) {
  if (ASTDecl *fileModule = declScope.getNearestDeclOfType<FileModuleOp>())
    return fileModule;

  Value mlirValue = value.getMlirValue();
  if (!mlirValue)
    return nullptr;

  Operation *op = mlirValue.getDefiningOp();
  if (!op) {
    if (Block *block = mlirValue.getParentBlock())
      op = block->getParentOp();
  }
  if (!op)
    return nullptr;

  auto fileMod = op->getParentOfType<FileModuleOp>();
  if (!fileMod)
    return nullptr;

  SymbolRefAttr fileSym = getFullyResolvedSymbolRef(
      cast<mlir::SymbolOpInterface>(fileMod.getOperation()));
  return shared.getDeclResolver().getDeclForTypeSymbolIfExists(fileSym);
}

/// This emits an implicit conversion to the specified type if the types
/// differ, including emitting any implicit constructor calls as well as
/// implicit promotions like origin conversions.
///
/// CAUTION: This method must line up with `canImplicitlyConvertToType`!!!
CValue IREmitter::emitImplicitConversionToType(
    ASTExprAnd<CValue> valueExpr, ASTType requiredType, ExprDest &dest,
    ArrayRef<ConstraintAttr> additionalAssumptions) {
  CValue value = valueExpr.ir;
  const ExprNode *expr = valueExpr.expr;

  // If converting to or from a TypeCheckError type, then there is an
  // already-diagnosed error about this expression.
  auto rvType = value.getRValueType();
  if (rvType.isTypeCheckErrorType() || requiredType.isTypeCheckErrorType()) {
    dest.resetForError(*this);
    return {};
  }

  // If the types are already identical, then we're done.
  if (requiredType.isEqualCanon(rvType))
    return emitCResult(value, expr, dest);

  // If we are dealing with types that differ only pre-elaboration,
  // we insert a rebind or equivalent.
  if (canZeroCostConvert(rvType, requiredType, additionalAssumptions)
          .isTrue()) {
    value = emitZeroCostConvert({value, expr}, requiredType);
    return emitCResult(value, expr, dest);
  }

  if (canConvertEmptyGeneratorToBody(valueExpr, requiredType, declScope,
                                     additionalAssumptions))
    return convertEmptyGeneratorToBody(valueExpr, requiredType, *this, dest);

  // Handle conversions between origins and origin sets.
  if (sugarIsa<OriginType>(rvType) && sugarIsa<OriginSetType>(requiredType)) {
    // This can only be done in the parameter domain.
    if (TypedAttr pv = value.getIfPValue()) {
      pv = OriginSetAttr::get(pv, sugarCast<OriginSetType>(requiredType));
      return emitCResult(pv, expr, dest);
    }
  }
  if (sugarIsa<OriginSetType>(rvType) && sugarIsa<OriginType>(requiredType)) {
    // This can only be done in the parameter domain.
    if (TypedAttr pv = value.getIfPValue()) {
      pv = OriginSetUnionAttr::get(pv, sugarCast<OriginType>(requiredType));
      return emitCResult(pv, expr, dest);
    }
  }

  FailureOr<PValue> typeValueCast =
      emitTypeValueUpCastToTrait(valueExpr, requiredType);
  // This handles nullptr case too.
  if (succeeded(typeValueCast))
    return emitCResult(*typeValueCast, expr, dest);

  // Conversions from function pointers to closures.
  if (sugarIsa<FnTypeGeneratorType, FnLiteralTypeGeneratorType>(rvType)) {
    // Functions can implicitly convert to their corresponding closure wrapper.
    if (auto structMeta =
            sugarDynCast<StructMetaType>(requiredType.extractMetaType())) {
      StructType structTy = structMeta.getType();
      auto target = valueExpr.ir.getIfPValue();
      if (auto fnLiteral = sugarDynCast<FnLiteralTypeGeneratorType>(rvType))
        target = PValue(fnLiteral.getSymbolConstantAttr());
      if (isClosureWrapperStruct(shared, target, structTy)) {
        return emitConstructorCall(
            structTy,
            CallOperands(CallSyntax::kTypeCall, expr, std::move(dest), {}));
      }
    }
  }

  // Extend one parametric closure trait to a structurally compatible one.
  if (auto anyTrait =
          sugarDynCast<AnyTraitType>(requiredType.extractMetaType())) {
    if (sugarIsa<AnyTraitType>(rvType.extractMetaType())) {
      if (std::optional<TraitDeclOp> targetTrait =
              ClosureEmitter::getClosureDecl(shared, anyTrait.getTraitType())) {
        ASTDecl *fileModule =
            getFileModuleForValue(shared, getDeclScope(), valueExpr.ir);
        if (fileModule) {
          if (CValue extension = shared.getClosureEmitter().createExtensionType(
                  *fileModule, valueExpr.ir, anyTrait.getTraitType(),
                  *targetTrait))
            return extension;
        }
      }
    }
  }

  if (sugarIsa<ParamListType>(rvType) &&
      sugarIsa<ParamListType>(requiredType)) {
    auto emitVariadicError = [&]() -> CValue {
      shared.emitError(valueExpr.expr->getLoc(), "can not convert ")
          << rvType << " to " << requiredType << valueExpr.expr->getRange();
      dest.resetForError(*this);
      return {};
    };

    auto dstVATp = sugarCast<ParamListType>(requiredType);
    ASTType fromEltTp = sugarCast<ParamListType>(rvType).getElementType();
    ASTType toEltTp = dstVATp.getElementType();
    TypedAttr srcVal = valueExpr.ir.getIfPValue().get();
    if (auto vVal = sugarDynCast<ParamListAttr>(srcVal)) {
      SmallVector<TypedAttr> converted;
      for (auto elt : vVal.getValues()) {
        if (!LIT::isTypeExpr(elt))
          return emitVariadicError();

        // TODO(MOCO-2742): overwriting the type below should not be necessary.
        fromEltTp = ASTType(elt).extractMetaType();
        if (fromEltTp.mlirType != toEltTp.mlirType) {
          FailureOr<PValue> castToOr =
              emitTypeValueUpCastToTrait({elt, valueExpr.expr}, toEltTp);
          if (failed(castToOr) || castToOr->isNull())
            return {};
          converted.push_back(*castToOr);
        } else {
          // Simple case such as: !Int : !AnyType -> !Int: !mt_Int
          converted.push_back(TypeParamAttr::get(ASTType(elt), fromEltTp));
        }
      }
      return emitCResult(ParamListAttr::get(converted, dstVATp), expr, dest);
    } else {
      // Must match the check in canImplicitlyConvertToType.
      FailureOr<TriBool> canUpCast =
          canMetaTypeUpCastTo(shared, valueExpr.expr->getLoc(), fromEltTp,
                              toEltTp, &getDeclScope());
      if (failed(canUpCast) || !canUpCast->isTrue())
        return emitVariadicError();

      // The source is not resolved yet, this is a simple upcast.
      // For example, we upcast a variadic of `Copyable`s to `AnyTypes` by
      // `#upcast<:param_list<!Copyable> T> :!param_list<!AnyType>`
      return emitCResult(UpcastAttr::get(requiredType, srcVal), expr, dest);
    }
  }

  // Support implicit conversions of generator types (incl. function
  // generators).
  if (auto requiredGenerator = sugarDynCast<GeneratorType>(requiredType)) {
    if (auto rvGeneratorType = sugarDynCast<GeneratorType>(rvType))
      if (canConvertGeneratorTypes(valueExpr, rvGeneratorType,
                                   requiredGenerator, declScope,
                                   additionalAssumptions)
              .isTrue())
        return convertGeneratorValue(value, expr, requiredGenerator, *this,
                                     dest, additionalAssumptions);
  }

  // Before we try emitting constructor call, makes sure that the required type
  // is concrete.
  if (llvm::find_if(requiredType.getParamBindings(), [](TypedAttr param) {
        return isa<UnboundAttr>(param);
      }) != requiredType.getParamBindings().end()) {
    emitError(expr->getLoc())
        << "cannot construct a value with parametric type: " << requiredType
        << expr->getRange();
    dest.resetForError(*this);
    return {};
  }
  //  We disable implicit conversions to prevent converting T -> S -> U in
  //  one step, and to avoid infinite conversion cycles.
  return emitConstructorCall(requiredType,
                             CallOperands(CallSyntax::kImplicitConvert, expr,
                                          std::move(dest), {valueExpr}));
}

TypedAttr IREmitter::emitStringExprAsDataToStr(CValue val, ExprNode *expr,
                                               SMLoc loc, ExprContext context) {
  if (!val)
    return {};

  // If the value is a t-string, convert it to String first. TString cannot be
  // implicitly converted to StringSpan, but String has an explicit constructor
  // that accepts TString.
  if (isa<TStringExprNode>(expr)) {
    auto stringType = shared.lookupBuiltinType("String", declScope, loc)
                          .getWithoutParameters(shared);
    if (val.getType().getDecl(shared) != stringType.getDecl(shared)) {
      val = emitConstructorCall(
          stringType,
          CallOperands(CallSyntax::kTypeCall, expr, context, {{val, expr}}));
      if (!val)
        return {};
    }
  }

  // Convert to StringSpan if not already.
  auto stringSpanType = shared.lookupBuiltinType("StringSpan", declScope, loc)
                            .getWithoutParameters(shared);
  if (val.getType().getDecl(shared) != stringSpanType.getDecl(shared)) {
    if (!canImplicitlyConvertToType({val, expr}, stringSpanType, declScope)) {
      emitError(expr->getLoc()) << "cannot implicitly convert " << val.getType()
                                << " to a `StringSpan`";
      return {};
    }

    val = emitConstructorCall(
        stringSpanType,
        CallOperands(CallSyntax::kTypeCall, expr, context, {{val, expr}}));
    if (!val)
      return {};
  }

  // Emit as a parameter value and wrap in DataToStr.
  PValue pval = emitPValue({val, expr}, context);
  if (!pval)
    return {};
  return ParamOperatorAttr::get(
      POC::DataToStr,
      {pval, ParamListAttr::get({}, ParamListType::get(pval.getType()))});
}
