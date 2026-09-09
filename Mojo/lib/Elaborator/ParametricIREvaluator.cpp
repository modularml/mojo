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

#include "ParametricIREvaluator.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/POPDialect/POPAttrs.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/POPDialect/POPUtils.h"
#include "Mojo/TransformUtils/ManglingUtils.h"
#include "Mojo/lib/Elaborator/IREvaluatorContext.h"
#include "ParametricElaborator.h"
#include "Support/Compiler/DiagnosticHandler.h"
#include "Support/MDialect/MTypeInterfaces.h"
#include "Support/StringExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/ScopeExit.h"

#include <regex>

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// IR Interpreter
//===----------------------------------------------------------------------===//

ErrorOr<std::pair<Region *, Operation *>>
ParametricIREvaluator::lookupParametricFunctionBody(SymbolRefAttr symbol) {
  StringAttr name = cast<FlatSymbolRefAttr>(symbol).getAttr();
  if (GeneratorOpInterface gen = elaborator->lookupGeneratorOp(name)) {
    return std::make_pair(&gen.getBodyRegion(), gen.getOperation());
  }

  InstantiatedOpInterface inst = elaborator->lookupInstantiatedOp(name);
  FuncOp func = cast<FuncOp>(inst);

  // Now we can return the function body.
  return std::make_pair(&func.getBodyRegion(), nullptr);
}

ErrorOr<Region *>
ParametricIREvaluator::lookupFunctionBody(SymbolRefAttr symbol) {
  ErrorOr<std::pair<Region *, Operation *>> result =
      lookupParametricFunctionBody(symbol);
  if (result.isError())
    return result.takeError();

  return result->first;
}

ErrorOr<Type>
ParametricIREvaluator::lookupFuncTypeGenerator(SymbolRefAttr symbol) {
  StringAttr name = cast<FlatSymbolRefAttr>(symbol).getAttr();
  if (GeneratorOpInterface itf = elaborator->lookupGeneratorOp(name)) {
    if (auto gen = dyn_cast<GeneratorOp>(itf.getOperation())) {
      return gen.getFuncTypeGenerator();
    }
    return Error("cannot find FuncTypeGeneratorType");
  }

  InstantiatedOpInterface inst = elaborator->lookupInstantiatedOp(name);
  FuncOp func = cast<FuncOp>(inst);
  return func.getFuncTypeGenerator();
}

ErrorTreeOr<TypedAttr> ParametricIREvaluator::interpretGeneratorWithResultSlot(
    Attribute calleeAttr, llvm::ArrayRef<TypedAttr> paramValues,
    ArrayRef<Attribute> arguments, Location loc) {

  ParametricIREvaluator nestedEvaluator(*this);
  nestedEvaluator.setErrorLoc(loc);
  auto callee = extractSymbolConstantAttr(cast<TypedAttr>(calleeAttr));
  ErrorOr<std::pair<Region *, Operation *>> bodyOr =
      nestedEvaluator.lookupParametricFunctionBody(callee.getSymbol());

  if (bodyOr.isError())
    return ErrorTree(loc, bodyOr.takeError());
  Region &body = *bodyOr->first;
  Operation *op = bodyOr->second;

  SmallVector<TypedAttr> operands(arguments.size());
  for (auto [idx, arg] : llvm::enumerate(arguments))
    operands[idx] = cast<TypedAttr>(arg);

  auto operandsAttr =
      ParameterExprArrayAttr::get(calleeAttr.getContext(), operands);

  TypedAttr cached =
      elaborator->lookupCachedInterpretation(op, operandsAttr, callee);
  if (!cached) {
    nestedEvaluator.pushParamValues(paramValues, true);
    nestedEvaluator.setDeclBindings(op, paramValues);

    Type resultPtrType =
        nestedEvaluator.getReboundType(body.getArguments().back().getType());
    auto ptr = dyn_cast<PointerType>(resultPtrType);
    Type elemType = ptr.getElementType();

    // Evaluate the iterator function body.
    auto result = nestedEvaluator.executeRegionWithResultSlot(
        body, arguments, elemType, resultPtrType);
    if (result.isError())
      return result.takeError();

    cached = result.takeValue();

    (void)elaborator->writeGlobalCachedInterpretation(op, operandsAttr, callee,
                                                      cached);
  }
  return cached;
}

ErrorTreeOr<TypedAttr> ParametricIREvaluator::interpretGenerator(
    Attribute calleeAttr, llvm::ArrayRef<TypedAttr> paramValues,
    ArrayRef<Attribute> arguments, Location loc) {

  ParametricIREvaluator nestedEvaluator(*this);
  nestedEvaluator.setErrorLoc(loc);

  auto callee = extractSymbolConstantAttr(cast<TypedAttr>(calleeAttr));
  auto bodyOr =
      nestedEvaluator.lookupParametricFunctionBody(callee.getSymbol());
  if (bodyOr.isError())
    return ErrorTree(loc, bodyOr.takeError());
  Region &body = *bodyOr->first;
  Operation *op = bodyOr->second;

  SmallVector<TypedAttr> operands(arguments.size());
  for (auto [idx, arg] : llvm::enumerate(arguments))
    operands[idx] = cast<TypedAttr>(arg);

  auto operandsAttr =
      ParameterExprArrayAttr::get(calleeAttr.getContext(), operands);

  TypedAttr cached =
      elaborator->lookupCachedInterpretation(op, operandsAttr, callee);

  if (!cached) {
    nestedEvaluator.pushParamValues(paramValues, true);
    nestedEvaluator.setDeclBindings(bodyOr->second, paramValues);

    auto result = nestedEvaluator.executeRegion(body, arguments);
    if (result.isError())
      return result.takeError();
    cached = cast<TypedAttr>(result.getValue().front());
    (void)elaborator->writeGlobalCachedInterpretation(op, operandsAttr, callee,
                                                      cached);
  }

  return cached;
}

void ParametricIREvaluator::setDeclBindings(Operation *op,
                                            ArrayRef<TypedAttr> paramValues) {
  if (auto gen = dyn_cast<GeneratorOpInterface>(op)) {
    for (auto [decl, attr] : llvm::zip(gen.getInputParams(), paramValues)) {
      getCurrentParamEvalFrame().evaluator.overwriteDeclBinding(decl, attr);
    }
  }
}

void ParametricIREvaluator::clearParameterCache() {}

void ParametricIREvaluator::pushEvalFrame(Operation *op, Region *region,
                                          llvm::ArrayRef<TypedAttr> paramValues,
                                          int id) {
  ParameterEvaluatorFrame curr(paramEvalFrames.back().evaluator);
  curr.evaluator.setEvaluationContext(this);

  ParameterExprArrayAttr paramValueAttr;
  SmallVector<TypedAttr> pValues;

  if (!frameParamInfos.empty()) {
    paramValueAttr = ParameterExprArrayAttr::get(
        op->getContext(), frameParamInfos.back().paramValues);
    pValues = frameParamInfos.back().paramValues;
  }

  curr.cachedOpKey = op;
  curr.cachedRegionKey = region;
  curr.cachedAttrKey = paramValueAttr;

  auto tlIter = elaborator->tlParamInterpCache->find({region, paramValueAttr});

  if (tlIter != elaborator->tlParamInterpCache->end()) {
    curr.evaluator.setRewritten(tlIter->second);
    curr.foundCached = true;
  } else {
    std::optional<DenseMap<std::pair<size_t, const void *>, const void *>>
        result = elaborator->paramInterpCache.read(
            [region, paramValueAttr](auto &map)
                -> std::optional<
                    DenseMap<std::pair<size_t, const void *>, const void *>> {
              auto result = map.find({region, paramValueAttr});
              if (result != map.end()) {
                return result->second;
              }
              return std::nullopt;
            });

    if (result) {
      curr.evaluator.setRewritten(std::move(*result));
      curr.foundCached = true;
    } else {
      bool clearCache = !isa<ParamIfOp>(op);
      if (auto gen = dyn_cast<GeneratorOpInterface>(op)) {
        FunctionParameterUseDefGraph &g = *elaborator->knownGraphs.get()[gen];
        clearCache = g.hasParams || !gen.getInputParams().empty();
      }
      if (clearCache)
        curr.evaluator.clearCache();
    }
  }

  paramEvalFrames.push_back(std::move(curr));
}

void ParametricIREvaluator::popEvalFrame() {
  ParameterEvaluatorFrame &back = paramEvalFrames.back();
  if (!back.foundCached && back.cachedRegionKey) {
    auto &rewritten = back.evaluator.getRewritten();
    if (!rewritten.empty()) {
      (*elaborator
            ->tlParamInterpCache)[{back.cachedRegionKey, back.cachedAttrKey}] =
          rewritten;
      elaborator->paramInterpCache.modify([back, rewritten](auto &map) {
        map.insert({{back.cachedRegionKey, back.cachedAttrKey}, rewritten});
      });
    }
  }

  paramEvalFrames.pop_back();
}

void ParametricIREvaluator::popEvalFrame(size_t size) {
  assert(paramEvalFrames.size() >= size && "popEvalFrame failed!");
  paramEvalFrames.erase(paramEvalFrames.begin() + size - 1,
                        paramEvalFrames.end());
}

void ParametricIREvaluator::pushParamValues(llvm::ArrayRef<TypedAttr> values,
                                            bool pushFrame, Operation *op) {
  if (pushFrame)
    frameParamInfos.emplace_back(FrameParamInfo{});

  FrameParamInfo &currFrame = frameParamInfos.back();

  currFrame.numParamsPerScope.emplace_back(std::make_pair(op, values.size()));
  currFrame.paramValues.append(values.begin(), values.end());
}

DenseSet<Operation *> *ParametricIREvaluator::getParamOps(Operation *op,
                                                          std::string &name) {
  auto &map = elaborator->knownGraphs.get();
  if (auto gen = dyn_cast<GeneratorOpInterface>(op)) {
    assert(map.contains(gen));
    return &map[gen]->paramOpsSet;
  }
  return nullptr;
}

void ParametricIREvaluator::setIsCurrOpParam(Operation *op) {
  isCurrOpParam = isa<ParamOpInterface>(op) || !stack.back().paramOps ||
                  stack.back().paramOps->contains(op);
}

void ParametricIREvaluator::popParamValues(bool popFrame, Operation *op,
                                           Operation *tillOp) {
  if (popFrame) {
    if (!frameParamInfos.empty())
      frameParamInfos.pop_back();
    return;
  }

  FrameParamInfo &currFrame = frameParamInfos.back();

  Operation *currOp = nullptr;

  do {
    assert(currFrame.numParamsPerScope.size() > 0 &&
           "popParamValues has wrong number of param values to pop");
    size_t numValuesToPop = currFrame.numParamsPerScope.back().second;
    currOp = currFrame.numParamsPerScope.back().first;

    assert(currFrame.paramValues.size() >= numValuesToPop &&
           "popParamValues has not enough param values to pop");

    currFrame.paramValues.erase(currFrame.paramValues.end() - numValuesToPop,
                                currFrame.paramValues.end());

    currFrame.numParamsPerScope.pop_back();
  } while (tillOp && currOp != tillOp);
}

void ParametricIREvaluator::appendParamValues(llvm::ArrayRef<TypedAttr> values,
                                              int id, Operation *op) {
  assert(frameParamInfos.size() > 0 && "no more frames to append to");
  FrameParamInfo &currFrame = frameParamInfos.back();

  currFrame.paramValues.append(values.begin(), values.end());
  if (currFrame.numParamsPerScope.empty())
    currFrame.numParamsPerScope.emplace_back(std::make_pair(op, values.size()));
  else
    currFrame.numParamsPerScope.back().second += values.size();
}

ErrorTreeOr<TypedAttr>
ParametricIREvaluator::evaluateFunction(FuncOp func,
                                        ArrayRef<TypedAttr> inputs) {
  if constexpr (KGEN::kIsTracingEnabled)
    auto ts = InterpreterTimeTraceScope("Launch interpreter",
                                        mlir::debugString(errorLoc));

  // Evaluate the function body.
  SmallVector<Attribute> arguments;
  for (TypedAttr input : inputs)
    arguments.push_back(input);
  ErrorTreeOr<SmallVector<Attribute>> result =
      executeRegion(func.getBodyRegion(), arguments);

  // Report an error if evaluation fails.
  if (result.isError()) {
    return ErrorTree(*errorLoc, "failed to compile-time evaluate function call",
                     result.takeError());
  }

  // Apply operators only return one result.
  return cast<TypedAttr>(result.getValue().front());
}

ErrorTreeOr<TypedAttr>
ParametricIREvaluator::evaluateGenerator(GeneratorOp func,
                                         ArrayRef<TypedAttr> inputs) {
  // Evaluate the function body.
  SmallVector<Attribute> arguments;
  for (TypedAttr input : inputs)
    arguments.push_back(input);

  std::optional<ErrorTree> error;
  if (!emitError)
    emitError = [&](ErrorTree err) { error = std::move(err); };

  ErrorTreeOr<SmallVector<Attribute>> result =
      executeRegion(func.getBodyRegion(), arguments);

  // Report an error if evaluation fails.
  if (result.isError()) {
    return ErrorTree(*errorLoc, "failed to compile-time evaluate function call",
                     result.takeError());
  } else if (error) {
    return ErrorTree(*errorLoc, "failed to compile-time evaluate function call",
                     std::move(*error));
  }

  // Apply operators only return one result.
  return cast<TypedAttr>(result.getValue().front());
}

ErrorTreeOr<TypedAttr> ParametricIREvaluator::evaluateGeneratorWithResultSlot(
    GeneratorOp func, ArrayRef<TypedAttr> inputs) {
  if constexpr (KGEN::kIsTracingEnabled)
    auto ts = InterpreterTimeTraceScope("Launch interpreter",
                                        mlir::debugString(errorLoc));

  // Evaluate the function body.
  SmallVector<Attribute> arguments;
  for (TypedAttr input : inputs)
    arguments.push_back(input);

  std::optional<ErrorTree> error;
  if (!emitError)
    emitError = [&](ErrorTree err) { error = std::move(err); };

  auto resultArg = func.getArguments().back();
  auto ptr = dyn_cast<PointerType>(resultArg.getType());
  if (!ptr)
    return ErrorTree(func.getLoc(), "result argument is not a pointer");

  Type elemType = getReboundType(ptr.getElementType());
  Type resultPtrType =
      getReboundType(func.getBodyRegion().getArguments().back().getType());

  ErrorTreeOr<TypedAttr> result = executeRegionWithResultSlot(
      func.getBodyRegion(), arguments, elemType, resultPtrType);

  // Report an error if evaluation fails.
  if (result.isError()) {
    return ErrorTree(
        *errorLoc,
        "failed to compile-time evaluate generator with resultslot call",
        result.takeError());
  } else if (error) {
    return ErrorTree(
        *errorLoc,
        "failed to compile-time evaluate generator with resultslot call",
        std::move(*error));
  }

  // Apply operators only return one result.
  return result.takeValue();
}

ErrorTreeOr<TypedAttr> ParametricIREvaluator::evaluateFunctionWithResultSlot(
    FuncOp func, ArrayRef<TypedAttr> inputs) {
  if constexpr (KGEN::kIsTracingEnabled)
    auto ts = InterpreterTimeTraceScope("Launch interpreter",
                                        mlir::debugString(errorLoc));

  // Evaluate the function body.
  SmallVector<Attribute> arguments;
  for (TypedAttr input : inputs)
    arguments.push_back(input);

  auto resultArg = func.getArguments().back();
  auto ptr = dyn_cast<PointerType>(resultArg.getType());
  if (!ptr)
    return ErrorTree(func.getLoc(), "result argument is not a pointer");
  ErrorTreeOr<TypedAttr> result = executeRegionWithResultSlot(
      func.getBodyRegion(), arguments, ptr.getElementType(),
      func.getBodyRegion().getArguments().back().getType());

  // Report an error if evaluation fails.
  if (result.isError()) {
    return ErrorTree(
        *errorLoc,
        "failed to compile-time evaluate function with resultslot call",
        result.takeError());
  }

  // Apply operators only return one result.
  return result.takeValue();
}

int ParametricIREvaluator::getErrorLimit() {
  return elaborator->options.elaborationErrorLimit;
}

bool ParametricIREvaluator::getElabErrorIncludePrelude() {
  return elaborator->options.elaborationErrorIncludePrelude;
}

//===----------------------------------------------------------------------===//
// Expression Evaluation
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr> ParametricIREvaluator::evaluateContextSpecific(
    ContextuallyEvaluatedAttrInterface attr) {
  // Don't try to evaluate a parameter operator that still contains parametric
  // things in it, since it may be transitory.
  if (EscapingReferenceFinder::check(attr))
    return cast<TypedAttr>(attr);

  if (auto genref = dyn_cast<TypeGeneratorRefAttr>(attr)) {
    // Attempt to concretize the function first.
    ErrorTreeOr<TypeInstanceRefAttr> symOr =
        elaborator->getConcreteStructTypeReference(parent, *errorLoc, genref);
    if (symOr.isError()) {
      emitError(symOr.takeError());
      return failure();
    }
    return cast<TypedAttr>(symOr.takeValue());
  }

  if (auto getLinkageNameAttr = dyn_cast<GetLinkageNameAttr>(attr))
    return evaluateGetLinkageNameAttr(getLinkageNameAttr);
  if (auto getSourceNameAttr = dyn_cast<GetSourceNameAttr>(attr))
    return evaluateGetSourceNameAttr(getSourceNameAttr);
  if (auto getTypeNameAttr = dyn_cast<GetTypeNameAttr>(attr))
    return evaluateGetTypeNameAttr(getTypeNameAttr);
  if (auto isStructTypeAttr = dyn_cast<IsStructTypeAttr>(attr))
    return evaluateIsStructTypeAttr(isStructTypeAttr);
  if (auto fnTypeIsCABIAttr = dyn_cast<FnTypeIsCABIAttr>(attr))
    return evaluateFnTypeIsCABIAttr(fnTypeIsCABIAttr);
  if (auto getBaseTypeNameAttr = dyn_cast<GetBaseTypeNameAttr>(attr))
    return evaluateGetBaseTypeNameAttr(getBaseTypeNameAttr);

  if (auto compileOffloadClosureAttr =
          dyn_cast<CompileOffloadClosureAttr>(attr))
    return evaluateCompileOffloadClosureAttr(compileOffloadClosureAttr);
  if (auto compileAssemblyAttr = dyn_cast<CompileAssemblyAttr>(attr))
    return evaluateCompileAssemblyAttr(compileAssemblyAttr);

  if (auto castAttr = dyn_cast<POP::CastAttr>(attr)) {
    auto outType = cast<SIMDType>(castAttr.getType());
    auto inType = cast<SIMDType>(castAttr.getArg().getType());
    if (auto fold =
            POP::foldCast(castAttr.getArg(), outType, inType, outType,
                          elaborator->getTarget().resolveIndexBitWidth())) {
      return cast<TypedAttr>(cast<Attribute>(fold));
    }
    emitError(ErrorTree(*errorLoc, "Unable to evaluate #pop.cast attribute"));
    return failure();
  }

  // Must be a parameter operator then. Otherwise, send back to the base class.
  auto op = dyn_cast<ParamOperatorAttr>(attr);
  if (!op)
    return failure();

  // Try to narrow this operator to an expression we can evaluate. We only need
  // to emit an error during the evaluation attempt.
  switch (op.getOpcode()) {
  case POC::CurrentTarget:
    // Retrieve the contextual compilation target info.
    return {TargetParamAttr::get(elaborator->getTarget())};
  case POC::AcceleratorArch:
    return {StringAttr::get(elaborator->options.targetAccelerator,
                            StringType::get(op.getContext()))};
  case POC::CrossCompilation:
    return {
        BoolAttr::get(op.getContext(), elaborator->options.isCrossCompilation)};
  case POC::GetEnv:
    return evaluateGetEnv(op);
  case POC::Apply:
    return evaluateApplyLike(op, /*withResultSlot=*/false);
  case POC::ApplyResultSlot:
    return evaluateApplyLike(op, /*withResultSlot=*/true);
  case POC::AttrToStr:
    return {StringAttr::get(mlir::debugString(op.getOperands().front()),
                            StringType::get(op.getContext()))};
  case POC::DataToStr:
    return evaluateDataToStr(op, /*reset=*/false);
  // Evaluate str_concat to support recursive string chains in deferred types.
  case POC::StrConcat: {
    // Succeeds with a null StringAttr when the operand is not ready yet, so
    // the caller can defer instead of reporting an unfoldable concatenation.
    auto evalToString = [&](TypedAttr attr) -> FailureOr<StringAttr> {
      attr = SugarAttr::strip(attr);
      if (auto strAttr = dyn_cast<StringAttr>(attr))
        return strAttr;
      if (auto evalAttr = dyn_cast<ContextuallyEvaluatedAttrInterface>(attr)) {
        FailureOr<TypedAttr> result = evaluateExpression(evalAttr);
        if (succeeded(result)) {
          if (!*result)
            return StringAttr();
          if (auto strAttr = dyn_cast<StringAttr>(*result))
            return strAttr;
        }
      }
      return failure();
    };
    FailureOr<StringAttr> lhs = evalToString(op.getOperands()[0]);
    FailureOr<StringAttr> rhs = evalToString(op.getOperands()[1]);
    if (failed(lhs) || failed(rhs))
      return failure();
    if (!*lhs || !*rhs)
      return TypedAttr();
    SmallString<64> buf;
    buf.append(lhs->strref());
    buf.append(rhs->strref());
    return TypedAttr{StringAttr::get(buf, StringType::get(op.getContext()))};
  }
  case POC::StringAddress:
    return evaluateStringAddress(op);
  case POC::Rebind:
    // Catch unfolded rebinds to emit a nicer error message.
    emitError(ErrorTree(
        *errorLoc, "error: rebind input type '" +
                       mlir::debugString(op.getOperands().front().getType()) +
                       "' does not match result type '" +
                       mlir::debugString(op.getType()) + "'"));
    return failure();
  case POC::LoadFromMem:
    if (auto memref = dyn_cast<MemRefAttr>(op.getOperands().front())) {
      ErrorOr<TypedAttr> value = loadAttributeFromMemRef(memref, op.getType());
      if (value.isError()) {
        emitError({*errorLoc, value.takeError()});
        return failure();
      }
      return value.takeValue();
    }
    return failure();
  case POC::Div:
  case POC::DivS:
  case POC::DivU: {
    Attribute operands[] = {op.getOperands()[0], op.getOperands()[1]};
    return foldAttrWithTarget(*this, operands, POP::foldSIMDDiv);
  }
  case POC::FloorDivS: {
    Attribute operands[] = {op.getOperands()[0], op.getOperands()[1]};
    return foldAttrWithTarget(*this, operands, POP::foldSIMDFloorDiv);
  }
  case POC::Shr: {
    Attribute operands[] = {op.getOperands()[0], op.getOperands()[1]};
    return foldAttrWithTarget(*this, operands, POP::foldSIMDShr);
  }
  case POC::Shl: {
    Attribute operands[] = {op.getOperands()[0], op.getOperands()[1]};
    return foldAttrWithTarget(*this, operands, POP::foldSIMDShl);
  }
  case POC::EQ:
    // Non-SIMD equality check (type equality)
    if (!isa<SIMDType>(op.getOperands()[0].getType()))
      return failure();
    [[fallthrough]];
  case POC::LT:
  case POC::LE: {
    auto pred = KGEN::toCmpPredicate(op.getOpcode());
    Attribute operands[] = {op.getOperands()[0], op.getOperands()[1]};
    return foldAttrWithTarget(*this, operands,
                              [&](FoldValues ops, TargetInfoAttr target) {
                                return foldSIMDCmp(pred, ops, target);
                              });
  }
  default:
    return failure();
  }
}

FailureOr<TypedAttr>
ParametricIREvaluator::evaluateApplyLike(ParamOperatorAttr op,
                                         bool withResultSlot) {
  auto symbol =
      extractSymbolConstantAttr(getReboundAttribute(op.getOperands().front()));
  StringAttr name = cast<FlatSymbolRefAttr>(symbol.getSymbol()).getAttr();
  auto gen = elaborator->oldSymTab.lookup<GeneratorOp>(name);

  // Attempt to lookup a cached value. This returns a thread local cached value.
  auto operandsAttr = cast<ParameterExprArrayAttr>(
      getReboundAttribute(ParameterExprArrayAttr::get(
          op.getContext(), op.getOperands().drop_front())));

  FuncOp func;
  if (!gen) {
    // if this is not a generator, get a concreteFunction
    ErrorTreeOr<FuncOp> funcOr =
        elaborator->getConcreteFunction(parent, *errorLoc, symbol);
    if (funcOr.isError()) {
      emitError(funcOr.takeError());
      return failure();
    }
    func = funcOr.takeValue();
    if (!func)
      return TypedAttr();
  }

  Operation *cacheOp = (!gen) ? func : gen;

  TypedAttr cached =
      elaborator->lookupCachedInterpretation(cacheOp, operandsAttr, symbol);
  if (cached) {
    return cached;
  }

  ParametricIREvaluator nestedEvaluator(*this);

  // Now invoke the interpreter.
  ErrorTreeOr<TypedAttr> result = [&]() -> ErrorTreeOr<TypedAttr> {
    if (!gen) {
      nestedEvaluator.pushEvalFrame(func, &func.getBodyRegion(), {}, 3);
      return withResultSlot
                 ? nestedEvaluator.evaluateFunctionWithResultSlot(func,
                                                                  operandsAttr)
                 : nestedEvaluator.evaluateFunction(func, operandsAttr);
    } else {
      nestedEvaluator.pushParamValues(symbol.getParamValues(), true);
      nestedEvaluator.pushEvalFrame(gen.getOperation(), &gen.getBodyRegion(),
                                    symbol.getParamValues(), 4);

      for (auto [decl, attr] :
           llvm::zip(gen.getInputParams(), symbol.getParamValues())) {
        nestedEvaluator.overwriteDeclBinding(decl, attr);
      }
      return withResultSlot
                 ? nestedEvaluator.evaluateGeneratorWithResultSlot(gen,
                                                                   operandsAttr)
                 : nestedEvaluator.evaluateGenerator(gen, operandsAttr);
    }
  }();

  // If we had a value, write it back.
  if (auto cached = result.tryGetValue()) {
    auto res = elaborator->writeGlobalCachedInterpretation(
        cacheOp, operandsAttr, symbol, cached);
    return res.first;
  }

  result.takeError().emit([](Location loc) { return mlir::emitError(loc); },
                          "interpreter failed.",
                          elaborator->options.elaborationErrorIncludePrelude);

  return TypedAttr();
}

FailureOr<TypedAttr>
ParametricIREvaluator::evaluateStringAddress(ParamOperatorAttr op) {
  // Ensure the string is null-terminated. This is safe because `StringAttr`
  // always stores a null terminator.
  auto value = dyn_cast<StringAttr>(op.getOperand(0));
  if (!value) {
    emitError({*errorLoc, "argument is not a concrete string"});
    return failure();
  }

  StringRef str(value.data(), value.size() + 1);
  if (value.getValue().empty())
    str = "\0";

  ParametricIREvaluator nestedEvaluator(*this);

  MemoryHandleAttr hdl = MemoryHandleAttr::get(getContext(), str);
  ErrorOr<int64_t> addr = nestedEvaluator.mapConstGlobalMemory(hdl);
  if (addr.isError()) {
    emitError({*errorLoc, addr.takeError()});
    return failure();
  }

  auto ptr = PointerAttr::get(getContext(), addr.takeValue(), op.getType());
  if (ErrorOrSuccess err = nestedEvaluator.externalizeMemory(ptr)) {
    emitError({*errorLoc, err.takeError()});
    return failure();
  }
  return {ptr};
}

//===----------------------------------------------------------------------===//
// ParameterEvaluationContext Hooks
//===----------------------------------------------------------------------===//

FailureOr<ResolvedStructHandle>
ParametricIREvaluator::resolveStructOp(TypedAttr typeValue,
                                       bool /*acceptAsync*/) {
  // ParametricIREvaluator works with already-instantiated types.
  // acceptAsync is ignored since we always look up from the generator.
  TypedAttr typeRef = getTypeRefForTypeValueIfResolved(typeValue);
  if (!typeRef)
    return failure();

  auto instanceRef = dyn_cast<TypeInstanceRefAttr>(typeRef);
  if (!instanceRef)
    return failure();
  PParamNode *genNode =
      elaborator->lookupImplNode(instanceRef.getSymbol())->parent;
  StructGeneratorOp gen = cast<StructGeneratorOp>(genNode->gen);
  return ResolvedStructHandle{gen, genNode->inputParams, genNode,
                              /*instance=*/nullptr};
}

void ParametricIREvaluator::withEvaluator(
    ArrayRef<ParamDeclAttr> paramDecls, ArrayRef<TypedAttr> paramValues,
    llvm::function_ref<void(ParameterEvaluator &)> callback) {
  ParametricIREvaluator nestedEvaluator(*elaborator, parent);
  nestedEvaluator.setErrorLoc(*errorLoc);
  nestedEvaluator.pushParamValues(paramValues, true);
  for (auto [param, value] : llvm::zip(paramDecls, paramValues))
    nestedEvaluator.overwriteDeclBinding(param, value);
  callback(nestedEvaluator.getCurrentParamEvalFrame().evaluator);
}

void ParametricIREvaluator::emitMaterializationError(const Twine &message) {
  IREvaluatorContext::emitError({*errorLoc, message.str()});
}

TargetInfoAttr ParametricIREvaluator::getTargetInfo() const {
  return elaborator->getTarget();
}

ParamNodeBase *
ParametricIREvaluator::lookupParamNodeBase(SymbolRefAttr symbol) {
  return elaborator->lookupImplNode(symbol)->parent;
}

GeneratorOp ParametricIREvaluator::getGenerator(SymbolRefAttr symbol) {
  return elaborator->oldSymTab.lookup<GeneratorOp>(
      cast<FlatSymbolRefAttr>(symbol).getAttr());
}

ErrorOr<CrossDeviceFunction>
ParametricIREvaluator::compileAsm(MLIRContext *ctx, GeneratorOp func,
                                  SymbolConstantAttr symbol, StringAttr name,
                                  TargetInfoAttr target, EmitAs emissionKind,
                                  EmissionOptions emissionOptions) {
  SymbolTable symtabCopy = elaborator->oldSymTab;
  return elaborator->compileAsmFn(
      func, symbol, name, symtabCopy, target, emissionKind, emissionOptions,
      elaborator->options, elaborator->getOptions());
}

void ParametricIREvaluator::addDeferredFunction(OwningOpRef<FuncOp> func) {
  elaborator->addDeferredFunction(std::move(func));
}

ImplNodeBase *ParametricIREvaluator::getParentNode() { return parent; }

//===----------------------------------------------------------------------===//
// ParametricIREvaluator
//===----------------------------------------------------------------------===//

ParametricIREvaluator::ParametricIREvaluator(ParametricElaborator &elaborator,
                                             PImplNode *parent)
    : IREvaluatorContext(elaborator.env, elaborator.getTarget().getContext(),
                         this),
      ParametricIRInterpreter(elaborator.config.maxDepth,
                              elaborator.getTarget()),
      elaborator(&elaborator), parent(parent) {
  paramEvalFrames.emplace_back();
  paramEvalFrames.back().evaluator.setEvaluationContext(this);
}

ParametricIREvaluator::ParametricIREvaluator(const ParametricIREvaluator &other)
    : IREvaluatorContext(other.elaborator->env, other.getTarget().getContext(),
                         this),
      ParametricIRInterpreter(other.maxDepth, other.getTarget()),
      elaborator(other.elaborator), parent(other.parent) {
  nestedStackDepth = other.nestedStackDepth + other.stack.size();
  this->errorLoc = other.errorLoc;
  this->emitError = other.emitError;
  if (!other.paramEvalFrames.empty()) {
    this->paramEvalFrames.emplace_back(other.paramEvalFrames.back().evaluator);
    this->paramEvalFrames.back().evaluator.setEvaluationContext(this);
  }
}

/// Given a generic parameter expression, simplify it by folding the
/// expression according to known parameter values.  This returns an error if
/// the expression cannot be folded for one reason or another.
ErrorTreeOr<Attribute>
ParametricIREvaluator::concretizeParameterExpr(PImplNode *parent, Location loc,
                                               Attribute expr) {
  // FIXME: Refactor ParameterEvaluator for better error propagation.
  this->parent = parent;
  errorLoc = loc;
  std::optional<ErrorTree> error;
  emitError = [&](ErrorTree err) { error = std::move(err); };

  Attribute result = getReboundAttribute(expr);
  if (error)
    return std::move(*error);

  if (!result)
    return Attribute();

  // If we can fold this to a simple constant result, do.
  if (ParameterAttr::isSimpleConstant(result))
    return result;

  // Otherwise we had an error folding the expression tree or we just have a
  // some foreign attribute that doesn't participate in the parameter system.
  // Walk the attribute tree postorder - if we see any attribute that has
  // all-simple-constant leaves, then we check to see if it is erroneous so we
  // can report the error.  We do this in postorder because you could have:
  //    add(4, div(8000000000, 4))
  // and the problem is that div isn't target invariant.  The problem isn't the
  // add outside it.
  result.walk<mlir::WalkOrder::PostOrder>(
      [&](Attribute attr) -> mlir::WalkResult {
        bool allSimple = true;
        attr.walkImmediateSubElements(
            [&](Attribute sub) {
              if (allSimple)
                allSimple = ParameterAttr::isSimpleConstant(sub);
            },
            [&](Type T) {});

        // If this is an attribute with simple operands that refused to fold,
        // see if we're able to get a custom error message from it to explain
        // what is going on.
        if (allSimple) {
          if (auto itf = ::dyn_cast<ParameterAttr>(attr)) {
            auto errorMessage = itf.validateForElaborator();
            if (errorMessage.isError()) {
              emitError(ErrorTree(loc, errorMessage.takeError()));
              return WalkResult::interrupt();
            }
          }
        }
        return WalkResult::advance();
      });

  if (error)
    return std::move(*error);

  return result;
}

ErrorTreeOr<Type>
ParametricIREvaluator::concretizeParameterExpr(PImplNode *parent, Location loc,
                                               Type expr) {
  // FIXME: Refactor ParameterEvaluator for better error propagation.
  this->parent = parent;
  errorLoc = loc;
  std::optional<ErrorTree> error;
  emitError = [&](ErrorTree err) { error = std::move(err); };

  Type result = getReboundType(expr);
  if (error)
    return std::move(*error);
  return result;
}
