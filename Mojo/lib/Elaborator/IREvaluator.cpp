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

#include "IREvaluator.h"
#include "Elaborator.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/POPDialect/POPAttrs.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/POPDialect/POPUtils.h"
#include "Mojo/TransformUtils/ManglingUtils.h"
#include "Mojo/lib/Elaborator/IREvaluatorContext.h"
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

ErrorOr<Region *> IREvaluator::lookupFunctionBody(SymbolRefAttr symbol) {
  InstantiatedOpInterface inst = elaborator->lookupInstantiatedOp(symbol);
  FuncOp func = cast<FuncOp>(inst);

  // Now we can return the function body.
  return &func.getBodyRegion();
}

ErrorTreeOr<TypedAttr>
IREvaluator::evaluateFunction(FuncOp func, ArrayRef<TypedAttr> inputs) {
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
IREvaluator::evaluateFunctionWithResultSlot(FuncOp func,
                                            ArrayRef<TypedAttr> inputs) {
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
      func.getBodyRegion(), arguments, ptr.getElementType());

  // Report an error if evaluation fails.
  if (result.isError()) {
    return ErrorTree(*errorLoc, "failed to compile-time evaluate function call",
                     result.takeError());
  }

  // Apply operators only return one result.
  return result.takeValue();
}

int IREvaluator::getErrorLimit() {
  return elaborator->options.elaborationErrorLimit;
}

bool IREvaluator::getElabErrorIncludePrelude() {
  return elaborator->options.elaborationErrorIncludePrelude;
}

//===----------------------------------------------------------------------===//
// Expression Evaluation
//===----------------------------------------------------------------------===//

FailureOr<TypedAttr>
IREvaluator::evaluateContextSpecific(ContextuallyEvaluatedAttrInterface attr) {

  // conforms_to can be folded without a concrete instance, aggressively fold it
  // here before checking "EscapingReferenceFinder"
  if (auto typeConformsToTraitAttr = dyn_cast<TypeConformsToTraitAttr>(attr))
    return typeConformsToTraitAttr.evaluateWithContext(*this);

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
    return evaluateDataToStr(op, /*reset=*/true);
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

FailureOr<TypedAttr> IREvaluator::evaluateApplyLike(ParamOperatorAttr op,
                                                    bool withResultSlot) {
  // Attempt to concretize the function first.
  ErrorTreeOr<FuncOp> funcOr = elaborator->getConcreteFunction(
      parent, *errorLoc, extractSymbolConstantAttr(op.getOperands().front()));
  if (funcOr.isError()) {
    emitError(funcOr.takeError());
    return failure();
  }
  FuncOp func = funcOr.takeValue();
  if (!func)
    return TypedAttr();

  // Attempt to lookup a cached value. This returns a thread local cached value.
  auto operandsAttr = ParameterExprArrayAttr::get(
      op.getContext(), op.getOperands().drop_front());
  TypedAttr &cached =
      elaborator->lookupCachedInterpretation(func, operandsAttr);
  if (cached)
    return cached;

  // Concretize symbols within the operands before invoking the function.
  ErrorTreeOr<Attribute> operandsOr =
      elaborator->concretizeSymbolsWithin(operandsAttr, parent, *errorLoc);
  if (operandsOr.isError()) {
    emitError(operandsOr.takeError());
    return failure();
  }
  operandsAttr = cast_or_null<ParameterExprArrayAttr>(operandsOr.takeValue());
  if (!operandsAttr)
    return TypedAttr();

  // Now invoke the interpreter.
  ErrorTreeOr<TypedAttr> result =
      withResultSlot ? evaluateFunctionWithResultSlot(func, operandsAttr)
                     : evaluateFunction(func, operandsAttr);

  // If we had a value, write it back.
  if ((cached = result.tryGetValue())) {
    return elaborator->writeGlobalCachedInterpretation(func, operandsAttr,
                                                       cached);
  }
  emitError(result.takeError());
  return TypedAttr();
}

FailureOr<TypedAttr> IREvaluator::evaluateStringAddress(ParamOperatorAttr op) {
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

  auto resetState = llvm::scope_exit([&] { reset(); });

  MemoryHandleAttr hdl = MemoryHandleAttr::get(getContext(), str);
  ErrorOr<int64_t> addr = mapConstGlobalMemory(hdl);
  if (addr.isError()) {
    emitError({*errorLoc, addr.takeError()});
    return failure();
  }

  auto ptr = PointerAttr::get(getContext(), addr.takeValue(), op.getType());
  if (ErrorOrSuccess err = externalizeMemory(ptr)) {
    emitError({*errorLoc, err.takeError()});
    return failure();
  }
  return {ptr};
}

//===----------------------------------------------------------------------===//
// ParameterEvaluationContext Hooks
//===----------------------------------------------------------------------===//

FailureOr<ResolvedStructHandle>
IREvaluator::resolveStructOp(TypedAttr typeValue, bool acceptAsync) {
  // IREvaluator works with already-instantiated types (TypeInstanceRefAttr).
  // Returns generator + instance if ready, generator alone if not ready and
  // acceptAsync=false, or null decl if not ready and acceptAsync=true.
  TypedAttr typeRef = getTypeRefForTypeValueIfResolved(typeValue);
  if (!typeRef)
    return failure();

  if (auto instRef = dyn_cast<TypeInstanceRefAttr>(typeRef)) {
    ParamNode *genNode =
        elaborator->lookupImplNode(instRef.getSymbol())->parent;
    StructGeneratorOp gen = cast<StructGeneratorOp>(genNode->gen);

    // Look up the impl node and try to get the StructInstanceOp.
    // Add parent node as waiter if the user accepts async elaboration.
    ErrorTreeOr<StructInstanceOp> structInstanceOr =
        elaborator->getConcreteStructTypeInstance(parent, *errorLoc, instRef,
                                                  acceptAsync);
    if (structInstanceOr.isError()) {
      emitError(structInstanceOr.takeError());
      return failure();
    }
    // If the struct instance is already done concretizing, return it.
    StructInstanceOp instance = structInstanceOr.takeValue();
    if (instance)
      return ResolvedStructHandle{gen, genNode->inputParams, genNode, instance};

    // If the struct instance is not yet ready and the user accepts async,
    // return null decl to signal that async concretization was triggered.
    if (acceptAsync)
      return ResolvedStructHandle{nullptr, {}, nullptr, nullptr};

    // Otherwise, return the generator.
    return ResolvedStructHandle{gen, genNode->inputParams, genNode, nullptr};
  }

  if (auto genRef = dyn_cast<TypeGeneratorRefAttr>(typeRef)) {
    // If the struct is not concretized yet, return the generator. things like
    // conforms_to does not requires a concretized struct to fold.
    StringAttr name = cast<FlatSymbolRefAttr>(genRef.getSymbol()).getAttr();
    auto gen = elaborator->oldSymTab.lookup<StructGeneratorOp>(name);
    if (!gen)
      return failure();
    return ResolvedStructHandle{gen, genRef.getParamValues(), nullptr, nullptr};
  }

  return failure();
}

void IREvaluator::withEvaluator(
    ArrayRef<ParamDeclAttr> paramDecls, ArrayRef<TypedAttr> paramValues,
    llvm::function_ref<void(ParameterEvaluator &)> callback) {
  ParameterEvaluator nestedEvaluator(paramDecls, paramValues);
  nestedEvaluator.setEvaluationContext(this);
  callback(nestedEvaluator);
}

void IREvaluator::emitMaterializationError(const Twine &message) {
  // Use the ErrorTree-based emitError from IREvaluatorContext.
  IREvaluatorContext::emitError({*errorLoc, message.str()});
}

TargetInfoAttr IREvaluator::getTargetInfo() const {
  return elaborator->getTarget();
}

//===----------------------------------------------------------------------===//
// Base Type Reflection Evaluators
//===----------------------------------------------------------------------===//

/// Helper to get the base generator symbol name from a type reference.
/// Returns the generator's symbol name. For types without a generator (e.g.,
/// MLIR primitives), returns a fallback based on the type's symbol or a
/// placeholder string.
StringAttr IREvaluator::getGeneratorSymbolName(TypedAttr typeRef) {
  typeRef = SugarAttr::strip(typeRef);

  // For TypeInstanceRefAttr, look up the generator through the elaborator.
  if (auto instanceRef = dyn_cast<TypeInstanceRefAttr>(typeRef)) {
    ParamNodeBase *genNode = lookupParamNodeBase(instanceRef.getSymbol());
    if (genNode && genNode->gen) {
      // StructGeneratorOp is the common case for type generators.
      if (auto structGen =
              dyn_cast<StructGeneratorOp>(genNode->gen.getOperation()))
        return structGen.getSymNameAttr();
      // Other generators may also be SymbolOpInterface.
      if (auto symOp =
              dyn_cast<mlir::SymbolOpInterface>(genNode->gen.getOperation()))
        return symOp.getNameAttr();
    }
    // Fallback: use the instance symbol directly. This handles types where
    // we can't look up the generator (e.g., non-struct parameterized types).
    return cast<FlatSymbolRefAttr>(instanceRef.getSymbol()).getAttr();
  }

  // For TypeGeneratorRefAttr, extract the symbol from the reference.
  if (auto genRef = dyn_cast<TypeGeneratorRefAttr>(typeRef))
    return cast<FlatSymbolRefAttr>(genRef.getSymbol()).getAttr();

  return nullptr;
}

FailureOr<TypedAttr>
IREvaluator::evaluateGetBaseTypeNameAttr(GetBaseTypeNameAttr attr) {
  // Get the underlying type reference.
  // Returns null for MLIR primitives (index, i64, etc.) since they're not type
  // refs, and also for types that aren't fully resolved yet.
  TypedAttr typeRef = getTypeRefForTypeValueIfResolved(attr.getTypeValue());

  // For primitives or non-struct types, return "<unknown>".
  // We don't defer here because primitives will never resolve to struct types.
  if (!typeRef)
    return {StringAttr::get("<unknown>", attr.getType())};

  // Get the generator symbol name using the common helper.
  StringAttr generatorSymbol = getGeneratorSymbolName(typeRef);

  // For types without a generator symbol, return "<unknown>".
  if (!generatorSymbol)
    return {StringAttr::get("<unknown>", attr.getType())};

  // Extract just the unqualified name (last component after "::")
  StringRef fullName = generatorSymbol.getValue();
  size_t lastSep = fullName.rfind("::");
  StringRef simpleName =
      (lastSep != StringRef::npos) ? fullName.substr(lastSep + 2) : fullName;

  return {StringAttr::get(simpleName, attr.getType())};
}

ParamNodeBase *IREvaluator::lookupParamNodeBase(SymbolRefAttr symbol) {
  return elaborator->lookupImplNode(symbol)->parent;
}

GeneratorOp IREvaluator::getGenerator(SymbolRefAttr symbol) {
  return elaborator->oldSymTab.lookup<GeneratorOp>(
      cast<FlatSymbolRefAttr>(symbol).getAttr());
}

ErrorOr<CrossDeviceFunction>
IREvaluator::compileAsm(MLIRContext *ctx, GeneratorOp func,
                        SymbolConstantAttr symbol, StringAttr name,
                        TargetInfoAttr target, EmitAs emissionKind,
                        EmissionOptions emissionOptions) {
  SymbolTable symtabCopy = elaborator->oldSymTab;
  ErrorOr<CrossDeviceFunction> closure = elaborator->compileAsmFn(
      func, symbol, name, symtabCopy, target, emissionKind, emissionOptions,
      elaborator->options, elaborator->getOptions());
  return closure;
}

void IREvaluator::addDeferredFunction(OwningOpRef<FuncOp> func) {
  elaborator->addDeferredFunction(std::move(func));
}

ImplNodeBase *IREvaluator::getParentNode() { return parent; }

//===----------------------------------------------------------------------===//
// IREvaluator
//===----------------------------------------------------------------------===//

IREvaluator::IREvaluator(Elaborator &elaborator, ImplNode *parent)
    : IREvaluatorContext(elaborator.env, elaborator.getTarget().getContext(),
                         this),
      BytecodeInterpreter(elaborator.getTarget(), &elaborator),
      elaborator(&elaborator), parent(parent) {
  evaluator.setEvaluationContext(this);
}

IREvaluator::IREvaluator(const IREvaluator &other)
    : IREvaluatorContext(other.elaborator->env, other.getTarget().getContext(),
                         this),
      BytecodeInterpreter(other.getTarget(), other.elaborator),
      evaluator(other.evaluator), elaborator(other.elaborator),
      parent(other.parent) {
  evaluator.setEvaluationContext(this);
}

/// Given a generic parameter expression, simplify it by folding the
/// expression according to known parameter values.  This returns an error if
/// the expression cannot be folded for one reason or another.
ErrorTreeOr<Attribute> IREvaluator::concretizeParameterExpr(ImplNode *parent,
                                                            Location loc,
                                                            Attribute expr) {
  // FIXME: Refactor ParameterEvaluator for better error propagation.
  this->parent = parent;
  errorLoc = loc;
  std::optional<ErrorTree> error;
  emitError = [&](ErrorTree err) { error = std::move(err); };

  Attribute result = evaluator.getReboundAttribute(expr);
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

ErrorTreeOr<Type> IREvaluator::concretizeParameterExpr(ImplNode *parent,
                                                       Location loc,
                                                       Type expr) {
  // FIXME: Refactor ParameterEvaluator for better error propagation.
  this->parent = parent;
  errorLoc = loc;
  std::optional<ErrorTree> error;
  emitError = [&](ErrorTree err) { error = std::move(err); };

  Type result = evaluator.getReboundType(expr);
  if (error)
    return std::move(*error);
  return result;
}
