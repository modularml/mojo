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

#include "IREvaluatorContext.h"
#include "ElaboratorHelper.h"
#include "Mojo/Interpreter/InterpreterState.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/POPDialect/POPAttrs.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/POPDialect/POPUtils.h"
#include "Mojo/Support/NameMangling.h"
#include "Mojo/TransformUtils/ManglingUtils.h"
#include "Support/Compiler/DiagnosticHandler.h"
#include "Support/StringExtras.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopeExit.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

SymbolConstantAttr KGEN::extractSymbolConstantAttr(TypedAttr attr) {
  if (auto literal = dyn_cast<FuncLiteralTypeGeneratorType>(attr.getType()))
    return literal.getSymbolConstantAttr();
  if (auto literal = dyn_cast<FuncLiteralType>(attr.getType())) {
    auto funcSymbol = cast<FuncSymbolAttr>(literal.getFuncLiteral());
    return SymbolConstantAttr::get(
        funcSymbol.getSymbol(),
        FuncTypeGeneratorType::get({}, funcSymbol.getType(), nullptr),
        funcSymbol.getParamValues());
  }
  if (auto genSymbol = dyn_cast<GeneratorAttr>(attr)) {
    assert(genSymbol.getInputParamTypes().empty());
    auto funcSymbol = cast<FuncSymbolAttr>(genSymbol.getBody());
    return SymbolConstantAttr::get(
        funcSymbol.getSymbol(),
        FuncTypeGeneratorType::get({}, funcSymbol.getType(), nullptr),
        funcSymbol.getParamValues());
  }

  return dyn_cast<SymbolConstantAttr>(attr);
}

ErrorTreeOr<OffloadFunc> KGEN::extractOffloadFunc(CompileOffloadOp op,
                                                  mlir::SymbolTable &symtab) {
  Location loc = op.getLoc();
  SymbolConstantAttr symbol = extractSymbolConstantAttr(op.getFuncAttr());
  if (!symbol) {
    return ErrorTree(loc, "'compile_offload' func argument did not resolve to "
                          "a concrete function");
  }

  if (!symbol.getType().isFullyBound()) {
    std::string errMsg;
    llvm::raw_string_ostream os(errMsg);
    os << "'compile_offload' func is not fully bound: "
       << symbol.getSymbol().getLeafReference().getValue() << " missing "
       << symbol.getType().getInputParamTypes().size()
       << " parameter binding(s)";
    return ErrorTree(loc, errMsg);
  }

  auto generator = symtab.lookup<GeneratorOp>(
      cast<FlatSymbolRefAttr>(symbol.getSymbol()).getAttr());
  if (!generator) {
    std::string errMsg;
    llvm::raw_string_ostream os(errMsg);
    os << "compile_offload must reference a valid GeneratorOp, but got "
       << symbol.getSymbol().getLeafReference().getValue();
    return ErrorTree(loc, errMsg);
  }

  return OffloadFunc{symbol, generator};
}

//===----------------------------------------------------------------------===//
// ImplNodeBase
//===----------------------------------------------------------------------===//

void ImplNodeBase::initialize(InstantiatedOpInterface inst,
                              ParameterUseDefGraph &&graph) {
  this->inst = inst;
  this->paramGraph = std::move(graph);
}

//===----------------------------------------------------------------------===//
// ParamNodeBase
//===----------------------------------------------------------------------===//

AsyncValueRef<Chain> ParamNodeBase::copy() const { return paramCh.copy(); }

StringAttr ParamNodeBase::getMangledName() {
  // Check cached result.
  if (const void *namePtr = mangledName.load())
    return StringAttr::getFromOpaquePointer(namePtr);

  // Bind all parameter values in this scope.
  ArrayRef<TypedAttr> inputParamValues = inputParams.getValue();
  [[maybe_unused]] ArrayRef<ParamDeclAttr> inputParamDecls =
      gen.getInputParams();
  assert(inputParamValues.size() == inputParamDecls.size() &&
         "incorrect # input parameter values");
  std::string baseName = mangleParameterValues(gen, inputParamValues);
  StringAttr name = StringAttr::get(gen->getContext(), baseName);

  const void *existing = nullptr;
  if (mangledName.compare_exchange_strong(existing, name.getAsOpaquePointer()))
    return name;
  return StringAttr::getFromOpaquePointer(existing);
}

FailureOr<TypedAttr> IREvaluatorContext::evaluateGetLinkageNameAttr(
    GetLinkageNameAttr getLinkageNameAttr) {
  TargetInfoAttr target =
      cast<TargetParamAttr>(getLinkageNameAttr.getTarget()).getTarget();

  ErrorTreeOr<StringAttr> nameOrError = evaluateMangledName(
      getLinkageNameAttr.getFunc(),
      /*sanitize=*/target.isGPU(), *errorLoc, "get_linkage_name");

  if (nameOrError.isError()) {
    emitError(nameOrError.takeError());
    return failure();
  }

  StringAttr mangledName = nameOrError.takeValue();

  if (!mangledName)
    return TypedAttr(); // Not ready yet — signal retry.

  return {
      StringAttr::get(mangledName.getValue(), getLinkageNameAttr.getType())};
}

FailureOr<TypedAttr> IREvaluatorContext::evaluateGetSourceNameAttr(
    GetSourceNameAttr getSourceNameAttr) {
  auto symbol = extractSymbolConstantAttr(getSourceNameAttr.getFunc());
  if (!symbol) {
    emitError({*errorLoc, "'get_source_name' function argument did not resolve "
                          "to a concrete function"});
    return failure();
  }

  auto func = getGenerator(symbol.getSymbol());

  std::optional<StringRef> sourceName = func.getSourceName();
  if (!sourceName) {
    emitError({*errorLoc, "function '" +
                              symbol.getSymbol().getLeafReference().getValue() +
                              "' has no source name"});
    return failure();
  }
  return {StringAttr::get(*sourceName, getSourceNameAttr.getType())};
}

static TypedAttr getElementTypeFromPointer(TypedAttr typeValue) {
  auto typeParam = dyn_cast<TypeParamAttr>(typeValue);
  if (!typeParam)
    return {};
  auto pointerType = dyn_cast<PointerType>(typeParam.getTypeValue());
  if (!pointerType)
    return {};
  auto typeValueType = dyn_cast<TypeValueType>(pointerType.getElementType());
  if (!typeValueType)
    return {};
  return typeValueType.getTypeValue();
}

FailureOr<TypedAttr>
IREvaluatorContext::evaluateGetTypeNameAttr(GetTypeNameAttr getTypeNameAttr) {
  auto qualifiedBuiltins =
      dyn_cast<SIMDAttr>(getTypeNameAttr.getQualifiedBuiltins());
  if (!qualifiedBuiltins) {
    emitError({*errorLoc, "'get_type_name' name did not narrow to a constant"});
    return failure();
  }

  bool qualified = qualifiedBuiltins.getAsBool();
  TypedAttr typeValue = getTypeNameAttr.getTypeValue();
  bool isRef = false;
  if (TypedAttr inner = getElementTypeFromPointer(typeValue)) {
    typeValue = inner;
    isRef = true;
  }

  TypedAttr typeRef = getTypeRefForTypeValueIfResolved(typeValue);
  if (auto instanceRef = dyn_cast_if_present<TypeInstanceRefAttr>(typeRef)) {
    std::string name;
    llvm::raw_string_ostream os(name);
    if (isRef)
      os << "ref[";
    os << stringifyTypeInstanceRef(instanceRef, qualified);
    if (isRef)
      os << "]";
    return {StringAttr::get(name, getTypeNameAttr.getType())};
  }

  emitError({*errorLoc, "'get_type_name' requires a concrete type, got " +
                            mlir::debugString(getTypeNameAttr)});
  return failure();
}

FailureOr<TypedAttr>
IREvaluatorContext::evaluateIsStructTypeAttr(IsStructTypeAttr attr) {
  MLIRContext *ctx = attr.getContext();

  if (auto typeParam = sugarDynCast<TypeParamAttr>(attr.getTypeValue())) {
    if (sugarDynCast<KGEN::StructType>(typeParam.getTypeValue()))
      return {BoolAttr::get(ctx, true)};
  }

  // Unwrap the type value to get the underlying reference.
  TypedAttr typeRef = getTypeRefForTypeValueIfResolved(attr.getTypeValue());

  // Check if it's a TypeInstanceRefAttr (concrete type instance).
  // Returns false for:
  // - MLIR primitive types (index, i64, etc.) - these are TypeAttr, not refs
  // - TypeGeneratorRefAttr (unbound generics) - can't reflect on unbound types
  // - nullptr/unresolved types
  auto instanceRef = dyn_cast_if_present<TypeInstanceRefAttr>(typeRef);
  if (!instanceRef)
    return {BoolAttr::get(ctx, false)};

  // Look up the symbol to verify it's a StructGeneratorOp.
  // Returns false for non-struct type generators (e.g., trait types).
  ParamNodeBase *genNode = lookupParamNodeBase(instanceRef.getSymbol());
  if (!isa<StructGeneratorOp>(genNode->gen))
    return {BoolAttr::get(ctx, false)};

  return {BoolAttr::get(ctx, true)};
}

FailureOr<TypedAttr>
IREvaluatorContext::evaluateFnTypeIsCABIAttr(FnTypeIsCABIAttr attr) {
  MLIRContext *ctx = attr.getContext();

  // After generic parameter substitution, typeValue is a TypeParamAttr.
  // TypeParamAttr.getMlirType() holds the concrete FuncTypeGeneratorType
  // (including its FnEffects bits such as CABI). For non-function types,
  // return false.
  TypedAttr typeValue = SugarAttr::strip(attr.getTypeValue());
  auto typeParam = dyn_cast<TypeParamAttr>(typeValue);
  if (!typeParam)
    return {BoolAttr::get(ctx, false)};

  auto funcType = dyn_cast<FuncTypeGeneratorType>(typeParam.getMlirType());
  if (!funcType)
    return {BoolAttr::get(ctx, false)};

  return {BoolAttr::get(ctx, funcType.getBody().isCABI())};
}

//===----------------------------------------------------------------------===//
// Base Type Reflection Evaluators
//===----------------------------------------------------------------------===//

/// Helper to get the base type name from a type reference.
/// Returns the unqualified name from the generator's valueDomainType.
/// For non-struct types, returns null.
StringAttr IREvaluatorContext::getBaseTypeName(TypedAttr typeRef) {
  typeRef = SugarAttr::strip(typeRef);

  // Helper to extract unqualified name from a StructGeneratorOp.
  auto getNameFromStructGen = [](StructGeneratorOp structGen) -> StringAttr {
    Type valueDomainType = structGen.getValueDomainType();
    if (auto structInstType = dyn_cast<StructInstanceType>(valueDomainType)) {
      StringRef fullName = structInstType.getName().getValue();
      // Strip module qualification (e.g., "std::builtin::int::Int" -> "Int")
      size_t lastSep = fullName.rfind("::");
      StringRef baseName = (lastSep != StringRef::npos)
                               ? fullName.substr(lastSep + 2)
                               : fullName;
      return StringAttr::get(structGen.getContext(), baseName);
    }
    return nullptr;
  };

  // For TypeInstanceRefAttr, look up the generator through the evaluator.
  if (auto instanceRef = dyn_cast<TypeInstanceRefAttr>(typeRef)) {
    ParamNodeBase *genNode = lookupParamNodeBase(instanceRef.getSymbol());
    if (genNode && genNode->gen) {
      if (auto structGen =
              dyn_cast<StructGeneratorOp>(genNode->gen.getOperation()))
        return getNameFromStructGen(structGen);
    }
    return nullptr;
  }

  // For TypeGeneratorRefAttr, look up the generator.
  if (auto genRef = dyn_cast<TypeGeneratorRefAttr>(typeRef)) {
    GeneratorOp gen = getGenerator(genRef.getSymbol());
    if (auto structGen = dyn_cast<StructGeneratorOp>(*gen))
      return getNameFromStructGen(structGen);
    return nullptr;
  }

  return nullptr;
}

FailureOr<TypedAttr>
IREvaluatorContext::evaluateGetBaseTypeNameAttr(GetBaseTypeNameAttr attr) {
  // Get the underlying type reference.
  // Returns null for primitives (index, i64, etc.) since they're not type refs.
  TypedAttr typeRef = getTypeRefForTypeValueIfResolved(attr.getTypeValue());

  // For primitives or non-struct types, return "<unknown>".
  if (!typeRef)
    return {StringAttr::get("<unknown>", attr.getType())};

  // Get the base type name from the generator's valueDomainType.
  StringAttr name = getBaseTypeName(typeRef);
  if (!name)
    return {StringAttr::get("<unknown>", attr.getType())};

  return {StringAttr::get(name.getValue(), attr.getType())};
}

static void emitDiagnosticToStream(raw_ostream &os, Diagnostic &diag) {
  os << "\n" << diag.getLocation() << ": " << diag;
  for (Diagnostic &note : diag.getNotes())
    emitDiagnosticToStream(os, note);
}

FailureOr<TypedAttr>
IREvaluatorContext::evaluateCompileAssemblyAttr(CompileAssemblyAttr attr) {
  // Cheeky copy. The state of the symbol table right at this moment is
  // sufficient to produce a standalone object for the generator being JIT'd.
  // Slice out a standalone module to re-elaborate with the new target.

  TargetInfoAttr target = cast<TargetParamAttr>(attr.getTarget()).getTarget();
  EmitAs emissionKind = cast<EmitAsAttr>(attr.getEmissionKind()).getValue();
  StringRef emissionOptionsStr =
      cast<StringAttr>(attr.getEmissionOptions()).getValue();
  bool propagateError = cast<BoolAttr>(attr.getPropagateError()).getValue();
  SymbolConstantAttr symbol = extractSymbolConstantAttr(attr.getFunc());
  ErrorTreeOr<StringAttr> nameOrError = evaluateMangledName(
      symbol, /*sanitize=*/false, *errorLoc, "compile_assembly");
  if (nameOrError.isError()) {
    getParentNode()->setToError(nameOrError.takeError());
    return failure();
  }

  StringAttr name = nameOrError.takeValue();

  if (!name)
    return TypedAttr(); // Not ready yet — signal retry.

  GeneratorOp func = getGenerator(symbol.getSymbol());

  // Construct the expected result type.
  MLIRContext *ctx = attr.getContext();
  Builder b(ctx);
  auto noneType = KGEN::NoneType::get(ctx);
  auto populateFnType = FuncTypeGeneratorType::get(
      /*inputParamTypes=*/{},
      b.getFunctionType(PointerType::get(noneType), noneType),
      {ArgConvention::ImmReg}, FnEffects().setCapturing());

  // Specialize the generator with another target by slicing it and its
  // transitive dependencies out of the IR and re-invoking the elaborator. If it
  // turns out that the specialization has more than one implementation, then
  // the elaborator invocation will fail due to multiple implementations of a
  // primary generator, and the functor will return an error.

  // Parse the emission options from a comma separated list of values.
  SmallVector<StringRef> emissionOptions;
  emissionOptionsStr.split(emissionOptions, /*Separator=*/",",
                           /*MaxSplit=*/-1, /*KeepEmpty=*/false);

  // Capture the diagnostics that may be emitted.
  DiagnosticHandler handler(ctx);
  ErrorOr<CrossDeviceFunction> closure = compileAsm(
      ctx, func, symbol, name, target, emissionKind, emissionOptions);
  handler.release();

  if (closure.isError()) {
    // Emit all the errors now.
    if (!propagateError) {
      handler.emitDiagnostics([&](Diagnostic &diag) {
        ctx->getDiagEngine().emit(std::move(diag));
      });
      emitError({*errorLoc, closure.takeError()});
      return failure();
    }
    // Concat all the errors together and return it as a variant.
    std::string error;
    llvm::raw_string_ostream os(error);
    os << closure.getError();
    handler.emitDiagnostics(
        [&](Diagnostic &diag) { emitDiagnosticToStream(os, diag); });
    // Note: return -1 to indicate an error state.
    return {StructAttr::get({StringAttr::get(os.str(), StringType::get(ctx)),
                             b.getIndexAttr(-1),
                             UninitMemAttr::get(populateFnType)})};
  }

  auto populate = cast<FuncOp>(closure->populateCapturesFn.release());
  auto populateFnRef = SymbolConstantAttr::get(populate);
  addDeferredFunction(populate);
  return {
      StructAttr::get({closure->contents, b.getIndexAttr(closure->numCaptures),
                       populateFnRef})};
}

/// Resolve the linkageName of @p gen to a concrete string for the given
/// symbol instantiation.
///
/// Precondition: @p gen must have a linkageName attribute. Call this only
/// when gen.getLinkageNameAttr() is non-null.
///
/// Returns:
///   - failure()          — linkage name could not be resolved
///   - success(null)      — not ready yet; a blocker has been registered
///                          and the caller should retry
///   - success(TypedAttr) — the resolved linkage name expression
static FailureOr<TypedAttr>
evaluateLinkageName(GeneratorOp gen, SymbolConstantAttr symbol,
                    ParameterEvaluationContext *evalCtx) {
  assert(gen.getLinkageNameAttr() &&
         "evaluateLinkageName requires gen to have a linkageName attribute");
  // Fall back to evaluating the generator's linkageName expression directly.
  LinkageNameAttr genLinkageName =
      dyn_cast_if_present<LinkageNameAttr>(gen.getLinkageNameAttr());
  if (!genLinkageName)
    return failure();

  // Substitute the generator's parameters with the symbol's concrete values.
  // getReboundAttribute evaluates any residual parametric expressions —
  // including DataToStr — via the evaluateContextSpecific hook in the
  // evaluation context, which is always the concrete IREvaluator subclass.
  ParameterEvaluator tempEval(gen.getInputParams(), symbol.getParamValues());
  tempEval.setEvaluationContext(evalCtx);
  Attribute rebound = tempEval.getReboundAttribute(genLinkageName);
  if (!rebound)
    return TypedAttr(); // not-ready: a sub-dependency is still being elaborated
  if (auto lna = dyn_cast_if_present<LinkageNameAttr>(rebound))
    if (auto prefix = dyn_cast<StringAttr>(lna.getName()))
      return TypedAttr(StringAttr::get(prefix.getValue(),
                                       StringType::get(gen.getContext())));
  return failure();
}

std::optional<ErrorTreeOr<SymbolConstantAttr>>
IREvaluatorContext::resolveTransparentThunkCallee(GeneratorOp generator,
                                                  SymbolConstantAttr symbol,
                                                  Location loc) {
  auto calleeExpr =
      generator->getAttrOfType<TypedAttr>(kTransparentThunkCalleeExprAttr);
  if (!calleeExpr)
    return std::nullopt;

  // Save/restore the error-capture state. The helper may be invoked while an
  // outer evaluation (e.g. `concretizeParameterExpr`) already has its own
  // `emitError` lambda installed; clobbering it without restoring would
  // leave the outer evaluation with a dangling lambda after we return.
  std::function<void(ErrorTree)> savedEmitError = std::move(emitError);
  std::optional<Location> savedErrorLoc = errorLoc;
  auto restore = llvm::scope_exit([&] {
    emitError = std::move(savedEmitError);
    errorLoc = savedErrorLoc;
  });

  errorLoc = loc;
  std::optional<ErrorTree> error;
  emitError = [&](ErrorTree err) { error = std::move(err); };

  // Plug `this` in as the evaluation context so the rebind dispatches
  // parameter operators through the subclass's `evaluateContextSpecific`
  // hook (and routes any materialization errors back through `emitError`).
  ParameterEvaluator evaluator(generator.getInputParams(),
                               symbol.getParamValues());
  evaluator.setEvaluationContext(this);
  Attribute rebound =
      extractSymbolConstantAttr(evaluator.getReboundAttribute(calleeExpr));

  if (error)
    return ErrorTreeOr<SymbolConstantAttr>(std::move(*error));
  if (!rebound)
    return ErrorTreeOr<SymbolConstantAttr>(SymbolConstantAttr());
  if (auto resolved = dyn_cast<SymbolConstantAttr>(rebound))
    return ErrorTreeOr<SymbolConstantAttr>(resolved);

  // Defensive: the callee expression rebound to a non-symbol attr. A
  // well-formed `kgen.transparent_thunk_callee_expr` should always resolve to
  // a `SymbolConstantAttr`; reaching this branch means the attribute is
  // malformed (likely a compiler bug at the producer site). Surface as an
  // internal error rather than crashing in `cast`.
  return ErrorTreeOr<SymbolConstantAttr>(ErrorTree(
      loc, "internal error: transparent thunk callee expression resolved to "
           "non-symbol attr: " +
               mlir::debugString(rebound)));
}

/// Evaluate the mangled name of a function. Returns an empty StringAttr to
/// signal "not ready yet" (caller should retry).
ErrorTreeOr<StringAttr>
IREvaluatorContext::evaluateMangledName(TypedAttr symCst, bool sanitize,
                                        Location errorLoc,
                                        StringRef errorContext) {
  auto symbol = extractSymbolConstantAttr(symCst);
  if (!symbol) {
    return ErrorTree(errorLoc,
                     "'" + errorContext +
                         "' function argument did not resolve to a concrete "
                         "function");
  }

  if (!symbol.getType().isFullyBound()) {
    std::string errMsg;
    llvm::raw_string_ostream os(errMsg);
    os << "'" << errorContext << "' function is not fully bound: "
       << symbol.getSymbol().getLeafReference().getValue() << " missing "
       << symbol.getType().getInputParamTypes().size()
       << " parameter binding(s)";
    return ErrorTree(errorLoc, errMsg);
  }

  GeneratorOp generator = getGenerator(symbol.getSymbol());
  if (!generator) {
    std::string errMsg;
    llvm::raw_string_ostream os(errMsg);
    os << "'" << errorContext
       << "' expected a valid generator reference, but got "
       << symbol.getSymbol().getLeafReference().getValue() << "\n";
    return ErrorTree(errorLoc, errMsg);
  }

  // For a transparent thunk, look through to the wrapped function.
  // `processCompileOffload` does the same redirect on the offload side — the
  // wrapped function is what gets compiled, so its mangled name is what the
  // host needs to look up.
  if (auto thunkCallee =
          resolveTransparentThunkCallee(generator, symbol, errorLoc)) {
    if (thunkCallee->isError())
      return thunkCallee->takeError();
    SymbolConstantAttr calleeSym = thunkCallee->takeValue();
    if (!calleeSym)
      return StringAttr(); // not ready: retry
    return evaluateMangledName(calleeSym, sanitize, errorLoc, errorContext);
  }

  if (generator.getLinkageNameAttr()) {
    // @__name is present: evaluate the (possibly parametric) name expression.
    FailureOr<TypedAttr> result = evaluateLinkageName(generator, symbol, this);
    if (failed(result)) {
      return ErrorTree(
          {errorLoc,
           "'" + errorContext + "' failed to resolve linkage name for '" +
               symbol.getSymbol().getLeafReference().getValue() + "'"});
    }
    if (!*result)
      return StringAttr(); // retry
    auto resolved = dyn_cast<StringAttr>(*result);
    if (!resolved) {
      return ErrorTree(
          {errorLoc, "'" + errorContext +
                         "' linkage name did not resolve to a string for '" +
                         symbol.getSymbol().getLeafReference().getValue() +
                         "'"});
    }
    // symName is the auto-mangled wrapper symbol used as a hash input
    // (mangle=true). Always use mangleParameterValues so the host and offload
    // paths hash the same seed. renameFunctions calls applyLinkageName with
    // func.getSymName() - the concrete function's MLIR sym - which equals
    // mangleParameterValues for both constant and parametric linkage names.
    // Using the linkage name literal here instead would produce a different
    // hash whenever sym_name != literal, causing a runtime kernel-launch
    // failure.
    std::string symName =
        mangleParameterValues(generator, symbol.getParamValues());
    auto lna = cast<LinkageNameAttr>(generator.getLinkageNameAttr());
    return applyLinkageName(resolved, lna, sanitize, symName,
                            symbol.getType().getBody().getValues());
  }

  // No @__name: use the auto-mangled sym_name, optionally sanitized.
  StringAttr name = StringAttr::get(
      generator.getContext(),
      mangleParameterValues(generator, symbol.getParamValues()));
  if (sanitize)
    name = sanitizeSymbolToAlnum(name);
  return name;
}

FailureOr<TypedAttr> IREvaluatorContext::evaluateCompileOffloadClosureAttr(
    CompileOffloadClosureAttr compileOffloadClosureAttr) {
  // Create the signature and an empty body of the populate capture for offload
  // closures as part of elaboration step.
  // We currently only support capturing closure as a parameter. So this closure
  // has to be created during elaboration time as a compile time constant.
  // However, bundling offload compilation means
  // that the actual compilation of the offload functions will happen later
  // once all of them are seen and collected, and the actual body of this
  // closure will not be known until the offload function is compiled
  // (so that we know what needs to be captured).
  // We will generated the actual body of this closure later.

  // Slice out a standalone module to re-elaborate with the new target later.

  // compile_offload_closure attrs are compiler-generated; these invariants
  // must hold.
  auto closureSymbolForLookup =
      extractSymbolConstantAttr(compileOffloadClosureAttr.getFunc());
  assert(closureSymbolForLookup &&
         "compile_offload_closure func must resolve to a SymbolConstantAttr");
  assert(closureSymbolForLookup.getType().isFullyBound() &&
         "compile_offload_closure func must be fully bound");
  GeneratorOp generator = getGenerator(closureSymbolForLookup.getSymbol());
  assert(generator &&
         "compile_offload_closure must reference a valid GeneratorOp");

  // If `generator` is a transparent thunk, mirror what `processCompileOffload`
  // does and look through it to the wrapped function. The offload-side
  // `writeCaptureArgs` names the populate function body after the *redirected*
  // (user-kernel) generator, so this stub must be named the same way for the
  // fill step to match them up.
  if (auto thunkCallee = resolveTransparentThunkCallee(
          generator, closureSymbolForLookup, *errorLoc)) {
    if (thunkCallee->isError()) {
      emitError(thunkCallee->takeError());
      return failure();
    }
    SymbolConstantAttr calleeSym = thunkCallee->takeValue();
    if (!calleeSym)
      return TypedAttr(); // Not ready yet — signal retry.
    if (GeneratorOp calleeGen = getGenerator(calleeSym.getSymbol())) {
      generator = calleeGen;
      closureSymbolForLookup = calleeSym;
    }
  }

  // Construct the expected result type.
  MLIRContext *ctx = compileOffloadClosureAttr.getContext();
  auto noneType = KGEN::NoneType::get(ctx);

  // The location to use for generated code. Remove all debuginfo from it.
  Location loc = DebugInfo::stripDebugScopesRecursively(*errorLoc);

  // The expected signature is `fn(Pointer[None]) capturing -> None`.
  ImplicitLocOpBuilder bb(loc, ctx);
  auto nonePtr = PointerType::get(noneType);
  auto sig = FuncType::get(bb.getFunctionType(nonePtr, noneType),
                           ArgConvention::ImmReg, FnEffects().setCapturing());

  // Use the auto-mangled sym_name (NOT @__name) for the stub name. Each
  // distinct closure instantiation gets its own stub keyed by the pre-rename
  // sym, even when multiple instantiations share the same @__name value
  // (e.g. closures capturing uint32 vs uint64). writeCaptureArgs in the
  // offload compilation path names the populate function body using the same
  // pre-rename sym, so the fill step can match stub to body by name.
  std::string stubBaseName =
      mangleParameterValues(generator, closureSymbolForLookup.getParamValues());

  OwningOpRef<FuncOp> populateFunc =
      FuncOp::create(bb, bb.getStringAttr(stubBaseName + "_populate_captures"),
                     sig, InlineLevel::Always);

  auto populate = cast<FuncOp>(populateFunc.get());
  auto populateFnRef = SymbolConstantAttr::get(populate);
  addDeferredFunction(std::move(populateFunc));
  return {populateFnRef};
}

/// Print a KGENDType, following the naming scheme in the Mojo DType struct.
/// NOTE: It would be better to have custom type name printing that can be
/// implemented on the struct directly.
static void printDType(raw_ostream &os, KGENDType dtype, bool qualified) {
  if (qualified)
    os << "std.builtin.dtype.";
  os << "DType." << dtype.getAsString(/*libForm=*/true);
}

void IREvaluatorContext::printParamValue(raw_ostream &os, ParamDeclAttr decl,
                                         TypedAttr value,
                                         bool qualifiedBuiltins) {
  TypeSwitch<TypedAttr>(value)
      .Case<DTypeConstantAttr>([&](auto dtypeConstant) {
        printDType(os, dtypeConstant.getDType(), qualifiedBuiltins);
      })
      .Case<IntegerAttr>([&](auto intAttr) {
        // Print booleans nicely.
        if (intAttr.getType().isSignlessInteger(1))
          os << (intAttr.getValue().isZero() ? "False" : "True");
        else
          intAttr.print(os, /*elideType=*/true);
      })
      .Case<NoneAttr>([&](auto noneAttr) { os << "None"; })
      .Case<UnboundAttr>([&](auto unboundAttr) { os << "?"; })
      .Case<TypeParamAttr>([&](auto typeAttr) {
        if (auto typeValue = dyn_cast<TypeValueType>(typeAttr.getTypeValue())) {
          if (auto instanceRef =
                  dyn_cast<TypeInstanceRefAttr>(typeValue.getTypeValue())) {
            os << stringifyTypeInstanceRef(instanceRef, qualifiedBuiltins);
            return;
          }
        }
        if (TypedAttr inner = getElementTypeFromPointer(typeAttr)) {
          if (auto instanceRef = dyn_cast<TypeInstanceRefAttr>(inner)) {
            os << "ref["
               << stringifyTypeInstanceRef(instanceRef, qualifiedBuiltins)
               << "]";
            return;
          }
        }

        // We print a placeholder for anything we don't know how to print.
        // NOTE: We could consider just printing the mlir for anything else. For
        // now, this is a more conservative approach, since it prevents leaking
        // IR details.
        os << "<unprintable>";
      })
      .Case<StructAttr>([&](auto structAttr) {
        os << "{";
        llvm::interleaveComma(structAttr.getValues(), os, [&](TypedAttr value) {
          printParamValue(os, decl, value, qualifiedBuiltins);
        });
        os << "}";
      })
      .Case<MemRefAttr>([&](auto memRefAttr) {
        MemoryBlobAttr memory =
            memRefAttr.getModel().getMemory()[memRefAttr.getIndex()];
        if (MemoryHandleAttr handle = memory.getHandle(); handle.isString()) {
          // NOTE: these strings should be null terminated.
          os << '"' << StringRef(handle.getData(), handle.getSize()) << '"';
          return;
        }

        os << "<unprintable>";
      })
      .Case<KGEN::SIMDAttr>([&](auto simdAttr) {
        ArrayRef<KGEN::DTypeValue> values = simdAttr.getValues();
        KGENDType dType = *simdAttr.getType().getResolvedDType();
        KGEN::printDTypeValues(os, values, dType);
        // Don't print Bool as a SIMD type.
        if (!isScalarOf<KGENDType::kBool>(simdAttr.getType())) {
          os << " : ";
          if (qualifiedBuiltins)
            os << "std.builtin.simd.";
          os << "SIMD[";
          printDType(os, dType, qualifiedBuiltins);
          os << ", " << values.size() << "]";
        }
      })
      .Default([&](auto value) { os << "<unprintable>"; });
}

std::string
IREvaluatorContext::stringifyTypeInstanceRef(TypeInstanceRefAttr instanceRef,
                                             bool qualifiedBuiltins) {
  ParamNodeBase *genNode = lookupParamNodeBase(instanceRef.getSymbol());
  StructGeneratorOp genOp = cast<StructGeneratorOp>(genNode->gen);

  llvm::SmallDenseMap<llvm::StringRef, llvm::StringRef> eligibleBuiltins = {
      {"std::builtin::bool::Bool", "Bool"},
      {"std::builtin::int::Int", "Int"},
      {"std::collections::list::List", "List"},
      {"std::builtin::simd::SIMD", "SIMD"},
      {"std::collections::string::string::String", "String"},
      {"std::builtin::uint::UInt", "UInt"},
  };

  // Print the type name first. A few common types can be printed more tersely.
  /// NOTE: It would be better to have custom type name printing that can be
  /// implemented on the struct directly.
  std::string name = genOp.getSymName().str();
  if (!qualifiedBuiltins && name.starts_with("std::")) {
    if (auto it = eligibleBuiltins.find(name); it != eligibleBuiltins.end())
      name = it->second;
  }
  replaceAll(name, "::", ".");

  ArrayRef<TypedAttr> paramValues = genNode->inputParams.getValue();
  if (!paramValues.empty()) {
    std::string paramValuesStr;
    llvm::raw_string_ostream os(paramValuesStr);
    auto paramDecls = genOp.getInputParams();

    // If the type is parameterized, print the parameter values.
    llvm::interleaveComma(llvm::zip(paramDecls, paramValues), os,
                          [&](auto pair) {
                            auto [decl, value] = pair;
                            printParamValue(os, decl, value, qualifiedBuiltins);
                          });

    name += "[" + paramValuesStr + "]";
  }
  return name;
}

//===----------------------------------------------------------------------===//
// IREvaluatorContext
//===----------------------------------------------------------------------===//

IREvaluatorContext::IREvaluatorContext(EnvAttr env, MLIRContext *mlirCtx,
                                       InterpreterState *state)
    : env(env), state(state), mlirCtx(mlirCtx) {}

FailureOr<TypedAttr> IREvaluatorContext::evaluateGetEnv(ParamOperatorAttr op) {
  // Grab the module from the elaborator. This is a read operation of memory
  // that is not modified during elaboration, so no synchronization is needed.
  auto name = dyn_cast<StringAttr>(op.getOperands().front());
  if (!name) {
    emitError({*errorLoc, "'get_env' name did not narrow to a constant"});
    return failure();
  }

  // Get the `StringRef` out of the `StringAttr` because the latter comes with
  // a `StringType` type that makes pointer comparisons fails.
  ErrorOr<TypedAttr> result = env.queryValue(name.getValue(), op.getType());

  if (result.isError()) {
    emitError({*errorLoc, result.getError()});
    return failure();
  }

  return result.get();
}

// See if we can decode the first 'numBytes' of the memory blob into a
// StringAttr.
static StringAttr getBytesOf(MemoryBlobAttr value, size_t numBytes) {
  // We don't bother handling these.
  if (!value.getPointerRegions().empty() || !value.getSymbolRegions().empty())
    return {};

  if (numBytes <= value.getHandle().getSize()) {
    return StringAttr::get(StringRef(value.getHandle().getData(), numBytes),
                           StringType::get(value.getContext()));
  }
  return {};
}

/// Extract a value of type `struct<(pointer<none>, index)>` into a StringAttr.
FailureOr<StringAttr> IREvaluatorContext::evaluateStringPart(TypedAttr part,
                                                             bool reset) {
  // Get the two parts of the struct, StructExtract will fold.
  TypedAttr lengthAttr = StructExtractAttr::get(part, 1);
  ErrorOr<int64_t> lengthOr = POP::getScalarIndexValue(lengthAttr);
  if (lengthOr.isError()) {
    emitError(
        {*errorLoc,
         Error(Twine("'data_to_str' length didn't resolve to a constant: ") +
               lengthOr.getError())});
    return failure();
  }

  size_t numBytes = *lengthOr;
  if (!numBytes)
    return {StringAttr::get("", StringType::get(mlirCtx))};

  MemRefAttr pointerAttr =
      dyn_cast<MemRefAttr>(StructExtractAttr::get(part, 0));
  if (!pointerAttr) {
    emitError({*errorLoc, "'data_to_str' did not narrow to a constant"});
    return failure();
  }

  // Check to see if we have a memref(interp.memory_handle(...)) because
  // we can just immediately fold it in common cases without materializing the
  // memory.
  // We don't handle index/offset yet.
  if (auto result =
          getBytesOf(pointerAttr.getModel().getMemory()[pointerAttr.getIndex()],
                     numBytes)) {
    if (pointerAttr.getOffset() == 0)
      return result;
  }

  // Reset memory upon exit.
  auto resetState = llvm::scope_exit([&] {
    if (reset)
      state->reset();
  });

  if (ErrorOrSuccess err = state->internalizeMemory(pointerAttr)) {
    emitError({*errorLoc, "'data_to_str' failed to read data"});
    return failure();
  }

  size_t address = cast<PointerAttr>(pointerAttr).getAddr();
  Type byteType = IntegerType::get(mlirCtx, 8);

  // Read each of the bytes into 'result' one at a time.  If any fail,
  // just bail out.
  std::string result;
  while (numBytes) {
    ErrorOr<TypedAttr> attrOr =
        state->readAttributeFromMemory(address, byteType);
    if (attrOr.isError() || !isa<IntegerAttr>(attrOr.get())) {
      emitError({*errorLoc, "'data_to_str' failed to read data"});
      return failure();
    }
    result.push_back((char)cast<IntegerAttr>(attrOr.get()).getInt());
    ++address;
    --numBytes;
  }

  // Success!
  return {StringAttr::get(result, StringType::get(mlirCtx))};
}

/// Evaluate POC::DataToStr "data_to_str" operator.
FailureOr<TypedAttr> IREvaluatorContext::evaluateDataToStr(ParamOperatorAttr op,
                                                           bool reset) {
  FailureOr<StringAttr> result = evaluateStringPart(op.getOperand(0), reset);
  if (failed(result))
    return failure();

  // Extra string parts, which will be a ParamListAttr of type
  // !kgen.param_list<>
  ParamListAttr extrasAttr = dyn_cast<ParamListAttr>(op.getOperand(1));
  if (!extrasAttr) {
    emitError(
        {*errorLoc, "'data_to_str' did not narrow to a variadic constant"});
    return failure();
  }

  // If there are no extra parts then we're done.
  if (extrasAttr.getValues().empty())
    return TypedAttr(*result);

  // Otherwise, we need to evaluate the extra parts and concatenate them.
  std::string concatStr = result->str();
  for (TypedAttr extra : extrasAttr.getValues()) {
    FailureOr<StringAttr> extraResult = evaluateStringPart(extra, reset);
    if (failed(extraResult))
      return failure();
    concatStr += extraResult->str();
  }
  return TypedAttr(StringAttr::get(concatStr, StringType::get(mlirCtx)));
}
