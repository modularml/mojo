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
// This file implements the LIT dialect operations.
//
//===----------------------------------------------------------------------===//

#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/CODialect/COUtils.h"
#include "Mojo/Interpreter/ParametricInterpreterState.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENPogUtils.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/KGENDialect/ParameterReplacer.h"
#include "Mojo/LITDialect/LITAttrs.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/LITDialect/SpecialFunctions.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/MDialect/ParserUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/DebugStringHelper.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

/// This is used by `ReferenceElementType` constraint to match a type range
/// against a single type.
static bool typeRangeMatches(Type type, TypeRange range) {
  return llvm::all_of(range, [&](Type e) { return type == e; });
}

/// Given an insertion point in a block, scan up the parent hierarchy to see if
/// this block is nested under the TryOp region that will handle a 'raise'd
/// error, or if this is in a function that is allowed to raise.  This returns
/// the TryOp or FuncOp if found, or null if raise is not valid.
Operation *LIT::findOpProcessingRaise(Block *currentBlock) {
  Operation *parentOp;
  while (currentBlock && (parentOp = currentBlock->getParentOp())) {
    // If we find a throwing function, return it.
    if (auto funcOp = dyn_cast<FnOp>(parentOp))
      return funcOp.isThrows() ? funcOp : nullptr;

    if (auto tryOp = dyn_cast<TryOp>(parentOp)) {
      Region &tryBody = tryOp.getTryRegion();
      // Throwing from an else or finally region of a try operation shouldn't
      // stop at the try.
      if (!tryBody.empty() && &tryBody.front() == currentBlock) {
        // If the except region has an UnreachableOp in it, then this is not
        // allowed to raise.  This must be for a 'with' or something else that
        // needs a finally but isn't itself in a throwing region.
        auto &exceptRegion = tryOp.getExceptRegion();
        if (exceptRegion.empty() || exceptRegion.front().empty() ||
            !isa<UnreachableOp>(exceptRegion.front().front()))
          return tryOp;
      }
    }
    currentBlock = parentOp->getBlock();
  }
  return nullptr;
}

FnTypeGeneratorType LIT::getCalleeType(Operation *op) {
  if (auto call = dyn_cast<LIT::CallOp>(op))
    return call.getCalleeType();
  return cast<LIT::CallIndirectOp>(op).getCalleeType();
}

ValueRange LIT::getCalleeArguments(Operation *op) {
  if (auto call = dyn_cast<LIT::CallOp>(op))
    return call.getOperands();
  return cast<LIT::CallIndirectOp>(op).getArguments();
}

SymbolRefAttr LIT::getFullyResolvedSymbolRef(mlir::SymbolOpInterface op) {
  SmallVector<FlatSymbolRefAttr> symbols;
  do {
    symbols.push_back(FlatSymbolRefAttr::get(op.getNameAttr()));
  } while ((op = dyn_cast<mlir::SymbolOpInterface>(op->getParentOp())));

  // Form a reference from the symbols we collected.
  if (symbols.size() == 1)
    return symbols.front();
  std::reverse(symbols.begin(), symbols.end());
  return SymbolRefAttr::get(symbols[0].getAttr(),
                            ArrayRef(symbols).drop_front());
}

/// Collect ancestor ops whose parameters are relevant, and create a
/// concatenated list of their parameters.
static SmallVector<std::pair<Operation *, PogListAttr>>
collectAncestorPogListForFnOp(Operation *op) {
  SmallVector<std::pair<Operation *, PogListAttr>> ret;
  while (op) {
    // If we are dealing with a struct, trait, or extension, we concatenate
    // their variadic masks. For extensions, use the target struct's variadic
    // information.
    PogListAttr paramListAttr;
    if (auto structDecl = ::dyn_cast<StructDeclOp>(op))
      paramListAttr = structDecl.getSignature().getParamListAttrs();
    else if (auto traitDecl = ::dyn_cast<TraitDeclOp>(op))
      paramListAttr = traitDecl.getSignature().getParamListAttrs();
    else if (auto extensionDecl = ::dyn_cast<ExtensionDeclOp>(op)) {
      // Extensions inherit parameters from their target struct.
      // The MojoParser already mirrors these parameters during resolution.
      paramListAttr = extensionDecl.getSignature().getParamListAttrs();
    }

    // break upon detecting an irrelevant operation
    if (!paramListAttr)
      break;

    // collect the pog list and the parent operation.
    ret.emplace_back(std::make_pair(op, paramListAttr));
    op = op->getParentOp();
  }
  return ret;
}

FnTypeGeneratorType LIT::getFullSignature(Operation *container,
                                          FnTypeGeneratorType signature) {
  // Collect contextual params, if there are none, the full signature is the
  // same as the local signature.
  SmallVector<std::pair<Operation *, PogListAttr>> opAndPogLists =
      collectAncestorPogListForFnOp(container);

  if (opAndPogLists.empty())
    return signature;

  SmallVector<StringAttr> paramNames;
  SmallVector<ParamDeclAttr> paramDecls;
  SmallVector<TypedAttr> paramDefaults;
  SmallVector<ConstraintAttr> bodyConstraints;
  for (auto [op, pogList] : opAndPogLists) {
    for (PogMetadataAttr pog : pogList.getPogs()) {
      paramNames.push_back(pog.getName());
      paramDefaults.push_back(pog.getDefaultValue());
    }
    llvm::append_range(bodyConstraints, pogList.getBodyConstraints());

    for (ParamDeclAttr param : cast<DeclInterface>(op).getInputParams())
      paramDecls.push_back(param);
  }

  assert(paramNames.size() == paramDecls.size());
  assert(paramDefaults.size() == paramDecls.size());
  return FnTypeGeneratorType::prependParams(signature, paramDecls, paramNames,
                                            paramDefaults, bodyConstraints);
}

//===----------------------------------------------------------------------===//
// FileModuleOp
//===----------------------------------------------------------------------===//

void FileModuleOp::build(OpBuilder &builder, OperationState &state,
                         StringAttr name) {
  state.addAttribute(getSymNameAttrName(state.name), name);
  state.addRegion()->push_back(new Block());
}

/// Modules don't have input parameters but do define a parameter scope.
ArrayRef<ParamDeclAttr> FileModuleOp::getInputParams() { return {}; }

//===----------------------------------------------------------------------===//
// PackageOp
//===----------------------------------------------------------------------===//

void PackageOp::build(OpBuilder &builder, OperationState &state,
                      StringAttr name) {
  build(builder, state, name, /*sym_visibility=*/nullptr, /*docString=*/{},
        /*dependencies=*/{}, /*externLLVMBitcodeModules=*/{});
  assert(state.regions.size() == 1);
  state.regions.back()->push_back(new Block());
}

void ImportOp::build(OpBuilder &builder, OperationState &state,
                     StringAttr symName, ImportPathAttr modulePath) {
  state.addAttribute(getSymNameAttrName(state.name), symName);
  state.addAttribute(getModulePathAttrName(state.name), modulePath);
  state.addRegion()->push_back(new Block());
}

/// Packages don't have input parameters but do define a parameter scope.
ArrayRef<ParamDeclAttr> PackageOp::getInputParams() { return {}; }

LogicalResult PackageOp::verify() {
  for (Operation &op : *getBody()) {
    if (!isa<FileModuleOp, PackageOp, ImportOp, UnresolvedImportOp,
             UnresolvedWildcardImportOp>(op)) {
      return emitOpError("expected only `lit.file_module`, `lit.package`, "
                         "`lit.unresolved_import`, or "
                         "`lit.unresolved_wildcard_import` in its body")
          .attachNote(op.getLoc())
          .append("see operation defined here");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

static ParseResult parseOriginParams(AsmParser &p,
                                     ParameterExprArrayAttr &implicitOrigins) {
  SmallVector<TypedAttr> values;
  if (p.parseCommaSeparatedList(
          AsmParser::Delimiter::OptionalSquare, [&]() -> ParseResult {
            return parseOriginParamValue(p, values.emplace_back());
          }))
    return failure();
  implicitOrigins = ParameterExprArrayAttr::get(p.getContext(), values);
  return success();
}

static void printOriginParams(AsmPrinter &p, Operation *op,
                              ParameterExprArrayAttr implicitOrigins) {
  if (implicitOrigins.empty())
    return;
  p << '[';
  llvm::interleaveComma(implicitOrigins, p, [&](TypedAttr value) {
    printOriginParamValue(p, value);
  });
  p << ']';
}

/// Infer call operation operand and result types from the signature,
/// substituting implicit origin parameters.
template <typename CalleeT>
static ParseResult
parseCallOpTypes(AsmParser &p, SmallVectorImpl<Type> &operandTypes,
                 SmallVectorImpl<Type> &resultTypes, CalleeT callee,
                 ArrayRef<TypedAttr> implicitOrigins) {
  FuncTypeGeneratorType calleeType;
  if constexpr (std::is_same_v<Type, CalleeT>)
    calleeType = cast<FuncTypeGeneratorType>(callee);
  else
    calleeType = cast<FuncTypeGeneratorType>(callee.getType());

  FunctionType values;
  if (implicitOrigins.empty()) {
    values = calleeType.getBody().getValues();
  } else {
    auto calleeLITTypeGen = dyn_cast<FnTypeGeneratorType>(calleeType);
    if (!calleeLITTypeGen)
      return p.emitError(p.getCurrentLocation(),
                         "expected a FnTypeGeneratorType");
    FuncType calleeLITType = calleeLITTypeGen.getBody();
    if (calleeLITType.getNumImplicitOriginDecls() != implicitOrigins.size())
      return p.emitError(p.getNameLoc())
             << implicitOrigins.size()
             << " origins specified, but signature expected "
             << calleeLITType.getNumImplicitOriginDecls();

    values = calleeLITType.substituteImplicitOriginsIntoValues(
        implicitOrigins, [&] { return p.emitError(p.getNameLoc()); });
    if (!values)
      return failure();
  }

  // Async calls don't provide result slots.
  llvm::append_range(operandTypes,
                     values.getInputs().drop_back(
                         calleeType.getBody().getNumAsyncReturnSlots()));
  llvm::append_range(resultTypes, values.getResults());
  return success();
}

/// Nothing to do on print.
template <typename CalleeT>
static void printCallOpTypes(AsmPrinter &, Operation *, TypeRange, TypeRange,
                             CalleeT, ArrayRef<TypedAttr>) {}

static ParseResult
parseCallOp(OpAsmParser &p, TypedAttr &calleeAttr,
            ParameterExprArrayAttr &implicitOrigins,
            SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
            SmallVectorImpl<Type> &operandTypes,
            SmallVectorImpl<Type> &resultTypes) {
  SymbolRefAttr callee;
  // Optionally parse the direct call syntax: `lit.call @abc`.
  OptionalParseResult optResult = p.parseOptionalAttribute(callee);
  if (!optResult.has_value()) {
    // Otherwise, parse the parametric call syntax `lit.call [...: @abc]`
    if (parseParametricCallee(p, calleeAttr))
      return failure();
  } else if (failed(*optResult)) {
    return failure();
  }

  ParameterExprArrayAttr paramValues;
  if (parseOriginParams(p, implicitOrigins))
    return failure();
  if (callee && parseParameterValues(p, paramValues))
    return failure();
  if (p.parseOperandList(operands, AsmParser::Delimiter::Paren))
    return failure();

  if (callee) {
    FuncTypeGeneratorType signature;
    FunctionType functionType;
    if (p.parseColon() ||
        parseKGENFuncTypeGenerator(p, functionType, signature))
      return failure();
    calleeAttr = SymbolConstantAttr::get(callee, signature, paramValues);
  }
  if (failed(parseCallOpTypes(p, operandTypes, resultTypes, calleeAttr,
                              implicitOrigins)))
    return failure();
  return success();
}

static void printCallOp(OpAsmPrinter &p, Operation *op, TypedAttr calleeAttr,
                        ParameterExprArrayAttr implicitOrigins,
                        ValueRange operands, TypeRange operandTypes,
                        TypeRange resultTypes) {
  auto symbolCst = dyn_cast<SymbolConstantAttr>(calleeAttr);
  // Optionally print the direct call syntax. Otherwise, print the parametric
  // call syntax.
  if (symbolCst)
    p << ' ' << symbolCst.getSymbol();
  else
    printParametricCallee(p, op, calleeAttr);
  printOriginParams(p, op, implicitOrigins);
  if (symbolCst)
    printParameterValues(p, symbolCst.getParamValues());
  p << '(';
  p.printOperands(operands);
  p << ')';
  if (symbolCst) {
    p << " : ";
    printSignatureValues(
        p, FunctionType::get(op->getContext(), operandTypes, resultTypes),
        symbolCst.getType());
  }
}

template <typename OpT>
static LogicalResult verifyOriginParams(OpT op, FuncType sig) {
  size_t numImplicit = sig.getNumImplicitOriginDecls();
  size_t numParams = op.getImplicitOrigins().size();
  if (numParams == numImplicit)
    return success();
  return op->emitOpError("operation has ")
         << numParams
         << " bindings for implicit origin parameters, but callee "
            "expected "
         << numImplicit;
}

template <typename OpT>
static LogicalResult verifyCallOp(OpT op, FuncType sig, ValueRange operands,
                                  std::optional<TypeRange> results) {
  FunctionType values = sig.substituteImplicitOriginsIntoValues(
      op.getImplicitOrigins(), [&] { return op.emitOpError(); });
  if (!values)
    return failure();

  auto verifyTypes = [&](StringRef kind, TypeRange types,
                         TypeRange expected) -> LogicalResult {
    if (types.size() != expected.size()) {
      return op.emitOpError("callee expected ")
             << expected.size() << " " << kind << "s but got " << types.size();
    }
    for (auto [i, type, exp] : llvm::enumerate(types, expected)) {
      if (type == exp)
        continue;
      return op.emitOpError("callee expected call ")
             << kind << " #" << i << " to be " << exp << " but got " << type;
    }
    return success();
  };

  // Async calls don't provide result slots.
  if (failed(verifyTypes(
          "argument", operands,
          values.getInputs().drop_back(sig.getNumAsyncReturnSlots()))) ||
      (results && failed(verifyTypes("result", *results, values.getResults()))))
    return failure();
  return success();
}

LogicalResult LIT::CallOp::verify() {
  auto sig = dyn_cast<FnTypeGeneratorType>(getCallee().getType());
  if (!sig)
    return emitOpError("callee type must be a FnTypeGeneratorType");
  if (failed(verifyOriginParams(*this, sig.getBody())))
    return failure();
  return verifyCallOp(*this, sig.getBody(), getOperands(), getResultTypes());
}

SymbolRefAttr LIT::CallOp::getDirectCallee() {
  if (auto symbolCst = dyn_cast<SymbolConstantAttr>(getCallee()))
    return symbolCst.getSymbol();
  return {};
}

FailureOr<InlineResult> LIT::CallOp::prepInline(mlir::RewriterBase &b) {
  // Inlining not supported for this op
  return failure();
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

LogicalResult LIT::CallIndirectOp::verify() {
  auto sig = cast<FnTypeGeneratorType>(getCallee().getType());
  if (failed(verifyOriginParams(*this, sig.getBody())))
    return failure();
  return verifyCallOp(*this, sig.getBody(), getArguments(), getResultTypes());
}

//===----------------------------------------------------------------------===//
// BindParamsOp
//===----------------------------------------------------------------------===//

static ParseResult
parseDischargedBodyConstraints(AsmParser &p, DenseBoolArrayAttr &discharged) {
  if (failed(p.parseOptionalVerticalBar()))
    return success();

  std::string mask;
  if (p.parseString(&mask))
    return failure();

  SmallVector<bool> values;
  values.reserve(mask.size());
  for (char bit : mask) {
    if (bit != '0' && bit != '1') {
      return p.emitError(
          p.getCurrentLocation(),
          "expected discharged mask to contain only '0' and '1'");
    }
    values.push_back(bit == '1');
  }
  discharged = Builder(p.getContext()).getDenseBoolArrayAttr(values);
  return success();
}

static DenseBoolArrayAttr
toDenseBoolArrayAttr(MLIRContext *context, std::optional<ArrayRef<bool>> mask) {
  if (!mask)
    return {};
  return DenseBoolArrayAttr::get(context, *mask);
}

/// Parse parameter bindings for `lit.bind_params` using the generator operand
/// type. This mirrors `parseBindParams` but omits the generator attribute.
static ParseResult
parseBindParamsOpValues(AsmParser &p, GeneratorType genType,
                        SmallVectorImpl<TypedAttr> &paramValues,
                        DenseBoolArrayAttr &discharged) {
  ParameterEvaluator evaluator;
  evaluator.setInputDepth(1);
  IndexDepthAdjuster minusOneAdjuster(/*adjustDepth=*/-1);
  size_t numUnboundParams = 0;
  for (Type inputType : genType.getInputParamTypes()) {
    if (failed(p.parseOptionalComma()))
      break;
    Type remappedDeclType = evaluator.getReboundType(inputType);
    Type defaultValueType = minusOneAdjuster.replace(remappedDeclType);
    Type valueType;
    if (parseColonTypeOrDefault(p, valueType, defaultValueType) ||
        parseParamValue(p, paramValues.emplace_back(), valueType))
      return failure();
    TypedAttr value = paramValues.back();
    if (::isa<UnboundAttr>(value)) {
      auto residual = ParamIndexRefAttr::get(
          /*depth=*/-1, numUnboundParams++, defaultValueType);
      evaluator.appendIndexBinding(residual);
      continue;
    }
    evaluator.appendIndexBinding(value);
  }
  return parseDischargedBodyConstraints(p, discharged);
}

static void printBindParamsOpValues(AsmPrinter &p, GeneratorType genType,
                                    ArrayRef<TypedAttr> paramValues,
                                    DenseBoolArrayAttr discharged) {
  ParameterEvaluator evaluator;
  evaluator.setInputDepth(1);
  IndexDepthAdjuster minusOneAdjuster(/*adjustDepth=*/-1);
  size_t numUnboundParams = 0;
  for (auto [inputType, value] :
       llvm::zip(genType.getInputParamTypes(), paramValues)) {
    p << ", ";
    Type remappedDeclType = evaluator.getReboundType(inputType);
    Type defaultValueType = minusOneAdjuster.replace(remappedDeclType);
    if (::isa<UnboundAttr>(value)) {
      printColonTypeOrDefault(p, value.getType(), defaultValueType);
      printParamValue(p, value);
      auto residual = ParamIndexRefAttr::get(
          /*depth=*/-1, numUnboundParams++, defaultValueType);
      evaluator.appendIndexBinding(residual);
      continue;
    }
    printColonTypeParamValue(p, value);
    evaluator.appendIndexBinding(value);
  }
  if (discharged && !discharged.empty()) {
    p << " | \"";
    for (bool value : discharged.asArrayRef())
      p << (value ? '1' : '0');
    p << '"';
  }
}

Type BindParamsOp::inferResultType(FnTypeGeneratorType generator,
                                   ArrayRef<TypedAttr> paramValues,
                                   DenseBoolArrayAttr discharged) {
  return BindParamsAttr::inferResultType(UnknownAttr::get(generator),
                                         paramValues, discharged,
                                         /*evaluationContext=*/nullptr);
}

LogicalResult
BindParamsOp::inferReturnTypes(MLIRContext *context,
                               std::optional<Location> loc, Adaptor adaptor,
                               SmallVectorImpl<Type> &inferredReturnTypes) {
  auto emitError = [&](const Twine &msg) -> LogicalResult {
    return mlir::emitOptionalError(loc, msg);
  };
  if (adaptor.getOperands().size() != 1)
    return emitError("expected one generator operand");
  auto generator =
      dyn_cast<FnTypeGeneratorType>(adaptor.getGenerator().getType());
  if (!generator)
    return emitError("generator operand must have FnTypeGeneratorType");

  inferredReturnTypes.push_back(
      inferResultType(generator, adaptor.getParamValues(),
                      toDenseBoolArrayAttr(context, adaptor.getDischarged())));
  return success();
}

ParseResult BindParamsOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand generatorOperand;
  Type generatorType, resultType;
  SmallVector<TypedAttr> paramValues;
  DenseBoolArrayAttr discharged;

  if (parser.parseOperand(generatorOperand) || parser.parseColon() ||
      parseGenerator(parser, generatorType))
    return failure();

  auto genType = cast<FnTypeGeneratorType>(generatorType);
  if (parseBindParamsOpValues(parser, genType, paramValues, discharged) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseKeyword("to") || parseGenerator(parser, resultType) ||
      parser.resolveOperand(generatorOperand, generatorType, result.operands))
    return failure();

  result.addAttribute("paramValues", ParameterExprArrayAttr::get(
                                         parser.getContext(), paramValues));
  if (discharged)
    result.addAttribute("discharged", discharged);
  result.addTypes(resultType);
  return success();
}

void BindParamsOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printOperand(getGenerator());
  p << " : ";
  printGenerator(p, getGenerator().getType());
  printBindParamsOpValues(p, getGenerator().getType(), getParamValues(),
                          toDenseBoolArrayAttr(getContext(), getDischarged()));
  p.printOptionalAttrDict(
      getOperation()->getAttrs(),
      /*elidedAttrs=*/{"generator", "paramValues", "discharged"});
  p << " to ";
  printGenerator(p, getResult().getType());
}

LogicalResult BindParamsOp::verify() {
  auto generator = getGenerator().getType();
  TypedAttr generatorAttr = UnknownAttr::get(generator);
  DenseBoolArrayAttr discharged =
      toDenseBoolArrayAttr(getContext(), getDischarged());
  if (failed(BindParamsAttr::verify([&] { return emitOpError(); },
                                    generatorAttr, getParamValues(),
                                    discharged)))
    return failure();

  Type expectedResult =
      inferResultType(generator, getParamValues(), discharged);
  if (getResult().getType() != expectedResult)
    return emitOpError("result type ")
           << getResult().getType() << " does not match inferred type "
           << expectedResult;
  return success();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

/// If this is a special function like __init__ return the enum that
/// identifies it, otherwise return kNormal.
SpecialFunctionKind SpecialFunctionInfo::lookupKind(StringRef name) {
  if (name.size() < 5 || !name.starts_with("__") || !name.ends_with("__"))
    return SpecialFunctionKind::kNormal;

  // Normalize deprecated spellings (e.g. '__del__') to their canonical form
  // before dispatching below. StmtParser::parseDefFnStmt canonicalizes the
  // name eagerly for the common case, but DeclResolver::resolveSignature(FnOp,
  // Lexer&, ASTDecl&) re-lexes and re-parses a lazily-resolved function's name
  // directly from source, independently of that canonicalization - so this is
  // the single choke point that covers both paths. The deprecation warning
  // itself is emitted where a source location is available
  // (TypeCheckedFnSignature::verifyFunctionNameBinding).
  name = getCanonicalSpelling(name);

#define SF(ENUM, NAME, MINOPERANDS, MAXOPERANDS, EXPRNODE, FLAGS)              \
  if (name == (NAME))                                                          \
    return SpecialFunctionKind::ENUM;
#include "Mojo/LITDialect/SpecialFunctions.def"

  // Otherwise, this declaration isn't known.
  return SpecialFunctionKind::kNormal;
}

// TODO(MOCO-4495): Remove when `__del__` is no longer supported.
StringRef SpecialFunctionInfo::getCanonicalSpelling(StringRef name) {
  return name == "__del__" ? "__deinit__" : name;
}

/// If this is a special function like __init__ return the enum that
/// identifies it, otherwise return kNormal.
const SpecialFunctionInfo &SpecialFunctionInfo::get(SpecialFunctionKind kind) {
  static const SpecialFunctionInfo infos[] = {
      {nullptr, SpecialFunctionKind::kNormal, /*minNumArguments=*/0,
       /*maxNumArguments=*/-1, /*flags=*/0},
#define SF(ENUM, NAME, MINOPERANDS, MAXOPERANDS, EXPRNODE, FLAGS)              \
  {NAME, SpecialFunctionKind::ENUM, (MINOPERANDS), (MAXOPERANDS), (FLAGS)},
#include "Mojo/LITDialect/SpecialFunctions.def"
  };

  assert(unsigned(kind) < sizeof(infos) / sizeof(infos[0]));
  return infos[unsigned(kind)];
}

/// Return the SpecialFunctionKind ID that indicates if this is a special
/// function like __init__ or __radd__.
SpecialFunctionKind FnOp::getSpecialFunctionKind() {
  return (SpecialFunctionKind)getSpecialFnKind();
}
const SpecialFunctionInfo &FnOp::getSpecialFunctionInfo() {
  return SpecialFunctionInfo::get(getSpecialFunctionKind());
}

/// Returns the user-defined result type, looking through implicit memory
/// results and stripping off the variant from error throwing results if needed.
Type FnOp::getUserResultType() {
  return LIT::getSignatureUserResultType(
      getFuncTypeGenerator(), getArgumentTypes(), getMLIRResultType());
}

TypedAttr FnOp::getBoundReference(ParameterEvaluationContext &evalContext,
                                  ParameterExprArrayAttr bindings) {
  if (!bindings) // We allow null for convenience.
    bindings = ParameterExprArrayAttr::get(getContext(), {});

  if (ParamDeclAttr decl = getParamDeclAttr()) {
    // KGEN expects different binding types than Lit can provide.
    SmallVector<TypedAttr> reboundBindings;
    ParameterEvaluator evaluator;
    evaluator.setEvaluationContext(&evalContext);
    for (auto [binding, type] :
         llvm::zip(bindings, getFullSignature().getInputParamTypes())) {
      TypedAttr value = binding;
      Type unboundType = evaluator.getReboundType(type);
      if (unboundType != value.getType())
        value = ParamOperatorAttr::getRebind(value, unboundType);
      evaluator.appendIndexBinding(value);
      reboundBindings.push_back(value);
    }

    TypedAttr generator = ParamDeclRefAttr::get(decl);
    return BindParamsAttr::get(generator.getContext(), generator,
                               reboundBindings, &evalContext);
  }

  SymbolRefAttr symbol = getFullyResolvedSymbolRef(*this);
  auto unboundSymbol = SymbolConstantAttr::get(symbol, getFullSignature());
  auto resultType = cast<FuncTypeGeneratorType>(
      BindParamsAttr::inferResultType(unboundSymbol, bindings,
                                      /*discharged=*/{}, &evalContext));
  return SymbolConstantAttr::get(symbol, resultType, bindings);
}

SymbolConstantAttr
FnOp::getBoundSymbolRef(ParameterEvaluationContext &evalContext,
                        ParameterExprArrayAttr bindings) {
  return cast<SymbolConstantAttr>(getBoundReference(evalContext, bindings));
}

TypedAttr FnOp::getFuncLiteralGenerator(
    ParameterEvaluationContext &evalContext, ParameterExprArrayAttr bindings,
    const llvm::BitVector &dischargedBodyConstraints) {
  // legacy @__parameter closure is not a function literal.
  if (getParamDeclAttr())
    return getBoundReference(evalContext, bindings);

  SmallVector<TypedAttr> paramValues;
  FnTypeGeneratorType fullSig = getFullSignature();
  SymbolRefAttr symbol = getFullyResolvedSymbolRef(*this);

  // Moving FnType out of the FnTypeGeneratorType, attaching the symbol and form
  // a FnLiteralTypeGeneratorType.
  for (auto [idx, type] : enumerate(fullSig.getInputParamTypes()))
    paramValues.push_back(ParamIndexRefAttr::get(idx, type));
  auto fnLiteral = SingletonAttr::get(FuncLiteralType::get(
      FuncSymbolAttr::get(symbol, fullSig.getBody(), paramValues)));

  auto unboundGen = GeneratorAttr::get(fullSig.getInputParamTypes(), fnLiteral,
                                       fullSig.getParamListAttrs());
  if (!bindings || llvm::all_of(bindings, [](TypedAttr binding) {
        return isa<UnboundAttr>(binding);
      })) {
    // If all the provided bindings are unbound, return the original generator.
    // FIXME: this is a hack to avoid the bug when UnboundAttr erased type
    // dependencies, which results in BindParamsAttr being folded in a wrong
    // way.
    //
    // E.g.,
    //
    // def takeClosure[
    //     origins: OriginSet,
    //     //,
    //     f: def() capturing[origins] -> None,
    // ]():
    //     pass
    //
    // where
    //
    // `f: def() capturing[origins]` will be replaced to
    // `f: def() capturing[  ?    ]`. This will make the type dependency on
    // `origins` unrecoverable.
    return unboundGen;
  }
  DenseBoolArrayAttr discharged = KGEN::getDenseBoolArrayAttr(
      unboundGen.getContext(), dischargedBodyConstraints);
  return BindParamsAttr::get(unboundGen.getContext(), unboundGen, bindings,
                             discharged, &evalContext);
}

bool FnOp::isSynthetic() { return getSynthetic(); }

/// Parse a fixed mutability specifier that occurs for implicit Origins.
// Implicit origin params are always known immut or mut, never parametric.
static ParseResult parseImplicitOriginMutability(AsmParser &p,
                                                 bool &isMutable) {
  llvm::SMLoc loc;
  StringRef mutability;
  if (p.getCurrentLocation(&loc) || p.parseKeyword(&mutability))
    return failure();
  if (mutability != "mut" && mutability != "imm")
    return p.emitError(loc, "expected 'mut' or 'imm' to indicate mutability");
  isMutable = mutability == "mut";
  return success();
}

static void printImplicitOriginMutability(AsmPrinter &p, OriginType type) {
  assert((type.isMutableKnown(true) || type.isMutableKnown(false)) &&
         "Implicit Origins are always known mut or imm");
  p << (type.isMutableKnown(true) ? "mut " : "imm ");
}

// These FuncOp attributes are disallowed while parsing since they can
// be inferred. Likewise while printing we ignore them.
static StringRef disallowedAttrNames[] = {
    "sym_name",          "exportKind",   "constraints", "implements",
    "funcTypeGenerator", "functionType", "sym_name",    "argNames",
    "paramNames",        "evaluator",    "defaultImpl", "inlineLevel",
    "paramDecl",         "params",       "decorators",  "argPassingKinds"};

static ParseResult parseLITFunctionSignature(
    OpAsmParser &p, SmallVectorImpl<OpAsmParser::Argument> &args,
    ParamDeclArrayAttr &params, FunctionType &functionType,
    FnTypeGeneratorType &signature) {
  llvm::SMLoc startLoc = p.getCurrentLocation();

  TypedAttr captureOrigins;
  auto originSet = OriginSetType::get(p.getContext());
  if (succeeded(p.parseOptionalColon())) {
    if (parseParamValue(p, captureOrigins, originSet) || p.parseColon())
      return failure();
  } else {
    captureOrigins = OriginSetAttr::get({}, originSet);
  }
  bool isNestedOriginsReadOnly =
      succeeded(p.parseOptionalKeyword("no_nested_origin_exclusivity"));
  bool definesInteriorOrigins =
      succeeded(p.parseOptionalKeyword("defines_interior_origins"));

  SmallVector<ParamDeclAttr> originDecls;
  auto parseOriginDecl = [&]() -> ParseResult {
    bool isMutable = false;
    StringAttr name;
    if (parseImplicitOriginMutability(p, isMutable) || parseParamName(p, name))
      return failure();
    originDecls.push_back(
        ParamDeclAttr::get(name, OriginType::get(p.getContext(), isMutable)));
    return success();
  };

  // Parse the parameter spec.
  PogListAttr paramListAttr;
  if (parseOptionalParameterSpec(p, params, paramListAttr))
    return failure();

  // Parse implicit origin decls.
  if (p.parseCommaSeparatedList(AsmParser::Delimiter::OptionalSquare,
                                parseOriginDecl))
    return failure();

  SmallVector<StringAttr> argNames;
  SmallVector<TypedAttr> defaults;
  SmallVector<ArgConvention> argConventions;
  SmallVector<VariadicKind> argVariadics;
  std::optional<ArgConvention> origVariadicConvention;

  PassingKindParser passingKindParser(p);
  size_t idx = 0;
  auto parseArg = [&](SmallVectorImpl<Type> &argTypes) -> ParseResult {
    if (OptionalParseResult res = passingKindParser.parseOptionalStarSlash();
        res.has_value())
      return res.value();

    // Parse the ssa name first.
    OpAsmParser::Argument &arg = args.emplace_back();
    StringAttr &argName = argNames.emplace_back();
    if (p.parseOperand(arg.ssaName, /*allowResultNumber=*/false))
      return failure();
    // A user defined name might follow in brackets, e.g. `%arg0[someName]`; if
    // omitted, we just use the SSA name.
    if (succeeded(p.parseOptionalLSquare())) {
      // The user defined names might be escaped, since we allow arbitrary
      // identifiers, e.g.: `%arg1[*"!415weirdname"]`.
      if (parseParamName(p, argName) || p.parseRSquare())
        return failure();
    } else {
      // The parsed SSA name comes prepended with '%', so drop it.
      argName = p.getBuilder().getStringAttr((arg.ssaName.name.drop_front()));
    }

    // A colon and type should come next, followed by an optional location,
    // input convention, and variadicness.
    if (p.parseColonType(arg.type) ||
        p.parseOptionalLocationSpecifier(arg.sourceLoc) ||
        parseConventionAndVariadicness(p, argConventions.emplace_back(),
                                       argVariadics.emplace_back(),
                                       origVariadicConvention, idx++))
      return failure();

    // Parse an optional default value.
    TypedAttr defaultVal;
    if (failed(parseOptionalDefaultValue(p, defaultVal, arg.type,
                                         hasAddress(argConventions.back()))))
      return failure();
    defaults.push_back(defaultVal);
    argTypes.push_back(arg.type);
    return success();
  };

  FnEffects effects;
  if (failed(parseSignatureValues(p, parseArg, functionType, effects,
                                  /*optionalResultList=*/true)))
    return failure();

  SmallVector<PassingKind> argPassingKinds;
  passingKindParser.populatePassingKinds(argPassingKinds);

  auto pogList =
      PogListAttr::get(p.getContext(), argNames, argPassingKinds, argVariadics,
                       defaults, origVariadicConvention,
                       /*bodyConstraints=*/{});
  auto metadata = FnMetaOriginDataAttr::get(
      p.getContext(), originDecls.size(), captureOrigins,
      isNestedOriginsReadOnly, definesInteriorOrigins);
  signature = FuncTypeGeneratorType::remapToFuncTypeGenerator(
      params, functionType, argConventions, effects, metadata, paramListAttr,
      [&] { return p.emitError(startLoc); }, pogList);
  if (!signature)
    return failure();

  // Replace named implicit origin parameter references with index-based
  // references in the signature.
  signature = signature.replaceImplicitOriginsWithIndexes(originDecls);

  // The formal params are the declared params + the implicit origin decls.
  SmallVector<ParamDeclAttr> allParams;
  allParams.reserve(params.size() + originDecls.size());
  llvm::append_range(allParams, params);
  llvm::append_range(allParams, originDecls);
  params = ParamDeclArrayAttr::get(p.getContext(), allParams);
  return success();
}

static void printLITFunctionSignature(OpAsmPrinter &p, Region *region,
                                      ArrayRef<ParamDeclAttr> params,
                                      FunctionType functionType,
                                      FnTypeGeneratorType signature) {
  ArrayRef<ParamDeclAttr> originDecls =
      params.drop_front(signature.getInputParamTypes().size());

  if (!isEmptyOriginSet(signature.getCaptureOrigins())) {
    p << ':';
    printParamValue(p, signature.getCaptureOrigins());
    p << ':';
  }
  if (signature.getIsNestedOriginsReadOnly())
    p << "no_nested_origin_exclusivity";
  if (signature.getDefinesInteriorOrigins()) {
    if (signature.getIsNestedOriginsReadOnly())
      p << ' ';
    p << "defines_interior_origins";
  }

  ParameterEvaluator evaluator;
  printOptionalParameterSpec(p, params.drop_back(originDecls.size()),
                             signature.getParamListAttrs(), evaluator);

  if (!originDecls.empty()) {
    p << '[';
    llvm::interleaveComma(originDecls, p, [&](ParamDeclAttr decl) {
      printImplicitOriginMutability(p, cast<OriginType>(decl.getType()));
      printParamName(p, decl.getName());
    });
    p << ']';
  }

  PogListAttr argListAttr = signature.getArgListAttrs();
  PassingKindPrinter passingKindPrinter(p, argListAttr, '|');
  auto printElt = [&](unsigned i) {
    passingKindPrinter.printOptionalStarSlash(i);

    StringAttr argName = signature.getArgName(i);
    Type argType = functionType.getInput(i);
    if (region) {
      // Print the SSA name first, which might have been automatically uniqued.
      BlockArgument arg = region->getArgument(i);
      std::string ssaName;
      llvm::raw_string_ostream ss(ssaName);
      p.printOperand(arg, ss);
      p << ssaName;

      // If different from the SSA name (e.g. because it was uniqued, or
      // because it contains characters that need escaping), also print the
      // user-defined argument name in brackets.
      if (StringRef(ssaName).drop_front() != argName) {
        p << "[";
        printParamName(p, argName);
        p << "]";
      }
      argType = arg.getType();
    } else {
      p << "%arg" << i;
      std::string syntheticName = ("arg" + llvm::Twine(i)).str();
      if (argName.getValue() != syntheticName) {
        p << "[";
        printParamName(p, argName);
        p << "]";
      }
    }

    // Print the type, then the optional location.
    p << ": ";
    p.printType(argType);
    if (region)
      p.printOptionalLocationSpecifier(region->getArgument(i).getLoc());
    auto argConv = signature.getArgConvention(i);

    if (argListAttr.isPack(i) || argListAttr.isPosVarArg(i))
      argConv = signature.getVariadicConvention(i);

    printConventionAndVariadicness(p, argConv, argListAttr.getVariadicKind(i));

    if (TypedAttr defaultOr = argListAttr.getDefault(i)) {
      printOptionalDefaultValue(p, evaluator.getReboundAttribute(defaultOr),
                                signature.getArgument(i), hasAddress(argConv));
    }

    // Check if we are at the end; if so, we might still have to print a '/'.
    passingKindPrinter.printOptionalTrailingSlash(i);
  };
  printSignatureValues(p, printElt, functionType, signature.getArgConventions(),
                       signature.getFnEffects(),
                       /*optionalResultList=*/true);
}

/// Parses a LIT Generator.
ParseResult FnOp::parse(OpAsmParser &parser, OperationState &result) {
  ExportKindAttr exportKind;
  if (parseSymbolExport(parser, exportKind))
    return failure();
  result.addAttribute(getExportKindAttrName(result.name), exportKind);

  // Parse the name as a symbol or a parameter declaration.
  StringAttr nameAttr;
  bool isParamDecl = false;
  if (parser.parseOptionalSymbolName(nameAttr)) {
    if (parseParamName(parser, nameAttr))
      return failure();
    isParamDecl = true;
  }
  if (!isParamDecl)
    result.addAttribute(getSymNameAttrName(result.name), nameAttr);

  // Parse the function signature.
  SmallVector<OpAsmParser::Argument> entryArgs;
  ParamDeclArrayAttr params;
  FunctionType functionType;
  FnTypeGeneratorType signature;
  if (parseLITFunctionSignature(parser, entryArgs, params, functionType,
                                signature))
    return failure();

  // Parse additional function attributes.
  InlineLevelAttr inlineLevel;
  DecoratorsAttr decorators;
  if (parseOptionalInline(parser, inlineLevel) ||
      parseOptionalDecorators(parser, decorators))
    return failure();
  result.addAttribute(getInlineLevelAttrName(result.name), inlineLevel);
  result.addAttribute(getDecoratorsAttrName(result.name), decorators);
  result.addAttribute(getParamsAttrName(result.name), params);
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(functionType));
  if (isParamDecl)
    result.addAttribute(getParamDeclAttrName(result.name),
                        ParamDeclAttr::get(nameAttr, signature));

  // If function attributes are present, parse them.
  NamedAttrList parsedAttributes;
  llvm::SMLoc attributeDictLocation = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDictWithKeyword(parsedAttributes))
    return failure();

  result.addAttribute(getFuncTypeGeneratorAttrName(result.name),
                      TypeAttr::get(signature));

  // Disallow attributes that are inferred from elsewhere in the attribute
  // dictionary.
  for (StringRef disallowed : disallowedAttrNames) {
    if (parsedAttributes.get(disallowed))
      return parser.emitError(attributeDictLocation, "'")
             << disallowed
             << "' is an inferred attribute and should not be specified in the "
                "explicit attribute dictionary";
  }
  result.attributes.append(parsedAttributes);

  // Parse the required function body.
  Region *region = result.addRegion();
  if (parser.parseRegion(*region, entryArgs))
    return failure();

  return success();
}

// Print the FnOp using the shared printing logic.
void FnOp::print(OpAsmPrinter &p) {
  using namespace mlir::function_interface_impl;

  // Print the operation and the function name.
  printSymbolExport(p, *this, getExportKindAttr());
  p << ' ';
  if (ParamDeclAttr decl = getParamDeclAttr())
    printParamName(p, decl.getName());
  else
    p.printSymbolName(*getSymName());

  // Print the function arguments. Here we need all the use defined names.
  printLITFunctionSignature(p, &getBodyRegion(), getParams(), getFunctionType(),
                            getFuncTypeGenerator());
  printOptionalInline(p, getInlineLevel());
  printOptionalDecorators(p, *this, getDecorators());

  // Don't print the following in lit.fn.
  SmallVector<StringRef> ignoredAttrNames(
      (ArrayRef<StringRef>(disallowedAttrNames)));
  if (getLLVMMetadataArray().empty())
    ignoredAttrNames.push_back(getLLVMMetadataArrayAttrName());
  if (getLLVMArgMetadataArray().empty())
    ignoredAttrNames.push_back(getLLVMArgMetadataArrayAttrName());
  if (!isImplicitConversion())
    ignoredAttrNames.push_back(getImplicitConversionAttrName());

  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                     ignoredAttrNames);

  p << ' ';
  p.printRegion(getBodyRegion(), /*printEntryBlockArgs=*/false);
}

// Name the arguments of the region with the argument names.
void FnOp::getAsmBlockArgumentNames(
    Region &region, llvm::function_ref<void(Value, StringRef)> setNameFn) {
  if (region.empty())
    return;

  // Set a name for each argument.
  for (auto [idx, arg] : llvm::enumerate(getBody()->getArguments()))
    setNameFn(arg, getFuncTypeGenerator().getArgName(idx).strref());
}

LogicalResult FnOp::verify() {
  if ((getLLVMMetadataArray().size() & 1) != 0)
    return emitOpError("expected an even number elements in LLVMMetadataArray");
  if (ArrayAttr argsArray = getLLVMArgMetadataArray();
      !argsArray.empty() && argsArray.size() != getNumArguments()) {
    return emitOpError("LLVMArgMetadataArray size does not equal number of "
                       "arguments, got ")
           << argsArray.size();
  }

  // Check that the number of argument labels matches the number of argument
  // types.
  if (getFuncTypeGenerator().getBody().getArgListAttrs().getPogs().size() !=
      getFunctionType().getNumInputs())
    return emitOpError("incorrect number of value parameter labels");

  // Verify the correct number of parameters.
  if (getFuncTypeGenerator().getInputParamTypes().size() +
          getFuncTypeGenerator().getNumImplicitOriginDecls() !=
      getInputParams().size()) {
    return emitOpError("incorrect number of input params: have ")
           << getParams().size() << ", but expected "
           << getFuncTypeGenerator().getNumImplicitOriginDecls()
           << " implicit origins and "
           << getFuncTypeGenerator().getInputParamTypes().size()
           << " input params";
  }

  return success();
}

void FnOp::walkDeclarations(function_ref<void(ParamDeclAttr)> walkDecl) {
  if (auto decl = getParamDeclAttr())
    walkDecl(decl);
}

void FnOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  if (auto decl = getParamDeclAttr())
    walkDef(decl, &getBodyRegion());
}

void FnOp::renameDeclarations(ArrayRef<ParamDeclAttr> decls) {
  if (getParamDecl()) {
    assert(decls.size() == 1);
    setParamDeclAttr(decls.front());
  } else {
    assert(decls.empty());
  }
}

/// This operation has no uses to collect in its current scope.
void FnOp::collectParameterUses(function_ref<void(Attribute)> scanAttr,
                                function_ref<void(Type)> scanType) {}

SmallVector<ParamDeclAttr> FnOp::collectAllParams(bool includeImplOrigins) {
  auto opAndPogLists =
      collectAncestorPogListForFnOp(getOperation()->getParentOp());

  SmallVector<ParamDeclAttr> result;
  for (auto [op, pogList] : opAndPogLists)
    for (ParamDeclAttr param : cast<DeclInterface>(op).getInputParams())
      result.push_back(param);

  auto params = getParams();
  if (!includeImplOrigins)
    params =
        params.drop_back(getFuncTypeGenerator().getNumImplicitOriginDecls());
  llvm::append_range(result, params);
  return result;
}

FnTypeGeneratorType FnOp::getFullSignature() {
  return LIT::getFullSignature((*this)->getParentOp(), getFuncTypeGenerator());
}

/// Build a function in a default configuration, used by member synthesis.
void FnOp::build(OpBuilder &builder, OperationState &result, StringAttr name,
                 StringAttr sourceName, FuncTypeGeneratorType signature) {
  MLIRContext *ctx = builder.getContext();
  UnitAttr none;
  build(builder, result, name, /*sym_visibility=*/nullptr, ParamDeclAttr(),
        TypeAttr::get(signature),
        TypeAttr::get(signature.getBody().getValues()),
        ParamDeclArrayAttr::get(ctx, {}), DecoratorsAttr::get(ctx, {}),
        /*isStatic=*/none, /*isSynthetic=*/none,
        ImplicitConversionKindAttr::get(ctx, ImplicitConversionKind::None),
        /*isExtern=*/none,
        /*isDefaultedTraitFn=*/none,
        ExportKindAttr::get(ctx, ExportKind::NotExported),
        InlineLevelAttr::get(ctx, InlineLevel::Automatic),
        builder.getI8IntegerAttr(uint8_t(SpecialFunctionKind::kNormal)),
        /*linkageName=*/LinkageNameAttr(), sourceName, /*inheritedFrom=*/{},
        /*defaultFnRef=*/{}, StringAttr(), DocStringAttr(),
        /*deprecationInfo=*/{}, /*unavailableInfo=*/{},
        /*hasStableDecorator=*/none, /*stableSinceVersion=*/{},
        ArrayAttr::get(ctx, {}), ArrayAttr::get(ctx, {}), Attribute());
  result.regions[0]->push_back(new Block());
}

//===----------------------------------------------------------------------===//
// StructDeclOp
//===----------------------------------------------------------------------===//

static ParseResult parseSymbol(AsmParser &p, SymbolConstantAttr &symbol) {
  TypedAttr value;
  if (parseColonTypeParamValue(p, value))
    return failure();
  symbol = cast<SymbolConstantAttr>(value);
  return success();
}

static void printSymbol(AsmPrinter &p, Operation *, SymbolConstantAttr symbol) {
  printColonTypeParamValue(p, symbol);
}

static ParseResult parseTypeConvention(AsmParser &p, TypeConvention &value) {
  StringRef str;
  value = TypeConvention::MemoryOnly;
  if (succeeded(p.parseOptionalKeyword(
          &str, {stringifyEnum(TypeConvention::MemoryOnly),
                 stringifyEnum(TypeConvention::RegisterPassable),
                 stringifyEnum(TypeConvention::RegisterPassableTrivial),
                 stringifyEnum(TypeConvention::Unspecified)})))
    value = *symbolizeTypeConvention(str);
  return success();
}

static void printTypeConvention(AsmPrinter &p, Operation *op,
                                TypeConvention value) {
  if (value != TypeConvention::MemoryOnly)
    p << ' ' << stringifyTypeConvention(value);
}

static ParseResult parseStructParameterSpec(AsmParser &p,
                                            ParamDeclArrayAttr &params,
                                            TypeAttr &signature,
                                            TypeAttr &canonicalTraitAttr) {
  llvm::SMLoc startLoc = p.getCurrentLocation();
  PogListAttr paramListAttr;
  if (parseOptionalParameterSpec(p, params, paramListAttr))
    return failure();

  TraitType canonicalTrait;
  if (succeeded(p.parseOptionalLParen())) {
    if (parseParamType(p, canonicalTrait) || p.parseRParen())
      return failure();
  } else {
    canonicalTrait = TraitType::get(p.getContext(), {});
  }
  canonicalTraitAttr = TypeAttr::get(canonicalTrait);

  auto sig = TypeSignatureType::remapToSignature(
      [&] { return p.emitError(startLoc); }, params, paramListAttr);
  if (!sig)
    return failure();
  signature = TypeAttr::get(sig);
  return success();
}

static ParseResult parseStructParameterSpec(AsmParser &p,
                                            ParamDeclArrayAttr &params,
                                            TypeAttr &signature) {
  TypeAttr canonicalTraitAttr = nullptr;
  return parseStructParameterSpec(p, params, signature, canonicalTraitAttr);
}

static void printStructParameterSpec(AsmPrinter &p, Operation *op,
                                     ArrayRef<ParamDeclAttr> params,
                                     TypeAttr signature,
                                     TypeAttr canonicalTraitAttr) {
  auto sig = cast<TypeSignatureType>(signature.getValue());
  ParameterEvaluator evaluator;
  printOptionalParameterSpec(p, params, sig.getParamListAttrs(), evaluator);

  if (canonicalTraitAttr) {
    TraitType canonicalTrait = cast<TraitType>(canonicalTraitAttr.getValue());
    if (!canonicalTrait.getSymbols().empty()) {
      p << '(';
      printParamType(p, canonicalTrait);
      p << ')';
    }
  }
}
static void printStructParameterSpec(AsmPrinter &p, Operation *op,
                                     ArrayRef<ParamDeclAttr> params,
                                     TypeAttr signature) {
  TypeAttr canonicalTraitAttr{};
  printStructParameterSpec(p, op, params, signature, canonicalTraitAttr);
}

bool StructDeclOp::isSynthetic() { return getSynthetic(); }

ArrayRef<ParamDeclAttr> StructDeclOp::getInputParams() { return getParams(); }

LIT::StructType StructDeclOp::bindReference(ArrayRef<TypedAttr> paramValues) {
  SymbolRefAttr symbol = getFullyResolvedSymbolRef(*this);
  TypeSignatureType sig = getSignature();

  if (paramValues.empty()) {
    // Create a fully unbound reference to the type.
    SmallVector<TypedAttr> unbound;
    ParameterEvaluator evaluator;
    for (Type type : sig.getParamTypes()) {
      unbound.push_back(UnboundAttr::get(evaluator.getReboundType(type)));
      evaluator.appendIndexBinding(unbound.back());
    }
    return LIT::StructType::get(symbol, unbound, sig);
  }

  // Compute the resultant signature.
  auto newSig = sig.bind(paramValues);
  return LIT::StructType::get(symbol, paramValues, newSig);
}

std::optional<size_t> StructDeclOp::findFieldIndex(StringRef name) {
  size_t index = 0;
  for (StructFieldOp field : getFieldDecls()) {
    if (field.getName() == name)
      return index;
    ++index;
  }
  return std::nullopt;
}

TypedAttr StructDeclOp::getFieldType(StringRef name, Type metaType) {
  for (StructFieldOp field : getFieldDecls())
    if (field.getName() == name)
      return TypeParamAttr::get(field.getType(), metaType);

  return nullptr;
}

void StructDeclOp::getFieldNames(SmallVectorImpl<StringAttr> &names) {
  for (StructFieldOp field : getFieldDecls())
    names.push_back(field.getNameAttr());
}

void StructDeclOp::getFieldTypes(SmallVectorImpl<TypedAttr> &types,
                                 Type metaType) {
  for (StructFieldOp field : getFieldDecls())
    types.push_back(TypeParamAttr::get(field.getType(), metaType));
}

/// Verify the debuginfo scope of an op that must be a top-level declaration.
static LogicalResult verifyTopLevelLocScope(Operation *op) {
  Location loc = op->getLoc();

  // If the decl doesn't contain a location scope, we don't verify it.
  auto fusedLoc = dyn_cast<mlir::FusedLocWith<DebugInfo::DIScopeAttr>>(loc);
  if (!fusedLoc)
    return success();

  DebugInfo::DIScopeAttr scope = fusedLoc.getMetadata();
  auto funcScope = dyn_cast<DebugInfo::DIFileAttr>(scope);
  if (funcScope)
    return success();
  return op->emitOpError("must have file scope in location, but got ") << scope;
}

/// Return the debuginfo scope of an op that must be a top-level declaration.
static DebugInfo::DIFileAttr getTopLevelScope(Operation *op) {
  if (auto fusedLoc =
          dyn_cast<mlir::FusedLocWith<DebugInfo::DIFileAttr>>(op->getLoc()))
    return fusedLoc.getMetadata();
  return {};
}

LogicalResult StructDeclOp::verify() {
  if (getFields().getNumArguments())
    return emitOpError("expected declaration body to have no arguments");
  return verifyTopLevelLocScope(*this);
}

DebugInfo::DIScopeAttr StructDeclOp::getLocScope() {
  return getTopLevelScope(*this);
}

/// Verify that there are no duplicate field names.
LogicalResult StructDeclOp::verifyRegions() {
  SmallDenseMap<StringAttr, StructFieldOp, 8> seenFields;
  for (Operation &op : getFields().front()) {
    auto field = dyn_cast<StructFieldOp>(&op);
    if (!field)
      continue;
    auto [it, inserted] = seenFields.try_emplace(field.getNameAttr(), field);
    if (!inserted) {
      return (field.emitError("duplicate struct field ") << field.getNameAttr())
                 .attachNote(it->second.getLoc())
             << "see previous declaration here";
    }
  }
  return success();
}

void StructDeclOp::build(OpBuilder &builder, OperationState &result,
                         StringAttr name) {
  MLIRContext *ctx = builder.getContext();
  build(builder, result, name, /*sym_visibility=*/nullptr,
        TypeAttr::get(TypeSignatureType::get(ctx)),
        ParamDeclArrayAttr::get(ctx, {}), DecoratorsAttr::get(ctx, {}),
        TypeAttr::get(TraitType::get(ctx, {})),
        /*isSynthetic=*/{},
        /*nonmaterializableTarget=*/{}, /*moveInit=*/{},
        /*copyInit=*/{}, /*linearTypeErrorMsg*/ {}, /*closureSignature=*/{},
        /*docString=*/{}, /*deprecationInfo=*/{}, /*unavailableInfo=*/{},
        /*hasStableDecorator=*/{}, /*stableSinceVersion=*/{}, /*sourceName=*/{},
        /*minAlignment=*/{}, /*convention=*/{}, /*definesClosure=*/{},
        /*registerPassableConstraint=*/{});
  result.regions[0]->push_back(new Block());
}

//===----------------------------------------------------------------------===//
// StructFieldOp
//===----------------------------------------------------------------------===//

/// Parse the struct field name as a keyword literal.
static ParseResult parseKeywordAsString(OpAsmParser &p, StringAttr &name) {
  StringRef value;
  if (p.parseKeyword(&value))
    return failure();
  name = p.getBuilder().getStringAttr(value);
  return success();
}

/// Print the struct field name as a keyword literal.
static void printKeywordAsString(OpAsmPrinter &p, Operation *op,
                                 StringAttr name) {
  p << name.getValue();
}

Type StructFieldOp::getReboundType(StructType structSelfType,
                                   ParameterEvaluationContext *ctx) {
  if (structSelfType.getParamValues().empty())
    return getType();
  ParameterEvaluator evaluator(getParentOp().getParams(),
                               structSelfType.getParamValues());
  evaluator.setEvaluationContext(ctx);
  return evaluator.getReboundType(getType());
}

void StructFieldOp::build(OpBuilder &builder, OperationState &odsState,
                          StringAttr name, Type type) {
  build(builder, odsState, name, type, /*docString=*/{}, /*isDocHidden=*/false,
        /*allowLegacyAnyOrigin=*/false);
}

void StructFieldOp::build(OpBuilder &builder, OperationState &odsState,
                          const Twine &name, Type type) {
  build(builder, odsState, builder.getStringAttr(name), type);
}

//===----------------------------------------------------------------------===//
// StructInsertOp
//===----------------------------------------------------------------------===//

/// Lookup the declaration for the struct. When checking field types, we can't
/// directly compare operation types to the struct field types because they are
/// parameterized under different domains. We have to rebind them.
static std::pair<StructDeclOp, ParameterEvaluator>
lookupStructDecl(SymbolTableCollection &symbolTable, Operation *user,
                 LIT::StructType ref) {
  auto module = KGENModule::from(user, symbolTable);
  auto decl = module.lookup<StructDeclOp>(ref.getSymbol());
  if (!decl) {
    user->emitOpError("expected to find a struct decl for ") << ref;
    return {};
  }
  ParameterEvaluator evaluator(decl.getParams(), ref.getParamValues());
  return {decl, std::move(evaluator)};
}

LogicalResult
StructInsertOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto [structDecl, evaluator] =
      lookupStructDecl(symbolTable, *this, getType());
  if (!structDecl)
    return emitOpError("expected to find a struct decl for ") << getType();

  auto module = getOperation()->getParentOfType<ModuleOp>();
  mlir::LockedSymbolTableCollection lockedSymtab(symbolTable);
  LITSymTabEvaluationContext evalCtx(module, lockedSymtab);
  evaluator.setEvaluationContext(&evalCtx);

  for (StructFieldOp fieldDecl : structDecl.getFieldDecls()) {
    if (fieldDecl.getName() != getFieldAttr())
      continue;
    Type reboundType = evaluator.getReboundType(fieldDecl.getType());
    if (!isEqualCanon(reboundType, getValue().getType()))
      return emitOpError("cannot insert value of type ")
             << getValue().getType() << " into struct field " << getFieldAttr()
             << " which expected " << reboundType;
    return success();
  }

  return emitOpError("struct ")
         << getType().getSymbol() << " has no field named " << getFieldAttr();
}

OpFoldResult StructInsertOp::fold(FoldAdaptor adaptor) {
  auto value = dyn_cast_if_present<LITStructAttr>(adaptor.getContainer());
  if (!value || !adaptor.getValue())
    return {};
  auto it = llvm::find_if(value.getValues(), [&](const auto &p) {
    return std::get<0>(p) == getFieldAttr();
  });
  if (it == value.getValues().end())
    return {};
  SmallVector<std::tuple<StringAttr, TypedAttr>> values(value.getValues());
  std::get<1>(values[std::distance(value.getValues().begin(), it)]) =
      cast<TypedAttr>(adaptor.getValue());
  return LITStructAttr::get(values, getType());
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

static LogicalResult
verifyStructFieldAndType(SymbolTableCollection &symbolTable, Operation *op,
                         LIT::StructType ref, StringAttr fieldName, Type type) {
  auto [structDecl, evaluator] = lookupStructDecl(symbolTable, op, ref);
  if (!structDecl)
    return op->emitOpError("struct ") << ref.getSymbol() << " cannot be found";

  auto module = op->getParentOfType<ModuleOp>();
  mlir::LockedSymbolTableCollection lockedSymtab(symbolTable);
  LITSymTabEvaluationContext evalCtx(module, lockedSymtab);
  evaluator.setEvaluationContext(&evalCtx);

  for (StructFieldOp fieldDecl : structDecl.getFieldDecls()) {
    if (fieldDecl.getName() != fieldName)
      continue;
    Type reboundType = evaluator.getReboundType(fieldDecl.getType());
    if (!isEqualCanon(reboundType, type))
      return op->emitOpError("cannot extract value of type ")
             << type << " from struct field " << fieldName << " which has type "
             << reboundType;
    return success();
  }

  return op->emitOpError("struct ")
         << ref.getSymbol() << " has no field named " << fieldName;
}

LogicalResult
LIT::StructExtractOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyStructFieldAndType(symbolTable, *this, getContainer().getType(),
                                  getFieldAttr(), getValue().getType());
}

OpFoldResult LIT::StructExtractOp::fold(FoldAdaptor adaptor) {
  if (auto value = adaptor.getContainer())
    return StructExtractAttr::get(cast<TypedAttr>(value), getFieldAttr(),
                                  getType());

  // Fold
  //    %S = lit.struct.insert %x, %S0[a]
  //    %y = lit.struct.extract %S[a]
  // into %x
  if (auto insert = getContainer().getDefiningOp<StructInsertOp>()) {
    if (insert.getFieldAttr() == getFieldAttr())
      return insert.getOperand(0);
  }
  return {};
}

//===----------------------------------------------------------------------===//
// ExtensionDeclOp
//===----------------------------------------------------------------------===//

DebugInfo::DIScopeAttr ExtensionDeclOp::getLocScope() {
  return getTopLevelScope(*this);
}

void ExtensionDeclOp::build(OpBuilder &builder, OperationState &result,
                            StringAttr name, StringAttr targetStructName) {
  MLIRContext *ctx = builder.getContext();
  build(builder, result, name, /*sym_visibility=*/nullptr,
        TypeAttr::get(TypeSignatureType::get(ctx)),
        ParamDeclArrayAttr::get(ctx, {}), targetStructName, /*targetStruct=*/{},
        /*immediateParents=*/TraitSymbolArrayAttr::get(ctx, {}),
        /*canonicalTrait=*/{});
  result.regions[0]->push_back(new Block());
}

//===----------------------------------------------------------------------===//
// RefStructGEROp
//===----------------------------------------------------------------------===//

/// Given a reference to a struct, return the reference type to the
/// specified field, maintaining origin and mutability, assuming the type
/// is already rebound to its final type.
RefType RefStructGEROp::getReboundFieldType(RefType structRefTy,
                                            StringAttr fieldName,
                                            Type reboundType) {
  // The origin of the struct reference incorporates field sensitivity.
  auto fieldOrigin = OriginFieldAttr::get(structRefTy.getOrigin(), fieldName);
  return RefType::get(reboundType, fieldOrigin, structRefTy.getAddressSpace());
}

RefType RefStructGEROp::getFieldType(RefType structRefTy, StructFieldOp field,
                                     ParameterEvaluationContext *ctx) {
  auto structTy = sugarCast<StructType>(structRefTy.getElementType());
  return getReboundFieldType(structRefTy, field.getNameAttr(),
                             field.getReboundType(structTy, ctx));
}

LogicalResult
RefStructGEROp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Symbol verification only applies to field name access
  if (!usesFieldAccess())
    return success();

  Type structType = getContainer().getType().getElementType();
  return verifyStructFieldAndType(symbolTable, *this,
                                  cast<StructType>(structType), getFieldAttr(),
                                  getResult().getType().getElementType());
}

void RefStructGEROp::build(OpBuilder &builder, OperationState &result,
                           Value structBaseRef, StructFieldOp field) {
  auto resultType = getFieldType(cast<RefType>(structBaseRef.getType()), field);
  result.addTypes(resultType);
  result.addAttribute("field", field.getNameAttr());
  // Don't add index attribute for field access
  result.addOperands(structBaseRef);
}

void RefStructGEROp::build(OpBuilder &builder, OperationState &result,
                           Type resultType, TypedAttr index, Value container) {
  result.addTypes(resultType);
  // Don't add field attribute for index access
  result.addAttribute("index", index);
  result.addOperands(container);
}

void RefStructGEROp::build(OpBuilder &builder, OperationState &result,
                           Type resultType, StringAttr field, Value container) {
  result.addTypes(resultType);
  result.addAttribute("field", field);
  // Don't add index attribute for field access
  result.addOperands(container);
}

LogicalResult RefStructGEROp::verify() {
  // Must have exactly one of field or index
  bool hasField = getField().has_value();
  bool hasIndex = getIndex().has_value();
  if (hasField == hasIndex)
    return emitOpError("must have exactly one of 'field' or 'index' attribute");

  auto containerRefType = getContainer().getType();
  Type elementType = containerRefType.getElementType();

  if (hasField) {
    // Field access requires a concrete struct type
    if (!isa<StructType>(elementType))
      return emitOpError(
          "field access requires container to be a reference to a struct type");

    // Verify the origin is field-sensitive
    if (getType() != getReboundFieldType(containerRefType, getFieldAttr(),
                                         getType().getElementType()))
      return emitOpError("invalid origin or address space");
  } else {
    // Index access allows both StructType and ParamType
    if (!isa<StructType, KGEN::StructType, KGEN::ParamType>(elementType))
      return emitOpError(
          "index access requires container to be a reference to a struct "
          "or parametric type");
  }

  return success();
}

ErrorTreeOrSuccess RefStructGEROp::interpret(ArrayRef<Attribute> operands,
                                             InterpreterState &state) {
  // This operation doesn't need special interpretation logic - it just
  // needs to exist in the IR. For index access, canonicalization will convert
  // to field access when possible.
  return success();
}

ErrorTreeOrSuccess
RefStructGEROp::parametric_interpret(ArrayRef<Attribute> operands,
                                     ParametricInterpreterState &state) {
  return interpret(operands, state);
}

namespace {
/// Canonicalization pattern that converts index access to field name access
/// when the index is a concrete integer and the struct type is known.
struct ResolveIndexToFieldName : public mlir::OpRewritePattern<RefStructGEROp> {
  using mlir::OpRewritePattern<RefStructGEROp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(RefStructGEROp op,
                  mlir::PatternRewriter &rewriter) const override {
    // Only applies to index access
    if (!op.usesIndexAccess())
      return failure();

    // Only lower when index is a concrete integer
    auto indexAttr = dyn_cast<IntegerAttr>(*op.getIndex());
    if (!indexAttr)
      return failure(); // Still parametric, wait for elaboration

    unsigned index = indexAttr.getInt();

    // Get the struct type from the container. If the element type is still
    // parametric (ParamType), wait for further elaboration.
    auto containerRefType = op.getContainer().getType();
    auto structType =
        dyn_cast<LIT::StructType>(containerRefType.getElementType());
    if (!structType)
      return failure();

    // Look up the struct declaration
    SymbolTableCollection symbolTable;
    auto module = KGENModule::from(op.getOperation(), symbolTable);
    auto structDecl = module.lookup<StructDeclOp>(structType.getSymbol());
    if (!structDecl)
      return op.emitOpError("could not find struct declaration for ")
             << structType.getSymbol();

    // Find the field at the given index
    StructFieldOp targetField;
    unsigned currentIdx = 0;
    for (StructFieldOp field : structDecl.getFieldDecls()) {
      if (currentIdx == index) {
        targetField = field;
        break;
      }
      ++currentIdx;
    }

    if (!targetField)
      return op.emitOpError("field index ") << index << " is out of bounds";

    // Replace with field name access
    rewriter.replaceOpWithNewOp<RefStructGEROp>(op, op.getContainer(),
                                                targetField);
    return success();
  }
};
} // namespace

void RefStructGEROp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                 MLIRContext *context) {
  patterns.add<ResolveIndexToFieldName>(context);
}

//===----------------------------------------------------------------------===//
// RefStructGEROp - Custom Assembly Format
//===----------------------------------------------------------------------===//

// Syntax:
//   Field access:  lit.ref.struct.ger %s[fieldname] : <@Struct, mut l> -> T
//   Index access:  lit.ref.struct.ger %s[idx 0] : <@Struct, mut l> -> T
//   Parametric:    lit.ref.struct.ger %s[idx I] : <!kgen.param<T>, mut l> -> R

ParseResult RefStructGEROp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand container;
  Type containerType, resultType;

  // Parse: %container
  if (parser.parseOperand(container))
    return failure();

  // Parse: '[' (field_name | 'idx' index) ']'
  if (parser.parseLSquare())
    return failure();

  StringAttr fieldAttr;
  TypedAttr indexAttr;
  bool alreadyParsedRSquare = false;

  // Check if this is index access (starts with 'idx' keyword followed by a
  // value). We need to be careful here because 'idx' could also be a field name
  // (e.g., struct IntTupleIter has a field named 'idx').
  //
  // Index access syntax: [idx VALUE]  - 'idx' keyword followed by index value
  // Field name access:   [idx]        - field named 'idx' (no value after)
  //                      [fieldname]  - any other field name
  //
  // To disambiguate: if we see 'idx' followed immediately by ']', it's a field
  // name. Otherwise, it's the keyword for index access.
  if (parser.parseOptionalKeyword("idx").succeeded()) {
    // Check if next token is ']' - if so, this is field name 'idx', not keyword
    if (parser.parseOptionalRSquare().succeeded()) {
      // Followed by ']', so 'idx' is a field name
      fieldAttr = parser.getBuilder().getStringAttr("idx");
      result.addAttribute("field", fieldAttr);
      alreadyParsedRSquare = true;
    } else {
      // Not followed by ']', so this is index access - parse the index value
      if (parseIndexParamValue(parser, indexAttr))
        return failure();
      result.addAttribute("index", indexAttr);
    }
  } else {
    // Parse field name as identifier
    StringRef fieldName;
    if (parser.parseKeyword(&fieldName))
      return failure();
    fieldAttr = parser.getBuilder().getStringAttr(fieldName);
    result.addAttribute("field", fieldAttr);
  }

  // Parse ']' if we haven't already consumed it
  if (!alreadyParsedRSquare) {
    if (parser.parseRSquare())
      return failure();
  }

  // Parse optional attributes
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse: ':' container_type '->' result_type
  if (parser.parseColon())
    return failure();

  containerType = RefType::parse(parser);
  if (!containerType)
    return failure();

  if (parser.parseArrow())
    return failure();

  // For field access, result type is just the element type (origin is computed)
  // For index access, result type is the full ref type
  if (fieldAttr) {
    Type fieldType;
    if (parseParamType(parser, fieldType))
      return failure();

    auto containerRefType = cast<RefType>(containerType);
    resultType = RefStructGEROp::getReboundFieldType(containerRefType,
                                                     fieldAttr, fieldType);
  } else {
    resultType = RefType::parse(parser);
    if (!resultType)
      return failure();
  }

  // Resolve operand and add result type
  if (parser.resolveOperand(container, containerType, result.operands))
    return failure();
  result.addTypes(resultType);

  return success();
}

void RefStructGEROp::print(OpAsmPrinter &p) {
  p << " " << getContainer() << "[";

  if (usesFieldAccess()) {
    // Field name access: print as identifier
    p << *getField();
  } else {
    // Index access: print with 'idx' keyword
    p << "idx ";
    printIndexParamValue(p, *getIndex());
  }

  p << "]";

  // Print optional attributes (excluding field and index which are handled
  // above)
  SmallVector<StringRef> elidedAttrs = {"field", "index"};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  // Print: ':' container_type '->' result_type
  p << " : ";
  getContainer().getType().print(p);
  p << " -> ";

  if (usesFieldAccess()) {
    // For field access, print just the element type
    printParamType(p, getType().getElementType());
  } else {
    // For index access, print the full ref type
    getType().print(p);
  }
}

//===----------------------------------------------------------------------===//
// RefImmutOp
//===----------------------------------------------------------------------===//

OpFoldResult RefImmutOp::fold(RefImmutOp::FoldAdaptor adaptor) {
  // If the operand is already known to be immutable then this is a noop.
  if (getRef().getType().isMutableKnown(false))
    return getRef();
  return {};
}

/// If 'value' is defined by one or more rebind-like ops, look through them.
Value RefImmutOp::stripRebinds(Value value) {
  while (1) {
    if (auto rebind = value.getDefiningOp<RebindOp>())
      value = rebind.getInput();
    else if (auto cast = value.getDefiningOp<RefImmutOp>())
      value = cast.getOperand();
    else if (auto upcast = value.getDefiningOp<RefUpcastOp>())
      value = upcast.getOperand();
    else
      return value;
  }
}

//===----------------------------------------------------------------------===//
// RefUpcastOp
//===----------------------------------------------------------------------===//

LogicalResult RefUpcastOp::verify() {
  auto srcType = getRef().getType();
  auto dstType = getResult().getType();

  // Element type and address space must be unchanged.
  if (!isEqualCanon(srcType.getElementType(), dstType.getElementType()))
    return emitOpError("element type mismatch: source has ")
           << srcType.getElementType() << " but result has "
           << dstType.getElementType();

  if (srcType.getAddressSpace() != dstType.getAddressSpace())
    return emitOpError("address space mismatch: source has ")
           << srcType.getAddressSpace() << " but result has "
           << dstType.getAddressSpace();

  // The destination origin must be an upcast of the source, which happens with
  // subtree origins, conversions to unions with other origins, and upcast
  // symbolic mutabilities, e.g. "x" -> "x&y" is less mutable than "x".  We can
  // check this by forming union(dst, src): it will collapse to dst when src is
  // already covered by dst.
  auto srcOrigin = getCanonicalAttr(srcType.getOrigin());
  auto dstOrigin = getCanonicalAttr(dstType.getOrigin());
  auto dstOriginType = cast<OriginType>(dstOrigin.getType());
  auto originUnion = OriginUnionAttr::get(
      {dstOrigin, OriginMutCastAttr::get(srcOrigin, dstOriginType)},
      dstOriginType);
  if (dstOrigin != originUnion)
    return emitOpError("result origin is not an upcast of the source origin: ")
           << srcOrigin << " is not covered by " << dstOrigin;

  return success();
}

//===----------------------------------------------------------------------===//
// RefFromPointerOp
//===----------------------------------------------------------------------===//

void RefFromPointerOp::build(OpBuilder &builder, OperationState &result,
                             Value pointer, TypedAttr origin, bool startsUninit,
                             bool endsUninit) {
  auto ptr = cast<PointerType>(pointer.getType());
  auto refType =
      RefType::get(ptr.getElementType(), origin, ptr.getAddressSpace());
  build(builder, result, refType, pointer, startsUninit, endsUninit);
}

//===----------------------------------------------------------------------===//
// RefToKgenPtrOp
//===----------------------------------------------------------------------===//

LogicalResult RefToKgenPtrOp::verify() {
  auto refType = getRef().getType();
  auto ptrType = getResult().getType();

  // Note: Element type correspondence between lit.ref and kgen.pointer is
  // intentionally not verified. The kgen element type should be the lowered
  // form of the lit element type (e.g., lit.struct -> kgen.struct), but this
  // correspondence is enforced by usage context rather than the verifier.
  // This flexibility enables using kgen.struct.gep on lit references.

  // Verify address spaces match - this is the critical safety invariant.
  if (refType.getAddressSpace() != ptrType.getAddressSpace())
    return emitOpError("address space mismatch: ref has address space ")
           << refType.getAddressSpace() << " but result has "
           << ptrType.getAddressSpace();

  return success();
}

//===----------------------------------------------------------------------===//
// RefFromKgenPtrOp
//===----------------------------------------------------------------------===//

LogicalResult RefFromKgenPtrOp::verify() {
  auto ptrType = getPointer().getType();
  auto refType = getResult().getType();

  // Note: Element type correspondence between kgen.pointer and lit.ref is
  // intentionally not verified. The lit element type should be the lifted
  // form of the kgen element type (e.g., kgen.struct -> lit.struct), but this
  // correspondence is enforced by usage context rather than the verifier.

  // Verify address spaces match - this is the critical safety invariant.
  if (ptrType.getAddressSpace() != refType.getAddressSpace())
    return emitOpError("address space mismatch: pointer has address space ")
           << ptrType.getAddressSpace() << " but result has "
           << refType.getAddressSpace();

  return success();
}

//===----------------------------------------------------------------------===//
// TraitDeclOp
//===----------------------------------------------------------------------===//

DebugInfo::DIScopeAttr TraitDeclOp::getLocScope() {
  return getTopLevelScope(*this);
}

void TraitDeclOp::build(OpBuilder &builder, OperationState &result,
                        StringAttr name) {
  MLIRContext *ctx = builder.getContext();
  UnitAttr none;
  build(builder, result, name, /*sym_visibility=*/nullptr,
        TypeAttr::get(TypeSignatureType::get(ctx)),
        ParamDeclArrayAttr::get(ctx, {}),
        TypeAttr::get(TraitType::get(ctx, {})),
        TraitSymbolArrayAttr::get(ctx, {}),
        /*convention=*/TypeConvention::Unspecified,
        /*definesClosure=*/none,
        /*docString=*/{},
        /*deprecationInfo=*/{}, /*unavailableInfo=*/{},
        /*hasStableDecorator=*/{}, /*stableSinceVersion=*/{},
        /*linearTypeErrorMsg*/ {}, /*closureSignature*/ {},
        /*sourceName=*/{});
  result.regions[0]->push_back(new Block());
}

TraitSymbolAttr TraitDeclOp::bindReference(ArrayRef<TypedAttr> paramValues) {
  assert(paramValues.size() == this->getInputParams().size() - 1);
  return TraitSymbolAttr::get(getFullyResolvedSymbolRef(*this), paramValues);
}

//===----------------------------------------------------------------------===//
// TryOp
//===----------------------------------------------------------------------===//

void TryOp::getEntryTargets(ArrayRef<Attribute> operands,
                            SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  targets.emplace_back(0, getTryRegion().getArguments());
}

ValueRange TryOp::getEntryArguments(std::optional<unsigned> target) {
  if (!target)
    return getResults();
  return getRegion(*target).getArguments();
}

ErrorTreeOrSuccess TryOp::interpret(ArrayRef<Attribute> operands,
                                    InterpreterState &state) {
  return state.transferControlFlowTo(getTryRegion(), operands);
}

ErrorTreeOrSuccess
TryOp::parametric_interpret(ArrayRef<Attribute> operands,
                            ParametricInterpreterState &state) {
  return interpret(operands, state);
}

bool TryOp::hasTrivialFinally() {
  Block &finally = getFinallyRegion().front();
  return llvm::hasSingleElement(finally) &&
         isa<TryYieldOp>(finally.getTerminator());
}

//===----------------------------------------------------------------------===//
// TryYieldOp
//===----------------------------------------------------------------------===//

bool TryYieldOp::isParentNode(Operation *op) { return isa<TryOp>(op); }

void TryYieldOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  Region *region = (*this)->getParentRegion();
  // Figure out which region this yield is in.
  if (!isa<TryOp>(region->getParentOp()))
    region = region->getParentRegion();

  switch (region->getRegionNumber()) {
  case TryOp::kTRY:
    // Yield from the 'try' region branches to the 'else' region.
    targets.emplace_back(TryOp::kELSE, getOperands());
    break;
  case TryOp::kEXCEPT:
  case TryOp::kELSE:
    // Yield from either the 'except' or 'else' regions branches back to the
    // parent operation.
    targets.emplace_back(std::nullopt, getOperands());
    break;
  case TryOp::kFINALLY:
    // The finally region is a no-op according to HLCF.
    break;
  default:
    llvm_unreachable("unknown lit.try region");
  }
}

ErrorTreeOrSuccess TryYieldOp::interpret(ArrayRef<Attribute> operands,
                                         InterpreterState &state) {
  // There should always be a parent TryOp.
  auto tryOp = cast<TryOp>((*this)->getParentOp());

  switch ((*this)->getParentRegion()->getRegionNumber()) {
  case TryOp::kTRY:
    // Yield from the 'try' region branches to the 'else' region.
    return state.transferControlFlowTo(tryOp.getElseRegion(), operands);
  case TryOp::kELSE:
    // Yield from either the 'except' or 'else' regions branches back to the
    // parent region which continues after the try.
    return state.transferControlFlowTo(tryOp, operands);
  case TryOp::kFINALLY:
    llvm_unreachable("Should be processed by LowerSemanticCF");
  default:
    llvm_unreachable("unknown lit.try region");
  }
}

ErrorTreeOrSuccess
TryYieldOp::parametric_interpret(ArrayRef<Attribute> operands,
                                 ParametricInterpreterState &state) {
  return interpret(operands, state);
}

//===----------------------------------------------------------------------===//
// TryRaiseOp
//===----------------------------------------------------------------------===//

/// Return true if the operation is a loop and has a matching label.
static bool isMatchingTry(Operation *op, StringAttr label) {
  if (auto tryOp = dyn_cast<LIT::TryOp>(op)) {
    assert(tryOp.getLabelAttr() && "LowerSemanticCF ensures labels on lit.try");
    return tryOp.getLabelAttr() == label;
  }
  return false;
}

bool TryRaiseOp::isParentNode(Operation *op) {
  return isMatchingTry(op, getLabelAttr());
}

void TryRaiseOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  targets.emplace_back(1, getOperands());
}

ErrorTreeOrSuccess TryRaiseOp::interpret(ArrayRef<Attribute> operands,
                                         InterpreterState &state) {
  // There should always be a parent TryOp.
  auto tryOp = (*this)->getParentOfType<TryOp>();
  while (tryOp) {
    if (isMatchingTry(tryOp, getLabelAttr()))
      break;
    tryOp = tryOp->getParentOfType<TryOp>();
  }
  assert(tryOp && "LowerSemanticCF ensures this before elaboration");

  return state.transferControlFlowTo(tryOp.getExceptRegion(), operands);
}

ErrorTreeOrSuccess
TryRaiseOp::parametric_interpret(ArrayRef<Attribute> operands,
                                 ParametricInterpreterState &state) {
  return interpret(operands, state);
}

//===----------------------------------------------------------------------===//
// AliasDeclOp
//===----------------------------------------------------------------------===//

static ParseResult parseAliasDeclOpValue(OpAsmParser &p,
                                         ParamDeclAttr &paramDecl,
                                         TypedAttr &value) {
  if (parseParamDecl(p, paramDecl))
    return failure();

  if (failed(p.parseOptionalEqual())) {
    // This is actually valid; an alias declaration in a trait is an associated
    // alias.
    return success();
  }

  if (p.parseLess() || parseParamValue(p, value, paramDecl.getType()) ||
      p.parseGreater())
    return failure();

  return success();
}

static void printAliasDeclOpValue(OpAsmPrinter &p, Operation *,
                                  ParamDeclAttr paramDecl, TypedAttr value) {
  printParamDecl(p, paramDecl);
  // Traits' alias declarations need no value, in which case they're associated
  // types.
  if (value) {
    p << " = <";
    printParamValue(p, value);
    p << ">";
  }
}

void AliasDeclOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  if (TypedAttr value = getValueAttr()) {
    walkDef(getParamDecl(), value);
  } else {
    // This could happen if we're in a trait's associated alias declaration.
    walkDef(getParamDecl(), ParamDefValue());
  }
}

LogicalResult AliasDeclOp::verify() {
  // Associated types in traits need no value.
  if (TypedAttr value = getValueAttr()) {
    if (getParamDecl().getType() != value.getType()) {
      return emitOpError("declares a parameter with type ")
             << getParamDecl().getType()
             << " but parameter expression has type " << value.getType();
    }
  }

  return success();
}

StringAttr AliasDeclOp::getDeclName() {
  StringRef demangled = demangleParameterName(getName().getValue());
  return StringAttr::get(getContext(), demangled);
}

//===----------------------------------------------------------------------===//
// VarDeclOp
//===----------------------------------------------------------------------===//

static ParseResult parseVarDeclType(AsmParser &p, Type &resultType,
                                    ParamDeclAttr &originDecl) {
  if (p.parseType(resultType))
    return failure();
  auto refType = dyn_cast<RefType>(resultType);
  if (!refType || !refType.isMutableKnown(true))
    return p.emitError(p.getNameLoc(),
                       "expected a mutable !lit.ref<> result type");
  // The origin must be a simple name, which becomes the name we are
  // declaring.
  auto origin = dyn_cast<ParamDeclRefAttr>(refType.getOrigin());
  if (!origin)
    return p.emitError(p.getNameLoc(),
                       "expected a !lit.ref<> with named origin");
  originDecl = ParamDeclAttr::get(origin);
  return success();
}

static void printVarDeclType(AsmPrinter &p, Operation *op, Type resultType,
                             ParamDeclAttr decl) {
  p.printType(resultType);
}

void VarDeclOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getResult(), getName());
}

void VarDeclOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  walkDef(getParamDecl(), ParamDefValue());
}

void VarDeclOp::build(OpBuilder &b, OperationState &state, Type elementType,
                      StringRef name, StringRef originName, VarDeclKind kind) {
  auto originType = b.getType<OriginType>(/*isMutable=*/true);
  auto originNameAttr = b.getAttr<StringAttr>(originName);
  auto originDecl = ParamDeclAttr::get(originNameAttr, originType);
  auto resultType = RefType::get(
      elementType, ParamDeclRefAttr::get(originNameAttr, originType));
  build(b, state, resultType, name, kind, originDecl);
}

bool VarDeclOp::isSynthetic() { return getKind() == VarDeclKind::Synthesized; }

/// Return true if this is non-synthetic variable, if its name starts with
/// something other than an underscore, and is not an argument shadow.
bool VarDeclOp::shouldWarnAboutUnused() {
  auto kind = getKind();
  // Don't warn about synthesized VarDecls, they aren't user-declared.
  return kind != VarDeclKind::Synthesized && kind != VarDeclKind::Arg &&
         kind != VarDeclKind::InitOutArg &&
         // Don't warn about things like _x, because this silences the warning.
         !getName().starts_with("_");
}

// Change the element type of the var decl to the specified RValue type,
// maintaining the origin of the vardecl.
void VarDeclOp::changeElementType(Type newElementType) {
  getResult().setType(getType().getWithElement(newElementType));
}

//===----------------------------------------------------------------------===//
// AsyncCallOp
//===----------------------------------------------------------------------===//

/// Use the result types to form the coroutine type, inheriting the throws bit.
static ParseResult parseAsyncCallOpTypes(AsmParser &p,
                                         SmallVectorImpl<Type> &operandTypes,
                                         TypedAttr callee,
                                         ArrayRef<TypedAttr> implicitOrigins) {
  SmallVector<Type> resultTypes;
  return parseCallOpTypes(p, operandTypes, resultTypes, callee,
                          implicitOrigins);
}

/// Nothing to do on print.
static void printAsyncCallOpTypes(AsmPrinter &, Operation *, TypeRange,
                                  TypedAttr, ArrayRef<TypedAttr>) {}

LogicalResult AsyncCallOp::verify() {
  auto sig = cast<FuncTypeGeneratorType>(getCallee().getType()).getBody();
  if (!sig.isAsync())
    return emitOpError("callable must be 'async'");
  if (auto litSigGen = dyn_cast<FnTypeGeneratorType>(sig)) {
    if (failed(verifyOriginParams(*this, litSigGen.getBody())) ||
        failed(verifyCallOp(*this, litSigGen.getBody(), getOperands(),
                            /*results=*/{})))
      return failure();
  }
  return success();
}

FailureOr<InlineResult> LIT::AsyncCallOp::prepInline(mlir::RewriterBase &b) {
  // Inlining not supported for this op
  return failure();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult LIT::ReturnOp::verify() {
  auto functionLike = (*this)->getParentOfType<FunctionLike>();
  if (!isa<FnOp>(functionLike))
    return emitOpError("expected to be nested inside a `lit.fn` operation");
  return checkOperandTypes(*this, functionLike.getResultTypes());
}

//===----------------------------------------------------------------------===//
// RaiseOp
//===----------------------------------------------------------------------===//

LogicalResult RaiseOp::verify() {
  Operation *op = *this;

  // Scan for an enclosing try block (where we're in the try part, not the
  // except) or a throwing function.
  while (Operation *parentOp = op->getParentOp()) {
    if (auto tryOp = dyn_cast<TryOp>(parentOp)) {
      if (&tryOp.getTryRegion().front() == op->getBlock())
        return success();
    }

    if (auto funcOp = dyn_cast<FnOp>(parentOp)) {
      if (funcOp.isThrows())
        return success();
    }
    op = parentOp;
  }

  return emitOpError("must be nested inside the 'try' region of a `lit.try` "
                     "operation or a throwing function");
}

//===----------------------------------------------------------------------===//
// UnboundRegionOp
//===----------------------------------------------------------------------===//

LogicalResult UnboundRegionOp::verify() {
  return emitOpError("is never valid. Was it not erased by the parser?");
}

//===----------------------------------------------------------------------===//
// ErrorReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ErrorReturnOp::verify() {
  auto func = (*this)->getParentOfType<FnOp>();
  if (!func)
    return emitOpError("expected to be nested inside a `lit.fn` operation");
  return checkOperandTypes(*this, func.getResultTypes());
}

bool ErrorReturnOp::isParentNode(Operation *op) { return isa<FnOp>(op); }

void ErrorReturnOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.size() == 1);
  targets.emplace_back(std::nullopt, getResult());
}

//===----------------------------------------------------------------------===//
// UnresolvedImportOp
//===----------------------------------------------------------------------===//

LogicalResult LIT::UnresolvedImportOp::verify() {
  if (getDeclNameLoc().has_value() && !getDeclName().has_value())
    return emitOpError("specified `declNameLoc` without `declName`");
  return success();
}

//===----------------------------------------------------------------------===//
// RefPackCreateOp
//===----------------------------------------------------------------------===//

/// Parses a kgen.pack.create op.
///
/// operation ::=
///   `lit.ref.pack.create` `(` operands `)` attr-dict `:` result-type
///
/// This is custom because we need to match operands at each index to the
/// resulting pack type element at that index.
static ParseResult parseRefPackCreateType(AsmParser &p, Type &resultType,
                                          SmallVectorImpl<Type> &elementTypes) {
  llvm::SMLoc loc = p.getCurrentLocation();
  if (p.parseType(resultType))
    return failure();
  auto type = dyn_cast<RefPackType>(resultType);
  if (!type)
    return p.emitError(loc, "expected a !lit.ref.pack type");

  auto variadic = type.getVariadicIfResolved();
  if (!variadic) {
    // We can only infer if we know the elements of the pack type (i.e.: it is
    // backed by a variadic attribute).
    return p.emitError(loc) << "operand types cannot be "
                               "inferred for resulting pack type "
                            << type;
  }

  // The operands have the same type as the elements but wrapped in a reference
  // with the specified origin and addr space.
  ArrayRef<TypedAttr> values = variadic.getValues();
  for (TypedAttr value : values) {
    Type eltType = type.getElementRefTypeFor(ParamType::get(value));
    elementTypes.push_back(eltType);
  }
  return success();
}

static void printRefPackCreateType(OpAsmPrinter &p, Operation *op,
                                   Type resultType, TypeRange elementTypes) {
  p << resultType;
}

LogicalResult RefPackCreateOp::verify() {
  RefPackType packType = getType();
  ParamListAttr elementTypesAttr = packType.getVariadicIfResolved();
  if (!elementTypesAttr)
    return emitOpError() << "cannot create pack with parametric element types";
  ArrayRef<TypedAttr> elementTypes = elementTypesAttr.getValues();
  if (elementTypes.size() != getNumOperands()) {
    return emitOpError() << "expected " << elementTypes.size()
                         << " operands, but got " << getNumOperands();
  }
  for (auto [i, expected, provided] :
       llvm::enumerate(elementTypes, getOperandTypes())) {
    Type type = packType.getElementRefTypeFor(ParamType::get(expected));
    if (isEqualCanon(type, provided))
      continue;
    return emitOpError() << "operand #" << i << " should have type " << type
                         << " but got " << provided;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// RefPackExtractOp
//===----------------------------------------------------------------------===//

LogicalResult
RefPackExtractOp::inferReturnTypes(MLIRContext *context,
                                   std::optional<Location> loc, Adaptor adaptor,
                                   SmallVectorImpl<Type> &inferredReturnTypes) {
  auto emitError = [&](const Twine &msg) -> LogicalResult {
    return mlir::emitOptionalError(loc, msg);
  };
  if (adaptor.getOperands().size() != 1 ||
      !isa<RefPackType>(adaptor.getPack().getType()))
    return emitError("expected 1 operand");

  auto indexAttr = dyn_cast_if_present<TypedAttr>(adaptor.getIndexAttr());
  if (!indexAttr || !indexAttr.getType().isIndex())
    return emitError("expected an index attribute");

  auto refPackTy = cast<RefPackType>(adaptor.getPack().getType());

  // The result type is a !lit.ref wrapping the type extracted from the
  // type list.  Extract the element from the type list.
  auto typeAttr = ParamListGetAttr::get(refPackTy.getVariadic(), indexAttr);
  Type type = ParamType::get(typeAttr);
  inferredReturnTypes.push_back(refPackTy.getElementRefTypeFor(type));
  return success();
}

//===----------------------------------------------------------------------===//
// EndFnOp
//===----------------------------------------------------------------------===//

LogicalResult EndFnOp::verify() {
  auto func = (*this)->getParentOfType<KGEN::FunctionLike>();
  if (!func)
    return emitOpError("expected to be nested inside a function");
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "Mojo/LITDialect/LIT.cpp.inc"
