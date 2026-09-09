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
// This file implements the KGEN dialect operations.
//
//===----------------------------------------------------------------------===//

#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/HLCFDialect/HLCFUtils.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/Compiler/VerifyUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/MDialect/ParserUtils.h"
#include "Support/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;

using HLCF::parseLoop;
using HLCF::printLoop;

//===----------------------------------------------------------------------===//
// ParamConstantOp
//===----------------------------------------------------------------------===//

static ParseResult parseParamConstantOpValue(OpAsmParser &p, TypedAttr &value,
                                             Type &resultType) {
  if (parseColonTypeOrIndex(p, resultType) || p.parseEqual() || p.parseLess() ||
      parseParamValue(p, value, resultType) || p.parseGreater())
    return failure();
  return success();
}

static void printParamConstantOpValue(OpAsmPrinter &p, Operation *,
                                      TypedAttr value, Type resultType) {
  printColonTypeOrIndex(p, value.getType());
  p << " = <";
  printParamValue(p, value);
  p << ">";
}

/// Parameter materialization operations are not allowed to materialize
/// capturing signature-typed values, since they can be inlined.
template <typename OpT>
static LogicalResult verifyParamValueOp(OpT op) {
  if (op.getValue().getType() == op.getType())
    return success();
  return op.emitOpError() << "parameter type " << op.getValue().getType()
                          << " does not match result type " << op.getType();
}

/// Return true if the parameter value contains symbol constants, making the
/// operation implicit parametric.
static bool containsSymbolConstants(TypedAttr value) {
  mlir::AttrTypeWalker walker;
  walker.addWalk([](SymbolConstantAttr) { return WalkResult::interrupt(); });
  return walker.walk(value).wasInterrupted();
}

LogicalResult ParamConstantOp::verify() { return verifyParamValueOp(*this); }

void ParamConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  // If the type of the value has a registered pretty name, use that for the SSA
  // value name.
  if (std::optional<StringRef> name =
          getContext()->getLoadedDialect<KGENDialect>()->getTypeName(
              getType().getTypeID())) {
    setNameFn(getResult(), *name);
    return;
  }

  // Otherwise, handle some common cases here.
  if (isa<IndexType>(getType())) {
    if (auto intVal = dyn_cast<IntegerAttr>(getValue()))
      setNameFn(getResult(), ("index" + Twine(intVal.getInt())).str());
    else
      setNameFn(getResult(), "index");
  }
}

bool ParamConstantOp::isImplicitlyParametric() {
  return containsSymbolConstants(getValue());
}

void ParamConstantOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {}

void ParamConstantOp::getEffects(
    SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {

  // This is a hack designed to stop the CSE of "none" in closures.
  //
  // See MSTDL-1123 for more context, but until cross-device compilation is
  // more robust, we need this to avoid some edge case behavior around None.
  //
  // TODO(MOCO-1101): remove the need for this and make op Pure again.
  if (isa<KGEN::NoneType>(getValue().getType()) &&
      getOperation()->getParentOfType<KGEN::ParamDeclareRegionOp>())
    effects.emplace_back(mlir::MemoryEffects::Allocate::get());
}

//===----------------------------------------------------------------------===//
// ParamMaterializeOp
//===----------------------------------------------------------------------===//

LogicalResult ParamMaterializeOp::verify() { return verifyParamValueOp(*this); }

bool ParamMaterializeOp::isImplicitlyParametric() {
  return containsSymbolConstants(getValue());
}

void ParamMaterializeOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {}

//===----------------------------------------------------------------------===//
// ParamDeclareOp
//===----------------------------------------------------------------------===//

static ParseResult parseParamDeclareOpValue(OpAsmParser &p,
                                            ParamDeclAttr &paramDecl,
                                            TypedAttr &value) {
  if (parseParamDecl(p, paramDecl) || p.parseEqual() || p.parseLess() ||
      parseParamValue(p, value, paramDecl.getType()) || p.parseGreater())
    return failure();

  return success();
}

static void printParamDeclareOpValue(OpAsmPrinter &p, Operation *,
                                     ParamDeclAttr paramDecl, TypedAttr value) {
  printParamDecl(p, paramDecl);
  p << " = <";
  printParamValue(p, value);
  p << ">";
}

void ParamDeclareOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  walkDef(getParamDecl(), getValue());
}

/// Verify that the type of the declaration matches the type of the attribute.
LogicalResult ParamDeclareOp::verify() {
  if (getParamDecl().getType() == getValue().getType())
    return success();
  return emitOpError("declares a parameter with type ")
         << getParamDecl().getType() << " but parameter expression has type "
         << getValue().getType();
}

//===----------------------------------------------------------------------===//
// ParamDeclareRegionOp
//===----------------------------------------------------------------------===//

static ParseResult parseRegionOnly(OpAsmParser &p,
                                   ParamDeclArrayAttr &inputParams,
                                   TypeAttr &functionType, TypeAttr &type,
                                   InlineLevelAttr &inlineLevel, Region &body) {
  SmallVector<OpAsmParser::Argument> args;
  FunctionType functionTypeValue;
  FuncTypeGeneratorType sigGenType;
  llvm::SMLoc bodyLoc;
  ParamDeclArrayAttr resultParams;
  if (parseFunctionFuncTypeGenerator(p, args, inputParams, resultParams,
                                     functionTypeValue, sigGenType) ||
      parseOptionalInline(p, inlineLevel) || p.getCurrentLocation(&bodyLoc) ||
      p.parseRegion(body, args))
    return failure();
  if (!resultParams.empty())
    return p.emitError(p.getCurrentLocation(), "invalid result parameters");

  // Form the Signature.
  SmallVector<Type> argTypes;
  for (const OpAsmParser::Argument &arg : args)
    argTypes.push_back(arg.type);
  functionType = TypeAttr::get(functionTypeValue);
  type = TypeAttr::get(sigGenType);
  return success();
}

static ParseResult
parseRegionDeclaration(OpAsmParser &p, ParamDeclAttr &paramDecl,
                       StringAttr &sourceName, ParamDeclArrayAttr &inputParams,
                       TypeAttr &functionType, TypeAttr &type,
                       InlineLevelAttr &inlineLevel, Region &body) {
  StringAttr paramName;
  if (parseParamName(p, paramName))
    return failure();

  if (p.parseOptionalLSquare()) {
    sourceName = paramName;
  } else {
    std::string str;
    if (p.parseKeywordOrString(&str) || p.parseRSquare())
      return failure();
    sourceName = StringAttr::get(p.getContext(), str);
  }

  if (p.parseEqual() ||
      parseRegionOnly(p, inputParams, functionType, type, inlineLevel, body))
    return failure();
  paramDecl = ParamDeclAttr::get(paramName, type.getValue());
  return success();
}

static void printRegionOnly(OpAsmPrinter &p, Operation *op,
                            ParamDeclArrayAttr inputParams,
                            TypeAttr functionType, TypeAttr type,
                            InlineLevelAttr inlineLevel, Region &body) {
  printFunctionFuncTypeGenerator(p, &body, inputParams, {},
                                 cast<FunctionType>(functionType.getValue()),
                                 cast<FuncTypeGeneratorType>(type.getValue()));
  printOptionalInline(p, inlineLevel.getValue());
  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false);
}

static void printRegionDeclaration(OpAsmPrinter &p, Operation *op,
                                   ParamDeclAttr paramDecl,
                                   StringAttr sourceName,
                                   ParamDeclArrayAttr inputParams,
                                   TypeAttr functionType, TypeAttr type,
                                   InlineLevelAttr inlineLevel, Region &body) {
  printParamName(p, paramDecl.getName());
  if (paramDecl.getName() != sourceName) {
    p << "[";
    p.printKeywordOrString(sourceName.getValue());
    p << "]";
  }
  p << " = ";
  printRegionOnly(p, op, inputParams, functionType, type, inlineLevel, body);
}

bool ParamDeclareRegionOp::isIsolatedFromAbove(unsigned regionNum) {
  assert(regionNum == 0);
  return getIsolated();
}

void ParamDeclareRegionOp::notifyKnownIsolatedFromAbove(unsigned regionNum) {
  assert(regionNum == 0);
  setIsolated(true);
}

void ParamDeclareRegionOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  walkDef(getParamDecl(), &getBodyRegion());
}

/// This operation has no uses to collect in its current scope.
void ParamDeclareRegionOp::collectParameterUses(
    function_ref<void(Attribute)> scanAttr, function_ref<void(Type)> scanType) {
}

LogicalResult ParamDeclareRegionOp::verify() {
  if (ArrayAttr argsArray = getLLVMArgMetadataArray();
      !argsArray.empty() && argsArray.size() != getNumArguments())
    return emitOpError("LLVMArgMetadataArray size does not equal number of "
                       "arguments, got ")
           << argsArray.size();
  return success();
}

//===----------------------------------------------------------------------===//
// ParamApplyOp
//===----------------------------------------------------------------------===//

static ParseResult parseParamApplyOp(AsmParser &p, ParamDeclAttr &paramDecl,
                                     TypedAttr &callee,
                                     ParameterExprArrayAttr &operands) {
  StringAttr paramName;
  FuncTypeGeneratorType calleeType;
  SmallVector<TypedAttr> operandValues;
  llvm::SMLoc sigLoc;
  if (parseParamName(p, paramName) || p.parseEqual() || p.parseLSquare() ||
      parseKGENType(p, calleeType) || p.parseColon() ||
      p.getCurrentLocation(&sigLoc) || parseParamValue(p, callee, calleeType) ||
      p.parseRSquare() || p.parseLParen() ||
      failableInterleave(
          calleeType.getBody().getArguments(),
          [&](Type type) {
            return parseParamValue(p, operandValues.emplace_back(), type);
          },
          [&] { return p.parseComma(); }) ||
      p.parseRParen())
    return failure();
  if (calleeType.getBody().getNumResults() != 1)
    return p.emitError(sigLoc, "expected callee to have 1 result");
  paramDecl =
      ParamDeclAttr::get(paramName, calleeType.getBody().getResults().front());
  operands = ParameterExprArrayAttr::get(p.getContext(), operandValues);
  return success();
}

static void printParamApplyOp(AsmPrinter &p, Operation *op,
                              ParamDeclAttr paramDecl, TypedAttr callee,
                              ParameterExprArrayAttr operands) {
  printParamName(p, paramDecl.getName());
  p << " = [";
  printKGENType(p, callee.getType());
  p << ": ";
  printParamValue(p, callee);
  p << "](";
  llvm::interleaveComma(operands, p,
                        [&](TypedAttr value) { printParamValue(p, value); });
  p << ')';
}

LogicalResult ParamApplyOp::verify() {
  auto type = cast<FuncTypeGeneratorType>(getCallee().getType());
  if (type.getInputParamTypes().empty())
    return success();
  return emitOpError("callee signature must be concrete");
}

void ParamApplyOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {
  ParamDefValue def;
  def.exprs.push_back(getCallee());
  llvm::append_range(def.exprs, getOperands());
  walkDef(getParamDecl(), def);
}

void ParamApplyOp::walkDeclarations(
    function_ref<void(ParamDeclAttr)> walkDecl) {
  walkDecl(getParamDecl());
}

void ParamApplyOp::renameDeclarations(ArrayRef<ParamDeclAttr> decls) {
  assert(decls.size() == 1);
  setParamDeclAttr(decls.front());
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

bool ReturnOp::isParentNode(Operation *op) { return isa<FunctionLike>(op); }

void ReturnOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  targets.emplace_back(std::nullopt, getOperands());
}

LogicalResult ReturnOp::verify() {
  auto func = (*this)->getParentOfType<KGEN::FunctionLike>();
  if (!func)
    return emitOpError("expected to be nested inside a function");
  return checkOperandTypes(*this, func.getResultTypes());
}

//===----------------------------------------------------------------------===//
// UnreachableOp
//===----------------------------------------------------------------------===//

/// Unreachable can terminate any control flow operation.
bool UnreachableOp::isParentNode(Operation *op) { return true; }

/// No branch targets.
void UnreachableOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {}

//===----------------------------------------------------------------------===//
// ParamAssertOp
//===----------------------------------------------------------------------===//

/// This operation defines no parameters.
void ParamAssertOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {}

/// This operation is implicitly parametric.
bool ParamAssertOp::isImplicitlyParametric() { return true; }

//===----------------------------------------------------------------------===//
// GeneratorOp
//===----------------------------------------------------------------------===//

static ParseResult
parseGeneratorOp(OpAsmParser &p, ExportKindAttr &exportKind,
                 StringAttr &symName, TypeAttr &signatureAttr,
                 TypeAttr &functionTypeAttr, ParamDeclArrayAttr &inputParams,
                 InlineLevelAttr &inlineLevel, DecoratorsAttr &decorators,
                 NamedAttrList &attrs, Region &body) {
  if (parseSymbolExport(p, exportKind) || p.parseSymbolName(symName))
    return failure();

  SmallVector<OpAsmParser::Argument> args;
  FuncTypeGeneratorType signature;
  FunctionType functionType;
  ParamDeclArrayAttr resultParams;
  if (parseFunctionFuncTypeGenerator(p, args, inputParams, resultParams,
                                     functionType, signature))
    return failure();
  if (!resultParams.empty())
    return p.emitError(p.getCurrentLocation(), "invalid result parameters");

  signatureAttr = TypeAttr::get(signature);
  functionTypeAttr = TypeAttr::get(functionType);

  if (parseOptionalInline(p, inlineLevel) ||
      parseOptionalDecorators(p, decorators) ||
      p.parseOptionalAttrDictWithKeyword(attrs) ||
      p.parseRegion(body, args, /*enableNameShadowing=*/true))
    return failure();
  return success();
}

static void printGeneratorOp(OpAsmPrinter &p, Operation *op,
                             ExportKindAttr exportKind, StringAttr symName,
                             TypeAttr signature, TypeAttr functionType,
                             ParamDeclArrayAttr inputParams,
                             InlineLevelAttr inlineLevel,
                             DecoratorsAttr decorators, DictionaryAttr attrs,
                             Region &body) {
  printSymbolExport(p, op, exportKind);
  p << ' ';
  p.printSymbolName(symName);
  printFunctionFuncTypeGenerator(
      p, &body, inputParams, /*resultParams=*/{},
      cast<FunctionType>(functionType.getValue()),
      cast<FuncTypeGeneratorType>(signature.getValue()));
  printOptionalInline(p, inlineLevel.getValue());
  printOptionalDecorators(p, op, decorators);

  auto gen = cast<GeneratorOp>(op);
  SmallVector<StringRef, 10> elidedAttrs{
      gen.getExportKindAttrName(),        gen.getSymNameAttrName(),
      gen.getFuncTypeGeneratorAttrName(), gen.getFunctionTypeAttrName(),
      gen.getInputParamsAttrName(),       gen.getInlineLevelAttrName(),
      gen.getDecoratorsAttrName()};
  ArrayAttr emptyArray = ArrayAttr::get(op->getContext(), {});
  if (attrs.get(gen.getLLVMMetadataArrayAttrName()) == emptyArray)
    elidedAttrs.push_back(gen.getLLVMMetadataArrayAttrName());
  if (attrs.get(gen.getLLVMArgMetadataArrayAttrName()) == emptyArray)
    elidedAttrs.push_back(gen.getLLVMArgMetadataArrayAttrName());
  p.printOptionalAttrDictWithKeyword(attrs.getValue(), elidedAttrs);

  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false);
}

LogicalResult GeneratorOp::verify() {
  if (ArrayAttr argsArray = getLLVMArgMetadataArray();
      !argsArray.empty() && argsArray.size() != getNumArguments())
    return emitOpError("LLVMArgMetadataArray size does not equal number of "
                       "arguments, got ")
           << argsArray.size();

  // The `funcTypeGenerator` must be self-contained: a by-name
  // `ParamDeclRefAttr` only resolves against this op's `inputParams`, so it
  // must not appear. `SugarAttr` subtrees are skipped: a `SugarAttr` retains
  // the original sugared form of an alias, which embeds a `ParamDeclRefAttr`
  // purely as a display label (paired with the resolved value), not as a real
  // reference. The walk must be pre-order so the skip prunes the subtree before
  // its refs are visited.
  ParamDeclRefAttr byNameRef;
  mlir::AttrTypeWalker walker;
  walker.addWalk([](SugarAttr) { return WalkResult::skip(); });
  walker.addWalk([&](ParamDeclRefAttr ref) {
    byNameRef = ref;
    return WalkResult::interrupt();
  });
  walker.walk<mlir::WalkOrder::PreOrder>(getFuncTypeGenerator());
  if (byNameRef)
    return emitOpError("funcTypeGenerator is not self-contained: it references "
                       "parameter '")
           << byNameRef.getName().getValue() << "' by name instead of by index";

  return success();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

static ParseResult parseFuncOp(OpAsmParser &p, ExportKindAttr &exportKind,
                               StringAttr &name, TypeAttr &signature,
                               InlineLevelAttr &inlineLevel,
                               DecoratorsAttr &decorators, NamedAttrList &attrs,
                               Region &body) {
  if (parseSymbolExport(p, exportKind) || p.parseSymbolName(name))
    return failure();

  SmallVector<OpAsmParser::Argument> args;
  FunctionType functionType;
  SmallVector<ArgConvention> conventions;
  FnEffects effects;
  auto parseArg = [&](SmallVectorImpl<Type> &argTypes) -> ParseResult {
    OpAsmParser::Argument &arg = args.emplace_back();
    if (p.parseArgument(arg, /*allowType=*/true) ||
        parseArgConvention(p, conventions.emplace_back()))
      return failure();
    argTypes.push_back(arg.type);
    return success();
  };
  llvm::SMLoc loc = p.getCurrentLocation();
  if (parseSignatureValues(p, parseArg, functionType, effects,
                           /*optionalResultList=*/true))
    return failure();
  auto sig = FuncType::getChecked([&] { return p.emitError(loc); },
                                  functionType, conventions, effects);
  if (!sig)
    return failure();
  signature = TypeAttr::get(GeneratorType::get(/*inputParamTypes=*/{}, sig));

  if (parseOptionalInline(p, inlineLevel) ||
      parseOptionalDecorators(p, decorators) ||
      p.parseOptionalAttrDictWithKeyword(attrs) ||
      p.parseRegion(body, args, /*enableNameShadowing=*/true))
    return failure();
  return success();
}

static void printFuncOp(OpAsmPrinter &p, Operation *op,
                        ExportKindAttr exportKind, StringAttr name,
                        TypeAttr signature, InlineLevelAttr inlineLevel,
                        DecoratorsAttr decorators, DictionaryAttr attrs,
                        Region &body) {
  FuncType sig = cast<FuncTypeGeneratorType>(signature.getValue()).getBody();
  auto func = cast<FuncOp>(op);

  printSymbolExport(p, op, exportKind);
  p << ' ';
  p.printSymbolName(name);
  auto printArg = [&](unsigned i) {
    p.printRegionArgument(body.getArgument(i));
    printArgConvention(p, sig.getArgConvention(i));
  };
  printSignatureValues(p, printArg, sig.getValues(), sig.getArgConventions(),
                       sig.getFnEffects(),
                       /*optionalResultList=*/true);
  printOptionalInline(p, inlineLevel.getValue());
  printOptionalDecorators(p, op, decorators);

  SmallVector<StringRef, 8> elidedAttrs{
      func.getExportKindAttrName(), func.getSymNameAttrName(),
      func.getFuncTypeGeneratorAttrName(), func.getInlineLevelAttrName(),
      func.getDecoratorsAttrName()};
  if (attrs.get(func.getLLVMMetadataAttrName()) ==
      DictionaryAttr::get(op->getContext()))
    elidedAttrs.push_back(func.getLLVMMetadataAttrName());
  if (attrs.get(func.getLLVMArgMetadataAttrName()) ==
      ArrayAttr::get(op->getContext(), {}))
    elidedAttrs.push_back(func.getLLVMArgMetadataAttrName());
  if (attrs.get(func.getCrossDeviceCapturesAttrName()) ==
      StringArrayAttr::get(func.getContext(), {}))
    elidedAttrs.push_back(func.getCrossDeviceCapturesAttrName());
  p.printOptionalAttrDictWithKeyword(attrs.getValue(), elidedAttrs);

  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false);
}

LogicalResult FuncOp::verify() {
  // Skip body-based checks for functions whose body has not been filled in yet
  // (e.g. populate_captures stubs created before offload compilation).
  if (getBodyRegion().empty())
    return success();
  FunctionType type = getFunctionType();
  if (type.getNumInputs() != getNumArguments()) {
    return mlir::emitError(getLoc(), "'kgen.func' op function type expected ")
           << type.getNumInputs() << " arguments but region has "
           << getNumArguments();
  }
  for (auto [i, arg, type] :
       llvm::enumerate(getArguments(), type.getInputs())) {
    if (arg.getType() != type) {
      return mlir::emitError(arg.getLoc(), "'kgen.func' op argument #")
             << i << " type is " << arg.getType()
             << " but function type expected " << type;
    }
  }
  if (ArrayAttr argsArray = getLLVMArgMetadata();
      !argsArray.empty() && argsArray.size() != getNumArguments()) {
    return emitOpError("LLVMArgMetadataArray size does not equal number of "
                       "arguments, got ")
           << argsArray.size();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ExternGeneratorOp
//===----------------------------------------------------------------------===//

static ParseResult parseExternGenerator(OpAsmParser &p, TypeAttr &signature,
                                        TypeAttr &functionType,
                                        ParamDeclArrayAttr &inputParams) {
  SmallVector<OpAsmParser::Argument> args;
  FunctionType funcType;
  FuncTypeGeneratorType sigType;
  ParamDeclArrayAttr resultParams;
  if (parseFunctionFuncTypeGenerator(p, args, inputParams, resultParams,
                                     funcType, sigType))
    return failure();
  if (!resultParams.empty())
    return p.emitError(p.getCurrentLocation(), "invalid result parameters");
  functionType = TypeAttr::get(funcType);
  signature = TypeAttr::get(sigType);
  return success();
}

static void printExternGenerator(OpAsmPrinter &p, Operation *op,
                                 TypeAttr signature, TypeAttr functionType,
                                 ParamDeclArrayAttr inputParams) {
  printFunctionFuncTypeGenerator(
      p, /*region=*/nullptr, inputParams,
      /*resultParams=*/{}, cast<FunctionType>(functionType.getValue()),
      cast<FuncTypeGeneratorType>(signature.getValue()));
}

//===----------------------------------------------------------------------===//
// StructGeneratorOp
//===----------------------------------------------------------------------===//

static ParseResult parseStructGeneratorSpec(OpAsmParser &p,
                                            ParamDeclArrayAttr &inputParams,
                                            TypeAttr &valueDomainTypeAttr,
                                            TypeAttr &metaTypeAttr,
                                            Region &body) {
  SmallVector<ParamDeclAttr> paramAttrs;
  if (succeeded(p.parseOptionalLess()) &&
      (parseParamDeclAttrs(p, paramAttrs) || p.parseGreater()))
    return failure();
  inputParams = ParamDeclArrayAttr::get(p.getContext(), paramAttrs);

  // Derive generator type input params.
  SmallVector<Type> inputParamTypes(llvm::map_range(
      paramAttrs, [](ParamDeclAttr decl) { return decl.getType(); }));

  Type metaType;
  Type valueDomainType;
  if (parseColonTypeOrDefault(p, metaType, TypeType::get(p.getContext())) ||
      p.parseEqual() || parseKGENType(p, valueDomainType))
    return failure();
  valueDomainTypeAttr = TypeAttr::get(valueDomainType);
  metaTypeAttr = TypeAttr::get(metaType);

  auto bodyResult = p.parseOptionalRegion(body);
  if (bodyResult.has_value() && failed(*bodyResult))
    return failure();
  if (body.empty())
    body.emplaceBlock();
  return success();
}

static void printStructGeneratorSpec(OpAsmPrinter &p, Operation *op,
                                     ParamDeclArrayAttr inputParams,
                                     TypeAttr valueDomainTypeAttr,
                                     TypeAttr metaTypeAttr, Region &body) {
  if (inputParams && !inputParams.empty()) {
    p << '<';
    printParamDeclAttrs(p, inputParams);
    p << '>';
  }

  printColonTypeOrDefault(p, metaTypeAttr.getValue(),
                          TypeType::get(op->getContext()));
  p << " = ";
  printKGENType(p, valueDomainTypeAttr.getValue());

  if (!body.empty() && !body.front().empty())
    p.printRegion(body);
}

/// StructGeneratorOps are not exported for now.
ExportKind StructGeneratorOp::getExportKind() {
  return ExportKind::NotExported;
}
void StructGeneratorOp::setExportKind(ExportKind kind) {
  assert(kind == ExportKind::NotExported &&
         "StructGeneratorOp is not exported");
}

std::optional<size_t> StructGeneratorOp::findFieldIndex(StringRef name) {
  auto structType = cast<StructInstanceType>(getValueDomainType());
  size_t index = 0;
  for (StructDefFieldAttr field : structType.getFields()) {
    if (field.getName().getValue() == name)
      return index;
    ++index;
  }
  return std::nullopt;
}

TypedAttr StructGeneratorOp::getFieldType(StringRef name, Type metaType) {
  auto structType = cast<StructInstanceType>(getValueDomainType());
  for (StructDefFieldAttr field : structType.getFields())
    if (field.getName().getValue() == name)
      return field.getTypeValue();
  return nullptr;
}

void StructGeneratorOp::getFieldNames(SmallVectorImpl<StringAttr> &names) {
  auto structType = cast<StructInstanceType>(getValueDomainType());
  for (StructDefFieldAttr field : structType.getFields())
    names.push_back(field.getName());
}

void StructGeneratorOp::getFieldTypes(SmallVectorImpl<TypedAttr> &types,
                                      Type metaType) {
  auto structType = cast<StructInstanceType>(getValueDomainType());
  for (StructDefFieldAttr field : structType.getFields()) {
    // Convert from !kgen.type to (eg) AnyType.
    types.push_back(
        ParamOperatorAttr::getRebind(field.getTypeValue(), metaType));
  }
}

//===----------------------------------------------------------------------===//
// StructInstanceOp
//===----------------------------------------------------------------------===//

static ParseResult parseStructInstanceSpec(OpAsmParser &p,
                                           TypeAttr &valueDomainTypeAttr,
                                           TypeAttr &metaTypeAttr,
                                           Region &body) {
  ParamDeclArrayAttr params;
  mlir::SMLoc startLoc = p.getCurrentLocation();
  if (failed(parseStructGeneratorSpec(p, params, valueDomainTypeAttr,
                                      metaTypeAttr, body)))
    return failure();
  if (!params.empty())
    return p.emitError(startLoc, "struct.instance cannot be parameterized");
  return success();
}

static void printStructInstanceSpec(OpAsmPrinter &p, Operation *op,
                                    TypeAttr valueDomainTypeAttr,
                                    TypeAttr metaTypeAttr, Region &body) {
  printStructGeneratorSpec(p, op, {}, valueDomainTypeAttr, metaTypeAttr, body);
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

static ParseResult
parseCallOp(OpAsmParser &p, SymbolConstantAttr &calleeCst,
            SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
            SmallVectorImpl<Type> &operandTypes,
            SmallVectorImpl<Type> &resultTypes) {
  SymbolRefAttr callee;
  ParameterExprArrayAttr paramValues;
  if (p.parseAttribute(callee) || parseParameterValues(p, paramValues) ||
      p.parseOperandList(operands, AsmParser::Delimiter::Paren) ||
      p.parseColon())
    return failure();

  FuncTypeGeneratorType signature;
  FunctionType functionType;
  if (parseKGENFuncTypeGenerator(p, functionType, signature))
    return failure();
  calleeCst = SymbolConstantAttr::get(callee, signature, paramValues);
  llvm::append_range(operandTypes, functionType.getInputs());
  llvm::append_range(resultTypes, functionType.getResults());
  return success();
}

static void printCallOp(OpAsmPrinter &p, Operation *op,
                        SymbolConstantAttr calleeCst, ValueRange operands,
                        TypeRange operandTypes, TypeRange resultTypes) {
  p << calleeCst.getSymbol();
  printParameterValues(p, calleeCst.getParamValues());
  p << '(';
  p.printOperands(operands);
  p << ") : ";
  printSignatureValues(
      p, FunctionType::get(op->getContext(), operandTypes, resultTypes),
      calleeCst.getType());
}

void CallOp::concretizeCallee(IRRewriter &b, SymbolConstantAttr callee) {
  setCalleeAttr(callee);
}

void CallOp::setCalleeAttr(TypedAttr callee) {
  setCalleeAttr(cast<SymbolConstantAttr>(callee));
}

FailureOr<InlineResult> CallOp::prepInline(mlir::RewriterBase &b) {
  StringAttr label = b.getStringAttr("inlined_cf_scope");
  auto op =
      HLCF::LoopOp::create(b, getLoc(), getResultTypes(), ValueRange(), label);
  return {{op, [label, &b](Operation *op) {
             b.replaceOpWithNewOp<HLCF::BreakOp>(op, op->getOperands(), label);
           }}};
}

static LogicalResult verifyCallOp(Operation *op, ValueRange args,
                                  FuncType signature) {
  if (failed(verifyCallOperands(op, args, signature)) ||
      failed(verifyCallResults(op, op->getResults(), signature)))
    return failure();
  return success();
}

LogicalResult CallOp::verify() {
  return verifyCallOp(*this, getOperands(), getCalleeType().getBody());
}

//===----------------------------------------------------------------------===//
// CallIndirectOp
//===----------------------------------------------------------------------===//

LogicalResult CallIndirectOp::verify() {
  return verifyCallOp(*this, getArguments(), getCallee().getType().getBody());
}

//===----------------------------------------------------------------------===//
// CallParamOp
//===----------------------------------------------------------------------===//

void CallParamOp::concretizeCallee(IRRewriter &b, SymbolConstantAttr callee) {
  b.replaceOpWithNewOp<CallOp>(*this, getResultTypes(), callee, getOperands());
}

FailureOr<InlineResult> CallParamOp::prepInline(mlir::RewriterBase &b) {
  // Inlining not supported for this op
  return failure();
}

LogicalResult CallParamOp::verify() {
  return verifyCallOp(*this, getOperands(), getCalleeType().getBody());
}

//===----------------------------------------------------------------------===//
// ParamForOp
//===----------------------------------------------------------------------===//

LogicalResult ParamForOp::verify() {
  if (getNumOperands() != getNumResults()) {
    return emitOpError("has ")
           << getNumOperands() << " operands but " << getNumResults()
           << " results; it should be the same";
  }
  for (auto [i, argTy, resTy] :
       llvm::enumerate(getOperandTypes(), getResultTypes())) {
    if (argTy == resTy)
      continue;
    return emitOpError("operand #")
           << i << " has type " << argTy
           << " but corresponding result has type " << resTy;
  }
  return success();
}

void ParamForOp::getEntryTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  targets.emplace_back(0, getOperands());
}

ValueRange ParamForOp::getEntryArguments(std::optional<unsigned> target) {
  if (!target)
    return getResults();
  if (*target == 0)
    return getBody().getArguments();
  assert(*target == 1);
  return getElseRegion().getArguments();
}

ArrayRef<ParamDeclAttr> ParamForOp::getInputParams() {
  // HACK: The interface requires an ArrayRef, but we only have a single
  // element. Returning `getParamDecl` will cause a reference to a temporary to
  // be formed. Grab the reference directly from the DictionaryAttr. We know
  // alphabetically it will be last attribute.
  assert((*this)->getRegisteredInfo()->getAttributeNames().back() ==
         getParamDeclAttrName());
  static_assert(sizeof(std::pair<Attribute, ParamDeclAttr>) ==
                        sizeof(NamedAttribute) &&
                    alignof(std::pair<Attribute, ParamDeclAttr>) ==
                        alignof(NamedAttribute),
                "hack doesn't work");
  return {&((const std::pair<Attribute, ParamDeclAttr> *)&(*this)
                ->getAttrs()
                .back())
               ->second,
          1};
}

void ParamForOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {}

bool ParamForOp::isImplicitlyParametric() { return true; }

void ParamForOp::collectParameterUsesBelow(
    function_ref<void(Attribute)> scanAttr, function_ref<void(Type)> scanType) {
}

bool ParamForOp::isIsolatedFromAbove(unsigned regionNum) {
  if (regionNum == 0)
    return getBodyIsolated();
  assert(regionNum == 1);
  return getElseIsolated();
}

void ParamForOp::notifyKnownIsolatedFromAbove(unsigned regionNum) {
  if (regionNum == 0)
    return setBodyIsolated(true);
  assert(regionNum == 1);
  return setElseIsolated(true);
}

bool ParamForBreakOp::isParentNode(Operation *op) {
  return isa<ParamForOp>(op);
}

void ParamForBreakOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  // Branch to after the loop operation.
  targets.emplace_back(std::nullopt, getOperands());
}

bool ParamForContinueOp::isParentNode(Operation *op) {
  return isa<ParamForOp>(op);
}

void ParamForContinueOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  // Branch to the beginning of the body region only (not the else region).
  targets.emplace_back(0, getOperands());
}

bool ParamForGotoElseOp::isParentNode(Operation *op) {
  return isa<ParamForOp>(op);
}

void ParamForGotoElseOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.empty() && "Shouldn't exist by mem2reg time");
  // Branch to the beginning of the else region.
  targets.emplace_back(1, ValueRange());
}

//===----------------------------------------------------------------------===//
// ParamIfOp
//===----------------------------------------------------------------------===//

bool ParamIfOp::isIsolatedFromAbove(unsigned regionNum) {
  switch (regionNum) {
  case 0:
    return getThenIsolated();
  case 1:
    return getElseIsolated();
  default:
    llvm_unreachable("unknown region number");
  }
}

void ParamIfOp::notifyKnownIsolatedFromAbove(unsigned regionNum) {
  switch (regionNum) {
  case 0:
    setThenIsolated(true);
    break;
  case 1:
    setElseIsolated(true);
    break;
  default:
    llvm_unreachable("unknown region number");
  }
}

void ParamIfOp::getEntryTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.empty());
  targets.emplace_back(0);
  targets.emplace_back(1);
}

ValueRange ParamIfOp::getEntryArguments(std::optional<unsigned> target) {
  if (!target)
    return getResults();
  assert(*target == 0 || *target == 1);
  return {};
}

void ParamIfOp::walkDefinitions(
    function_ref<void(ParamDeclAttr, const ParamDefValue &)> walkDef) {}

bool ParamIfOp::isImplicitlyParametric() { return true; }

/// This operation has no uses to collect in the scopes it defines.
void ParamIfOp::collectParameterUsesBelow(
    function_ref<void(Attribute)> scanAttr, function_ref<void(Type)> scanType) {
}

//===----------------------------------------------------------------------===//
// ParamYieldOp
//===----------------------------------------------------------------------===//

bool ParamYieldOp::isParentNode(Operation *op) {
  return isa<ParamForOp, ParamIfOp>(op);
}

void ParamYieldOp::getBranchTargets(
    ArrayRef<Attribute> operands,
    SmallVectorImpl<HLCF::ControlFlowTarget> &targets) {
  assert(operands.size() == getNumOperands());
  // Branch to after the if operation.
  targets.emplace_back(std::nullopt, getOperands());
}

//===----------------------------------------------------------------------===//
// StageClosureOp
//===----------------------------------------------------------------------===//

static ParseResult parseStageClosureOp(OpAsmParser &p, Type &resultType,
                                       Region &body) {
  // we expect the following syntax:
  // kgen.stage_closure = () capturing -> index {
  // } { name = foo }
  FuncTypeGeneratorType signatureType;
  ParamDeclArrayAttr inputParams;
  ParamDeclArrayAttr resultParams;
  FunctionType functionTypeValue;
  SmallVector<OpAsmParser::Argument> args;
  llvm::SMLoc bodyLoc;
  if (p.parseEqual() ||
      parseFunctionFuncTypeGenerator(p, args, inputParams, resultParams,
                                     functionTypeValue, signatureType) ||
      p.getCurrentLocation(&bodyLoc) || p.parseRegion(body, args))
    return failure();
  if (!inputParams.empty() || !resultParams.empty())
    return p.emitError(bodyLoc, "staged closures cannot have parameters");
  resultType = signatureType;
  return success();
}

static void printStageClosureOp(OpAsmPrinter &p, Operation *op,
                                FuncTypeGeneratorType resultType,
                                Region &body) {
  p << "= ";
  printFunctionFuncTypeGenerator(p, &body, {}, {},
                                 resultType.getBody().getValues(), resultType);
  p << ' ';
  p.printRegion(body, /*printEntryBlockArgs=*/false);
}

//===----------------------------------------------------------------------===//
// CreateClosureOp
//===----------------------------------------------------------------------===//

LogicalResult
CreateClosureOp::inferReturnTypes(MLIRContext *ctx, std::optional<Location> loc,
                                  Adaptor adaptor,
                                  SmallVectorImpl<Type> &results) {
  auto callee = dyn_cast_or_null<TypedAttr>(adaptor.getCalleeAttr());
  if (!callee) {
    return mlir::emitOptionalError(
        loc, "'create_closure' expected TypedAttr 'callee'");
  }
  auto sigGen = dyn_cast<FuncTypeGeneratorType>(callee.getType());
  if (!sigGen) {
    return mlir::emitOptionalError(
        loc,
        "'create_closure' attribute 'callee' must have FuncTypeGeneratorType");
  }

  ValueRange captures = adaptor.getOperands();
  FuncType sig = sigGen.getBody();
  unsigned numCaptures = captures.size();
  if (numCaptures > sig.getNumArguments()) {
    return mlir::emitOptionalError(loc, "provided ", numCaptures,
                                   " operands but callee only has ",
                                   sig.getNumArguments(), " to bind");
  }

  ArrayRef<Type> newArgTypes = sig.getArguments().drop_front(numCaptures);
  ArrayRef<ArgConvention> newArgConvs =
      sig.getArgConventions().drop_front(numCaptures);

  FnEffects effects = sig.getFnEffects();
  if (!captures.empty())
    effects.setCapturing();

  // Drop the first `numCaptures` positional pogs from `argListAttrs` (they're
  // now bound by the closure).
  PogListAttr argListAttrs = sig.getArgListAttrs();
  if (argListAttrs && !argListAttrs.getPogs().empty()) {
    ArrayRef<PogMetadataAttr> newPogs =
        argListAttrs.getPogs().drop_front(numCaptures);
    argListAttrs =
        PogListAttr::get(ctx, newPogs, argListAttrs.getBodyConstraints(),
                         argListAttrs.getOrigVariadicConvention());
  }
  results.push_back(FuncTypeGeneratorType::get(
      sigGen.getInputParamTypes(),
      Builder(ctx).getFunctionType(newArgTypes, sig.getResults()), newArgConvs,
      effects, sig.getMetadata(), sigGen.getParamListAttrs(), argListAttrs));
  return mlir::success();
}

static ParseResult
parseClosureCaptureTypes(AsmParser &p, TypedAttr callee,
                         ArrayRef<OpAsmParser::UnresolvedOperand> captures,
                         SmallVectorImpl<Type> &captureTypes) {
  auto sigGen = dyn_cast<FuncTypeGeneratorType>(callee.getType());
  if (!sigGen) {
    return p.emitError(p.getCurrentLocation(),
                       "expected type of callee to be FuncTypeGeneratorType");
  }
  FuncType sig = sigGen.getBody();

  unsigned numCaptures = captures.size();
  if (numCaptures > sig.getNumArguments()) {
    return p.emitError(p.getCurrentLocation(), "provided ")
           << numCaptures << " operands but callee only has "
           << sig.getNumArguments() << " to bind";
  }

  ArrayRef<Type> inputs = sig.getArguments().take_front(numCaptures);
  captureTypes.append(inputs.begin(), inputs.end());
  return success();
}

static void printClosureCaptureTypes(AsmPrinter &p, Operation *,
                                     TypedAttr callee, ValueRange captures,
                                     TypeRange captureTypes) {}

LogicalResult CreateClosureOp::verify() {
  FuncType calleeSig = getCalleeType().getBody();
  if (getNumOperands() > calleeSig.getNumArguments()) {
    return emitOpError("provided ")
           << getNumOperands() << " operands but callee only has "
           << calleeSig.getNumArguments() << " to bind";
  }
  unsigned expectedArgs = calleeSig.getNumArguments() - getNumOperands();
  FuncType sig = getType().getBody();
  if (sig.getNumArguments() != expectedArgs) {
    return emitOpError("result signature has ")
           << sig.getNumArguments() << " arguments but expected "
           << expectedArgs;
  }

  for (auto [i, type, argType] :
       llvm::enumerate(getOperandTypes(),
                       calleeSig.getArguments().take_front(getNumOperands()))) {
    if (type != argType) {
      return emitOpError("operand #")
             << i << " has type " << type
             << " but callee argument type expected " << argType;
    }
  }
  for (auto [i, type, argType] :
       llvm::enumerate(sig.getArguments(),
                       calleeSig.getArguments().drop_front(getNumOperands()))) {
    if (type != argType) {
      return emitOpError("result signature argument #")
             << i << " type is " << argType << " but expected to be " << type;
    }
  }

  if (!getCaptures().empty() && !sig.isCapturing())
    return emitOpError("has captures, so result signature must be 'capturing'");
  return success();
}

FailureOr<InlineResult> CreateClosureOp::prepInline(mlir::RewriterBase &b) {
  auto op = StageClosureOp::create(b, getLoc(), getType());
  return {{op, [](Operation *) {}}};
}

//===----------------------------------------------------------------------===//
// CreateRegStubOp
//===----------------------------------------------------------------------===//

LogicalResult CreateRegStubOp::verify() {
  FuncType calleeSig =
      cast<FuncTypeGeneratorType>(getCallee().getType()).getBody();
  FuncType resSig = getType().getBody();

  if (calleeSig.isThrows() || resSig.isThrows())
    return emitOpError("throwing function not supported");
  for (Type ty : resSig.getResults())
    if (!isa<NoneType>(ty))
      return emitOpError("result signature with output types not supported");

  bool expectPromotedMemOutputs =
      resSig.hasMemoryOnlyResult() && !calleeSig.hasMemoryOnlyResult();
  unsigned expectedArgsCount =
      calleeSig.getNumArguments() + unsigned(expectPromotedMemOutputs);
  if (resSig.getNumArguments() != expectedArgsCount) {
    return emitOpError("result signature has ")
           << resSig.getNumArguments()
           << " arguments, but the expected count is " << expectedArgsCount;
  }

  for (unsigned i = 0, e = resSig.getNumArguments(); i < e; ++i) {
    Type argTy = getOriginalArgType(i);
    Type calleeTy = getCalleeArgType(i);
    if (argTy == calleeTy)
      continue;

    PointerType argPtrTy = dyn_cast<PointerType>(argTy);
    if (!argPtrTy || argPtrTy.getElementType() != calleeTy) {
      return emitOpError("result signature argument #")
             << i << " type is " << argTy
             << " but callee signature argument is " << calleeTy;
    }
  }

  return success();
}

void CreateRegStubOp::build(mlir::OpBuilder &builder,
                            mlir::OperationState &state,
                            mlir::TypedAttr callee) {
  FuncType resultSig = getStubSignatureType(
      cast<FuncTypeGeneratorType>(callee.getType()).getBody());
  auto resultTy = GeneratorType::get(/*inputParamTypes=*/{}, resultSig);
  build(builder, state, resultTy, callee);
}

FuncType CreateRegStubOp::getStubSignatureType(FuncType calleeSign) {
  FunctionType values = calleeSign.getValues();

  // Check if type is a memory type that can be promoted to value.
  // These types will be wrapped in a memory struct.
  auto canLowerToRegPassable = [](Type ty, ArgConvention conv) {
    if (!hasAddress(conv))
      return false;

    PointerType ptrTy = dyn_cast<PointerType>(ty);
    if (!ptrTy)
      return false;
    auto structElemTy = dyn_cast<StructType>(ptrTy.getElementType());
    if (structElemTy && structElemTy.isDefinitelyMemoryOnly())
      return false;

    return true;
  };

  SmallVector<Type> newArgTypes;
  for (unsigned i = 0, e = values.getNumInputs(); i < e; ++i) {
    Type argTy = values.getInput(i);
    // Replace register-passable `!kgen.pointer<T> owned_in_mem` with
    // `!kgen.pointer<struct<(T) memoryOnly>> owned_in_mem`:
    // - It guarantees the pointer arguments won't be lowered to by-value.
    // - It also tells LLVM that arguments don't alias.
    if (canLowerToRegPassable(argTy, calleeSign.getArgConvention(i))) {
      PointerType ptrTy = cast<PointerType>(argTy);
      newArgTypes.push_back(PointerType::get(
          StructType::get(calleeSign.getContext(), ptrTy.getElementType(),
                          /*isMemoryOnly=*/true)));
    } else {
      // Other types aren't changed.
      newArgTypes.push_back(argTy);
    }
  }

  return FuncType::get(FunctionType::get(calleeSign.getContext(), newArgTypes,
                                         values.getResults()),
                       calleeSign.getArgConventions());
}

Type CreateRegStubOp::getOriginalArgType(unsigned index) {
  Type rawArgTy = getType().getBody().getValues().getInput(index);
  Type calleeArgTy = getCalleeArgType(index);
  // The type isn't transformed if it's identical to callee.
  if (rawArgTy == calleeArgTy)
    return rawArgTy;

  // Wrapped types are memory types of the form `pointer<struct<(T)
  // memoryOnly>`.
  if (!hasAddress(getType().getBody().getArgConvention(index)))
    return rawArgTy;

  PointerType ptrTy = dyn_cast<PointerType>(rawArgTy);
  if (!ptrTy)
    return rawArgTy;
  auto structElemTy = dyn_cast<StructType>(ptrTy.getElementType());
  auto numElements =
      structElemTy ? structElemTy.getNumElements() : std::nullopt;
  if (!structElemTy || !numElements || *numElements != 1 ||
      !structElemTy.isDefinitelyMemoryOnly())
    return rawArgTy;

  // Returns pointer<T>.
  auto elementTypes = structElemTy.getElementTypes();
  assert(elementTypes && "numElements succeeded, so elementTypes must too");
  return PointerType::get((*elementTypes)[0]);
}

Type CreateRegStubOp::getCalleeArgType(unsigned index) {
  // Some arguments might be promoted to outputs.
  FuncType calleeSigBase = getCalleeSignature().getBody();
  FuncType resSig = getType().getBody();
  bool promotedOutputs =
      resSig.hasMemoryOnlyResult() && !calleeSigBase.hasMemoryOnlyResult();
  if (!promotedOutputs)
    return calleeSigBase.getValues().getInput(index);

  ArgConvention conv = resSig.getArgConvention(index);
  // If `conv` is ByRefResult, the promoted output has to be this argument.
  if (conv == ArgConvention::ByRefResult)
    return calleeSigBase.getValues().getResult(0);

  // A different argument is promoted.
  return calleeSigBase.getValues().getInput(index);
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

/// Given a struct type, return the type of the field at the specified index,
/// which may be parametric.
///
/// This uses ParamListGetAttr to extract from the struct's type list, which
/// automatically folds when both the struct and index are constant. For
/// parametric cases (e.g. parametric indices), it returns a ParamType that
/// will be resolved during elaboration.
static Type getStructFieldTypeAtIndex(StructType structType, TypedAttr index) {
  // The result type is the type extracted from the type list.  Extract the
  // element from the type list.  This automatically folds if constant.
  auto typeAttr =
      ParamListGetAttr::get(structType.getElementTypesVariadic(), index);
  return ParamType::get(typeAttr);
}

void StructExtractOp::build(OpBuilder &b, OperationState &state,
                            Value structVal, unsigned fieldIdx) {
  build(b, state, structVal, b.getIndexAttr(fieldIdx));
}

/// Verify the value type matches the struct element type at the given index.
static LogicalResult verifyStructValueType(Operation *op, StructType container,
                                           Attribute indexAttrGeneric,
                                           Type valueType,
                                           StringRef valueKind) {
  // Parametric indices cannot be verified until elaboration resolves them.
  auto indexAttr = dyn_cast_or_null<IntegerAttr>(indexAttrGeneric);
  if (!indexAttr)
    return success();

  if (auto elementTypesOpt = container.getElementTypes()) {
    if (!elementTypesOpt)
      return success(); // Cannot verify without resolved element types.
    SmallVector<Type> elementTypes = *elementTypesOpt;
    // If the index is concrete then we can verify it and the result type.
    if (auto intAttr = dyn_cast_if_present<IntegerAttr>(indexAttr)) {
      size_t index = intAttr.getInt();
      if (index >= elementTypes.size())
        return op->emitOpError("element index ")
               << index << " out of bounds (>=" << elementTypes.size() << ")";
    }
  }

  auto expectedType = getStructFieldTypeAtIndex(container, indexAttr);
  if (expectedType != valueType) {
    return op->emitOpError(valueKind)
           << " type " << valueType
           << " does not match struct element type at index " << indexAttr
           << ": " << expectedType;
  }

  return success();
}

LogicalResult StructExtractOp::verify() {
  return verifyStructValueType(*this, getContainer().getType(), getIndexAttr(),
                               getType(), "result");
}

LogicalResult StructExtractOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, Adaptor adaptor,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto emitError = [&](const Twine &msg) -> LogicalResult {
    return mlir::emitOptionalError(location, msg);
  };

  ValueRange operands = adaptor.getOperands();
  if (operands.size() != 1)
    return emitError("expected 1 operand");
  auto structType = dyn_cast<StructType>(operands.front().getType());
  if (!structType)
    return emitError("expected struct operand");

  TypedAttr indexAttr = adaptor.getIndexAttr();
  if (!indexAttr)
    return emitError("expected an index attribute");

  inferredReturnTypes.push_back(
      getStructFieldTypeAtIndex(structType, indexAttr));
  return success();
}

//===----------------------------------------------------------------------===//
// StructReplaceOp
//===----------------------------------------------------------------------===//

static ParseResult parseStructValueType(AsmParser &p, Type &valueType,
                                        Type structType, IntegerAttr index) {
  auto elementTypesOpt = llvm::cast<StructType>(structType).getElementTypes();
  if (!elementTypesOpt)
    return p.emitError(p.getCurrentLocation(),
                       "cannot infer element type from parametric struct");
  SmallVector<Type> elementTypes = *elementTypesOpt;
  if (index.getInt() > static_cast<int64_t>(elementTypes.size()))
    return p.emitError(p.getCurrentLocation(), "element index out of bounds (")
           << index.getInt() << " >= " << elementTypes.size() << ")";
  // Infer the value type from the struct type and index.
  valueType = elementTypes[index.getInt()];
  return success();
}

static void printStructValueType(AsmPrinter &p, Operation *op, Type valueType,
                                 Type structType, IntegerAttr index) {}

LogicalResult StructReplaceOp::verify() {
  return verifyStructValueType(*this, getContainer().getType(), getIndexAttr(),
                               getValue().getType(), "operand");
}

//===----------------------------------------------------------------------===//
// StructGEPOp
//===----------------------------------------------------------------------===//

LogicalResult StructGEPOp::verify() {
  auto pointerType = dyn_cast<PointerType>(getContainer().getType());
  if (!pointerType)
    return emitOpError("expected pointer operand");

  Type elementType = pointerType.getElementType();

  // If the index is a concrete integer, we can verify more strictly.
  if (auto indexAttr = dyn_cast<IntegerAttr>(getIndex())) {
    auto structType = dyn_cast<StructType>(elementType);
    if (!structType) {
      // Allow identity case: when input and output types are the same,
      // this is a no-op that can arise from single-element struct flattening.
      // For example, a single-element TrivialRegisterPassable struct
      // with an Int field gets flattened to just `index`, and accessing
      // field 0 becomes an identity operation.
      if (getContainer().getType() == getType())
        return success();

      if (isa<ParamType>(elementType))
        return success();

      return emitOpError("constant index requires pointer to concrete struct "
                         "type, got ")
             << elementType;
    }

    auto numElements = structType.getNumElements();
    auto elementTypes = structType.getElementTypes();
    // Skip verification for parametric structs.
    if (!numElements || !elementTypes)
      return success();

    unsigned index = indexAttr.getInt();
    if (index >= *numElements)
      return emitOpError("struct field index ")
             << index << " is out of bounds for struct with " << *numElements
             << " elements";

    // Verify result type matches the element type at the index.
    Type expectedEltType = (*elementTypes)[index];
    if (getType().getElementType() != expectedEltType &&
        !isa<ParamType>(getType().getElementType())) {
      return emitOpError("result element type ")
             << getType().getElementType()
             << " does not match struct element type " << expectedEltType
             << " at index " << index;
    }
  } else {
    // Parametric index: allow both StructType and ParamType for generic
    // contexts.
    if (!isa<StructType>(elementType) && !isa<ParamType>(elementType))
      return emitOpError("expected pointer to struct or parametric type, got ")
             << elementType;
  }

  return success();
}

void StructGEPOp::build(OpBuilder &builder, OperationState &result,
                        Value container, unsigned index) {
  auto pointerType = cast<PointerType>(container.getType());
  auto structType = cast<StructType>(pointerType.getElementType());
  auto elementTypes = structType.getElementTypes();
  assert(elementTypes &&
         "TODO: build requires concrete struct type for no reason");
  Type resultEltType = (*elementTypes)[index];
  Type resultType =
      PointerType::get(resultEltType, pointerType.getAddressSpace());

  result.addOperands(container);
  result.addAttribute("index",
                      builder.getIntegerAttr(builder.getIndexType(), index));
  result.addTypes(resultType);
}

void StructGEPOp::build(OpBuilder &builder, OperationState &result,
                        Type resultType, Value container, TypedAttr index) {
  result.addOperands(container);
  result.addAttribute("index", index);
  result.addTypes(resultType);
}

//===----------------------------------------------------------------------===//
// StructGEPOp - Custom Assembly Format
//===----------------------------------------------------------------------===//

// Syntax:
//   Constant index:   kgen.struct.gep %s[0] : <struct<(i32, i64)>>
//   Parametric index: kgen.struct.gep %s[I] : <struct<(i32, i64)>> -> <i64>
//
// For constant indices with concrete struct types, result type is omitted
// (can be inferred). For parametric indices, result type must be specified.

ParseResult StructGEPOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand container;
  TypedAttr indexAttr;
  Type containerType;

  // Parse: %container '[' index ']'
  if (parser.parseOperand(container) || parser.parseLSquare() ||
      parseIndexParamValue(parser, indexAttr) || parser.parseRSquare())
    return failure();

  // Parse optional attributes
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Parse: ':' container_type
  if (parser.parseColon())
    return failure();

  containerType = PointerType::parse(parser);
  if (!containerType)
    return failure();

  // Determine result type: either inferred (for constant index) or explicit
  Type resultType;
  auto pointerType = cast<PointerType>(containerType);

  if (succeeded(parser.parseOptionalArrow())) {
    // Explicit result type: -> <element_type>
    resultType = PointerType::parse(parser);
    if (!resultType)
      return failure();
  } else {
    // Infer result type from struct and constant index
    auto indexIntAttr = dyn_cast<IntegerAttr>(indexAttr);
    auto structType = dyn_cast<StructType>(pointerType.getElementType());
    if (!indexIntAttr || !structType)
      return parser.emitError(parser.getCurrentLocation(),
                              "parametric index requires explicit result type");

    auto numElements = structType.getNumElements();
    auto elementTypes = structType.getElementTypes();
    if (!numElements || !elementTypes)
      return parser.emitError(
          parser.getCurrentLocation(),
          "parametric struct requires explicit result type");

    unsigned index = indexIntAttr.getInt();
    if (index >= *numElements)
      return parser.emitError(parser.getCurrentLocation(),
                              "struct field index out of bounds");

    resultType =
        PointerType::get((*elementTypes)[index], pointerType.getAddressSpace());
  }

  // Resolve operand and set result
  if (parser.resolveOperand(container, containerType, result.operands))
    return failure();

  result.addAttribute("index", indexAttr);
  result.addTypes(resultType);
  return success();
}

void StructGEPOp::print(OpAsmPrinter &p) {
  p << " " << getContainer() << "[";
  printIndexParamValue(p, getIndex());
  p << "]";

  // Print optional attributes (excluding index which is handled above)
  p.printOptionalAttrDict((*this)->getAttrs(), {"index"});

  // Print: ':' container_type
  p << " : ";
  getContainer().getType().print(p);

  // Print result type if it can't be inferred (parametric index or parametric
  // struct type)
  auto indexAttr = dyn_cast<IntegerAttr>(getIndex());
  auto structType =
      dyn_cast<StructType>(getContainer().getType().getElementType());
  if (!indexAttr || !structType) {
    p << " -> ";
    getType().print(p);
  }
}

//===----------------------------------------------------------------------===//
// StructLoadIndirectOp
//===----------------------------------------------------------------------===//

LogicalResult StructLoadIndirectOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location> loc, Adaptor adaptor,
    SmallVectorImpl<Type> &types) {
  auto structType = dyn_cast<StructType>(adaptor.getStructValue().getType());
  if (!structType)
    return mlir::emitError(loc.value_or(adaptor.getStructValue().getLoc()),
                           "expected !kgen.struct operand, not ")
           << adaptor.getStructValue().getType();
  // The result type is the same as the input type, but with a layer of pointers
  // stripped off. The struct type may be parametric, so we need to use the
  // ParamOperatorAttr to remove the pointer layer. Preserve `isParamPack` on
  // the result when the operand is a variadic pack (same as former
  // `kgen.pack.load`).
  auto mappedTypes = ParamOperatorAttr::get(
      POC::VariadicPtrRemoveMap, structType.getElementTypesVariadic());
  types.push_back(StructType::get(ctx, mappedTypes, /*memOnly*/ {},
                                  /*minAlign*/ {},
                                  structType.getIsParamPack()));
  return success();
}

//===----------------------------------------------------------------------===//
// VariantCreateOp
//===----------------------------------------------------------------------===//

static LogicalResult verifyVariantIndex(Operation *op, VariantType type,
                                        unsigned index) {
  if (index < type.getNumTypes())
    return success();
  return op->emitOpError("variant index ")
         << index << " is out of bounds in range [0, " << type.getNumTypes()
         << ")";
}

LogicalResult VariantCreateOp::verify() {
  if (failed(verifyVariantIndex(*this, getType(), getIndex())))
    return failure();
  Type elementType = getType().getType(getIndex());
  if (elementType == getOperand().getType())
    return success();
  return emitOpError("variant element at index ")
         << getIndex() << " expected type " << elementType
         << " but operand has type " << getOperand().getType();
}

static ParseResult parseVariantElementType(AsmParser &p, Type &type,
                                           Type variantType,
                                           IntegerAttr index) {
  unsigned i = index.getInt();
  auto variant = cast<VariantType>(variantType);
  if (i >= variant.getNumTypes()) {
    return p.emitError(p.getCurrentLocation(),
                       "variant index is out of bounds: ")
           << i;
  }
  type = variant.getType(i);
  return success();
}

static void printVariantElementType(AsmPrinter &p, Operation *op, Type type,
                                    Type variantType, IntegerAttr index) {}

//===----------------------------------------------------------------------===//
// VariantIsOp
//===----------------------------------------------------------------------===//

LogicalResult VariantIsOp::verify() {
  return verifyVariantIndex(*this, getVariant().getType(), getIndex());
}

//===----------------------------------------------------------------------===//
// VariantGetOp
//===----------------------------------------------------------------------===//

LogicalResult VariantGetOp::verify() {
  if (failed(verifyVariantIndex(*this, getVariant().getType(), getIndex())))
    return failure();
  Type elementType = getVariant().getType().getType(getIndex());
  if (elementType == getType())
    return success();
  return emitOpError("variant element at index ")
         << getIndex() << " expected type " << elementType
         << " but operand has type " << getType();
}

LogicalResult VariantGetOp::inferReturnTypes(MLIRContext *,
                                             std::optional<Location> loc,
                                             Adaptor adaptor,
                                             SmallVectorImpl<Type> &types) {
  unsigned index = adaptor.getIndex();
  auto variant = cast<VariantType>(adaptor.getVariant().getType());
  if (index >= variant.getNumTypes())
    return mlir::emitOptionalError(loc, "variant element index ", index,
                                   " is out of bounds");
  types.push_back(variant.getType(index));
  return success();
}

//===----------------------------------------------------------------------===//
// CompileOffloadOp
//===----------------------------------------------------------------------===//

static ParseResult parseCompileOffloadOp(OpAsmParser &p, TypedAttr &targetType,
                                         TypedAttr &emissionKind,
                                         TypedAttr &emissionOption,
                                         TypedAttr &emissionLinkOption,
                                         TypedAttr &func) {

  if (parseParamValue(p, targetType, KGEN::TargetType::get(p.getContext())) ||
      p.parseComma() || parseIndexParamValue(p, emissionKind) ||
      p.parseComma() || parseStringParam(p, emissionOption) || p.parseComma() ||
      parseStringParam(p, emissionLinkOption) || p.parseComma() ||
      parseTypeParamValue(p, func))
    return failure();
  return success();
}

static void printCompileOffloadOp(OpAsmPrinter &p, Operation *op,
                                  TypedAttr targetType, TypedAttr emissionKind,
                                  TypedAttr emissionOption,
                                  TypedAttr emissionLinkOption,
                                  TypedAttr func) {

  printParamValue(p, targetType, KGEN::TargetType::get(op->getContext()));
  p << ", ";
  printIndexParamValue(p, emissionKind);
  p << ", ";
  printParamValue(p, emissionOption);
  p << ", ";
  printParamValue(p, emissionLinkOption);
  p << ", ";
  printTypeParamValue(p, func);
}

/// If 'value' is defined by one or more rebinds, look through them.
Value RebindOp::strip(Value value) {
  while (auto rebind = value.getDefiningOp<RebindOp>())
    value = rebind.getInput();
  return value;
}

//===----------------------------------------------------------------------===//
// ConformanceOp
//===----------------------------------------------------------------------===//

ParseResult ConformanceOp::parse(OpAsmParser &parser, OperationState &result) {
  TraitSymbolAttr traitSymbol;
  if (parseTraitSymbol(parser, traitSymbol))
    return failure();
  result.addAttribute(getTraitSymbolAttrName(result.name), traitSymbol);

  // Parse the body region.
  Region *body = result.addRegion();
  if (parser.parseRegion(*body))
    return failure();

  // Ensure the region has at least one block (required by SizedRegion<1>).
  if (body->empty())
    body->emplaceBlock();

  // Parse optional "where" clause for conditional conformance.
  // Use a trivially true constraint when no "where" clause is present.
  ConstraintAttr constraint;
  if (succeeded(parser.parseOptionalKeyword("where"))) {
    if (parser.parseAttribute(constraint))
      return failure();
  } else {
    constraint = getUnconditionalConstraint(parser.getContext());
  }
  result.addAttribute(getConstraintAttrName(result.name), constraint);

  // Parse the optional attribute dictionary.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  return success();
}

void ConformanceOp::print(OpAsmPrinter &p) {
  p << " ";
  printTraitSymbol(p, getTraitSymbol());
  p << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);

  // Print "where <constraint>" only for conditional conformance constraints.
  // Trivially true constraints (unconditional conformance) are not printed.
  if (ConstraintAttr constraint = getConstraint();
      !isTriviallyTrueConstraint(constraint)) {
    p << " where ";
    p.printAttribute(constraint);
  }

  // Print the attribute dictionary, excluding the constraint and trait symbol
  // since they're handled specially above.
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(), {getTraitSymbolAttrName(), getConstraintAttrName()});
}

///===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Mojo/KGENDialect/KGEN.cpp.inc"
