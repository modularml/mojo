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
// This file contains core logic to parameterized generators into concrete
// function implementations.
//
//===----------------------------------------------------------------------===//

#include "ParametricElaborator.h"
#include "ElaboratorHelper.h"
#include "ParametricIREvaluator.h"

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/ForkJoin.h"
#include "Mojo/HLCFDialect/HLCFDialect.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Mojo/Support/NameMangling.h"
#include "Mojo/ToolCommon/CLOptions.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Mojo/TransformUtils/ManglingUtils.h"
#include "Support/Compiler/DiagnosticHandler.h"
#include "Support/Compiler/Error.h"
#include "Support/Compiler/ErrorTree.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/StringExtras.h"

using namespace M;
using namespace KGEN;
using namespace AsyncRT;

/// Short living attribute that is needed to set on KGEN::FuncOp or
/// KGEN::DeclareRegeionOp. This attribute will be converted to LLVMetadata
/// after concretization and will be removed from the operation, therefore won't
/// survive Elaborator.
static constexpr StringRef kLLVMMetadataArrayAttrName =
    "kgen.elaborator.llvm_metadata_array";
static constexpr StringRef kLLVMArgMetadataArrayAttrName =
    "kgen.elaborator.llvm_arg_metadata_array";

//===----------------------------------------------------------------------===//
// ExpansionGraph
//===----------------------------------------------------------------------===//

PImplNode::PImplNode(PParamNode *parent)
    : ImplNodeBase(parent->gen.getBodyRegion()), parent(parent) {}

[[maybe_unused]] static std::mutex &getGlobalMutex() {
  static std::mutex mutex;
  return mutex;
}

void PImplNode::setToError(ErrorTree &&err) {
  if (error) {
#ifndef MODULAR_PRODUCTION
    std::lock_guard<std::mutex> guard(getGlobalMutex());
    llvm::errs() << "INTERNAL ELABORATOR ERROR PROCESSING: ";
    if (parent && parent->gen)
      llvm::errs() << parent->getMangledName().strref();
    else if (inst)
      llvm::errs() << inst.getName();
    else
      llvm::errs() << "[ROOT NODE]";
    llvm::errs() << "\n";
    ErrorLimit errorLimit{this->getEvaluator().getErrorLimit(), 0};

    emitLimitedError(
        [&] {
          return std::move(*error).emit(
              [](Location loc) { return mlir::emitError(loc); }, "HERE",
              this->getEvaluator().getElabErrorIncludePrelude());
        },
        errorLimit);
#endif // MODULAR_PRODUCTION
    llvm_unreachable("impl node already has an error");
  }
  hasError.store(true);
  error = std::move(err);
}

void PParamNode::emplace() {
  if (done.exchange(DoneState::DONE) == DoneState::NOT_DONE)
    paramCh.copy().emplace();
}

void PParamNode::setToError() {
  if (done.exchange(DoneState::ERROR) == DoneState::NOT_DONE)
    paramCh.copy().emplace();
}

void PParamNode::andThenAsync(AsyncValue::Waiter &&waiter) {
  expansionGraph->didAddTask();
  paramCh.andThenAsync([waiter = std::move(waiter), this]() mutable {
    waiter();
    expansionGraph->didCompleteTask();
  });
}

ParametricExpansionGraph::~ParametricExpansionGraph() {
  if (--numOutstandingResources == 0) {
    quiesceChain.copy().emplace();
    return;
  }
  // If we have outstanding tasks at destruction time, set all outstanding
  // tasks to the error state and await completion.
  for (auto &[key, node] : nodes.get())
    node->setToError();
  AsyncRT::await(quiesceChain);
}

void ParametricExpansionGraph::didCompleteTask() {
  if (--numOutstandingResources == 0)
    quiesceChain.copy().emplace();
}

void ParametricExpansionGraph::didAddTask() { ++numOutstandingResources; }

ErrorTreeOr<PImplNode *> PParamNode::getFirstConcreteNode() {
  if (!impl.error)
    return &impl;
  return ErrorTree(gen.getLoc(), "function instantiation failed 2",
                   impl.error->copy());
}

ErrorTreeOr<FuncOp> PParamNode::getFirstConcreteFunc() {
  ErrorTreeOr<PImplNode *> impl = getFirstConcreteNode();
  if (impl.isError())
    return impl.takeError();
  FuncOp func = dyn_cast<FuncOp>(*impl.takeValue()->inst);
  assert(func && "concrete instance not a FuncOp");
  return func;
}

ErrorTreeOrSuccess PParamNode::collectErrorsOrSuccess() {
  if (!impl.error)
    return success();
  return ErrorTree(gen.getLoc(), "function instantiation failed 1",
                   impl.error->copy());
}

#define HANDLE_EVALUATOR_CONC(VAR, INODE, LOC, EXPR)                           \
  do {                                                                         \
    auto exprResult =                                                          \
        (INODE)->getEvaluator().concretizeParameterExpr(INODE, LOC, EXPR);     \
    if (exprResult.isError()) {                                                \
      (INODE)->setToError(exprResult.takeError());                             \
      return ElaborationState::error();                                        \
    }                                                                          \
    if (!*exprResult)                                                          \
      return ElaborationState::skipNode();                                     \
    VAR = *exprResult;                                                         \
  } while (0)

//===----------------------------------------------------------------------===//
// processParamDeclareOp
//===----------------------------------------------------------------------===//

/// Process a param.declare op by setting its parameter value in the provided
/// evaluator.
static ElaborationState processParamDeclareOp(PImplNode *inode,
                                              ParamDeclareOp op) {
  // Simplify the input expression.
  Attribute value;
  HANDLE_EVALUATOR_CONC(value, inode, op.getLoc(), op.getValue());

  // Bind it to the parameter declaration it is setting.
  inode->getEvaluator().setDeclBinding(op.getParamDecl(),
                                       cast<TypedAttr>(value));

  // The kgen.param.declare operation serves no other purpose: remove it.
  op->erase();
  return ElaborationState::advance();
}

//===----------------------------------------------------------------------===//
// processRebindOp
//===----------------------------------------------------------------------===//

static ElaborationState processRebindOp(PImplNode *inode, RebindOp op) {
  if (!op.getInput()) {
    // FIXME MOCO-2053: This should be an error.
    // This rebind op was removed, but is still traversed due to flaw in
    // ParameterUseDefGraph's collect function.
    // Mojo/stdlib/test/cpuDevice/test_locks.mojo
    return ElaborationState::advance();
  }
  Type outType;
  HANDLE_EVALUATOR_CONC(outType, inode, op.getLoc(), op.getType());
  Type inType;
  HANDLE_EVALUATOR_CONC(inType, inode, op.getLoc(), op.getInput().getType());
  if (outType != inType) {
    inode->setToError(ErrorTree(
        op.getLoc(), "error: rebind input type '" + mlir::debugString(inType) +
                         "' does not match result type '" +
                         mlir::debugString(outType) + "'"));
    return failure();
  }
  op.replaceAllUsesWith(op.getOperand());
  op.erase();
  return ElaborationState::advance();
}

//===----------------------------------------------------------------------===//
// processParamAssertOp
//===----------------------------------------------------------------------===//

/// Process a param.assert op by folding its parameter expression and checking
/// its constraint. Returns the appropriate error if the constraint failed.
static ElaborationState processParamAssertOp(PImplNode *inode,
                                             ParamAssertOp op) {
  // Check the condition expression.
  Attribute value;
  HANDLE_EVALUATOR_CONC(value, inode, op.getLoc(), op.getCond());

  // If the constraint evaluated to false then the assert fails.
  auto resultBool = cast<SIMDAttr>(value);
  if (!resultBool.getAsBool()) {
    // Evaluate the string to report it.
    HANDLE_EVALUATOR_CONC(value, inode, op.getLoc(), op.getMessage());
    inode->setToError(
        ErrorTree(op.getLoc(),
                  "constraint failed: " + cast<StringAttr>(value).getValue()));
    return failure();
  }

  // The kgen.param.assert op serves no further purpose, so we can remove it.
  op->erase();
  return ElaborationState::advance();
}

//===----------------------------------------------------------------------===//
// processGenericOp
//===----------------------------------------------------------------------===//

/// Convert llvm metadata array attrs into dicts by treating every pair of
/// attributes in the array as (key, value) pairs, where the key is always a
/// StringAttr.
static ErrorTreeOr<DictionaryAttr>
concretizeLLVMMetadataArrays(Location loc, ArrayAttr array) {
  NamedAttrList llvmMetadata;
  DenseSet<StringAttr> seenMetadataNames;
  for (int i = 0, e = array.size(); i < e; i += 2) {
    auto name = dyn_cast<StringAttr>(array[i]);
    if (!name)
      return ErrorTree(loc, "cannot concretize name in 'llvm_metadata'");
    if (seenMetadataNames.contains(name)) {
      // NOTE: @llvm_metadata are processed and added in reverse order by the
      // parser.
      InFlightDiagnostic diag =
          mlir::emitWarning(loc, "duplicate LLVM metadata attribute for ")
          << name << ". Value of the last occurrence will be used.";
      continue;
    }
    seenMetadataNames.insert(name);
    llvmMetadata.append(name, array[i + 1]);
  }
  return llvmMetadata.getDictionary(array.getContext());
}

/// Unknown operations are allowed to use types and attributes with parameter
/// references. Substitute in concrete values for their references. Optionally
/// elaborate their locations.
static ElaborationState processGenericOp(PImplNode *parent, Operation *op) {
  // Scan all the attributes and types to look for uses of parameters.  We let
  // the walker scan the region hierarchy.
  SmallVector<NamedAttribute> newAttrs;
  bool changedAttrs = false;
  for (const NamedAttribute &namedAttr : op->getAttrs()) {
    Attribute value;
    HANDLE_EVALUATOR_CONC(value, parent, op->getLoc(), namedAttr.getValue());
    newAttrs.emplace_back(namedAttr.getName(), value);
    changedAttrs |= namedAttr.getValue() != newAttrs.back().getValue();
  }
  if (changedAttrs)
    op->setAttrs(newAttrs);

  if (auto func = dyn_cast<FuncOp>(op)) {
    if (auto llvmMetadataArray = dyn_cast_or_null<ArrayAttr>(
            func->getAttr(kLLVMMetadataArrayAttrName))) {
      ErrorTreeOr<DictionaryAttr> result =
          concretizeLLVMMetadataArrays(op->getLoc(), llvmMetadataArray);
      if (result.isError()) {
        parent->setToError(result.takeError());
        return ElaborationState::error();
      }
      func.setLLVMMetadataAttr(result.takeValue());
      func->removeAttr(kLLVMMetadataArrayAttrName);
    }
    if (auto llvmArgMetadataArray = dyn_cast_or_null<ArrayAttr>(
            func->getAttr(kLLVMArgMetadataArrayAttrName))) {
      SmallVector<Attribute> resultArray;
      for (Attribute perArgMetadataArray : llvmArgMetadataArray) {
        ErrorTreeOr<DictionaryAttr> result = concretizeLLVMMetadataArrays(
            op->getLoc(), cast<ArrayAttr>(perArgMetadataArray));
        if (result.isError()) {
          parent->setToError(result.takeError());
          return ElaborationState::error();
        }
        resultArray.push_back(result.takeValue());
      }
      func.setLLVMArgMetadataAttr(
          ArrayAttr::get(op->getContext(), resultArray));
      func->removeAttr(kLLVMArgMetadataArrayAttrName);
    }
  }

  // Check the types of results to find any parameters embedded in their
  // types.  We don't have to check operands because they are always checked
  // when being defined.
  for (OpResult result : op->getResults()) {
    Type type;
    HANDLE_EVALUATOR_CONC(type, parent, op->getLoc(), result.getType());
    result.setType(type);
  }

  // Scan the region list if present.  The walker will automatically recurse
  // for us, but we have to check the block arguments.
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Value arg : block.getArguments()) {
        Type type;
        HANDLE_EVALUATOR_CONC(type, parent, op->getLoc(), arg.getType());
        arg.setType(type);
      }
    }
  }

  // Post-rebind, a struct.gep into a flattened single-element register-passable
  // struct is an identity; fold it (as StructGEPOp::fold does) before the
  // verifier.
  if (auto gep = dyn_cast<StructGEPOp>(op)) {
    if (gep.getContainer().getType() == gep.getType()) {
      gep.replaceAllUsesWith(gep.getContainer());
      gep.erase();
      return ElaborationState::advance();
    }
  }

  return ElaborationState::advance();
}

//===----------------------------------------------------------------------===//
// collectOpsToProcess
//===----------------------------------------------------------------------===//

/// This simply walks the ParameterUseDefGraph and collects the list of ops that
/// need to be rewritten.
static void collectOpsToProcess(Region *scope, const ParameterUseDefGraph &uses,
                                std::vector<Operation *> &opsToRewrite) {
  // FIXME: The elaborator does not correctly handle the new parameter use-def
  // graph. Process the parameters in reverse: the same operation can define
  // multiple parameters, so punt those according to their most dominated
  // definition.
  opsToRewrite.reserve(opsToRewrite.size() + uses.params.size() +
                       uses.paramOps.size());
  llvm::SetVector<Operation *, SmallVector<Operation *, 8>,
                  SmallPtrSet<Operation *, 8>>
      defOps;
  for (StringAttr param : llvm::reverse(uses.params)) {
    auto it = uses.defs.find(param);
    assert(it != uses.defs.end());
    // Ignore the scope parent operation. Input parameters are set contextually.
    if (it->second.defOp == scope->getParentOp())
      continue;
    defOps.insert(it->second.defOp);
  }

  llvm::append_range(opsToRewrite, defOps.getArrayRef());
}

static void collectOpsToProcessInside(Region &toProcess, PImplNode *parent,
                                      std::vector<Operation *> &opsToRewrite) {
  auto &nestedScopes = parent->paramGraph.nestedScopes;
  auto it = nestedScopes.find(&toProcess);
  assert(it != nestedScopes.end());
  const ParameterUseDefGraph &uses = it->second;

  // Only process the ops in the branch that we ended up taking.
  for (Operation *paramOp : llvm::reverse(uses.paramOps)) {
    // Check if this op is in a region that is a child of the region we care
    // about. If not, don't process it.
    if (!toProcess.isAncestor(paramOp->getParentRegion()))
      continue;

    opsToRewrite.push_back(paramOp);
  }
  collectOpsToProcess(&toProcess, uses, opsToRewrite);
}

//===----------------------------------------------------------------------===//
// ParametricElaborator Implementation
//===----------------------------------------------------------------------===//

ParametricElaborator::ParametricElaborator(
    SymbolTable &symtab, ParameterCollector::Analysis &paramCache,
    TargetInfoAttr target, const CompilationOptions &options,
    ElaboratorCompileAsmFn compileAsmFn,
    ElaboratorCompileOffloadFn compileOffloadFn,
    const ElaborateGeneratorsOptions &config)
    : InterpreterCache(target, config.optimizeInterpreter), target(target),
      options(options), config(config), oldSymTab(symtab),
      env(symtab.getOp()->getAttrOfType<EnvAttr>(EnvAttr::getEnvAttrName())),
      cpuDevice(*loadContext(target.getContext())->get<AsyncRT::CPUDevice>()),
      g(this->cpuDevice),
      paramCache(paramCache, cpuDevice.getWorkQueue()->getParallelismLevel()),
      compileAsmFn(compileAsmFn), compileOffloadFn(compileOffloadFn) {}

//===----------------------------------------------------------------------===//
// ParametricElaborator::finalizeInstance
//===----------------------------------------------------------------------===//

void ParametricElaborator::finalizeInstance(PImplNode *node) {
  VerboseCompilerTimeTraceScope traceScope("finalizeInstance");
  // Erase everything but the entry blocks of each region.
  if (!node->inst)
    return;
  node->inst.walk<mlir::WalkOrder::PreOrder>([](Operation *op) {
    for (Region &region : op->getRegions())
      for (Block &block : llvm::make_early_inc_range(llvm::drop_begin(region)))
        block.erase();
  });
}

//===----------------------------------------------------------------------===//
// Elaborator::getConcreteFunction
//===----------------------------------------------------------------------===//

ErrorTreeOr<FuncOp>
ParametricElaborator::getConcreteFunction(PImplNode *parent, Location loc,
                                          SymbolConstantAttr symbol) {
  StringAttr name = cast<FlatSymbolRefAttr>(symbol.getSymbol()).getAttr();
  auto gen = oldSymTab.lookup<GeneratorOpInterface>(name);
  // If this doesn't reference anything in the existing module, then it must
  // refer to a concrete function in the new module.
  if (!gen) {
    return concreteNodes.read([name](auto &insts) {
      auto iter = insts.find(name);
      if (iter == insts.end())
        return FuncOp();
      return cast<FuncOp>(iter->second->inst);
    });
  }

  auto vals =
      ParameterExprArrayAttr::get(loc.getContext(), symbol.getParamValues());

  // Lookup the node if it already exists.
  PParamNode *node = getOrCreateNode(vals, gen, /*depth=*/0);
  // If the node has already been elaborated, just use that result.

  (void)specializeGenerator(parent, node, loc, /*addWaiter=*/true);

  return node->getFirstConcreteFunc();
}

ErrorTreeOr<TypeInstanceRefAttr>
ParametricElaborator::getConcreteStructTypeReference(
    PImplNode *parent, Location loc, TypeGeneratorRefAttr genref) {
  StringAttr name = cast<FlatSymbolRefAttr>(genref.getSymbol()).getAttr();
  auto gen = oldSymTab.lookup<GeneratorOpInterface>(name);
  assert(gen && "expected a valid generator reference");

  auto vals =
      ParameterExprArrayAttr::get(loc.getContext(), genref.getParamValues());
  PParamNode *calleeNode =
      getOrCreateNode(vals, gen, parent->parent->depth + 1);

  // Ensure elaboration is dispatched but return immediately. Track as an
  // eventual dependency.
  (void)specializeGenerator(parent, calleeNode, loc, /*addWaiter=*/false);

  calleeNode->emplace();

  return TypeInstanceRefAttr::get(
      SymbolRefAttr::get(loc->getContext(), calleeNode->getMangledName()),
      genref.getType());
}

ErrorTreeOr<Attribute>
ParametricElaborator::concretizeSymbolsWithin(Attribute value,
                                              PImplNode *parent, Location loc) {
  mlir::AttrTypeReplacer replacer;
  std::optional<ErrorTree> error;
  replacer.addReplacement(
      [&](SymbolConstantAttr cst) -> std::pair<Attribute, WalkResult> {
        // Ignore parametric constants.
        if (!cst.getType().getInputParamTypes().empty())
          return {cst, WalkResult::advance()};

        ErrorTreeOr<FuncOp> func = getConcreteFunction(parent, loc, cst);
        if (func.isError()) {
          error = func.takeError();
          return {cst, WalkResult::interrupt()};
        }
        if (!*func) {
          return {cst, WalkResult::interrupt()};
        }
        auto funcSym = SymbolConstantAttr::get(func.takeValue());
        auto result = parent->getEvaluator().concretizeParameterExpr(
            parent, loc, funcSym);
        if (result.isError()) {
          error = func.takeError();
          return {cst, WalkResult::interrupt()};
        }
        return {*result, WalkResult::skip()};
      });

  if (Attribute result = replacer.replace(value)) {
    return result;
  }

  if (error)
    return std::move(*error);

  return Attribute();
}

//===----------------------------------------------------------------------===//
// ParametricElaborator::addDeferredFunction
//===----------------------------------------------------------------------===//

void ParametricElaborator::addDeferredFunction(OwningOpRef<FuncOp> func) {
  FuncOp op = func.release();
  StringAttr name = op.getSymNameAttr();

  concreteNodes.modify([&op, name, this](auto &map) {
    if (addConcreteFunc(op, name, map)) {
      deferredSymbols.push_back(op);
      addRegion(op.getBodyRegion());
    } else {
      op.erase();
    }
  });
}

//===----------------------------------------------------------------------===//
// ParametricElaborator::processParamConstantOp
//===----------------------------------------------------------------------===//

template <typename OpT>
ElaborationState ParametricElaborator::processParamConstantOp(PImplNode *parent,
                                                              OpT op) {
  Attribute attr;
  HANDLE_EVALUATOR_CONC(attr, parent, op->getLoc(), op.getValue());
  auto value = cast<TypedAttr>(attr);

  // Root elaboration at the constant value and concretize any generator
  // references inside it. Multi-versioning is disallowed.
  ErrorTreeOr<Attribute> concrete =
      concretizeSymbolsWithin(value, parent, op.getLoc());

  if (concrete.isError()) {
    parent->setToError(concrete.takeError());
    return ElaborationState::error();
  }
  value = cast_or_null<TypedAttr>(concrete.takeValue());
  if (!value)
    return ElaborationState::skipNode();

  op.getResult().setType(value.getType());
  op.setValueAttr(value);
  return ElaborationState::advance();
}

//===----------------------------------------------------------------------===//
// ParametricElaborator::instantiateGeneratorReference
//===----------------------------------------------------------------------===//

std::pair<ElaborationState, PImplNode *>
ParametricElaborator::instantiateGeneratorReference(
    PImplNode *parent, Operation *user, SymbolConstantAttr calleeSymbol,
    ParameterExprArrayAttr &inputParamKey, GeneratorOpInterface &gen,
    function_ref<bool(PParamNode *)> shouldWait) {
  // Lookup the callee.
  StringAttr name = cast<FlatSymbolRefAttr>(calleeSymbol.getSymbol()).getAttr();
  Operation *calleeOp = oldSymTab.lookup(name);

  if (!calleeOp || !isa<GeneratorOpInterface>(calleeOp)) {
    PImplNode *node =
        concreteNodes.read([name](auto &map) { return map.at(name); });

    return {ElaborationState::advance(), node};
  }

  // Add in the mapping for parameters in the calls.
  inputParamKey = ParameterExprArrayAttr::get(user->getContext(),
                                              calleeSymbol.getParamValues());

  // If we already have a binding for this, we're done.
  gen = cast<GeneratorOpInterface>(calleeOp);

  // Check for excessive instantiation depth.
  if (parent->parent->depth > config.maxDepth) {
    parent->setToError(ErrorTree(parent->parent->gen.getLoc(),
                                 "elaborator expansion is " +
                                     Twine(config.maxDepth + 1) +
                                     " levels deep - infinite recursion?"));
    return {ElaborationState::error(), nullptr};
  }

  // Find the tree node that corresponds to the thing we're calling.
  PParamNode *calleeNode =
      getOrCreateNode(inputParamKey, gen, parent->parent->depth + 1);

  (void)specializeGenerator(parent, calleeNode, user->getLoc(),
                            shouldWait(calleeNode));

  return {ElaborationState::advance(), &calleeNode->impl};
}

//===----------------------------------------------------------------------===//
// ParametricElaborator::collectConcreteImplementations
//===----------------------------------------------------------------------===//

FailureOr<PImplNode *> ParametricElaborator::collectConcreteImplementations(
    Location loc, PImplNode *parent, PParamNode *calleeNode) {
  // Get all valid implementations of the callee node.
  ErrorTreeOr<PImplNode *> concrete = calleeNode->getFirstConcreteNode();
  if (concrete.isError()) {
    std::string str = printSimpleParamAttrValues(
        calleeNode->gen.getInputParams(), calleeNode->inputParams,
        options.elaborationErrorVerbose);

    if (str.empty()) {
      parent->setToError(
          ErrorTree(loc, "call expansion failed 3", concrete.takeError()));
    } else {
      parent->setToError(ErrorTree(
          loc, Twine("call expansion failed with parameter value(s): " + str),
          concrete.takeError()));
    }
    return failure();
  }

  return concrete.takeValue();
}

//===----------------------------------------------------------------------===//
// ParametricElaborator::processGeneratorUser
//===----------------------------------------------------------------------===//

ElaborationState
ParametricElaborator::processGeneratorUser(GeneratorUserOpInterface user,
                                           SymbolConstantAttr calleeSymbol,
                                           PImplNode *parent) {
  // Not all operations can verify their callee type, if for instance, it is a
  // generic type. Verify here as a fallback.
  if (!calleeSymbol.getType().getInputParamTypes().empty()) {
    parent->setToError(
        ErrorTree(user.getLoc(), "cannot reference parametric function"));
    return ElaborationState::error();
  }

  ParameterExprArrayAttr inputParamKey;
  GeneratorOpInterface gen;
  // bool isBlocking = false;
  PParamNode *calleeNode = nullptr;
  auto [result, concrete] = instantiateGeneratorReference(
      parent, user, calleeSymbol, inputParamKey, gen, [&](PParamNode *genNode) {
        calleeNode = genNode;
        return false;
      });

  if (result.isError())
    return result;

  for (auto [i, resultType] : llvm::enumerate(user->getResultTypes())) {
    Type type;
    HANDLE_EVALUATOR_CONC(type, parent, user.getLoc(), resultType);
    user->getResult(i).setType(type);
  }

  StringAttr concreteSymName;

  //  This resolved to a direct function call.
  FuncOp newCalleeFunc;
  if (!gen) {
    newCalleeFunc = dyn_cast<FuncOp>(*concrete->inst);
    assert(newCalleeFunc && "expected FuncOp as instantiated callee");

    // If this is a `kgen.param.apply`, bind its result here.
    concreteSymName = newCalleeFunc.getNameAttr();
  } else {
    concreteSymName = calleeNode->getMangledName();
  }

  // Regardless if the callee node is ready or not, we can concretize the callee
  // symbol reference immediately.
  IRRewriter b{OpBuilder(user)};
  auto newCallee =
      SymbolConstantAttr::get(concreteSymName, calleeSymbol.getType());
  user.concretizeCallee(b, newCallee);
  return ElaborationState::advance();
}

//===----------------------------------------------------------------------===//
// ParametricElaborator::processParamApplyOp
//===----------------------------------------------------------------------===//

/// Complete processing of a `kgen.param.apply` operation by invoking the
/// interpreter on the concrete callee and binding its result.
ElaborationState ParametricElaborator::processParamApplyOp(PImplNode *inode,
                                                           ParamApplyOp op) {
  Attribute calleeSymbol;
  HANDLE_EVALUATOR_CONC(calleeSymbol, inode, op.getLoc(), op.getCallee());

  StringAttr name =
      cast<FlatSymbolRefAttr>(
          extractSymbolConstantAttr(cast<TypedAttr>(calleeSymbol)).getSymbol())
          .getAttr();
  auto genItf = oldSymTab.lookup<GeneratorOpInterface>(name);
  // If this doesn't reference anything in the existing module, then it must
  // refer to a concrete function in the new module.
  FuncOp func;

  if (!genItf) {
    func = concreteNodes.read([name](auto &insts) {
      auto iter = insts.find(name);
      if (iter == insts.end())
        return FuncOp();
      return cast<FuncOp>(iter->second->inst);
    });
  }

  auto gen = cast<GeneratorOp>(genItf);

  // First concretize the operands.
  Attribute value;
  HANDLE_EVALUATOR_CONC(value, inode, op.getLoc(), op.getOperandsAttr());
  Attribute callee;
  HANDLE_EVALUATOR_CONC(callee, inode, op.getLoc(), op.getCalleeAttr());

  // Attempt to lookup a cached value. This returns a thread local cached value.
  auto operandsAttr = cast<ParameterExprArrayAttr>(value);
  auto calleeAttr = extractSymbolConstantAttr(cast<TypedAttr>(callee));

  TypedAttr cached = lookupCachedInterpretation(((!gen) ? func : gen),
                                                operandsAttr, calleeAttr);
  if (!cached) {
    // Should probably use a nestedEvaluator here???
    ParametricIREvaluator nestedEvaluator(inode->getEvaluator());
    if (gen) {
      nestedEvaluator.pushParamValues(calleeAttr.getParamValues(), true);
      // bool overwrite = false;
      for (auto [decl, attr] :
           llvm::zip(gen.getInputParams(), calleeAttr.getParamValues())) {
        nestedEvaluator.overwriteDeclBinding(decl, attr);
      }
    }

    nestedEvaluator.setErrorLoc(op.getLoc());
    ErrorTreeOr<TypedAttr> result =
        (!gen) ? nestedEvaluator.evaluateFunction(func, operandsAttr)
               : nestedEvaluator.evaluateGenerator(gen, operandsAttr);

    if (result.isError()) {
      inode->setToError(result.takeError());
      return failure();
    }

    cached = result.takeValue();
    (void)writeGlobalCachedInterpretation(((!gen) ? func : gen), operandsAttr,
                                          calleeAttr, cached);
  }

  // Bind the result and erase the operation.
  inode->getEvaluator().setDeclBinding(op.getParamDecl(), cached);
  op.erase();
  return ElaborationState::advance();
}

//===----------------------------------------------------------------------===//
// ParametricElaborator::processCallOp
//===----------------------------------------------------------------------===//

/// Process a call_param op.
ElaborationState
ParametricElaborator::processCallOp(PImplNode *parent,
                                    GeneratorUserOpInterface call) {
  Attribute symbol;
  HANDLE_EVALUATOR_CONC(symbol, parent, call.getLoc(), call.getCallee());
  return processGeneratorUser(
      call, extractSymbolConstantAttr(cast<TypedAttr>(symbol)), parent);
}

//===----------------------------------------------------------------------===//
// Locations and DebugInfo
//===----------------------------------------------------------------------===//

/// Concretizes the attribute that may contains parameters. If unsuccessful,
/// sets the ImplNode to the error state and returns null.
template <typename AttrType>
static AttrType concretizeAttr(AttrType attr, mlir::Location loc,
                               PImplNode *inode) {
  auto exprResult =
      inode->getEvaluator().concretizeParameterExpr(inode, loc, attr);
  if (exprResult.isError()) {
    inode->setToError(exprResult.takeError());
    return {};
  }
  if (LLVM_UNLIKELY(!*exprResult)) {
    // FIXME MOCO-2054: Report error after problem of compiling
    // test_layout_tensor_copy_nvidia.mojo with -debug-level=full is fixed.
    return cast<AttrType>(UnknownLoc::get(attr.getContext()));
  }
  return cast<AttrType>(*exprResult);
}

/// Concretizes the location of an op or a block argument.
template <typename ArgOrOp>
static LogicalResult concretizeLocOf(ArgOrOp &argOrOp, PImplNode *inode) {
  LocationAttr loc = argOrOp.getLoc();
  if (LocationAttr newLocAttr = concretizeAttr<LocationAttr>(loc, loc, inode)) {
    argOrOp.setLoc(newLocAttr);
    return success();
  }
  return failure();
}

static LogicalResult
concretizeLocsInScope(iterator_range<Block::iterator> scope, PImplNode *inode) {
  // Location concretization cannot yield and restart. Add a blocker to ensure
  // no blockers are set for this node while concretizing locations. Empty
  // concretization results will result in UnknownLoc.
  inode->blocker = std::make_pair(inode->inst.getLoc(), nullptr);
  for (Operation &op : scope) {
    op.walk([&](Operation *op) {
      if (failed(concretizeLocOf(*op, inode)))
        return WalkResult::interrupt();

      // Update the ValueInfo attr since they contain types.
      if (isa<DebugInfo::ValueOp, DebugInfo::KillOp>(op)) {
        op->setAttrs(
            concretizeAttr(op->getAttrDictionary(), op->getLoc(), inode));
        return WalkResult::advance();
      }

      // To be defensive, we only concretize location attributes if we know
      // what we are dealing with.
      if (auto inlined = dyn_cast<DebugInfo::InlinedSubprogramScoped>(op)) {
        if (LocationAttr callLoc = inlined.getCallLocAttr()) {
          inlined.setCallLocAttr(
              concretizeAttr<LocationAttr>(callLoc, op->getLoc(), inode));
        }
      }
      // When elaboration is complete, only the first block in any region is
      // valid (any other block may be illegal, e.g. due to how kgen.param.if
      // is handled). So we only need to go through the region arguments.
      for (Region &r : op->getRegions()) {
        for (BlockArgument arg : r.getArguments())
          if (failed(concretizeLocOf(arg, inode)))
            return WalkResult::interrupt();
      }

      // Walk over nested scopes.
      if (isa<DeclInterface>(op))
        return WalkResult::skip();

      return WalkResult::advance();
    });
  }
  inode->blocker.reset();
  return success(!inode->error);
}

/// Concretizes the locations of all operations within scope bound by the
/// specified block.
static LogicalResult concretizeLocsInScope(Block &scope, PImplNode *inode) {
  return concretizeLocsInScope({scope.begin(), scope.end()}, inode);
}

//===----------------------------------------------------------------------===//
// ParametricElaborator::processParamIfOp
//===----------------------------------------------------------------------===//

/// We always erase this op and its nested scopes from the parameter graph -
/// it's been handled, and we don't want anyone else touching it later
/// considering we're about to delete the op itself.
static void recursivelyEraseFromNestedScopes(PImplNode *node, Operation *op) {
  ParameterUseDefGraph &paramGraph = node->paramGraph;
  auto eraseScopes = [op](ParameterUseDefGraph &graph) mutable {
    // Erase any regions from the nested scopes that belong either to this op
    // or under this op.
    for (auto &[r, _] : graph.nestedScopes)
      if (op->isAncestor(r->getParentOp()))
        graph.nestedScopes.erase(r);

    // Do the same for nested decls. These two are somehow not always in sync,
    // so we have to check both separately.
    auto newEnd = llvm::remove_if(graph.nestedDecls, [&](Region *r) {
      return op->isAncestor(r->getParentOp());
    });
    graph.nestedDecls.erase(newEnd, graph.nestedDecls.end());
  };
  // Delete references to this nested declaration from all nested graphs.
  eraseScopes(paramGraph);
  for (auto &[scope, graph] : paramGraph.nestedScopes)
    eraseScopes(graph);
}

ElaborationState ParametricElaborator::processParamIfOp(PImplNode *parent,
                                                        ParamIfOp op) {
  // Check the condition expression.
  Attribute value;
  HANDLE_EVALUATOR_CONC(value, parent, op.getLoc(), op.getCond());

  // Take whichever branch the condition indicated, and simply inline those ops
  // then elaborate them. We can do this by splicing the op list into the parent
  // block. We splice it this way to avoid remapping the ops when we process
  // them later.
  bool resultBool = cast<SIMDAttr>(value).getAsBool();
  // Get the appropriate region.
  Region &toProcess = op->getRegion(!resultBool);

  // Push a new node and skip over the current frame until it completes.
  PImplNode::WorkItem item{{}, nullptr, parent->getEvaluator()};
  collectOpsToProcessInside(toProcess, parent, item.ops);

  // When the nested scope completes processing, finish processing the current
  // parameter if.
  item.onComplete = [resultBool, debug = config.elaborateDebugInfo](
                        PImplNode *node) -> LogicalResult {
    assert(node->stack.size() >= 2 && "expected at least two work items");
    // Retrieve the current state.
    PImplNode::WorkItem &parentFrame = *std::next(node->stack.rbegin());
    auto op = cast<ParamIfOp>(parentFrame.ops.back());

    // Splice the ops into the parent. Grab the terminator before the iterators
    // invalidate.
    Block::iterator iter = op->getIterator();
    Block &block = op->getRegion(!resultBool).front();

    // First update the locations if necessary
    if (debug) {
      if (failed(concretizeLocsInScope(block, node)))
        return failure();
    }

    Operation *terminator = block.getTerminator();
    op->getBlock()->getOperations().splice(iter, block.getOperations());

    // Update the values for the result parameters and do other processing
    // necessary for param.yield.
    if (auto yieldOp = dyn_cast<ParamYieldOp>(terminator)) {
      // RAUW the op's results with the terminator's inputs.
      op->getResults().replaceAllUsesWith(yieldOp.getOperands());

      // Erase the terminator.
      terminator->erase();
    } else if (auto hlcfTerm =
                   dyn_cast<HLCF::ControlFlowTerminator>(terminator)) {
      // If it's an kgen.return op, we have to split the block after the return.
      hlcfTerm->getBlock()->splitBlock(++hlcfTerm->getIterator());
      // Drop all uses of the if op because any of its uses will be null and
      // void at this point.
      op->dropAllDefinedValueUses();
    } else {
      node->setToError(ErrorTree(terminator->getLoc(),
                                 "unknown terminator kind for comptime if "
                                 "(compiler bug, please report!)"));
      return failure();
    }

    // The callback to the current frame finishes processing the current
    // operation, so take it off the parent frame's worklist.
    recursivelyEraseFromNestedScopes(node, op);
    op->erase();
    parentFrame.ops.pop_back();
    return success();
  };

  parent->stack.push_back(std::move(item));
  return ElaborationState::skipFrame();
}

//===----------------------------------------------------------------------===//
// ParametricElaborator::processParamForOp
//===----------------------------------------------------------------------===//

// Take a look at the HLCF::Loop operation generated by unrolling one iteration
// of a kgen.param.for. If the loop is pointless, we can inline it into the
// containing region to reduce control flow complexity from downstream passes.
//
// This returns true if the rest of the containing logic is dead.
static bool simplifyParamForLoop(HLCF::LoopOp loop, HLCF::LoopOp outerLoop) {
  auto isBreakFrom = [&](Operation *op, HLCF::LoopOp l) {
    auto breakOp = dyn_cast<HLCF::BreakOp>(op);
    return breakOp && breakOp.getLabelAttr() == l.getLabelAttr();
  };

  Block &body = loop.getBodyBlock();

  // Simplify the loop.
  size_t numBreaks = 0;
  body.walk([&](Operation *op) {
    if (isBreakFrom(op, loop))
      ++numBreaks;
  });

  // If there are no breaks out of this loop, but we (nonetheless) end up with
  // a break from the outer loop, then we can inline the entire body of this
  // loop and break directly out of the outer loop.  This notably happens on the
  // final iteration, but can technically happen anywhere.
  if (numBreaks == 0 &&
      isBreakFrom(outerLoop.getBodyBlock().getTerminator(), outerLoop)) {
    // Change all the loop body argument values to be the initial values.
    for (auto [bbarg, value] :
         llvm::zip(body.getArguments(), loop.getOperands()))
      bbarg.replaceAllUsesWith(value);

    // Inline the loop into the containing region, including this break.
    loop->getBlock()->getOperations().splice(Block::iterator(loop),
                                             body.getOperations());
    // Remove the loop and anything after it, it is unreachable. Do this
    // bottom-up, so defs are removed before uses.
    auto &outerBody = outerLoop.getBodyBlock();
    while (1) {
      auto &op = outerBody.back();
      bool stop = &op == loop;
      op.erase();
      if (stop) // Stop when we remove the loop itself.
        break;
    }

    // Nothing in the original block should be scanned, we removed it.
    return true;
  }

  // If there is exactly one break from the HLCF loop and if it is the
  // terminator, then we know the loop is pointless - we can inline it into the
  // containing region.
  if (numBreaks != 1 || !isBreakFrom(body.getTerminator(), loop))
    return false;

  // Change all the loop body argument values to be the initial values.
  for (auto [bbarg, value] : llvm::zip(body.getArguments(), loop.getOperands()))
    bbarg.replaceAllUsesWith(value);

  // Replace any uses of the loop result with the final break operands.
  auto theBreak = cast<HLCF::BreakOp>(body.getTerminator());
  loop->replaceAllUsesWith(theBreak.getOperands());

  // Inline the loop into the containing region, dropping the break.
  body.getTerminator()->erase();
  loop->getBlock()->getOperations().splice(Block::iterator(loop),
                                           body.getOperations());
  // Remove the loop itself.
  loop.erase();
  return false;
}

ElaborationState ParametricElaborator::processParamForOp(PImplNode *parent,
                                                         ParamForOp op) {
  // First, concretize the iterator value and the hasnext/getnext expressions.
  Attribute initial, hasNext, getNextIter;
  HANDLE_EVALUATOR_CONC(initial, parent, op.getLoc(), op.getInitial());
  HANDLE_EVALUATOR_CONC(hasNext, parent, op.getLoc(), op.getHasNext());
  HANDLE_EVALUATOR_CONC(getNextIter, parent, op.getLoc(), op.getGetNextIter());

  // Get the result types of the for loop.  These are the values passed from
  // kgen.param.for.break/continue across loop iterations and to the result of
  // the kgen.param.for.  These are created by mem2reg promoting stack objects
  // in the body of the loop.
  SmallVector<Type> resultTypes;
  for (Type type : op.getResultTypes())
    HANDLE_EVALUATOR_CONC(resultTypes.emplace_back(), parent, op.getLoc(),
                          type);

  //// Concretize the paramfor_has_next generator function.
  auto hasNextSymbol = extractSymbolConstantAttr(cast<TypedAttr>(hasNext));
  StringAttr hasNextName =
      cast<FlatSymbolRefAttr>(hasNextSymbol.getSymbol()).getAttr();
  auto hasNextGen = oldSymTab.lookup<GeneratorOp>(hasNextName);

  FuncOp hasNextFunc;
  FuncType hasNextType;
  if (!hasNextGen) {
    // Concretize the sequence generator function.
    ErrorTreeOr<FuncOp> funcOr =
        getConcreteFunction(parent, op.getLoc(), hasNextSymbol);
    if (funcOr.isError()) {
      parent->setToError(funcOr.takeError());
      return failure();
    }
    if (!*funcOr)
      return ElaborationState::skipNode();

    if (LLVM_UNLIKELY(!FuncOp(*funcOr)
                           .getFuncTypeGenerator()
                           .getBody()
                           .hasMemoryOnlyResult())) {
      parent->setToError(ErrorTree(
          op.getLoc(),
          "INTERNAL ERROR: paramfor_has_next should have register result"));
      return failure();
    }
    hasNextFunc = *funcOr;
    hasNextType = hasNextFunc.getFuncTypeGenerator().getBody();
  } else {
    hasNextType = hasNextGen.getFuncTypeGenerator().getBody();
  }

  if (hasNextType.hasMemoryOnlyResult()) {
    parent->setToError(ErrorTree(
        op.getLoc(), "INTERNAL ERROR: paramfor_has_next should return a bool"));
    return failure();
  }

  auto getNextIterSymbol =
      extractSymbolConstantAttr(cast<TypedAttr>(getNextIter));
  StringAttr getNextIterName =
      cast<FlatSymbolRefAttr>(getNextIterSymbol.getSymbol()).getAttr();
  auto getNextIterGen = oldSymTab.lookup<GeneratorOp>(getNextIterName);

  FuncOp getNextIterFunc;
  if (!getNextIterGen) {
    // Concretize the sequence generator function.
    ErrorTreeOr<FuncOp> funcOr =
        getConcreteFunction(parent, op.getLoc(), getNextIterSymbol);
    if (funcOr.isError()) {
      parent->setToError(funcOr.takeError());
      return failure();
    }
    if (!*funcOr)
      return ElaborationState::skipNode();

    if (LLVM_UNLIKELY(!FuncOp(*funcOr)
                           .getFuncTypeGenerator()
                           .getBody()
                           .hasMemoryOnlyResult())) {
      parent->setToError(
          ErrorTree(op.getLoc(),
                    "INTERNAL ERROR: __next__ should have memory-only result"));
      return failure();
    }
    getNextIterFunc = *funcOr;
  }

  // Generate the series of values.
  auto iterator = cast<TypedAttr>(initial);

  SmallVector<TypedAttr> values;
  int64_t loopUnrollCount = 0;

  while (true) {
    // We will unroll the loop N+1 times, because we have to run the body on the
    // final iterator value.
    values.push_back(iterator);
    assert(iterator.getType() == cast<TypedAttr>(initial).getType() &&
           "each iterator value should match the initial value");

    // Check to see if we are supposed to stop here.
    parent->getEvaluator().setErrorLoc(op.getLoc());

    // Check if the iterator is a memory-only type, then hasNextFunc will take
    // a pointer input.
    TypedAttr hasNextInput = iterator;
    if (hasAddress(hasNextType.getArgConvention(0)))
      hasNextInput =
          StoreToMemAttr::get(iterator, hasNextType.getArguments()[0]);

    ErrorTreeOr<TypedAttr> hasNextResult = [&]() {
      if (!hasNextGen) {
        parent->getEvaluator().pushEvalFrame(
            hasNextFunc, &hasNextFunc.getBodyRegion(), {}, 0);
        return parent->getEvaluator().evaluateFunction(hasNextFunc, iterator);
      } else {
        ParametricIREvaluator nestedEvaluator(parent->getEvaluator());
        SmallVector<TypedAttr> values;
        for (auto [decl, val] : llvm::zip(hasNextGen.getInputParams(),
                                          hasNextSymbol.getParamValues())) {
          values.push_back(parent->getEvaluator().getReboundAttribute(val));
          nestedEvaluator.overwriteDeclBinding(decl, values.back());
        }
        nestedEvaluator.pushParamValues(values, true);

        return nestedEvaluator.evaluateGenerator(hasNextGen, hasNextInput);
      }
    }();

    if (hasNextResult.isError()) {
      parent->setToError(hasNextResult.takeError());
      return failure();
    }
    if (!cast<BoolAttr>(*hasNextResult).getValue())
      break;

    // Get the next iterator value.
    iterator =
        StoreToMemAttr::get(iterator, PointerType::get(iterator.getType()));
    parent->getEvaluator().setErrorLoc(op.getLoc());

    ErrorTreeOr<TypedAttr> getNextIterResult = [&]() {
      if (!getNextIterGen) {
        parent->getEvaluator().pushEvalFrame(
            getNextIterFunc, &getNextIterFunc.getBodyRegion(), {}, 1);
        return parent->getEvaluator().evaluateFunctionWithResultSlot(
            getNextIterFunc, iterator);
      } else {
        SmallVector<TypedAttr> values;
        ParametricIREvaluator nestedEvaluator(parent->getEvaluator());
        for (auto [decl, val] : llvm::zip(getNextIterGen.getInputParams(),
                                          getNextIterSymbol.getParamValues())) {
          values.push_back(parent->getEvaluator().getReboundAttribute(val));
          nestedEvaluator.overwriteDeclBinding(decl, values.back());
        }
        nestedEvaluator.pushParamValues(values, true);

        return nestedEvaluator.evaluateGeneratorWithResultSlot(getNextIterGen,
                                                               iterator);
      }
    }();

    if (getNextIterResult.isError()) {
      parent->setToError(getNextIterResult.takeError());
      return failure();
    }

    iterator = *getNextIterResult;
    ++loopUnrollCount;
  }

  if (config.loopUnrollingWarnThreshold > 0 &&
      loopUnrollCount > config.loopUnrollingWarnThreshold) {
    InFlightDiagnostic diag = mlir::emitWarning(
        op->getLoc(), "comptime for unrolling loop more than " +
                          Twine(config.loopUnrollingWarnThreshold) +
                          " times may cause long "
                          "compilation time and large code size. (use "
                          "'--loop-unrolling-warn-threshold' to increase the "
                          "threshold or set to `0` to disable this warning)");
  }

  // The else body should be unreachable after LowerSemanticCF.
  assert(isa<UnreachableOp>(op.getElseRegion().front().front()) &&
         "LowerSemanticCF didn't lower the else block of param.for?");

  // Lower the `kgen.param.for` into an outer loop and wrapper loops for each
  // generated iteration. This way, we can lower `continue` to a break to the
  // wrapper loop to model exiting a single iteration and lower `break` to a
  // break to the outer loop to model exiting the whole loop.
  IRRewriter b{OpBuilder(op)};
  StringAttr outerLabel = b.getStringAttr("param_for_outer");
  auto outerLoop =
      HLCF::LoopOp::create(b, op.getLoc(), resultTypes, outerLabel);
  b.createBlock(&outerLoop.getBody());

  // Now generate the loop bodies and set up their elaboration at the same time.
  // Start by taking the current op off the worklist. It will be deleted by the
  // end of this function.
  parent->stack.back().ops.pop_back();

  // Add a worklist item to delete the param.for op when all the iterations are
  // processed and done and cleanup the result IR.
  PImplNode::WorkItem finalItem{{}, nullptr, parent->getEvaluator()};
  finalItem.onComplete = [op, parent,
                          outerLoop](PImplNode *node) mutable -> LogicalResult {
    // Remove the original kgen.param.for now that it has been lowered.
    recursivelyEraseFromNestedScopes(parent, op);
    op.erase();

    // Simplify each of the hlcf.loop ops in the result.
    for (auto loop : llvm::make_early_inc_range(
             outerLoop.getBody().getOps<HLCF::LoopOp>())) {
      if (simplifyParamForLoop(loop, outerLoop))
        break;
    }

    // Check to see if we can simplify the final outer loop.
    simplifyParamForLoop(outerLoop, outerLoop);

    // Each iteration of the kgen.param.for will be turned into a nested
    // hlcf.loop and many of them will be trivial (no breaks/continues out of
    // them other than the final one).  Clean up the IR to improve compile time.
    return success();
  };
  parent->stack.push_back(std::move(finalItem));

  // Upon completion of elaboration of each such generated loop, replace the
  // `kgen.param.for` terminators with the appropriate HLCF ones.
  auto makeCompletion =
      [debug = config.elaborateDebugInfo, outerLabel](
          Region &region) -> std::function<LogicalResult(PImplNode *)> {
    return [debug, &region, outerLabel](PImplNode *node) -> LogicalResult {
      if (debug) {
        if (failed(concretizeLocsInScope(region.front(), node)))
          return failure();
      }

      auto thisLoop = cast<HLCF::LoopOp>(region.getParentOp());

      // Replace the `kgen.param.for` terminators with the HLCF equivalent.
      region.walk([&](Operation *op) {
        if (isa<ParamForOp>(op))
          return WalkResult::skip();
        if (isa<ParamForBreakOp>(op)) {
          IRRewriter b{OpBuilder(op)};
          b.replaceOpWithNewOp<HLCF::BreakOp>(op, op->getOperands(),
                                              outerLabel);
          return WalkResult::advance();
        }
        if (isa<ParamForContinueOp>(op)) {
          IRRewriter b{OpBuilder(op)};
          b.replaceOpWithNewOp<HLCF::BreakOp>(op, op->getOperands(),
                                              thisLoop.getLabelAttr());
          return WalkResult::advance();
        }
        return WalkResult::advance();
      });

      return success();
    };
  };

  // Compute the ops that need to be processed in the body.
  std::vector<Operation *> opsToRewrite;
  collectOpsToProcessInside(op.getBody(), parent, opsToRewrite);
  ParamDeclAttr iterParamDecl = op.getParamDecl();

  auto replaceArgs = [](Region &body, ValueRange argValues) {
    // Replace the arguments with the results of the previous loop. Then erase
    // the arguments.
    for (auto [arg, res] : llvm::zip(body.getArguments(), argValues))
      arg.replaceAllUsesWith(res);
    body.front().eraseArguments(0, body.getNumArguments());
  };

  // Finally, stamp out all of the iterations into a HLCF loop for each.
  IRMapping mapping;
  auto &nestedScopes = parent->paramGraph.nestedScopes;
  SmallVector<DeclInterface> nestedDecls;
  op.getBody().walk([&](DeclInterface decl) { nestedDecls.push_back(decl); });
  ParametricIREvaluator evaluator = parent->getEvaluator();

  // Forward the result of one iteration into the next.
  ValueRange nextOperands = op.getOperands();
  for (TypedAttr value : values) {
    // Create the loop op for this iteration and clone the body into it.
    auto loop = HLCF::LoopOp::create(b, op.getLoc(), resultTypes);
    mapping.clear();
    op.getBody().cloneInto(&loop.getBody(), mapping);
    replaceArgs(loop.getBody(), nextOperands);
    nextOperands = loop.getResults();

    // Map the ops to rewrite from the original body into the clone one.
    PImplNode::WorkItem nextItem{{}, makeCompletion(loop.getBody()), evaluator};

    for (Operation *op : opsToRewrite)
      nextItem.ops.push_back(mapping.lookup(op));

    // If any DeclInterface got cloned, we also have to make sure to clone its
    // parameter use-def list.
    for (DeclInterface nestedDecl : nestedDecls) {
      Operation *cloned = mapping.lookup(nestedDecl);
      for (auto [declRegion, clonedRegion] :
           llvm::zip(nestedDecl->getRegions(), cloned->getRegions()))
        nestedScopes.try_emplace(&clonedRegion,
                                 nestedScopes.at(&declRegion).copy(mapping));
    }

    // Now schedule the work item for this body, binding this iterator value
    // to the loop decl parameter. Concretize the decl's type through the
    // current parameter bindings before comparing: the raw decl type can
    // carry parametric references (e.g. pack expansions, or
    // `#kgen.param_list.tabulate` inside `Tuple<...>`) that reduce to the
    // concretized form the iterator values carry only after substitution.
    Type concreteIterType;
    HANDLE_EVALUATOR_CONC(concreteIterType, parent, op.getLoc(),
                          iterParamDecl.getType());
    assert(concreteIterType == value.getType() &&
           "iterator value type should match the loop decl type");
    nextItem.evaluator.setDeclBinding(iterParamDecl, value);
    parent->stack.push_back(std::move(nextItem));
  }

  HLCF::BreakOp::create(b, op.getLoc(), nextOperands, outerLabel);
  op.replaceAllUsesWith(outerLoop.getResults());
  return ElaborationState::skipFrame();
}

//===----------------------------------------------------------------------===//
// ParametricElaborator::processScope
//===----------------------------------------------------------------------===//

void ParametricElaborator::completeImplNodeProcessing(PImplNode *inode) {
  PParamNode *p = inode->parent;
  // This waiter was triggered in an error scenario. No further action is needed
  // because we are destroying the tree.
  if (p->getIsError())
    return;
  // If the node resulted in an error or all outstanding dependencies are
  // done, complete node processing. Otherwise, if the node has an error state,
  // it could end up completing early. Avoid double-completion by using a flag.
  //
  // NOTE: This is one of the two spots where an ImplNode may be accessed in
  // parallel. Synchronize the error state check using an atomic. Any data race
  // here is benign but this makes TSAN happy.
  bool hasError = inode->hasError.load();
  if ((!hasError && (--inode->numDependencies != 0)) ||
      inode->done.exchange(true)) {
    signalWorklist();
    return;
  }

  if (!hasError) {
    // If this node is part of an SCC, we need to wait for the chain to
    // complete. We know we're the only thread in here due to the atomic. When
    // we reset `done` to false, it's possible an error state will cause another
    // thread to enter, but that should be okay.
    if (inode->sccCh) {
      inode->numDependencies = 1;
      inode->done = false;
      std::move(inode->sccCh).emplace();
      return;
    }

    // Complete processing of outstanding dependencies. Process in reverse with
    // `pop_back` so that forks will end up in the same state.
    while (!inode->dependencies.empty()) {
      auto [loc, genNode] = inode->dependencies.back();
      inode->dependencies.pop_back();

      // Check for errors in dependencies.
      FailureOr<PImplNode *> concrete =
          collectConcreteImplementations(loc, inode, genNode);
      if (failed(concrete))
        break;
    }

    // Check intra-SCC edges that were removed to break the cycle. Any callee
    // that errored during the first pass needs to propagate its error here.
    for (auto [loc, genNode] : inode->sccRemovedDeps) {
      if (inode->error)
        break;
      if (failed(collectConcreteImplementations(loc, inode, genNode)))
        break;
    }
    inode->sccRemovedDeps.clear();

    if (!inode->error)
      finalizeInstance(inode);
  }

  // If this is the last implementation node for its parent parameter node to
  // complete, then the parameter node is done.
  g.numWorkItems.fetch_add(p->state.markDone());
  p->emplace();
  signalWorklist();
}

void ParametricElaborator::processImplNodeTask(PImplNode *inode) {
  // Process the node. If processing the node got pre-empted, then return. It
  // will get scheduled again later.
  if (succeeded(processImplNode(inode))) {
    g.numWorkItems.fetch_add(1);
    completeImplNodeProcessing(inode);
  }
  signalWorklist();
}

void ParametricElaborator::scheduleImplNode(PImplNode *inode) {
  cpuDevice.getWorkQueue()->addTask(
      [inode, this] { processImplNodeTask(inode); });
}

LogicalResult ParametricElaborator::processImplNode(PImplNode *inode) {
  if (!inode->inst) {
    // Begin specialization of the parameter node. Immediately suspend
    // execution by returning `failure`.
    (void)specializeGenerator(inode, inode->parent, inode->parent->gen.getLoc(),
                              /*addWaiter=*/true);
  }

  if (inode->stack.empty())
    return success();

  VerboseCompilerTimeTraceScope traceScope(
      "processImplNode", [inode] { return inode->inst.getName().str(); });

  while (!inode->stack.empty()) {
    PImplNode::WorkItem &item = inode->stack.back();
    [[maybe_unused]] size_t size = inode->stack.size();
    ElaborationState result = processScope(inode, item);
    if (result.isError()) {
      // Interrupt indicates a fatal error.
      assert(inode->error && "node processing interrupted but no error set");
      return success();
    }
    if (result.shouldSkipFrame()) {
      // Skip indicates we need to move to another frame first.
      assert(inode->stack.size() > size && "skip with no new frame");
      continue;
    }
    if (result.shouldSkipNode()) {
      // Node skip indicates to suspend elaboration of the current function
      // and come back later.
      return failure();
    }
    // Advance indicates the current work item's operation list was exhausted.
    assert(inode->stack.size() == size && "new frame with no skip");
    assert(item.ops.empty() && "advance did not exhaust worklist");
    if (failed(item.onComplete(inode))) {
      assert(inode->error && "callback failed but no error set");
      return success();
    }
    inode->stack.pop_back();
  }
  assert(!inode->error && "unexpected error");
  return success();
}

ElaborationState ParametricElaborator::processScope(PImplNode *node,
                                                    PImplNode::WorkItem &item) {
  VerboseCompilerTimeTraceScope traceScope("processScope", [&item]() {
    return std::to_string(item.ops.size()) + " ops";
  });

  // Processing an op may generate more stuff, or even delete the op being
  // processed.
  while (!item.ops.empty()) {
    Operation *op = item.ops.back();
    ElaborationState result = processOp(node, op);
    if (result.isError() || result.shouldSkipFrame() || result.shouldSkipNode())
      return result;
    item.ops.pop_back();
  }

  return ElaborationState::advance();
}

ElaborationState ParametricElaborator::processOp(PImplNode *node,
                                                 Operation *op) {
  if (Block *block = op->getBlock())
    if (!block->isEntryBlock())
      return ElaborationState::advance();

  // ParamDeclareOp declaring a new parameter, in param space
  // ParamConstantOp cpuDevice constant value
  // ParamMaterializeOp same as ParamConstantOp, cpuDevice constant? value
  // RebindOp  check during elab, no-op
  // ParamAssertOp: similar to RebindOp, just do checks
  // ParamIfOp:
  // ParamForOp: nested regions. unwrap things, no interpreting the body
  // GeneratorUserOpInterface:
  //  ParamApply
  //  CallOp
  //  ...
  // These all result in callee get schedule, but don't need to be block for
  // concretizing anymore. DeferredOp: the op that can only be created during
  // elaboration
  //
  if (auto declare = dyn_cast<ParamDeclareOp>(op))
    return processParamDeclareOp(node, declare);
  if (auto constant = dyn_cast<ParamConstantOp>(op))
    return processParamConstantOp(node, constant);
  if (auto constant = dyn_cast<ParamMaterializeOp>(op))
    return processParamConstantOp(node, constant);
  if (auto rebindOp = dyn_cast<RebindOp>(op))
    return processRebindOp(node, rebindOp);
  if (auto assertOp = dyn_cast<ParamAssertOp>(op))
    return processParamAssertOp(node, assertOp);
  if (auto ifOp = dyn_cast<ParamIfOp>(op))
    return processParamIfOp(node, ifOp);
  if (auto forOp = dyn_cast<ParamForOp>(op))
    return processParamForOp(node, forOp);
  if (auto apply = dyn_cast<ParamApplyOp>(op))
    return processParamApplyOp(node, apply);
  if (auto call = dyn_cast<GeneratorUserOpInterface>(op))
    return processCallOp(node, call);
  if (auto compileOffload = dyn_cast<CompileOffloadOp>(op))
    return processCompileOffload(node, compileOffload);
  if (auto deferred = dyn_cast<DeferredOp>(op))
    return processDeferredOp(node, deferred);
  if (auto codegenReachable = dyn_cast<CodeGenReachableOp>(op))
    return processCodeGenReachableOp(node, codegenReachable);

  // Delay elaboration of the DILocalVariableAttr until when locations are
  // elaborated.
  if (isa<DebugInfo::ValueOp, DebugInfo::KillOp>(op))
    return ElaborationState::advance();

  // NOTE: We only need to elaborate locations manually for generic ops if we
  // don't do it globally.
  return processGenericOp(node, op);
}

//===----------------------------------------------------------------------===//
// ParametricElaborator::specializeGenerator
//===----------------------------------------------------------------------===//

PParamNode *ParametricElaborator::getOrCreateNode(ParameterExprArrayAttr values,
                                                  GeneratorOpInterface gen,
                                                  size_t depth) {
  // TODO: Split this into `get` and `create` methods, so that some can be
  // read-only accesses.
  PParamNode *paramNode = g.nodes.modify([&](auto &map) {
    std::unique_ptr<PParamNode> &n = map[{values, gen}];
    if (!n)
      n = std::make_unique<PParamNode>(cpuDevice, gen, values, depth, &g);
    return n.get();
  });
  // Add the node to the concrete nodes map regardless of whether it was
  // created or not. This guarantees that both nodes (PParamNode and PImplNode)
  // are in the corresponding maps (g.nodes and concreteNodes) when this
  // function returns.
  StringAttr name = paramNode->getMangledName();
  PImplNode *implNode = &paramNode->impl;
  concreteNodes.modify(
      [name, implNode](auto &map) { map.try_emplace(name, implNode); });
  return paramNode;
}

ElaborationState ParametricElaborator::specializeGenerator(PImplNode *inode,
                                                           PParamNode *genNode,
                                                           Location from,
                                                           bool addWaiter) {
  llvm::sys::SmartScopedWriter<true> lock(genNode->mutex);
  auto state = genNode->state.markInProgress();
  switch (state) {
  case ParamNodeState::DONE:
    return ElaborationState::advance();
  case ParamNodeState::IN_PROGRESS:
    // If the worker hit a parameter node that is already in progress, this
    // could mean two things:
    //
    // 1. The parameter node is being handled by another worker.
    // 2. A generator recursively calls into the same instantiation of itself.
    //
    // The first case is impossible in single-threaded, DFS traversal of the
    // expansion graph, because the elaborator will process generator
    // instantiations as soon as they are encountered.
    //
    // In that situation, the elaborator assumes the recursive generator
    // instantiation will have at most one successful candidate. This is valid
    // because:
    //
    // 1. If there is more than one, the total number of candidates is infinity
    //    due to recursion.
    // 2. If there are zero successful candidates, then elaboration of the rest
    //    of the function will fail anyways, and the error will be propagated
    //    up.
    //
    // However, the elaborator does not know will candidate will succeed, so it
    // must defer the processing of the recursive call to the end of the
    // worklist. The elaborator also places the restriction that recursive calls
    // cannot have result parameters. Although the following is well-formed:
    //
    // ```mlir
    // kgen.generator @foo<() -> x>() {
    //   kgen.call @foo<() -> y>()
    //   %0 = kgen.param.constant = <y>
    //   kgen.param.result_bind<2>
    //   kgen.return
    // }
    // ```
    //
    // It will be rejected as forbidden, because analyzing which operations to
    // defer would be too complex, and it could result in recursively deferring
    // operations if, for example, another recursive call would depend on `y`.
    //
    // In multi-threaded execution, call resolution is also deferred as late as
    // possible. This maximizes parallelism on the expansion graph (without
    // intra-node parallelism) while correctly handling recursion.

    return ElaborationState::skipNode();
  default:
    break;
  }

  GeneratorOpInterface gen = genNode->gen;

  ArrayRef<TypedAttr> inputParamValues = genNode->inputParams.getValue();
  ArrayRef<ParamDeclAttr> inputParamDecls = gen.getInputParams();

  VerboseCompilerTimeTraceScope traceScope(
      "specializeGenerator", [name = gen.getName()] { return name.str(); });

  // TODO (low prio): Some day we could mangle "instantiated from here"
  // information into the location.
  OpBuilder b(gen.getContext());
  StringAttr mangledName = genNode->getMangledName();

  // Whether the body of the generator needs to be instantiated too. If false,
  // regions of the generator will not be carried over to the specialized
  // instance.
  bool instantiateBody;
  InstantiatedOpInterface instance;
  if (auto generatorOp = dyn_cast<GeneratorOp>(*gen)) {
    instance = cast<InstantiatedOpInterface>(*FuncOp::create(
        b, gen.getLoc(), mangledName,
        FuncType::get(
            generatorOp.getFunctionType(),
            generatorOp.getFuncTypeGenerator().getBody().getArgConventions(),
            generatorOp.getFuncTypeGenerator().getBody().getFnEffects()),
        generatorOp.getInlineLevel(), generatorOp.getExportKind(),
        generatorOp.getExternal(), /*convergent=*/false,
        generatorOp.getLinkageNameAttr(), generatorOp.getDecorators(),
        DictionaryAttr::get(b.getContext())));
    // Process LLVM metadata recorded in the generator by fusing names and
    // values from the LLVMetadataName and LLVMMetadataValue dictionaries.
    auto newFunc = cast<FuncOp>(*instance);
    if (!generatorOp.getLLVMMetadataArray().empty()) {
      newFunc->setAttr(kLLVMMetadataArrayAttrName,
                       generatorOp.getLLVMMetadataArray());
    }
    if (!generatorOp.getLLVMArgMetadataArray().empty()) {
      newFunc->setAttr(kLLVMArgMetadataArrayAttrName,
                       generatorOp.getLLVMArgMetadataArray());
    }
    instantiateBody = true;
  } else {
    auto structGenOp = dyn_cast<StructGeneratorOp>(*gen);
    instance = cast<InstantiatedOpInterface>(*StructInstanceOp::create(
        b, gen.getLoc(), mangledName, /*sym_visibility=*/nullptr,
        structGenOp.getValueDomainType(), structGenOp.getMetaType()));
    instantiateBody = false;
  }

  ParameterUseDefGraph childGraph(instance.getBodyRegion());
  std::vector<Operation *> opsToRewrite;
  if (instantiateBody) {
    // Get a partial ordering of parameter definitions and uses that are listed
    // "top down" in our evaluation order, if we don't have one already. This
    // should happen exactly once for each  node. This will be tricky to
    // parallelize as-is - we should change the approach a bit to have a
    // ParametricNode (or similar) that doesn't store the input parameters, in
    // which we could store the ParameterUseDefGraph.
    FunctionParameterUseDefGraph *genNodeGraph = knownGraphs.read(
        [gen](const auto &map) -> FunctionParameterUseDefGraph * {
          if (auto it = map.find(gen); it != map.end())
            return it->second.get();
          return nullptr;
        });
    if (!genNodeGraph) {
      // Compute a new graph. The computed graph could end up getting discarded
      // if two threads end up here at the same time for the same generator.
      auto newGraph =
          std::make_unique<FunctionParameterUseDefGraph>(gen.getBodyRegion());
      newGraph->calculate(paramCache.getThreadLocalCache());

      // Make sure to use whichever graph ended up in the map.
      genNodeGraph = knownGraphs.modify([gen, newGraph = std::move(newGraph)](
                                            auto &map) mutable {
        return map.try_emplace(gen, std::move(newGraph)).first->second.get();
      });
    }

    // Clone the body of the generator into the function.
    // TODO: is there a nice way for us to avoid cloning this?
    IRMapping map;
    gen.getBodyRegion().cloneInto(&instance.getBodyRegion(), map);

    // Map from the generator to the new function for the parameter graph copy.
    map.map(gen.getOperation(), instance.getOperation());
    // Copy over the parameter use-def graph for this clone.
    childGraph = genNodeGraph->copy(map);

    // Collect the operations to rewrite from this function.
    llvm::append_range(opsToRewrite, llvm::reverse(childGraph.paramOps));
    opsToRewrite.push_back(instance);
    collectOpsToProcess(&instance.getBodyRegion(), childGraph, opsToRewrite);
  } else {
    // If body instantiation is not needed, the childGraph should just contain
    // the instance op itself as the only op to process.
    instance.getBodyRegion().push_back(new Block());
    childGraph.paramOps.push_back(instance);
    opsToRewrite.push_back(instance);
  }

  addRegion(instance.getBodyRegion());

  PImplNode *newFuncNode = &genNode->impl;
  assert(!newFuncNode->fromLoc && "from location is already set.");
  newFuncNode->fromLoc = from;

  newFuncNode->initialize(instance, std::move(childGraph));

  // Since the symbol will have a new name, we need to update the linkage name
  // in the subprogram information (if any).
  if (auto newFunc = dyn_cast<FuncOp>(*instance)) {
    if (auto scope = newFunc.getSubprogramScope()) {
      SmallVector<StringAttr> paramValues;
      for (TypedAttr value : inputParamValues) {
        std::string result;
        llvm::raw_string_ostream os(result);
        prettyPrintParameter(value, os);
        paramValues.push_back(b.getStringAttr(result));
      }
      DebugInfo::SourceNameAttr sourceName = scope.getSourceName();
      sourceName = DebugInfo::SourceNameAttr::get(
          sourceName.getName(), sourceName.getParamTypes(),
          sourceName.getArgTypes(), paramValues, sourceName.getParent(),
          sourceName.getKind(), sourceName.getDecorators());
      StringRef linkageName = newFunc.getSymName();
      if (inputParamValues.empty())
        linkageName.consume_back("_concrete");
      DebugInfo::updateSubprogram(newFunc, b.getStringAttr(linkageName),
                                  sourceName);
    }
  }

  std::function<LogicalResult(PImplNode *)> onComplete;
  if (config.elaborateDebugInfo) {
    // We need to recursively elaborate locations within nested regions, both on
    // ops and block arguments. We do this after the worklist is processed, to
    // ensure that all parameter computation is completed, e.g. we have
    // processed all kgen.param.decl ops.
    onComplete = [](PImplNode *inode) -> LogicalResult {
      if (failed(concretizeLocOf(*inode->inst, inode)))
        return failure();
      if (failed(concretizeLocsInScope(inode->inst.getBodyRegion().front(),
                                       inode)))
        return failure();
      return success();
    };
  } else {
    onComplete = [](PImplNode *) { return success(); };
  }

  ParametricIREvaluator evaluator(*this, newFuncNode);
  for (auto [decl, val] : llvm::zip(inputParamDecls, inputParamValues))
    evaluator.overwriteDeclBinding(decl, val);
  evaluator.pushParamValues(inputParamValues, true);

  PImplNode::WorkItem item{std::move(opsToRewrite), std::move(onComplete),
                           std::move(evaluator)};
  newFuncNode->stack.push_back(std::move(item));

  g.numWorkItems.fetch_add(1);

  scheduleImplNode(newFuncNode);
  return ElaborationState::advance();
}

//===----------------------------------------------------------------------===//
// ParametricElaborator::bundleOffloadModules
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess ParametricElaborator::bundleOffloadModules() {
  return sortAndBundleOffloadOps(compileOffloadOps.get(), oldSymTab,
                                 [&](CompileOffloadOp op, StringAttr name) {
                                   return bundleCompileOffloadOp(op, name);
                                 });
}

//===----------------------------------------------------------------------===//
// ParametricElaborator::bundleCompileOffloadOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess
ParametricElaborator::bundleCompileOffloadOp(CompileOffloadOp op,
                                             StringAttr name) {

  TargetInfoAttr target =
      cast<TargetParamAttr>(op.getTargetTypeAttr()).getTarget();
  EmitAs emissionKind = cast<EmitAsAttr>(op.getEmissionKindAttr()).getValue();

  StringRef emissionOptionsStr =
      cast<StringAttr>(op.getEmissionOptionAttr()).getValue();

  StringRef emissionLinkOptionsStr =
      cast<StringAttr>(op.getEmissionLinkOptionAttr()).getValue();

  ErrorTreeOr<OffloadFunc> offloadFuncOr = extractOffloadFunc(op, oldSymTab);
  if (offloadFuncOr.isError())
    return offloadFuncOr.takeError();
  auto [symbol, func] = offloadFuncOr.takeValue();

  // Construct the expected result type.
  MLIRContext *ctx = op.getContext();
  Builder b(ctx);
  auto noneType = KGEN::NoneType::get(ctx);
  auto populateFnType = FuncTypeGeneratorType::get(
      {}, b.getFunctionType(PointerType::get(noneType), noneType),
      {ArgConvention::ImmReg}, FnEffects().setCapturing());

  // Specialize the generator with another target by slicing it and its
  // transitive dependencies out of the IR and re-invoking the elaborator. If it
  // turns out that the specialization has more than one implementation, then
  // the elaborator invocation will fail due to multiple implementations of a
  // primary generator, and the functor will return an error.

  targetOffloadInfos.modify([&](auto &info) {
    std::string groupKey =
        emissionOptionsStr.str() + emissionLinkOptionsStr.str();
    OffloadInfo::Group &offloadInfo = info[target].groups[groupKey];
    offloadInfo.emissionOptions = emissionOptionsStr;
    offloadInfo.emissionLinkOptions = emissionLinkOptionsStr;

    // Slice out a pre-elaboration module for the new target to compile for.
    ExportMap &exportedSymbols = offloadInfo.exportedSymbols;
    exportedSymbols.insert_or_assign(func.getSymNameAttr(),
                                     ExportKind::Exported);

    // Make sure to slice out anything referenced in the input parameters. When
    // generator references are instantiated in the standalone module, they are
    // instantiated with the new target.
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement(
        [&](SymbolConstantAttr ref)
            -> std::optional<std::pair<Attribute, WalkResult>> {
          if (ref != symbol)
            exportedSymbols.insert(
                {ref.getSymbol().getRootReference(), ExportKind::NotExported});
          return std::nullopt;
        });
    replacer.addReplacement(
        [&](TypeGeneratorRefAttr ref)
            -> std::optional<std::pair<Attribute, WalkResult>> {
          exportedSymbols.insert(
              {ref.getSymbol().getRootReference(), ExportKind::NotExported});
          return std::nullopt;
        });
    replacer.addReplacement(
        [&](TypeInstanceRefAttr ref)
            -> std::optional<std::pair<Attribute, WalkResult>> {
          // Upgrade the instance reference to a generator reference so that we
          // may slice out the struct generator. We do not support slicing out
          // instances yet (see WEASOOM for more details).
          PImplNode *impl = lookupImplNode(ref.getSymbol());
          auto gen = cast<StructGeneratorOp>(impl->parent->gen);
          TypeGeneratorRefAttr genRef = TypeGeneratorRefAttr::get(
              SymbolRefAttr::get(gen.getSymNameAttr()),
              impl->parent->inputParams, gen.getMetaType());
          Attribute newGenRef = replacer.replace(genRef);
          return std::make_pair(newGenRef, WalkResult::skip());
        });
    symbol = cast<SymbolConstantAttr>(replacer.replace(symbol));

    auto iter =
        offloadInfo.symbols.insert({func, OffloadInfo::Group::SymbolInfo{}})
            .first;

    std::optional<StringAttr> sourceName;
    if (auto srcName = func.getSourceName())
      sourceName = StringAttr::get(ctx, *srcName);

    auto pair = iter->second.insert(
        {symbol,
         OffloadInfo::KernelInfo{
             name, offloadInfo.numKernels, populateFnType, {}, sourceName}});

    if (pair.second)
      offloadInfo.numKernels += 1;

    pair.first->second.emissionKinds.insert(emissionKind);

    OpBuilder b(ctx);
    op.setKernelIDAttr(b.getIndexAttr(pair.first->second.kernelId));
  });
  return success();
}

//===----------------------------------------------------------------------===//
// ParametricElaborator::processCompileOffload
//===----------------------------------------------------------------------===//

ElaborationState
ParametricElaborator::processCompileOffload(PImplNode *parent,
                                            CompileOffloadOp op) {

  SmallVector<NamedAttribute> newAttrs;
  bool changedAttrs = false;
  for (const NamedAttribute &namedAttr : op->getAttrs()) {
    Attribute value;
    HANDLE_EVALUATOR_CONC(value, parent, op->getLoc(), namedAttr.getValue());
    newAttrs.emplace_back(namedAttr.getName(), value);
    changedAttrs |= namedAttr.getValue() != newAttrs.back().getValue();
  }
  if (changedAttrs)
    op->setAttrs(newAttrs);

  ErrorTreeOr<OffloadFunc> offloadFuncOr = extractOffloadFunc(op, oldSymTab);
  if (offloadFuncOr.isError()) {
    parent->setToError(offloadFuncOr.takeError());
    return ElaborationState::error();
  }
  auto [symbol, offloadGenerator] = offloadFuncOr.takeValue();

  // If `func` is a transparent thunk, reroute the offload to its wrapped
  // function so the offload side compiles the user's kernel directly. Mutating
  // the op here keeps `bundleCompileOffloadOp` thunk-agnostic.
  if (auto thunkCallee = parent->getEvaluator().resolveTransparentThunkCallee(
          offloadGenerator, symbol, op.getLoc())) {
    if (thunkCallee->isError()) {
      parent->setToError(thunkCallee->takeError());
      return ElaborationState::error();
    }
    SymbolConstantAttr calleeSym = thunkCallee->takeValue();
    if (!calleeSym)
      return ElaborationState::skipNode();
    symbol = calleeSym;
    op.setFuncAttr(symbol);
  }

  compileOffloadOps.modify([&](auto &set) { set.insert(op); });

  return ElaborationState::advance();
}

//===----------------------------------------------------------------------===//
// processDeferredOp
//===----------------------------------------------------------------------===//

ElaborationState ParametricElaborator::processDeferredOp(PImplNode *inode,
                                                         DeferredOp op) {
  Location loc = op.getLoc();
  Attribute dict;
  HANDLE_EVALUATOR_CONC(dict, inode, loc, op.getOpAttrs());
  assert(isa<DictionaryAttr>(dict) && "expected dictionary attribute");
  Attribute propsDict;
  if (auto props = op.getOpPropertiesAttr())
    HANDLE_EVALUATOR_CONC(propsDict, inode, loc, props);

  // At this point remove all deferred attributes by replacing them with their
  // content. It's essential to do this before operation is constructed,
  // otherwise attribute may not be set if it's not concretized.
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement(
      [](DeferredAttr attr) -> std::pair<Attribute, WalkResult> {
        return {attr.getAttr(), WalkResult::advance()};
      });
  dict = replacer.replace(dict);
  if (propsDict)
    propsDict = replacer.replace(propsDict);

  // Do have to call to attr replacer again as AttrTypeReplacer does not visit
  // just replaced attribute and goes directly to its sub attributes. That
  // probably has to be fixed in upstream.
  mlir::AttrTypeReplacer concretizeAttrs;
  concretizeAttrs.addReplacement(
      [](AttrCtorDeferredAttr attr) -> std::pair<Attribute, WalkResult> {
        std::string attrString;
        llvm::raw_string_ostream os(attrString);
        for (Attribute str : attr.getStrings()) {
          if (auto strAttr = dyn_cast<StringAttr>(str)) {
            // Avoid strAttr.print as it will print quotes.
            os << strAttr.str();
          } else if (auto toStrAttr = dyn_cast<ToStringDeferredAttr>(str)) {
            Attribute val = toStrAttr.getAttr();
            bool elideType = toStrAttr.getNeedElideType() != nullptr;
            // Special case when deferred attr was evaluated to a string, but
            // user requested to omit the type. Likewise for a case above,
            // printing of that StringAttr would also print quotes that will
            // make parser fail.
            if (auto strAttr = dyn_cast<StringAttr>(val); strAttr && elideType)
              os << strAttr.str();
            else
              val.print(os, elideType);
          } else {
            llvm_unreachable("unexpected attribute type");
          }
        }
        SmallString<64> tmpBuf(attrString.begin(), attrString.end());
        tmpBuf.push_back(0);
        size_t bytesRead;
        Attribute resultAttr =
            mlir::parseAttribute(StringRef(tmpBuf).drop_back(),
                                 attr.getContext(), Type(), &bytesRead);
        if (!resultAttr)
          return {nullptr, WalkResult::interrupt()};
        return {resultAttr, WalkResult::advance()};
      });

  DiagnosticHandler handler(op.getContext());
  dict = concretizeAttrs.replace(dict);
  if (propsDict)
    propsDict = concretizeAttrs.replace(propsDict);

  if (handler.hasDiagnostics()) {
    // FIXME: Should report all errors encountered during construction of
    // attributes. Cannot do this now as PImplNode can only have one error.
    inode->setToError(
        ErrorTree(loc, "invalid MLIR attribute: " +
                           handler.getDiagnostics().back().str()));
    return failure();
  }

  OperationState state(loc, op.getOpName(), op.getOperands(),
                       op.getResultTypes());

  for (auto &attr : cast<DictionaryAttr>(dict))
    state.addAttribute(attr.getName(), attr.getValue());

  OpBuilder b(op);
  Operation *resultOp = b.create(state);

  if (propsDict) {
    if (failed(resultOp->setPropertiesFromAttribute(
            propsDict, [&]() { return resultOp->emitError(); }))) {
      inode->setToError(ErrorTree(loc, "failed to set properties on '" +
                                           op.getOpName() + "'"));
      return failure();
    }
  }

  // It's essential to elaborate result types are they're not going to be
  // elaborated later.
  for (auto [i, resultType] : llvm::enumerate(op->getResultTypes())) {
    Type type;
    HANDLE_EVALUATOR_CONC(type, inode, loc, resultType);
    resultOp->getResult(i).setType(type);
  }

  // NOTE: this operation is not verified at this stage. KGENVerifier,
  // which is invoked after Elaborator, will verify all operations
  // from dialects used by Mojo: KGEN, LIT, POP, NVVM, HLCF, ROCDL.

  DenseSet<StringAttr> inherentAttrs;
  inherentAttrs.insert_range(resultOp->getName().getAttributeNames());
  for (NamedAttribute &attr : state.attributes) {
    if (!inherentAttrs.contains(attr.getName())) {
      inode->setToError(ErrorTree(loc, "unexpected attribute '" +
                                           Twine(attr.getName().getValue()) +
                                           "' on operation"));
      return failure();
    }
  }

  op.replaceAllUsesWith(resultOp);
  op->erase();
  return ElaborationState::advance();
}

//===----------------------------------------------------------------------===//
// processDeferredOp
//===----------------------------------------------------------------------===//

ElaborationState
ParametricElaborator::processCodeGenReachableOp(PImplNode *inode,
                                                CodeGenReachableOp op) {
  // Check the condition expression.
  Attribute value;
  HANDLE_EVALUATOR_CONC(value, inode, op.getLoc(), op.getCond());

  // If the constraint evaluated to zero then we are not allowing to generate
  // cpuDevice code.
  auto resultInt = cast<IntegerAttr>(value);
  if (resultInt.getValue().isZero()) {
    // Evaluate the string to report it.
    HANDLE_EVALUATOR_CONC(value, inode, op.getLoc(), op.getMessage());
    inode->setToError(
        ErrorTree(op.getLoc(), "codegen unreachable: " +
                                   cast<StringAttr>(value).getValue()));
    return failure();
  }

  // The kgen.codegen.reachable op serves no further purpose, so we can remove
  // it.
  op->erase();
  return ElaborationState::advance();
}

//===----------------------------------------------------------------------===//
// ParametricElaborator::diagnoseAndBreakRecursion
//===----------------------------------------------------------------------===//

namespace {
/// This struct represents an edge in the partially instantiated concrete
/// callgraph in the elaborator. It is represented as a pointer to one of the
/// dependencies of a PParamNode. Note that the edge actually acts as a "node"
/// as far as `llvm::GraphTraits` is concerned. It preserves the same graph
/// properties, but this allows us to iterate over edges in graph SCCs, which is
/// what we want to do.
struct GraphEdge {
  /// In the graph edge, this PParamNode represents the caller node.
  PParamNode *pnode;
  /// This is the index into the concatenated range over
  /// `[*dependencies, blocker]` pointing to the callee PParamNode.
  size_t depIdx;

  /// This function returns the callee PParamNode by indexing into the
  /// appropriate dependency list.
  PParamNode *getPointee() const {
    auto &inode = pnode->impl;
    if (depIdx < inode.dependencies.size())
      return inode.dependencies[depIdx].second;
    return inode.blocker->second;
  }
  /// Return the location on the callee side representing where the edge
  /// originates from, to be used for diagnostic reporting.
  Location getLoc() const {
    auto &inode = pnode->impl;
    if (depIdx < inode.dependencies.size())
      return inode.dependencies[depIdx].first;
    return inode.blocker->first;
  }
  /// Return true if this edge is a blocker/interpreter edge.
  bool isBlockerEdge() const {
    auto &inode = pnode->impl;
    return depIdx >= inode.dependencies.size();
  }

  // Comparison operators for GraphTraits.
  bool operator==(const GraphEdge &rhs) const {
    return pnode == rhs.pnode && depIdx == rhs.depIdx;
  }
  bool operator!=(const GraphEdge &rhs) const { return !(*this == rhs); }

  /// Iterate over the children of the edge by iterating the dependencies of the
  /// callee node. This returns the first dependency.
  GraphEdge begin() const {
    PParamNode *next = getPointee();
    return {next, 0};
  }
  /// Iterate over the children of the edge by iterating the dependencies of the
  /// callee node. This returns the past-the-end iterator, where the index is
  /// equal to the number of dependencies.
  GraphEdge end() const {
    PParamNode *next = getPointee();
    PImplNode &inode = next->impl;
    return {next, inode.dependencies.size() + inode.blocker.has_value()};
  }

  /// GraphEdge is its own iterator.
  GraphEdge operator*() const { return *this; }

  // Increment operators required by GraphTraits.
  GraphEdge operator++() {
    ++depIdx;
    return *this;
  }
  GraphEdge operator++(int) {
    GraphEdge tmp = *this;
    ++*this;
    return tmp;
  }
};

/// This struct just wraps the root nodes and edges of the partial expansion
/// graph so we can iterate over them with GraphTraits.
struct PartialExpansionGraph {
  PartialExpansionGraph(ArrayRef<PParamNode *> roots) {
    // Gross hack to create a virtual root edge to all root generators.
    // This node has an edge to each of the root nodes.
    for (PParamNode *root : roots)
      virtualRoot.impl.dependencies.emplace_back(root->gen.getLoc(), root);

    // The base node just has an edge to the virtual root.
    baseNode.impl.dependencies.emplace_back(roots.front()->gen.getLoc(),
                                            &virtualRoot);
  }

  PParamNode virtualRoot;
  PParamNode baseNode;
};
} // namespace

namespace llvm {
template <>
struct DenseMapInfo<GraphEdge> {
  static unsigned getHashValue(GraphEdge node) {
    return DenseMapInfo<std::pair<PParamNode *, size_t>>::getHashValue(
        {node.pnode, node.depIdx});
  }
  static bool isEqual(GraphEdge lhs, GraphEdge rhs) { return lhs == rhs; }
};

template <>
struct GraphTraits<PartialExpansionGraph> {
  using NodeRef = GraphEdge;
  using ChildIteratorType = GraphEdge;

  static NodeRef getEntryNode(const PartialExpansionGraph &g) {
    return {const_cast<PParamNode *>(&g.baseNode), 0};
  }

  static ChildIteratorType child_begin(NodeRef node) { return node.begin(); }
  static ChildIteratorType child_end(NodeRef node) { return node.end(); }
};
} // namespace llvm

/// Build an error stack showing the recursion path that cannot be resolved.
static ErrorTree buildRecursionError(GraphEdge offending,
                                     ArrayRef<GraphEdge> edges,
                                     const DenseSet<GraphEdge> &inSCC) {
  SmallVector<GraphEdge> path;
  llvm::SmallDenseSet<GraphEdge, 4> edgesInPath;
  GraphEdge nextEdge = offending;

  // Find a path in the SCC that loops from `offending` back to itself.
  while (edgesInPath.insert(nextEdge).second) {
    GraphEdge it = nextEdge.begin();
    while (!inSCC.contains(*it)) {
      ++it;
      assert(it != nextEdge.end());
    }
    path.push_back(it);
    nextEdge = *it;
  }

  // Use the path to construct a stack of errors showing the user the path.
  ErrorTree err(offending.getLoc(), "function instantiation in parameter "
                                    "domain that recursively requires itself");
  ErrorTree *stack = &err;
  for (GraphEdge edge : path) {
    const char *diag = "recursively instantiated through here";
    if (path.size() == 1)
      diag = "function recursively calls itself in the parameter domain";
    else if (edge == offending)
      diag = "back to parameter domain function call here";

    stack->addCause({edge.getLoc(), diag});
    stack = &stack->getCauses().back();
  }
  return err;
}

bool ParametricElaborator::diagnoseAndBreakRecursion(
    unsigned generation, ArrayRef<PParamNode *> roots) {
  PartialExpansionGraph graph(roots);

  // Re-used data structures to reduce memory pressure.
  DenseSet<GraphEdge> inSCC;
  std::vector<AnyAsyncValueRef> sccChains;
  llvm::SetVector<PParamNode *> sccNodes; // this one gets moved

  // These are the nodes we are going to reschedule at the end.
  std::vector<PImplNode *> reschedule;

  // Early increment since we will modify the graph as we go.
  for (auto sccIt = llvm::scc_begin(graph); !sccIt.isAtEnd();) {
    if (!sccIt.hasCycle()) {
      ++sccIt;
      continue;
    }
    std::vector<GraphEdge> scc = *sccIt;
    ++sccIt;

    // First build a set of edges in the SCC for convenient lookup.
    inSCC.clear();
    sccChains.clear();
    std::optional<GraphEdge> badEdge;
    for (GraphEdge edge : scc) {
      inSCC.insert(edge);
      sccNodes.insert(edge.pnode);
      // Check if we have an invalid edge in the SCC.
      if (edge.isBlockerEdge())
        badEdge = edge;
    }
    // If we found an invalid edge, diagnose and set an error. Mark the node as
    // completed with an error.
    if (badEdge) {
      PImplNode *inode = &badEdge->pnode->impl;
      inode->setToError(buildRecursionError(*badEdge, scc, inSCC));
      inode->stack.clear();
      reschedule.push_back(inode);
      break;
    }

    // Now, we break all the edges in the SCC for each node in the SCC.
    for (PParamNode *node : sccNodes) {
      PImplNode *inode = &node->impl;
      std::vector<std::pair<Location, PParamNode *>> newDeps;
      std::vector<std::pair<Location, PParamNode *>> sccDeps;
      for (auto [idx, dep] : llvm::enumerate(inode->dependencies)) {
        if (!inSCC.contains(GraphEdge{node, idx})) {
          newDeps.push_back(dep);
        } else {
          inode->sccRemovedDeps.push_back(dep);
        }
      }

      // Decrement the number of dependencies and set the new dependencies.
      inode->numDependencies -=
          (inode->dependencies.size() - newDeps.size() - 1);
      inode->dependencies = std::move(newDeps);
      inode->sccCh = AsyncValueRef<Chain>::allocate(cpuDevice);
      sccChains.push_back(inode->sccCh.copy());
      reschedule.push_back(inode);
    }

    // When all of them are done as individual nodes, they will reset their
    // dependency counter to 1 and wait for all chains to complete. Then
    // sccRemovedDeps on each node is checked to propagate any errors from
    // callees that errored during the first pass.
    AsyncRT::andThenAsyncMoving(sccChains,
                                [this, nodes = sccNodes.takeVector()](
                                    MutableArrayRef<AnyAsyncValueRef>) {
                                  for (PParamNode *node : nodes)
                                    completeImplNodeProcessing(&node->impl);
                                });
  }

  // Now reschedule the nodes outside the loop to avoid races.
  for (PImplNode *inode : reschedule) {
    g.numWorkItems.fetch_add(1);
    scheduleImplNode(inode);
  }
  return !reschedule.empty();
}

//===----------------------------------------------------------------------===//
// ParametricElaborator::run
//===----------------------------------------------------------------------===//

static WalkResult rewriteCompileOffloadOp(
    CompileOffloadOp op, Location loc,
    DenseMap<TargetInfoAttr,
             llvm::DenseMap<std::string,
                            DenseMap<uint64_t, OffloadCompilationResult>>>
        &compiledOffload,
    bool &failed, ErrorLimit &errorLimit, bool errorIncludePrelude) {
  // Plug offload compilation results as strings back to the elaborated IR.
  auto kernelId = cast<IntegerAttr>(op.getKernelIDAttr()).getInt();
  EmitAs emissionKind = cast<EmitAsAttr>(op.getEmissionKindAttr()).getValue();
  TargetInfoAttr target =
      cast<TargetParamAttr>(op.getTargetTypeAttr()).getTarget();
  StringRef emissionOptionsStr =
      cast<StringAttr>(op.getEmissionOptionAttr()).getValue();
  StringRef emissionLinkOptionsStr =
      cast<StringAttr>(op.getEmissionLinkOptionAttr()).getValue();
  std::string groupKey =
      emissionOptionsStr.str() + emissionLinkOptionsStr.str();

  auto targetIter = compiledOffload.find(target);
  if (targetIter == compiledOffload.end()) {
    ErrorTree compileOffloadError(loc, "compile offload result missing target");

    emitLimitedError(
        [&] {
          return std::move(compileOffloadError)
              .emit([](Location loc) { return mlir::emitError(loc); },
                    "Compile offload failed.", errorIncludePrelude);
        },
        errorLimit);
    failed = true;
    return WalkResult::interrupt();
  }

  auto iter0 = targetIter->second.find(groupKey);
  if (iter0 == targetIter->second.end()) {
    ErrorTree compileOffloadError(
        loc,
        "compile offload result missing emissionOptions \"" + groupKey + "\"");
    emitLimitedError(
        [&] {
          return std::move(compileOffloadError)
              .emit([](Location loc) { return mlir::emitError(loc); },
                    "Compile offload failed.", errorIncludePrelude);
        },
        errorLimit);
    failed = true;
    return WalkResult::interrupt();
  }

  auto iter = iter0->second.find(kernelId);
  if (iter == iter0->second.end()) {
    ErrorTree compileOffloadError(loc,
                                  "compile offload result missing kernelId " +
                                      std::to_string(kernelId));
    emitLimitedError(
        [&] {
          return std::move(compileOffloadError)
              .emit([](Location loc) { return mlir::emitError(loc); },
                    "Compile offload failed.", errorIncludePrelude);
        },
        errorLimit);
    failed = true;
    return WalkResult::interrupt();
  }

  ImplicitLocOpBuilder b(op.getLoc(), OpBuilder(op));
  StringAttr content = iter->second.contents[emissionKind];
  StringAttr moduleName = iter->second.moduleNames[emissionKind];
  IntegerAttr numCaptures = iter->second.numCaptures;
  mlir::DenseI64ArrayAttr captureSizes = iter->second.captureSizes;
  PointerType nonePtr = PointerType::get(KGEN::NoneType::get(op->getContext()));
  auto structType = StructType::get(op->getContext(), {
                                                          content.getType(),
                                                          moduleName.getType(),
                                                          numCaptures.getType(),
                                                          nonePtr,
                                                      });

  SmallVector<Value> values;
  auto constantV = ParamConstantOp::create(b, content);
  auto moduleNameV = ParamConstantOp::create(b, moduleName);
  auto numCapturesV = ParamConstantOp::create(b, numCaptures);
  // Allocate an array numCaptures elements of int64 integers to store capture
  // sizes
  auto captureSizesV = POP::StackAllocationOp::create(
      b, PointerType::get(captureSizes.getElementType()), numCaptures.getInt());
  for (auto [i, typeSize] : llvm::enumerate(captureSizes.asArrayRef())) {
    Value gep = POP::OffsetOp::create(
        b, captureSizesV, ParamConstantOp::create(b, b.getIndexAttr(i)));
    POP::StoreOp::create(
        b, ParamConstantOp::create(b, b.getI64IntegerAttr(typeSize)), gep);
  }
  auto opaqueCaptureSizesV =
      POP::PointerBitcastOp::create(b, nonePtr, captureSizesV);

  assert(numCaptures.getInt() == captureSizes.size() &&
         "Num captures and number of their sizes mismatch");

  values.push_back(constantV);
  values.push_back(moduleNameV);
  values.push_back(numCapturesV);
  values.push_back(opaqueCaptureSizesV);
  auto newOp = StructCreateOp::create(b, op->getLoc(), structType, values);

  op->replaceUsesOfWith(op.getResult(), newOp);
  op.replaceAllUsesWith(newOp.getResult());
  op.erase();
  return WalkResult::advance();
}

LogicalResult ParametricElaborator::run(
    ModuleOp theModule, ArrayRef<std::pair<GeneratorOp, ParameterExprArrayAttr>>
                            primaryGenerators) {
  auto l = [&](GeneratorOp gen) {
    auto newGraph =
        std::make_unique<FunctionParameterUseDefGraph>(gen.getBodyRegion());
    newGraph->calculate(paramCache.getThreadLocalCache());
    knownGraphs.modify(
        [gen, newGraph = std::move(newGraph)](auto &map) mutable {
          return map.try_emplace(gen, std::move(newGraph));
        });
  };

  mlir::parallelForEach(theModule.getContext(), theModule.getOps<GeneratorOp>(),
                        l);

  // Find any kgen.func we have already - they're already elaborated, and we do
  // not want to re-process them. Add concrete PImplNodes for each one.
  for (FuncOp func : theModule.getOps<FuncOp>()) {
    (void)addConcreteFunc(func, func.getSymNameAttr(), concreteNodes.get());
    addRegion(func.getBodyRegion());
  }

  std::vector<AnyAsyncValueRef> primaryChs;
  std::vector<std::unique_ptr<PImplNode>> rootNodes;
  std::vector<PParamNode *> primaryNodes;
  primaryChs.reserve(primaryGenerators.size());
  primaryNodes.reserve(primaryGenerators.size());

  for (auto [gen, params] : primaryGenerators) {
    // This has no input parameters, so we can create the expansion node with
    // no input parameters.
    PParamNode *genNode = getOrCreateNode(params, gen, /*depth=*/0);
    primaryNodes.push_back(genNode);

    // Create a special root node for this primary generator.
    PImplNode *root =
        rootNodes.emplace_back(std::make_unique<PImplNode>(genNode)).get();

    // Now we can begin to construct the expansion tree rooted at this
    // generator. Emit as many errors as possible.
    g.numWorkItems.fetch_add(1);
    scheduleImplNode(root);
    primaryChs.push_back(genNode->copy());
  }

  // Process all current work.
  {
    VerboseCompilerTimeTraceScope traceScope("doElaboration");
    unsigned cycleGeneration = 0;
    while (true) {
      signalWorklist();
      AsyncRT::await(g.worklistCh);
      assert(g.numWorkItems == 0);

      // Check if all primary generators are done. If so, break.
      if (llvm::all_of(primaryChs, [](auto &ch) { return ch.isReady(); }))
        break;
      g.numWorkItems = 1;

      // Re-initialize the worklist chain.
      g.worklistCh = AsyncValueRef<Chain>::allocate(cpuDevice);

      // The only other possibility is a cycle due to recursion.
      if (diagnoseAndBreakRecursion(++cycleGeneration, primaryNodes))
        continue;
      // Anything else indicates a bug/race condition.
      llvm_unreachable("no work left, no deferred search, and no recursion?");
    }
  }

  // Check for any errors and emit them. Emit as many errors as possible.
  bool failed = false;
  ErrorLimit errorLimit{.errorLimit = options.elaborationErrorLimit,
                        .errorCount = 0};

  for (PParamNode *genNode : primaryNodes) {
    ErrorTreeOrSuccess err = genNode->collectErrorsOrSuccess();
    if (err.isError()) {
      failed = true;
      emitLimitedError(
          [&] {
            return err.takeError().emit(
                [](Location loc) { return mlir::emitError(loc); },
                "call expansion failed 1",
                options.elaborationErrorIncludePrelude);
          },
          errorLimit);
    }
  }

  for (PImplNode *node : llvm::make_second_range(concreteNodes.get())) {
    if (!node->parent)
      continue;
    ErrorTreeOrSuccess err = node->parent->collectErrorsOrSuccess();
    if (err.isError()) {
      failed = true;
      emitLimitedError(
          [&]() {
            return err.takeError().emit(
                [](Location loc) { return mlir::emitError(loc); },
                "call expansion failed 2",
                options.elaborationErrorIncludePrelude);
          },
          errorLimit);
    }
  }

  if (failed) {
    for (PImplNode *node : llvm::make_second_range(concreteNodes.get()))
      node->inst.erase();

    return failure();
  }

  ErrorTreeOrSuccess bundleOr = bundleOffloadModules();

  if (bundleOr.isError()) {

    emitLimitedError(
        [&]() {
          return bundleOr.takeError().emit(
              [](Location loc) { return mlir::emitError(loc); },
              "Bundle CompileOffload failed.",
              options.elaborationErrorIncludePrelude);
        },
        errorLimit);

    for (PImplNode *node : llvm::make_second_range(concreteNodes.get()))
      node->inst.erase();
    return failure();
  }

  // Compile the offload functions here.
  MLIRContext *ctx = theModule.getContext();
  DiagnosticHandler handler(ctx, /*capturePerThread=*/false);
  ElaboratorCompileOffloadRetType compiledOffloadOr = compileOffloadFn(
      theModule, targetOffloadInfos.get(), oldSymTab, options, getOptions());
  // Release the handler from MLIRContext
  handler.release();
  if (compiledOffloadOr.isError()) {
    ErrorTree compileOffloadError(theModule->getLoc(),
                                  compiledOffloadOr.takeError());
    emitLimitedError(
        [&] {
          return std::move(compileOffloadError)
              .emit([](Location loc) { return mlir::emitError(loc); },
                    "Compile offload failed.",
                    options.elaborationErrorIncludePrelude);
        },
        errorLimit);
    for (PImplNode *node : llvm::make_second_range(concreteNodes.get()))
      node->inst.erase();

    handler.emitDiagnostics([&](Diagnostic &diag) {
      // Emit diagnostics using another diagnostic handler that should be set.
      theModule->getContext()->getDiagEngine().emit(std::move(diag));
    });
    return failure();
  }

  DenseMap<
      TargetInfoAttr,
      llvm::DenseMap<std::string, DenseMap<uint64_t, OffloadCompilationResult>>>
      compiledOffload = compiledOffloadOr.takeValue();

  // The fill step runs after renameFunctions below, anchored to pre-rename syms
  // so that @__name renaming does not break the host-stub lookup.

  // Cleanup pass - we want to remove generators and interfaces by replacing
  // them with their concrete implementations. Only handle the primary
  // generators - everything else we don't care about.
  // Sort instantiations of each generator to ensure we have a deterministic
  // output in multithreaded execution.
  struct SuccessfulInstances {
    std::string paramStr;
    InstantiatedOpInterface inst;
  };
  auto *newBlock = new Block;
  llvm::MapVector<GeneratorOpInterface, std::vector<SuccessfulInstances>>
      genInstantiations;
  for (Operation &op : llvm::make_early_inc_range(*theModule.getBody())) {
    if (auto gen = dyn_cast<GeneratorOpInterface>(op)) {
      genInstantiations[gen];
    } else {
      op.remove();
      newBlock->push_back(&op);
    }
  }
  for (PParamNode &node :
       llvm::make_pointee_range(llvm::make_second_range(g.nodes.get()))) {
    VerboseCompilerTimeTraceScope traceScope(
        "processGen", [name = node.gen.getName()] { return name.str(); });
    // Erase all erroneous instances.
    if (node.impl.error) {
      node.impl.inst.erase();
      continue;
    }

    genInstantiations[node.gen].push_back(SuccessfulInstances{
        mlir::debugString(node.inputParams), node.impl.inst});
  }

  // Now reorder all instantiations of each generator to be deterministic.
  for (auto &[gen, instantiations] : genInstantiations) {
    llvm::sort(instantiations, [](auto &lhs, auto &rhs) {
      return lhs.paramStr < rhs.paramStr;
    });
    for (auto &[_, func] : instantiations)
      newBlock->push_back(func);
  }

  // Sort and then push on all the deferred functions.
  llvm::sort(deferredSymbols, [](FuncOp lhs, FuncOp rhs) {
    return lhs.getSymName() < rhs.getSymName();
  });
  for (FuncOp func : deferredSymbols)
    newBlock->push_back(func);

  // Update the symbol table with the new one.
  theModule.getBody()->erase();
  theModule.getBodyRegion().push_back(newBlock);

  renameFunctions(theModule, target.isGPU(), failed);

  // Fill populate_captures stubs with the body generated during offload
  // compilation.
  //
  // @__name kernels are renamed by renameFunctions above, so this fill step
  // runs after renaming. Both sides of the name lookup are deliberately
  // anchored to the pre-rename auto-mangled sym to survive that rename:
  //
  //   - The host stub was created in evaluateCompileOffloadClosureAttr using
  //     the pre-rename sym (mangleParameterValues of the generator).
  //   - writeCaptureArgs (in KGENCompiler) names the offload-side populate
  //     function using the same mangled kernel.name.
  //
  // So populate.getSymName() here is the pre-rename sym, and lookupSymbolIn
  // finds the host stub by that same name.
  for (auto &[fillTarget, fillResult] : compiledOffload) {
    for (auto &[_, fillGroup] : fillResult) {
      for (auto &[_, kernel] : fillGroup) {
        auto populate = cast<FuncOp>(kernel.func.get());
        StringRef populateName = populate.getSymName();
        StringAttr lookupName =
            StringAttr::get(theModule.getContext(), populateName);
        // A host stub only exists for kernels that had a
        // compile_offload_closure declared on the host side. Kernels compiled
        // only for their info (e.g. via compile_offload without a matching
        // closure) have no stub; skip.
        Operation *stubOp =
            mlir::SymbolTable::lookupSymbolIn(theModule, lookupName);
        if (!stubOp)
          continue;
        auto stubFunc = cast<FuncOp>(stubOp);
        // Fill in the actual body of the populate closure which is
        // generated while compiling all the offload functions.
        stubFunc.getBodyRegion().takeBody(
            cast<FuncOp>(*kernel.func).getBodyRegion());
      }
    }
  }

  theModule.walk([&](Operation *op) {
    if (auto offloadOp = dyn_cast<CompileOffloadOp>(op)) {
      // Plug offload compilation results as strings back to the elaborated IR.
      rewriteCompileOffloadOp(offloadOp, theModule.getLoc(), compiledOffload,
                              failed, errorLimit,
                              options.elaborationErrorIncludePrelude);
    } else if (auto isCompileTime =
                   dyn_cast<IsRunInComptimeInterpreterOp>(op)) {
      // Rewrite IsCompileTimeOp to cpuDevice value as always false.
      OpBuilder b(op);
      isCompileTime->replaceAllUsesWith(ParamConstantOp::create(
          b, op->getLoc(), SIMDAttr::getScalarBool(b.getContext(), false)));
      op->erase();
    }
  });

  if (failed)
    return failure();

  // Recompute the new symbol table.
  oldSymTab = SymbolTable(theModule);
  return success();
}
