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

#include "Elaborator.h"
#include "ElaboratorHelper.h"
#include "IREvaluator.h"
#include "ParametricElaborator.h"

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
#include "Mojo/ToolCommon/MLIROpFString.h"
#include "Mojo/TransformUtils/EliminateDeadSymbolUtils.h"
#include "Mojo/TransformUtils/ManglingUtils.h"
#include "Support/Compiler/DiagnosticHandler.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/ScopeExit.h"
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
// InterpreterCache
//===----------------------------------------------------------------------===//

ErrorTreeOr<const FunctionIRBytecode *>
ConcreteFunction::CompiledRegion::compileIfNecessary(Region &region,
                                                     TargetInfoAttr target,
                                                     bool optimize) {
  // Try to minimize writer contention by checking quickly if the region is
  // already compiled.
  if (compiled)
    return &*bytecode.get();

  // Let in only one thread at a time.
  using Result = ErrorTreeOr<const FunctionIRBytecode *>;
  return bytecode.modify([&](auto &bc) -> Result {
    // If another thread got in here first, just exit.
    if (compiled)
      return &*bc;

    auto func = cast<FuncOp>(region.getParentOp());
    CompilerTimeTraceScope traceScope(
        "compileBytecode", [func] { return FuncOp(func).getSymName().str(); });

    // If the compilation mode is -O0, quickly optimize the function.
    if (optimize) {
      clone = func.clone();
      func = *clone;
      mlir::PassManager mgr(clone->getContext());
      mgr.enableVerifier(false);
      mgr.addPass(createCanonicalizer());
      mgr.addPass(createSROA());
      mgr.addPass(createMem2Reg());
      mgr.addPass(createCanonicalizer());
      mgr.addPass(mlir::createCSEPass());
      (void)mgr.run(*clone);
    }

    ErrorTreeOr<FunctionIRBytecode> result =
        FunctionIRBytecode::compile(func.getBodyRegion(), target);
    if (result.isError())
      return result.takeError();
    bc.emplace(result.takeValue());
    compiled = true;
    return &*bc;
  });
}

//===----------------------------------------------------------------------===//
// ExpansionGraph
//===----------------------------------------------------------------------===//

ImplNode::ImplNode(ParamNode *parent)
    : ImplNodeBase(parent->gen.getBodyRegion()), parent(parent) {}

[[maybe_unused]] static std::mutex &getGlobalMutex() {
  static std::mutex mutex;
  return mutex;
}

void ImplNode::setToError(ErrorTree &&err) {
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

void ParamNode::emplace() {
  if (done.exchange(DoneState::DONE) == DoneState::NOT_DONE)
    paramCh.copy().emplace();
}

void ParamNode::setToError() {
  if (done.exchange(DoneState::ERROR) == DoneState::NOT_DONE) {
    state.markDone();
    paramCh.copy().emplace();
  }
}

void ParamNode::andThenAsync(AsyncValue::Waiter &&waiter) {
  expansionGraph->didAddTask();
  paramCh.andThenAsync([waiter = std::move(waiter), this]() mutable {
    waiter();
    expansionGraph->didCompleteTask();
  });
}

ExpansionGraph::~ExpansionGraph() {
  auto resetState = llvm::scope_exit([&] {
    if (eraseNodeInst) {
      for (auto &[_, node] : nodes.get())
        node->impl.inst.erase();
    }
  });

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

void ExpansionGraph::didCompleteTask() {
  if (--numOutstandingResources == 0)
    quiesceChain.copy().emplace();
}

void ExpansionGraph::didAddTask() { ++numOutstandingResources; }

ErrorTreeOr<ImplNode *> ParamNode::getFirstConcreteNode() {
  if (!impl.error)
    return &impl;
  return ErrorTree(gen.getLoc(), "function instantiation failed",
                   impl.error->copy());
}

ErrorTreeOr<FuncOp> ParamNode::getFirstConcreteFunc() {
  ErrorTreeOr<ImplNode *> impl = getFirstConcreteNode();
  if (impl.isError())
    return impl.takeError();
  FuncOp func = dyn_cast<FuncOp>(*impl.takeValue()->inst);
  assert(func && "concrete instance not a FuncOp");
  return func;
}

ErrorTreeOrSuccess ParamNode::collectErrorsOrSuccess() {
  if (!impl.error)
    return success();
  return ErrorTree(gen.getLoc(), "function instantiation failed",
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
static ElaborationState processParamDeclareOp(ImplNode *inode,
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

static ElaborationState processRebindOp(ImplNode *inode, RebindOp op) {
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
static ElaborationState processParamAssertOp(ImplNode *inode,
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
static ElaborationState processGenericOp(ImplNode *parent, Operation *op) {
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

static void collectOpsToProcessInside(Region &toProcess, ImplNode *parent,
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
// Elaborator Implementation
//===----------------------------------------------------------------------===//

Elaborator::Elaborator(SymbolTable &symtab,
                       ParameterCollector::Analysis &paramCache,
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
// Elaborator::finalizeInstance
//===----------------------------------------------------------------------===//

void Elaborator::finalizeInstance(ImplNode *node) {
  VerboseCompilerTimeTraceScope traceScope("finalizeInstance");
  // Erase everything but the entry blocks of each region.
  node->inst.walk<mlir::WalkOrder::PreOrder>([](Operation *op) {
    for (Region &region : op->getRegions())
      for (Block &block : llvm::make_early_inc_range(llvm::drop_begin(region)))
        block.erase();
  });
}

//===----------------------------------------------------------------------===//
// Elaborator::getConcreteFunction
//===----------------------------------------------------------------------===//

ErrorTreeOr<FuncOp> Elaborator::getConcreteFunction(ImplNode *parent,
                                                    Location loc,
                                                    SymbolConstantAttr symbol) {
  StringAttr name = cast<FlatSymbolRefAttr>(symbol.getSymbol()).getAttr();
  auto gen = oldSymTab.lookup<GeneratorOpInterface>(name);
  // If this doesn't reference anything in the existing module, then it must
  // refer to a concrete function in the new module.
  if (!gen) {
    ImplNode *implNode = concreteNodes.read([name](auto &insts) -> ImplNode * {
      auto iter = insts.find(name);
      return iter == insts.end() ? nullptr : iter->second;
    });
    if (!implNode)
      return FuncOp();
    // A generator instance is registered under its concrete name before it
    // has elaborated (or failed), so it must go through the same
    // readiness/error protocol as a call by generator name. Only an instance
    // with no expansion node is an already-concrete function.
    if (implNode->parent)
      return concretizeCallee(parent, implNode->parent, loc);
    return cast<FuncOp>(implNode->inst);
  }

  auto vals =
      ParameterExprArrayAttr::get(loc.getContext(), symbol.getParamValues());

  // Lookup the node if it already exists.
  ParamNode *node = getOrCreateNode(vals, gen, /*depth=*/0);
  return concretizeCallee(parent, node, loc);
}

ErrorTreeOr<FuncOp>
Elaborator::concretizeCallee(ImplNode *parent, ParamNode *node, Location loc) {
  // If the node has already been elaborated, just use that result.
  ElaborationState result =
      specializeGenerator(parent, node, loc, /*addWaiter=*/true);
  if (result.shouldSkipNode())
    return FuncOp();
  if (result.isError()) {
    // A failure not recorded on the node must still surface as an error,
    // never as an unfinished body.
    ErrorTreeOr<FuncOp> func = node->getFirstConcreteFunc();
    if (func.isError())
      return func;
    return ErrorTree(loc, "failed to elaborate callee function");
  }
  return node->getFirstConcreteFunc();
}

ErrorTreeOr<StructInstanceOp>
Elaborator::getConcreteStructTypeInstance(ImplNode *parent, Location loc,
                                          TypeInstanceRefAttr instref,
                                          bool addWaiter) {
  ImplNode *implNode = lookupImplNode(instref.getSymbol());
  assert(implNode && "the existence of an instance reference implies the "
                     "struct instance must have been created already");

  ElaborationState result =
      specializeGenerator(parent, implNode->parent, loc, addWaiter);
  if (result.shouldSkipNode())
    return StructInstanceOp();
  if (result.isError())
    return ErrorTree(loc, "failed to concretize struct type");
  return cast<StructInstanceOp>(implNode->inst);
}

ErrorTreeOr<TypeInstanceRefAttr>
Elaborator::getConcreteStructTypeReference(ImplNode *parent, Location loc,
                                           TypeGeneratorRefAttr genref) {
  StringAttr name = cast<FlatSymbolRefAttr>(genref.getSymbol()).getAttr();
  auto gen = oldSymTab.lookup<GeneratorOpInterface>(name);
  assert(gen && "expected a valid generator reference");
  // If this doesn't reference anything in the existing module, then it must
  // already refer to a concrete struct type in the new module.

  auto vals =
      ParameterExprArrayAttr::get(loc.getContext(), genref.getParamValues());
  ParamNode *calleeNode = getOrCreateNode(vals, gen, parent->parent->depth + 1);

  // Ensure elaboration is dispatched but return immediately. Track as an
  // eventual dependency.
  ElaborationState result =
      specializeGenerator(parent, calleeNode, loc, /*addWaiter=*/false);
  if (result.shouldSkipNode()) {
    // The callee node is not done yet, but we just need to track the dependency
    // and move on.
    assert(parent->numDependencies >= 1 && "impossible for impl to be done");
    parent->dependencies.emplace_back(loc, calleeNode);
    callInstantiationGraph.modify(
        [&](auto &map) { map[calleeNode][parent->parent].push_back(loc); });

    if (calleeNode->state.addWaiter()) {
      ++parent->numDependencies;
      calleeNode->andThenAsync(
          [this, parent] { completeImplNodeProcessing(parent); });
    }
  } else if (result.isError()) {
    return ErrorTree(gen.getLoc(), "struct type instantiation failed");
  }

  return TypeInstanceRefAttr::get(
      SymbolRefAttr::get(loc->getContext(), calleeNode->getMangledName()),
      genref.getType());
}

ErrorTreeOr<Attribute> Elaborator::concretizeSymbolsWithin(Attribute value,
                                                           ImplNode *parent,
                                                           Location loc) {
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
        if (!*func)
          return {cst, WalkResult::interrupt()};

        return {SymbolConstantAttr::get(func.takeValue()), WalkResult::skip()};
      });
  if (Attribute result = replacer.replace(value))
    return result;
  if (error)
    return std::move(*error);
  return Attribute();
}

//===----------------------------------------------------------------------===//
// Elaborator::addDeferredFunction
//===----------------------------------------------------------------------===//

void Elaborator::addDeferredFunction(OwningOpRef<FuncOp> func) {
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
// Elaborator::processParamConstantOp
//===----------------------------------------------------------------------===//

template <typename OpT>
ElaborationState Elaborator::processParamConstantOp(ImplNode *parent, OpT op) {
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
// Elaborator::instantiateGeneratorReference
//===----------------------------------------------------------------------===//

std::pair<ElaborationState, ImplNode *>
Elaborator::instantiateGeneratorReference(
    ImplNode *parent, Operation *user, SymbolConstantAttr calleeSymbol,
    ParameterExprArrayAttr &inputParamKey, GeneratorOpInterface &gen,
    function_ref<bool(ParamNode *)> shouldWait) {
  // Lookup the callee.
  StringAttr name = cast<FlatSymbolRefAttr>(calleeSymbol.getSymbol()).getAttr();
  Operation *calleeOp = oldSymTab.lookup(name);

  ParamNode *calleeNode;
  if (!calleeOp || !isa<GeneratorOpInterface>(calleeOp)) {
    ImplNode *node =
        concreteNodes.read([name](auto &map) { return map.at(name); });

    // An instance is registered under its concrete name before its expansion
    // node has elaborated, so reaching it by that name must wait on the same
    // readiness protocol as a call by generator name; otherwise its body
    // reaches the interpreter with `kgen.param.*` ops still in it. An instance
    // without an expansion node is already concrete.
    if (!node->parent)
      return {ElaborationState::advance(), node};
    calleeNode = node->parent;
  } else {
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
    calleeNode = getOrCreateNode(inputParamKey, gen, parent->parent->depth + 1);
  }
  ElaborationState result = specializeGenerator(
      parent, calleeNode, user->getLoc(), shouldWait(calleeNode));
  if (result.shouldSkipNode())
    return {ElaborationState::skipNode(), nullptr};

  FailureOr<ImplNode *> concrete =
      collectConcreteImplementations(user->getLoc(), parent, calleeNode);
  if (failed(concrete))
    return {failure(), nullptr};
  return {ElaborationState(success()), *concrete};
}

//===----------------------------------------------------------------------===//
// Elaborator::collectConcreteImplementations
//===----------------------------------------------------------------------===//

FailureOr<ImplNode *>
Elaborator::collectConcreteImplementations(Location loc, ImplNode *parent,
                                           ParamNode *calleeNode) {
  // Get all valid implementations of the callee node.
  ErrorTreeOr<ImplNode *> concrete = calleeNode->getFirstConcreteNode();
  if (concrete.isError()) {
    std::string str = printSimpleParamAttrValues(
        calleeNode->gen.getInputParams(), calleeNode->inputParams,
        options.elaborationErrorVerbose);

    if (str.empty()) {
      parent->setToError(
          ErrorTree(loc, "call expansion failed", concrete.takeError()));
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
// Elaborator::processGeneratorUser
//===----------------------------------------------------------------------===//

ElaborationState
Elaborator::processGeneratorUser(GeneratorUserOpInterface user,
                                 SymbolConstantAttr calleeSymbol,
                                 ImplNode *parent) {
  // Not all operations can verify their callee type, if for instance, it is a
  // generic type. Verify here as a fallback.
  if (!calleeSymbol.getType().getInputParamTypes().empty()) {
    parent->setToError(
        ErrorTree(user.getLoc(), "cannot reference parametric function"));
    return ElaborationState::error();
  }

  ParameterExprArrayAttr inputParamKey;
  GeneratorOpInterface gen;
  bool isBlocking = false;
  ParamNode *calleeNode;
  auto [result, concrete] = instantiateGeneratorReference(
      parent, user, calleeSymbol, inputParamKey, gen, [&](ParamNode *genNode) {
        calleeNode = genNode;
        return isBlocking = isa<ParamApplyOp>(user);
      });
  if (result.isError() || (result.shouldSkipNode() && isBlocking))
    return result;

  for (auto [i, resultType] : llvm::enumerate(user->getResultTypes())) {
    Type type;
    HANDLE_EVALUATOR_CONC(type, parent, user.getLoc(), resultType);
    user->getResult(i).setType(type);
  }

  StringAttr concreteSymName;
  if (result.shouldSkipNode()) {
    // The callee node is not done yet, but `isBlocking` is false, which means
    // we just need to track the dependency and move on.
    assert(parent->numDependencies >= 1 && "impossible for impl to be done");
    parent->dependencies.emplace_back(user->getLoc(), calleeNode);
    callInstantiationGraph.modify([&](auto &map) {
      map[calleeNode][parent->parent].push_back(user->getLoc());
    });

    if (calleeNode->state.addWaiter()) {
      ++parent->numDependencies;
      calleeNode->andThenAsync(
          [this, parent] { completeImplNodeProcessing(parent); });
    }
    concreteSymName = calleeNode->getMangledName();
  } else {
    // This resolved to a direct function call.
    FuncOp newCalleeFunc = dyn_cast<FuncOp>(*concrete->inst);
    assert(newCalleeFunc && "expected FuncOp as instantiated callee");

    // If this is a `kgen.param.apply`, bind its result here.
    if (auto apply = dyn_cast<ParamApplyOp>(*user))
      return processParamApplyOp(parent, apply, newCalleeFunc);
    concreteSymName = newCalleeFunc.getNameAttr();
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
// Elaborator::processParamApplyOp
//===----------------------------------------------------------------------===//

/// Complete processing of a `kgen.param.apply` operation by invoking the
/// interpreter on the concrete callee and binding its result.
ElaborationState Elaborator::processParamApplyOp(ImplNode *inode,
                                                 ParamApplyOp op, FuncOp func) {
  // First concretize the operands.
  Attribute value;
  HANDLE_EVALUATOR_CONC(value, inode, op.getLoc(), op.getOperandsAttr());

  // Attempt to lookup a cached value. This returns a thread local cached value.
  auto operandsAttr = cast<ParameterExprArrayAttr>(value);
  TypedAttr &cached = lookupCachedInterpretation(func, operandsAttr);
  if (!cached) {
    ErrorTreeOr<Attribute> operandsOr =
        concretizeSymbolsWithin(operandsAttr, inode, op.getLoc());
    if (operandsOr.isError()) {
      inode->setToError(operandsOr.takeError());
      return failure();
    }
    operandsAttr = cast_or_null<ParameterExprArrayAttr>(operandsOr.takeValue());
    if (!operandsAttr)
      return ElaborationState::skipNode();

    inode->getEvaluator().setErrorLoc(op.getLoc());
    ErrorTreeOr<TypedAttr> result =
        inode->getEvaluator().evaluateFunction(func, operandsAttr);
    if (result.isError()) {
      inode->setToError(result.takeError());
      return failure();
    }
    // Get the value from the global cache and use it below. This way, all
    // call sites to this function use exactly the same value.
    cached =
        writeGlobalCachedInterpretation(func, operandsAttr, result.takeValue());
  }

  // Bind the result and erase the operation.
  inode->getEvaluator().setDeclBinding(op.getParamDecl(), cached);
  op.erase();
  return ElaborationState::advance();
}

//===----------------------------------------------------------------------===//
// Elaborator::processCallOp
//===----------------------------------------------------------------------===//

/// Process a call_param op.
ElaborationState Elaborator::processCallOp(ImplNode *parent,
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
                               ImplNode *inode) {
  auto exprResult =
      inode->getEvaluator().concretizeParameterExpr(inode, loc, attr);
  if (exprResult.isError()) {
    inode->setToError(exprResult.takeError());
    return {};
  }
  if (LLVM_UNLIKELY(!*exprResult)) {
    inode->setToError(ErrorTree(
        loc, "concretized parameter expression in attribute is null"));
    return {};
  }
  return cast<AttrType>(*exprResult);
}

/// Concretizes the location of an op or a block argument.
template <typename ArgOrOp>
static LogicalResult concretizeLocOf(ArgOrOp &argOrOp, ImplNode *inode) {
  LocationAttr loc = argOrOp.getLoc();
  if (LocationAttr newLocAttr = concretizeAttr<LocationAttr>(loc, loc, inode)) {
    argOrOp.setLoc(newLocAttr);
    return success();
  }
  return failure();
}

static LogicalResult
concretizeLocsInScope(iterator_range<Block::iterator> scope, ImplNode *inode) {
  // Location concretization cannot yield and restart. Suppress blocker
  // registration for this node while concretizing locations. Empty
  // concretization results will result in UnknownLoc.
  inode->suppressBlockers = true;
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
  inode->suppressBlockers = false;
  return success(!inode->error);
}

/// Concretizes the locations of all operations within scope bound by the
/// specified block.
static LogicalResult concretizeLocsInScope(Block &scope, ImplNode *inode) {
  return concretizeLocsInScope({scope.begin(), scope.end()}, inode);
}

//===----------------------------------------------------------------------===//
// Elaborator::processParamIfOp
//===----------------------------------------------------------------------===//

/// We always erase this op and its nested scopes from the parameter graph -
/// it's been handled, and we don't want anyone else touching it later
/// considering we're about to delete the op itself.
static void recursivelyEraseFromNestedScopes(ImplNode *node, Operation *op) {
  ParameterUseDefGraph &paramGraph = node->paramGraph;

  // Compute the set of regions that live within `op` (including `op`'s own
  // regions). This is exactly { r : op->isAncestor(r->getParentOp()) }, the
  // predicate the old code tested against every scope in the graph — but
  // computed once in O(size of op's subtree).
  //
  // `ParameterUseDefGraph::calculate` flattens the graph so that the top-level
  // `nestedScopes` map holds the scopes for ALL nesting depths (see the comment
  // in `ParameterUseDefGraph::dump`: "Nested graphs are actually contained on
  // the top-level map"), and `copy` preserves that flattening when `param.for`
  // unrolling clones scopes. So every scope to erase is a key in the top-level
  // map, and erasing a key drops its whole sub-graph. We therefore only need a
  // direct, keyed erase against `regionsWithinOp` rather than scanning every
  // scope and walking each one's ancestor chain — the latter was quadratic in
  // the size of the enclosing function (every param.if/param.for resolution
  // rescanned the whole function's scope graph).
  llvm::SmallPtrSet<Region *, 8> regionsWithinOp;
  for (Region &r : op->getRegions())
    regionsWithinOp.insert(&r);
  op->walk([&](Operation *o) {
    for (Region &r : o->getRegions())
      regionsWithinOp.insert(&r);
  });

  llvm::SmallPtrSet<Region *, 8> parentScopesToPrune;
  for (Region *r : regionsWithinOp) {
    paramGraph.nestedScopes.erase(r);
    if (Region *parent = r->getParentOp()->getParentRegion())
      parentScopesToPrune.insert(parent);
  }

  auto pruneNestedDecls = [&](ParameterUseDefGraph &g) {
    auto newEnd = llvm::remove_if(
        g.nestedDecls, [&](Region *r) { return regionsWithinOp.contains(r); });
    g.nestedDecls.erase(newEnd, g.nestedDecls.end());
  };
  pruneNestedDecls(paramGraph);
  for (Region *parentScope : parentScopesToPrune) {
    if (parentScope == paramGraph.scope)
      continue;
    auto it = paramGraph.nestedScopes.find(parentScope);
    if (it != paramGraph.nestedScopes.end())
      pruneNestedDecls(it->second);
  }
}

ElaborationState Elaborator::processParamIfOp(ImplNode *parent, ParamIfOp op) {
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
  ImplNode::WorkItem item{{}, nullptr, parent->getEvaluator()};
  collectOpsToProcessInside(toProcess, parent, item.ops);

  // When the nested scope completes processing, finish processing the current
  // parameter if.
  item.onComplete = [resultBool, debug = config.elaborateDebugInfo](
                        ImplNode *node) -> LogicalResult {
    assert(node->stack.size() >= 2 && "expected at least two work items");
    // Retrieve the current state.
    ImplNode::WorkItem &parentFrame = *std::next(node->stack.rbegin());
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
// Elaborator::processParamForOp
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

ElaborationState Elaborator::processParamForOp(ImplNode *parent,
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

  // Concretize the paramfor_has_next generator function.
  auto hasNextSymbol = extractSymbolConstantAttr(cast<TypedAttr>(hasNext));
  if (!hasNextSymbol) {
    parent->setToError(ErrorTree(op.getLoc(),
                                 "INTERNAL ERROR: paramfor_has_next should "
                                 "resolve to a concrete function"));
    return failure();
  }
  ErrorTreeOr<FuncOp> hasNextFunc =
      getConcreteFunction(parent, op.getLoc(), hasNextSymbol);
  if (hasNextFunc.isError()) {
    parent->setToError(hasNextFunc.takeError());
    return failure();
  }
  if (!*hasNextFunc)
    return ElaborationState::skipNode();

  // Concretize the sequence generator function.
  auto getNextIterSymbol =
      extractSymbolConstantAttr(cast<TypedAttr>(getNextIter));
  if (!getNextIterSymbol) {
    parent->setToError(
        ErrorTree(op.getLoc(), "INTERNAL ERROR: paramfor_get_next_iter should "
                               "resolve to a concrete function"));
    return failure();
  }
  ErrorTreeOr<FuncOp> getNextIterFunc =
      getConcreteFunction(parent, op.getLoc(), getNextIterSymbol);
  if (getNextIterFunc.isError()) {
    parent->setToError(getNextIterFunc.takeError());
    return failure();
  }
  if (!*getNextIterFunc)
    return ElaborationState::skipNode();

  // has_next should return a bool.
  FuncType hasNextType = FuncOp(*hasNextFunc).getFuncTypeGenerator().getBody();
  if (hasNextType.hasMemoryOnlyResult()) {
    parent->setToError(ErrorTree(
        op.getLoc(), "INTERNAL ERROR: paramfor_has_next should return a bool"));
    return failure();
  }
  // The generator should return a well-known struct.
  if (!FuncOp(*getNextIterFunc)
           .getFuncTypeGenerator()
           .getBody()
           .hasMemoryOnlyResult()) {
    parent->setToError(
        ErrorTree(op.getLoc(),
                  "INTERNAL ERROR: iterator should have memory-only result"));
    return failure();
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

    ErrorTreeOr<TypedAttr> hasNextResult =
        parent->getEvaluator().evaluateFunction(*hasNextFunc, hasNextInput);
    if (hasNextResult.isError()) {
      parent->setToError(hasNextResult.takeError());
      return failure();
    }
    if (!cast<SIMDAttr>(*hasNextResult).getAsBool())
      break;

    // Get the next iterator value.
    iterator =
        StoreToMemAttr::get(iterator, PointerType::get(iterator.getType()));
    parent->getEvaluator().setErrorLoc(op.getLoc());
    ErrorTreeOr<TypedAttr> result =
        parent->getEvaluator().evaluateFunctionWithResultSlot(*getNextIterFunc,
                                                              iterator);
    if (result.isError()) {
      parent->setToError(result.takeError());
      return failure();
    }
    iterator = *result;
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
  ImplNode::WorkItem finalItem{{}, nullptr, parent->getEvaluator()};
  finalItem.onComplete = [op, parent,
                          outerLoop](ImplNode *node) mutable -> LogicalResult {
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
      [debug = config.elaborateDebugInfo,
       outerLabel](Region &region) -> std::function<LogicalResult(ImplNode *)> {
    return [debug, &region, outerLabel](ImplNode *node) -> LogicalResult {
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
  IREvaluator evaluator = parent->getEvaluator();

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
    ImplNode::WorkItem nextItem{{}, makeCompletion(loop.getBody()), evaluator};

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
// Elaborator::processScope
//===----------------------------------------------------------------------===//

void Elaborator::completeImplNodeProcessing(ImplNode *inode) {
  ParamNode *p = inode->parent;
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
      FailureOr<ImplNode *> concrete =
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

void Elaborator::processImplNodeTask(ImplNode *inode) {
  // Hold a resume guard for the duration of the pass so blocker completions
  // cannot re-run the node while it is still being processed.
  inode->pendingResumes.fetch_add(1);
  // Process the node. If processing the node got pre-empted, then return. It
  // will get scheduled again later.
  // If processImplNode returns success, it means the node is either fully
  // processed or in error state, no more to do here so we can complete it.
  // Otherwise, it means the node is being suspended and will be resumed later
  // once its blockers are done.
  if (succeeded(processImplNode(inode))) {
    // Only increment numWorkItems if inode is not in error state because we
    // don't put an errored-out node's dependencies back onto the worklist. This
    // also means the worklist does not keep track of error nodes. If we put it
    // onto the worklist and signal the worklist below, this can cause
    // numWorkItems become 0 multiple times and lead to g.worklistCh to be
    // emplaced multiple times which is illegal.
    if (!inode->parent->getIsError())
      g.numWorkItems.fetch_add(1);
    completeImplNodeProcessing(inode);
  }
  // Release the guard. If every registered blocker completed while this pass
  // ran, re-run the node here — no resume is left to do it. The nested
  // invocation signals the worklist once, so back it with a work item.
  if (inode->pendingResumes.fetch_sub(1) == 1 && !inode->done &&
      !inode->blockers.empty() && !inode->parent->getIsError()) {
    inode->blockers.clear();
    g.numWorkItems.fetch_add(1);
    processImplNodeTask(inode);
  }
  // Signal the worklist that the work is complete.
  if (!inode->parent->getIsError())
    signalWorklist();
}

void Elaborator::scheduleImplNode(ImplNode *inode) {
  cpuDevice.getWorkQueue()->addTask(
      [inode, this] { processImplNodeTask(inode); });
}

void Elaborator::registerBlocker(ImplNode *inode, Location from,
                                 ParamNode *genNode) {
  inode->blockers.emplace_back(from, genNode);
  inode->pendingResumes.fetch_add(1);
  genNode->andThenAsync([inode, this] {
    // Re-process only once every registered blocker has completed and no
    // processing pass is in flight (a pass holds a guard count).
    if (inode->pendingResumes.fetch_sub(1) == 1) {
      inode->blockers.clear();
      processImplNodeTask(inode);
      return;
    }
    // Another blocker completion or the in-flight pass resumes the node;
    // consume the work item added for this waiter on callee completion.
    if (!inode->parent->getIsError())
      signalWorklist();
  });
}

LogicalResult Elaborator::processImplNode(ImplNode *inode) {
  // Check for a root node.
  if (!inode->inst) {
    // Begin specialization of the parameter node. Immediately suspend
    // execution by returning `failure`.
    (void)specializeGenerator(inode, inode->parent, inode->parent->gen.getLoc(),
                              /*addWaiter=*/true);
    return failure();
  }
  if (inode->stack.empty())
    return success();

  VerboseCompilerTimeTraceScope traceScope(
      "processImplNode", [inode] { return inode->inst.getName().str(); });

  while (!inode->stack.empty()) {
    ImplNode::WorkItem &item = inode->stack.back();
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

ElaborationState Elaborator::processScope(ImplNode *node,
                                          ImplNode::WorkItem &item) {
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

ElaborationState Elaborator::processOp(ImplNode *node, Operation *op) {
  if (Block *block = op->getBlock())
    if (!block->isEntryBlock())
      return ElaborationState::advance();

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
  if (auto call = dyn_cast<GeneratorUserOpInterface>(op))
    return processCallOp(node, call);
  if (auto compileOffload = dyn_cast<CompileOffloadOp>(op))
    return processCompileOffload(node, compileOffload);
  if (auto deferred = dyn_cast<DeferredOp>(op))
    return processDeferredOp(node, deferred);

  // Delay elaboration of the DILocalVariableAttr until when locations are
  // elaborated.
  if (isa<DebugInfo::ValueOp, DebugInfo::KillOp>(op))
    return ElaborationState::advance();

  // NOTE: We only need to elaborate locations manually for generic ops if we
  // don't do it globally.
  return processGenericOp(node, op);
}

//===----------------------------------------------------------------------===//
// Elaborator::specializeGenerator
//===----------------------------------------------------------------------===//

ParamNode *Elaborator::getOrCreateNode(ParameterExprArrayAttr values,
                                       GeneratorOpInterface gen, size_t depth) {
  // TODO: Split this into `get` and `create` methods, so that some can be
  // read-only accesses.
  ParamNode *paramNode = g.nodes.modify([&](auto &map) {
    std::unique_ptr<ParamNode> &n = map[{values, gen}];
    if (!n)
      n = std::make_unique<ParamNode>(cpuDevice, gen, values, depth, &g);
    return n.get();
  });
  // Add the node to the concrete nodes map regardless of whether it was
  // created or not. This guarantees that both nodes (ParamNode and ImplNode)
  // are in the corresponding maps (g.nodes and concreteNodes) when this
  // function returns.
  StringAttr name = paramNode->getMangledName();
  ImplNode *implNode = &paramNode->impl;
  concreteNodes.modify(
      [name, implNode](auto &map) { map.try_emplace(name, implNode); });
  return paramNode;
}

ElaborationState Elaborator::specializeGenerator(ImplNode *inode,
                                                 ParamNode *genNode,
                                                 Location from,
                                                 bool addWaiter) {
  switch (genNode->state.markInProgress()) {
  case ParamNodeState::DONE:
    if (genNode->getIsError())
      return ElaborationState::error();

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
    if (addWaiter && !inode->suppressBlockers) {
      if (genNode->state.addWaiter()) {
        registerBlocker(inode, from, genNode);
        return ElaborationState::skipNode();
      }
      // Raced with node completion.
      return ElaborationState::advance();
    }
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

    ParameterUseDefGraph *genNodeGraph = knownGraphs.modify(
        [gen, &paramCache = this->paramCache](auto &map) mutable {
          if (auto it = map.find(gen); it != map.end())
            return it->second.get();
          // Compute a new graph. The computed graph could end up getting
          // discarded if two threads end up here at the same time for the same
          // generator.
          auto newGraph =
              std::make_unique<ParameterUseDefGraph>(gen.getBodyRegion());

          // calculate() modifies gen.getBodyRegion(), so this needs to be done
          // in a thread safe manner to avoid tsan failures.
          newGraph->calculate(paramCache.getThreadLocalCache());

          return map.try_emplace(gen, std::move(newGraph)).first->second.get();
        });

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

  ImplNode *newFuncNode = &genNode->impl;
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

  std::function<LogicalResult(ImplNode *)> onComplete;
  if (config.elaborateDebugInfo) {
    // We need to recursively elaborate locations within nested regions, both on
    // ops and block arguments. We do this after the worklist is processed, to
    // ensure that all parameter computation is completed, e.g. we have
    // processed all kgen.param.decl ops.
    onComplete = [](ImplNode *inode) -> LogicalResult {
      if (failed(concretizeLocOf(*inode->inst, inode)))
        return failure();
      if (failed(concretizeLocsInScope(inode->inst.getBodyRegion().front(),
                                       inode)))
        return failure();
      return success();
    };
  } else {
    onComplete = [](ImplNode *) { return success(); };
  }

  IREvaluator evaluator(*this, newFuncNode);
  for (auto [decl, val] : llvm::zip(inputParamDecls, inputParamValues))
    evaluator.setDeclBinding(decl, val);

  ImplNode::WorkItem item{std::move(opsToRewrite), std::move(onComplete),
                          std::move(evaluator)};
  newFuncNode->stack.push_back(std::move(item));

  if (addWaiter && !inode->suppressBlockers) {
    [[maybe_unused]] bool added = genNode->state.addWaiter();
    assert(added);
    registerBlocker(inode, from, genNode);
  }
  g.numWorkItems.fetch_add(1);
  scheduleImplNode(newFuncNode);
  return ElaborationState::skipNode();
}

//===----------------------------------------------------------------------===//
// Elaborator::bundleOffloadModules
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess Elaborator::bundleOffloadModules() {
  return sortAndBundleOffloadOps(compileOffloadOps.get(), oldSymTab,
                                 [&](CompileOffloadOp op, StringAttr name) {
                                   return bundleCompileOffloadOp(op, name);
                                 });
}

//===----------------------------------------------------------------------===//
// Elaborator::bundleCompileOffloadOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess Elaborator::bundleCompileOffloadOp(CompileOffloadOp op,
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
        [&](FuncSymbolAttr ref)
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
          ImplNode *impl = lookupImplNode(ref.getSymbol());
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
// Elaborator::processCompileOffload
//===----------------------------------------------------------------------===//

ElaborationState Elaborator::processCompileOffload(ImplNode *parent,
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

/// Replace every `DeferredAttr` in `a` with its wrapped attribute.
static Attribute stripDeferredAttrs(Attribute a) {
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement(
      [](DeferredAttr attr) -> std::pair<Attribute, WalkResult> {
        return {attr.getAttr(), WalkResult::advance()};
      });
  return replacer.replace(a);
}

/// Print one deferred-string fragment to `os`: a bare `StringAttr` (raw, no
/// quotes) or a `ToStringDeferredAttr` (unwrapped, honoring its elide-type
/// request, with the same no-quotes rule when the wrapped value is a string).
/// Returns false if `a` is neither, leaving the fallback to the caller.
static bool printDeferredFragment(Attribute a, llvm::raw_ostream &os) {
  if (auto strAttr = dyn_cast<StringAttr>(a)) {
    os << strAttr.str();
    return true;
  }
  if (auto toStr = dyn_cast<ToStringDeferredAttr>(a)) {
    Attribute val = toStr.getAttr();
    bool elideType = toStr.getNeedElideType() != nullptr;
    if (auto sv = dyn_cast<StringAttr>(val); sv && elideType)
      os << sv.str();
    else
      val.print(os, elideType);
    return true;
  }
  return false;
}

ElaborationState Elaborator::processDeferredOp(ImplNode *inode, DeferredOp op) {
  Location loc = op.getLoc();

  // F-string MLIR op: template lives in `opName`. Concretize result types,
  // inline any `%param<N>` placeholders, then parse via the shared helper.
  if (KGEN::isFStringTemplate(op.getOpName())) {
    SmallVector<Type> resultTypes;
    resultTypes.reserve(op->getNumResults());
    for (Type t : op->getResultTypes()) {
      Type concrete;
      HANDLE_EVALUATOR_CONC(concrete, inode, loc, t);
      resultTypes.push_back(concrete);
    }
    OpBuilder b(op);
    SmallVector<Value> operands(op.getOperands().begin(),
                                op.getOperands().end());

    // `fstring_params` carries the comptime-PValue substitutions; skip
    // entirely when no `%param<N>` marker survived (the common case).
    std::string templateStr = op.getOpName().str();
    if (templateStr.find("%param<") != std::string::npos) {
      // Strip `#kgen.deferred<…>` wrappers so we print bare values.
      Attribute opAttrsConc;
      HANDLE_EVALUATOR_CONC(opAttrsConc, inode, loc, op.getOpAttrs());
      opAttrsConc = stripDeferredAttrs(opAttrsConc);
      auto opAttrsDict = dyn_cast<DictionaryAttr>(opAttrsConc);
      auto params =
          opAttrsDict
              ? opAttrsDict.getAs<ArrayAttr>(KGEN::getFStringParamsAttrName())
              : ArrayAttr();
      if (params) {
        SmallVector<std::string> paramStrings;
        for (Attribute a : params) {
          std::string s;
          llvm::raw_string_ostream ss(s);
          if (!printDeferredFragment(a, ss))
            a.print(ss);
          paramStrings.push_back(std::move(s));
        }

        // Inline `%param<N>`; `%arg<N>` / `%type_of(arg<N>)` are left for
        // `lowerFStringMLIROp`. A malformed `%param<` means the emitter and
        // elaborator are out of sync (internal error).
        std::string paramErr;
        std::string substituted;
        llvm::raw_string_ostream sos(substituted);
        KGEN::scanFStringTemplate(
            templateStr, sos,
            [&](StringRef rest, llvm::raw_ostream &os) -> int {
              if (!rest.consume_front("param<"))
                return 0;
              size_t len = 0;
              while (len < rest.size() && llvm::isDigit(rest[len]))
                ++len;
              if (len == 0 || len >= rest.size() || rest[len] != '>') {
                paramErr = "internal: malformed `%param<...>` placeholder";
                return -1;
              }
              size_t n = 0;
              if (rest.slice(0, len).getAsInteger(10, n)) {
                paramErr =
                    "internal: failed to parse `%param` placeholder index";
                return -1;
              }
              if (n >= paramStrings.size()) {
                llvm::raw_string_ostream eos(paramErr);
                eos << "internal: `%param<" << n
                    << ">` references a parameter beyond the `fstring_params` "
                       "array (have "
                    << paramStrings.size() << " entries)";
                return -1;
              }
              os << paramStrings[n];
              return static_cast<int>(StringRef("param<").size() + len + 1);
            });
        if (!paramErr.empty()) {
          inode->setToError(ErrorTree(loc, paramErr));
          return failure();
        }
        templateStr = std::move(substituted);
      }
    }

    std::string errorMsg;
    Operation *parsedOp = KGEN::lowerFStringMLIROp(
        b, loc, templateStr, operands, resultTypes, errorMsg);
    if (!parsedOp) {
      inode->setToError(ErrorTree(loc, errorMsg));
      return failure();
    }
    op.replaceAllUsesWith(parsedOp);
    op->erase();
    return ElaborationState::advance();
  }

  Attribute dict;
  HANDLE_EVALUATOR_CONC(dict, inode, loc, op.getOpAttrs());
  assert(isa<DictionaryAttr>(dict) && "expected dictionary attribute");
  Attribute propsDict;
  if (auto props = op.getOpPropertiesAttr())
    HANDLE_EVALUATOR_CONC(propsDict, inode, loc, props);

  // At this point remove all deferred attributes by replacing them with their
  // content. It's essential to do this before operation is constructed,
  // otherwise attribute may not be set if it's not concretized.
  dict = stripDeferredAttrs(dict);
  if (propsDict)
    propsDict = stripDeferredAttrs(propsDict);

  // Do have to call to attr replacer again as AttrTypeReplacer does not visit
  // just replaced attribute and goes directly to its sub attributes. That
  // probably has to be fixed in upstream.
  mlir::AttrTypeReplacer concretizeAttrs;
  concretizeAttrs.addReplacement(
      [](AttrCtorDeferredAttr attr) -> std::pair<Attribute, WalkResult> {
        std::string attrString;
        llvm::raw_string_ostream os(attrString);
        // Print bare (no quotes); a StringAttr or elide-type string would
        // otherwise emit quotes that make the attribute parser fail.
        for (Attribute str : attr.getStrings()) {
          if (!printDeferredFragment(str, os))
            llvm_unreachable("unexpected attribute type");
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
    // attributes. Cannot do this now as ImplNode can only have one error.
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
// Elaborator::diagnoseAndBreakRecursion
//===----------------------------------------------------------------------===//

namespace {
/// This struct represents an edge in the partially instantiated concrete
/// callgraph in the elaborator. It is represented as a pointer to one of the
/// dependencies of a ParamNode. Note that the edge actually acts as a "node" as
/// far as `llvm::GraphTraits` is concerned. It preserves the same graph
/// properties, but this allows us to iterate over edges in graph SCCs, which is
/// what we want to do.
struct GraphEdge {
  /// In the graph edge, this ParamNode represents the caller node.
  ParamNode *pnode;
  /// This is the index into the concatenated range over
  /// `[*dependencies, *blockers]` pointing to the callee ParamNode.
  size_t depIdx;

  /// This function returns the callee ParamNode by indexing into the
  /// appropriate dependency list.
  ParamNode *getPointee() const {
    auto &inode = pnode->impl;
    if (depIdx < inode.dependencies.size())
      return inode.dependencies[depIdx].second;
    return inode.blockers[depIdx - inode.dependencies.size()].second;
  }
  /// Return the location on the callee side representing where the edge
  /// originates from, to be used for diagnostic reporting.
  Location getLoc() const {
    auto &inode = pnode->impl;
    if (depIdx < inode.dependencies.size())
      return inode.dependencies[depIdx].first;
    return inode.blockers[depIdx - inode.dependencies.size()].first;
  }
  /// Return true if this edge is a blocker/interpreter edge.
  bool isBlockerEdge() const {
    auto &inode = pnode->impl;
    // Try to find the node in the scc that can break the scc loop.
    // This node should not be an already errored out node since
    // we may break it previously and marked it as errored, we don't
    // want to keep breaking the same node then come back
    // to enter an infinite loop here.
    return depIdx >= inode.dependencies.size() && !inode.error;
  }

  // Comparison operators for GraphTraits.
  bool operator==(const GraphEdge &rhs) const {
    return pnode == rhs.pnode && depIdx == rhs.depIdx;
  }
  bool operator!=(const GraphEdge &rhs) const { return !(*this == rhs); }

  /// Iterate over the children of the edge by iterating the dependencies of the
  /// callee node. This returns the first dependency.
  GraphEdge begin() const {
    ParamNode *next = getPointee();
    return {next, 0};
  }
  /// Iterate over the children of the edge by iterating the dependencies of the
  /// callee node. This returns the past-the-end iterator, where the index is
  /// equal to the number of dependencies.
  GraphEdge end() const {
    ParamNode *next = getPointee();
    ImplNode &inode = next->impl;
    return {next, inode.dependencies.size() + inode.blockers.size()};
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
  PartialExpansionGraph(ArrayRef<ParamNode *> roots) {
    // Gross hack to create a virtual root edge to all root generators.
    // This node has an edge to each of the root nodes.
    for (ParamNode *root : roots)
      virtualRoot.impl.dependencies.emplace_back(root->gen.getLoc(), root);

    // The base node just has an edge to the virtual root.
    baseNode.impl.dependencies.emplace_back(roots.front()->gen.getLoc(),
                                            &virtualRoot);
  }

  ParamNode virtualRoot;
  ParamNode baseNode;
};
} // namespace

namespace llvm {
template <>
struct DenseMapInfo<GraphEdge> {
  static unsigned getHashValue(GraphEdge node) {
    return DenseMapInfo<std::pair<ParamNode *, size_t>>::getHashValue(
        {node.pnode, node.depIdx});
  }
  static bool isEqual(GraphEdge lhs, GraphEdge rhs) { return lhs == rhs; }
};

template <>
struct GraphTraits<PartialExpansionGraph> {
  using NodeRef = GraphEdge;
  using ChildIteratorType = GraphEdge;

  static NodeRef getEntryNode(const PartialExpansionGraph &g) {
    return {const_cast<ParamNode *>(&g.baseNode), 0};
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

bool Elaborator::diagnoseAndBreakRecursion(unsigned generation,
                                           ArrayRef<ParamNode *> roots) {
  PartialExpansionGraph graph(roots);

  // Re-used data structures to reduce memory pressure.
  DenseSet<GraphEdge> inSCC;
  std::vector<AnyAsyncValueRef> sccChains;
  llvm::SetVector<ParamNode *> sccNodes; // this one gets moved

  // These are the nodes we are going to reschedule at the end.
  std::vector<ImplNode *> reschedule;
  // Suspended SCC members are woken by their blocker resumes rather than a
  // reschedule, so progress is not implied by `reschedule` alone.
  bool brokeScc = false;

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
      ImplNode *inode = &badEdge->pnode->impl;
      inode->setToError(buildRecursionError(*badEdge, scc, inSCC));
      inode->stack.clear();
      reschedule.push_back(inode);
      break;
    }

    // Now, we break all the edges in the SCC for each node in the SCC.
    brokeScc = true;
    for (ParamNode *node : sccNodes) {
      ImplNode *inode = &node->impl;
      std::vector<std::pair<Location, ParamNode *>> newDeps;
      for (auto [idx, dep] : llvm::enumerate(inode->dependencies)) {
        if (!inSCC.contains(GraphEdge{node, idx}))
          newDeps.push_back(dep);
        else
          inode->sccRemovedDeps.push_back(dep);
      }

      // A node suspended on blockers still holds its unconsumed processing
      // count and is re-run by its blocker resume, so it gets neither a
      // reschedule nor the extra count reserved for the rescheduled pass.
      bool suspended = inode->pendingResumes.load() > 0;
      inode->numDependencies -=
          (inode->dependencies.size() - newDeps.size() - (suspended ? 0 : 1));
      inode->dependencies = std::move(newDeps);
      inode->sccCh = AsyncValueRef<Chain>::allocate(cpuDevice);
      sccChains.push_back(inode->sccCh.copy());
      if (!suspended)
        reschedule.push_back(inode);
    }

    // When all of them are done as individual nodes, they will reset their
    // dependency counter to 1 and wait for all chains to complete. Then
    // sccRemovedDeps on each node is checked to propagate any errors from
    // callees that errored during the first pass.
    AsyncRT::andThenAsyncMoving(sccChains,
                                [this, nodes = sccNodes.takeVector()](
                                    MutableArrayRef<AnyAsyncValueRef>) {
                                  for (ParamNode *node : nodes)
                                    completeImplNodeProcessing(&node->impl);
                                });
  }

  // Now reschedule the nodes outside the loop to avoid races.
  for (ImplNode *inode : reschedule) {
    g.numWorkItems.fetch_add(1);
    scheduleImplNode(inode);
  }
  return brokeScc || !reschedule.empty();
}

//===----------------------------------------------------------------------===//
// Elaborator::run
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

bool Elaborator::checkCodeGenUnreachable(ModuleOp module,
                                         DenseSet<StringAttr> &usedSymbols,
                                         ErrorLimit &errorLimit,
                                         SymbolTable &symtab) {
  DenseMap<InstantiatedOpInterface, ParamNode *> nodeMap;
  for (auto &node : llvm::make_second_range(concreteNodes.get())) {
    nodeMap.insert({node->inst, node->parent});
  }

  std::function<void(ParamNode *, Location, ErrorTree, DenseSet<ParamNode *> &)>
      emitError = [&](ParamNode *node, Location loc, ErrorTree cause,
                      DenseSet<ParamNode *> &seenParents) -> void {
    DenseMap<ParamNode *, DenseMap<ParamNode *, std::vector<Location>>> &graph =
        callInstantiationGraph.get();
    auto iter = graph.find(node);
    if (iter == graph.end() || iter->second.empty() ||
        seenParents.contains(node)) {
      emitLimitedError(
          [&] {
            return std::move(cause).emit(
                [](Location loc) { return mlir::emitError(loc); },
                "call expansion failed",
                options.elaborationErrorIncludePrelude);
          },
          errorLimit);
      return;
    }
    seenParents.insert(node);
    DenseMap<ParamNode *, std::vector<Location>> &parents = iter->second;
    for (auto &[parent, locs] : parents) {
      // callInstantiationGraph can have cycles for recursive function calls at
      // cpuDevice (with proper cpuDevice leaf condition which is totally legal)
      // Stop recursing to the parent while emitting the errors if the parent
      // has already seen.

      std::string str = printSimpleParamAttrValues(
          parent->gen.getInputParams(), parent->inputParams,
          options.elaborationErrorVerbose);
      std::string msg = "call expansion failed";
      if (!str.empty())
        msg += " with parameter value(s): " + str;

      for (auto loc : locs) {
        emitError(parent, loc, ErrorTree(loc, Twine(msg), cause.copy()),
                  seenParents);
      }
    }
  };

  bool result = true;
  for (auto name : usedSymbols) {
    // Check all the live functions.
    if (auto func = symtab.lookup<FuncOp>(name)) {
      func->walk([&](CodeGenReachableOp op) {
        auto cond = cast<IntegerAttr>(op.getCond());
        if (cond.getValue().isZero()) {
          // Emit error;
          auto iter = nodeMap.find(func);
          assert(iter != nodeMap.end());
          DenseSet<ParamNode *> seenParents;

          emitError(
              iter->second, op->getLoc(),
              ErrorTree(op.getLoc(),
                        "codegen unreachable: " +
                            cast<StringAttr>(op.getMessageAttr()).getValue()),
              seenParents);
          result = false;
        }
      });
    }
  }

  return result;
}

LogicalResult Elaborator::run(
    ModuleOp theModule,
    ArrayRef<std::pair<GeneratorOp, ParameterExprArrayAttr>> primaryGenerators,
    mlir::SymbolTableAnalysis &analysis) {
  // Find any kgen.func we have already - they're already elaborated, and we
  // do not want to re-process them. Add concrete ImplNodes for each one.
  for (FuncOp func : theModule.getOps<FuncOp>()) {
    (void)addConcreteFunc(func, func.getSymNameAttr(), concreteNodes.get());
    addRegion(func.getBodyRegion());
  }

  std::vector<AnyAsyncValueRef> primaryChs;
  std::vector<std::unique_ptr<ImplNode>> rootNodes;
  std::vector<ParamNode *> primaryNodes;
  primaryChs.reserve(primaryGenerators.size());
  primaryNodes.reserve(primaryGenerators.size());
  for (auto [gen, params] : primaryGenerators) {
    // This has no input parameters, so we can create the expansion node with
    // no input parameters.
    ParamNode *genNode = getOrCreateNode(params, gen, /*depth=*/0);
    primaryNodes.push_back(genNode);

    // Create a special root node for this primary generator.
    ImplNode *root =
        rootNodes.emplace_back(std::make_unique<ImplNode>(genNode)).get();

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

  for (ParamNode *genNode : primaryNodes) {
    ErrorTreeOrSuccess err = genNode->collectErrorsOrSuccess();
    if (err.isError()) {
      failed = true;
      emitLimitedError(
          [&]() {
            return err.takeError().emit(
                [](Location loc) { return mlir::emitError(loc); },
                "call expansion failed",
                options.elaborationErrorIncludePrelude);
          },
          errorLimit);
    }
  }

  if (failed) {
    g.eraseNodeInst = true;
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

    g.eraseNodeInst = true;
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
        [&]() {
          return std::move(compileOffloadError)
              .emit([](Location loc) { return mlir::emitError(loc); },
                    "Compile offload failed.",
                    options.elaborationErrorIncludePrelude);
        },
        errorLimit);

    g.eraseNodeInst = true;
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
  for (ParamNode &node :
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

  // Keep the old module body alive for checkCodeGenUnreachable erroring.
  mlir::OwningOpRef<mlir::ModuleOp> tmpModule =
      mlir::ModuleOp::create(theModule->getLoc());
  theModule.getBody()->moveBefore(tmpModule->getBody());

  // Update the symbol table with the new one.
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
      // Plug offload compilation results as strings back to the elaborated
      // IR.
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

  for (Operation &op : theModule.getOps())
    if (op.hasTrait<OpTrait::SymbolTable>())
      analysis.getSymbolTables().invalidateSymbolTable(&op);

  oldSymTab = SymbolTable(theModule);

  DenseSet<StringAttr> usedSymbols = getUsedSymbols(analysis, theModule);

  failed |=
      !checkCodeGenUnreachable(theModule, usedSymbols, errorLimit, oldSymTab);

  return success(!failed);
}

//===----------------------------------------------------------------------===//
// ElaborateGeneratorsPass
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_ELABORATEGENERATORS
#define GEN_PASS_DEF_RESOLVEINCLUDES
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {

/// Run the elaborator as a pass. The elaborator requires imports to be
/// resolved, so first resolve imports and then elaborate.
class ElaborateGeneratorsPass
    : public KGEN::impl::ElaborateGeneratorsBase<ElaborateGeneratorsPass> {
public:
  ElaborateGeneratorsPass(const ElaborateGeneratorsOptions &elabOpts = {},
                          TargetInfoAttr target = nullptr,
                          const CompilationOptions &options = {},
                          ElaboratorCompileAsmFn compileAsmFn = {},
                          ElaboratorCompileOffloadFn compileOffloadFn = {})
      : ElaborateGeneratorsBase(elabOpts), target(target), options(options),
        compileAsmFn(std::move(compileAsmFn)),
        compileOffloadFn(std::move(compileOffloadFn)) {}

  void runOnOperation() override {
    ModuleOp theModule = getOperation();
    target = lookupTargetInfo(theModule);
    // Default to the host target if one was not specified
    if (!target) {
      ErrorOr<TargetInfoAttr> targetOr = getTargetInfoFor(
          theModule.getContext(), llvm::sys::getDefaultTargetTriple(),
          llvm::sys::getHostCPUName(), getHostCPUFeatures());
      if (targetOr.isError()) {
        mlir::emitError(UnknownLoc::get(theModule.getContext()),
                        targetOr.getError());
        return signalPassFailure();
      }

      target = targetOr.takeValue();
    }

    auto &analysis = getAnalysis<mlir::SymbolTableAnalysis>();
    auto &paramCache = getAnalysis<ParameterCollector::Analysis>();
    SymbolTable &symtab = analysis.getTopLevelSymbolTable();

    // Root elaboration on exports and global variables. These are the
    // generators that elaboration will start from. If there are no such
    // generators, then elaborate anything with no input parameters.
    llvm::SetVector<std::pair<GeneratorOp, ParameterExprArrayAttr>> roots;
    auto emptyParams = ParameterExprArrayAttr::get(&getContext(), {});
    for (Operation &op : theModule.getOps()) {
      if (auto gen = dyn_cast<GeneratorOp>(op);
          gen && gen.isExported() && gen.getInputParams().empty()) {
        roots.insert({gen, emptyParams});
      }
    }

    // Extract the top-level, parameterless generators from the main module.
    // These are the only generators that will be elaborated.
    if (roots.empty()) {
      for (auto gen : theModule.getOps<GeneratorOp>())
        if (gen.getInputParams().empty())
          roots.insert({gen, emptyParams});
    }

    // Elaboration is the compilation phase in which the IR goes from
    // target-non-specific to target-specific: in order to fully concretize
    // the IR, we must evaluate compile-time expressions, which is a
    // target-specific operation. Make the IR target-specific by attaching the
    // required target specification.
    if (TargetInfoAttr targetInfo = getTargetInfo(theModule))
      target = targetInfo;
    else
      setTargetInfo(theModule, target);

    // If the module is missing an environment attribute, set an empty one.
    if (!theModule->hasAttrOfType<EnvAttr>(EnvAttr::getEnvAttrName())) {
      theModule->setAttr(EnvAttr::getEnvAttrName(),
                         EnvAttr::get(DictionaryAttr::get(&getContext())));
    }

    ElaborateGeneratorsOptions config{maxDepth,
                                      elaborateDebugInfo,
                                      optimizeInterpreter,
                                      useParametricInterpreter,
                                      errorIncludePrelude,
                                      errorVerbose,
                                      loopUnrollingWarnThreshold};

    VerboseCompilerTimeTraceScope traceScope("elaborate-generators");
    options.elaborationErrorIncludePrelude = errorIncludePrelude;
    options.elaborationErrorVerbose = errorVerbose;
    options.elaborationMaxDepth = maxDepth;

    // Now, construct and run the elaborator.
    if (useParametricInterpreter) {
      ParametricElaborator impl(symtab, paramCache, target, options,
                                compileAsmFn, compileOffloadFn, config);
      if (failed(impl.run(theModule, roots.takeVector())))
        return signalPassFailure();

      // Invalidate nested symbol tables and recompute a new top level symbol
      // table.
      for (Operation &op : theModule.getOps())
        if (op.hasTrait<OpTrait::SymbolTable>())
          analysis.getSymbolTables().invalidateSymbolTable(&op);
      symtab = SymbolTable(theModule);

    } else {
      Elaborator impl(symtab, paramCache, target, options, compileAsmFn,
                      compileOffloadFn, config);
      if (failed(impl.run(theModule, roots.takeVector(), analysis)))
        return signalPassFailure();
    }
  }

private:
  /// The compilation target.
  TargetInfoAttr target;
  /// The compilation options.
  CompilationOptions options;
  /// The functor used to compile a module to assembly.
  ElaboratorCompileAsmFn compileAsmFn;

  /// The functor used to compile bundled offload functions.
  ElaboratorCompileOffloadFn compileOffloadFn;
};
} // namespace

std::unique_ptr<mlir::Pass> KGEN::createElaborateGenerators(
    TargetInfoAttr target, const ElaborateGeneratorsOptions &elabOpts,
    const CompilationOptions &options, ElaboratorCompileAsmFn compileAsmFn,
    ElaboratorCompileOffloadFn compileOffloadFn) {
  return std::make_unique<ElaborateGeneratorsPass>(elabOpts, target, options,
                                                   std::move(compileAsmFn),
                                                   std::move(compileOffloadFn));
}
