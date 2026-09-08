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

#include "AsyncRT/CompilerSupport/Context.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Mojo/TransformUtils/SCCUtils.h"
#include "Mojo/TransformUtils/Walkers.h"
#include "Support/Context.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/SCCIterator.h"

using namespace M;
using namespace KGEN;

namespace {

//===----------------------------------------------------------------------===//
// CallLikeOp
//===----------------------------------------------------------------------===//

/// Op-like wrapper for call operations that must behave as edges in the
/// callgraph.
class CallLikeOp {
public:
  // Required methods for LLVM-style RTTI.
  CallLikeOp(Operation *op) : op(op) {
    assert((!op || classof(op)) && "not a call-like op");
  }
  static bool classof(Operation *op) {
    return isa_and_nonnull<KGENCallOpInterface>(op);
  }
  operator bool() const { return op; }
  operator Operation *() const { return op; }
  Operation &operator*() { return *op; }
  Operation *operator->() { return op; }

  /// Required CallGraph interface.
  TypedAttr getCallee() { return cast<KGENCallOpInterface>(op).getCallee(); }

  FlatSymbolRefAttr getCalleeSymbol() {
    return cast<FlatSymbolRefAttr>(
        cast<SymbolConstantAttr>(getCallee()).getSymbol());
  }

private:
  /// The underlying operation.
  Operation *op;
};

//===----------------------------------------------------------------------===//
// CallGraphNode
//===----------------------------------------------------------------------===//

/// A basic callgraph node, containing state for this dataflow analysis across
/// the callgraph.
struct CallGraphNode : public SCCNode<CallGraphNode, FuncOp, CallLikeOp> {
  using SCCNode::SCCNode;

  /// The current set of required promises for the function.
  llvm::MapVector<StringAttr, std::pair<Type, unsigned>> requiredPromises;
};

//===----------------------------------------------------------------------===//
// CallGraph
//===----------------------------------------------------------------------===//

struct CallGraph : public SCCGraph<CallGraph, CallGraphNode> {
  CallGraph(const SymbolTable &symtab) : symtab(symtab) {}

  /// Only add nodes to the graph that are to functions that are capturing,
  /// since those are the only functions we need to handle.
  bool shouldAddToGraph(CallLikeOp call, CallGraphNode *node) {
    return node->func.getFuncTypeGenerator().getBody().isCapturing();
  }

  /// Lookup the call graph node for the given symbol reference.
  CallGraphNode *getCalleeNode(TypedAttr symbol) {
    auto callee = symtab.lookup<FuncOp>(
        cast<SymbolConstantAttr>(symbol).getSymbol().getRootReference());
    assert(callee);
    return &nodes.find(callee)->second;
  }

  /// Run an iteration of the analysis and transformation on a single node.
  /// Return true if anything changed.
  bool doAnalysis(CallGraphNode *node);
  void doRewrite(const CallGraphNode *node);

  const SymbolTable &symtab;
};
} // namespace

void CallGraph::doRewrite(const CallGraphNode *node) {
  FuncOp func = node->func;
  func.walk([](Operation *op) {
    if (isa<POP::CompilerGlobalStoreOp>(op))
      op->erase();
  });
}

static size_t getNumReturnSlots(FuncType funcType) {
  ArrayRef<ArgConvention> conventions = funcType.getArgConventions();
  size_t result = 0;
  while (!conventions.empty() && isResultSlot(conventions.back())) {
    ++result;
    conventions = conventions.drop_back();
  }
  return result;
}

bool CallGraph::doAnalysis(CallGraphNode *node) {
  FuncOp func = node->func;
  llvm::MapVector<StringAttr, SmallVector<POP::CompilerGlobalLoadOp>>
      requiredPromises;
  unsigned curNumPromises = node->requiredPromises.size();

  VerboseCompilerTimeTraceScope traceScope(
      "resolvePromises", [func]() mutable { return func.getSymName().str(); });

  // This functor will, given an operation that points to another node, create
  // new required promises based on any additional required promises from the
  // callee.
  auto computeRequiredCaptures = [&requiredPromises](Operation *op,
                                                     CallGraphNode *calleeNode,
                                                     unsigned fulfilled) {
    ImplicitLocOpBuilder b(op->getLoc(), OpBuilder(op));
    // Create new loads to keep the state. Because the walk uses an early
    // inc, it will not visit the loads twice.
    SmallVector<Value> captures;
    for (auto [name, type] :
         llvm::drop_begin(calleeNode->requiredPromises, fulfilled)) {
      auto load = POP::CompilerGlobalLoadOp::create(b, type.first, name);
      requiredPromises[name].push_back(load);
      captures.push_back(load);
    }
    return captures;
  };

  /// This functor will, given the current set of required promises, transfer
  /// any new ones to the enclosing function.
  auto consumeRequiredPromises = [node,
                                  &requiredPromises](ValueRange fulfilled) {
    SmallVector<std::pair<Type, SmallVector<POP::CompilerGlobalLoadOp>>>
        newTypes;
    for (auto &[name, loads] : requiredPromises) {
      // If this required promise is already fulfilled on the node, then replace
      // the request immediately.
      if (auto it = node->requiredPromises.find(name);
          it != node->requiredPromises.end()) {
        auto [type, index] = it->second;
        for (POP::CompilerGlobalLoadOp load : loads) {
          load.replaceAllUsesWith(fulfilled[index]);
          load.erase();
        }
        continue;
      }
      // Otherwise, save the request for the new promise.
      Type type = loads.front().getType();
      node->requiredPromises.insert(
          {name, {type, node->requiredPromises.size()}});
      newTypes.emplace_back(type, std::move(loads));
    }
    // Consume the current state.
    requiredPromises.clear();
    return newTypes;
  };

  reversePostOrderWalk(func, [&](Operation *op) {
    // When we encounter a load, mark it as a requested promise within the
    // function.
    if (auto load = dyn_cast<POP::CompilerGlobalLoadOp>(op)) {
      requiredPromises[load.getNameAttr()].push_back(load);
      return;
    }

    // When a store is encountered, resolve every load that requested this
    // value.
    if (auto store = dyn_cast<POP::CompilerGlobalStoreOp>(op)) {
      auto it = requiredPromises.find(store.getNameAttr());
      if (it == requiredPromises.end())
        return;

      SmallVector<POP::CompilerGlobalLoadOp> leftover;
      for (POP::CompilerGlobalLoadOp load : it->second) {
        // Make sure the store dominates the load in terms of regions.
        if (store->getParentRegion()->isAncestor(load->getParentRegion())) {
          load.replaceAllUsesWith(store.getValue());
          load.erase();
        } else {
          leftover.push_back(load);
        }
      }
      // If there no leftover ops, then the promise is not pending anymore.
      if (leftover.empty())
        requiredPromises.erase(it);
      else
        it->second = std::move(leftover);
      return;
    }

    // When a call is encountered, look up the required promises of the
    // function it is calling. Rewrite the call to provide them.
    if (auto call = dyn_cast<KGENCallOpInterface>(op)) {
      auto symbol = cast<SymbolConstantAttr>(call.getCallee());
      FuncType sig = symbol.getType().getBody();
      // Calls to functions that are not capturing cannot have captures.
      if (!sig.isCapturing())
        return;
      CallGraphNode *calleeNode = getCalleeNode(symbol);
      FuncOp callee = calleeNode->func;

      // Exit early if there is nothing to do.
      if (calleeNode->requiredPromises.empty())
        return;

      unsigned fulfilled =
          calleeNode->requiredPromises.size() -
          (calleeNode->func.getNumArguments() - sig.getNumArguments());
      SmallVector<Value> captures =
          computeRequiredCaptures(call, calleeNode, fulfilled);
      // Insert any new captures and update the callee signature on the call.
      // The function already has the updated signature.
      // New capture args are inserted immediately before any return slots.
      size_t argInsertionIndex = sig.getNumArguments() - getNumReturnSlots(sig);
      call->insertOperands(argInsertionIndex, captures);
      call.setCalleeAttr(SymbolConstantAttr::get(
          symbol.getSymbol(), callee.getFuncTypeGenerator()));
      return;
    }
  });

  if (!func.getFuncTypeGenerator().getBody().isCapturing())
    return node->requiredPromises.size() != curNumPromises;

  // At the end of the walk, assess the leftover required promises. Insert
  // them to the signature and block arguments (inserted before any return
  // slots).
  size_t numResultArgs =
      getNumReturnSlots(func.getFuncTypeGenerator().getBody());
  auto newTypes = consumeRequiredPromises(
      func.getArguments()
          .take_back(node->requiredPromises.size() + numResultArgs)
          .drop_back(numResultArgs));
  if (newTypes.empty())
    return false;

  Block *body = func.getBody();
  unsigned argInsertionIndex = body->getNumArguments() - numResultArgs;
  // Insert arguments for any new captures to propagate.
  for (auto &[type, loads] : newTypes) {
    Value arg = body->insertArgument(argInsertionIndex++, type, func.getLoc());
    for (POP::CompilerGlobalLoadOp load : loads) {
      load.replaceAllUsesWith(arg);
      load.erase();
    }
  }

  FuncType sig = func.getFuncTypeGenerator().getBody();
  // TODO: What conventions do we use for captures.
  SmallVector<ArgConvention> convs(sig.getArgConventions());
  convs.insert(std::prev(convs.end(), numResultArgs), newTypes.size(),
               ArgConvention::ImmReg);
  assert(body->getNumArguments() == convs.size());

  // Update the function signature.
  auto fnType = FunctionType::get(func.getContext(), body->getArgumentTypes(),
                                  func.getResultTypes());
  func.setFuncTypeGenerator(GeneratorType::get(
      /*inputParamTypes=*/{},
      FuncType::get(fnType, convs, sig.getFnEffects())));

  // Extend LLVM per-arg metadata.
  if (ArrayRef<Attribute> oldLLVMArgMetadata =
          func.getLLVMArgMetadata().getValue();
      !oldLLVMArgMetadata.empty()) {
    SmallVector<Attribute> llvmArgMetadata(oldLLVMArgMetadata);
    llvmArgMetadata.insert(llvmArgMetadata.end(), newTypes.size(),
                           DictionaryAttr::get(func.getContext()));
    func.setLLVMArgMetadataAttr(
        ArrayAttr::get(func->getContext(), llvmArgMetadata));
  }

  // If captures went up to an exported function, propagate them through
  // the ABI boundary by encoding the capture names on the function.
  if (func.isExported() && !node->requiredPromises.empty()) {
    SmallVector<StringAttr> captures;
    // Store the expected capture type in the StringAttr.
    for (auto [name, type] : node->requiredPromises)
      captures.push_back(StringAttr::get(name.getValue(), type.first));
    func.setCrossDeviceCaptures(captures);
  }

  return true;
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_RESOLVECOMPILERPROMISES
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct ResolveCompilerPromisesPass
    : impl::ResolveCompilerPromisesBase<ResolveCompilerPromisesPass> {
  void runOnOperation() override;
};
} // namespace

void ResolveCompilerPromisesPass::runOnOperation() {
  VerboseCompilerTimeTraceScope traceScope(
      "ResolveCompilerPromisesPass::runOnOperation");
  const SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

  AsyncRT::CPUDevice &cpuDevice =
      *loadContext(&getContext())->get<AsyncRT::CPUDevice>();
  CallGraph cg(symtab);
  cg.build(getOperation(), symtab);
  cg.run(cpuDevice);
}
