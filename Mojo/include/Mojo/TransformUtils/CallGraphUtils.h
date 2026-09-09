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

#ifndef KGEN_TRANSFORMUTILS_CALLGRAPHUTILS_H
#define KGEN_TRANSFORMUTILS_CALLGRAPHUTILS_H

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Support/RWMutex.h"

namespace M::KGEN {

template <typename DerivedT, typename FuncT, typename CallT>
struct CallGraphEdge;
template <typename DerivedT, typename FuncT, typename CallT>
struct CallGraphEdgeIterator;

//===----------------------------------------------------------------------===//
// CallGraphNode
//===----------------------------------------------------------------------===//

template <typename DerivedT, typename FuncT, typename CallT>
struct CallGraphEdgeBase {
  CallGraphEdgeBase(CallT call, DerivedT *node) : call(call), node(node) {}

  CallT call;
  DerivedT *node;

  using BaseT = CallGraphEdgeBase<DerivedT, FuncT, CallT>;

  auto begin() { return node->begin(); }
  auto end() { return node->end(); }

  bool operator==(const BaseT &rhs) const {
    return std::tie(call, node) == std::tie(rhs.call, rhs.node);
  }
  bool operator!=(const BaseT &rhs) const { return !(*this == rhs); }
};

/// A node in a call graph contains a function, edges to its callers, and edges
/// to its callees. A node is ready to inline its callees when all of its
/// callees have been processed.
template <typename DerivedT, typename FuncT, typename CallT>
struct CallGraphNodeBase {
  using FuncOpT = FuncT;
  using CallOpT = CallT;

  using BaseT = CallGraphNodeBase<DerivedT, FuncT, CallT>;
  using EdgeT = CallGraphEdgeBase<DerivedT, FuncT, CallT>;
  using EdgeListT = SmallVector<EdgeT>;
  using EdgeIteratorT = typename EdgeListT::iterator;

  /// Create the node for the given function.
  explicit CallGraphNodeBase(FuncT func) : func(func) {}

  /// This class is only move-constructed when the node map in
  /// `InliningGraphBase` is resized. That occurs before any references are
  /// taken to instances of this object, so just default-construct all other
  /// members of this class.
  CallGraphNodeBase(CallGraphNodeBase &&other) : func(other.func) {
    other.func = nullptr;
  }

  auto begin() { return callsites.begin(); }
  auto end() { return callsites.end(); }

  /// The function represented by the node.
  FuncOpT func;

  /// Nodes of functions that inline call this function. These are the child
  /// edges.
  SmallVector<DerivedT *> callers;
  /// Calls and callees to inline inside this function. These are the parent
  /// edges.
  EdgeListT callsites;
  /// This mutex guards `callsites` and `callers` during parallel graph
  /// construction.
  llvm::sys::SmartRWMutex<true> mutex;
};

//===----------------------------------------------------------------------===//
// CallGraph
//===----------------------------------------------------------------------===//

/// A callgraph is a graph where the nodes represent functions and the edges
/// represent calls between functions. This is a generic callgraph that provides
/// a build method.
template <typename DerivedT, typename NodeT>
struct CallGraphBase {
  using FuncOpT = typename NodeT::FuncOpT;
  using CallOpT = typename NodeT::CallOpT;
  using BaseT = CallGraphBase<DerivedT, NodeT>;
  using EdgeT = typename NodeT::EdgeT;

  /// Get a reference to the derived class.
  DerivedT &getDerived() { return *static_cast<DerivedT *>(this); }

  /// This method handles an operation that is not a call in a function. The
  /// base implementation does nothing.
  void checkNonCallOp(Operation *op) {}

  /// This method is called on a call graph edge and determines whether it
  /// should be added to the graph. That is, whether the analysis cares about
  /// this edge. The base implementation always returns true.
  bool shouldAddToGraph(CallOpT call, NodeT *node) { return true; }

  /// Build the inlining graph for a module.
  template <typename... NodeArgTs>
  void build(ModuleOp module, const SymbolTable &symtab, NodeArgTs &&...args);

  /// Dump the callgraph. For debugging.
  void dump();

  /// Lookup the NodeT corresponding to the given FuncOp.
  NodeT *lookup(FuncOpT func) {
    auto it = nodes.find(func);
    return it == nodes.end() ? nullptr : &it->second;
  }

  /// Lookup the NodeT corresponding to the given FuncOp.
  const NodeT *lookup(FuncOpT func) const {
    auto it = nodes.find(func);
    return it == nodes.end() ? nullptr : &it->second;
  }

  /// The nodes in the graph. The map does not resize after it is constructed,
  /// so references always remain valid.
  llvm::MapVector<FuncOpT, NodeT> nodes;
};

template <typename DerivedT, typename NodeT>
template <typename... NodeArgTs>
void CallGraphBase<DerivedT, NodeT>::build(ModuleOp module,
                                           const SymbolTable &symtab,
                                           NodeArgTs &&...args) {
  VerboseCompilerTimeTraceScope traceScope("CallGraphBase::build");

  // Instantiate the nodes for each generator first.
  for (auto func : llvm::make_early_inc_range(module.getOps<FuncOpT>())) {
    nodes.insert(
        std::make_pair(func, NodeT(func, std::forward<NodeArgTs>(args)...)));
  }

  // Build the graph by walking all the calls in each function and adding edges
  // as appropriate.
  auto workFn = [this, &symtab](NodeT *callerNode) {
    FuncOpT func = callerNode->func;
    func.getBodyRegion().walk([&](Operation *op) {
      auto call = dyn_cast<CallOpT>(op);
      if (!call) {
        getDerived().checkNonCallOp(op);
        return;
      }

      auto symbol =
          dyn_cast_if_present<FlatSymbolRefAttr>(call.getCalleeSymbol());
      assert(symbol && "call op not using flat symbol references");

      Operation *calleeOp = symtab.lookup(symbol.getAttr());
      assert(calleeOp && "invalid IR?");
      // Only add the edge if the symbol we found is of the type we expect.
      auto callee = dyn_cast<FuncOpT>(calleeOp);
      if (!callee)
        return;

      NodeT *calleeNode = &nodes.find(callee)->second;
      // Filter calls that do not satisfy the inlining level.
      if (!getDerived().shouldAddToGraph(call, calleeNode))
        return;
      {
        llvm::sys::SmartScopedWriter<true> lock(callerNode->mutex);
        callerNode->callsites.emplace_back(call, calleeNode);
      }
      {
        llvm::sys::SmartScopedWriter<true> lock(calleeNode->mutex);
        calleeNode->callers.push_back(callerNode);
      }
    });
  };
  std::vector<NodeT *> work;
  work.reserve(nodes.size());
  for (auto &[_, node] : nodes)
    work.push_back(&node);
  mlir::parallelForEach(module.getContext(), work, workFn);
}

template <typename DerivedT, typename NodeT>
void CallGraphBase<DerivedT, NodeT>::dump() {
  for (auto &[func, node] : nodes) {
    llvm::errs() << "@" << func.getSymName() << ":\n";
    for (auto [call, callee] : node.callsites)
      llvm::errs() << "  -> @" << callee->func.getSymName() << "\n";
    llvm::errs() << "\n";
  }
}

} // namespace M::KGEN

namespace llvm {
template <typename DerivedT, typename FuncT, typename CallT>
struct DenseMapInfo<M::KGEN::CallGraphEdgeBase<DerivedT, FuncT, CallT>>
    : DenseMapInfo<std::pair<CallT, DerivedT *>> {
  using EdgeT = M::KGEN::CallGraphEdgeBase<DerivedT, FuncT, CallT>;

  static unsigned getHashValue(const EdgeT &node) {
    return DenseMapInfo<std::pair<CallT, DerivedT *>>::getHashValue(
        {node.call, node.node});
  }
  static bool isEqual(const EdgeT &lhs, const EdgeT &rhs) { return lhs == rhs; }
};

template <typename DerivedT, typename FuncT, typename CallT>
struct GraphTraits<M::KGEN::CallGraphNodeBase<DerivedT, FuncT, CallT> *> {
  using NodeT = M::KGEN::CallGraphNodeBase<DerivedT, FuncT, CallT>;
  using NodeRef = typename NodeT::EdgeT;
  using ChildIteratorType = typename NodeT::EdgeListT::iterator;

  static NodeRef getEntryNode(DerivedT *node) { return {nullptr, node}; }
  static ChildIteratorType child_begin(NodeRef edge) { return edge.begin(); }
  static ChildIteratorType child_end(NodeRef edge) { return edge.end(); }
};

template <typename DerivedT, typename NodeT>
struct GraphTraits<M::KGEN::CallGraphBase<DerivedT, NodeT> *> {
  static NodeT *getEntryNode(DerivedT *graph) { return &graph->externalNode; }
  static NodeT *getNode(typename NodeT::EdgeT edge) { return edge.node; }

  using NodeRef = NodeT *;
  using ChildIteratorType =
      llvm::mapped_iterator<typename NodeT::EdgeIteratorT, decltype(&getNode)>;

  static ChildIteratorType child_begin(NodeRef node) {
    return ChildIteratorType(node->begin(), &getNode);
  }
  static ChildIteratorType child_end(NodeRef node) {
    return ChildIteratorType(node->end(), &getNode);
  }
};
} // namespace llvm

#endif // KGEN_TRANSFORMUTILS_CALLGRAPHUTILS_H
