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

#ifndef KGEN_TRANSFORMUTILS_SCCUTILS_H
#define KGEN_TRANSFORMUTILS_SCCUTILS_H

#include "AsyncRT/Runtime/ForkJoin.h"
#include "Mojo/TransformUtils/CallGraphUtils.h"

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// SCCNode
//===----------------------------------------------------------------------===//

template <typename DerivedT, typename FuncOpT, typename CallOpT>
struct SCCNode : CallGraphNodeBase<DerivedT, FuncOpT, CallOpT> {
  using ParentT = CallGraphNodeBase<DerivedT, FuncOpT, CallOpT>;

  SCCNode(FuncOpT func) : ParentT(func) {}
  SCCNode(SCCNode &&other) : ParentT(std::move(other)) {}

  /// The unit of analysis for this pass is a single SCC in the callgraph. This
  /// struct contains the nodes in an SCC and is the dependency node for it in
  /// the work processing graph.
  struct SCC {
    SCC() {}
    SCC(SCC &&other) {} // only used in `reserve`

    /// The nodes in the SCC. In most cases, SCCs should have a single node (no
    /// or self recursion).
    SmallVector<DerivedT *, 2> nodes;

    std::vector<SCC *> callers;
    bool hasRecursion = false;
    std::atomic<unsigned> numCallees = 0;
    std::atomic<unsigned> numReady = 0;
  };

  /// Pointer to the SCC the node is contained in.
  SCC *scc = nullptr;
};

//===----------------------------------------------------------------------===//
// SCCGraph
//===----------------------------------------------------------------------===//

template <typename DerivedT, typename NodeT, bool BottomUpRewrite = false>
struct SCCGraph : public CallGraphBase<DerivedT, NodeT> {
  using SCC = typename NodeT::SCC;
  using FuncOpT = typename NodeT::FuncOpT;
  using ParentT = CallGraphBase<DerivedT, NodeT>;

  /// Run the analysis on a single SCC.
  void doAnalysis(SCC *scc) {
    // The common case of a single node in the SCC.
    if (LLVM_LIKELY(scc->nodes.size() == 1)) {
      NodeT *node = scc->nodes.front();
      bool changed = ParentT::getDerived().doAnalysis(node);
      // The common case of no self-recursion. We can exit immediately
      // regardless of whether anything changed.
      if (LLVM_LIKELY(!scc->hasRecursion) || !changed)
        return;
      // Fixed-point iterate the same function. We expect it to converge after
      // at most a single iteration.
      changed = ParentT::getDerived().doAnalysis(node);
      assert(!changed && "self recursion didn't converge in 2 iterations");
      return;
    }

    // Each node get visited at least once.
    SmallVector<NodeT *, 2> worklist;
    llvm::append_range(worklist, scc->nodes);

    while (!worklist.empty()) {
      NodeT *node = worklist.pop_back_val();
      if (!ParentT::getDerived().doAnalysis(node))
        continue;

      // The node changed, so reschedule its dependendees in the SCC.
      for (NodeT *caller : node->callers)
        if (caller->scc == scc)
          worklist.push_back(caller);
    }
  }

  /// Process a single SCC.
  void doWork(SCC *scc, AsyncRT::ForkJoin &state) {
    // Skip the root node.
    if (!scc->nodes.front()->func)
      return;
    doAnalysis(scc);

    // For a bottom-up rewrite, perform rewriting as soon as the analysis is
    // complete as rewriting is expected to complete before rewriting any
    // callers are rewritten.
    // Otherwise, queue the rewrites to occur asynchronously.
    if constexpr (BottomUpRewrite) {
      for (NodeT *node : scc->nodes)
        ParentT::getDerived().doRewrite(node);
    } else {
      for (NodeT *node : scc->nodes)
        state.fork([this, node] { ParentT::getDerived().doRewrite(node); });
    }

    for (SCC *caller : scc->callers)
      if (++caller->numReady == caller->numCallees)
        state.fork([this, &state, caller] { doWork(caller, state); });
  }

  /// This runs the analysis on the full call graph in post-order SCC. It starts
  /// by computing the SCCs and then scheduling them for analysis.
  void run(AsyncRT::CPUDevice &cpuDevice) {
    // Add every node as a child of the virtual root node.
    for (auto &[func, node] : this->nodes)
      externalNode.callsites.emplace_back(nullptr, &node);

    // There cannot be more SCCs than there are nodes (plus 1 for the root
    // node). Reserve to avoid indirection and iterator invalidation.
    std::vector<SCC> sccs;
    sccs.reserve(this->nodes.size() + 1);

    llvm::SetVector<NodeT *, SmallVector<NodeT *, 2>,
                    llvm::SmallPtrSet<NodeT *, 2>>
        sccNodes;
    llvm::SetVector<SCC *, std::vector<SCC *>> callers;
    for (auto sccIt = llvm::scc_begin(this); !sccIt.isAtEnd(); ++sccIt) {
      SCC &scc = sccs.emplace_back(SCC{});
      for (NodeT *node : *sccIt) {
        sccNodes.insert(node);
        node->scc = &scc;
      }
      scc.nodes = sccNodes.takeVector();
      scc.hasRecursion = sccIt.hasCycle();
    }

    // Compute the dependency graph between the SCCs.
    for (SCC &scc : sccs) {
      for (NodeT *node : scc.nodes) {
        for (NodeT *caller : node->callers) {
          if (caller->scc != node->scc) {
            if (callers.insert(caller->scc))
              ++caller->scc->numCallees;
          }
        }
      }
      scc.callers = callers.takeVector();
    }

    AsyncRT::ForkJoin state(cpuDevice);

    // Because SCCs are visited in reverse topological order by the SCC
    // iterator, this will schedule leaf nodes first, which is good for
    // utilization.
    for (SCC &scc : sccs)
      if (!scc.numCallees)
        state.fork([this, &state, scc = &scc] { doWork(scc, state); });

    state.join();
  }

  /// The virtual root node of the callgraph. This node points to every other
  /// node in the callgraph.
  NodeT externalNode{nullptr};
};

} // namespace M::KGEN

namespace llvm {
template <typename DerivedT, typename NodeT, bool BottomUpRewrite>
struct GraphTraits<M::KGEN::SCCGraph<DerivedT, NodeT, BottomUpRewrite> *>
    : public GraphTraits<typename M::KGEN::SCCGraph<DerivedT, NodeT,
                                                    BottomUpRewrite>::BaseT *> {
  static NodeT *
  getEntryNode(M::KGEN::SCCGraph<DerivedT, NodeT, BottomUpRewrite> *graph) {
    return &graph->externalNode;
  }
};
} // namespace llvm

#endif // KGEN_TRANSFORMUTILS_SCCUTILS_H
