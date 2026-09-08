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

#ifndef KGEN_HLCFDIALECT_ANALYSIS_CONTROLFLOWTREE_H
#define KGEN_HLCFDIALECT_ANALYSIS_CONTROLFLOWTREE_H

#include "Mojo/HLCFDialect/HLCFInterfaces.h"
#include "mlir/Pass/AnalysisManager.h"
#include "llvm/ADT/SmallVector.h"

namespace M::HLCF {
/// This analysis contains information about the control-flow tree rooted at the
/// given operation.
class ControlFlowTree {
public:
  /// Build the tree at the given operation.
  explicit ControlFlowTree(Operation *op);

  /// A map of operation ID to the operation. The ID is the depth-first visit
  /// order of the operation.
  SmallVector<ControlFlowNode> ops;

  /// A map of terminators to their branch target and a flag indicating whether
  /// the target is before or after the operation.
  SmallVector<std::pair<unsigned, SmallVector<ControlFlowTarget, 1>>> targets;

private:
  /// Build the control-flow relations.
  void buildTree(ControlFlowNode node, unsigned &nodeId,
                 SmallVectorImpl<unsigned> &nodeIds);
};

/// This analysis contains cached control-flow tree analyses mapped by root
/// operation. This class ensures that nested analyses are preserved across
/// passes that may change parent operations of root operations, such as
/// rewriting `kgen.func` to `llvm.func`. The analysis is always assumed to be
/// preserved unless indicated otherwise. The analysis should be invalidated
/// when ControlFlowNode operations are deleted, inserted, or modified.
class ControlFlowTreeAnalysis {
public:
  ControlFlowTreeAnalysis(Operation *) {}

  /// Get the control-flow tree analysis for the given node, creating it if it
  /// has not already been computed.
  const ControlFlowTree &getOrCreate(ControlFlowNode node) {
    auto it = analyses.find(node);
    if (it != analyses.end())
      return it->second;
    return analyses.try_emplace(node, ControlFlowTree(node)).first->second;
  }

  /// Never automatically invalidate the analysis.
  bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &pa) {
    return false;
  }

private:
  /// This is the map of root operation to its control-flow tree analysis.
  DenseMap<Operation *, ControlFlowTree> analyses;
};
} // namespace M::HLCF

#endif // KGEN_HLCFDIALECT_ANALYSIS_CONTROLFLOWTREE_H
