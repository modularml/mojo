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

#include "Mojo/HLCFDialect/Analysis/ControlFlowTree.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace M;
using namespace HLCF;

ControlFlowTree::ControlFlowTree(Operation *op) {
  unsigned nodeId = 0;
  SmallVector<unsigned> nodeIds;
  buildTree(cast<ControlFlowNode>(op), nodeId, nodeIds);
}

void ControlFlowTree::buildTree(ControlFlowNode node, unsigned &nodeId,
                                SmallVectorImpl<unsigned> &nodeIds) {
  ops.push_back(node);
  nodeIds.push_back(nodeId);

  // Process the immediate terminators and then the nested nodes. This order has
  // to be mirrored in the rewrite walk.
  for (Region &region : node->getRegions()) {
    for (Block &block : region) {
      auto terminator = dyn_cast<ControlFlowTerminator>(block.getTerminator());
      if (!terminator || terminator->hasTrait<mlir::OpTrait::ReturnLike>())
        continue;

      std::optional<unsigned> nodeId;
      for (unsigned id : llvm::reverse(nodeIds)) {
        if (terminator.isParentNode(ops[id])) {
          nodeId = id;
          break;
        }
      }
      assert(nodeId);

      SmallVector<ControlFlowTarget, 1> branchTargets;
      terminator.getBranchTargets(
          SmallVector<Attribute>(terminator->getNumOperands()), branchTargets);
      targets.emplace_back(*nodeId, std::move(branchTargets));
    }
  }
  for (Region &region : node->getRegions()) {
    for (Operation &op : region.getOps()) {
      if (!isa<ControlFlowNode>(op))
        continue;
      ++nodeId;
      buildTree(cast<ControlFlowNode>(op), nodeId, nodeIds);
    }
  }

  nodeIds.pop_back();
}
