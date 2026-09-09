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

#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace M;
using namespace HLCF;

//===----------------------------------------------------------------------===//
// Control-Flow Verification
//===----------------------------------------------------------------------===//

namespace {
/// This object contains context about control-flow trees and uses that to
/// verify them. Keep a stack of control-flow scopes as we walk the tree and
/// verify that each terminator has a valid parent somewhere up the stack and
/// check that the return types match.
class ControlFlowVerifier {
public:
  explicit ControlFlowVerifier(KGEN::FunctionLike root) : root(root) {}

  /// Verify a node.
  LogicalResult verifyNode(ControlFlowNode op);

private:
  /// Verify a terminator.
  LogicalResult verifyTerminator(ControlFlowTerminator op);

  /// Return the nearest operation that is a valid parent for the terminator.
  ControlFlowNode findNearestParentFor(ControlFlowTerminator op);

  /// The root of the control-flow tree.
  KGEN::FunctionLike root;

  /// The current stack of control-flow scopes.
  SmallVector<ControlFlowNode> scopes;
};
} // namespace

/// Verify two type ranges match along a control-flow edge.
static LogicalResult verifyTypesAlongEdge(TypeRange lhs, TypeRange rhs,
                                          Operation *op, Operation *parent,
                                          ControlFlowTarget target) {
  auto attachEdgeNote = [&](InFlightDiagnostic diag) {
    diag << " along control-flow edge from here";
    if (target.index)
      diag.attachNote(parent->getRegion(*target.index).front().front().getLoc())
          << "to beginning of region #" << *target.index << " here";
    else
      diag.attachNote(parent->getLoc()) << "to end of parent operation here";
    return failure();
  };

  if (lhs.size() != rhs.size()) {
    return attachEdgeNote(op->emitOpError("specifies ")
                          << lhs.size() << " branch inputs but target expected "
                          << rhs.size());
  }
  for (auto [idx, lhsType, rhsType] :
       llvm::zip(llvm::seq<unsigned>(0, lhs.size()), lhs, rhs)) {
    if (lhsType == rhsType)
      continue;
    return attachEdgeNote(op->emitOpError("branch input #")
                          << idx << " has type " << lhsType
                          << " but target expected " << rhsType);
  }
  return success();
}

ControlFlowNode
ControlFlowVerifier::findNearestParentFor(ControlFlowTerminator op) {
  for (ControlFlowNode node : llvm::reverse(scopes))
    if (op.isParentNode(node))
      return node;
  return nullptr;
}

LogicalResult ControlFlowVerifier::verifyTerminator(ControlFlowTerminator op) {
  // Returns are modeled differently. Handle them here.
  if (op->hasTrait<OpTrait::ReturnLike>())
    return success();

  ControlFlowNode parent = findNearestParentFor(op);
  if (!parent) {
    return op->emitOpError("is not nested within a suitable parent operation")
               .attachNote(root->getLoc())
           << "see control-flow root here";
  }
  SmallVector<ControlFlowTarget, 1> targets;
  op.getBranchTargets(SmallVector<Attribute>(op->getNumOperands(), {}),
                      targets);

  for (const ControlFlowTarget &target : targets) {
    ValueRange args = parent.getEntryArguments(target.index);
    if (failed(verifyTypesAlongEdge(target.inputs.getTypes(), args.getTypes(),
                                    op, parent, target)))
      return failure();
  }
  return success();
}

LogicalResult ControlFlowVerifier::verifyNode(ControlFlowNode op) {
  scopes.push_back(op);

  SmallVector<ControlFlowTarget, 2> targets;
  op.getEntryTargets(SmallVector<Attribute>(op->getNumOperands(), {}), targets);

  for (auto [target, inputs] : targets) {
    ValueRange args = op.getEntryArguments(target);
    if (failed(verifyTypesAlongEdge(inputs.getTypes(), args.getTypes(), op, op,
                                    target)))
      return failure();
  }

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      if (block.empty() || !block.back().hasTrait<OpTrait::IsTerminator>())
        return success(); // another trait will emit an error
      if (auto terminator =
              dyn_cast<ControlFlowTerminator>(block.getTerminator()))
        if (failed(verifyTerminator(terminator)))
          return failure();
      for (Operation &op : block.without_terminator())
        if (auto node = dyn_cast<ControlFlowNode>(&op))
          if (failed(verifyNode(node)))
            return failure();
    }
  }

  scopes.pop_back();
  return success();
}

//===----------------------------------------------------------------------===//
// Interface Verifiers
//===----------------------------------------------------------------------===//

LogicalResult HLCF::verifyControlFlowNode(ControlFlowNode op) {
  // Verify that there are no empty regions or blocks.
  for (Region &region : op->getRegions()) {
    if (region.empty())
      return op->emitOpError("unexpected empty region #")
             << region.getRegionNumber();
    for (Block &block : region)
      if (block.empty())
        return op->emitOpError("unexpected empty block");
  }
  return success();
}

LogicalResult HLCF::verifyControlFlowTerminator(ControlFlowTerminator op) {
  // Verify that the terminator's parent is an HLCF operation.
  Operation *parent = op->getParentOp();
  if (isa<ControlFlowNode>(parent))
    return success();

  // Special case `kgen.return` and `kgen.unreachable`. These are the only
  // terminators allowed for a function-like.
  if ((op->hasTrait<OpTrait::ReturnLike>() || isa<KGEN::UnreachableOp>(op)) &&
      isa<KGEN::FunctionLike>(parent))
    return success();

  return (op->emitOpError("expected parent operation to be a control-flow "
                          "operation but got '")
          << parent->getName() << "'")
             .attachNote(parent->getLoc())
         << "see invalid parent here";
}

LogicalResult HLCF::verifyControlFlow(KGEN::FunctionLike root) {
  ControlFlowVerifier verifier(root);
  for (auto node : root.getBodyRegion().getOps<ControlFlowNode>())
    if (failed(verifier.verifyNode(node)))
      return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Mojo/HLCFDialect/HLCFInterfaces.cpp.inc"
