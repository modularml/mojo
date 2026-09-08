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

#include "Mojo/HLCFDialect/HLCFUtils.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"

using namespace M;
using namespace HLCF;

/// Return true if the operation is a loop and has a matching label.
bool HLCF::isMatchingLoop(Operation *op, StringAttr label) {
  if (auto loop = dyn_cast<LoopOp>(op))
    return !label || loop.getLabelAttr() == label;
  return false;
}

/// Return the nearest enclosing matching loop or nullptr if nothing found.
LoopOp HLCF::getParentLoop(Operation *op, StringAttr label) {
  LoopOp loop = op->getParentOfType<LoopOp>();
  while (!isMatchingLoop(loop, label))
    loop = loop->getParentOfType<LoopOp>();
  return loop;
}

/// Check if the child loop is nested in the parentToCheck loop.
bool HLCF::isParentLoop(LoopOp child, LoopOp parentToCheck) {
  LoopOp parent = child;
  while (parent && parent != parentToCheck)
    parent = parent->getParentOfType<LoopOp>();
  return parent == parentToCheck;
}

/// Get the parent operation of a terminator.
Operation *HLCF::getParentNode(HLCF::ControlFlowTerminator term) {
  Operation *op = term->getParentOp();
  while (!term.isParentNode(op))
    op = op->getParentOp();
  return op;
}

HLCF::IfOp HLCF::replaceElifWithIfOps(ElifOp elifOp) {
  ImplicitLocOpBuilder builder(elifOp->getLoc(), elifOp);
  builder.setInsertionPoint(elifOp);

  // First condition is an SSA operand; build the outermost if from it.
  HLCF::IfOp outerMostIfOp =
      HLCF::IfOp::create(builder, elifOp.getResultTypes(), elifOp.getCond());
  outerMostIfOp.getThenRegion().takeBody(elifOp.getThenRegion());
  Region *currentRegion = &outerMostIfOp.getElseRegion();

  // Nest additional (cond, then) pairs into the current else region.
  for (Region &region : elifOp.getElifRegions()) {
    currentRegion->takeBody(region);
    builder.setInsertionPointToEnd(&currentRegion->front());
    Operation *terminator = currentRegion->front().getTerminator();
    if (auto elifYieldOp = dyn_cast<HLCF::ElifYieldOp>(terminator)) {
      auto newIfOp = HLCF::IfOp::create(builder, elifOp.getResultTypes(),
                                        elifYieldOp->getOperand(0));
      IRRewriter rewriter{builder};
      rewriter.replaceOp(elifYieldOp,
                         HLCF::YieldOp::create(builder, newIfOp.getResults()));
      currentRegion = &newIfOp.getThenRegion();
      continue;
    }
    // Moved a then region into If's Then region; continue into its Else.
    auto ifOpParent = terminator->getParentOfType<HLCF::IfOp>();
    currentRegion = &ifOpParent.getElseRegion();
  }
  currentRegion->takeBody(elifOp.getElseRegion());
  builder.setInsertionPoint(elifOp);
  IRRewriter rewriter{builder};
  rewriter.replaceOp(elifOp, outerMostIfOp);
  return outerMostIfOp;
}
