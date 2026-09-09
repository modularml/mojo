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
#include "Mojo/ToolCommon/CLOptions.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Mojo/TransformUtils/CallGraphUtils.h"
#include "Mojo/TransformUtils/SCCUtils.h"
#include "Target/TargetLowering.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"

#define DEBUG_TYPE "kgen-infer-function-attrs"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// InferFunctionAttrsPass
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_INFERFUNCTIONATTRS
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct FunctionAttrs {
  bool isConvergent = false;
};

struct CallGraphNode
    : public SCCNode<CallGraphNode, FuncOp, KGENCallOpInterface> {
  using SCCNode::SCCNode;

  FunctionAttrs attrs;
};

struct CallGraph : public SCCGraph<CallGraph, CallGraphNode> {
  CallGraph(const SymbolTable &symtab, const TargetLowering *lowering)
      : symtab(symtab), lowering(lowering) {}

  /// Analyze function body to determine which attributes need to be added or
  /// removed.
  bool doAnalysis(CallGraphNode *node);

  /// Propagate all attributes that need to be added or removed to a function.
  void doRewrite(const CallGraphNode *node);

  /// Symbol table for function lookup.
  const SymbolTable &symtab;

  /// Target lowering for the module's target, or null if none is registered.
  const TargetLowering *lowering;
};

/// Propagate attributes from function body to a function.
/// One of the crucial attribute to be propagated is `convergent` that tells
/// that function cannot be made control-dependent on any other value (see
/// https://llvm.org/docs/ConvergentOperations.html)
bool CallGraph::doAnalysis(CallGraphNode *node) {
  bool changed = false;
  FuncOp func = node->func;
  if (node->attrs.isConvergent || func.isConvergent()) {
    node->attrs.isConvergent = true;
    return false;
  }

  llvm::LLVMContext llvmCtx;
  func.walk([&](Operation *op) -> WalkResult {
    // Target-specific convergent ops (e.g. GPU barriers).
    if (lowering && lowering->isConvergentOp(op)) {
      node->attrs.isConvergent = true;
      changed = true;
      return WalkResult::interrupt();
    }

    // Propagate `convergent` attribute from intrinsic.
    if (auto intrinsic = dyn_cast<POP::CallLLVMIntrinsicOp>(op)) {
      auto intrinsicName = cast<StringAttr>(intrinsic.getIntrin());
      llvm::Intrinsic::ID intrinsicID =
          llvm::Intrinsic::lookupIntrinsicID(intrinsicName.getValue());
      llvm::AttributeSet attrSet =
          llvm::Intrinsic::getFnAttributes(llvmCtx, intrinsicID);
      // Check if intrinsic is convergent
      if (attrSet.hasAttribute(llvm::Attribute::Convergent)) {
        node->attrs.isConvergent = true;
        changed = true;
        return WalkResult::interrupt();
      }
    }

    // Propagate `convergent` attribute from callee.
    if (auto call = dyn_cast<KGEN::CallOp>(op)) {
      auto callee = symtab.lookup<FuncOp>(call.getCalleeSymbol().getAttr());
      const CallGraphNode &calleeNode = nodes.find(callee)->second;
      if (calleeNode.attrs.isConvergent) {
        node->attrs.isConvergent = true;
        changed = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return changed;
}

/// Propagate all attributes that need to be added or removed to a function.
void CallGraph::doRewrite(const CallGraphNode *node) {
  FuncOp func = node->func;
  if (node->attrs.isConvergent)
    func.setConvergent(true);
}

struct InferFunctionAttrsPass
    : impl::InferFunctionAttrsBase<InferFunctionAttrsPass> {
  void runOnOperation() override;
};

void InferFunctionAttrsPass::runOnOperation() {
  AsyncRT::CPUDevice &cpuDevice =
      *loadContext(&getContext())->get<AsyncRT::CPUDevice>();
  const SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
  TargetInfoAttr target = lookupTargetInfo(getOperation());
  const TargetLowering *lowering = nullptr;
  if (target) {
    ErrorOr<const TargetLowering *> loweringOr =
        TargetLoweringRegistry::get().lookup(target.getTriple());
    lowering = loweringOr.isError() ? nullptr : *loweringOr;
  }
  CallGraph cg(symtab, lowering);
  cg.build(getOperation(), symtab);
  cg.run(cpuDevice);
}

} // namespace
