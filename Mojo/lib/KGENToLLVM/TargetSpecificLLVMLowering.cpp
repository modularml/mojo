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

// Target-agnostic shim that lets each target attach additional MLIR passes to
// the LLVM-dialect lowering pipeline. The pass resolves the module's target
// lowering from `TargetLoweringRegistry` (dispatched by triple) and runs that
// target's `addPostLowerToLLVMPasses` on a nested `PassManager`. A compiler
// plugin is one such target: its late passes are contributed the same way.
//
// The pass itself does nothing target-specific: it forwards to the resolved
// `TargetLowering`, which populates a nested `PassManager`. It is a safe no-op
// on triples that no target lowering claims (or that contribute no late
// passes).

#include "Target/TargetLowering.h"

#include "Mojo/ToolCommon/KGENPasses.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/PassManager.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_TARGETSPECIFICLLVMLOWERING
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {

struct TargetSpecificLLVMLoweringPass
    : public KGEN::impl::TargetSpecificLLVMLoweringBase<
          TargetSpecificLLVMLoweringPass> {
  using TargetSpecificLLVMLoweringBase::TargetSpecificLLVMLoweringBase;

  void runOnOperation() override;
};

} // namespace

void TargetSpecificLLVMLoweringPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();

  TargetInfoAttr targetInfo = lookupTargetInfo(module);
  if (!targetInfo)
    return; // No target info — nothing to attach.

  ErrorOr<const TargetLowering *> loweringOr =
      TargetLoweringRegistry::get().lookup(targetInfo.getTriple());
  const TargetLowering *lowering = loweringOr.isError() ? nullptr : *loweringOr;
  if (!lowering)
    return;

  // Let the target contribute late passes on a nested, module-anchored pass
  // manager. Skip the run when it adds none.
  mlir::PassManager nested(&getContext(), mlir::ModuleOp::getOperationName());
  lowering->addPostLowerToLLVMPasses(nested);
  if (nested.size() == 0)
    return;

  if (failed(nested.run(module)))
    signalPassFailure();
}
