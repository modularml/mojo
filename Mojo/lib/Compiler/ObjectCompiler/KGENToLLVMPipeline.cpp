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

#include "KGENToLLVMPipeline.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Support/DebugInfoDialect/DebugInfoToLLVM/DebugInfoToLLVM.h"
#include "Support/DebugInfoDialect/Transforms/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace M;

void KGEN::buildLowerToLLVMPipeline(mlir::OpPassManager &pm,
                                    const LowerToLLVMOptions &options) {
  using mlir::LLVM::LLVMFuncOp;

  // Run all LLVM lowering passes.
  pm.addPass(createLowerKGENToLLVM(LowerKGENToLLVMOptions{
      options.globalCtorFnName, options.globalDtorFnName}));
  pm.addNestedPass<LLVMFuncOp>(createKGENVerifierPass());
  pm.addPass(createLowerRuntimeClosures());
  pm.addNestedPass<LLVMFuncOp>(createLegalizePOPOperations());
  pm.addPass(createLowerGlobalPOPToLLVM());
  pm.addNestedPass<LLVMFuncOp>(createLowerPOPToLLVM());
  pm.addNestedPass<LLVMFuncOp>(createLowerControlFlow());
  pm.addNestedPass<LLVMFuncOp>(mlir::createReconcileUnrealizedCastsPass());
  pm.addNestedPass<LLVMFuncOp>(createLowerSuspensionPoints());

  // And finally canonicalize again.
  // FIXME(#25742): The MLIR region simplifier has exponential behaviour.
  mlir::GreedyRewriteConfig config;
  config.setRegionSimplificationLevel(
      mlir::GreedySimplifyRegionLevel::Disabled);
  pm.addNestedPass<LLVMFuncOp>(mlir::createCanonicalizerPass(config));
  pm.addNestedPass<LLVMFuncOp>(mlir::createCSEPass());

  // Let each target contribute late MLIR passes on the LLVM-dialect IR before
  // translation to LLVM IR, dispatched through the target lowering.
  pm.addPass(createTargetSpecificLLVMLowering());

  // Run the LLVM lowering for debug info last.
  pm.addPass(DebugInfo::createDebugInfoToLLVM(
      {/*tradeoffPerfForVariableDI=*/options.optimizationLevel == 0}));
}

void KGEN::registerLowerToLLVMPipeline() {
  mlir::PassPipelineRegistration<LowerToLLVMOptions>(
      "lower-to-llvm", "Lower KGEN IR to LLVM IR.",
      static_cast<void (*)(mlir::OpPassManager &, const LowerToLLVMOptions &)>(
          buildLowerToLLVMPipeline));
}
