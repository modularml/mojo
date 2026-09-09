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

#include "Mojo/Compiler/KGENCompiler.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Support/DebugInfoDialect/DebugInfoToLLVM/DebugInfoToLLVM.h"
#include "Support/DebugInfoDialect/Transforms/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"

using namespace M;
using namespace KGEN;

void KGEN::registerDefaultKGENPasses(const std::string &cacheBaseExtra) {
  // Register the standard passes we want.
  mlir::registerCSEPass();
  mlir::registerCanonicalizerPass();
  mlir::registerConvertIndexToLLVMPass();
  mlir::registerReconcileUnrealizedCastsPass();
  mlir::registerPrintOpStatsPass();
  mlir::registerStripDebugInfoPass();

  // Register opt passes.
  KGEN::registerApplyInliner();
  KGEN::registerArgPromotion();
  KGEN::registerCanonicalizer();
  KGEN::registerCheckLifetimes();
  KGEN::registerEliminateDeadSymbols();
  KGEN::registerEliminateDuplicateFunctions();
  KGEN::registerEnsureNoParameters();
  KGEN::registerFunctionStats();
  KGEN::registerHoistTrivialInvariants();
  KGEN::registerLiftAndFoldApply();
  KGEN::registerKGENVerifierPass();
  KGEN::registerLoopUnrolling();
  KGEN::registerLowerAsyncFunctions();
  KGEN::registerLowerCallingConventions();
  KGEN::registerLowerClosures();
  KGEN::registerLowerControlFlow();
  KGEN::registerLowerGlobalPOPToLLVM();
  KGEN::registerLowerArgConventions();
  KGEN::registerLowerLoops();
  KGEN::registerLowerKGENToLLVM();
  KGEN::registerLowerLIT();
  KGEN::registerLegalizePOPOperations();
  KGEN::registerLowerPOPToLLVM();
  KGEN::registerLowerRuntimeClosures();
  KGEN::registerLowerSemanticCF();
  KGEN::registerMem2Reg();
  KGEN::registerOutlineClosures();
  KGEN::registerRaiseForLoops();
  KGEN::registerRemoveUnusedParams();
  KGEN::registerReorderParamOps();
  KGEN::registerSROA();
  KGEN::registerSetFastMathFlags();
  KGEN::registerSimplifyCF();
  KGEN::registerStackReuse();
  KGEN::registerSynthesizeDebugInfo();
  KGEN::registerVerifyParameters();
  KGEN::registerLowerSuspensionPoints();
  KGEN::registerTargetSpecificLLVMLowering();
  KGEN::registerLowerToLLVMPipeline();
  KGEN::registerIPDF();
  KGEN::registerSCCP();
  KGEN::registerStripParserMetadata();
  DebugInfo::registerDebugInfoToLLVM();
  DebugInfo::registerDebugInfoStrip();

  // Passes that require a runtime.
  mlir::registerPass([cacheBaseExtra] {
    return KGEN::createElaborateGeneratorsWithDefaultJIT(cacheBaseExtra);
  });
  KGEN::registerInlineParametric();
  KGEN::registerAutomaticInline();
  KGEN::registerDeadArgumentElimination();
  KGEN::registerResolveCompilerPromises();
  KGEN::registerUpdateMaterialization();
  KGEN::registerInferFunctionAttrs();
}
