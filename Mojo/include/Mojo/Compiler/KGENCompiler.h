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

#ifndef KGEN_COMPILER_KGENCOMPILER_H
#define KGEN_COMPILER_KGENCOMPILER_H

#include "Cache/CachedTransform.h"
#include "Mojo/ExecutionEngine/ExecutionEngine.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Mojo/ToolCommon/PassManagerConfigOptions.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

namespace M::KGEN {

class KGENCompiler {
public:
  KGENCompiler(
      MLIRContext &context, CompilationOptions options,
      PassManagerConfigOptions pmConfigOptions = PassManagerConfigOptions());

  /// Run KGEN compilation pipeline, including pre-elaboration passes,
  /// elaboration, and post-elaboration pass. Get the theModule ready before
  /// llvm lowering.
  ErrorOrSuccess runKGENPipeline(ModuleOp theModule, TargetInfoAttr target,
                                 RCRef<Cache::TransformCache> transformCache,
                                 AnyAsyncValueRef chain);

  ErrorOrSuccess runKGENPipeline(ModuleOp theModule, TargetInfoAttr target);

  /// Run the library generation pipeline on the given module. If
  /// `materializeDependencies` is true, the pipeline will ensure all
  /// dependencies are materialized in the final module.
  ErrorOrSuccess runGenerateLibraryPipeline(ModuleOp module);

  /// Run post-parser pipeline that checks and lowers source-level
  /// LIT constructs.
  LogicalResult runCheckLITPipeline(ModuleOp module);

  /// Run the elaboration and post-elaboration pipeline
  /// This doesn't not include check LIT and pre-elaboration passes.
  /// This allows the transform to be cached if chain is provided.
  ErrorOrSuccess runElaborationPipeline(
      ModuleOp module, TargetInfoAttr target, AsyncRT::CPUDevice &cpuDevice,
      std::optional<AnyAsyncValueRef> chain = std::nullopt,
      std::function<void(Operation *)> moreOnMiss = [](Operation *) {},
      std::function<void(Operation *)> moreOnHit = [](Operation *) {});

private:
  /// Compilation options.
  CompilationOptions options;

  /// PassManager configuration options.
  PassManagerConfigOptions pmConfigOptions;

  MLIRContext &context;
};

//===----------------------------------------------------------------------===//
// Default JIT Configuration
//===----------------------------------------------------------------------===//

/// Sets up an ExecutionEngine instance for compiling Mojo. It handles
/// initializing the target machine, the cache backends, and the execution
/// engine itself. On success, the execution engine is returned.
ErrorOr<std::unique_ptr<ExecutionEngine>> initializeExecutionEngine(
    MLIRContext &context, const KGEN::CompilationOptions &compilationOptions,
    ExecutionEngineOptions executionEngineOptions, bool isJIT,
    PassManagerConfigOptions pmOptions = PassManagerConfigOptions());

/// Create an instance of the elaborator pass using the given configuration.
/// The created elaborator pass uses a default specialization executor that
/// JITs and executes in-process.
std::unique_ptr<Pass> createElaborateGeneratorsWithDefaultJIT();
std::unique_ptr<Pass>
createElaborateGeneratorsWithDefaultJIT(const std::string &name);

} // namespace M::KGEN

#endif // KGEN_COMPILER_KGENCOMPILER_H
