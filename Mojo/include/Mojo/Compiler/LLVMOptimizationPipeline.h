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

#ifndef KGEN_COMPILER_LLVMPASSESPIPELINE_H
#define KGEN_COMPILER_LLVMPASSESPIPELINE_H

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Target/TargetMachine.h"

namespace M::KGEN {

class CompilationOptions;

/// Build a module pass pipeline for a given set of compilation options.
llvm::ModulePassManager
buildLLVMOptimizationPipeline(llvm::PassBuilder &passBuilder,
                              const CompilationOptions &options);

/// Add LLVMIRDowngradePass to the pass manager.
void addLLVMIRDowngradePass(llvm::ModulePassManager &mpm);

/// Register all custom LLVM passes with \p passBuilder so they are available by
/// name when using PassBuilder's pipeline parsing (e.g. the -passes option).
///
/// Registers the target-agnostic kgen-llvmir-downgrade pass directly; each
/// registered backend adds its own passes via registerPipelinePasses.
void registerKGENLLVMPasses(llvm::PassBuilder &passBuilder);

} // namespace M::KGEN

#endif // KGEN_COMPILER_LLVMPASSESPIPELINE_H
