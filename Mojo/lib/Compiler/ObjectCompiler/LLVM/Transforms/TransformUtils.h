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
//
// LLVM IR Transform Utils
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_LLVMIR_TRANSFORMS_TRANSFORMUTILS_H
#define KGEN_COMPILER_LLVMIR_TRANSFORMS_TRANSFORMUTILS_H

#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

namespace M::KGEN {

/// Base class for updating call graph with new functions that have different
/// signatures.
class CallGraphUpdater {
protected:
  llvm::Module &module;
  llvm::ModuleAnalysisManager &mam;
  llvm::LazyCallGraph &graph;
  llvm::SetVector<llvm::Function *> functionsToUpdate;

public:
  explicit CallGraphUpdater(llvm::Module &module,
                            llvm::ModuleAnalysisManager &mam)
      : module(module), mam(mam),
        graph(mam.getResult<llvm::LazyCallGraphAnalysis>(module)) {
    graph.buildRefSCCs();
  }

  virtual ~CallGraphUpdater() {}

  /// Return new function with updated body/signature.
  virtual llvm::Function *updateFunction(llvm::Function &f) = 0;

  /// Return new call instruction that corresponds to updated callee.
  virtual llvm::Value *updateCall(llvm::CallInst &call, llvm::Function &newFunc,
                                  llvm::Function &callerFunc) = 0;

  /// Analyze entire call graph to identify functions that require update.
  virtual bool analyze() = 0;

  /// Update all functions and their callsites that were previously identified.
  void update();
};

} // namespace M::KGEN

#endif // KGEN_COMPILER_LLVMIR_TRANSFORMS_TRANSFORMUTILS_H
