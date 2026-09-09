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

#include "TransformUtils.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace M::KGEN;
using namespace llvm;

/// Update all functions and their callsites that were previously identified.
void CallGraphUpdater::update() {
  assert(!functionsToUpdate.empty() &&
         "No functions to update. Run analyze() first ?.");

  DenseMap<llvm::Function *, llvm::Function *> funcMap;

  for (llvm::Function *f : functionsToUpdate) {
    llvm::Function *newF = updateFunction(*f);
    funcMap[f] = newF;
  }

  // Replace call sites with calls that include thread parameters
  for (llvm::Function *func : functionsToUpdate) {
    llvm::Function *newFunc = funcMap[func];
    SmallVector<llvm::CallInst *> calls;
    for (llvm::BasicBlock &bb : *newFunc) {
      for (llvm::Instruction &inst : bb) {
        if (llvm::CallInst *call = llvm::dyn_cast<llvm::CallInst>(&inst)) {
          llvm::Function *callee = call->getCalledFunction();
          // Type-mismatched calls use a bitcast wrapper; strip it.
          if (!callee) {
            callee = llvm::dyn_cast<llvm::Function>(
                call->getCalledOperand()->stripPointerCasts());
          }
          if (!callee || !functionsToUpdate.contains(callee))
            continue;
          calls.push_back(call);
        }
      }
    }

    llvm::LLVMContext &ctx = module.getContext();
    llvm::IRBuilder<> builder(ctx);
    for (llvm::CallInst *call : calls) {
      llvm::Function *resolvedCallee = call->getCalledFunction();
      if (!resolvedCallee) {
        resolvedCallee = llvm::dyn_cast<llvm::Function>(
            call->getCalledOperand()->stripPointerCasts());
      }
      assert(resolvedCallee && funcMap.count(resolvedCallee) &&
             "resolved callee not in funcMap; missing from functionsToUpdate?");
      llvm::Value *newCall =
          updateCall(*call, *funcMap[resolvedCallee], *newFunc);

      if (newCall != call) {
        builder.SetInsertPoint(call);
        builder.Insert(newCall);
        if (call->getType() == cast<llvm::Value>(newCall)->getType()) {
          call->replaceAllUsesWith(newCall);
        } else {
          // Return type changed; RAUW asserts equal types so replace uses via
          // Use::set() which bypasses the type check. Users are expected to
          // either be empty (result discarded) or already rebuilt by
          // rewriteFunctionBody with the updated type.
          SmallVector<Use *> uses;
          for (Use &u : call->uses())
            uses.push_back(&u);
          for (Use *u : uses)
            u->set(newCall);
        }
        call->eraseFromParent();
      }
    }
  }
  // Replace all uses of old functions with new functions and restore the
  // original name
  for (auto [func, newFunc] : funcMap) {
    if (newFunc == func)
      continue;
    func->replaceAllUsesWith(newFunc);
    newFunc->takeName(func);
    func->eraseFromParent();
  }
  // Clear all cached analyses to force their recomputation later.
  mam.clear();
}
