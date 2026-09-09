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
// LLVM IR Downgrade Pass - Transform LLVM IR for backend compilation
// that takes older version of LLVM IR.
//
//===----------------------------------------------------------------------===//

#include "LLVMIRDowngradePass.h"
#include "TransformUtils.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
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
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define KGEN_DEBUG_TYPE "llvm-ir-downgrade-pass"

using namespace llvm;
using namespace M::KGEN;

static bool isLifetimeIntrinsic(Function &f) {
  SmallSet<StringRef, 2> lifetimeIntrinsicNames{"llvm.lifetime.end",
                                                "llvm.lifetime.start"};
  for (auto name : lifetimeIntrinsicNames) {
    if (f.getName().contains(name))
      return true;
  }
  return false;
}
namespace {

/// ----------------------------------------------------------
/// Downgrade Lifetime intrinsics to older version:
/// From:
/// declare void @llvm.lifetime.end.p0(ptr captures(none))
/// to (add i64 value which is the size of the memory ptr is pointing):
/// declare void @llvm.lifetime.end.p0(i64, ptr captures(none))
/// and
/// From:
/// declare void @llvm.lifetime.start.p0(ptr captures(none))
/// to (add i64 value which is the size of the memory ptr is pointing):
/// declare void @llvm.lifetime.start.p0(i64, ptr captures(none))
///
/// Newer LLVM IR doesn't require this value anymore since it can be
/// inferred from the alloca size where ptr is assigned.
/// https://github.com/llvm/llvm-project/pull/149310
/// The updater here also update the callsite of the intrinsics to
/// with the alloca size.

class LifetimeIntrinsicUpdater : public CallGraphUpdater {
  const DataLayout &dataLayout;
  llvm::LLVMContext &ctx;

public:
  explicit LifetimeIntrinsicUpdater(llvm::Module &module,
                                    llvm::ModuleAnalysisManager &mam)
      : CallGraphUpdater(module, mam), dataLayout(module.getDataLayout()),
        ctx(module.getContext()) {}

  /// Analyze entire call graph to find functions that requires update
  virtual bool analyze() final {
    for (llvm::Function &f : module) {
      if (!f.isDeclaration())
        continue;

      if (isLifetimeIntrinsic(f))
        functionsToUpdate.insert(&f);
    }

    auto hasLifetimeMarker = [&](llvm::Function &f) {
      for (llvm::BasicBlock &bb : f) {
        for (llvm::Instruction &inst : bb) {
          if (llvm::CallInst *callInst =
                  llvm::dyn_cast<llvm::CallInst>(&inst)) {
            llvm::Function *callee = callInst->getCalledFunction();
            if (callee && (functionsToUpdate.contains(callee) ||
                           isLifetimeIntrinsic(*callee))) {
              return true;
            }
          }
        }
      }
      return false;
    };

    SmallVector<llvm::Function *, 8> worklist;
    for (llvm::Function &f : module) {
      if (isLifetimeIntrinsic(f) || hasLifetimeMarker(f)) {
        functionsToUpdate.insert(&f);
        worklist.push_back(&f);
      }
    }

    return !functionsToUpdate.empty();
  }

  virtual llvm::Value *updateCall(llvm::CallInst &call, llvm::Function &newFunc,
                                  llvm::Function &callerFunc) final {
    if (!isLifetimeIntrinsic(*call.getCalledFunction()))
      return &call;

    Value *ptr = call.args().begin()->get();
    if (auto alloca = dyn_cast<llvm::AllocaInst>(ptr)) {
      SmallVector<llvm::Value *> args;
      // Infer the ptr memory size from alloca.
      args.push_back(ConstantInt::get(Type::getInt64Ty(ctx),
                                      *alloca->getAllocationSize(dataLayout)));

      for (auto &arg : call.args())
        args.push_back(arg);
      return CallInst::Create(&newFunc, args);
    }

    return &call;
  }

  /// Create a new function with the "i64 size" parameter.
  /// func function into it.
  virtual llvm::Function *updateFunction(llvm::Function &func) final {
    if (!isLifetimeIntrinsic(func))
      return &func;

    FunctionType *oldFT = func.getFunctionType();
    SmallVector<Type *> newParams;
    newParams.push_back(IntegerType::get(ctx, 64));
    newParams.append(oldFT->param_begin(), oldFT->param_end());
    FunctionType *newFT =
        FunctionType::get(oldFT->getReturnType(), newParams, oldFT->isVarArg());

    // Store the original name before creating new function
    std::string originalName = func.getName().str();

    // Create new function with void return type
    Function *newFunc = Function::Create(newFT, func.getLinkage(),
                                         originalName + ".temp", &module);

    // Update the arg attribute.
    AttributeSet originalAttr = func.getAttributes().getParamAttrs(0);
    newFunc->setAttributes(func.getAttributes().removeParamAttributes(ctx, 0));

    AttrBuilder builder(ctx, originalAttr);
    newFunc->addParamAttrs(1, builder);

    return newFunc;
  }
};
} // namespace

static void downgradeLifetimeIntrinsics(Module &module,
                                        llvm::ModuleAnalysisManager &mam) {

  LifetimeIntrinsicUpdater updater(module, mam);
  if (updater.analyze()) {
    updater.update();
  }
}

namespace M::KGEN {

// Implementation of MetalAIRPass::run
PreservedAnalyses LLVMIRDowngradePass::run(Module &module,
                                           ModuleAnalysisManager &mam) {

  downgradeLifetimeIntrinsics(module, mam);
  return PreservedAnalyses::all();
}

} // namespace M::KGEN
