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
#include "SetFunctionAttributes.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/TypedPointerType.h"

using namespace llvm;
using namespace llvm::codegen;
using namespace M::KGEN;

//===----------------------------------------------------------------------===//
// SetFunctionAttributes
//===----------------------------------------------------------------------===//

PreservedAnalyses SetFunctionAttributes::run(Module &module,
                                             ModuleAnalysisManager &MAM) {

  llvm::DenseMap<llvm::StringRef, llvm::cl::Option *> &options =
      llvm::cl::getRegisteredOptions();

  runImpl(module, options);
  return PreservedAnalyses::none();
}

static std::optional<DenormalMode::DenormalModeKind> getDenormalKind(
    const llvm::DenseMap<llvm::StringRef, llvm::cl::Option *> &options) {

  auto denormIter = options.find("denormal-fp-math-f32");
  if (denormIter == options.end())
    return std::nullopt;

  auto *denormalIntVal = static_cast<llvm::cl::opt<int> *>(denormIter->second);
  if (!denormalIntVal || denormalIntVal->getNumOccurrences() == 0)
    return std::nullopt;

  return (DenormalMode::DenormalModeKind)denormalIntVal->getValue();
}

void SetFunctionAttributes::runImpl(
    llvm::Module &module,
    const llvm::DenseMap<llvm::StringRef, llvm::cl::Option *> &options) {
  // Set function denormal-fp-math-f32 attributes based on cl option value.
  // Clang does similar thing for `-fdenormal-fp-math-f32`
  // https://github.com/llvm/llvm-project/blob/main/clang/lib/CodeGen/CGCall.cpp#L1942-L1948
  std::optional<DenormalMode::DenormalModeKind> denormalKind =
      getDenormalKind(options);

  if (denormalKind.has_value()) {
    DenormalMode f32Mode(*denormalKind, *denormalKind);

    for (Function &func : module) {
      DenormalFPEnv existing = func.getDenormalFPEnv();
      // Skip if the f32 mode already matches what we want to set.
      if (existing.F32Mode == f32Mode)
        continue;
      // Override the f32 mode while preserving the existing default mode.
      DenormalFPEnv fpEnv(existing.DefaultMode, f32Mode);
      AttrBuilder attrs(func.getContext());
      attrs.addDenormalFPEnvAttr(fpEnv);
      func.addFnAttrs(attrs);
    }
  }
}
