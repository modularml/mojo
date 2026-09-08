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

#ifndef KGEN_COMPILER_LLVMIR_TRANSFORMS_LLVMIRDOWNGRADEPASS_H
#define KGEN_COMPILER_LLVMIR_TRANSFORMS_LLVMIRDOWNGRADEPASS_H

#include "llvm/IR/PassManager.h"

namespace llvm {
class Module;
class ModulePass;
} // namespace llvm

namespace M::KGEN {

/// New pass manager pass that transforms LLVM IR for backend compilation
// that takes older version of LLVM IR.
///
/// This pass:
/// - Transforms llvm.lifetime related intrinsics
///
class LLVMIRDowngradePass
    : public llvm::RequiredPassInfoMixin<LLVMIRDowngradePass> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
};

} // namespace M::KGEN

#endif // KGEN_COMPILER_LLVMIR_TRANSFORMS_LLVMIRDOWNGRADEPASS_H
