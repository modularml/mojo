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

#ifndef KGEN_COMPILER_LLVMIR_TRANSFORMS_SETFUNCTIONATTRIBUTES_H
#define KGEN_COMPILER_LLVMIR_TRANSFORMS_SETFUNCTIONATTRIBUTES_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {
class Module;
} // namespace llvm

namespace M::KGEN {

/// Pass to set some function attributes that are needed for compilation.
class SetFunctionAttributes
    : public llvm::RequiredPassInfoMixin<SetFunctionAttributes> {
public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);

  static llvm::StringRef name() { return "KGEN::SetFunctionAttributes"; }

private:
  void
  runImpl(llvm::Module &M,
          const llvm::DenseMap<llvm::StringRef, llvm::cl::Option *> &options);
};

} // namespace M::KGEN

#endif // KGEN_COMPILER_LLVMIR_TRANSFORMS_SETFUNCTIONATTRIBUTES_H
