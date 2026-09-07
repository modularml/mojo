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

#ifndef KGEN_COMPILER_LLVMIR_TRANSFORMS_POINTERREWRITER_H
#define KGEN_COMPILER_LLVMIR_TRANSFORMS_POINTERREWRITER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class Value;
class Module;
class TypedPointerType;
} // namespace llvm

namespace M::KGEN {

/// Pass to rewrite opaque pointers to typed pointers that constructs a map
/// between opaque pointer and its intended type.
class PointerRewriter : public llvm::RequiredPassInfoMixin<PointerRewriter> {
public:
  using PointerTypeMap =
      llvm::DenseMap<const llvm::Value *, llvm::TypedPointerType *>;

  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);

  static llvm::StringRef name() { return "PointerRewriter"; }

  static PointerTypeMap buildPointerMap(const llvm::Module &M);

private:
  bool runImpl(llvm::Module &M);
  bool cleanupTypedPointerMetadata(llvm::Module &M);
};

} // namespace M::KGEN

#endif // KGEN_COMPILER_LLVMIR_TRANSFORMS_POINTERREWRITER_H
