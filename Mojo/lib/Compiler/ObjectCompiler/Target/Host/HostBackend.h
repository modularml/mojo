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
// TargetBackend for host (CPU) targets: llc to an object, then link a shared
// object. The default backend for CPU triples, which `HostTraits::matches`
// enumerates, and the emission base other backends reuse when their object
// lowering matches the host path.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_COMPILER_TARGET_HOST_HOSTBACKEND_H
#define KGEN_COMPILER_TARGET_HOST_HOSTBACKEND_H

#include "Mojo/Compiler/Target/TargetBackend.h"

namespace M::KGEN {

class HostBackend : public TargetBackend {
public:
  const TargetTraits *traits() const override;

  SplitStrategy
  splitStrategy(const CompilationOptions &options) const override {
    return options.enableLLVMPerFunctionSplitting ? SplitStrategy::PerFunction
                                                  : SplitStrategy::PerExported;
  }

  ErrorOr<BufferRef> emitAssembly(llvm::Module &module,
                                  EmitContext &ctx) const override;
  ErrorOr<BufferRef> emitObject(llvm::Module &module,
                                EmitContext &ctx) const override;
  ErrorOr<BufferRef> createArchive(llvm::MutableArrayRef<BufferRef> objects,
                                   llvm::StringRef moduleName,
                                   EmitContext &ctx) const override;

protected:
  // bast target that is always registered
  bool isBaseTarget() const override { return true; }
};

} // namespace M::KGEN

#endif // KGEN_COMPILER_TARGET_HOST_HOSTBACKEND_H
