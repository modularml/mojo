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

#include "Target/TargetLowering.h"

#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"
#include "Target/TargetTraits.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/TargetParser/Triple.h"

namespace M::KGEN {

// Default: no target-specific debug type. Defined out-of-line so the header
// only needs a forward declaration of `DebugInfo::DIType`.
DebugInfo::DIType
TargetLowering::buildDebugTypeForDType(mlir::MLIRContext *ctx,
                                       KGENDType dtype) const {
  return {};
}

static llvm::ManagedStatic<TargetLoweringRegistry> theLoweringRegistry;

TargetLoweringRegistry &TargetLoweringRegistry::get() {
  return *theLoweringRegistry;
}

void TargetLoweringRegistry::add(std::unique_ptr<TargetLowering> lowering) {
  Targets.push_back(std::move(lowering));
}

ErrorOr<const TargetLowering *>
TargetLoweringRegistry::lookup(const llvm::Triple &triple) const {
  // A lowering that resolves `triple` to one it owns takes precedence over a
  // direct self-match.
  const TargetLowering *result = [&]() -> const TargetLowering * {
    const TargetLowering *directMatch = nullptr;
    for (const std::unique_ptr<TargetLowering> &target : Targets) {
      const TargetLowering *resolved = target->resolve(triple);
      if (!resolved)
        continue;
      if (resolved != target.get())
        return resolved;
      if (!directMatch)
        directMatch = resolved;
    }
    return directMatch;
  }();
  if (!result) {
    return Error("target '" + triple.str() +
                 "' is not supported by this build");
  }
  if (ErrorOrSuccess e = requireMaxForAccelerator(!result->isBaseTarget()))
    return Error(e.getError());
  return result;
}

} // namespace M::KGEN
