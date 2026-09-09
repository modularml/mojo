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

#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/ToolCommon/CLOptions.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Target/TargetLowering.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/MathExtras.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_KGENVERIFIERPASS
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

/// True for first-party + target specific dialects; others may be mid-lowering.
static bool shouldVerifyOp(Operation *op) {
  Dialect *dialect = op->getDialect();
  if (!dialect)
    return false;
  // Curated subset of registerAllKGENDialects(); keep in sync.
  return llvm::StringSwitch<bool>(dialect->getNamespace())
      .Case("kgen", true)
      .Case("pop", true)
      .Case("lit", true)
      .Case("hlcf", true)
      .Case("co", true)
      .Case("interp", true)
      .Case("M", true)
      .Case("debuginfo", true)
      .Case("nvvm", true)
      .Case("rocdl", true)
      .Default(false);
}

/// Reject SIMD types whose resolved length cannot be lowered: non-positive
/// lengths, lengths which aren't a power of two, and lengths beyond LLVM
/// SelectionDAG's 2^15 operand limit (MOCO-1388).
/// FIXME: These should be correct on construction from the Mojo code itself,
/// but as per MOCO-2839 we're (often) silently dropping assertions during the
/// folding of @always_inline("builtin") functions. This is a clumsy fallback
/// but a good backstop to ensure we don't generate invalid code.
static LogicalResult verifySIMDLengths(Operation *op,
                                       DenseSet<Type> &checkedTypes) {
  constexpr int64_t maxSIMDLength = 1LL << 15;
  SIMDType invalid;
  auto checkType = [&](Type type) {
    if (!checkedTypes.insert(type).second)
      return;
    type.walk([&](SIMDType simd) {
      std::optional<int64_t> size = simd.getResolvedSize();
      if (size &&
          (*size <= 0 || *size > maxSIMDLength || !llvm::isPowerOf2_64(*size)))
        invalid = simd;
    });
  };

  for (Type type : op->getOperandTypes())
    checkType(type);
  for (Type type : op->getResultTypes())
    checkType(type);
  for (Region &region : op->getRegions())
    for (Block &block : region.getBlocks())
      for (BlockArgument arg : block.getArguments())
        checkType(arg.getType());

  if (!invalid)
    return success();
  return op->emitError("SIMD vector length must be a power of two between 1 "
                       "and 2^15, found ")
         << invalid;
}

namespace {
struct KGENVerifierPass : public impl::KGENVerifierPassBase<KGENVerifierPass> {
  using KGENVerifierPassBase::KGENVerifierPassBase;

  void runOnOperation() override {
    Operation *op = getOperation();

    size_t numErrors = 0;
    const size_t maxErrors = *KGENPassCLOptions::kgenVerifierMaxErrors();

    // Per-op (non-recursive): skips off-allowlist ops; structural/dominance
    // checks come from the PM verifier (off in MODULAR_PRODUCTION).
    // SIMD lengths are checked on every op regardless of dialect, since
    // mid-lowering ops can still carry KGEN types.
    DenseSet<Type> checkedTypes;
    if (op->walk([&](Operation *operation) {
            if (failed(verifySIMDLengths(operation, checkedTypes)))
              ++numErrors;
            if (shouldVerifyOp(operation) &&
                failed(mlir::verify(operation, /*verifyRecursively=*/false)))
              ++numErrors;
            if (numErrors >= maxErrors)
              return WalkResult::interrupt();
            return WalkResult::advance();
          }).wasInterrupted()) {
      signalPassFailure();
    }

    TargetInfoAttr target = lookupTargetInfo(op);
    const TargetLowering *lowering = nullptr;
    if (target) {
      ErrorOr<const TargetLowering *> loweringOr =
          TargetLoweringRegistry::get().lookup(target.getTriple());
      lowering = loweringOr.isError() ? nullptr : *loweringOr;
    }
    if (useMLIRVerifierOnly || !lowering || !lowering->needsVerification()) {
      if (numErrors > 0)
        signalPassFailure();
      return;
    }

    // Target-specific verification.
    if (op->walk([&](Operation *operation) {
            if (failed(lowering->verifyOp(operation)))
              ++numErrors;
            if (numErrors >= maxErrors)
              return WalkResult::interrupt();
            return WalkResult::advance();
          }).wasInterrupted()) {
      signalPassFailure();
    }
    if (numErrors > 0)
      signalPassFailure();
  }
};
} // namespace
