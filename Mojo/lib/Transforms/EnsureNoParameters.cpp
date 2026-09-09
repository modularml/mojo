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

#include "Mojo/KGENDialect/KGENOps.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_ENSURENOPARAMETERS
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct EnsureNoParametersPass
    : impl::EnsureNoParametersBase<EnsureNoParametersPass> {
  void runOnOperation() override;
};
} // namespace

static LogicalResult legalizeOp(Operation *op) {
  mlir::AttrTypeWalker legalizer;
  bool isFailure = false;

  // FuncTypeGeneratorType must not be parameterized anymore.
  legalizer.addWalk([&](Type type) {
    if (auto sig = dyn_cast<FuncTypeGeneratorType>(type)) {
      if (!sig.getInputParamTypes().empty()) {
        mlir::emitError(op->getLoc(),
                        "parameterized functions cannot be used at runtime");
        isFailure = true;
        return WalkResult::skip();
      }
    }
    return WalkResult::advance();
  });

  // Parameter references should not exist anymore.
  legalizer.addWalk([&](ParamDeclRefAttr attr) {
    op->emitError("dangling parameter reference post-elaboration: ") << attr;
    isFailure = true;
    return WalkResult::skip();
  });

  if (isa<ParamConstantOp, ParamMaterializeOp>(op)) {
    // Capturing closure references are not materialize-able.
    legalizer.addWalk([&](SymbolConstantAttr ref) {
      if (ref.getType().getBody().isCapturing()) {
        mlir::emitError(op->getLoc(),
                        "capturing closures cannot be materialized at runtime");
        isFailure = true;
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
  } else if (auto createClosure = dyn_cast<CreateClosureOp>(op)) {
    if (createClosure.getCalleeSignature().getBody().isCapturing()) {
      mlir::emitError(op->getLoc(),
                      "capturing closures cannot be materialized at runtime");
      isFailure = true;
    }
  }

  // Walk the op attrs, results, block arguments, and locations.
  legalizer.walk(op->getAttrDictionary());
  legalizer.walk(op->getLoc());

  for (Value result : op->getResults())
    legalizer.walk(result.getType());

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (BlockArgument &arg : block.getArguments()) {
        legalizer.walk(arg.getLoc());
        legalizer.walk(arg.getType());
      }
    }
  }

  return failure(isFailure);
}

void EnsureNoParametersPass::runOnOperation() {
  getOperation()->walk([&](Operation *op) {
    if (failed(legalizeOp(op)))
      signalPassFailure();
  });
}
