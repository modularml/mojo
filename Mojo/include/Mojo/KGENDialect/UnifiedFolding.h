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
// C++ trait for UnifiedFolding. Provides default interpret,
// parametric_interpret (via unifiedFold), and fold (via foldTrait) for ops
// that implement a single `unifiedFold` method.
//
// Ops must separately list InterpreterOpInterface for interpreter discovery.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_KGENDIALECT_UNIFIEDFOLDING_H
#define KGEN_KGENDIALECT_UNIFIEDFOLDING_H

#include "Mojo/KGENDialect/FoldUtils.h"
#include "Support/MDialect/MAttrs.h"

namespace M::detail {

/// The trait mixed into concrete ops via NativeOpTrait.  Provides:
///   - Default `interpret` / `parametric_interpret` that delegate to
///     `unifiedFold` (overridable by shadowing on the op class).
///   - `foldTrait` so that MLIR's fold infrastructure folds through
///     `unifiedFold` without requiring `hasFolder = 1`.
template <typename ConcreteOp>
struct UnifiedFoldingOpInterfaceTrait {

  /// Default interpret hook — delegates to unifiedFold.
  ErrorTreeOrSuccess interpret(ArrayRef<Attribute> operands,
                               InterpreterState &state) {
    auto *self = static_cast<ConcreteOp *>(this);
    auto foldFn = [self](KGEN::FoldValues fvs, TargetInfoAttr target) {
      return self->unifiedFold(fvs, target);
    };
    return KGEN::interpretOpWithFold((*self)->getLoc(),
                                     (*self)->getName().getStringRef(),
                                     operands, state, foldFn);
  }

  /// Default parametric_interpret hook — delegates to unifiedFold.
  ErrorTreeOrSuccess parametric_interpret(ArrayRef<Attribute> operands,
                                          ParametricInterpreterState &state) {
    auto *self = static_cast<ConcreteOp *>(this);
    auto foldFn = [self](KGEN::FoldValues fvs, TargetInfoAttr target) {
      return self->unifiedFold(fvs, target);
    };
    return KGEN::interpretOpWithFold((*self)->getLoc(),
                                     (*self)->getName().getStringRef(),
                                     operands, state, foldFn);
  }

  /// Trait-based fold hook.  Called by MLIR when the op does not have
  /// `hasFolder = 1`.  Delegates to the concrete op's `unifiedFold`.
  static LogicalResult foldTrait(Operation *op, ArrayRef<Attribute> operands,
                                 SmallVectorImpl<OpFoldResult> &results) {
    assert(op->getNumResults() == 1 &&
           "UnifiedFolding only supports single-result ops");
    auto foldFn = [op](KGEN::FoldValues fvs, TargetInfoAttr target) {
      return cast<ConcreteOp>(op).unifiedFold(fvs, target);
    };
    OpFoldResult result =
        KGEN::foldOpWithTarget(KGEN::FoldValues(operands, op->getOperands()),
                               lookupTargetInfo(op), foldFn);
    if (!result)
      return failure();
    results.push_back(result);
    return success();
  }
};

} // namespace M::detail

#endif // KGEN_KGENDIALECT_UNIFIEDFOLDING_H
