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

#ifndef SUPPORT_DEBUGINFODIALECT_DEBUGINFOTOLLVM_TARGET_TARGETADAPTER_H
#define SUPPORT_DEBUGINFODIALECT_DEBUGINFOTOLLVM_TARGET_TARGETADAPTER_H

#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include <functional>

namespace M::DebugInfo {
//===----------------------------------------------------------------------===//
// TargetAdapter
//===----------------------------------------------------------------------===//
struct TargetAdapter {
  /// Conversion patterns.
  std::function<void(DIAttrTypeReplacer &, RewritePatternSet &)>
      populateConversionPatterns;

  /// Custom massaging.
  using DebugAdapterFn = std::function<void(mlir::ModuleOp)>;
  DebugAdapterFn preTranslationAdapter;
  DebugAdapterFn postTranslationAdapter;
};

/// Get the corresponding adapter for the target.
TargetAdapter getTargetAdapter(M::TargetInfoAttr target,
                               bool tradeoffPerfForVariableDI);

/// Default adapter for targets without a more specific adapter.
TargetAdapter getFallbackAdapter(bool tradeoffPerfForVariableDI);

//===----------------------------------------------------------------------===//
// Common Routines
//===----------------------------------------------------------------------===//
void populateFallbackConversionPatterns(DIAttrTypeReplacer &replacer,
                                        RewritePatternSet &patterns);

/// Sink kill Debug Value ops so that they are the last instructions from
/// their source line. This way variables are guaranteed to be killed only at
/// the end of the line.
void sinkDebugKills(ModuleOp module);
void sinkDebugKills(mlir::Operation *op);

/// This function converts instances of llvm.dbg.value to llvm.dbg.declare when
/// desirable. LLVM optimizations and codegen often muck up the use of
/// llvm.dbg.value (and other debug intrinsics), which creates subpar debugging
/// experiences. Converting to llvm.dbg.declare provides a more stable debugging
/// environment, and more closely matches what a traditional frontend would
/// provide in O0 modes.
///
/// The current conversion policy considers:
/// - Whether undef dbg.values appear after non-undef ones, implying a limited
/// scope incompatible with dbg.declare.
/// - Whether the exprLocation paths make it possible for the entire variable to
/// be placed in a single allocation without performing loads.
///
/// Additionally, regardless of whether dbg.value ops are switched to
/// dbg.declare, we put all variables in stack allocations for visibility in GPU
/// debugging.
///
/// TODO: As we grow support we may want to consider making this optional
/// depending on the debug mode.
void convertDbgValueToDeclare(mlir::ModuleOp module);
} // namespace M::DebugInfo

#endif // SUPPORT_DEBUGINFODIALECT_DEBUGINFOTOLLVM_TARGET_TARGETADAPTER_H
