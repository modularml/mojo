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

#ifndef SUPPORT_COMPILER_DIAGNOSTICHANDLER_H
#define SUPPORT_COMPILER_DIAGNOSTICHANDLER_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "mlir/IR/Diagnostics.h"
#include <cstdint>
#include <vector>

namespace M {
/// This diagnostic handler captures MLIR diagnostics emitted into a vector.
class DiagnosticHandler {
public:
  DiagnosticHandler(MLIRContext *ctx, bool capturePerThread = true);
  ~DiagnosticHandler();

  /// Emit the diagnostics.
  void emitDiagnostics(function_ref<void(Diagnostic &)> emitFn);

  /// Manually remove the handler from the context.
  void release();

  /// Get the global HandlerID which is a unique identifier for this Handler.
  mlir::DiagnosticEngine::HandlerID getHandlerID();

  /// Return true if there is any diagnostic to emit
  bool hasDiagnostics() const { return !diagnostics.empty(); }

  /// Return the captured diagnostics
  const std::vector<Diagnostic> &getDiagnostics() const { return diagnostics; }

private:
  /// The MLIR context.
  MLIRContext *ctx;
  /// The ID of the registered handler.
  mlir::DiagnosticEngine::HandlerID handlerID = 0;
  /// The thread ID of the thread that registered the handler.
  uint64_t threadID = 0;
  /// Whether to capture diagnostics from all threads.
  bool capturePerThread = true;
  /// The captured diagnostics.
  std::vector<Diagnostic> diagnostics;
};
} // namespace M

#endif // SUPPORT_COMPILER_DIAGNOSTICHANDLER_H
