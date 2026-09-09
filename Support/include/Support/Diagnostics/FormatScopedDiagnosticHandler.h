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

#ifndef SUPPORT_DIAGNOSTICS_FORMATSCOPEDDIAGNOSTICHANDLER_H
#define SUPPORT_DIAGNOSTICS_FORMATSCOPEDDIAGNOSTICHANDLER_H

#include "mlir/IR/Diagnostics.h"

#include <string>
#include <vector>

namespace M {

/// Capture the diagnostics in a given scope. Later, generate a formatted
/// message that can be displayed to a user. No formatting is done in the
/// diagnostics handler in order to save time. The diagnostics are captured in
/// the handler, but only formatted when formatMessage is called.
class FormatScopedDiagnosticHandler : public mlir::ScopedDiagnosticHandler {
public:
  FormatScopedDiagnosticHandler(mlir::MLIRContext *ctx);
  std::string formatMessage() const;

  static void emitDiagnosticSeverityToStream(llvm::raw_ostream &os,
                                             const mlir::Diagnostic &diag);

  /// Emit all details of the diagnostic to a single stream. Note that this
  /// can be a very large message, so dump it to the console with care.
  static void emitDiagnosticToStream(llvm::raw_ostream &fullOutputStream,
                                     const mlir::Diagnostic &diag);

  /// Emit the diagnostic to two streams: one for the minimal output and one
  /// for the full output. The minimal output will include one line per
  /// diagnostic, and the full output will include the entire diagnostic.
  ///
  /// The minimal output is intended to be used for display in a terminal, and
  /// the full output is intended to be used for display in a file or other
  /// non-terminal output.
  static void emitDiagnosticToStream(llvm::raw_ostream &minimalOutputStream,
                                     llvm::raw_ostream &fullOutputStream,
                                     const mlir::Diagnostic &diag);

private:
  std::vector<mlir::Diagnostic> diagnostics;
};

} // namespace M

#endif // SUPPORT_DIAGNOSTICS_FORMATSCOPEDDIAGNOSTICHANDLER_H
