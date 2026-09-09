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

#ifndef SUPPORT_DRIVER_DIAGNOSTICFORMAT_H
#define SUPPORT_DRIVER_DIAGNOSTICFORMAT_H

#include "llvm/Support/SourceMgr.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llvm::json {
class Value;
} // namespace llvm::json

namespace M {

//===----------------------------------------------------------------------===//
// JSON protocol
//===----------------------------------------------------------------------===//

namespace json {

/// The diagnostic kind, such as "error" or "warning."
enum class DiagnosticKind {
  /// A purely informational message; neither a warning nor an error.
  Note = 0,
  /// Not quite an error, but still a potential problem of which the user should
  /// be made aware.
  Warning = 1,
  /// A hard error that prevents the tool from completing its task.
  Error = 2,
};

/// Serializes the given diagnostic kind as a JSON string value.
llvm::json::Value toJSON(DiagnosticKind kind);

/// A highlighted range of text in a diagnostic.
struct DiagnosticRange {
  /// The index of the starting character in the range, 0-indexed.
  int64_t start;
  /// The index of the end of the range, which is the index of the last
  /// character (0-indexed), plus one.
  int64_t end;
};

/// Serializes the given range as a JSON object.
llvm::json::Value toJSON(const DiagnosticRange &range);

/// A line and column number pair that is being pointed out in a diagnostic.
struct DiagnosticLocation {
  /// The line number, 1-indexed.
  int64_t line;
  /// The column number, 1-indexed.
  int64_t column;
};

/// Serializes the given location as a JSON object.
llvm::json::Value toJSON(const DiagnosticLocation &location);

/// A range of text to replace, and the text to replace it with.
struct FixIt {
  /// The replacement text.
  std::string text;
  /// The start of the text to replace.
  DiagnosticLocation start;
  /// One character past the end of the text to replace.
  DiagnosticLocation end;
};

/// Serializes the given fix-it as a JSON object.
llvm::json::Value toJSON(const FixIt &fixIt);

/// A diagnostic that points to a specific location in a source program.
struct SourceDiagnostic {
  /// The path to the file for which this diagnostic is being output.
  std::string file;
  /// The line of text for which this diagnostic is being output. (Note that
  /// LLVM does not support diagnostics that refer to ranges that span multiple
  /// lines.)
  std::string text;
  /// The location for which this diagnostic is being output.
  DiagnosticLocation location;
  /// Zero or more ranges of text that are related to the diagnostic. These all
  /// appear on the same line as the diagnostic itself.
  std::vector<DiagnosticRange> ranges;
  /// Zero or more fix-it suggestions that, when applied, address the problem
  /// raised by the diagnostic.
  std::vector<FixIt> fixIts;
};

/// Serializes a source location diagnostic as a JSON object.
llvm::json::Value toJSON(const SourceDiagnostic &diagnostic);

/// A diagnostic that can be represented with JSON.
struct Diagnostic {
  /// The diagnostic type, such as "warning" or "error."
  DiagnosticKind kind;
  /// The message the diagnostic is to convey to the reader.
  std::string message;
  /// Not all diagnostics point to source locations, but those that do represent
  /// that location with this field.
  std::optional<SourceDiagnostic> diagnostic = std::nullopt;
};

/// Serializes a diagnostic as a JSON object.
llvm::json::Value toJSON(const Diagnostic &diagnostic);
} // namespace json

//===----------------------------------------------------------------------===//
// Driver utilities
//===----------------------------------------------------------------------===//

/// The format with which to print diagnostics.
enum class DiagnosticFormat {
  /// Print diagnostics as plain text, with no strict format or specification.
  Text,
  /// Print diagnostics as JSON Lines (https://jsonlines.org), where each line
  /// is a textual JSON representation of the `Diagnostic` struct defined above.
  JSON
};

/// Returns an LLVM source manager diagnostic handler for the given diagnostic
/// format. Text diagnostics are printed in LLVM's default manner, while JSON
/// diagnostics are printed as textual JSON, each separated by a newline.
llvm::SourceMgr::DiagHandlerTy getDiagHandler(DiagnosticFormat format);
} // namespace M

#endif // SUPPORT_DRIVER_DIAGNOSTICFORMAT_H
