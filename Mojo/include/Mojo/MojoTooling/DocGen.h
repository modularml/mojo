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
// Shared doc-generation pipeline used by both `mojo doc` and `kgen-doc`.
// Callers are responsible for creating the Init context, resolving the input
// path, setting up the MLIRContext (with registerAllKGENDialects), and managing
// the output file lifetime. This helper constructs a SourceMgr and
// LIT::ParserConfig internally for the parse/serialize steps.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOTOOLING_DOCGEN_H
#define KGEN_MOJOTOOLING_DOCGEN_H

#include "Support/Driver/DiagnosticFormat.h"

#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <string>
#include <vector>

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace M {

/// Configuration for Mojo documentation generation. Mirrors the flags exposed
/// by both `mojo doc` and `kgen-doc`; callers populate this from their
/// respective CLI parsers.
struct DocGenConfig {
  bool diagnoseMissingDocStrings = false;
  bool warningsAsErrors = false;
  unsigned maxNotesPerDiagnostic = 10;
  std::string stripFilePrefix;
  std::string docsBasePath;
  std::vector<std::string> includePaths;
  DiagnosticFormat diagnosticFormat = DiagnosticFormat::Text;
  std::vector<std::string> ignoredDeprecations;
};

/// Parses the Mojo file or package at \p resolvedPath, serializes its public
/// API documentation as JSON (including the top-level version wrapper) to
/// \p os, and returns true on success.
///
/// \p context must have all KGEN dialects already registered (e.g. via
/// registerAllKGENDialects). Diagnostics are written to stderr through the
/// SourceMgr handler selected by \p config.diagnosticFormat.
bool generateMojoDocJSON(const std::filesystem::path &resolvedPath,
                         mlir::MLIRContext &context, const DocGenConfig &config,
                         llvm::raw_ostream &os);

} // namespace M

#endif // KGEN_MOJOTOOLING_DOCGEN_H
