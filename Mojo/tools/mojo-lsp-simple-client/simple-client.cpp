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

#include "../mojo-lsp-test-client/LSPBatchClient.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"

using namespace M;
namespace lsp = llvm::lsp;

int main(int argc, char **argv) {
  llvm::InitLLVM il(argc, argv, /*InstallPipeSignalExitHandler=*/false);
  llvm::PrettyStackTraceProgram x(argc, argv);

  llvm::cl::opt<bool> attachDebugger{
      "attach-debugger",
      llvm::cl::desc("Launch the LSP and start a debug session attached to "
                     "it on VS Code."),
      llvm::cl::init(false),
  };

  llvm::cl::opt<bool> keepIOFiles{
      "keep-io-files",
      llvm::cl::desc(
          "Preserve the server's stdin/stdout/stderr temp files and print "
          "their paths, even on success. Useful for inspecting raw LSP "
          "traffic. On failure the files are always preserved."),
      llvm::cl::init(false),
  };

  llvm::cl::opt<bool> failOnDiagnostics{
      "fail-on-diagnostics",
      llvm::cl::desc(
          "Exit with failure if the server reports any error-severity "
          "diagnostics. Useful for verifying that a file parses cleanly."),
      llvm::cl::init(false),
  };

  llvm::cl::opt<bool> noDocstringChecks{
      "no-docstring-checks",
      llvm::cl::desc(
          "Skip parsing and type-checking of code blocks inside doc strings. "
          "Use with --fail-on-diagnostics to check that a file parses cleanly "
          "without requiring every docstring example to be valid Mojo code."),
      llvm::cl::init(false),
  };

  llvm::cl::opt<std::string> inputFile{
      "inputFile", llvm::cl::desc("The input file to be processed by the LSP."),
      llvm::cl::Positional, llvm::cl::Required};

  llvm::cl::ParseCommandLineOptions(
      argc, argv,
      "This simple LSP client receives an input file and spawns an LSP server "
      "to process it. This tool is intended to be used for debugging purposes "
      "and it's supposed to be modified to replicate any desired workflows.");

  auto bufferOr = toModularErrorOr(llvm::MemoryBuffer::getFile(inputFile));
  if (failed(bufferOr))
    llvm::report_fatal_error(Twine("Error reading the file ") + inputFile +
                             ": " + bufferOr.getError());
  llvm::MemoryBuffer &buffer = *bufferOr->get();
  // Convert to an absolute path so "file://" + path produces a valid
  // file:///abs/path URI.  A relative path yields file://host/path where the
  // first component is mis-parsed as the URI authority, causing the LSP server
  // to lose track of the file and skip import resolution entirely.
  llvm::SmallString<256> absPath(inputFile);
  if (std::error_code ec = llvm::sys::fs::make_absolute(absPath))
    llvm::report_fatal_error(Twine("Failed to make path absolute: ") +
                             ec.message());
  Document doc("file://" + absPath.str().str(), buffer.getBuffer());

  if (keepIOFiles)
    setenv("PRESERVE_LSP_IO_FILES", "1", /*overwrite=*/true);

  // Register a diagnostics handler when --fail-on-diagnostics is set so that
  // any error-severity diagnostic causes a non-zero exit.  The handler must be
  // registered before execute() is called; the server sends
  // textDocument/publishDiagnostics in response to textDocument/didOpen.
  bool hasDiagnosticErrors = false;
  LSPBatchClient client(/*attachDebugger=*/attachDebugger);
  client.setCheckDocstrings(!noDocstringChecks);
  client.open(doc);
  if (failOnDiagnostics)
    client.onDiagnostics(doc, [&](const std::vector<lsp::Diagnostic> &diags) {
      for (const auto &diag : diags) {
        if (diag.severity == lsp::DiagnosticSeverity::Error) {
          llvm::errs() << inputFile << ":" << (diag.range.start.line + 1) << ":"
                       << (diag.range.start.character + 1)
                       << ": error: " << diag.message << "\n";
          hasDiagnosticErrors = true;
        }
      }
    });

  // By default, we include the following requests that don't require any
  // special input.
  auto result =
      client
          .documentSymbol(doc,
                          [](const std::vector<llvm::lsp::DocumentSymbol> &) {
                            // This is left here for demonstrative purposes.
                            // Whenever you need to use this client, just
                            // specify the requests you want to send. You can
                            // use this lambda to print the results, but you can
                            // probably more easily just inspect the
                            // stdout/stderr files.
                          })
          .semanticTokensFull(doc, [](ArrayRef<Mojo::LSP::SemanticToken>) {})
          .hoverNullable(doc, {0, 0}, [](const std::optional<lsp::Hover2> &) {})
          .execute();

  if (failed(result.err)) {
    llvm::errs() << result.err.getError() << "\n";
    // Stream the server's stderr inline so crash details are immediately
    // visible without manually cat-ing the temp file.
    if (result.serverIOFiles) {
      if (auto stderrBuf =
              llvm::MemoryBuffer::getFile(result.serverIOFiles->serverStderr)) {
        llvm::errs() << (*stderrBuf)->getBuffer();
      }
    }
  } else if (!hasDiagnosticErrors) {
    llvm::errs() << "Success\n";
  }

  return (failed(result.err) || hasDiagnosticErrors) ? EXIT_FAILURE
                                                     : EXIT_SUCCESS;
}
