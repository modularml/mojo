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

#include "AsyncRT/Runtime/CPUDevice.h"
#include "Config/Version.h"
#include "LSPServer.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Mojo/Support/Debugging.h"
#include "Mojo/ToolCommon/OOMHandler.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/LSP/Logging.h"
#include "llvm/Support/LSP/Transport.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"

#include <unistd.h>

using namespace M;
using namespace M::KGEN::LIT;
using namespace llvm::lsp;

int main(int argc, char **argv) {
  llvm::InitLLVM il(argc, argv, /*InstallPipeSignalExitHandler=*/false);
  llvm::PrettyStackTraceProgram x(argc, argv);

  KGEN::installOOMHandler();

  llvm::setBugReportMsg(
      "Please submit a bug report to https://github.com/modular/modular/issues "
      "and include the crash backtrace along with all the relevant source "
      "codes with the contents they had at crash time.\n");

  llvm::cl::OptionCategory category{"Mojo language server options"};

  llvm::cl::opt<JSONStreamStyle> inputStyle{
      "input-style",
      llvm::cl::desc("Input JSON stream encoding"),
      llvm::cl::values(clEnumValN(JSONStreamStyle::Standard, "standard",
                                  "usual LSP protocol"),
                       clEnumValN(JSONStreamStyle::Delimited, "delimited",
                                  "messages delimited by `// -----` lines, "
                                  "with // comment support")),
      llvm::cl::init(JSONStreamStyle::Standard),
      llvm::cl::Hidden,
      llvm::cl::cat(category),
  };
  llvm::cl::opt<bool> mojoTest{
      "mojo-test",
      llvm::cl::desc(
          "This flags sets up the server in test mode. It effectively sets the "
          "options `-input-style=delimited -pretty -log=verbose`, and "
          "indicates the LSP server to run in single-thread mode and to ensure "
          "that all the requests are resolved once the shutdown packet is "
          "received, to avoid early invalidations."),
      llvm::cl::init(false),
      llvm::cl::cat(category),
  };
  llvm::cl::opt<Logger::Level> logLevel{
      "log",
      llvm::cl::desc("Verbosity of log messages written to stderr"),
      llvm::cl::values(
          clEnumValN(Logger::Level::Error, "error", "Error messages only"),
          clEnumValN(Logger::Level::Info, "info",
                     "High level execution tracing"),
          clEnumValN(Logger::Level::Debug, "verbose", "Low level details")),
      // We are still in basic development mode, so we set the logLevel to Debug
      // to get more additional information for troubleshooting. When we become
      // more confident of the LSP, we can switch this back to Info.
      llvm::cl::init(Logger::Level::Debug),
      llvm::cl::cat(category),
  };
  llvm::cl::opt<bool> prettyPrint{
      "pretty",
      llvm::cl::desc("Pretty-print JSON output"),
      llvm::cl::init(false),
      llvm::cl::cat(category),
  };
  llvm::cl::opt<bool> attach{
      "attach-debugger-on-startup",
      llvm::cl::desc("Launch the server and start a debug session attached to "
                     "it on VS Code"),
      llvm::cl::init(false),
      llvm::cl::cat(category),
  };
  llvm::cl::opt<bool> waitOnShutdown{
      "wait-on-shutdown",
      llvm::cl::desc("Wait for pending requests to complete before shutting "
                     "down the server."),
      llvm::cl::init(false),
      llvm::cl::cat(category),
  };
  llvm::cl::list<std::string> includeDirs{
      "I",
      llvm::cl::desc("Append directory to the search path list used to "
                     "resolve imported modules in a document"),
      llvm::cl::cat(category),
  };
  llvm::cl::opt<bool> checkDocstrings{
      "check-docstrings",
      llvm::cl::desc(
          "Parse and type-check code blocks inside doc strings. Defaults to "
          "false: The server validates file structure but ignores errors "
          "inside docstring examples."),
      llvm::cl::init(false),
      llvm::cl::cat(category),
  };
  llvm::cl::opt<bool> version{
      "mojo-version",
      llvm::cl::desc("Output the Mojo version and exit."),
      llvm::cl::init(false),
      llvm::cl::cat(category),
  };

  llvm::cl::HideUnrelatedOptions(category);
  llvm::cl::ParseCommandLineOptions(argc, argv, "Mojo LSP Language Server");

  if (version) {
    // Print the version and exit.
    const char *versionStr = getMojoVersionString();
    llvm::outs() << llvm::formatv("Mojo {0}\n", versionStr);
    return 0;
  }

  if (isatty(STDOUT_FILENO)) {
    llvm::errs()
        << "The Mojo Language Server is not intended to be executed directly. "
           "Please refer to your editor documentation for instructions on "
           "integrating the language server with your editor.\n";
  }

  // Unconditionally enable tracing when we are being built with tracing.
  auto traceProfiler =
      std::make_unique<KGEN::TraceProfiler>(KGEN::kIsTracingEnabled, 3);

  if (attach)
    attachToNewRemoteDebugSession(true);

  // When testing, updating flags that make the server a bit easier to interact
  // with.
  if (mojoTest) {
    inputStyle = JSONStreamStyle::Delimited;
    logLevel = Logger::Level::Debug;
    prettyPrint = true;
    waitOnShutdown = true;
  }

  // Configure the logger.
  Logger::setLogLevel(logLevel);

  // Configure the transport used for communication.
  llvm::sys::ChangeStdinToBinary();
  JSONTransport transport(stdin, llvm::outs(), inputStyle, prettyPrint);

  // Register the additionally supported URI schemes for the server.
  URIForFile::registerSupportedScheme("vscode-notebook-cell");

  // Start the server.
  // When testing we use a single thread to provide deterministic output.
  return failed(runMojoLSPServer(transport, /*singleThreaded=*/mojoTest,
                                 waitOnShutdown, includeDirs,
                                 std::move(traceProfiler),
                                 /*checkDocstringCodeBlocks=*/checkDocstrings));
}
