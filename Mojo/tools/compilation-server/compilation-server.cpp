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

#include "CompilationServer.h"
#include "Mojo/Support/Debugging.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/LSP/Logging.h"
#include "llvm/Support/LSP/Transport.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"

using namespace M;
using namespace M::KGEN;
using namespace llvm::lsp;

int main(int argc, char **argv) {
  llvm::InitLLVM il(argc, argv, /*InstallPipeSignalExitHandler=*/false);
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  llvm::PrettyStackTraceProgram x(argc, argv);
  llvm::setBugReportMsg(
      "Compilation server has encountered an internal error. "
      "Please submit a Modular internal bug report "
      "and include the crash backtrace along with your command line"
      "invocation.\n");

  llvm::cl::opt<JSONStreamStyle> inputStyle{
      "input-style",
      llvm::cl::desc("Input JSON stream encoding"),
      llvm::cl::values(clEnumValN(JSONStreamStyle::Standard, "standard",
                                  "usual Compilation Server Protocol"),
                       clEnumValN(JSONStreamStyle::Delimited, "delimited",
                                  "messages delimited by `// -----` lines, "
                                  "with // comment support "
                                  "to facilitate debugging")),
      llvm::cl::init(JSONStreamStyle::Standard),
      llvm::cl::Hidden,
  };
  llvm::cl::opt<Logger::Level> logLevel{
      "log",
      llvm::cl::desc("Verbosity of log messages written to stderr"),
      llvm::cl::values(
          clEnumValN(Logger::Level::Error, "error", "Error messages only"),
          clEnumValN(Logger::Level::Info, "info",
                     "High level execution tracing"),
          clEnumValN(Logger::Level::Debug, "verbose", "Low level details")),
      // Print maximum info, while compilation server is under development.
      llvm::cl::init(Logger::Level::Debug),
  };
  llvm::cl::opt<bool> prettyPrint{
      "pretty",
      llvm::cl::desc("Pretty-print JSON output"),
      llvm::cl::init(false),
  };
  llvm::cl::opt<bool> singleThreaded{
      "single-threaded",
      llvm::cl::desc("Use single-threaded mode for the runtime"),
      llvm::cl::init(false),
  };
  llvm::cl::opt<bool> testMode{
      "test",
      llvm::cl::desc("This flags sets up the server in test mode. It "
                     "effectively sets the "
                     "options `-input-style=delimited -pretty -log=verbose "),
      llvm::cl::init(false),
  };
  llvm::cl::opt<bool> attach{
      "attach-debugger-on-startup",
      llvm::cl::desc("Launch the server and start a debug session attached to "
                     "it on VS Code"),
      llvm::cl::init(false),
  };

  llvm::cl::ParseCommandLineOptions(argc, argv, "Compilation Server");

  // When testing, set the flags that make it easier to interact with server.
  if (testMode) {
    inputStyle = JSONStreamStyle::Delimited;
    logLevel = Logger::Level::Debug;
    prettyPrint = true;
  }

  // Configure the logger.
  Logger::setLogLevel(logLevel);

  // Configure the transport used for communication.
  if (llvm::sys::ChangeStdinToBinary())
    return -1;

  JSONTransport transport(stdin, llvm::outs(), inputStyle, prettyPrint);

  if (attach)
    attachToNewRemoteDebugSession();

  // Start the server.
  return failed(runCompilationServer(transport, singleThreaded));
}
