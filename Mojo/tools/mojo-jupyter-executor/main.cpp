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
// This file defines an entry point for a dummy executable used by the Mojo
// REPL. This provides an anchor point for the debugger to run REPL expressions,
// as LLDB requires an in-memory target.
//
//===----------------------------------------------------------------------===//

#include "Init/Init.h"
#include "Mojo/MojoJupyter/Kernel.h"
#include "Mojo/Support/Configuration.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

using namespace M;
using namespace M::Mojo::Jupyter;

/// The leak sanitizer shows errors because we load libpython through this
/// plugin.
extern "C" const char *__asan_default_options() { return "detect_leaks=0"; }

/// Forward declaration so that the REPL mode can call into the notebook mode.
static LogicalResult executeNotebook(MojoKernel &kernel, StringRef notebookPath,
                                     bool debugOnFailure);

//===----------------------------------------------------------------------===//
// REPL Executor
//===----------------------------------------------------------------------===//

/// This function provides a REPL-like experience that calls into the Jupyter
/// kernel. You can enter code at the prompt and execute it - the prompt itself
/// will tell you the current 'cell'. This does not change automatically because
/// we need to be able to (for example) print the logs generated for the current
/// cell. In order to switch cells, you can use `:next-cell` or
/// `:prev-cell`. In order to exit cleanly, use `:exit`. If you want to begin
/// executing a notebook, use `:notebook /path/to/notebook`.
static void executeAsREPL(MojoKernel &kernel, StringRef currentCell = "") {
  int idx = 0;
  std::string cellPrefix =
      currentCell.empty() ? "[0] > " : ("[" + currentCell + "] > ").str();
  std::cout << cellPrefix;
  std::string expression;
  for (std::string line; std::getline(std::cin, line);) {
    // Allow the program to exit cleanly.
    if (line == ":exit")
      break;

    // Print the prompt at the end.
    auto scope = llvm::scope_exit([&]() { std::cout << cellPrefix; });

    // Allow the user control over which cell is executing. This is useful for
    // things like executing an expression, and then dumping the logs.
    if (line == ":next-cell") {
      cellPrefix = "[" + std::to_string(++idx) + "] > ";
      continue;
    }
    if (line == ":prev-cell") {
      cellPrefix = "[" + std::to_string(--idx) + "] > ";
      continue;
    }

    StringRef lineRef(line);
    if (lineRef.consume_front(":notebook")) {
      if (failed(executeNotebook(kernel, lineRef.ltrim().str(),
                                 /*debugOnFailure=*/true)))
        continue;
    }

    // Add the individual lines to the full expression - this allows us to
    // support multiline expressions in REPL mode.
    expression += line + "\n";
    if (!StringRef(line).rtrim().empty())
      continue;

    if (!kernel.startExecution(cellPrefix, expression)) {
      while (!kernel.checkExecutionFinished())
        continue;
    }
    expression.clear();
  }
}

//===----------------------------------------------------------------------===//
// Command Handlers
//===----------------------------------------------------------------------===//

/// Check the code completion results for the end position of the given code
/// cell. The computed completion results are printed to outs.
static void checkCodeCompletion(MojoKernel &kernel, StringRef code) {
  kernel.codeComplete(code, code.size(), [](StringRef completion) {
    llvm::outs() << "completion: " << completion << "\n";
  });
}

//===----------------------------------------------------------------------===//
// Jupyter Executor
//===----------------------------------------------------------------------===//

static LogicalResult executeNotebook(MojoKernel &kernel, StringRef notebookPath,
                                     bool debugOnFailure) {
  std::string errorMsg;
  std::unique_ptr<llvm::MemoryBuffer> notebookFile =
      mlir::openInputFile(notebookPath, &errorMsg);
  if (!notebookFile) {
    llvm::errs() << "error opening notebook file: " << errorMsg << "\n";
    return failure();
  }

  // Parse the notebook file into a json object.
  auto notebookJSON = llvm::json::parse(notebookFile->getBuffer());
  if (!notebookJSON) {
    llvm::errs() << "error parsing notebook file: " << notebookJSON.takeError()
                 << "\n";
    return failure();
  }
  auto notebook = notebookJSON->getAsObject();
  if (!notebook) {
    llvm::errs() << "error parsing notebook file: not a json object\n";
    return failure();
  }

  // Check that we can actually find the cells.
  auto *cells = notebook->getArray("cells");
  if (!cells) {
    llvm::errs() << "error parsing notebook file: no cells found\n";
    return failure();
  }

  for (const auto &[index, cell] : llvm::enumerate(*cells)) {
    // We only care about code cells.
    auto *cellObj = cell.getAsObject();
    auto *source = cellObj->getArray("source");
    if (cellObj->getString("cell_type") != "code" || !source)
      continue;

    // Concatenate all of the lines of the cell into a single string.
    std::string codeStr;
    for (const auto &line : *source)
      if (std::optional<StringRef> lineStr = line.getAsString())
        codeStr += *lineStr;
    if (codeStr.empty())
      continue;
    codeStr += "\n\n";
    StringRef code(codeStr);

    // Process special commands.
    // Check if this cell is testing code completion results.
    if (code.consume_front("%%test_code_completion\n")) {
      checkCodeCompletion(kernel, code.trim());
      continue;
    }

    // Otherwise, execute the cell code.
    std::string cellName = ("notebook_cell_" + Twine(index)).str();
    int finishState = kernel.startExecution(cellName, code);

    // If we finish with an error and we're in debug mode, drop into REPL mode
    // in the current cell.
    while (finishState != kFinishedSuccessfully) {
      // We hit an error, exit failure.
      if (finishState == kFinishedError) {
        // Drop into REPL mode if requested.
        if (debugOnFailure)
          executeAsREPL(kernel, cellName);

        return failure();
      }

      finishState = kernel.checkExecutionFinished();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

int main(int argc, char *argv[]) {
  llvm::cl::opt<std::string> notebookPath(
      llvm::cl::desc("Optional notebook to execute, if not present the "
                     "executor will run in REPL mode"),
      llvm::cl::Positional, llvm::cl::Optional);
  llvm::cl::opt<bool> debugOnFailure(
      "debug-on-failure", llvm::cl::desc("Drop into REPL mode on cell failure"),
      llvm::cl::init(false));
  llvm::cl::opt<std::string> lldbInitFile(
      "lldb-init-file", llvm::cl::init(""),
      llvm::cl::desc("Optional LLDB initialization file."),
      llvm::cl::value_desc("filename"), llvm::cl::Optional, llvm::cl::Hidden);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "mojo-jupyter-executor", Init::Options().withCPUDeviceOptions());
  if (ctxOr.isError()) {
    llvm::errs() << "failed to create context: " << ctxOr.getError() << "\n";
    return 1;
  }
  ContextRef ctx = ctxOr.takeValue();

  // Determine the path of the repl entry point.
  KGEN::MojoConfig config = KGEN::MojoConfig::fromContext(ctx);
  StringRef exePath = config.getREPLEntryPoint();

  // If this is a notebook, add the working directory to the kernel.
  std::string workingDirectory;
  if (notebookPath.getNumOccurrences() != 0) {
    workingDirectory =
        std::filesystem::path(notebookPath.getValue()).parent_path().string();
  }

  // Initialize the kernel.
  MojoKernel kernel([](StringRef kind, StringRef msg) {
    llvm::outs() << "[" << kind << "] " << msg << "\n";
  });
  if (failed(kernel.initialize(ctx, exePath, workingDirectory,
                               /*additionalDirectories=*/{}, lldbInitFile)))
    return 1;

  // If we have a notebook path, execute it, otherwise run in REPL mode.
  if (notebookPath.getNumOccurrences() != 0) {
    if (failed(executeNotebook(kernel, notebookPath, debugOnFailure)))
      return 1;
  } else {
    executeAsREPL(kernel);
  }
  return 0;
}
