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

#include "mojo-format.h"

#include "Init/Init.h"
#include "Mojo/Support/Configuration.h"
#include "Support/Driver/DriverSupport.h"

#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Program.h"

#include <filesystem>

using namespace M;

#define DRIVER_OPTIONS_PATH "Format/FormatOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct FormatOptTable : public llvm::opt::PrecomputedOptTable {
  FormatOptTable()
      : llvm::opt::PrecomputedOptTable(OptionStrTable, OptionPrefixesTable,
                                       InfoTable, OptionPrefixesUnion) {}
};
} // namespace

/// Resolve the path to the mblack formatter binary via modular.cfg.
/// On success, writes the path to \p outPath and returns 0.
/// On failure, reports the error and returns a non-zero exit code.
static int resolveMBlackPath(const State &state, std::string &outPath) {
  ErrorOr<KGEN::MojoConfig> configOr = KGEN::MojoConfig::open();
  if (failed(configOr))
    return state.reportError(Twine("failed to parse 'modular.cfg': ") +
                             configOr.getError());
  StringRef mblackPath = configOr->getMBlackPath();
  std::error_code ec;
  if (!std::filesystem::exists(mblackPath.str(), ec) || ec ||
      !llvm::sys::fs::can_execute(mblackPath))
    return state.reportError(
        "unable to resolve Mojo formatter in PATH, try installing `mojo`.");
  outPath = mblackPath.str();
  return 0;
}

/// Format a set of Mojo source files. Returns an integer representing a
/// successful exit code if formatting succeeded, otherwise returns a failure
/// code.
static int format(const State &state) {
  // Parse command line arguments.
  FormatOptTable options;
  unsigned missingIndex = 0;
  unsigned missingCount = 0;
  llvm::opt::InputArgList args =
      options.ParseArgs(state.arguments, missingIndex, missingCount);

  if (args.hasArg(options::OPT_help)) {
    return state.printHelp(
#include "Format/FormatOptionsHelpText.inc"
    );
  } else if (args.hasArg(options::OPT_help_hidden)) {
    return state.printHelp(
#include "Format/FormatOptionsHelpHiddenText.inc"
    );
  }

  if (int result = state.rejectUnknownArguments(args, options::OPT_UNKNOWN))
    return result;

  // Handle --print-cache-dir by forwarding to mblack.
  if (args.hasArg(options::OPT_print_cache_dir)) {
    std::string mblackPath;
    if (int result = resolveMBlackPath(state, mblackPath))
      return result;
    SmallVector<StringRef> printArgs = {mblackPath, "--print-cache-dir"};
    return llvm::sys::ExecuteAndWait(mblackPath, printArgs);
  }

  // Process the input files.
  std::vector<std::string> inputs = args.getAllArgValues(options::OPT_INPUT);
  if (!args.hasArg(options::OPT_INPUT))
    return state.reportError("no inputs provided");

  // Create our context.
  ErrorOr<ContextRef> ctxOr =
      Init::createContext("mojo", Init::Options(), "format");
  if (ctxOr.isError())
    return state.reportError(ctxOr.getError());
  ContextRef ctx = std::move(*ctxOr);

  // Check that the inputs are all valid Mojo files, or directories.
  std::error_code ec;
  for (const std::string &input : inputs) {
    // Allow "-" to represent stdin.
    if (input == "-") {
      if (inputs.size() > 1)
        return state.reportError("cannot mix '-' with other inputs");
      break;
    }

    std::filesystem::path inputPath(input);
    if (!std::filesystem::exists(inputPath, ec)) {
      return state.reportError(
          llvm::formatv("input '{0}' does not exist", input));
    }

    if (std::filesystem::is_directory(inputPath, ec))
      continue;
    if (ec)
      return state.reportError(ec.message());

    if (inputPath.extension().string() != ".mojo") {
      return state.reportError(
          llvm::formatv("invalid input '{0}', expected a source .mojo "
                        "file, or a directory",
                        input));
    }
  }

  StringRef lineLengthArg = args.getLastArgValue(options::OPT_line_length);
  if (!lineLengthArg.empty()) {
    int lineLength = 0;
    if (lineLengthArg.getAsInteger(10, lineLength)) {
      return state.reportError(llvm::formatv(
          "expected integer value for --line-length, but got '{0}'",
          lineLengthArg));
    }
  }

  // Check for additional options.
  bool isQuiet = args.hasArg(options::OPT_quiet);

  // Assert that we've parsed all command line arguments.
  state.assertNoUnusedArguments(args);

  std::string mblack;
  if (int result = resolveMBlackPath(state, mblack))
    return result;

  // Forward the curated options to mblack.
  SmallVector<StringRef> mblackArgs = {mblack, "--fast", "--preview"};
  if (!lineLengthArg.empty()) {
    mblackArgs.push_back("--line-length");
    mblackArgs.push_back(lineLengthArg);
  }
  // Tell mblack to only format Mojo files, not Python files.
  llvm::append_range(mblackArgs, ArrayRef<StringRef>{"-t", "mojo"});
  if (isQuiet)
    mblackArgs.push_back("-q");
  llvm::append_range(mblackArgs, inputs);
  return llvm::sys::ExecuteAndWait(mblack, mblackArgs);
}

void M::registerFormatSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("format", format);
}
