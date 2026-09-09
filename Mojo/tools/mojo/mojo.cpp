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

#include "Build/mojo-build.h"
#include "Debug/mojo-debug.h"
#include "Demangle/mojo-demangle.h"
#include "Doc/mojo-doc.h"
#include "Format/mojo-format.h"
#include "Precompile/mojo-precompile.h"
#include "REPL/mojo-repl.h"
#include "Run/mojo-run.h"

#include "Config/Version.h"
#include "Mojo/Support/CLOptionUtils.h"
#include "Mojo/Support/Configuration.h"
#include "Mojo/Support/Constants.h"
#include "Mojo/Support/ForceLinkMLIRC.h"
#include "Mojo/ToolCommon/OOMHandler.h"
#include "Support/Configuration.h"
#include "Support/CrashReporting/CrashReporting.h"
#include "Support/Driver/DriverSupport.h"
#include "Support/LogicalResult.h"
#include "Support/Process.h"

#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/InitLLVM.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <string>

using namespace M;

#define DRIVER_OPTIONS_PATH "DriverOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct DriverOptTable : public llvm::opt::PrecomputedOptTable {
  DriverOptTable()
      : llvm::opt::PrecomputedOptTable(OptionStrTable, OptionPrefixesTable,
                                       InfoTable, OptionPrefixesUnion) {}
};
} // namespace

//===----------------------------------------------------------------------===//
// `main` entry point
//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  // Force linking of MLIR C symbols to JIT Mojo code relying on the mlir
  // bindings.
  KGEN::forceLinkMLIRC();
  M::registerCommandFlags();

  // Install LLVM signal handlers and convert `argc` and `argv` for Windows
  // hosts.
  llvm::InitLLVM initLLVM(argc, argv);

  KGEN::installOOMHandler();

  llvm::setBugReportMsg(
      "Please submit a bug report to https://github.com/modular/modular/issues "
      "and include the crash backtrace along with all the relevant source "
      "codes.\n");

  // Store command line arguments and record the program name.
  SmallVector<const char *, 256> argvStorage(argv, argv + argc);
  const char *programName = argvStorage.front();
  ArrayRef<const char *> arguments = ArrayRef(argvStorage).slice(1);

  // Register subcommands and their options.
  SubcommandRegistry registry;
  registerBuildSubcommand(registry);
  registerDemangleSubcommand(registry);
  registerDocSubcommand(registry);
  registerFormatSubcommand(registry);
  registerPrecompileSubcommand(registry);
  registerREPLSubcommand(registry);
  registerDebugSubcommand(registry);
  registerRunSubcommand(registry);

  // If the user hasn't provided any arguments, treat this as the `repl`
  // subcommand.
  if (arguments.empty())
    return registry.getCallback("repl").get()(
        State(programName, "repl", arguments));

  // Otherwise, parse the first argument; it could be:
  // - One of a handful of top-level driver options that we allow in this first
  //   position.
  // - One of the registered subcommands.
  // - A positional ("input") argument, or an option, that we don't recognize.
  DriverOptTable options;
  llvm::opt::InputArgList args(arguments.begin(), arguments.end());
  unsigned index = 0;
  std::unique_ptr<llvm::opt::Arg> firstArg = options.ParseOneArg(args, index);
  switch (firstArg->getOption().getID()) {
  case options::OPT_version: {
    // Print the version and exit.
    ProjectVersion version = getMojoVersion();
    const char *versionStr = getMojoVersionString();
    llvm::outs() << llvm::formatv("Mojo {0} ({1})\n", versionStr,
                                  version.revision);
    return 0;
  }
  case options::OPT_print_cache_location: {
    // Resolve the same cache root used by ObjectCompiler / KGENCompiler at
    // runtime, and report where the `.mojo_cache` directory lives.
    ErrorOr<Config> cfg = Config::open();
    if (cfg.isError()) {
      llvm::errs() << "error: failed to open Modular configuration: "
                   << cfg.getError() << "\n";
      return 1;
    }
    auto cacheFolder = cfg->getModularCacheFolderPath();
    if (cacheFolder.isError()) {
      llvm::errs() << "error: failed to resolve cache folder: "
                   << cacheFolder.getError() << "\n";
      return 1;
    }
    llvm::outs()
        << (*cacheFolder / (KGEN::kMojoCacheBaseDirName.str())).string()
        << "\n";
    return 0;
  }
  case options::OPT_clear_cache: {
    // Accept '-f' / '--force' as a trailing modifier to skip the prompt.
    // Anything else after `--clear-cache` is rejected so we don't silently
    // ignore typos like `--forced`.
    bool force = false;
    for (const char *rest : arguments.slice(index)) {
      StringRef restRef(rest);
      if (restRef == "-f" || restRef == "--force") {
        force = true;
      } else {
        llvm::errs() << "error: unexpected argument '" << restRef
                     << "' for --clear-cache; expected '-f' or '--force'.\n";
        return 1;
      }
    }

    ErrorOr<Config> cfg = Config::open();
    if (cfg.isError()) {
      llvm::errs() << "error: failed to open Modular configuration: "
                   << cfg.getError() << "\n";
      return 1;
    }
    auto cacheFolder = cfg->getModularCacheFolderPath();
    if (cacheFolder.isError()) {
      llvm::errs() << "error: failed to resolve cache folder: "
                   << cacheFolder.getError() << "\n";
      return 1;
    }
    std::filesystem::path mojoCachePath =
        *cacheFolder / KGEN::kMojoCacheBaseDirName.str();

    std::error_code ec;
    if (!std::filesystem::exists(mojoCachePath, ec)) {
      llvm::outs() << "Mojo cache at " << mojoCachePath.string()
                   << " does not exist; nothing to do.\n";
      return 0;
    }

    if (!force) {
      llvm::outs() << "This will remove the Mojo compile cache at:\n  "
                   << mojoCachePath.string() << "\nProceed? [y/N] ";
      llvm::outs().flush();

      std::string response;
      if (!std::getline(std::cin, response)) {
        llvm::errs() << "error: failed to read confirmation; aborting.\n";
        return 1;
      }
      // Trim leading/trailing whitespace and lowercase.
      auto isWS = [](unsigned char c) { return std::isspace(c) != 0; };
      while (!response.empty() && isWS(response.front()))
        response.erase(response.begin());
      while (!response.empty() && isWS(response.back()))
        response.pop_back();
      std::transform(response.begin(), response.end(), response.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      if (response != "y" && response != "yes") {
        llvm::outs() << "Aborted.\n";
        return 0;
      }
    }

    std::filesystem::remove_all(mojoCachePath, ec);
    if (ec) {
      llvm::errs() << "error: failed to remove " << mojoCachePath.string()
                   << ": " << ec.message() << "\n";
      return 1;
    }
    llvm::outs() << "Removed " << mojoCachePath.string() << "\n";
    return 0;
  }
  case options::OPT_help:
    // Print the top level driver help text and exit.
    return State(programName, ArrayRef(arguments).slice(1))
        .printHelp(
#include "DriverOptionsHelpText.inc"
        );
  case options::OPT_help_hidden:
    // Print the top level driver help hidden text and exit.
    return State(programName, ArrayRef(arguments).slice(1))
        .printHelp(
#include "DriverOptionsHelpHiddenText.inc"
        );

  case options::OPT_INPUT: {
    // This could be a subcommand, or it could be an input file for the `run`
    // subcommand.
    std::string arg = firstArg->getAsString(args);
    ErrorOr<SubcommandRegistry::Callback> callback = registry.getCallback(arg);
    // If it's a subcommand, invoke its callback.
    if (succeeded(callback))
      return callback.get()(
          State(programName, arg.c_str(), arguments.slice(index)));

    // If it looks like a Mojo source file, invoke the `run` subcommand.
    State state(programName, "run", arguments);
    StringRef argRef(arg);
    if (argRef.ends_with(".mojo"))
      return registry.getCallback("run").get()(state);

    // Otherwise, we don't know what this is; return an error.
    return state.reportError(callback.getError());
  }
  default:
    // This is some sort of option, so we'll pass it along to the `run` command
    // to parse. This allows for invocations such as `mojo -Ifoo Foo.mojo`.
    return registry.getCallback("run").get()(
        State(programName, "run", arguments));
  }
}
