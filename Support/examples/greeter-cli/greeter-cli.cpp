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
// This is the main entry point for the `greeter-cli` executable. It parses the
// first command line argument to determine which subcommand to invoke.
//
//===----------------------------------------------------------------------===//

#include "Bye/bye.h"
#include "Hi/hi.h"

#include "Support/Driver/DriverSupport.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"

#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <memory>
#include <string>

using namespace M;

// Create the `llvm::opt::PrecomputedOptTable` class that LLVMOption needs for
// option parsing.
#define DRIVER_OPTIONS_PATH "DriverOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct DriverOptTable : public llvm::opt::PrecomputedOptTable {
  DriverOptTable()
      : llvm::opt::PrecomputedOptTable(OptionStrTable, OptionPrefixesTable,
                                       InfoTable, OptionPrefixesUnion) {}
};
} // namespace

int main(int argc, char **argv) {
  // First, set up LLVM.
  llvm::InitLLVM initLLVM(argc, argv);
  llvm::setBugReportMsg(
      "This example program shouldn't be crashing. Please submit an issue to "
      "https://github.com/modular/max/issues and include the crash "
      "backtrace. Thanks!\n");

  // Collect the command line arguments for parsing.
  SmallVector<const char *, 256> argvStorage(argv, argv + argc);
  const char *programName = argvStorage.front();
  ArrayRef<const char *> arguments = ArrayRef(argvStorage).slice(1);

  // MSupportDriver provides helpers for associating a subcommand name with a
  // function. Initialize this registry and add our `hi` and `bye` subcommands
  // to it.
  SubcommandRegistry registry;
  registerHiSubcommand(registry);
  registerByeSubcommand(registry);

  // The start of our custom argument parsing behavior: invoking just
  // `greeter-cli` behaves as if the user invoked `greeter-cli hi`.
  if (arguments.empty())
    return registry.getCallback("hi").get()(
        State(programName, "hi", arguments));

  // Otherwise, the user provided arguments. We wish to parse only the first
  // one:
  // 1. If it's `--help`, we'll print help text for this top-level command.
  // 2. If it's `--busy`, we'll print a custom greeting and exit.
  // 3. If it's a subcommand name like "hi" or "bye", we'll delegate argument
  //    parsing to those subcommands.
  // 4. If it's anything else, we'll behave as if the user invoked
  //    `greeter-cli hi [arguments]`.
  DriverOptTable options;
  llvm::opt::InputArgList args(arguments.begin(), arguments.end());
  unsigned index = 0;
  std::unique_ptr<llvm::opt::Arg> firstArg = options.ParseOneArg(args, index);
  switch (firstArg->getOption().getID()) {
  case options::OPT_help:
    // 1: The first argument is `--help`, so print the help text string that we
    //    generated with TableGen, and exit successfully.
    return State(programName, ArrayRef(arguments).slice(1))
        .printHelp(
#include "DriverOptionsHelpText.inc"
        );
  case options::OPT_busy:
    // 2: The first argument is `--busy`, so print this message and exit. Note
    //    that we're too "busy" to check if any of the following arguments may
    //    be `--help`.
    llvm::outs() << "Sorry, but now is NOT the time!\n";
    return EXIT_SUCCESS;
  case options::OPT_INPUT: {
    // 3 or 4: This is an argument that doesn't start with any option prefix,
    //         like "-" or "--".
    std::string arg = firstArg->getAsString(args);

    // 3: If it's a subcommand, invoke the subcommand callback.
    ErrorOr<SubcommandRegistry::Callback> callback = registry.getCallback(arg);
    if (succeeded(callback))
      return callback.get()(
          State(programName, arg.c_str(), arguments.slice(index)));

    // 4: It's not a subcommand, so we behave as if the user meant it to be an
    //    argument to the `hi` subcommand.
    State state(programName, "hi", arguments);
    return registry.getCallback("hi").get()(state);
  }
  default:
    // 4: This is some option we don't know about, so we behave as if the user
    //    meant it to be an argument to the `hi` subcommand.
    return registry.getCallback("hi").get()(
        State(programName, "hi", arguments));
  }
}
