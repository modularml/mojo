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

#include "hi.h"
#include "Support/Driver/DriverSupport.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <string>
#include <vector>

using namespace M;

// As with the top-level `greeter-cli.cpp`, we must generate the C++ boilerplate
// that LLVMOption requires for option parsing.
#define DRIVER_OPTIONS_PATH "Hi/HiOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct HiOptTable : public llvm::opt::PrecomputedOptTable {
  HiOptTable()
      : llvm::opt::PrecomputedOptTable(OptionStrTable, OptionPrefixesTable,
                                       InfoTable, OptionPrefixesUnion) {}
};
} // namespace

static int hi(const State &state) {
  // The top-level main function only parses the first argument, to determine
  // which subcommand to invoke. We parse the remaining arguments here.
  HiOptTable options;
  unsigned missingIndex = 0;
  unsigned missingCount = 0;
  llvm::opt::InputArgList args =
      options.ParseArgs(state.arguments, missingIndex, missingCount);

  // Before anything else, if the `--help` option appears anywhere, print help
  // text and exit successfully. We want people to be able to add `--help`
  // anywhere in their CLI invocation and get help, even if other arguments are
  // invalid.
  if (args.hasArg(options::OPT_help)) {
    return state.printHelp(
#include "Hi/HiOptionsHelpText.inc"
    );
  }

  // Exit unsuccessfully if we have unrecognized options anywhere.
  // (The flipside of being able to write any kind of command line parsing
  // behavior we wish is that rejecting incorrect options is not done "for you"
  // at any point; you must include explicit code such as this. After all, some
  // CLI wish to `--allow any -and --all /arguments`.)
  if (int result = state.rejectUnknownArguments(args, options::OPT_UNKNOWN))
    return result;

  // Our `hi` subcommand can take a single subject of a greeting; verify that we
  // have only one, and then print the greeting.
  std::vector<std::string> inputs = args.getAllArgValues(options::OPT_INPUT);
  if (inputs.size() > 1)
    return state.reportError(llvm::formatv(
        "too many things to greet, cannot handle both '{0}' and '{1}'",
        inputs[0], inputs[1]));

  std::string greeting = "Hi";
  if (args.hasArg(options::OPT_hello))
    greeting = "Hello";

  if (inputs.empty())
    llvm::outs() << greeting << "!\n";
  else
    llvm::outs() << greeting << ", " << inputs.front() << "!\n";

  return EXIT_SUCCESS;
}

void M::registerHiSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("hi", hi);
}
