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

#include "mojo-doc.h"

#include "AsyncRT/Runtime/Allocator.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/WorkQueue.h"
#include "Init/Init.h"
#include "Mojo/MojoTooling/DocGen.h"
#include "Mojo/ToolCommon/InitAllDialects.h"
#include "Support/Driver/DriverSupport.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ToolOutputFile.h"

#include <filesystem>

using namespace M;

#define DRIVER_OPTIONS_PATH "Doc/DocOptions.inc"
#include "Support/Driver/OptTable.inc"

namespace {
struct DocOptTable : public llvm::opt::PrecomputedOptTable {
  DocOptTable()
      : llvm::opt::PrecomputedOptTable(OptionStrTable, OptionPrefixesTable,
                                       InfoTable, OptionPrefixesUnion) {}
};
} // namespace

/// Given the path to a Mojo source file, opens and parses that file's doc
/// strings in order to generate structured output (currently JSON). Returns an
/// integer representing a successful exit code is documentation generation
/// succeeded, otherwise returns a failure code.
static int doc(const State &subcommandState) {
  // Parse command line arguments.
  State state = subcommandState;
  DocOptTable options;
  unsigned missingIndex = 0;
  unsigned missingCount = 0;
  llvm::opt::InputArgList args =
      options.ParseArgs(state.arguments, missingIndex, missingCount);

  if (args.hasArg(options::OPT_help)) {
    return state.printHelp(
#include "Doc/DocOptionsHelpText.inc"
    );
  } else if (args.hasArg(options::OPT_help_hidden)) {
    return state.printHelp(
#include "Doc/DocOptionsHelpHiddenText.inc"
    );
  }

  if (int result = state.parseDiagnosticFormatArguments(
          args, options::OPT_diagnostic_format,
          /*disableWarningsId=*/llvm::opt::OptSpecifier(), options::OPT_werror,
          options::OPT_wno_error))
    return result;

  // Handle deprecated --validate-doc-strings flag as an alias for -Werror.
  // Only apply if user hasn't explicitly specified -Werror or -Wno-error.
  if (args.hasArg(options::OPT_validate_doc_strings) &&
      !args.hasArg(options::OPT_werror) &&
      !args.hasArg(options::OPT_wno_error)) {
    state.reportWarning(
        "--validate-doc-strings is deprecated, use -Werror instead");
    state.warningsAsErrors = true;
  }

  if (int result = state.rejectUnknownArguments(args, options::OPT_UNKNOWN))
    return result;

  if (!args.hasArg(options::OPT_INPUT))
    return state.reportError("no input file provided");
  if (args.hasMultipleArgs(options::OPT_INPUT)) {
    std::vector<std::string> inputs = args.getAllArgValues(options::OPT_INPUT);
    return state.reportError(llvm::formatv(
        "too many input files, cannot process both '{0}' and '{1}'", inputs[0],
        inputs[1]));
  }

  // Create our context.
  ErrorOr<ContextRef> ctxOr =
      Init::createContext("mojo", Init::Options(), "doc");
  if (ctxOr.isError())
    return state.reportError(ctxOr.getError());
  // Keep ctx alive for the duration of the pipeline; it holds init/runtime
  // state that the parser depends on.
  ContextRef ctx = std::move(*ctxOr);

  // Resolve the input, or exit with an error.
  auto pathOrErr =
      resolveMojoInputFileOrPackage(args.getLastArgValue(options::OPT_INPUT));
  if (pathOrErr)
    return state.reportError(pathOrErr.getError());

  mlir::DialectRegistry registry;
  registerAllKGENDialects(registry);
  mlir::MLIRContext context{registry};

  // Open the output file, or exit with an error.
  std::string outputError;
  std::unique_ptr<llvm::ToolOutputFile> out = mlir::openOutputFile(
      args.getLastArgValue(options::OPT_o, "-"), &outputError);
  if (!out)
    return state.reportError(outputError);

  DocGenConfig config;
  config.warningsAsErrors = state.areWarningsAsErrors();
  config.diagnoseMissingDocStrings =
      args.hasArg(options::OPT_diagnose_missing_doc_strings);
  int maxNotes = 0;
  if (!args.getLastArgValue(options::OPT_max_notes).getAsInteger(10, maxNotes))
    config.maxNotesPerDiagnostic = maxNotes;
  config.stripFilePrefix = args.getLastArgValue(options::OPT_strip_file_prefix);
  config.docsBasePath = args.getLastArgValue(options::OPT_docs_base_path);
  config.includePaths = args.getAllArgValues(options::OPT_I);
  config.diagnosticFormat = state.diagnosticFormat;
  config.ignoredDeprecations =
      args.getAllArgValues(options::OPT_ignore_deprecated);

  // Note: timing scope removed during DocGen extraction — it was never
  // surfaced to users.
  if (!generateMojoDocJSON(*pathOrErr, context, config, out->os()))
    return state.reportError("could not generate documentation");

  out->keep();

  // Assert that we've parsed all command line arguments.
  state.assertNoUnusedArguments(args);

  return EXIT_SUCCESS;
}

void M::registerDocSubcommand(SubcommandRegistry &registry) {
  registry.addCallback("doc", doc);
}
