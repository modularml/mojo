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

#include "Config/Version.h"
#include "Mojo/Compiler/LLVMIRUtils.h"
#include "Mojo/ExecutionEngine/ExecutionEngine.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Mojo/ToolCommon/CLOptions.h"
#include "Support/CommonCLOptions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <utility>

using namespace M;
using namespace KGEN;
using namespace mlir;
using namespace llvm;

//===----------------------------------------------------------------------===//
// Module Splitter
//===----------------------------------------------------------------------===//

namespace {
class CLOptions : public CLOptionsBase {

public:
  CLOptions(int argc, char **argv, bool skipInitLLVM = false)
      : CLOptionsBase(argc, argv, options, skipInitLLVM) {}

  OptionsBase options;
  std::string inputFilename{"-"};
  std::string outputPrefix{"-"};
  bool perFunctionSplit = false;
  bool timeTrace = false;
  int timeTraceGranularity = 0;

private:
  llvm::cl::OptionCategory cat{"Common command line options"};

  M::cl::MOpt<std::string, true> inputFilenameOpt{
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::location(inputFilename), llvm::cl::cat(cat)};

  M::cl::MOpt<std::string, true> outputPrefixOpt{
      "output-prefix", llvm::cl::desc("output prefix"),
      llvm::cl::value_desc("output prefix"), llvm::cl::location(outputPrefix),
      llvm::cl::cat(cat)};

  M::cl::MOpt<bool, true> perFunctionSplitOpt{
      "per-func", llvm::cl::desc("split each function into separate modules"),
      llvm::cl::value_desc("split each function into separate modules"),
      llvm::cl::location(perFunctionSplit), llvm::cl::cat(cat)};

  M::cl::MOpt<bool, true> timeTraceOpt{
      "time-trace",
      llvm::cl::desc("Turn on time profiler. Generates JSON file "
                     "called kgen.trace.json in the derived directory."),
      llvm::cl::location(timeTrace), llvm::cl::cat(cat)};

  M::cl::MOpt<int, true> timeTraceGranularityOpt{
      "time-trace-granularity",
      llvm::cl::desc("Minimum time granularity (in microseconds) "
                     "traced by time profiler."),
      llvm::cl::location(timeTraceGranularity), llvm::cl::cat(cat)};
};

} // namespace

/// Reads a module from a file.  On error, messages are written to stderr
/// and null is returned.
static std::unique_ptr<Module> readModule(LLVMContext &Context,
                                          StringRef Name) {
  SMDiagnostic diag;
  std::unique_ptr<Module> m = parseIRFile(Name, diag, Context);
  if (!m)
    diag.print("llvm-module-split", errs());
  return m;
}

int main(int argc, char **argv) {
  CLOptions clOptions(argc, argv, true);

  // Override the default version printer.
  llvm::cl::SetVersionPrinter([](raw_ostream &os) {
    ProjectVersion version = getMojoVersion();
    os << "LLVM Module Split Tool:\n  ";
    os << "Mojo version: " << version.major << '.' << version.minor << '.'
       << version.patch << version.label << "\n  ";
    os << "Git SHA: " << version.revision << "\n  ";
    os << "Build config: " << version.buildType << "\n\n";

    // Print the host target config.
    llvm::sys::printDefaultTargetAndDetectedCPU(os);
    // Print all registered targets.
    llvm::TargetRegistry::printRegisteredTargetsForVersion(os);
  });

  // Enable command line options for various MLIR internals.
  llvm::cl::ParseCommandLineOptions(argc, argv);
  TraceProfiler tracer(clOptions.timeTrace, clOptions.timeTraceGranularity);

  LLVMModuleAndContext module;
  ErrorOrSuccess err = module.create(
      [&](LLVMContext &ctx) -> M::ErrorOr<std::unique_ptr<Module>> {
        if (std::unique_ptr<Module> module =
                readModule(ctx, clOptions.inputFilename))
          return module;
        return M::Error("could not load LLVM file");
      });
  if (err) {
    llvm::errs() << err.getError() << "\n";
    return -1;
  }

  std::unique_ptr<llvm::ToolOutputFile> output = nullptr;
  if (clOptions.outputPrefix == "-") {
    std::error_code error;
    output = std::make_unique<llvm::ToolOutputFile>(
        clOptions.outputPrefix, error, llvm::sys::fs::OF_None);
    if (error)
      exit(clOptions.options.reportError("Cannot open output file: '" +
                                         clOptions.outputPrefix +
                                         "':" + error.message()));
  }

  auto outputLambda =
      [&](llvm::unique_function<LLVMModuleAndContext()> produceModule,
          std::optional<int64_t> idx, unsigned numFunctionsBase) mutable {
        LLVMModuleAndContext subModule = produceModule();
        if (clOptions.outputPrefix == "-") {
          output->os() << "##############################################\n";
          if (idx)
            output->os() << "# [LLVM Module Split: submodule " << *idx << "]\n";
          else
            output->os() << "# [LLVM Module Split: main module]\n";
          output->os() << "##############################################\n";
          output->os() << *subModule;
          output->os() << "\n";
        } else {
          std::string outPath;
          if (!idx) {
            outPath = clOptions.outputPrefix + ".ll";
          } else {
            outPath =
                (clOptions.outputPrefix + "." + Twine(*idx) + ".ll").str();
          }
          auto outFile = mlir::openOutputFile(outPath);
          if (!outFile) {
            exit(clOptions.options.reportError("Cannot open output file: '" +
                                               outPath + "."));
          }
          outFile->os() << *subModule;
          outFile->keep();
          llvm::outs() << "Write llvm module to " << outPath << "\n";
        }
      };

  llvm::StringMap<llvm::GlobalValue::LinkageTypes> symbolLinkageTypes;
  if (clOptions.perFunctionSplit)
    splitPerFunction(std::move(module), outputLambda, symbolLinkageTypes);
  else
    splitPerExported(std::move(module), outputLambda);

  if (output)
    output->keep();
  return 0;
}
