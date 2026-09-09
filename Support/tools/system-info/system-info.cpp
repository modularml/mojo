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

#include "Support/CommandLine.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/MArchTarget/Host.h"
#include "Support/MArchTarget/MArchTargetMinimal.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/MDialect/MDialect.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>

using namespace M;
using namespace llvm;

namespace {

enum class OutputFormat { YAML, JSON };

struct SystemInfoCLIOptions {

  llvm::cl::opt<std::string> outputFilename{
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-")};

  llvm::cl::opt<std::string> arch{
      "march", llvm::cl::desc("Architecture to show info for."),
      llvm::cl::init("")};

  llvm::cl::opt<std::string> cpu{
      "mcpu", llvm::cl::desc("CPU to show info for."), llvm::cl::init("")};

  llvm::cl::opt<OutputFormat> format{
      "format", llvm::cl::desc("Format. Defaults to yaml"),
      M::cl::values(clEnumValN(OutputFormat::YAML, "yaml", "YAML format")),
      M::cl::values(clEnumValN(OutputFormat::JSON, "json", "JSON format")),
      llvm::cl::init(OutputFormat::YAML)};

  llvm::cl::list<HostProperty> QueryProperty{
      "query", M::cl::desc("Available Queries:"),
      M::cl::values(
          clEnumValN(HostProperty::TargetTriple, "target-triple",
                     "Host target triple"),
          clEnumValN(HostProperty::OS, "os", "Host operating system"),
          clEnumValN(HostProperty::Arch, "arch", "Host CPU architecture"),
          clEnumValN(HostProperty::CPUModel, "cpu-model",
                     "Host CPU model name"),
          clEnumValN(HostProperty::SIMDBitWidth, "simd-bitwidth",
                     "Host CPU SIMD bitwidth"),
          clEnumValN(HostProperty::Features, "features",
                     "Host CPU features printed as comma-separated values"),
          clEnumValN(HostProperty::CoreCount, "core-count",
                     "Host number of cores"),
          clEnumValN(HostProperty::L1CacheSize, "l1-cache-size",
                     "Host L1 DCache size"),
          clEnumValN(HostProperty::L2CacheSize, "l2-cache-size",
                     "Host L2 DCache size"),
          clEnumValN(HostProperty::L3CacheSize, "l3-cache-size",
                     "Host L3 DCache size"),
          clEnumValN(HostProperty::L4CacheSize, "l4-cache-size",
                     "Host L4 DCache size"),
          clEnumValN(HostProperty::Affinities, "affinities",
                     "Preferred CPU ids for numPhysicalCores threads if both "
                     "CPUSystemInfo and thread affinities are supported.")),
      llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated};
};
} // namespace

static int reportError(const Twine &errorMessage) {
  llvm::errs() << "system-info: " << errorMessage << "\n";
  return EXIT_FAILURE;
}

int main(int argc, char **argv) {
  SystemInfoCLIOptions cli;

  llvm::cl::ParseCommandLineOptions(argc, argv, "Modular System Info Tool");

  auto outFilePathStr = cli.outputFilename.getValue();
  OutputFormat format = cli.format.getValue();

  std::error_code ec;
  if (outFilePathStr != "-" && !std::filesystem::exists(outFilePathStr, ec) &&
      !ec) {
    auto outFilePath = std::filesystem::path(outFilePathStr);
    if (outFilePath.has_parent_path())
      std::filesystem::create_directories(outFilePath.parent_path(), ec);
  }
  // If anything failed, report the failure.
  if (ec)
    exit(reportError("std::filesystem: " + ec.message() + ": " +
                     outFilePathStr));

  std::error_code error;
  auto outputFile = std::make_unique<llvm::ToolOutputFile>(
      outFilePathStr, error, llvm::sys::fs::OF_None);
  if (error)
    exit(reportError("Cannot open output file: '" + outFilePathStr +
                     "': " + error.message()));

  HostMachineInfo hostInfo;

  // If arch/cpu is specified generate hostInfo from that.
  if (!cli.arch.empty()) {
    // Initialize the LLVM targets so we can look up the current target machine.
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();

    MLIRContext ctx{MLIRContext::Threading::DISABLED};
    ctx.loadDialect<MDialect>();
    std::string cpu = cli.cpu;
    if (cpu.empty())
      cpu = cli.arch;
    std::string hostTriple = llvm::sys::getDefaultTargetTriple();
    auto targetInfoOr =
        M::getMArchTargetInfo(hostTriple, cli.arch, cpu, /*mtune=*/{});
    if (targetInfoOr.isError())
      return reportError(targetInfoOr.getError());
    hostInfo = HostMachineInfo::fromTargetInfo(*targetInfoOr);
  } else {
    // Get info from host machine.
    auto hostMachineOr = getHostMachineInfo();
    if (hostMachineOr.isError())
      return reportError(hostMachineOr.getError());

    hostInfo = hostMachineOr.takeValue();
  }

  if (format == OutputFormat::YAML) {
    auto &os = outputFile ? outputFile->os() : llvm::outs();
    if (cli.QueryProperty.empty())
      hostInfo.print(os);

    for (auto query : cli.QueryProperty)
      hostInfo.print(query, os);
    os.flush();
    outputFile->keep();
    return EXIT_SUCCESS;
  }

  auto os = outputFile ? llvm::json::OStream(outputFile->os())
                       : llvm::json::OStream(llvm::outs());
  if (cli.QueryProperty.empty())
    hostInfo.print(os);
  else {
    os.objectBegin();
    for (auto query : cli.QueryProperty)
      hostInfo.print(query, os);
    os.objectEnd();
  }
  os.flush();
  outputFile->keep();
  return EXIT_SUCCESS;
}
