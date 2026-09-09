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

#ifndef SUPPORT_COMMONCLOPTIONS_H
#define SUPPORT_COMMONCLOPTIONS_H

#include "Support/CommandLine.h"
#include "Support/Config.h"
#include "Support/ErrorOr.h"
#include "Support/FileSystemExtras.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/FileUtilities.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace llvm {
class ToolOutputFile;
}

namespace M {

/// Opens an intermediate text file for writing if max.temps_dir config is set.
/// The filename is generated from `baseName` and `extension`, with a
/// numerical suffix added if the file already exists to avoid overwriting.
/// Returns nullptr if max.temps_dir is not set or on error.
/// Caller is responsible for calling keep() on the returned file after writing.
std::unique_ptr<llvm::ToolOutputFile>
openIntermediateTextFile(StringRef baseName, StringRef extension);

class OptionsBase {
public:
  StringRef getProgramName() const { return programName; }
  int reportError(const Twine &errorMessage) const {
    llvm::errs() << programName << ": " << errorMessage << "\n";
    return EXIT_FAILURE;
  }

  /// This is the value of argv[0] when the program launches, used for reporting
  /// error messages.
  StringRef programName;
  /// This tells LLVM to print stack traces on crashes, and also handles
  /// multibyte command line options on windows.
  std::optional<llvm::InitLLVM> llvmInitializer;
};

/// Contains functionality that's common to all tools.
class CLOptionsBase {
public:
  OptionsBase &options;
  /// When the 'skipInitLLVM' flag is true, this initializer does not call
  /// InitLLVM.
  CLOptionsBase(int &argc, char **argv, OptionsBase &o,
                bool skipInitLLVM = false)
      : options(o) {
    if (!skipInitLLVM)
      options.llvmInitializer.emplace(argc, argv);
    // On windows, InitLLVM may mutate argv, so make sure to get the fresh
    // value.
    options.programName = argv[0];

    static constexpr StringLiteral bugReportMsg =
        "PLEASE submit a bug report to "
        "https://github.com/modular/modular/issues and include the crash "
        "backtrace.\n";

    llvm::setBugReportMsg(bugReportMsg.data());
  }
};

class CommonOptions : public OptionsBase {

public:
  bool verifyDiagnostics{false};
  // Specify the input file for a given binary
  std::string inputFilename{"-"};
  // Specify the alignment for a given binary file.
  int inputFileAlignment{0};

  /// Open the filename specified on the command line and return a memory
  /// buffer, or an error message on failure.
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
  openInputFile(StringRef inputFile,
                std::optional<llvm::Align> align = std::nullopt) {
    align = (inputFileAlignment != 0) ? llvm::Align(inputFileAlignment) : align;
    return M::openInputFile(inputFile, align.value_or(defaultAlignment));
  }

  /// The common case for all our driver-like tools is to fail early with an
  /// exit error status.  This takes care of that bit of boilerplate.
  /// Takes an optional alignment with priority:
  /// CLI alignment > align argument > default alignment.
  std::unique_ptr<llvm::MemoryBuffer>
  openInputFileOrExit(std::optional<llvm::Align> align = std::nullopt) {
    return openInputFileOrExit(inputFilename, align);
  }

  std::unique_ptr<llvm::MemoryBuffer>
  openInputFileOrExit(StringRef inputFile,
                      std::optional<llvm::Align> align = std::nullopt) {
    auto errorOrInputFile = openInputFile(inputFile, align);
    if (failed(errorOrInputFile))
      exit(reportError(Twine(errorOrInputFile.getError())));
    return errorOrInputFile.takeValue();
  }

  //===--------------------------------------------------------------------===//
  // Emission Options
  //===--------------------------------------------------------------------===//

  std::string outputFilename{"-"};

  /// Determine an output file name and open it.
  std::unique_ptr<llvm::ToolOutputFile>
  getOutputFile(bool hasBinaryOutput, StringRef fileExtension = ".mef") const;

  /// This method creates an MLIR context with the specified memory buffer as
  /// the primary file configured in the source mgr.  It configures it for
  /// diagnostic printing based on the setting of the -verify-diagnostics flag.
  /// This invokes the `bodyFn` callable with the MLIRContext that is set up.
  template <typename BodyFn>
  LogicalResult
  configureMLIRContextAndExecute(std::unique_ptr<llvm::MemoryBuffer> &&buffer,
                                 BodyFn &&bodyFn) const {
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
    return configureMLIRContextAndExecute(sourceMgr,
                                          std::forward<BodyFn>(bodyFn));
  }

  // This method creates an MLIR context with the specified memory buffer as
  // the primary file configured in the source mgr.  It configures it for
  // diagnostic printing based on the setting of the -verify-diagnostics flag.
  // This invokes the `bodyFn` callable with the MLIRContext and SourceMgr that
  // is set up.
  template <typename BodyFn>
  LogicalResult configureMLIRContextAndSourceMgrAndExecute(
      std::unique_ptr<llvm::MemoryBuffer> &&buffer, BodyFn &&bodyFn) const {
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());
    return configureMLIRContextAndExecute(
        sourceMgr,
        [&sourceMgr, bodyFn = std::forward<BodyFn>(bodyFn)](
            mlir::MLIRContext *ctx) -> LogicalResult {
          return bodyFn(ctx, sourceMgr);
        });
  }

  // This method creates an MLIR context and configures it for diagnostic
  // printing based on the setting of the -verify-diagnostics flag.  This
  // invokes the `bodyFn` callable with the MLIRContext that is set up.
  template <typename BodyFn>
  LogicalResult configureMLIRContextAndExecute(llvm::SourceMgr &sourceMgr,
                                               BodyFn &&bodyFn) const {
    mlir::MLIRContext context{mlir::MLIRContext::Threading::DISABLED};
    configureMLIRContext(context);

    if (verifyDiagnostics) {
      mlir::SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr,
                                                                &context);
      // If diagnostic verification is enabled, we don't propagate the
      // result.
      (void)bodyFn(&context);
      return sourceMgrHandler.verify();
    }

    mlir::SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    return bodyFn(&context);
  }

  //===--------------------------------------------------------------------===//
  // Intermediate Files Options
  //===--------------------------------------------------------------------===//

  bool saveTemps{false};

  // Whether to use local scope when printing MLIR dumps.
  // Local scope prevents accesses and printing to global module resources.
  bool printTempsLocalScope{false};

  std::string tempsDir{""};

  /// Determine an intermediate file with extension `ext` and open it.
  std::unique_ptr<llvm::ToolOutputFile>
  getIntermediateFile(StringRef inputName, StringRef ext) const;

  LogicalResult emitArchive(StringRef object) const;

private:
  /// Default alignment for input files.
  /// Used only when both client code and CLI do not specify alignment.
  static constexpr llvm::Align defaultAlignment = llvm::Align::Constant<64>();
};

/// Contains command-line options that are shared among most of our binaries.
class CommonCLOptions : public CLOptionsBase {
public:
  CommonOptions &options;
  CommonCLOptions(int argc, char **argv, CommonOptions &o,
                  bool skipInitLLVM = false)
      : CLOptionsBase(argc, argv, o, skipInitLLVM), options(o) {}

private:
  llvm::cl::OptionCategory CommonOptionsCategory{"Common command line options"};
  M::cl::MOpt<bool, true> verifyDiagnosticsOpt{
      "verify-diagnostics",
      cl::desc("Check that emitted diagnostics match "
               "expected-* lines on the corresponding line"),
      llvm::cl::location(options.verifyDiagnostics),
      llvm::cl::cat(CommonOptionsCategory)};

  // Specify the input file for a given binary
  M::cl::MOpt<std::string, true> inputFilenameOpt{
      llvm::cl::Positional, cl::desc("<input file>"),
      llvm::cl::location(options.inputFilename),
      llvm::cl::cat(CommonOptionsCategory)};

  // Specify the alignment for a given binary file.
  M::cl::MOpt<int, true> inputFileAlignmentOpt{
      "input-file-alignment", cl::desc("Alignment for opening input file"),
      llvm::cl::location(options.inputFileAlignment),
      llvm::cl::cat(CommonOptionsCategory)};

  //===--------------------------------------------------------------------===//
  // Emission Options
  //===--------------------------------------------------------------------===//

  M::cl::MOpt<std::string, true> outputFilenameOpt{
      "o", cl::desc("Output filename"), cl::value_desc("filename"),
      llvm::cl::location(options.outputFilename),
      llvm::cl::cat(CommonOptionsCategory)};

  //===--------------------------------------------------------------------===//
  // Intermediate Files Options
  //===--------------------------------------------------------------------===//

  M::cl::MOpt<bool, true> saveTempsOpt{
      "save-temps",
      cl::desc("Store the usual 'temporary' intermediate files permanently in "
               "the directory specified by -temps-dir (defaults to the output "
               "directory); name them as auxiliary output files."),
      llvm::cl::Optional, llvm::cl::location(options.saveTemps),
      llvm::cl::cat(CommonOptionsCategory)};

  M::cl::MOpt<std::string, true> tempsDirOpt{
      "temps-dir",
      cl::desc(
          "The directory in which to store 'temporary' intermediate files. No "
          "files will be saved here unless `-save-temps` is also specified."),
      llvm::cl::location(options.tempsDir), llvm::cl::Optional,
      llvm::cl::cat(CommonOptionsCategory)};
};

} // namespace M

#endif // SUPPORT_COMMONCLOPTIONS_H
