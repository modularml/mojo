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
// This file provides various utilities for configuring and compiling Mojo.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLS_MOJO_COMMON_COMPILATION_H
#define KGEN_TOOLS_MOJO_COMMON_COMPILATION_H

#include "Mojo/ExecutionEngine/ExecutionEngine.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Mojo/ToolCommon/PassManagerConfigOptions.h"
#include "Support/Driver/DriverSupport.h"
#include "Support/ErrorOr.h"
#include "mlir/Support/Timing.h"
#include "llvm/Option/ArgList.h"

namespace llvm {
class SourceMgr;
} // namespace llvm

namespace mlir {
class PassManager;
} // namespace mlir

namespace M {
namespace AsyncRT {
class CPUDevice;
struct CPUDeviceOptions;
} // namespace AsyncRT

namespace KGEN::LIT {
struct ParserConfig;
} // namespace KGEN::LIT

class TargetInfoAttr;

/// Holds the option IDs that are common between mojo build and mojo run.
/// These are passed to parseCommonMojoArguments to avoid duplicating the
/// option ID mappings.
struct CommonOptionIDs {
  llvm::opt::OptSpecifier help;
  llvm::opt::OptSpecifier helpHidden;
  llvm::opt::OptSpecifier diagnosticFormat;
  llvm::opt::OptSpecifier disableWarnings;
  llvm::opt::OptSpecifier warningsAsErrors;
  llvm::opt::OptSpecifier noWarningsAsErrors;
  llvm::opt::OptSpecifier ignoreIncompatiblePrecompiledFileErrors;
  llvm::opt::OptSpecifier unknown;
  llvm::opt::OptSpecifier input;

  // Compilation options
  llvm::opt::OptSpecifier includeDirs;
  llvm::opt::OptSpecifier optimizationLevel;
  llvm::opt::OptSpecifier fpMode;
  llvm::opt::OptSpecifier debugLevel;
  llvm::opt::OptSpecifier sanitize;
  llvm::opt::OptSpecifier sharedLibasan;
  llvm::opt::OptSpecifier externalLibasan;
  llvm::opt::OptSpecifier bitcodeLibs;
  llvm::opt::OptSpecifier debugInfoLanguage;
  llvm::opt::OptSpecifier numThreads;
  llvm::opt::OptSpecifier mojoSearchPaths;
  llvm::opt::OptSpecifier loopUnrollingWarnThreshold;
  llvm::opt::OptSpecifier elaborationErrorLimit;
  llvm::opt::OptSpecifier elaborationErrorIncludePrelude;
  llvm::opt::OptSpecifier elaborationErrorVerbose;
  llvm::opt::OptSpecifier elaborationMaxDepth;

  // Target options
  llvm::opt::OptSpecifier targetTriple;
  llvm::opt::OptSpecifier targetCpu;
  llvm::opt::OptSpecifier targetFeatures;
  llvm::opt::OptSpecifier targetAbi;
  llvm::opt::OptSpecifier march;
  llvm::opt::OptSpecifier mcpu;
  llvm::opt::OptSpecifier mtune;
  llvm::opt::OptSpecifier targetAccelerator;
  llvm::opt::OptSpecifier mcmodel;
  llvm::opt::OptSpecifier largeDataThreshold;
  llvm::opt::OptSpecifier relocationModel;

  // Parser options
  llvm::opt::OptSpecifier diagnoseMissingDocStrings;
  llvm::opt::OptSpecifier maxNotes;
  llvm::opt::OptSpecifier defines;
  llvm::opt::OptSpecifier stripFilePrefix;
  llvm::opt::OptSpecifier disableBuiltins;
  llvm::opt::OptSpecifier fixit;
  llvm::opt::OptSpecifier exportFixit;

  // Stability options
  llvm::opt::OptSpecifier warnOnUnstableAPIs;
  llvm::opt::OptSpecifier ignoreDeprecated;

  // Linker options
  llvm::opt::OptSpecifier lldPath;
};

/// Configuration flags for common argument parsing behavior.
struct CommonParseConfig {
  /// If true, parse all arguments normally. If false (for `mojo run`), only
  /// parse arguments up to and including the input file, treating remaining
  /// arguments as program arguments to pass to the Mojo executable.
  bool parseAllArguments = true;

  /// If true, require exactly one input file. If false, allow zero or more.
  bool requireSingleInput = true;
};

/// Result of parsing common Mojo arguments.
struct CommonParseResult {
  /// If set, the caller should exit immediately with this code.
  std::optional<int> exitCode;

  /// The parsed argument list. For `mojo run`, this includes only arguments
  /// up to and including the input file.
  llvm::opt::InputArgList args;

  /// The parsed compilation options.
  KGEN::CompilationOptions compilationOptions;

  /// The parsed target information.
  TargetInfoAttr target;
};

/// Parse arguments common to both mojo build and mojo run.
///
/// This function extracts the common argument parsing logic shared between
/// the two commands, including:
/// - Diagnostic format parsing
/// - Unknown argument rejection
/// - Input file validation and opening
/// - Source manager setup
/// - Compilation option parsing
/// - Target option parsing
///
/// Note: Help text handling is intentionally left to the caller, as each
/// command has different help text files that cannot be easily parameterized.
/// Callers should check for help flags before calling this function.
///
/// Returns a CommonParseResult containing either an exit code (if parsing
/// failed) or the parsed arguments and options.
ErrorOr<CommonParseResult> parseCommonMojoArguments(
    State &state, llvm::SourceMgr &sourceManager, MLIRContext &ctx,
    const llvm::opt::PrecomputedOptTable &optTable,
    const CommonOptionIDs &optionIDs, const CommonParseConfig &config);

/// Parse the common configuration options for Mojo related to compilation,
/// populating the provided `compilationOptions` argument. An error is returned
/// if any of the provided option values are invalid.
ErrorOrSuccess parseCompilationOptions(
    const State &state, const llvm::opt::InputArgList &args,
    KGEN::CompilationOptions &compilationOptions, llvm::SourceMgr &sourceMgr,
    MLIRContext &ctx, llvm::opt::OptSpecifier includeDirsId,
    llvm::opt::OptSpecifier optimizationLevelId = {},
    llvm::opt::OptSpecifier debugLevelId = {},
    llvm::opt::OptSpecifier sanitizeId = {},
    llvm::opt::OptSpecifier sharedLibasan = {},
    llvm::opt::OptSpecifier externalLibasan = {},
    llvm::opt::OptSpecifier bitcodeLibs = {},
    llvm::opt::OptSpecifier debugInfoLanguageId = {},
    llvm::opt::OptSpecifier numThreadsId = {},
    llvm::opt::OptSpecifier stdLibPath = {},
    llvm::opt::OptSpecifier loopUnrollingWarnThresholdId = {},
    llvm::opt::OptSpecifier elaborationErrorLimitId = {},
    llvm::opt::OptSpecifier elaborationErrorIncludePreludeId = {},
    llvm::opt::OptSpecifier elaborationErrorVerbose = {},
    llvm::opt::OptSpecifier elaborationMaxDepth = {},
    llvm::opt::OptSpecifier ignoreIncompatiblePrecompiledFilesId = {},
    llvm::opt::OptSpecifier fpModeId = {},
    llvm::opt::OptSpecifier ignoreDeprecatedId = {});

/// Warn users when doing debug builds with a compiler in debug mode.
void warnBuildingForDebugWithDebugBuiltCompiler(
    const State &state,
    KGEN::CompilationOptions::DebugInfoLevel debugInfoLevel);

/// Parse the common configuration options for Mojo related to target info,
/// populating the provided `compilationOptions` argument. On success, `target`
/// is populated with the selected compilation target.
ErrorOrSuccess parseTargetOptions(
    const State &state, const llvm::opt::InputArgList &args,
    KGEN::CompilationOptions &compilationOptions, llvm::SourceMgr &sourceMgr,
    MLIRContext &ctx, TargetInfoAttr &target, llvm::opt::OptSpecifier tripleId,
    llvm::opt::OptSpecifier cpuId, llvm::opt::OptSpecifier featuresId,
    llvm::opt::OptSpecifier marchId, llvm::opt::OptSpecifier mcpuId,
    llvm::opt::OptSpecifier mtuneId,
    llvm::opt::OptSpecifier targetAcceleratorId,
    llvm::opt::OptSpecifier mcmodelId,
    llvm::opt::OptSpecifier largeDataThresholdId,
    llvm::opt::OptSpecifier relocationModelId = {},
    llvm::opt::OptSpecifier abiId = {});

/// This class holds the MLIR timing manager for one compiler command, and
/// gives the root scope that the steps of the compilation nest under. The
/// timing is off unless the `--mlir-timing` option is present. A command
/// without the option nests the parse under an empty scope, which records
/// nothing, and leaves `PassManagerConfigOptions::timingScope` unset, so no
/// pass manager gains timing instrumentation.
class MLIRPassTiming {
public:
  // `passManagerOptions` gives the address of `root` to other code. A move of
  // this object makes that address invalid. A copy is also not correct,
  // because each copy prints the same report again.
  MLIRPassTiming() = default;
  MLIRPassTiming(const MLIRPassTiming &) = delete;
  MLIRPassTiming &operator=(const MLIRPassTiming &) = delete;
  MLIRPassTiming(MLIRPassTiming &&) = delete;
  MLIRPassTiming &operator=(MLIRPassTiming &&) = delete;

  /// Makes the timing active if the `--mlir-timing` option is present. Sets
  /// the display mode from the `--mlir-timing-display` option. Gives an error
  /// if the display mode is not correct.
  ErrorOrSuccess configure(const llvm::opt::InputArgList &args,
                           llvm::opt::OptSpecifier timingId,
                           llvm::opt::OptSpecifier displayModeId);

  /// The root scope for the steps of the compilation. The scope is empty if
  /// the timing is off.
  mlir::TimingScope &rootScope() { return root; }

  /// The pass manager configuration that sends the times to this scope. The
  /// scope stays empty if the timing is off. Then a build that has the trace
  /// function keeps its own timing manager.
  KGEN::PassManagerConfigOptions passManagerOptions();

  /// Prints the report and stops the timing, for a command that keeps working
  /// after the compilation. `mojo run` executes the program next: the wall
  /// time of the program does not belong in the report, and an `exit()` from
  /// the program would suppress the report. A command that ends with the
  /// compilation leaves this to the destructor.
  void finish();

  ~MLIRPassTiming() { finish(); }

private:
  mlir::DefaultTimingManager manager;
  // `root` comes after the manager in this declaration. Thus the program
  // destroys `root` first, and the manager sees a scope that is not active if
  // it prints from its own destructor.
  mlir::TimingScope root;
};

/// This class turns on the pass timing of LLVM for one compiler command.
/// `StandardInstrumentations` and the legacy pass manager already install the
/// instrumentation that collects the times. LLVM keeps the times in timer
/// groups that are global to the process. Therefore this class only starts the
/// collection and prints the report.
class LLVMPassTiming {
public:
  // The destructor prints the report and changes state that is global to the
  // process. A copy or a move of this object prints the report again.
  LLVMPassTiming() = default;
  LLVMPassTiming(const LLVMPassTiming &) = delete;
  LLVMPassTiming &operator=(const LLVMPassTiming &) = delete;
  LLVMPassTiming(LLVMPassTiming &&) = delete;
  LLVMPassTiming &operator=(LLVMPassTiming &&) = delete;

  /// Turns on the timing if the command has the `--llvm-timing` option. Also
  /// sets the compilation to one thread, because the timers of LLVM are
  /// global to the process and are not safe for more than one thread. Call
  /// this before the program makes the CPU device from `options`.
  void configure(const llvm::opt::InputArgList &args,
                 llvm::opt::OptSpecifier timingId,
                 KGEN::CompilationOptions &options);

  /// Prints the report to stderr and clears the timers. Does nothing if the
  /// timing is off. Does nothing on a second call. The destructor calls this
  /// function for a command that ends with the compilation.
  void finish();

  ~LLVMPassTiming() { finish(); }

private:
  bool enabled = false;
};

/// Wrap a parser invocation to Mojo, populating the necessary parsing context,
/// and attaching post parse metadata. On success, returns the parsed module
/// operation. If the `autoFixIt` flag is set and the parser collects any
/// fix-its, they will be applied, and the returned module will be null.
/// If `exportFixit` is set, fix-its will be exported to a YAML file instead.
ErrorOr<OwningOpRef<ModuleOp>> invokeMojoParser(
    const State &state, const llvm::opt::InputArgList &args,
    KGEN::CompilationOptions &compilationOptions, MLIRContext *ctx,
    AsyncRT::CPUDevice &cpuDevice, llvm::opt::OptSpecifier docDiagnoseMissingId,
    llvm::opt::OptSpecifier maxNotesId, llvm::opt::OptSpecifier definesId,
    llvm::opt::OptSpecifier stripFilePrefixId,
    llvm::opt::OptSpecifier disableBuiltins, llvm::opt::OptSpecifier stdlibPath,
    llvm::opt::OptSpecifier autoFixIt, llvm::opt::OptSpecifier exportFixit,
    mlir::TimingScope *timingScope,
    function_ref<OwningOpRef<ModuleOp>(KGEN::LIT::ParserConfig &,
                                       mlir::TimingScope &)>
        parseFn);

/// Configure cpuDevice options based on compilation options.
/// Currently handles thread pool configuration based on numThreads.
void configureCPUDeviceOptions(AsyncRT::CPUDeviceOptions &cpuDeviceOptions,
                               const KGEN::CompilationOptions &options);

} // namespace M

#endif // KGEN_TOOLS_MOJO_COMMON_COMPILATION_H
