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

#include "Compilation.h"
#include "Mojo/Compiler/KGENCompiler.h"
#include "Mojo/Compiler/ObjectCompiler.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/MojoParser/EntryPoint.h"
#include "Mojo/Support/Configuration.h"
#include "Mojo/ToolCommon/InitAllDialects.h"
#include "Mojo/ToolCommon/PipelineTiming.h"
#include "Support/Compiler/Diags.h"
#include "Support/MArchTarget/MArchTarget.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/Timing.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/PassTimingInfo.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Timer.h"
#include "llvm/Target/TargetMachine.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

ErrorOr<CommonParseResult> M::parseCommonMojoArguments(
    State &state, llvm::SourceMgr &sourceManager, MLIRContext &ctx,
    const llvm::opt::PrecomputedOptTable &optTable,
    const CommonOptionIDs &optionIDs, const CommonParseConfig &config) {
  CommonParseResult result;

  // Parse arguments based on the configuration.
  unsigned unused = 0;

  // Handle the special case for `mojo run` where we need to find the input
  // argument and only parse up to that point.
  llvm::opt::InputArgList args(nullptr, nullptr);
  if (!config.parseAllArguments) {
    llvm::opt::InputArgList allArgs =
        optTable.ParseArgs(state.arguments, unused, unused);

    // LLVMOption treats all "positional arguments" (arguments that do not have
    // a "-" or "--" prefix) as `INPUT`. The very first of these is our Mojo
    // source file, and each remaining positional argument is an argument being
    // passed to the Mojo executable produced from that source file.
    auto inputArgs = allArgs.filtered(optionIDs.input);
    if (inputArgs.empty()) {
      return Error("no input file provided");
    }

    // We now have the index of the Mojo source file argument, so we can parse
    // the arguments up to and including that argument "normally."
    args = optTable.ParseArgs(
        state.arguments.slice(0, (*inputArgs.begin())->getIndex() + 1), unused,
        unused);
  } else {
    // Parse all arguments normally for `mojo build`.
    args = optTable.ParseArgs(state.arguments, unused, unused);
  }

  // Parse diagnostic format arguments.
  if (int exitCode = state.parseDiagnosticFormatArguments(
          args, optionIDs.diagnosticFormat, optionIDs.disableWarnings,
          optionIDs.warningsAsErrors, optionIDs.noWarningsAsErrors)) {
    result.exitCode = exitCode;
    return result;
  }

  // Reject unknown arguments.
  if (int exitCode = state.rejectUnknownArguments(args, optionIDs.unknown)) {
    result.exitCode = exitCode;
    return result;
  }

  // Validate input file arguments based on configuration.
  if (!args.hasArg(optionIDs.input)) {
    if (config.requireSingleInput) {
      return Error("no input file provided");
    }
  } else if (config.requireSingleInput &&
             args.hasMultipleArgs(optionIDs.input)) {
    std::vector<std::string> inputs = args.getAllArgValues(optionIDs.input);
    return Error(llvm::formatv(
        "too many input files, cannot process both '{0}' and '{1}'", inputs[0],
        inputs[1]));
  }

  // Open the provided input file path, or exit with an error if it's not a
  // valid argument that can be opened.
  if (args.hasArg(optionIDs.input)) {
    auto bufferOrErr = openMojoInputFile(args.getLastArgValue(optionIDs.input));
    if (failed(bufferOrErr))
      return bufferOrErr.takeError();

    // Initialize the source manager with the input file buffer, as well as the
    // appropriate diagnostic handler.
    sourceManager.setDiagHandler(getDiagHandler(state.diagnosticFormat));
    sourceManager.AddNewSourceBuffer(std::move(*bufferOrErr), llvm::SMLoc());
  }

  // Parse compilation options.
  if (ErrorOrSuccess err = parseCompilationOptions(
          state, args, result.compilationOptions, sourceManager, ctx,
          optionIDs.includeDirs, optionIDs.optimizationLevel,
          optionIDs.debugLevel, optionIDs.sanitize, optionIDs.sharedLibasan,
          optionIDs.externalLibasan, optionIDs.bitcodeLibs,
          optionIDs.debugInfoLanguage, optionIDs.numThreads,
          optionIDs.mojoSearchPaths, optionIDs.loopUnrollingWarnThreshold,
          optionIDs.elaborationErrorLimit,
          optionIDs.elaborationErrorIncludePrelude,
          optionIDs.elaborationErrorVerbose, optionIDs.elaborationMaxDepth,
          optionIDs.ignoreIncompatiblePrecompiledFileErrors, optionIDs.fpMode,
          optionIDs.ignoreDeprecated))
    return err.takeError();

  // Parse target options.
  if (ErrorOrSuccess err = parseTargetOptions(
          state, args, result.compilationOptions, sourceManager, ctx,
          result.target, optionIDs.targetTriple, optionIDs.targetCpu,
          optionIDs.targetFeatures, optionIDs.march, optionIDs.mcpu,
          optionIDs.mtune, optionIDs.targetAccelerator, optionIDs.mcmodel,
          optionIDs.largeDataThreshold, optionIDs.relocationModel,
          optionIDs.targetAbi))
    return err.takeError();

  // Parse stability options.
  if (optionIDs.warnOnUnstableAPIs.isValid())
    result.compilationOptions.warnOnUnstableAPIs =
        args.hasArg(optionIDs.warnOnUnstableAPIs);

  // Apply the lld path override (equivalent to MODULAR_MOJO_MAX_LLD_PATH). This
  // installs a process-wide override so every later `MojoConfig::open()` (for
  // example, in the object compiler) observes it.
  if (optionIDs.lldPath.isValid()) {
    StringRef lldPath = args.getLastArgValue(optionIDs.lldPath);
    if (!lldPath.empty())
      KGEN::MojoConfig::setLLDPathOverride(lldPath);
  }

  // Store the parsed args for the caller.
  result.args = std::move(args);

  // Success - no exit code, continue with compilation.
  return result;
}

ErrorOrSuccess M::parseCompilationOptions(
    const State &state, const llvm::opt::InputArgList &args,
    CompilationOptions &compilationOptions, llvm::SourceMgr &sourceMgr,
    MLIRContext &ctx, llvm::opt::OptSpecifier includeDirsId,
    llvm::opt::OptSpecifier optimizationLevelId,
    llvm::opt::OptSpecifier debugLevelId, llvm::opt::OptSpecifier sanitizeId,
    llvm::opt::OptSpecifier sharedLibasan,
    llvm::opt::OptSpecifier externalLibasan,
    llvm::opt::OptSpecifier bitcodeLibs,
    llvm::opt::OptSpecifier debugInfoLanguageId,
    llvm::opt::OptSpecifier numThreadsId, llvm::opt::OptSpecifier stdLibPath,
    llvm::opt::OptSpecifier loopUnrollingWarnThresholdId,
    llvm::opt::OptSpecifier elaborationErrorLimitId,
    llvm::opt::OptSpecifier elaborationErrorIncludePreludeId,
    llvm::opt::OptSpecifier elaborationErrorVerboseId,
    llvm::opt::OptSpecifier elaborationMaxDepthId,
    llvm::opt::OptSpecifier ignoreIncompatiblePrecompiledFileErrorsId,
    llvm::opt::OptSpecifier fpModeId,
    llvm::opt::OptSpecifier ignoreDeprecatedId) {

  // Parse the deprecation-warning ignore list.
  if (ignoreDeprecatedId.isValid())
    compilationOptions.ignoredDeprecations = llvm::to_vector_of<std::string>(
        args.getAllArgValues(ignoreDeprecatedId));

  // Process the sanitizers.
  if (sanitizeId.isValid()) {
    StringRef sanitizer = args.getLastArgValue(sanitizeId);
    if (args.hasMultipleArgs(sanitizeId))
      return Error("too many specified sanitizers, expected exactly one");
    if (!sanitizer.empty()) {
      if (!llvm::is_contained({"address", "thread"}, sanitizer)) {
        return Error("invalid sanitizer '" + sanitizer +
                     "', expected one of: `address` or `thread`");
      }
      if (sanitizer == "address")
        compilationOptions.sanitizers.enable(Sanitizers::kAddress);
      else if (sanitizer == "thread")
        compilationOptions.sanitizers.enable(Sanitizers::kThread);
    }
  }

  if (sharedLibasan.isValid() && args.hasArg(sharedLibasan)) {
    if (!compilationOptions.sanitizers.has(Sanitizers::kAddress))
      return Error(
          "cannot enable --shared-libasan without enabling --sanitize=address");
    compilationOptions.sharedLibasan = true;
  }

  if (externalLibasan.isValid()) {
    StringRef libPath = args.getLastArgValue(externalLibasan);
    if (args.hasMultipleArgs(externalLibasan))
      return Error("too many external libasan paths, expected exactly one");
    if (!libPath.empty()) {
      if (compilationOptions.sharedLibasan)
        return Error("--external-libasan cannot be used with --shared-libasan");
      if (!compilationOptions.sanitizers.has(Sanitizers::kAddress))
        return Error("cannot use --external-libasan without enabling "
                     "--sanitize=address");
      compilationOptions.externalLibasan = libPath;
    }
  }

  if (bitcodeLibs.isValid()) {
    compilationOptions.bitcodeLibs =
        llvm::to_vector_of<std::string>(args.getAllArgValues(bitcodeLibs));
  }

  // Enable overwriting of the auto-imported paths, which is where the compiler
  // first looks for builtins.
  if (stdLibPath.isValid()) {
    StringRef value = args.getLastArgValue(stdLibPath);
    compilationOptions.searchPaths = value;
  }

  // Process the debug info language.
  if (debugInfoLanguageId.isValid()) {
    StringRef debugInfoLanguage = args.getLastArgValue(debugInfoLanguageId);
    if (args.hasMultipleArgs(debugInfoLanguageId))
      return Error(
          "too many specified debug info languages, expected exactly one");
    if (!debugInfoLanguage.empty()) {
      if (!llvm::is_contained({"C", "Mojo"}, debugInfoLanguage)) {
        return Error("invalid debug info language '" + debugInfoLanguage +
                     "', expected one of: `C` or `Mojo`");
      }
      if (debugInfoLanguage == "C")
        compilationOptions.debugInfoLanguage = CompilationOptions::kLangC;
    }
  }

  // Set up the optimization level.
  if (optimizationLevelId.isValid()) {
    StringRef levelStr = args.getLastArgValue(optimizationLevelId, "3");
    int level = -1;
    if (levelStr.size() == 1)
      level = levelStr[0] - '0';
    if (0 <= level && level <= 3) {
      compilationOptions.optimizationLevel = level;
    } else {
      return Error(llvm::formatv("invalid optimization level '{0}', expected "
                                 "number from 0-3 inclusive",
                                 levelStr));
    }
  }

  // Set up the floating-point mode from a list of `feature=on|off` items.
  if (fpModeId.isValid()) {
    for (StringRef value : args.getAllArgValues(fpModeId)) {
      SmallVector<StringRef> items;
      value.split(items, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
      for (StringRef item : items) {
        switch (parseFpModeItem(item, compilationOptions.fpMode)) {
        case FpModeItemStatus::Ok:
          break;
        case FpModeItemStatus::UnknownFeature:
          return Error(
              llvm::formatv("invalid fp-mode '{0}', the only supported "
                            "feature is 'contract'",
                            item));
        case FpModeItemStatus::InvalidValue:
          return Error(llvm::formatv("invalid fp-mode '{0}', expected "
                                     "'contract=fast' or 'contract=off'",
                                     item));
        }
      }
    }
  }

  // Setup the debug level.
  if (debugLevelId.isValid()) {
    StringLiteral kDebugLevelNone = "none";
    StringLiteral kDebugLevelLineTables = "line-tables";
    StringLiteral kDebugLevelFull = "full";
    StringRef level = args.getLastArgValue(debugLevelId, kDebugLevelNone);
    if (!llvm::is_contained(
            {kDebugLevelNone, kDebugLevelLineTables, kDebugLevelFull}, level)) {
      return Error(llvm::formatv("invalid debug level '{0}', expected one of: "
                                 "`{1}` (the default value), `{2}`, or `{3}`",
                                 level, kDebugLevelNone, kDebugLevelLineTables,
                                 kDebugLevelFull));
    }
    compilationOptions.debugLevel =
        llvm::StringSwitch<CompilationOptions::DebugInfoLevel>(level)
            .Case(kDebugLevelNone, CompilationOptions::kNoDebug)
            .Case(kDebugLevelLineTables, CompilationOptions::kLineTablesOnly)
            .Case(kDebugLevelFull, CompilationOptions::kFullDebugInfo);
  }

  if (numThreadsId.isValid()) {
    if (args.hasMultipleArgs(numThreadsId)) {
      return Error("Number of threads can only be specified once");
    }
    StringRef numThreadsStr = args.getLastArgValue(numThreadsId);
    if (!numThreadsStr.empty()) {
      // Next line has a side effect of setting compilationOptions.numThreads
      if (!llvm::to_integer(numThreadsStr, compilationOptions.numThreads))
        return Error("Invalid number of threads: " + numThreadsStr.str());
    }
  }

  if (loopUnrollingWarnThresholdId.isValid()) {
    if (args.hasMultipleArgs(loopUnrollingWarnThresholdId)) {
      return Error("too many specified loop unroll factor threshold, expected "
                   "exactly one");
    }
    StringRef loopUnrollingWarnThreshold =
        args.getLastArgValue(loopUnrollingWarnThresholdId);
    if (!loopUnrollingWarnThreshold.empty()) {
      if (!llvm::to_integer(loopUnrollingWarnThreshold,
                            compilationOptions.loopUnrollingWarnThreshold)) {
        return Error("invalid loop-unrolling-warn-threshold'" +
                     loopUnrollingWarnThreshold +
                     "', expected an integer number");
      }
    }
  }

  if (elaborationErrorLimitId.isValid()) {
    if (args.hasMultipleArgs(elaborationErrorLimitId)) {
      return Error(
          "too many specified elaboration-error-limit, expected exactly one");
    }

    StringRef elabErrorLimit = args.getLastArgValue(elaborationErrorLimitId);
    if (!elabErrorLimit.empty()) {
      if (!llvm::to_integer(elabErrorLimit,
                            compilationOptions.elaborationErrorLimit)) {
        return Error("invalid elaboration-error-limit'" + elabErrorLimit +
                     "', expected an integer number");
      }
    }
  }

  if (elaborationErrorIncludePreludeId.isValid()) {
    if (args.hasMultipleArgs(elaborationErrorIncludePreludeId)) {
      return Error("too many specified elaboration-error-include-prelude , "
                   "expected exactly one");
    }
    compilationOptions.elaborationErrorIncludePrelude =
        args.hasArg(elaborationErrorIncludePreludeId);
  }

  if (elaborationErrorVerboseId.isValid()) {
    if (args.hasMultipleArgs(elaborationErrorVerboseId)) {
      return Error(
          "too many specified elaboration-error-verbose, expected exactly one");
    }

    StringLiteral kNoParams = "no-params";
    StringLiteral kSimpleParams = "simple-params";
    StringLiteral kAllParams = "all-params";

    StringRef elabErrorVerbose =
        args.getLastArgValue(elaborationErrorVerboseId, kSimpleParams);

    if (!llvm::is_contained({kNoParams, kSimpleParams, kAllParams},
                            elabErrorVerbose)) {

      return Error(llvm::formatv(
          "invalid elaboration-error-verbose '{0}', expected one of: "
          "`{1}` (the default value), `{2}`, or `{3}`",
          elabErrorVerbose, kNoParams, kSimpleParams, kAllParams));
    }

    compilationOptions.elaborationErrorVerbose =
        llvm::StringSwitch<CompilationOptions::ErrorVerboseLevel>(
            elabErrorVerbose)
            .Case(kNoParams, CompilationOptions::kNoParams)
            .Case(kSimpleParams, CompilationOptions::kSimpleParams)
            .Case(kAllParams, CompilationOptions::kAllParams);
  }

  if (elaborationMaxDepthId.isValid()) {
    if (args.hasMultipleArgs(elaborationMaxDepthId)) {
      return Error(
          "too many specified elaboration-max-depth, expected exactly one");
    }

    StringRef elabMaxDepth = args.getLastArgValue(elaborationMaxDepthId);
    if (!elabMaxDepth.empty()) {
      if (!llvm::to_integer(elabMaxDepth,
                            compilationOptions.elaborationMaxDepth)) {
        return Error("invalid elaboration-max-depth'" + elabMaxDepth +
                     "', expected an unsigned number");
      }
    }
  }

  compilationOptions.ignoreIncompatiblePrecompiledFileErrors =
      args.hasArg(ignoreIncompatiblePrecompiledFileErrorsId);

  // Unike other command line options, disableWarnings is parsed
  // in State::parseDiagnosticFormatArguments(), because there is another
  // warning emission mechanism in State::reportWarning(), and the option
  // is single-sourced to affect both places.
  compilationOptions.disableWarnings = state.areWarningsDisabled();
  compilationOptions.warningsAsErrors = state.areWarningsAsErrors();

  sourceMgr.setIncludeDirs(args.getAllArgValues(includeDirsId));

  // Initialize the MLIR context.
  DialectRegistry registry;
  registerAllKGENDialects(registry);
  registerKGENToLLVMTranslation(registry);
  ctx.appendDialectRegistry(registry);
  ctx.loadDialect<MDialect>();

  return success();
}

void M::warnBuildingForDebugWithDebugBuiltCompiler(
    const State &state,
    KGEN::CompilationOptions::DebugInfoLevel debugInfoLevel) {
#ifdef MODULAR_DEBUG
  if (debugInfoLevel == M::KGEN::CompilationOptions::kFullDebugInfo)
    state.reportWarning(
        "Performing a debug build with the Mojo compiler built in debug mode:\n"
        "It is not necessary to use a debug build of the Mojo compiler to "
        "produce debuggable Mojo programs.\n"
        "You can safely use a release build of the compiler for this, which "
        "will result in faster build times.\n");
#endif
}

ErrorOrSuccess M::parseTargetOptions(
    const State &state, const llvm::opt::InputArgList &args,
    CompilationOptions &compilationOptions, llvm::SourceMgr &sourceMgr,
    MLIRContext &ctx, TargetInfoAttr &target, llvm::opt::OptSpecifier tripleId,
    llvm::opt::OptSpecifier cpuId, llvm::opt::OptSpecifier featuresId,
    llvm::opt::OptSpecifier marchId, llvm::opt::OptSpecifier mcpuId,
    llvm::opt::OptSpecifier mtuneId,
    llvm::opt::OptSpecifier targetAcceleratorId,
    llvm::opt::OptSpecifier mcmodelId,
    llvm::opt::OptSpecifier largeDataThresholdId,
    llvm::opt::OptSpecifier relocationModelId, llvm::opt::OptSpecifier abiId) {
  StringRef targetTriple = args.getLastArgValue(tripleId);
  if (args.hasMultipleArgs(tripleId))
    return Error("too many specified target triples, expected exactly one");

  StringRef targetCpu = args.getLastArgValue(cpuId);
  if (args.hasMultipleArgs(cpuId))
    return Error("too many specified target CPUs, expected exactly one");

  StringRef targetFeatures = args.getLastArgValue(featuresId);
  if (args.hasMultipleArgs(featuresId))
    return Error("too many specified target features, expected exactly one");

  StringRef targetAbi = args.getLastArgValue(abiId);
  if (args.hasMultipleArgs(abiId))
    return Error("too many specified target ABIs, expected exactly one");

  StringRef mArch = args.getLastArgValue(marchId);
  if (args.hasMultipleArgs(marchId))
    return Error(
        "too many specified target architectures, expected exactly one");

  StringRef mCpu = args.getLastArgValue(mcpuId);
  if (args.hasMultipleArgs(mcpuId))
    return Error("too many specified target cpus, expected exactly one");

  StringRef mTune = args.getLastArgValue(mtuneId);
  if (args.hasMultipleArgs(mtuneId))
    return Error("too many specified tune cpus, expected exactly one");

  // Validate that LLVM-style options (--target-cpu, --target-features) are not
  // mixed with GCC/Clang-style options (--march, --mcpu, --mtune).
  // These are two separate option families that should not be mixed.
  bool hasGccStyleOptions = !mArch.empty() || !mCpu.empty();
  if (hasGccStyleOptions) {
    if (!targetCpu.empty()) {
      return Error("--target-cpu cannot be used with --march or --mcpu; "
                   "use either --target-cpu/--target-features or "
                   "--march/--mcpu/--mtune");
    }
    if (!targetFeatures.empty()) {
      return Error(
          "--target-features cannot be used with --march or --mcpu; "
          "use --target-cpu with --target-features, or use --march/--mcpu with "
          "extension syntax (e.g., --march=skylake-avx512)");
    }
  }

  StringRef targetAccelerator = args.getLastArgValue(targetAcceleratorId);
  if (args.hasMultipleArgs(targetAcceleratorId))
    return Error(
        "too many specified target accelerators, expected exactly one");

  StringRef mcmodel = args.getLastArgValue(mcmodelId);
  if (args.hasMultipleArgs(mcmodelId))
    return Error("too many specified CodeModel, expected exactly one");

  StringRef largeDataThreshold = args.getLastArgValue(largeDataThresholdId);
  if (args.hasMultipleArgs(largeDataThresholdId))
    return Error(
        "too many specified large data threshold, expected exactly one");

  // If the user specified the triple, the target CPU, or the target feature
  // set, use those to override the defaults.
  if (!targetTriple.empty())
    compilationOptions.targetTriple = targetTriple.str();
  if (!targetCpu.empty())
    compilationOptions.targetCpu = targetCpu.str();
  else if (!mCpu.empty()) {
    // Use -mcpu to set targetCpu when --target-cpu is not specified.
    // Strip any extensions (e.g., "haswell+avx512f" -> "haswell").
    compilationOptions.targetCpu = mCpu.split('+').first.str();
  } else
    compilationOptions.setDefaultCPU();

  if (!targetAbi.empty())
    compilationOptions.targetAbi = targetAbi.str();

  if (!targetFeatures.empty())
    compilationOptions.targetFeatures = targetFeatures.str();
  else if (mArch.empty() && mCpu.empty()) {
    // Only compute features here if not using -march/-mcpu, since those will
    // be handled later by getMArchFeatures() which computes the correct
    // features for the specified architecture/CPU.
    //
    // TODO: Reconsider whether `mojo build` with no target CPU/arch should
    // default to the host CPU at all. Clang defaults to the architecture
    // baseline (e.g. x86-64 with SSE2 only), which is conservative and
    // portable, and relies on -march=native to opt into host-specific
    // features. Mojo currently defaults to the host CPU, which is aggressive
    // and matches the common case of compiling and running on the same machine,
    // but diverges from Clang's convention.
    //
    // Short-term fix: when the default CPU is the host CPU, route through
    // getHostTargetInfo() so the runtime CPUID cross-check is applied. Without
    // this, Clang's CPU model database would enable features (e.g. AVX-512 on
    // znver4) that the OS may not have made available.
    if (compilationOptions.targetCpu == llvm::sys::getHostCPUName()) {
      ErrorOr<TargetInfo> hostInfoOr = M::getHostTargetInfo();
      if (hostInfoOr)
        return hostInfoOr.takeError();
      compilationOptions.targetFeatures = encodeFeatures(*hostInfoOr);
    } else {
      ErrorOr<std::vector<std::string>> featuresOr = M::getFeatures(
          compilationOptions.targetTriple, compilationOptions.targetCpu);
      if (featuresOr)
        return featuresOr.takeError();
      compilationOptions.targetFeatures =
          encodeFeatures(TargetInfo({}, {}, std::move(*featuresOr)));
    }
  }
  if (!targetAccelerator.empty()) {
    compilationOptions.targetAccelerator = targetAccelerator.str();
    compilationOptions.isCrossCompilation = true;
  }
#if MLRT_ACCELERATOR_SUPPORT
  else {
    compilationOptions.targetAccelerator =
        M::Driver::Device::getAcceleratorArchOrEmpty();
  }
#endif

  if (!mcmodel.empty()) {
    if (!llvm::is_contained({"small", "medium", "large"}, mcmodel)) {
      return Error("invalid mcmodel'" + mcmodel +
                   "', expected one of: `small`, `medium` or `large`");
    }
    if (mcmodel == "small")
      compilationOptions.mcmodel = llvm::CodeModel::Small;
    else if (mcmodel == "medium")
      compilationOptions.mcmodel = llvm::CodeModel::Medium;
    else if (mcmodel == "large")
      compilationOptions.mcmodel = llvm::CodeModel::Large;
  }

  if (!largeDataThreshold.empty()) {
    uint64_t value;
    if (!llvm::to_integer(largeDataThreshold, value)) {
      return Error("invalid large-data-threshold'" + largeDataThreshold +
                   "', expected a positive integer number");
    }
    compilationOptions.largeDataThreshold = value;
  }

  StringRef relocationModel = args.getLastArgValue(relocationModelId);
  if (args.hasMultipleArgs(relocationModelId))
    return Error("too many specified relocation models, expected exactly one");

  if (!relocationModel.empty()) {
    ErrorOr<llvm::Reloc::Model> model =
        M::symbolizeRelocationModel(relocationModel);
    if (model.isError())
      return model.takeError();

    compilationOptions.relocModel = *model;
  }

  // Initialize targets first - we rely on this for getTargetInfo as well as for
  // the ExecutionEngine.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  // Construct a target specification using the command line options.
  ErrorOr<TargetInfoAttr> targetOr = nullptr;
  if (!mArch.empty() || !mCpu.empty()) {
    // Use `-march` to determine the feature set.
    targetOr = getMArchFeatures(
        &ctx, compilationOptions.targetTriple, mArch, mCpu, mTune,
        compilationOptions.targetAccelerator, compilationOptions.relocModel,
        compilationOptions.targetAbi);
  } else {
    // Use the full triple, specific CPU, and manually specified features to
    // get the target info.
    targetOr = getTargetInfoFor(
        &ctx, compilationOptions.targetTriple, compilationOptions.targetCpu,
        compilationOptions.targetFeatures, /*tuneCpu=*/"",
        /*acceleratorArch=*/compilationOptions.targetAccelerator,
        compilationOptions.relocModel, compilationOptions.targetAbi);
  }
  if (targetOr.isError())
    return targetOr.takeError();
  target = targetOr.takeValue();

  return success();
}

ErrorOrSuccess
M::MLIRPassTiming::configure(const llvm::opt::InputArgList &args,
                             llvm::opt::OptSpecifier timingId,
                             llvm::opt::OptSpecifier displayModeId) {
  StringLiteral kTree = "tree";
  StringLiteral kList = "list";
  // The code reads the display mode also when the timing is off. An argument
  // that stays unclaimed makes `State::assertNoUnusedArguments` fail. An
  // incorrect mode is also an error when the compiler does not use the mode.
  StringRef displayMode = args.getLastArgValue(displayModeId, kTree);
  if (!llvm::is_contained({kTree, kList}, displayMode)) {
    return Error(llvm::formatv(
        "invalid mlir-timing-display '{0}', expected one of: `{1}` (the "
        "default value), or `{2}`",
        displayMode, kTree, kList));
  }

  if (!args.hasArg(timingId))
    return success();

  manager.setEnabled(true);
  if (displayMode == kList)
    manager.setDisplayMode(mlir::DefaultTimingManager::DisplayMode::List);
  root = manager.getRootScope();
  // The pass manager of an offload target is deep inside the elaboration of
  // the host. The function that makes this pass manager gets no scope. That
  // function reads the root from here. Then that function puts its own scope
  // under the root.
  KGEN::setMLIRTimingRoot(&root);
  return success();
}

KGEN::PassManagerConfigOptions M::MLIRPassTiming::passManagerOptions() {
  KGEN::PassManagerConfigOptions options;
  // The scope stays empty when the timing is off. Then `configurePassManager`
  // uses the time-trace manager, which the trace builds need.
  if (manager.isEnabled())
    options.timingScope = &root;
  return options;
}

/// The reports of MLIR and of LLVM are this many columns wide.
static constexpr unsigned kTimingTitleWidth = 79;

/// Writes a title above a timing report. MLIR and LLVM both write a report
/// that does not name the tool that made it, and a command can ask for both
/// reports. The title tells the two reports apart.
static void printTimingReportTitle(StringRef title) {
  // The title is one line. A report starts with a box of three lines, and a
  // box above a box looks like one damaged box.
  std::string prefix = ("===--- " + title + " ").str();
  unsigned fill = prefix.size() < kTimingTitleWidth - 3
                      ? kTimingTitleWidth - 3 - prefix.size()
                      : 0;
  llvm::errs() << "\n" << prefix << std::string(fill, '-') << "===\n\n";
}

void M::MLIRPassTiming::finish() {
  if (!manager.isEnabled())
    return;
  // Stop the root scope first. Then the total covers the compilation, and the
  // scope is empty, so a later nest operation records nothing.
  root.stop();
  // A scope must not go under `root` after `root` stops.
  KGEN::setMLIRTimingRoot(nullptr);
  printTimingReportTitle("MLIR pass timing (--mlir-timing)");
  manager.print();
  // The destructor of the manager also prints. A manager that is off prints
  // nothing.
  manager.setEnabled(false);
}

void M::LLVMPassTiming::configure(const llvm::opt::InputArgList &args,
                                  llvm::opt::OptSpecifier timingId,
                                  KGEN::CompilationOptions &options) {
  if (!args.hasArg(timingId))
    return;

  enabled = true;
  // The legacy pass manager that does the code generation reads this global
  // variable. `StandardInstrumentations` also reads the variable when it makes
  // the `TimePassesHandler` for the optimization pipeline. Both of them write
  // to the same timer groups.
  llvm::TimePassesIsEnabled = true;

  // The timers of LLVM are global to the process, and a timer of a code
  // generation region has one object for each name. Two threads that run the
  // same region thus start one timer two times, and LLVM stops with an
  // assertion. One thread also keeps one row for each pass, because the report
  // does not merge the rows of the units that a parallel build makes. The
  // split of the report into one part for each pipeline needs one thread as
  // well, because it reads the same global timers.
  options.numThreads = 1;

  KGEN::enableLLVMTimingRegions();
}

void M::LLVMPassTiming::finish() {
  if (!enabled)
    return;
  enabled = false;
  // Stop the collection before the report. Then no more times go into a
  // report that no one prints.
  llvm::TimePassesIsEnabled = false;

  // Each part holds one pipeline: the host, and one for each offload target.
  // A pipeline that the compilation cache answered gives no part, so a run
  // that gets all of its object code from the cache writes nothing at all.
  for (const KGEN::LLVMTimingReportPart &part : KGEN::takeLLVMTimingReport()) {
    printTimingReportTitle("LLVM pass timing (--llvm-timing): " + part.label);
    llvm::errs() << part.report;
  }
}

ErrorOr<OwningOpRef<ModuleOp>> M::invokeMojoParser(
    const State &state, const llvm::opt::InputArgList &args,
    KGEN::CompilationOptions &compilationOptions, MLIRContext *ctx,
    AsyncRT::CPUDevice &cpuDevice, llvm::opt::OptSpecifier docDiagnoseMissingId,
    llvm::opt::OptSpecifier maxNotesId, llvm::opt::OptSpecifier definesId,
    llvm::opt::OptSpecifier stripFilePrefixId,
    llvm::opt::OptSpecifier disableBuiltins, llvm::opt::OptSpecifier stdLibPath,
    llvm::opt::OptSpecifier autoFixIt, llvm::opt::OptSpecifier exportFixit,
    mlir::TimingScope *timingScope,
    function_ref<OwningOpRef<ModuleOp>(ParserConfig &, mlir::TimingScope &)>
        parseFn) {
  // Mutual exclusion check for fixit flags.
  if (args.hasArg(autoFixIt) && args.hasArg(exportFixit)) {
    return Error("cannot use both --experimental-fixit and "
                 "--experimental-export-fixit simultaneously");
  }

  // A command with the `--mlir-timing` option holds the timing manager. Thus
  // the report shows all of the compilation, not only the parse. The other
  // commands use an empty scope, which records no data.
  mlir::TimingScope disabledScope;
  mlir::TimingScope &timing = timingScope ? *timingScope : disabledScope;

  DialectRegistry registry;
  registerAllKGENDialects(registry);
  ctx->appendDialectRegistry(registry);
  if (stdLibPath.isValid()) {
    StringRef value = args.getLastArgValue(stdLibPath);
    compilationOptions.searchPaths = value;
  }
  // Parse the input Mojo file into an MLIR module.
  ParserConfig parseConfig(ctx, compilationOptions);
  parseConfig.diagnoseMissingDocStrings = args.hasArg(docDiagnoseMissingId);
  int maxNotes = 0;
  if (!args.getLastArgValue(maxNotesId).getAsInteger(10, maxNotes))
    parseConfig.maxNotesPerDiagnostic = maxNotes;
  parseConfig.stripFilePrefix = args.getLastArgValue(stripFilePrefixId);
  parseConfig.useBuiltinModule = !args.hasArg(disableBuiltins);

  std::unique_ptr<AutoFixItHandler> fixItHandler;
  if (args.hasArg(autoFixIt)) {
    fixItHandler = std::make_unique<AutoFixItHandler>();
    parseConfig.autoFixItHandler = fixItHandler.get();
  } else if (args.hasArg(exportFixit)) {
    StringRef exportPath = args.getLastArgValue(exportFixit);
    fixItHandler = std::make_unique<AutoFixItHandler>(exportPath);
    parseConfig.autoFixItHandler = fixItHandler.get();
  }

  mlir::TimingScope mojoScope = timing.nest("Import Mojo");
  OwningOpRef<ModuleOp> module = parseFn(parseConfig, mojoScope);

  // Handle fix-it output based on mode.
  if (fixItHandler) {
    if (fixItHandler->isApplyMode()) {
      // Apply mode: apply fix-its and return a null module
      // (re-run is expected after applying fixes).
      if (fixItHandler->hasFixIts()) {
        fixItHandler->applyFixIts();
        llvm::outs() << "Fixits applied.\n";
      } else {
        llvm::outs() << "No fixits to apply.\n";
      }
      return OwningOpRef<ModuleOp>();
    }

    // Export mode: write the YAML file but continue with normal execution.
    // Unliek clang-tidy, we create the file even if there are no fix-its.
    if (auto err = fixItHandler->exportFixIts())
      return err.takeError();
    StringRef yamlPath = args.getLastArgValue(exportFixit);
    llvm::outs() << "Fix-its exported to: " << yamlPath << "\n";
    llvm::outs() << "Apply with: 'clang-apply-replacements "
                 << llvm::sys::path::parent_path(yamlPath) << "'\n";
  }

  if (!module)
    return Error("failed to parse the provided Mojo source module");

  // Create LLVMBitcodeLibAttr instances from command line and package bitcode
  // libraries.
  SmallVector<LLVMBitcodeLibAttr> bitcodeLibAttrs;

  // Add command line bitcode libraries as StringAttr.
  for (const std::string &libPath : compilationOptions.bitcodeLibs) {
    StringAttr pathAttr = StringAttr::get(ctx, libPath);
    LLVMBitcodeLibAttr libAttr = LLVMBitcodeLibAttr::get(false, pathAttr);
    bitcodeLibAttrs.push_back(libAttr);
  }

  // Extract external LLVM bitcode modules from imported packages and add as
  // DenseResourceElementsAttr.
  module->walk([&](LIT::PackageOp packageOp) {
    if (auto bitcodeModules = packageOp.getExternLLVMBitcodeModulesAttr()) {
      for (auto bitcodeAttr : bitcodeModules) {
        LLVMBitcodeLibAttr libAttr =
            LLVMBitcodeLibAttr::get(false, bitcodeAttr);
        bitcodeLibAttrs.push_back(libAttr);
      }
    }
  });

  // Set the LLVMBitcodeLibArrayAttr on the ModuleOp if there are any bitcode
  // libraries.
  if (!bitcodeLibAttrs.empty()) {
    LLVMBitcodeLibArrayAttr arrayAttr =
        LLVMBitcodeLibArrayAttr::get(ctx, bitcodeLibAttrs);
    (*module)->setAttr(LLVMBitcodeLibArrayAttr::getBitcodeLibsAttrName(),
                       arrayAttr);
  }

  // Tag the module with the environment, which includes any definitions the
  // user may have specified on the command line.
  ctx->loadDialect<KGENDialect>();
  if (definesId.isValid()) {
    ErrorOr<EnvAttr> envOrErr = compilationOptions.parseDefinesWithDefaults(
        ctx, args.getAllArgValues(definesId));
    if (failed(envOrErr)) {
      return Error(
          llvm::formatv("an internal error occurred when initializing the Mojo "
                        "MLIR module: {0}",
                        envOrErr.getError()));
    }
    (*module)->setAttr(EnvAttr::getEnvAttrName(), *envOrErr);
  }
  return module;
}

void M::configureCPUDeviceOptions(AsyncRT::CPUDeviceOptions &cpuDeviceOptions,
                                  const KGEN::CompilationOptions &options) {
  if (options.numThreads != 0) {
    // AsyncRT has a sophisticated thread pool configuration system.
    // There are two parameters: numThreads and maxThreads.
    // Setting numThreads to non-zero means "Use exactly this many threads"
    // Setting maxThreads to non-zero means numThreads=min(maxThreads,
    // heuristicBasedOnNumCores)
    //
    // Here we take the stance that mojo build -j 5 means "Use at most 5
    // threads"
    cpuDeviceOptions.numThreads = 0;
    cpuDeviceOptions.maxThreads = options.numThreads;
  }
}
