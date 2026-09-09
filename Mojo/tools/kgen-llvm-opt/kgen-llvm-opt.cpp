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
// The kgen-llvm-opt tool is similar to LLVM's opt tool. It supports two modes:
//
//  1. Full KGEN optimization pipeline: specify -O0/-O1/-O2/-O3 to run the
//     complete Mojo compilation pipeline for that optimization level.
//
//  2. Custom pass pipeline via -passes: specify an explicit pass pipeline using
//     LLVM's pass pipeline syntax (same as `opt -passes=...`). The
//     target-agnostic KGEN pass is:
//         kgen-llvmir-downgrade  - LLVMIRDowngradePass
//     Additional passes are registered by the active target backends.
//
//     Example: kgen-llvm-opt -passes="kgen-llvmir-downgrade" in.bc

#include "Mojo/Compiler/LLVMOptimizationPipeline.h"
#include "Mojo/Compiler/ObjectCompiler.h"
#include "Mojo/ToolCommon/CLOptions.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Support/CommonCLOptions.h"
#include "Target/TargetTraits.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/RuntimeLibcallInfo.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/LibcallLoweringInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IRPrinter/IRPrintingPasses.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/InitializePasses.h"
#include "llvm/LinkAllIR.h"
#include "llvm/LinkAllPasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Remarks/HotnessThresholdParser.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/PluginLoader.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

namespace {
// Debug emission kind
enum class BCVersionNo : uint16_t {
  DEFAULT = 0,
  LLVM17 = 17,
  LLVM19 = 19,
  LLVM21 = 21,
};
struct CLOptions : public M::CLOptionsBase {
  CLOptions(int argc, char **argv, bool skipInitLLVM = true)
      : M::CLOptionsBase(argc, argv, options, skipInitLLVM) {}

  M::OptionsBase options;
  std::string inputFilename{"-"};
  std::string outputFilename{"-"};
  std::string passPipeline;
  bool optLevelO0 = false;
  bool optLevelO1 = false;
  bool optLevelO2 = false;
  bool optLevelO3 = false;
  std::string targetTriple;
  std::string targetAccelerator;
  std::string dataLayout;
  bool noOutput = false;
  bool outputAssembly = false;
  unsigned codeGenOptLevel = 0;
  bool disableOptimizationPasses = false;
  BCVersionNo outputBCVersion =
      BCVersionNo::DEFAULT; // 0 means the same as current upstream main.
  bool downgradeIR = false;

private:
  llvm::cl::OptionCategory cat{"Common command line options"};

  M::cl::MOpt<std::string, true> inputFilenameOp{
      cl::Positional, cl::desc("<input bitcode file>"),
      cl::value_desc("filename"), cl::location(inputFilename), cl::cat(cat)};

  M::cl::MOpt<std::string, true> outputFilenameOpt{
      "o", cl::desc("Override output filename"), cl::value_desc("filename"),
      cl::location(outputFilename), cl::cat(cat)};

  M::cl::MOpt<bool, true> OptLevelO0Opt{
      "O0", cl::desc("Optimization level 0. Similar to mojo -O0. "),
      cl::location(optLevelO0), cl::cat(cat)};

  M::cl::MOpt<bool, true> OptLevelO1Opt{
      "O1", cl::desc("Optimization level 1. Similar to mojo -O1. "),
      cl::location(optLevelO1), cl::cat(cat)};

  M::cl::MOpt<bool, true> OptLevelO2Opt{
      "O2", cl::desc("Optimization level 2. Similar to mojo -O2. "),
      cl::location(optLevelO2), cl::cat(cat)};

  M::cl::MOpt<bool, true> OptLevelO3Opt{
      "O3", cl::desc("Optimization level 3. Similar to mojo -O3. "),
      cl::location(optLevelO3), cl::cat(cat)};

  M::cl::MOpt<std::string, true> targetTripleOpt{
      "mtriple", cl::desc("Override target triple for module"),
      cl::location(targetTriple), cl::cat(cat)};

  M::cl::MOpt<std::string, true> dataLayoutOpt{
      "data-layout", cl::desc("data layout string to use"),
      cl::value_desc("layout-string"), cl::location(dataLayout), cl::cat(cat)};

  M::cl::MOpt<std::string, true> passPipelineOpt{
      "passes",
      cl::desc(
          "A textual description of the pass pipeline (same syntax as "
          "opt -passes=...). Mutually exclusive with -O0/-O1/-O2/-O3.\n"
          "Target-agnostic KGEN pass:\n"
          "  kgen-llvmir-downgrade    Downgrade IR for older LLVM backends\n"
          "Additional passes are registered by the active target backends."),
      cl::value_desc("pipeline"), cl::location(passPipeline), cl::cat(cat)};

  M::cl::MOpt<bool, true> noOutputOpt{
      "disable-output", cl::desc("Do not write result bitcode file"),
      cl::Hidden, cl::location(noOutput), cl::cat(cat)};

  M::cl::MOpt<bool, true> outputAssemblyOpt{
      "S", cl::desc("Write output as LLVM assembly"),
      cl::location(outputAssembly), cl::cat(cat)};

  M::cl::MOpt<unsigned, true> codeGenOptLevelOpt{
      "codegen-opt-level",
      cl::desc("Override optimization level for codegen hooks, legacy PM only"),
      cl::location(codeGenOptLevel), cl::cat(cat)};

  M::cl::MOpt<bool, true> disableOptimizationPassesOpt{
      "disable-optimization-passes",
      cl::desc("Disable optimization passes and print input module. Useful to "
               "test Bitcode Writer"),
      cl::location(disableOptimizationPasses), cl::cat(cat)};

  M::cl::MOpt<BCVersionNo, true> irVersion{
      "output-bc-version",
      cl::desc("output bitcode llvm version"),
      cl::Hidden,
      llvm::cl::values(
          clEnumValN(BCVersionNo::DEFAULT, "default",
                     "Default bitcode version, no downgrading."),
          clEnumValN(BCVersionNo::LLVM17, "llvm17", "Bitcode version 17."),
          clEnumValN(BCVersionNo::LLVM19, "llvm19", "Bitcode version 19."),
          clEnumValN(BCVersionNo::LLVM21, "llvm21", "Bitcode version 21.")),
      cl::location(outputBCVersion),
      cl::init(BCVersionNo::DEFAULT),
      cl::cat(cat)};

  M::cl::MOpt<bool, true> downgradeIROpt{
      "downgrade-llvm-ir",
      cl::desc(
          "Run LLVMIRDowngrade pass for llvm backends with older versions."),
      cl::location(downgradeIR), cl::cat(cat)};
};

enum OutputKind {
  OK_NoOutput,
  OK_OutputAssembly,
  OK_OutputBitcode,
  OK_OutputThinLTOBitcode,
};
} // anonymous namespace

static CodeGenOptLevel getCodeGenOptLevel(const CLOptions &clOptions) {
  return static_cast<CodeGenOptLevel>(unsigned(clOptions.codeGenOptLevel));
}

static ModulePassManager
buildPipeline(PassBuilder &pb, const CLOptions &clOptions, Triple triple) {
  ModulePassManager mpm;

  // If an explicit pass pipeline is specified via -passes, use it directly.
  if (!clOptions.passPipeline.empty()) {
    if (auto err = pb.parsePassPipeline(mpm, clOptions.passPipeline)) {
      errs() << "error: failed to parse pass pipeline '"
             << clOptions.passPipeline << "': " << toString(std::move(err))
             << "\n";
      exit(1);
    }
    return mpm;
  }

  // Otherwise, build the full KGEN optimization pipeline for the given level.
  M::KGEN::CompilationOptions options(/*optimizationLevel=*/-1U);
  options.targetTriple = triple.str();
  if (clOptions.optLevelO0)
    options.optimizationLevel = 0;
  if (clOptions.optLevelO1)
    options.optimizationLevel = 1;
  if (clOptions.optLevelO2)
    options.optimizationLevel = 2;
  if (clOptions.optLevelO3)
    options.optimizationLevel = 3;

  if (options.optimizationLevel == -1U) {
    llvm_unreachable(
        "Specify an optimization level (-O0/-O1/-O2/-O3) or a custom pipeline "
        "(-passes=...).");
  }
  mpm = M::KGEN::buildLLVMOptimizationPipeline(pb, options);

  return mpm;
}

// Normalize a target triple for codegen via its registered TargetTraits (a
// target may compile through a different LLVM triple).
static std::string fixTargetTriple(StringRef triple) {
  if (M::ErrorOr<const M::KGEN::TargetTraits *> traitsOr =
          M::KGEN::TargetTraitsRegistry::get().lookup(Triple(triple));
      !traitsOr.isError())
    return (*traitsOr)->codegenTriple(triple);
  return triple.str();
}

int main(int argc, char **argv) {
  static codegen::RegisterCodeGenFlags cfg;
  CLOptions clOptions(argc, argv);
  InitLLVM llvm(argc, argv);

  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  PassRegistry &registry = *PassRegistry::getPassRegistry();
  initializeCore(registry);
  initializeScalarOpts(registry);
  initializeVectorization(registry);
  initializeIPO(registry);
  initializeAnalysis(registry);
  initializeTransformUtils(registry);
  initializeInstCombine(registry);
  initializeTarget(registry);
  // For codegen passes, only passes that do IR to IR transformation are
  // supported.
  initializeExpandIRInstsLegacyPassPass(registry);
  initializeScalarizeMaskedMemIntrinLegacyPassPass(registry);
  initializeSelectOptimizePass(registry);
  initializeInlineAsmPreparePass(registry);
  initializeCodeGenPrepareLegacyPassPass(registry);
  initializeAtomicExpandLegacyPass(registry);
  initializeWinEHPreparePass(registry);
  initializeDwarfEHPrepareLegacyPassPass(registry);
  initializeSafeStackLegacyPassPass(registry);
  initializeSjLjEHPreparePass(registry);
  initializePreISelIntrinsicLoweringLegacyPassPass(registry);
  initializeGlobalMergePass(registry);
  initializeIndirectBrExpandLegacyPassPass(registry);
  initializeInterleavedLoadCombinePass(registry);
  initializeInterleavedAccessPass(registry);
  initializePostInlineEntryExitInstrumenterPass(registry);
  initializeUnreachableBlockElimLegacyPassPass(registry);
  initializeExpandReductionsPass(registry);
  initializeWasmEHPreparePass(registry);
  initializeWriteBitcodePassPass(registry);
  initializeReplaceWithVeclibLegacyPass(registry);
  initializeJMCInstrumenterPass(registry);
  initializeRuntimeLibraryInfoWrapperPass(registry);
  initializeLibcallLoweringInfoWrapperPass(registry);

  // Register the Target and CPU printer for --version.
  cl::AddExtraVersionPrinter(sys::printDefaultTargetAndDetectedCPU);

  cl::ParseCommandLineOptions(
      argc, argv, "llvm .bc -> .bc modular optimizer and analysis printer\n");

  LLVMContext context;
  SMDiagnostic err;
  std::unique_ptr<Module> module;
  auto setDataLayout = [&](StringRef irTriple,
                           StringRef irLayout) -> std::optional<std::string> {
    if (!clOptions.dataLayout.empty())
      return std::nullopt;

    // If an explicit data layout is already defined in the IR, don't infer.
    if (!irLayout.empty())
      return std::nullopt;

    // If an explicit triple was specified (either in the IR or on the
    // command line), use that to infer the default data layout. However, the
    // command line target triple should override the IR file target triple.
    std::string tripleStr = clOptions.targetTriple.empty()
                                ? irTriple.str()
                                : Triple::normalize(clOptions.targetTriple);

    tripleStr = fixTargetTriple(tripleStr);

    // If the triple string is still empty, we don't fall back to
    // sys::getDefaultTargetTriple() since we do not want to have differing
    // behaviour dependent on the configured default triple. Therefore, if the
    // user did not pass -mtriple or define an explicit triple/datalayout in
    // the IR, we should default to an empty (default) DataLayout.
    if (tripleStr.empty())
      return std::nullopt;

    // Otherwise we infer the DataLayout from the target machine.
    Expected<std::unique_ptr<TargetMachine>> expectedTM =
        codegen::createTargetMachineForTriple(Triple(tripleStr),
                                              getCodeGenOptLevel(clOptions));
    if (!expectedTM) {
      errs() << argv[0] << ": warning: failed to infer data layout: "
             << toString(expectedTM.takeError()) << "\n";
      return std::nullopt;
    }
    return (*expectedTM)->createDataLayout().getStringRepresentation();
  };

  module = parseIRFile(clOptions.inputFilename, err, context,
                       ParserCallbacks(setDataLayout));

  if (!module) {
    err.print(argv[0], errs());
    return 1;
  }

  OutputKind outputKind = OK_NoOutput;
  if (!clOptions.noOutput)
    outputKind =
        clOptions.outputAssembly ? OK_OutputAssembly : OK_OutputBitcode;

  std::unique_ptr<ToolOutputFile> out;
  if (clOptions.noOutput) {
    if (!clOptions.outputFilename.empty())
      errs() << "WARNING: The -o (output filename) option is ignored when\n"
                "the --disable-output option is used.\n";
  } else {
    // Default to standard output.
    if (clOptions.outputFilename.empty())
      clOptions.outputFilename = "-";

    std::error_code errorCode;
    sys::fs::OpenFlags flags =
        clOptions.outputAssembly ? sys::fs::OF_TextWithCRLF : sys::fs::OF_None;
    out.reset(new ToolOutputFile(clOptions.outputFilename, errorCode, flags));
    if (errorCode) {
      errs() << errorCode.message() << '\n';
      return 1;
    }
  }

  if (!clOptions.targetTriple.empty())
    module->setTargetTriple(Triple(Triple::normalize(clOptions.targetTriple)));

  M::ErrorOr<const M::KGEN::TargetTraits *> traitsOr =
      M::KGEN::TargetTraitsRegistry::get().lookup(module->getTargetTriple());
  const M::KGEN::TargetTraits *traits =
      traitsOr.isError() ? nullptr : *traitsOr;

  // A target may force a specific bitcode version, overriding any CLI
  // selection.
  unsigned forcedBCVersion = traits ? traits->forcedBitcodeVersion() : 0;
  BCVersionNo bcVersion = clOptions.outputBCVersion;
  switch (forcedBCVersion) {
  case 0:
    break;
  case 17:
    bcVersion = BCVersionNo::LLVM17;
    break;
  case 19:
    bcVersion = BCVersionNo::LLVM19;
    break;
  case 21:
    bcVersion = BCVersionNo::LLVM21;
    break;
  default:
    errs() << argv[0] << ": unsupported forced bitcode version "
           << forcedBCVersion << " for target '"
           << (traits ? traits->name() : "<unknown>") << "'\n";
    return 1;
  }

  bool useExplicitBitcodeWriter = bcVersion != BCVersionNo::DEFAULT;
  Triple moduleTriple(fixTargetTriple(module->getTargetTriple().str()));
  TargetLibraryInfoImpl tlii(moduleTriple);
  std::string cpuStr, featuresStr;
  std::unique_ptr<TargetMachine> targetMachine;

  if (moduleTriple.getArch()) {
    const TargetOptions options =
        codegen::InitTargetOptionsFromCodeGenFlags(moduleTriple);
    cpuStr = codegen::getCPUStr();
    featuresStr = codegen::getFeaturesStr();
    Expected<std::unique_ptr<TargetMachine>> expectedTM =
        codegen::createTargetMachineForTriple(moduleTriple,
                                              getCodeGenOptLevel(clOptions));
    if (auto e = expectedTM.takeError()) {
      errs() << argv[0] << ": WARNING: failed to create target machine for '"
             << moduleTriple.str() << "': " << toString(std::move(e)) << "\n";
    } else {
      targetMachine = std::move(*expectedTM);
    }
  } else if (moduleTriple.getArchName() != "unknown" &&
             moduleTriple.getArchName() != "") {
    errs() << argv[0] << ": unrecognized architecture '"
           << moduleTriple.getArchName() << "' provided.\n";
    return 1;
  }

  // Override function attributes based on cpuStr, featuresStr, and command line
  // flags.
  codegen::setFunctionAttributes(*module, cpuStr, featuresStr);

  llvm::PassInstrumentationCallbacks pic;
  PassBuilder pb(targetMachine.get(), PipelineTuningOptions(),
                 /*PGOOpt=*/std::nullopt, &pic);
  M::KGEN::registerKGENLLVMPasses(pb);
  ModulePassManager mpm;
  if (!clOptions.disableOptimizationPasses)
    mpm = buildPipeline(pb, clOptions, module->getTargetTriple());
  if (clOptions.downgradeIR)
    M::KGEN::addLLVMIRDowngradePass(mpm);

  switch (outputKind) {
  case OK_NoOutput:
    break; // No output pass needed.
  case OK_OutputAssembly:
    mpm.addPass(PrintModulePass(out->os(), "",
                                /*ShouldPreserveAssemblyUseListOrder=*/false,
                                /*EmitSummaryIndex=*/false));
    break;
  case OK_OutputBitcode:
    // With no explicit bitcode version requested, use the standard writer here;
    // an explicit version is written after the run below.
    if (!useExplicitBitcodeWriter) {
      mpm.addPass(BitcodeWriterPass(out->os(),
                                    /*ShouldPreserveBitcodeUseListOrder=*/false,
                                    /*EmitSummaryIndex=*/false,
                                    /*EmitModuleHash=*/false));
    }
    break;
  case OK_OutputThinLTOBitcode:
    llvm_unreachable("Not implemented.");
  }

  AAManager aa;
  LoopAnalysisManager lam;
  FunctionAnalysisManager fam;
  CGSCCAnalysisManager cgam;
  ModuleAnalysisManager mam;

  StandardInstrumentations standardInstrumentations(module->getContext(),
                                                    /*DebugLogging=*/false);
  standardInstrumentations.registerCallbacks(pic, &mam);

  fam.registerPass([&] { return std::move(aa); });
  fam.registerPass([&] { return TargetLibraryAnalysis(tlii); });
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  mpm.run(*module, mam);

  // Don't verify IR with upstream main if we downgrade it.
  if (!clOptions.downgradeIR && verifyModule(*module, &llvm::errs()))
    return 1;

  if (useExplicitBitcodeWriter && outputKind == OK_OutputBitcode) {
    switch (bcVersion) {
    case BCVersionNo::LLVM17:
      M::KGEN::LLVM::WriteBitcode17ToFile(
          *module, out->os(),
          /*ShouldPreserveUseListOrder = */ false,
          /*ModuleSummaryIndex =*/nullptr,
          /*GenerateHash = */ false,
          /*ModuleHash = */ nullptr);
      break;

    case BCVersionNo::LLVM19:
      M::KGEN::LLVM::WriteBitcode19ToFile(
          *module, out->os(),
          /*ShouldPreserveUseListOrder = */ false,
          /*ModuleSummaryIndex =*/nullptr,
          /*GenerateHash = */ false,
          /*ModuleHash = */ nullptr);
      break;
    case BCVersionNo::LLVM21:
      M::KGEN::LLVM::WriteBitcode21ToFile(
          *module, out->os(),
          /*ShouldPreserveUseListOrder = */ false,
          /*ModuleSummaryIndex =*/nullptr,
          /*GenerateHash = */ false,
          /*ModuleHash = */ nullptr);
      break;
    case BCVersionNo::DEFAULT:
      break;
    }
  }

  // Declare success.
  if (outputKind != OK_NoOutput)
    out->keep();
  return 0;
}
