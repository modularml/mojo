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

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "AsyncRT/Runtime/RuntimeCLOptions.h"
#include "Config/Version.h"
#include "Init/Init.h"
#include "Mojo/Compiler/KGENCompiler.h"
#include "Mojo/Compiler/ObjectCompiler.h"
#include "Mojo/ExecutionEngine/ExecutionEngine.h"
#include "Mojo/ExecutionEngine/JIT/StaticArchiveLayer.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/MojoParser/EntryPoint.h"
#include "Mojo/MojoTooling/ParserDriver.h"
#include "Mojo/MojoTooling/PublicASTDecl.h"
#include "Mojo/Support/CLOptionUtils.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Mojo/Support/Configuration.h"
#include "Mojo/Support/Constants.h"
#include "Mojo/Support/ForceLinkMLIRC.h"
#include "Mojo/ToolCommon/CLOptions.h"
#include "Mojo/ToolCommon/Debug.h"
#include "Mojo/ToolCommon/InitAllDialects.h"
#include "Support/CommonCLOptions.h"
#include "Support/Compiler/TimeProfilerTimingManager.h"
#include "Support/FileSystemExtras.h"
#include "Support/MArchTarget/MArchTarget.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/Process.h"
#include "Target/TargetTraits.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/Timing.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace M;
using namespace KGEN;
using namespace mlir;

namespace {

/// Parser listener used for the `lsp` command: restricts interest to the main
/// file's locations, mirroring how the language server parses an open document
/// (and matching kgen-translate's -lsp listener). Installing a listener also
/// exercises the parser's eager listener-driven resolution paths.
class MainFileParserListener : public LIT::ParserListener {
public:
  explicit MainFileParserListener(const llvm::SourceMgr &sourceMgr)
      : sourceMgr(sourceMgr) {}

  bool isInterestedInLoc(llvm::SMLoc loc) override {
    return sourceMgr.FindBufferContainingLoc(loc) == sourceMgr.getMainFileID();
  }

private:
  const llvm::SourceMgr &sourceMgr;
};

class CLOptions : public KGENOptions {
public:
  KGENCLOptions parser;

  CLOptions(int argc, char **argv, bool skipInitLLVM = false)
      : parser(argc, argv, *this, skipInitLLVM) {}

  M::cl::MListOpt<std::string> inputFiles{llvm::cl::Positional,
                                          cl::desc("<input files>")};

  M::cl::MOpt<bool> emitTextualAsm{
      "S", cl::desc("Print MLIR output files in textual form")};

  M::cl::MOpt<bool> ignoreFailures{
      "ignore-failure",
      cl::desc("Ignore execution failures. Any messages are still printed, but "
               "failures don't mean the tool fails to execute.")};

  M::cl::MOpt<bool> disablePrebuiltPackages{
      "disable-prebuilt-packages",
      cl::desc("Disable prebuilt packages when parsing the input Mojo file."),
      llvm::cl::init(false)};

  M::cl::MOpt<std::string> dependencyFilename{
      "d", llvm::cl::desc("Path of the dependency file to generate"),
      llvm::cl::value_desc("filename"), llvm::cl::init("")};

  /// We default to printing diagnostics through llvm::SourceMgr to enable
  /// source ranges and fixit hints, but allow disabling this for testing.
  M::cl::MOpt<bool> enableMLIRDiagnostics{
      "enable-mlir-diagnostics",
      cl::desc("Print .mojo diagnostics through MLIR."), llvm::cl::init(false)};

  M::cl::MListOpt<std::string> bitcodeLibs{
      "bitcode-libs",
      cl::desc(
          "External bitcode libraries to link into the generated LLVM IR.")};

  /// Add all the input files provided on the command line to the SourceMgr.
  /// This is how MLIR parses multiple files.
  ErrorOrSuccess addInputFilesToSourceMgr(llvm::SourceMgr &mgr);
  void addInputFilesToSourceMgrOrExit(llvm::SourceMgr &mgr);

  /// Set default CPU value.
  ErrorOrSuccess setDefaultCPU();
};
} // namespace

ErrorOrSuccess CLOptions::addInputFilesToSourceMgr(llvm::SourceMgr &mgr) {
  if (inputFiles.empty())
    mgr.AddNewSourceBuffer(openInputFileOrExit(), llvm::SMLoc());

  for (StringRef in : inputFiles) {
    std::error_code ec;
    std::filesystem::path fullPath = std::filesystem::absolute(in.str(), ec);
    if (ec) {
      return Error(
          llvm::formatv("failed to resolve the absolute path for '{0}': {1}",
                        in.str(), ec.message()));
    }
    std::string errorMsg;
    auto result = mlir::openInputFile(fullPath.string(), &errorMsg);
    if (!result)
      return Error(errorMsg);

    mgr.AddNewSourceBuffer(std::move(result), llvm::SMLoc());
  }

  return M::success();
}

void CLOptions::addInputFilesToSourceMgrOrExit(llvm::SourceMgr &mgr) {
  if (auto err = addInputFilesToSourceMgr(mgr))
    exit(reportError(err.getError()));
}

ErrorOrSuccess CLOptions::setDefaultCPU() {
  llvm::Triple triple(targetTriple);
  // A registered target supplies its own default CPU (e.g. the host target's
  // ARM baseline, or a plugin target); otherwise default to `generic`.
  ErrorOr<const TargetTraits *> traits =
      TargetTraitsRegistry::get().lookup(triple);
  if (traits.isError())
    return traits.takeError();

  llvm::StringRef targetDefault = (*traits)->defaultCPU(triple);
  targetCpu = targetDefault.empty() ? "generic" : targetDefault.str();
  return M::success();
}

/// Emit the IR for `theModule` to a file.
static LogicalResult emitModuleIR(ModuleOp theModule, const CLOptions &opts) {
  CompilerTimeTraceScope traceScope("emit-module",
                                    theModule.getSymName().value_or(""));
  if (opts.emitTextualAsm.getValue()) {
    auto outFile = opts.getOutputFile(/*hasBinaryOutput=*/false, ".mlir");
    if (!outFile)
      return failure();

    theModule.print(outFile->os());
    // `print` does not insert a newline, so add one here.
    outFile->os() << "\n";
    outFile->keep();
  } else {
    auto outFile = opts.getOutputFile(/*hasBinaryOutput=*/true, ".mlirbc");
    if (!outFile)
      return failure();

    if (failed(mlir::writeBytecodeToFile(theModule, outFile->os())))
      return failure();
    outFile->keep();
  }

  // Try to save the textual IR as an intermediate file.
  if (auto irFile = opts.getIntermediateFile(opts.outputFilename, ".mlir")) {
    theModule.print(irFile->os());
    irFile->keep();
  }

  return mlir::success();
}

/// Create a dependency file for the `-d` option.
///
/// This functionality is generally only for the benefit of the build system,
/// and informs it of the dependencies of the input files.
static LogicalResult createDependencyFile(const CLOptions &clOptions,
                                          ArrayRef<std::string> includedFiles) {
  // It only makes sense to output a dependency file that can map inputs to
  // outputs. If the output file already exists and is not a regular file --
  // like `"-"` for stdout, or a character file like `/dev/null` -- then fail.
  if (clOptions.outputFilename == "-" ||
      (llvm::sys::fs::exists(clOptions.outputFilename) &&
       !llvm::sys::fs::is_regular_file(clOptions.outputFilename))) {
    return failure(clOptions.reportError(
        "can only create dependency file for outputs written to files"));
  }

  std::string errorMessage;
  std::unique_ptr<llvm::ToolOutputFile> outputFile =
      openOutputFile(clOptions.dependencyFilename, &errorMessage);
  if (!outputFile)
    return failure(clOptions.reportError(errorMessage));

  // Resolve each of the dependencies and add them to the file.
  outputFile->os() << clOptions.outputFilename << ":";
  for (StringRef includeFile : includedFiles)
    outputFile->os() << ' ' << includeFile;

  outputFile->os() << "\n";
  outputFile->keep();
  return mlir::success();
}

/// Runs the tool pipeline on the file fragment passed in. The pipeline does not
/// output to the specific ostream provided to it, rather it opens and writes to
/// files that are designated by the funcs it operates on.
static LogicalResult runToolPipeline(MLIRContext *ctx, llvm::SourceMgr &mgr,
                                     CLOptions &clOptions) {
  DialectRegistry registry;
  TraceProfiler tracer(clOptions.timeTrace, clOptions.timeTraceGranularity);

  if (clOptions.enableMLIRCrashReproducer || KGEN::debugFlag) {
    // If the reproducer is enable, turn off all threading.
    ctx->disableMultithreading();
    clOptions.withSingleThreaded();
  }

  // `-lsp` already forces MLIR pass-manager multithreading off below (there is
  // exactly one file, one clone, one check pipeline run: no parallel work to
  // exploit). Without this, the AsyncRT work queue still defaults to a full
  // thread pool, so every invocation pays thread-pool startup/teardown cost
  // for threads that never do anything -- a measured ~15-20% of wall time on
  // real kernel files, matching the real LSP server's `-mojo-test` mode, which
  // does the same.
  if (clOptions.cmd == Command::kLSP || clOptions.cmd == Command::kLSPNoDump)
    clOptions.withSingleThreaded();

  // Register MLIR stuff
  registerAllKGENDialects(registry);
  registerKGENToLLVMTranslation(registry);

  // Create our context, with a cpuDevice; this should not fail.
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "kgen", Init::Options().withCPUDeviceOptions(
                  clOptions.parser.options.withCPUAffinity(false)));
  if (ctxOr.isError())
    return failure();
  registerContext(registry, *ctxOr,
                  /*enableThreadPool=*/!clOptions.enableMLIRCrashReproducer);

  // Set up the dialects in the context.
  ctx->appendDialectRegistry(registry);

  CompilationOptions options = clOptions.getCompilationOptions();
  if (clOptions.saveTemps)
    options.saveTempsPrefix = clOptions.tempsDir;
  options.verboseOutput = (clOptions.cmd == Command::kEmitAssemblyVerbose);
  options.bitcodeLibs = llvm::to_vector_of<std::string>(clOptions.bitcodeLibs);
  options.cacheBaseExtra = "kgen";
  options.warnOnUnstableAPIs = clOptions.warnOnUnstableAPIs;
  options.ignoredDeprecations = clOptions.ignoredDeprecations;

  OwningOpRef<ModuleOp> theModule;
  auto inputFileName = llvm::StringRef(clOptions.inputFilename);

  // Initialize the timing manager.
  std::unique_ptr<mlir::TimingManager> timingManager;
  if (clOptions.timeTrace) {
    timingManager = std::make_unique<TimeProfilerTimingManager>();
  } else {
    auto defaultManager = std::make_unique<mlir::DefaultTimingManager>();
    applyDefaultTimingManagerCLOptions(*defaultManager);
    timingManager = std::move(defaultManager);
  }

  if (!clOptions.mcmodel.empty()) {
    if (!llvm::is_contained({"small", "medium", "large"}, clOptions.mcmodel)) {
      return Error("invalid mcmodel'" + clOptions.mcmodel +
                   "', expected one of: `small`, `medium` or `large`");
    }
    if (clOptions.mcmodel == "small")
      options.mcmodel = llvm::CodeModel::Small;
    else if (clOptions.mcmodel == "medium")
      options.mcmodel = llvm::CodeModel::Medium;
    else if (clOptions.mcmodel == "large")
      options.mcmodel = llvm::CodeModel::Large;
  }

  if (!clOptions.largeDataThreshold.empty()) {
    uint64_t value;
    if (!llvm::to_integer(clOptions.largeDataThreshold, value)) {
      return Error("invalid large-data-threshold'" +
                   clOptions.largeDataThreshold +
                   "', expected a positive integer number");
    }
    options.largeDataThreshold = value;
  }

  if (!clOptions.relocationModel.empty()) {
    ErrorOr<llvm::Reloc::Model> model =
        M::symbolizeRelocationModel(clOptions.relocationModel);
    if (model.isError())
      return model.takeError();

    options.relocModel = *model;
  }

  if (!clOptions.loopUnrollingWarnThreshold.empty()) {
    uint64_t value;
    if (!llvm::to_integer(clOptions.loopUnrollingWarnThreshold, value)) {
      return Error("invalid loop-unrolling-warn-threshold'" +
                   clOptions.loopUnrollingWarnThreshold +
                   "', expected an integer number");
    }
    options.loopUnrollingWarnThreshold = value;
  }

  TimingScope timing = timingManager->getRootScope();

  PassManagerConfigOptions pmOptions;
  pmOptions.applyPassManagerCLOptions = true;
  pmOptions.enableTiming = false;
  pmOptions.timingScope = &timing;
  pmOptions.crashReproducerOptions.enable = clOptions.enableMLIRCrashReproducer;
  pmOptions.crashReproducerOptions.inputFileName = clOptions.inputFilename;
  pmOptions.crashReproducerOptions.enableLocalMLIRReproducer =
      clOptions.enableLocalMLIRReproducer;

  options.isCrossCompilation = !clOptions.targetAccelerator.empty();
  if (clOptions.cmd == Command::kElaborateUseParametricInterpreter)
    options.useParametricInterpreter = true;
  else if (clOptions.cmd == Command::kElaborateNoUseParametricInterpreter)
    options.useParametricInterpreter = false;

  KGENCompiler compiler(*ctx, options, std::move(pmOptions));

  // The `lsp`/`lsp=no-dump` commands reproduce how the language server
  // processes an open document (MojoDocument::checkModuleSemantics), all
  // in-process: the lazy, error-tolerant parse (parseFileForLSP) followed by
  // the check pipeline (runCheckLITPipeline) on the cloned module.
  // Diagnostics are reported on stderr and (for plain `-lsp`) the resulting
  // checked module IR is printed to stdout; it then returns before the
  // normal import/elaborate/emit path (no elaboration or lowering happens,
  // matching the server, which stops at semantic checking). Note: docstring
  // code blocks are not checked here (that is the server's processDocStrings
  // step).
  //
  // Unlike the server (a long-running process with no exit code, for which
  // diagnostics are the only signal), this CLI command reports any
  // error-severity diagnostic as a non-zero exit -- useful for scripts/tests
  // that want a pass/fail signal without scraping stderr.
  if (clOptions.cmd == Command::kLSP || clOptions.cmd == Command::kLSPNoDump) {
    if (!inputFileName.ends_with(".mojo"))
      return failure(
          clOptions.reportError("lsp command requires a .mojo file"));

    ctx->disableMultithreading();
    // Route both parse and check diagnostics through one handler so they print
    // to stderr with source locations, mirroring checkModuleSemantics.
    mlir::SourceMgrDiagnosticHandler diagHandler(mgr, ctx);

    // Track whether any error-severity diagnostic was emitted, without
    // changing how diagHandler prints it. Handlers run most-recently-
    // registered-first and stop at the first success(); returning failure()
    // here always defers to diagHandler, which is registered first.
    bool sawError = false;
    mlir::ScopedDiagnosticHandler errorTracker(
        ctx, [&](mlir::Diagnostic &diag) {
          if (diag.getSeverity() == mlir::DiagnosticSeverity::Error)
            sawError = true;
          return failure();
        });

    LIT::ParserConfig config(ctx, options);
    config.stripFilePrefix = clOptions.stripFilePrefix;
    config.useMLIRDiagnostics = true;
    config.disablePrebuiltPackages = clOptions.disablePrebuiltPackages;
    MainFileParserListener lspListener(mgr);
    config.parserListener = &lspListener;

    MojoParserContext parserContext(mgr, config);
    MojoASTDeclRef decl = parserContext.parseFileForLSP(mgr.getMainFileID());
    if (!decl)
      return failure(clOptions.reportError("could not parse the module"));

    // Skip checking on a failed parse, like the actual server would do.
    if (!sawError) {
      // Run the real check pipeline on the same DCE'd per-decl clone the server
      // checks. Pipeline failure is non-fatal (the server only debug-logs it).
      OwningOpRef<ModuleOp> clone = LIT::cloneDeclModuleForCompilation(*decl);
      (void)compiler.runCheckLITPipeline(*clone);

      if (clOptions.cmd != Command::kLSPNoDump) {
        clone->print(llvm::outs());
        llvm::outs() << "\n";
      }
    }

    parserContext.ensureSignaturesResolved();

    return failure(sawError);
  }

  // The set of files included during processing, used to generate the
  // dependency file.
  SmallVector<std::string> includedFiles;

  if (inputFileName.ends_with(".mojo")) {
    TimingScope litScope = timing.nest("Import Mojo source");
    LIT::ParserConfig config(ctx, options);
    config.stripFilePrefix = clOptions.stripFilePrefix;
    config.useMLIRDiagnostics = clOptions.enableMLIRDiagnostics;
    config.disablePrebuiltPackages = clOptions.disablePrebuiltPackages;
    theModule = importMojoFile(*ctxOr, mgr, config, litScope, &includedFiles);
  } else {
    theModule = parseSourceFile<ModuleOp>(mgr, ctx);
  }
  if (!theModule)
    return failure(clOptions.reportError("could not parse the module"));

  // Tag the module with the environment parsed from the defines.
  ctx->loadDialect<KGENDialect>();

  // Populate the module with the user-provided -D options.
  ErrorOr<EnvAttr> env =
      options.parseDefinesWithDefaults(ctx, clOptions.defines);
  if (env.isError())
    return failure(clOptions.reportError(env.takeError().get()));
  theModule.get()->setAttr(EnvAttr::getEnvAttrName(), env.takeValue());

  // Extend the module with the Module env-attrs.
  extendWithModularEnvAttr(theModule.get(),
                           (*ctxOr)->get<CompilationContext>());

  // If we are generating a dependency file, do so now.
  if (!clOptions.dependencyFilename.empty()) {
    if (failed(createDependencyFile(clOptions, includedFiles)))
      return failure(
          clOptions.reportError("failed to create a dependency file"));
  }

  // Find a target specification or construct one using the commandline options.
  TargetInfoAttr target = getTargetInfo(*theModule);
  if (target) {
    if (!clOptions.march.empty()) {
      mlir::emitWarning(theModule->getLoc(),
                        "overriding module target specification with -march");
    } else if (target.getTripleStr() != clOptions.targetTriple ||
               target.getArch() != clOptions.targetCpu ||
               target.getFeatures() != clOptions.targetFeatures) {
      mlir::emitWarning(theModule->getLoc(),
                        "module target does not match command line "
                        "specification and will be overwritten");
    }
    target = nullptr;
  }

  if (!target) {
    ErrorOr<TargetInfoAttr> targetOr = nullptr;
    if (!clOptions.march.empty() || !clOptions.mcpu.empty()) {
      // Detect if the user accidentally specified any of the `--target-*`.
      if (options.targetCpu != llvm::sys::getHostCPUName() ||
          options.targetFeatures != getHostCPUFeatures())
        return failure(clOptions.reportError(
            "--target-cpu or --target-features specified at "
            "the same time as -march or -mcpu"));

      // Use `-march` to determine the feature set.
      targetOr = getMArchFeatures(ctx, clOptions.targetTriple, clOptions.march,
                                  clOptions.mcpu, clOptions.mtune,
                                  clOptions.targetAccelerator,
                                  options.relocModel, options.targetAbi);
    } else {
      if (clOptions.targetTriple != llvm::sys::getDefaultTargetTriple()) {
        if (clOptions.targetCpu == llvm::sys::getHostCPUName()) {
          ErrorOrSuccess result = clOptions.setDefaultCPU();
          if (result.isError())
            return failure(clOptions.reportError(result.getError()));
        }

        if (clOptions.targetFeatures == getHostCPUFeatures())
          clOptions.targetFeatures = "";
      }

      // When --target-cpu differs from the host but --target-features was not
      // explicitly provided, clear the default host features so they are
      // re-derived from the specified CPU. Without this, the host's feature
      // set (e.g. avx512) leaks into cross-CPU compilations (e.g. x86-64-v3).
      if (clOptions.targetCpu != llvm::sys::getHostCPUName() &&
          !clOptions.parser.targetFeaturesWasSet())
        clOptions.targetFeatures = "";

      // `targetCpu` defaults to the host CPU name, which LLVM reports as
      // "generic" for a part it cannot identify -- a name that fails
      // validation. Fall back to the triple's baseline instead of refusing to
      // compile on such a host, and take the resolved name with it so
      // getTargetInfoFor below is not handed one the target already rejected.
      ErrorOr<M::ResolvedCpu> cpuOr =
          M::resolveCpu(clOptions.targetTriple, clOptions.targetCpu);
      if (cpuOr)
        return failure(clOptions.reportError(cpuOr.takeError().get()));
      M::ResolvedCpu resolvedCpu = cpuOr.takeValue();
      clOptions.targetCpu = resolvedCpu.name;

      // TODO: This always overwrites any user-specified features
      if (clOptions.targetFeatures.empty())
        clOptions.targetFeatures =
            encodeFeatures(TargetInfo({}, {}, std::move(resolvedCpu.features)));

      // Use the full triple, specific CPU, and manually specified features to
      // get the target info.
      targetOr = getTargetInfoFor(ctx, clOptions.targetTriple,
                                  clOptions.targetCpu, clOptions.targetFeatures,
                                  clOptions.mtune, options.targetAccelerator,
                                  options.relocModel, options.targetAbi);
    }

    if (targetOr.isError())
      return failure(clOptions.reportError(targetOr.getError()));
    target = targetOr.takeValue();
    options.targetTriple = target.getTripleStr();
    options.targetCpu = target.getArch();
    options.targetFeatures = target.getFeatures();
    options.targetAbi = target.getAbi();
    options.targetAccelerator = clOptions.targetAccelerator;
  }

  auto compilerOr = ObjectCompiler::create(kMojoCacheBaseDirName, options,
                                           clOptions.cmd == Command::kExecute,
                                           *ctx, pmOptions);
  if (failed(compilerOr))
    return failure(clOptions.reportError(compilerOr.getError()));
  ObjectCompiler &objCompiler = **compilerOr;

  // Compiles the module through KGEN compiler pipeline.
  // We don't need to try to look anything up.
  if (ErrorOrSuccess err = compiler.runKGENPipeline(*theModule, target))
    return failure(clOptions.reportError(err.getError()));

  // If all we're doing is generating a library file or elaborating, we're done
  // now.
  if (clOptions.cmd == Command::kElaborate ||
      clOptions.cmd == Command::kElaborateUseParametricInterpreter ||
      clOptions.cmd == Command::kElaborateNoUseParametricInterpreter)
    return emitModuleIR(*theModule, clOptions);

  // Construct the symbol table and the export map.
  SymbolTable symtab(*theModule);
  ExportMap exportedSymbols = getExportedSymbols(*theModule);

  // Handle LLVM output.
  if (clOptions.cmd == Command::kEmitLLVM ||
      clOptions.cmd == Command::kEmitLLVMBitcode) {
    llvm::LLVMContext llvmCtx;
    ErrorOr<std::unique_ptr<llvm::Module>> llvmModuleOr =
        objCompiler.lowerAllFuncsToLLVM(llvmCtx, *theModule);

    if (llvmModuleOr)
      return failure(clOptions.reportError(
          Twine("could not lower funcs to LLVM, ") + llvmModuleOr.getError()));

    auto outFile = clOptions.getOutputFile(/*hasBinaryOutput=*/false, ".ll");
    if (!outFile)
      return failure(clOptions.reportError("could not open .ll output file"));

    std::unique_ptr<llvm::Module> llvmModule = llvmModuleOr.takeValue();
    if (clOptions.cmd == Command::kEmitLLVMBitcode) {
      if (ErrorOrSuccess err =
              objCompiler.emitBitcode(*llvmModule, outFile->os()))
        return failure(clOptions.reportError(err.takeError().get()));
    } else {
      llvmModule->print(outFile->os(), nullptr);
    }
    outFile->keep();
    return mlir::success();
  }

  if (clOptions.cmd == Command::kEmitLLVMOpt ||
      clOptions.cmd == Command::kEmitLLVMOptBitcode) {
    auto outFile = clOptions.getOutputFile(/*hasBinaryOutput=*/false, ".ll");
    if (!outFile)
      return failure(clOptions.reportError("could not open .ll output file"));

    LLVMModuleAndContext llvmModule;
    if (ErrorOrSuccess err = objCompiler.lowerAllFuncsToLLVMAndOptimize(
            *theModule, llvmModule)) {
      return failure(clOptions.reportError("failed to generate LLVMIR: " +
                                           Twine(err.getError())));
    }
    if (clOptions.cmd == Command::kEmitLLVMOptBitcode) {
      if (ErrorOrSuccess err =
              objCompiler.emitBitcode(*llvmModule, outFile->os()))
        return failure(clOptions.reportError(err.takeError().get()));
    } else {
      llvmModule->print(outFile->os(), nullptr);
    }
    outFile->keep();
    return mlir::success();
  }

  // Handle assembly output.
  if (clOptions.cmd == Command::kEmitAssembly ||
      clOptions.cmd == Command::kEmitAssemblyVerbose) {
    auto outFile = clOptions.getOutputFile(/*hasBinaryOutput=*/false, ".s");
    if (!outFile)
      return failure(clOptions.reportError("could not open .s output file"));

    ErrorOrSuccess standaloneOr =
        objCompiler.emitAssembly(std::move(theModule), outFile->os());
    if (failed(standaloneOr))
      return failure(
          clOptions.reportError("could not produce standalone asm: " +
                                Twine(standaloneOr.getError())));
    outFile->keep();
    return mlir::success();
  }

  // Handle header emission, we don't need to generate an archive for this.
  if (clOptions.cmd == Command::kEmitHeader) {
    LogicalResult result = failure();
    auto writeFn = [&](raw_ostream &os) {
      result =
          objCompiler.emitCXXHeader(*theModule, clOptions.outputFilename, os);
    };
    if (clOptions.outputFilename == "-") {
      auto writeContents = [&](raw_ostream &os) {
        writeFn(os);
        os.flush();
        return llvm::Error::success();
      };
      if (llvm::Error err =
              llvm::writeToOutput(clOptions.outputFilename, writeContents)) {
        return failure(
            clOptions.reportError(toModularError(std::move(err)).get()));
      }

      // Safely process creating the header, taking into account that we may
      // have different processes trying to produce this header in parallel.
    } else if (ErrorOr<std::filesystem::path> err =
                   writeFileUnderLock(clOptions.outputFilename, writeFn);
               err.isError()) {
      return failure(clOptions.reportError(err.getError()));
    }
    return mlir::success();
  }

  // If there are no exported symbols, there's nothing to codegen. Report this
  // as an error.
  if (exportedSymbols.empty()) {
    return failure(
        clOptions.reportError("module does not `@export` any symbols or define "
                              "a `main` function; nothing to codegen"));
  }

  // If we need to execute, grab the function metadata before the module is
  // consumed.
  struct FunctionExecution {
    StringAttr name;
    Location loc;
    FunctionType type;
    CommandLineFunc clFunc;
  };
  SmallVector<FunctionExecution> funcExecs;
  StringSet<> foundFuncs;
  if (clOptions.cmd == Command::kExecute) {
    for (auto fn : theModule->getOps<FuncOp>()) {
      StringAttr name = fn.getSymNameAttr();
      // See if we were asked to execute this function.
      if (std::optional<CommandLineFunc> clFunc =
              clOptions.shouldExecuteFunc(name)) {
        funcExecs.push_back(FunctionExecution{name, fn.getLoc(),
                                              fn.getFunctionType(), *clFunc});
        foundFuncs.insert(name);
        if (auto err = clFunc->verifyFuncSignature(funcExecs.back().type)) {
          mlir::emitError(fn.getLoc(), err.getError());
          if (!clOptions.ignoreFailures)
            return failure();
        }
      }
    }
    // If we didn't find a function the user asked to execute, emit an error.
    for (const auto &fn : clOptions.funcs) {
      if (!foundFuncs.count(fn.name)) {
        return mlir::emitError(theModule->getLoc(),
                               "could not find func '@" + fn.name + "'");
      }
    }
  }

  if (clOptions.cmd == Command::kEmitSharedObject) {
    auto outFile = clOptions.getOutputFile(/*hasBinaryOutput=*/true, ".so");
    if (!outFile)
      return failure(clOptions.reportError("could not open .so output file"));

    ErrorOrSuccess sharedObjOr =
        objCompiler.emitSharedObject(std::move(theModule), outFile->os());
    if (failed(sharedObjOr))
      return failure(clOptions.reportError(
          "could not produce standalone shared object binary: " +
          Twine(sharedObjOr.getError())));
    outFile->keep();
    return mlir::success();
  }

  // -emit and -execute both require compiled objects.
  ErrorOr<BufferRef> archiveOr = objCompiler.emitArchive(std::move(theModule));
  if (failed(archiveOr)) {
    return failure(clOptions.reportError("failed to emit archive: " +
                                         Twine(archiveOr.getError())));
  }
  BufferRef archive = archiveOr.takeValue();

  // If we're emitting the archive, do it.
  if (clOptions.cmd == Command::kEmit) {
    // Look up the first item in the exported symbols to trigger archive
    // generation.
    auto outFile = clOptions.getOutputFile(/*hasBinaryOutput=*/false, ".o");
    if (!outFile)
      return failure(clOptions.reportError("could not open .o output file"));

    outFile->os() << archive->getBuffer();
    outFile->keep();
    return mlir::success();
  }

  ExecutionEngineOptions eeOptions;
  if (options.debugLevel != CompilationOptions::kNoDebug)
    eeOptions.registerDebugPlugins = true;
  // Detect cross-compilation by checking whether the target CPU is the same as
  // the host CPU.
  eeOptions.crossCompiling = options.targetCpu != llvm::sys::getHostCPUName();

  auto engineOr = initializeExecutionEngine(*ctx, options, std::move(eeOptions),
                                            /*isJIT=*/true, pmOptions);
  if (failed(engineOr))
    return failure(clOptions.reportError(engineOr.getError()));
  ExecutionEngine &engine = **engineOr;

  // Helper to execute a func.
  auto execFunc = [&](const FunctionExecution &func, StringAttr name,
                      const CommandLineFunc &clFunc) -> LogicalResult {
    CompilerTimeTraceScope traceScope("execute-function", name);
    // Trigger compilation so we can pull out the archive.
    ErrorOr<CompiledFunc> funcOr = engine.lookup(name);
    if (failed(funcOr))
      return failure(clOptions.reportError(funcOr.getError()));

    if (auto err = clFunc.executeAndPrint(*funcOr)) {
      mlir::emitError(func.loc, err.getError());
      return failure(!clOptions.ignoreFailures);
    }
    return mlir::success();
  };

  // Pass the compiled archive to the execution engine.
  if (ErrorOrSuccess err =
          engine.addIfAbsent<StaticArchiveLayer>("exec", std::move(archive)))
    return failure(clOptions.reportError(err.getError()));

  // Loop over the functions, executing as necessary.
  for (const FunctionExecution &func : funcExecs) {
    if (failed(execFunc(func, func.name, func.clFunc))) {
      return failure(
          clOptions.reportError("failed to execute " + func.name.strref()));
    }
  }

  return mlir::success();
}

int main(int argc, char **argv) {
  KGEN::forceLinkMLIRC();
  M::registerCommandFlags();

  CLOptions clOptions(argc, argv);

  // Initialize targets first, so that --version shows registered targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  // Override the default version printer.
  llvm::cl::SetVersionPrinter([](raw_ostream &os) {
    ProjectVersion version = getMojoVersion();
    os << "KGEN compiler:\n  ";
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
  registerMLIRContextCLOptions();
  registerAsmPrinterCLOptions();
  registerDefaultTimingManagerCLOptions();
  KGEN::registerDefaultKGENPasses("kgen");
  registerPassManagerCLOptions();
  KGEN::initializeDebugOptions();
  KGEN::KGENPassCLOptions::registerOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Set up the input file(s).
  llvm::SourceMgr sourceManager;
  sourceManager.setIncludeDirs(clOptions.getIncludePaths());
  clOptions.addInputFilesToSourceMgrOrExit(sourceManager);
  if (clOptions.targetAccelerator.empty()) {
#if MLRT_ACCELERATOR_SUPPORT
    clOptions.targetAccelerator = Driver::Device::getAcceleratorArchOrEmpty();
#endif
  }

  return failed(clOptions.configureMLIRContextAndExecute(
      sourceManager, [&](MLIRContext *ctx) -> LogicalResult {
        ctx->printOpOnDiagnostic(true);
        return runToolPipeline(ctx, sourceManager, clOptions);
      }));
}
