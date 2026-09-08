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

#include "Mojo/Compiler/ObjectCompiler.h"
#include "Mojo/Compiler/SaveAsmOutput.h"
#include "Support/CacheLog.h"

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/Algorithms.h"
#include "KGENToLLVMPipeline.h"
#include "LLVMAccessorHelper.h"
#include "LLVMPassesPipeline.h"
#include "MCLinker.h"
#include "Mojo/Compiler/LLVMIRUtils.h"
#include "Mojo/Compiler/LLVMOptimizationPipeline.h"
#include "Mojo/Compiler/Target/TargetBackend.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/Support/BuildInfo.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Mojo/Support/Configuration.h"
#include "Mojo/Support/FileUtils.h"
#include "Mojo/ToolCommon/CLOptions.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Mojo/ToolCommon/Debug.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Mojo/ToolCommon/PipelineTiming.h"
#include "Support/Context.h"
#include "Support/FileSystemExtras.h"
#include "Target/TargetTraits.h"

#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/RuntimeLibcallInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <dlfcn.h>
#include <fstream>
#include <string>

using namespace M;
using namespace KGEN;
using namespace Cache;

#define DEBUG_TYPE "object-compiler"
#define KGEN_DEBUG_TYPE "object-compiler"

//===----------------------------------------------------------------------===//
// ObjectCompiler
//===----------------------------------------------------------------------===//

ErrorOr<std::unique_ptr<ObjectCompiler>>
ObjectCompiler::create(StringRef basePath, CompilationOptions options,
                       bool isJIT, MLIRContext &context,
                       PassManagerConfigOptions pmOptions) {
  MODULAR_CACHE_LOG("mojo")
      << "ObjectCompiler::create: basePath=" << basePath << "\n";
  std::filesystem::path cachePath(basePath.str());
  if (!options.cacheBaseExtra.empty())
    cachePath = cachePath / options.cacheBaseExtra;
  cachePath = cachePath / "transform";
  MODULAR_CACHE_LOG("mojo")
      << "ObjectCompiler::create: cachePath=" << cachePath.string()
      << " cacheBaseExtra=" << options.cacheBaseExtra << "\n";

  auto transformCache =
      Cache::getLocalDefaultBackendChain(cachePath, getVersionString());
  if (failed(transformCache)) {
    MODULAR_CACHE_LOG("mojo")
        << "ObjectCompiler::create: backend creation failed\n";
    return transformCache.takeError();
  }
  MODULAR_CACHE_LOG("mojo")
      << "ObjectCompiler::create: transform cache created\n";

  ErrorOr<MojoConfig> configOr = MojoConfig::open();
  if (failed(configOr)) {
    return Error(Twine("failed to parse 'modular.cfg': ") +
                 configOr.getError());
  }

  MojoConfig config = std::move(*configOr);

  StringRef linkerFileName = "ld.lld";
  if (llvm::Triple(options.targetTriple).getObjectFormat() ==
      llvm::Triple::MachO) {
    linkerFileName = "ld64.lld";
  }

  // Find the linker.
  llvm::ErrorOr<std::string> lldPath = config.getLLDPath().str();
  if (lldPath->empty()) {
    lldPath = llvm::sys::findProgramByName(linkerFileName);
    if (!lldPath) {
      return Error("unable to find linker for linking");
    }
  }

  std::string linker = *lldPath;

  // Resolve the target backend once, up front. A missing backend means the
  // target is not supported by this build; fail here rather than silently
  // falling back deeper in the pipeline.
  ErrorOr<const TargetBackend *> backendOr =
      TargetBackendRegistry::get().lookup(llvm::Triple(options.targetTriple));
  if (backendOr.isError())
    return Error(backendOr.getError());
  const TargetBackend *backend = *backendOr;

  return std::unique_ptr<ObjectCompiler>(new ObjectCompiler(
      std::move(*transformCache), std::move(options), *backend, isJIT, context,
      linker, std::move(pmOptions)));
}

ObjectCompiler::ObjectCompiler(RCRef<Cache::BlobCacheBackend> transformCache,
                               CompilationOptions options,
                               const TargetBackend &backend, bool isJIT,
                               MLIRContext &context, const std::string &linker,
                               PassManagerConfigOptions pmOptions)
    : transformCache(
          decltype(this->transformCache)::create(std::move(transformCache))),
      options(std::move(options)), backend(backend), isJIT(isJIT),
      pmOptions(std::move(pmOptions)), context(context),
      cpuDevice(*loadContext(&context)->get<AsyncRT::CPUDevice>()),
      linker(linker) {}

//===----------------------------------------------------------------------===//
// Time Trace Instrumentation
//===----------------------------------------------------------------------===//

/// Given a reference to an LLVM IR unit, return a string representation of
/// the name of the unit.
static std::string getLLVMIRName(llvm::IRUnitRef ir) {
  if (const auto *m = llvm::dyn_cast<llvm::Module>(ir))
    return ("[module](" + m->getName() + ")").str();
  if (const auto *fn = llvm::dyn_cast<llvm::Function>(ir))
    return fn->getName().str();
  if (const auto *scc = llvm::dyn_cast<llvm::LazyCallGraph::SCC>(ir))
    return scc->getName();
  if (const auto *loop = llvm::dyn_cast<llvm::Loop>(ir))
    return loop->getName().str();
  llvm_unreachable("unknown wrapped IR type");
}

namespace {
class LLVMTimeTraceInstrumentation {
public:
  LLVMTimeTraceInstrumentation(llvm::PassInstrumentationCallbacks &pic) {
    pic.registerBeforeNonSkippedPassCallback(
        [=](StringRef passID, llvm::IRUnitRef ir) {
          runBeforePass(passID, ir);
        });
    pic.registerAfterPassCallback(
        [=](StringRef, llvm::IRUnitRef, const llvm::PreservedAnalyses &) {
          runAfterPass();
        },
        /*ToFront=*/true);
    pic.registerAfterPassInvalidatedCallback(
        [=](StringRef, const llvm::PreservedAnalyses &) { runAfterPass(); },
        true);
    pic.registerBeforeAnalysisCallback(
        [=](StringRef passID, llvm::IRUnitRef ir) {
          runBeforePass(passID, ir);
        });
    pic.registerAfterAnalysisCallback(
        [=](StringRef, llvm::IRUnitRef) { runAfterPass(); }, /*ToFront=*/true);
  }

private:
  static void runBeforePass(StringRef passID, llvm::IRUnitRef ir) {
    VerboseCompilerProfilerEntry::createAndPush(passID, getLLVMIRName(ir));
  }

  static void runAfterPass() { VerboseCompilerProfilerEntry::endAndPop(); }
};
} // namespace

/// Run the default LLVM optimization pipeline based on the select optimization
/// level.
static LogicalResult runLLVMOptPasses(llvm::Module &module,
                                      llvm::TargetMachine &targetMachine,
                                      const CompilationOptions &options,
                                      AsyncRT::CPUDevice &cpuDevice) {
  CompilerTimeTraceScope traceScope("llvm-optimize", module.getName());
  using namespace llvm;

  LoopAnalysisManager loopAnalysisMgr;
  FunctionAnalysisManager funcAnalysisMgr;
  CGSCCAnalysisManager sccAnalysisMgr;
  ModuleAnalysisManager moduleAnalysisMgr;

  llvm::PassInstrumentationCallbacks pic;
  LLVMTimeTraceInstrumentation timeTraceInstrumentation(pic);

  StandardInstrumentations standardInstrumentations(module.getContext(),
                                                    /*DebugLogging=*/false);
  standardInstrumentations.registerCallbacks(pic, &moduleAnalysisMgr);

  TargetLibraryInfoImpl targetLibInfo(Triple(module.getTargetTriple()));
  PassBuilder passBuilder(&targetMachine, PipelineTuningOptions(),
                          /*PGOOpt=*/std::nullopt, &pic);

  // Specially handle the alias analysis manager so that we can register
  // a custom pipeline of AA passes with it.
  AAManager analysisAnalysisMgr;
  if (llvm::Error err =
          passBuilder.parseAAPipeline(analysisAnalysisMgr, "default")) {
    errs() << toString(std::move(err)) << "\n";
    return failure();
  }

  // Register the alias analysis manager first so that our version is the one
  // used.
  funcAnalysisMgr.registerPass([&] { return std::move(analysisAnalysisMgr); });
  // Register our TargetLibraryInfoImpl.
  funcAnalysisMgr.registerPass(
      [&] { return TargetLibraryAnalysis(targetLibInfo); });

  // Register all the basic analyses with the managers.
  passBuilder.registerModuleAnalyses(moduleAnalysisMgr);
  passBuilder.registerCGSCCAnalyses(sccAnalysisMgr);
  passBuilder.registerFunctionAnalyses(funcAnalysisMgr);
  passBuilder.registerLoopAnalyses(loopAnalysisMgr);
  passBuilder.crossRegisterProxies(loopAnalysisMgr, funcAnalysisMgr,
                                   sccAnalysisMgr, moduleAnalysisMgr);

  ModulePassManager modulePassMgr =
      buildLLVMOptimizationPipeline(passBuilder, options);

  // Now that we have all of the passes ready, run them.
  modulePassMgr.run(module, moduleAnalysisMgr);
  return mlir::success();
}

/// Run the default llc passes required to generate object code.
static LogicalResult
runLlcPasses(llvm::Module &module, CompilationOptions &options,
             llvm::TargetMachine &targetMachine, llvm::raw_pwrite_stream &os,
             std::unique_ptr<llvm::MachineModuleInfo> &machineModuleInfo,
             std::unique_ptr<llvm::MCContext> &mcContext,
             llvm::CodeGenFileType fileType, bool stopBeforeAsmPrint,
             unsigned numFunctionsBase = 0,
             llvm::TargetMachine *sharedTargetMachine = nullptr) {
  CompilerTimeTraceScope traceScope("llvm-codegen", module.getName());
  using namespace llvm;

  // Build up all of the passes that we want to do to the module.
  legacy::PassManager passMgr;

  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  TargetLibraryInfoImpl targetLibInfo(Triple(module.getTargetTriple()));
  passMgr.add(new TargetLibraryInfoWrapperPass(targetLibInfo));

  // Add RuntimeLibraryInfoWrapper for libcall lowering decisions.
  // This is required by passes like ExpandFp and PreISelIntrinsicLowering.
  const llvm::TargetOptions &tmOptions = targetMachine.Options;
  passMgr.add(new RuntimeLibraryInfoWrapper(
      tmOptions.ExceptionModel, tmOptions.EABIVersion,
      tmOptions.MCOptions.ABIName, tmOptions.VecLib));

#ifndef MODULAR_PRODUCTION
  // Verify module immediately to catch problems before doInitialization() is
  // called on any passes.
  if (verifyModule(module, &errs()))
    return failure();
#endif

  TargetMachine &llvmTargetMachine =
      static_cast<TargetMachine &>(targetMachine);

  MachineModuleInfoWrapperPass *machineModInfoPass;

  if (stopBeforeAsmPrint) {
    if (sharedTargetMachine) {
      mcContext = std::make_unique<llvm::MCContext>(
          sharedTargetMachine->getTargetTriple(),
          sharedTargetMachine->getMCAsmInfo(),
          sharedTargetMachine->getMCRegisterInfo(),
          sharedTargetMachine->getMCSubtargetInfo(), nullptr, false);

    } else {
      mcContext = std::make_unique<llvm::MCContext>(
          llvmTargetMachine.getTargetTriple(), llvmTargetMachine.getMCAsmInfo(),
          llvmTargetMachine.getMCRegisterInfo(),
          llvmTargetMachine.getMCSubtargetInfo(), nullptr, false);
    }

    machineModInfoPass =
        new MachineModuleInfoWrapperPass(&llvmTargetMachine, &(*mcContext));

    mcContext->setObjectFileInfo(llvmTargetMachine.getObjFileLowering());

    if (KGEN::addPassesToEmitMC(options, llvmTargetMachine, passMgr, os, true,
                                machineModInfoPass, numFunctionsBase))

      return failure();
  } else {
    machineModInfoPass = new MachineModuleInfoWrapperPass(&llvmTargetMachine);
    if (KGEN::addPassesToEmitFile(options, llvmTargetMachine, passMgr, os,
                                  nullptr, fileType, true, machineModInfoPass))
      return failure();
  }

  const_cast<TargetLoweringObjectFile *>(llvmTargetMachine.getObjFileLowering())
      ->Initialize(machineModInfoPass->getMMI().getContext(), targetMachine);

  passMgr.run(module);

  if (stopBeforeAsmPrint) {
    machineModuleInfo = std::make_unique<llvm::MachineModuleInfo>(
        std::move(machineModInfoPass->getMMI()));
  }

  return mlir::success();
}

/// Compile optimized llvm::Module module to object through the llc pipeline
/// asynchronously and cache the transformation.
static AsyncRT::AnyAsyncValueRef compileOptimizedLLVMModuleToObject(
    LLVMModuleAndContext module, Location loc,
    llvm::TargetMachine &targetMachine, std::mutex &tmMutex,
    AsyncRT::CPUDevice &cpuDevice, bool isJIT, bool isParLLC,
    CompilationOptions options, RCRef<Cache::TransformCache> transformCache,
    std::optional<size_t> moduleIdx, std::optional<size_t> splitIdx,
    unsigned numFunctionBase, const TargetBackend &backend) {
  WriteableBufferRef keyBuf;
  size_t nonBitcodeKeySize = 0;

  // No need to reload the module to a different context if we are not
  // going to further parallelizing compilation.
  // This is essential for some backends to avoid false hit
  // with stale AnnotationCache which is populated during both
  // llvm-opt and llc pipeline passes but is only cleared at the end of
  // codegen in AsmPrint. We need to make sure that llvm-opt and llc
  // are using the same llvm::Module so that the cache can be properly
  // cleaned.
  if (isParLLC) {
    keyBuf = WriteableBuffer::get();
    options.print(*keyBuf << "compileOptimizedLLVMModuleToObject(");
    *keyBuf << ")";
    nonBitcodeKeySize = keyBuf->getBufferSize();
    llvm::WriteBitcodeToFile(*module, *keyBuf);
    // Release memory.
    module.reset();
  }

  auto output = AsyncRT::AsyncValueRef<MCInfo>::allocate(cpuDevice);

  cpuDevice.getWorkQueue()->addTask(
      [nonBitcodeKeySize, loc, keyBuf = keyBuf.copy(), output = output.copy(),
       options, isJIT, isParLLC, moduleIdx, splitIdx, numFunctionBase, &backend,
       inputModule = std::move(module), &targetMachine, &tmMutex]() mutable {
        if (backend.isCodegenInterprocedural() && isParLLC) {
          return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
              "cannot do per function codegen for an inter-procedural backend.",
              loc));
        }
        LLVMModuleAndContext moduleAndContext;
        if (isParLLC) {
          BufferRef keyBufRef(std::move(keyBuf));
          StringRef bitcodeBuffer = keyBufRef->getBuffer();
          bitcodeBuffer = bitcodeBuffer.drop_front(nonBitcodeKeySize);

          // Load the cached bytecode into a new context.
          // This is necessary to avoid data races during multi-threading with
          // per function parallelization.
          ErrorOrSuccess createModuleResult = moduleAndContext.create(
              [&](llvm::LLVMContext &ctx)
                  -> ErrorOr<std::unique_ptr<llvm::Module>> {
                llvm::Expected<std::unique_ptr<llvm::Module>> moduleOr =
                    llvm::parseBitcodeFile(
                        llvm::MemoryBufferRef(bitcodeBuffer, ""), ctx);
                if (!moduleOr)
                  return Error("failed to create LLVMModuleAndContext");
                return std::move(*moduleOr);
              });
          if (createModuleResult) {
            return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
                "failed to load LLVM IR bitcode", loc));
          }
        } else {
          moduleAndContext = std::move(inputModule);
        }

        // Create TargetMachine for this module. This is also necessary to
        // avoid data races during multi-threading.
        ErrorOr<std::unique_ptr<llvm::TargetMachine>> machineOr =
            createTargetMachine(options, isJIT);
        if (failed(machineOr)) {
          return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
              "failed to create TargetMachine", loc));
        }

        llvm::TargetMachine &llvmTargetMachine =
            static_cast<llvm::TargetMachine &>(**machineOr);
        llvmTargetMachine.Options.MCOptions.AsmVerbose = options.verboseOutput;
        llvmTargetMachine.Options.MCOptions.PreserveAsmComments =
            options.verboseOutput;

        std::string saveTempsPrefix = options.saveTempsPrefix;
        if (!options.saveTempsPrefix.empty()) {
          if (moduleIdx)
            saveTempsPrefix += "_" + std::to_string(*moduleIdx);
          if (splitIdx)
            saveTempsPrefix += "__" + std::to_string(*splitIdx);
        }

        if (failed(writeTempModule(saveTempsPrefix, ".pre-llc",
                                   *moduleAndContext))) {
          return std::move(output).setToError(
              AsyncRT::getMLIRDiagnostic("failed save pre-llc llvm IR", loc));
        }

        std::unique_ptr<llvm::MachineModuleInfo> machineModuleInfo;
        std::unique_ptr<llvm::MCContext> mcContext;
        auto buf = WriteableBuffer::get();

        // Run llc passes.
        if (failed(runLlcPasses(*moduleAndContext, options, **machineOr, *buf,
                                machineModuleInfo, mcContext,
                                llvm::CodeGenFileType::ObjectFile,
                                /*stopBeforeAsmPrint=*/true, numFunctionBase,
                                &targetMachine))) {
          return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
              "llc failed to codegen LLVM IR to object code", loc));
        }

        if (!options.saveTempsPrefix.empty()) {
          if (failed(writeTempModule(saveTempsPrefix, ".post-llc",
                                     *moduleAndContext))) {
            return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
                "failed save post-llc llvm IR", loc));
          }
        }

        llvm::StringMap<const llvm::Function *> fnNameToFnPtr;
        auto wbuf = WriteableBuffer::get();

        for (llvm::Function &fn : moduleAndContext->functions())
          fnNameToFnPtr.insert({fn.getName().str(), &fn});

        llvm::WriteBitcodeToFile(*moduleAndContext, *wbuf);

        // Move and reset SubtargetInfo for MachineFunctions to
        // shared TargetMachine so as to reduce memory footprint
        // as soon as possible before reaching the mclinking barrier.
        {
          // Need to use a mutex here while modifying the shared targetMachine.
          std::lock_guard<std::mutex> lock(tmMutex);
          resetSubtargetInfo(targetMachine, *machineModuleInfo);
        }

        // Release more memory before reaching the mclinking barrier.
        releaseTargetMachineConstants(**machineOr);

        std::move(output).emplace(
            wbuf, std::move(machineModuleInfo),
            // Keep the original llvm::Module alive so that the MachineFunction
            // reference to llvm::Function is still valid.
            std::move(moduleAndContext), fnNameToFnPtr, std::move(*machineOr),
            std::move(mcContext), splitIdx);
      });

  return output;
}

/// Optimize the llvm module to prepare for codegen object file.
static LogicalResult optimizeLLVMModule(llvm::Module &module,
                                        llvm::TargetMachine &targetMachine,
                                        CompilationOptions &options,
                                        AsyncRT::CPUDevice &cpuDevice,
                                        std::optional<size_t> moduleIdx) {
  llvm::DataLayout targetDataLayout =
      options.targetDataLayout.empty()
          ? targetMachine.createDataLayout()
          : llvm::DataLayout(options.targetDataLayout);
  module.setDataLayout(targetDataLayout);

  std::string saveTempsPrefix = options.saveTempsPrefix;
  if (moduleIdx && !options.saveTempsPrefix.empty())
    saveTempsPrefix += "." + std::to_string(moduleIdx.value());

  if (failed(writeTempModule(saveTempsPrefix, ".pre-opt", module)))
    return failure();

  if (failed(runLLVMOptPasses(module, targetMachine, options, cpuDevice)))
    return failure();

  if (failed(writeTempModule(saveTempsPrefix, ".post-opt", module)))
    return failure();

  return success();
}

/// Compile the given LLVM module to object files and return the async values
/// that contains the compiled object file.
/// isParLLC is true: split module into per function for parallel llc lowering
///                   and return multiple object files.
/// isParLLC is false: compile module without splitting into one object file.
static SmallVector<AsyncRT::AnyAsyncValueRef> compileOptimizedLLVMToObjects(
    LLVMModuleAndContext module, mlir::Location loc,
    llvm::TargetMachine &targetMachine, std::mutex &tmMutex,
    CompilationOptions &options, AsyncRT::CPUDevice &cpuDevice,
    RCRef<Cache::TransformCache> transformCache, bool isParLLC, bool isJIT,
    std::optional<size_t> moduleIdx, SymbolAndMCInfo &symbolAndMirInfo,
    unsigned numFunctionBase, const TargetBackend &backend) {
  CompilerTimeTraceScope traceScope("compile-optimized-llvm-to-object",
                                    module->getName());

  // Perform module materialization in another task.
  auto launchCompilation = [&](llvm::unique_function<LLVMModuleAndContext()>
                                   produceModule,
                               std::optional<int64_t> idx,
                               unsigned numFunctions, bool isParLLC) {
    auto result = AsyncRT::AsyncValueRef<MCInfo>::allocate(cpuDevice);

    cpuDevice.getWorkQueue()->addTask([produceModule = std::move(produceModule),
                                       loc, &cpuDevice, isJIT, isParLLC,
                                       &options, cache = transformCache.copy(),
                                       moduleIdx, idx, result = result.copy(),
                                       numFunctions, &targetMachine, &backend,
                                       &tmMutex]() mutable {
      AsyncRT::AnyAsyncValueRef output = compileOptimizedLLVMModuleToObject(
          produceModule(), loc, targetMachine, tmMutex, cpuDevice, isJIT,
          isParLLC, options, cache, moduleIdx, idx, numFunctions, backend);
      andThenSyncMoving(
          output, [result = std::move(result)](
                      MutableArrayRef<AnyAsyncValueRef> outputs) mutable {
            for (auto &out : outputs) {
              if (out.isError())
                return std::move(result).setToError(out.takeDiagnostic());
            }
            std::move(result).emplace(std::move(outputs.front().get<MCInfo>()));
          });
    });
    return result;
  };

  SmallVector<AsyncRT::AnyAsyncValueRef> cacheResults;
  if (!isParLLC) {
    cacheResults.push_back(launchCompilation(forwardModule(std::move(module)),
                                             std::nullopt, numFunctionBase,
                                             isParLLC));
  } else {
    if (failed(writeTempModule(options.saveTempsPrefix, ".pre-llc-split",
                               *module))) {
      auto error = AsyncRT::AnyAsyncValueRef::createError(
          cpuDevice,
          AsyncRT::getMLIRDiagnostic(
              "writing module to file before llc split failed", loc));
      cacheResults.push_back(std::move(error));
      return cacheResults;
    }
    splitPerFunction(
        std::move(module),
        [&](llvm::unique_function<LLVMModuleAndContext()> produceModule,
            std::optional<int64_t> idx, unsigned numFunctions) {
          cacheResults.push_back(launchCompilation(
              std::move(produceModule), idx, numFunctions, isParLLC));
        },
        symbolAndMirInfo.symbolLinkageTypes, moduleIdx ? *moduleIdx : 0,
        numFunctionBase);
  }
  return cacheResults;
}

//===----------------------------------------------------------------------===//
// createTargetMachine
//===----------------------------------------------------------------------===//

ErrorOr<std::unique_ptr<llvm::TargetMachine>>
KGEN::createTargetMachine(const CompilationOptions &options, bool isJIT) {
  ErrorOr<const TargetBackend *> backendOr =
      TargetBackendRegistry::get().lookup(llvm::Triple(options.targetTriple));
  if (backendOr.isError())
    return Error(backendOr.getError());
  // TargetMachine creation runs in the host's LLVM instance (it uses the target
  // registry); the backend only adjusts the options.
  return defaultCreateTargetMachine(
      (*backendOr)
          ->adjustOptionsForTargetMachine(options, options.targetTriple),
      isJIT);
}

//===----------------------------------------------------------------------===//
// lowerAllFuncsToLLVM
//===----------------------------------------------------------------------===//

/// If requested, attach sanitizer, etc. instrumentations to the given
/// module.
/// TODO: Eventually we should explore attaching this information at a higher
/// level of the stack.
static void attachInstrumentationAttributes(llvm::Module &module,
                                            const CompilationOptions &options) {
  if (!options.sanitizers)
    return;

  for (llvm::Function &f : module.functions()) {
    if (f.isDeclaration())
      continue;
    if (options.sanitizers.has(Sanitizers::kAddress))
      f.addFnAttr(llvm::Attribute::SanitizeAddress);
    if (options.sanitizers.has(Sanitizers::kThread))
      f.addFnAttr(llvm::Attribute::SanitizeThread);
  }
}

static mlir::PassManager
createPassManager(const std::optional<std::string> &operationName,
                  MLIRContext *context) {
  if (operationName)
    return {context, *operationName};
  return {context};
}

static ErrorOr<std::unique_ptr<llvm::Module>>
loadBitcodeFile(llvm::LLVMContext &context, StringRef path) {
  if (!llvm::sys::fs::is_regular_file(path)) {
    return Error("Bitcode file path: " + path +
                 " does not exist or is not a file.");
  }

  llvm::SMDiagnostic error;
  std::unique_ptr<llvm::Module> library =
      llvm::getLazyIRFileModule(path, error, context);
  if (!library) {
    return Error("Failed loading bitcode file from " + path +
                 ", error: " + error.getMessage());
  }

  return library;
}

/// Load an LLVM module from a DenseResourceElementsAttr containing bitcode.
static ErrorOr<std::unique_ptr<llvm::Module>>
loadBitcodeFromResource(llvm::LLVMContext &context,
                        DenseResourceElementsAttr bitcodeAttr) {
  mlir::AsmResourceBlob *blob = bitcodeAttr.getRawHandle().getBlob();
  if (!blob)
    return Error("Failed to get bitcode blob from resource attribute");

  ArrayRef<char> bitcodeData = blob->getData();
  // Wrap the borrowed blob bytes in an owning MemoryBuffer handle. This does
  // not copy the data (RequiresNullTerminator=false wraps the existing bytes);
  // the lazy module takes ownership of the handle so the buffer stays alive
  // for on-demand materialization during linking.
  std::unique_ptr<llvm::MemoryBuffer> buffer = llvm::MemoryBuffer::getMemBuffer(
      StringRef(bitcodeData.begin(), bitcodeData.size()), "package_bitcode",
      /*RequiresNullTerminator=*/false);

  llvm::Expected<std::unique_ptr<llvm::Module>> moduleOr =
      llvm::getOwningLazyBitcodeModule(std::move(buffer), context);
  if (!moduleOr)
    return Error("Failed to parse bitcode from package: " +
                 llvm::toString(moduleOr.takeError()));

  return std::move(*moduleOr);
}

/// Link vendor-provided LLVM bitcode libraries into the LLVM module when
/// necessary.
static ErrorOrSuccess
linkBitcodeLibraries(Location loc, llvm::Module &llvmModule,
                     const CompilationOptions &options,
                     SmallVector<std::pair<bool, Attribute>> &bitcodeLibs,
                     const TargetBackend &backend) {
  // Vendor device/runtime bitcode is linked by the target backend; backends
  // without device libraries no-op. Then fall through to the standard logic
  // for our custom bitcode libraries.
  ErrorOrSuccess result =
      backend.linkRuntimeLibraries(loc, llvmModule, options);
  if (failed(result))
    return result;

  // Use standard linking procedure for custom bitcode libraries.
  if (bitcodeLibs.empty())
    return success();

  // Check if there are any extern functions in the module.
  bool hasExternFunctions = false;
  for (auto &fn : llvmModule.functions()) {
    if (fn.isDeclaration() &&
        (fn.hasExternalLinkage() || fn.hasHiddenVisibility()) &&
        !fn.isIntrinsic()) {
      hasExternFunctions = true;
      break;
    }
  }

  // By default a backend links only the bitcode-library symbols referenced by
  // unresolved extern functions, and skips linking entirely when there are
  // none. Some targets must link the full library instead; they opt out
  // through this backend policy.
  bool onlyLinkExtern = backend.onlyLinkExternFunctionsInBitcodeLibs(options);
  if (!hasExternFunctions && onlyLinkExtern)
    return success();

  llvm::Linker linker(llvmModule);

  // Lambda to link a loaded module, handling target triple check and linking
  auto linkModule =
      [&](std::unique_ptr<llvm::Module> libModule) -> ErrorOrSuccess {
    // Only check target triple for external bitcode libraries, the data layout
    // may be different e.g. for functions intended to be called from PTX.
    if (libModule->getTargetTriple() != llvmModule.getTargetTriple())
      return success();

    llvm::Linker::Flags linkerFlags = onlyLinkExtern
                                          ? llvm::Linker::Flags::LinkOnlyNeeded
                                          : llvm::Linker::Flags::None;

    bool err = linker.linkInModule(
        std::move(libModule), linkerFlags,
        [](llvm::Module &m, const StringSet<> &gvs) {
          llvm::internalizeModule(m, [&gvs](const llvm::GlobalValue &gv) {
            return !gv.hasName() || (gvs.count(gv.getName()) == 0);
          });
        });

    if (err)
      return Error("Unrecoverable failure during bitcode linking.");
    return success();
  };

  // Link bitcode libraries and track usage.
  for (auto &[used, library] : bitcodeLibs) {
    if (auto stringAttr = dyn_cast<StringAttr>(library)) {
      // Handle file path libraries.
      ErrorOr<std::unique_ptr<llvm::Module>> loadResult =
          loadBitcodeFile(llvmModule.getContext(), stringAttr.getValue());
      if (failed(loadResult))
        return loadResult.takeError();

      ErrorOrSuccess linkResult = linkModule(std::move(*loadResult));
      if (failed(linkResult))
        return linkResult.takeError();
      used |= true;
    } else if (auto resourceAttr =
                   dyn_cast<DenseResourceElementsAttr>(library)) {
      // Handle package bitcode libraries.
      ErrorOr<std::unique_ptr<llvm::Module>> loadResult =
          loadBitcodeFromResource(llvmModule.getContext(), resourceAttr);
      if (failed(loadResult))
        return loadResult.takeError();

      ErrorOrSuccess linkResult = linkModule(std::move(*loadResult));
      if (failed(linkResult))
        return linkResult.takeError();
      used |= true;
    }
  }
  return success();
}

static std::unique_ptr<llvm::Module>
translateModuleToLLVMIR(llvm::LLVMContext &ctx, ModuleOp module,
                        const CompilationOptions &options) {
  // Use the input filename for the module name if possible.
  StringRef moduleName = "LLVMDialectModule";
  if (auto moduleLoc = module.getLoc()->findInstanceOf<FileLineColLoc>())
    moduleName = llvm::sys::path::filename(moduleLoc.getFilename());

  // Translate the operation into an LLVM module.
  CompilerTimeTraceScope mlirScope("mlir-to-llvmir");
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(module, ctx, moduleName);
  if (!llvmModule)
    return nullptr;

  // Attach any necessary instrumentation to the module.
  attachInstrumentationAttributes(*llvmModule, options);
  return llvmModule;
}

ErrorOr<std::unique_ptr<llvm::Module>>
ObjectCompiler::lowerAllFuncsToLLVM(llvm::LLVMContext &ctx, ModuleOp module) {
  CompilerTimeTraceScope traceScope("lower-to-llvm");

  mlir::PassManager mgr = createPassManager(pmOptions.operationName, &context);

  // Ignore potential error that could happen when `mojo` tool is called,
  // which does not register pass manager options.
  (void)pmOptions.configurePassManager(mgr);

  backend.prepareModuleForLowering(module, options);

  LowerToLLVMOptions llvmOptions(
      options.optimizationLevel, options.getDIEmissionKind(),
      options.debugAtLevel,
      static_cast<llvm::dwarf::SourceLanguage>(options.debugInfoLanguage));
  llvmOptions.globalCtorFnName = ExecutionEngine::getGlobalCtorFnName();
  llvmOptions.globalDtorFnName = ExecutionEngine::getGlobalDtorFnName();

  buildLowerToLLVMPipeline(mgr, llvmOptions);

  if (failed(writeTempModule(options.saveTempsPrefix, ".pre-llvm-dialect",
                             module, ".mlir")))
    return Error(
        "writing module to file before converting to LLVM Dialect failed");

  if (failed(mgr.run(module)))
    return Error("run LowerToLLVMPipeline failed");

  if (failed(writeTempModule(options.saveTempsPrefix, ".pre-llvm-ir", module,
                             ".mlir")))
    return Error("writing module to file before converting to LLVM IR failed");

  // Translate the operation into an LLVM module.
  std::unique_ptr<llvm::Module> llvmModule =
      translateModuleToLLVMIR(ctx, module, options);
  if (!llvmModule)
    return Error("translate module to LLVMIR failed");

  return llvmModule;
}

SmallVector<AsyncRT::AnyAsyncValueRef>
ObjectCompiler::emitArchiveParallelCompilation(
    LLVMModuleAndContext llvmModule, Location opLoc,
    llvm::TargetMachine &targetMachine,
    llvm::StringMap<llvm::GlobalValue::LinkageTypes> &symbolLinkageTypes) {
  CompilerTimeTraceScope traceScope("split-input-module");

  bool noSplitting = cpuDevice.getWorkQueue()->getParallelismLevel() < 2;

  // Disable parLLC for inter-procedural backends.
  bool parLLC = cpuDevice.getWorkQueue()->getParallelismLevel() >= 2 &&
                options.enableParallelLLC &&
                !backend.isCodegenInterprocedural();

  // How the module is divided into independently-codegen'd units.
  SplitStrategy strategy = backend.splitStrategy(options);

  SmallVector<AsyncRT::AnyAsyncValueRef> cacheResults;

  if (noSplitting || strategy == SplitStrategy::None) {
    cacheResults.push_back(lowerLLVMModuleToObjects(
        forwardModule(std::move(llvmModule)), opLoc, targetMachine, parLLC,
        std::nullopt, /*numFunctionsBase=*/0));
  } else {
    (void)writeTempModule(options.saveTempsPrefix, ".pre-split", *llvmModule);

    auto handleSplit =
        [&](llvm::unique_function<LLVMModuleAndContext()> produceModule,
            std::optional<int64_t> idx, unsigned numFunctionsBase) {
          cacheResults.push_back(lowerLLVMModuleToObjects(
              std::move(produceModule), opLoc, targetMachine,
              strategy != SplitStrategy::PerFunction && parLLC, idx,
              numFunctionsBase));
        };
    if (strategy == SplitStrategy::PerFunction)
      splitPerFunction(std::move(llvmModule), handleSplit, symbolLinkageTypes);
    else
      splitPerExported(std::move(llvmModule), handleSplit);
  }
  return cacheResults;
}

ErrorOr<WriteableBufferRef> ObjectCompiler::emitArchiveMCLinking(
    MutableArrayRef<AnyAsyncValueRef> values, StringRef moduleName,
    bool emitAssembly,
    llvm::StringMap<llvm::GlobalValue::LinkageTypes> &symbolLinkageTypes,
    const llvm::StringMap<unsigned> &originalFnOrdering) {
  // If any of the cache results failed, propagate the error.
  for (auto &result : values) {
    if (result.isError())
      return Error(result.takeDiagnostic().getMessage().get());
  }
  CompilerTimeTraceScope traceScope("concatenate-object-files");

  // Link MC before printing.
  auto machineOr = createTargetMachine(options, /*isJIT=*/isJIT);
  if (failed(machineOr)) {
    return Error("failed to create TargetMachine");
  }

  SmallVector<SymbolAndMCInfo *> symbolAndMCInfos;
  symbolAndMCInfos.reserve(values.size());

  for (auto [i, result] : llvm::enumerate(values)) {
    auto &symbolAndMCInfo = result.get<SymbolAndMCInfo>();
    symbolAndMCInfos.emplace_back(&symbolAndMCInfo);
  }

  MCLinker mcLinker(symbolAndMCInfos, **machineOr, options, symbolLinkageTypes,
                    originalFnOrdering);
  ErrorOr<WriteableBufferRef> mcLinkResult =
      mcLinker.linkAndPrint(moduleName, emitAssembly);
  if (mcLinkResult.isError()) {
    return Error(mcLinkResult.getError());
  }

  return *mcLinkResult;
}

ErrorOrSuccess ObjectCompiler::emitArchiveSaveTemps(ModuleOp module,
                                                    StringRef moduleName) {
  // Generate saveTempsPrefix file for the assembly result of compilation.
  // This is expensive to do because  we need to go through llvm compilation
  // from the top so that AsmPrint can codegen properly for assembly output.
  // We can't use the compilation for AsmPrint with binary here because
  // AsmPrint writes back to the MC results such as SymbolTables etc. which
  // is not reusable for a second run of AsmPrint.
  auto output = AsyncRT::AsyncValueRef<BufferRef>::allocate(cpuDevice);
  LLVMModuleAndContext llvmModule;
  if (auto err = llvmModule.create([&](llvm::LLVMContext &ctx) {
        return translateModuleToLLVMIR(ctx, module, options);
      })) {
    return Error(
        Twine("failed to lower module to LLVM IR for archive compilation, ") +
        err.getError());
  }
  auto tmOr = createTargetMachine(options, isJIT);
  if (failed(tmOr))
    return tmOr.takeError();

  llvm::TargetMachine &tm = **tmOr;

  llvm::StringMap<llvm::GlobalValue::LinkageTypes> symbolLinkageTypes;

  SmallVector<AsyncRT::AnyAsyncValueRef> cachedResults =
      emitArchiveParallelCompilation(std::move(llvmModule), module->getLoc(),
                                     tm, symbolLinkageTypes);

  andThenSyncMoving(
      cachedResults,
      [this, moduleName, module, output = output.copy(), options = options,
       symbolLinkageTypes = std::move(symbolLinkageTypes)](
          MutableArrayRef<AnyAsyncValueRef> values) mutable {
        // If any of the cache results failed, propagate the error.
        for (auto &result : values) {
          if (result.isError())
            return std::move(output).setToError(result.takeDiagnostic());
        }

        ErrorOr<WriteableBufferRef> mcLinkResult =
            emitArchiveMCLinking(values, moduleName, /*emitAssembly=*/true,
                                 symbolLinkageTypes, /*originalFnOrdering=*/{});

        if (mcLinkResult.isError()) {
          return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
              Error(mcLinkResult.getError()), module->getLoc()));
        }

        WriteableBufferRef linkedObj = *mcLinkResult;
        StringRef toEmit(linkedObj->getBufferStart(),
                         linkedObj->getBufferSize());
        if (failed(writeBytesToTempWithHash(options.saveTempsPrefix, ".s",
                                            toEmit))) {
          return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
              "failed to save asm to saveTempsPrefix", module->getLoc()));
        }
        std::move(output).emplace(linkedObj.copy());
      });
  await(output);
  if (output.isError())
    return Error(output.takeDiagnostic().getMessage().get());

  return {};
}

// Compute the original order of Function in an llvm::Module.
// This is needed to help sort the linkedModule's functions list for backends
// that require the original function order (see requiresOriginalFunctionOrder).
static void computeFnOrdering(llvm::Module &module,
                              llvm::StringMap<unsigned> &result) {
  unsigned idx = 0;
  for (auto &func : module.functions()) {
    if (func.isDeclaration())
      continue;
    result.insert({func.getName(), idx++});
  }
}

//===----------------------------------------------------------------------===//
// emitArchive
//===----------------------------------------------------------------------===//

ErrorOr<BufferRef> ObjectCompiler::emitArchive(OwningOpRef<ModuleOp> module,
                                               bool emitAssembly,
                                               std::string *outKeyHash) {
  LLVMTimingRegion timingRegion(options);
  CompilerTimeTraceScope traceScope("produce-archive");

  auto tmOr = createTargetMachine(options, isJIT);
  if (failed(tmOr))
    return tmOr.takeError();

  llvm::TargetMachine &tm = **tmOr;

  // Perform a cache aware transformation to translate the module to an archive
  // file.
  auto runTransformation = [&](Operation *op, WriteableBufferRef buf,
                               AsyncRT::AnyAsyncValueRef chain) {
    auto output = AsyncRT::AsyncValueRef<BufferRef>::allocate(cpuDevice);
    chain.andThenSync([this, op, output = output.copy(), buf = buf.copy(), &tm,
                       emitAssembly]() mutable {
      // Lower the module to LLVM.
      LLVMModuleAndContext llvmModule;
      Location moduleLoc = op->getLoc();

      if (auto err = llvmModule.create([&](llvm::LLVMContext &ctx) {
            return lowerAllFuncsToLLVM(ctx, cast<ModuleOp>(op));
          })) {
        op->erase();
        return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
            Twine(
                "failed to lower module to LLVM IR for archive compilation, ") +
                err.getError(),
            moduleLoc));
      }

      // Link bitcode libraries.
      if (failed(linkBitcodeLibraries(moduleLoc, *llvmModule, options,
                                      bitcodeLibs, backend)))
        return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
            Error("failed to link bitcode libraries"), moduleLoc));

      // Split the module into multiple slices and compile each in parallel.
      [[maybe_unused]] bool needsOriginalOrder =
          backend.requiresOriginalFunctionOrder();
      assert((!needsOriginalOrder || emitAssembly) &&
             "backends requiring original function order should only emit "
             "assembly here");

      // Release mlir::ModuleOp before codegen happens to reduce memory
      // pressure.
      if (options.saveTempsPrefix.empty() || emitAssembly)
        op->erase();

      CompilerTimeTraceScope traceScope("split-input-module");

      std::string moduleName = llvmModule->getName().str();

      llvm::StringMap<unsigned> originalFnOrdering;

      // MCLinker changes function ordering in the linkedModule, but some
      // backends may need the original order to generate function
      // declarations properly and avoid use before def/decl illegal
      // instructions. Keep record of the ordering here so that we can sort the
      // linkedModule to its original order.
      if (needsOriginalOrder)
        computeFnOrdering(*llvmModule, originalFnOrdering);

      // Split the module into multiple slices and compile each in parallel.
      llvm::StringMap<llvm::GlobalValue::LinkageTypes> symbolLinkageTypes;
      SmallVector<AsyncRT::AnyAsyncValueRef> cachedResults =
          emitArchiveParallelCompilation(std::move(llvmModule), moduleLoc, tm,
                                         symbolLinkageTypes);

      andThenSyncMoving(
          cachedResults, [this, moduleName = std::move(moduleName), op,
                          moduleLoc, buf = buf.copy(), output = output.copy(),
                          emitAssembly, options = options, symbolLinkageTypes,
                          originalFnOrdering = std::move(originalFnOrdering)](
                             MutableArrayRef<AnyAsyncValueRef> values) mutable {
            ErrorOr<WriteableBufferRef> mcLinkResult =
                emitArchiveMCLinking(values, moduleName, emitAssembly,
                                     symbolLinkageTypes, originalFnOrdering);
            if (mcLinkResult.isError()) {

              return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
                  Error(mcLinkResult.getError()), moduleLoc));
            }

            WriteableBufferRef linkedObj = *mcLinkResult;
            if (emitAssembly) {
              ErrorOr<const TargetTraits *> traitsOr =
                  TargetTraitsRegistry::get().lookup(
                      llvm::Triple(options.targetTriple));
              if (traitsOr.isError()) {
                return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
                    Error(traitsOr.getError()), moduleLoc));
              }
              std::string postfix = (*traitsOr)->getAsmExtension().str();
              StringRef toEmit(linkedObj->getBufferStart(),
                               linkedObj->getBufferSize());
              if (failed(writeBytesToTempWithHash(options.saveTempsPrefix,
                                                  postfix, toEmit))) {
                return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
                    "failed to save asm to saveTempsPrefix", moduleLoc));
              }
              *buf << linkedObj->Buffer::getBuffer();
              std::move(output).emplace(buf.copy());
              return;
            }

            // Print assembly for saveTemps if needed.
            if (!options.saveTempsPrefix.empty()) {
              // Clear seenCodeGenFns since we are going to do the whole codegen
              // step all over again for printing saveTemps.
              seenCodeGenFns.clear();
              ErrorOrSuccess saveTempsResult =
                  emitArchiveSaveTemps(cast<ModuleOp>(op), moduleName);
              op->erase();
              if (saveTempsResult.isError()) {
                return std::move(output).setToError(AsyncRT::getMLIRDiagnostic(
                    saveTempsResult.takeError(), moduleLoc));
              }
            }

            // Copy the result into the output buffer.
            *buf << linkedObj->Buffer::getBuffer();
            std::move(output).emplace(buf.copy());
          });
    });
    return output;
  };
  auto onCacheHit = [](Operation *op, BufferRef buf) {
    op->erase();
    return buf.copy();
  };

  WriteableBufferRef produceArchiveKey = WriteableBuffer::get();
  options.print(*produceArchiveKey << "emitArchive(");
  *produceArchiveKey << ", isJIT=" << isJIT
                     << ", enableLLVMPerFunctionSplitting="
                     << options.enableLLVMPerFunctionSplitting
                     << ", emitAssembly=" << emitAssembly
                     << ", relocModel=" << options.relocModel
                     << ", verboseOutput=" << options.verboseOutput << ')';

  AsyncRT::AnyAsyncValueRef output = cachedTransform(
      module.release(), transformCache.copy(),
      AsyncRT::AsyncValueRef<Chain>::createReady(cpuDevice),
      std::move(produceArchiveKey), runTransformation, onCacheHit, outKeyHash);
  await(output);

  if (output.isError())
    return {std::move(output.takeDiagnostic().getMessage())};
  return {std::move(output.get<BufferRef>())};
}

//===----------------------------------------------------------------------===//
// lowerLLVMModuleToObjects
//===----------------------------------------------------------------------===//

AsyncRT::AsyncValueRef<SymbolAndMCInfo>
ObjectCompiler::lowerLLVMModuleToObjects(
    llvm::unique_function<LLVMModuleAndContext()> produceModule, Location loc,
    llvm::TargetMachine &targetMachine, bool parLLC,
    std::optional<size_t> moduleIdx, unsigned numFunctionsBase) {

  auto result = AsyncRT::AsyncValueRef<SymbolAndMCInfo>::allocate(cpuDevice);

  cpuDevice.getWorkQueue()->addTask(
      [this, result = result.copy(), produceModule = std::move(produceModule),
       loc, moduleIdx, parLLC, numFunctionsBase, &targetMachine]() mutable {
        CompilerTimeTraceScope traceScope("optimizeLLVMTask");

        // Materialize the module first.
        LLVMModuleAndContext module = produceModule();

        auto tmOr = createTargetMachine(this->options, isJIT);
        if (failed(tmOr)) {
          return std::move(result).setToError(
              AsyncRT::getMLIRDiagnostic(tmOr.takeError(), loc));
        }
        llvm::TargetMachine &tm = **tmOr;

        // Optimize the llvm Module.
        if (failed(optimizeLLVMModule(*module, tm, options, cpuDevice,
                                      moduleIdx))) {
          return std::move(result).setToError(
              AsyncRT::getMLIRDiagnostic("failed to optimize LLVM IR.", loc));
        }

        {
          // Deduplicate functions between splits.
          // A mutex is needed here to make access to seenCodeGenFns
          // thread-safe.
          std::lock_guard<std::mutex> lock(dedupMutex);
          for (auto &fn : module->functions()) {
            if (fn.isDeclaration())
              continue;
            if (!seenCodeGenFns.insert(fn.getName()).second)
              module.duplicatedFns.insert(fn.getName());
          }
        }

        SymbolAndMCInfo symbolAndMirInfo;
        SmallVector<AnyAsyncValueRef> buffers = compileOptimizedLLVMToObjects(
            std::move(module), loc, targetMachine, tmMutex, this->options,
            cpuDevice, transformCache, parLLC, isJIT, moduleIdx,
            symbolAndMirInfo, numFunctionsBase, backend);

        andThenAsyncMoving(
            buffers, [result = std::move(result),
                      symbolAndMirInfo = std::move(symbolAndMirInfo)](
                         MutableArrayRef<AnyAsyncValueRef> values) mutable {
              for (AnyAsyncValueRef &result : values)
                symbolAndMirInfo.mcInfos.emplace_back(
                    std::make_unique<MCInfo>(std::move(result.get<MCInfo>())));
              std::move(result).emplace(std::move(symbolAndMirInfo));
            });
      });

  return result;
}

//===----------------------------------------------------------------------===//
// lowerAllFuncsToLLVMAndOptimize
//===----------------------------------------------------------------------===//

ErrorOrSuccess ObjectCompiler::lowerAllFuncsToLLVMAndOptimize(
    ModuleOp module, LLVMModuleAndContext &llvmModule) {
  LLVMTimingRegion timingRegion(options);
  CompilerTimeTraceScope traceScope("lowerAllFuncsToLLVMAndOptimize");

  if (auto err = llvmModule.create([&](llvm::LLVMContext &ctx) {
        return lowerAllFuncsToLLVM(ctx, module);
      }))
    return err.takeError();

  // Link bitcode libraries.
  if (failed(linkBitcodeLibraries(module->getLoc(), *llvmModule, options,
                                  bitcodeLibs, backend)))
    return Error("failed to link bitcode libraries");

  auto machineOr = createTargetMachine(options, /*isJIT=*/false);
  if (failed(machineOr))
    return machineOr.takeError();

  if (failed(runLLVMOptPasses(*llvmModule, **machineOr, options, cpuDevice)))
    return Error("failed to run LLVM opt passes");
  return success();
}

//===----------------------------------------------------------------------===//
// emitLLVMIR
//===----------------------------------------------------------------------===//

ErrorOrSuccess ObjectCompiler::emitLLVMIR(ModuleOp module,
                                          llvm::raw_pwrite_stream &os) {
  LLVMTimingRegion timingRegion(options);
  CompilerTimeTraceScope traceScope("emitLLVMIR");

  LLVMModuleAndContext llvmModule;
  if (ErrorOrSuccess err = lowerAllFuncsToLLVMAndOptimize(module, llvmModule))
    return err.takeError();

  llvmModule->print(os, /*AAW=*/nullptr);
  return success();
}

//===----------------------------------------------------------------------===//
// emitAssembly
//===----------------------------------------------------------------------===//

ErrorOrSuccess ObjectCompiler::emitAssembly(OwningOpRef<ModuleOp> module,
                                            llvm::raw_pwrite_stream &os) {
  LLVMTimingRegion timingRegion(options);
  CompilerTimeTraceScope traceScope("emitAssembly");
  ErrorOr<BufferRef> buf =
      ObjectCompiler::emitArchive(std::move(module), /*emitAssembly=*/true);
  if (buf.isError())
    return Error(Twine("failed to lower LLVM IR to assembly:") +
                 buf.takeError().get());
  os << buf->getPointer()->getBuffer();
  return success();
}

ErrorOrSuccess ObjectCompiler::emitBitcode(llvm::Module &llvmModule,
                                           llvm::raw_pwrite_stream &os) {
  CompilerTimeTraceScope traceScope("emitBitcode");
  // The backend owns the bitcode format (e.g. Metal emits AIR bitcode).
  ErrorOr<const TargetBackend *> backendOr =
      TargetBackendRegistry::get().lookup(llvmModule.getTargetTriple());
  if (backendOr.isError())
    return Error(backendOr.getError());
  (*backendOr)->emitBitcode(llvmModule, os);
  return success();
}

//===----------------------------------------------------------------------===//
// emitSharedObject
//===----------------------------------------------------------------------===//

/// Utility function for creating shared object from buf
static ErrorOr<BufferRef> createSharedObject(BufferRef buf,
                                             CompilationOptions options,
                                             StringRef moduleName,
                                             const std::string &linker,
                                             const TargetBackend &backend) {
  llvm::StringRef libInExt = ".o";
  llvm::StringRef libOutExt = ".so";
  std::string objName = moduleName.str() + "-%%%%%%%" + libInExt.str();

  // Write .o to a file.
  auto objFileOr = writeTempFile(objName, buf->getBuffer());

  if (objFileOr.isError())
    return Error("failed to write object binary into a file");

  std::string objFilePath = objFileOr->getPath().string();
  std::string sharedObjName =
      objFileOr->getPath().stem().string() + libOutExt.str();
  std::error_code ec;
  std::filesystem::path sharedObjPath =
      std::filesystem::temp_directory_path(ec);
  sharedObjPath = sharedObjPath / sharedObjName;

  auto triple = llvm::Triple(options.targetTriple);
  std::string version = triple.getOSVersion().getAsString();
  std::string arch = "unknown";
  if (triple.getArch() == llvm::Triple::ArchType::aarch64)
    arch = "arm64";
  else if (triple.getArch() == llvm::Triple::ArchType::x86_64)
    arch = "x86_64";

  StringRef linkerFlavor = "gnu";
  if (triple.getObjectFormat() == llvm::Triple::MachO) {
    linkerFlavor = "darwin";
  }

  // Call lld to generate a dynamic library.
  // For ELF:
  //  ld.lld -shared tmp.o -o tmp.so
  // For MACHO (on MacOS)
  //  ld64.lld -platform_version macos 16.0 16.0 -arch arm64
  //           -dylib tmp.o -o tmp.so -undefined dynamic_lookup
  SmallVector<StringRef> lldArgs = [&]() -> SmallVector<StringRef> {
    if (triple.getObjectFormat() == llvm::Triple::MachO) {
      SmallVector<StringRef> args{
          linker,       "-flavor",       linkerFlavor,     "-platform_version",
          "macos",      version.c_str(), version.c_str(),  "-arch",
          arch.c_str(), "-undefined",    "dynamic_lookup", "-dylib"};

      if (!options.emissionLinkOptions.empty())
        args.push_back(options.emissionLinkOptions.c_str());
      args.push_back(objFilePath.c_str());
      args.push_back("-o");
      args.push_back(sharedObjPath.c_str());
      return args;
    }
    // Build ELF linker args, plus any backend-specific arguments.
    SmallVector<StringRef> args = {linker, "-flavor", linkerFlavor};
    backend.appendLinkArgs(args, options);
    args.push_back("-shared");
    if (!options.emissionLinkOptions.empty())
      args.push_back(options.emissionLinkOptions.c_str());
    args.push_back(objFilePath.c_str());
    args.push_back("-o");
    args.push_back(sharedObjPath.c_str());
    return args;
  }();

  std::string errorMsg;
  ErrorOr<TempFile> linkerErrorFileOr =
      TempFile::create("linker-error-%%%%%%.log");
  std::optional<TempFile> linkerErrorFile;
  if (!linkerErrorFileOr.isError()) {
    linkerErrorFile.emplace(std::move(*linkerErrorFileOr));
  }
  // If we couldn't create the error file, continue anyway - it's not critical.

  int linkExitCode = llvm::sys::ExecuteAndWait(
      lldArgs[0], lldArgs, /*Env=*/std::nullopt,
      /*Redirects=*/
      {/*stdin=*/std::nullopt, /*stdout=*/std::nullopt,
       /*stderr=*/
       linkerErrorFile ? std::make_optional(linkerErrorFile->getPath().string())
                       : std::nullopt},
      /*SecondsToWait=*/0, /*MemoryLimit=*/0, /*ErrMsg=*/&errorMsg);

  if (linkExitCode) {
    if (!errorMsg.empty())
      errorMsg.insert(0, ": ");
    std::string errorPrefix =
        ("failed to generate " + backend.name() + " shared object binary")
            .str();
    if (linkerErrorFile) {
      std::string linkerOutput =
          llvm::MemoryBuffer::getFile(linkerErrorFile->getPath().string())
              .get()
              ->getBuffer()
              .str();
      if (!linkerOutput.empty())
        return Error(Twine(errorPrefix) + errorMsg + ":\n" + linkerOutput);
    }
    return Error(Twine(errorPrefix) + errorMsg);
  }

  // Read linked dynamic library in to memory.
  ErrorOr<BufferRef> sharedObjBufOr =
      M::Buffer::getFile(sharedObjPath, std::nullopt, 0);
  if (sharedObjBufOr.isError())
    return Error("failed to open shared object binary");

  // Save to temp file if needed.
  if (failed(writeBytesToTempWithHash(options.saveTempsPrefix,
                                      std::string(".") +
                                          sharedObjPath.stem().c_str() + ".so",
                                      (*sharedObjBufOr)->getBuffer())))
    return Error("failed to write shared object binary to saveTemps");

  return sharedObjBufOr;
}

ErrorOrSuccess ObjectCompiler::emitSharedObject(OwningOpRef<ModuleOp> module,
                                                llvm::raw_pwrite_stream &os) {
  LLVMTimingRegion timingRegion(options);
  llvm::Triple triple(options.targetTriple);

  // Only ELF and MachO object formats are currently supported for
  // shared-object emission. Generalize to other platforms+formats when needed.
  if (!llvm::is_contained({llvm::Triple::ELF, llvm::Triple::MachO},
                          triple.getObjectFormat()))
    return Error("cannot create shared object binary from target triple that "
                 "is not ELF or MachO");

  CompilerTimeTraceScope traceScope("emitSharedObj");

  StringRef moduleName = "mojo-object";
  if (auto moduleLoc = module->getLoc()->findInstanceOf<FileLineColLoc>())
    moduleName = llvm::sys::path::filename(moduleLoc.getFilename());

  // Generate .o in memory.
  ErrorOr<BufferRef> bufOr =
      ObjectCompiler::emitArchive(std::move(module), /*emitAssembly=*/false);

  if (bufOr.isError())
    return Error("failed to lower LLVM IR to object binary");

  // Create shared object in buffer.
  ErrorOr<BufferRef> sharedObjBufOr =
      createSharedObject(*bufOr, options, moduleName, linker, backend);

  if (sharedObjBufOr.isError())
    return sharedObjBufOr.takeError();

  // Send dynamic library to output stream.
  os << sharedObjBufOr->getPointer()->getBuffer();

  return success();
}

static ErrorOr<std::pair<uint64_t, llvm::Function *>>
getKernelIDFromLLVMModule(llvm::Module &module) {
  for (llvm::Function &func : module) {
    if (func.isDeclaration())
      continue;

    const llvm::AttributeList &funcAttrs = func.getAttributes();

    for (auto &attr :
         funcAttrs.getAttributes(llvm::AttributeList::FunctionIndex)) {
      if (!attr.isStringAttribute())
        continue;
      if (attr.getKindAsString() == "kgen.offload.kernelid") {
        uint64_t kernelId;
        if (llvm::to_integer(attr.getValueAsString(), kernelId)) {
          // Remove the ID attribute so that caching won't take this into
          // consideration.
          llvm::AttributeList newList = funcAttrs.removeAttribute(
              func.getContext(), llvm::AttributeList::FunctionIndex,
              attr.getKindAsString());

          func.setAttributes(newList);
          return std::make_pair(kernelId, &func);
        }
      }
    }
  }

  return Error("Can't find kgen.offload.kernelid from the llvm split.");
}

static AnyAsyncValueRef
lowerLLVMModuleToObject(llvm::Module &inputModule, Location loc,
                        RCRef<Cache::TransformCache> transformCache,
                        size_t moduleIdx, AsyncRT::CPUDevice &cpuDevice,
                        CompilationOptions options, bool isJIT,
                        bool shouldDeserialize, EmitAs emissionKind,
                        std::string &linker, const TargetBackend &backend) {
  WriteableBufferRef keyBuf = WriteableBuffer::get();
  options.print(*keyBuf << "compileLLVMModuleToObject(");
  *keyBuf << ")";
  *keyBuf << " emitAs = " << emissionKind;
  *keyBuf << " isJIT = " << isJIT;
  if (!options.emissionOptions.empty())
    *keyBuf << " emissionOptions = " << options.emissionOptions;
  if (!options.emissionLinkOptions.empty())
    *keyBuf << " emissionLinkOptions = " << options.emissionLinkOptions;

  size_t nonBitcodeKeySize = keyBuf->getBufferSize();

  llvm::WriteBitcodeToFile(inputModule, *keyBuf);

  auto runTransformation = [loc, moduleIdx, isJIT, options, &cpuDevice,
                            emissionKind, keyBuf = keyBuf.copy(), &inputModule,
                            nonBitcodeKeySize, shouldDeserialize, &backend,
                            &linker](WriteableBufferRef buf,
                                     AsyncRT::AnyAsyncValueRef chain) mutable {
    auto output = AsyncRT::AsyncValueRef<BufferRef>::allocate(cpuDevice);

    chain.andThenAsync([loc, &cpuDevice, emissionKind, output = output.copy(),
                        buf = buf.copy(), keyBuf = std::move(keyBuf), options,
                        isJIT, moduleIdx, &inputModule, nonBitcodeKeySize,
                        shouldDeserialize, &backend, &linker]() mutable {
      CompilerTimeTraceScope traceScope("lowerLLVMModuleToObjectKernels");

      LLVMModuleAndContext deserializedModule;
      // We need to deserialize the llvm::Module into a separate copy here
      // when the same module split needs different emission kinds to avoid
      // data race on running optimization son the same module for different
      // parallel emission kind tasks. If there is only one emission kind,
      // we don't nee to do the extra deserialize since there will not be
      // any data race.
      if (shouldDeserialize) {
        if (auto err = deserializedModule.create(
                [&](llvm::LLVMContext &ctx)
                    -> ErrorOr<std::unique_ptr<llvm::Module>> {
                  BufferRef keyBufRef(std::move(keyBuf));
                  StringRef bitcodeBuffer = keyBufRef->getBuffer();
                  bitcodeBuffer = bitcodeBuffer.drop_front(nonBitcodeKeySize);

                  // Load the cached bytecode into a new context. This is
                  // necessary to avoid data races during multi-threading.
                  llvm::Expected<std::unique_ptr<llvm::Module>> moduleOr =
                      llvm::parseBitcodeFile(
                          llvm::MemoryBufferRef(bitcodeBuffer,
                                                inputModule.getName()),
                          ctx);
                  if (!moduleOr) {
                    return Error("failed to load LLVM IR bitcode");
                  }
                  return std::move(*moduleOr);
                })) {
          return std::move(output).setToError(
              AsyncRT::getMLIRDiagnostic(err.takeError(), loc));
        }
      }

      llvm::Module &module =
          shouldDeserialize ? *deserializedModule : inputModule;

      if (emissionKind == EmitAs::LLVM) {
        *buf << module;
        std::move(output).emplace(buf.copy());
        return;
      }
      if (emissionKind == EmitAs::LLVM_BITCODE) {
        if (ErrorOrSuccess err = ObjectCompiler::emitBitcode(module, *buf)) {
          return std::move(output).setToError(
              AsyncRT::getMLIRDiagnostic(err.takeError(), loc));
        }
        std::move(output).emplace(buf.copy());
        return;
      }

      // Create the target machine.
      std::string moduleTriple = module.getTargetTriple().getTriple();

      CompilationOptions adjustedOptions =
          backend.adjustOptionsForTargetMachine(options, moduleTriple);
      module.setTargetTriple(llvm::Triple(adjustedOptions.targetTriple));

      // `adjustedOptions` is already target-adjusted above, so build the
      // TargetMachine directly without re-dispatching through the backend.
      auto tmOr = defaultCreateTargetMachine(adjustedOptions, isJIT);
      if (failed(tmOr)) {
        return std::move(output).setToError(
            AsyncRT::getMLIRDiagnostic(tmOr.takeError(), loc));
      }
      llvm::TargetMachine &tm = **tmOr;

      backend.finalizeModuleForTarget(module, tm, moduleTriple);

      // Optimize the llvm Module.
      if (failed(
              optimizeLLVMModule(module, tm, options, cpuDevice, moduleIdx))) {
        return std::move(output).setToError(
            AsyncRT::getMLIRDiagnostic("failed to optimize LLVM IR.", loc));
      }

      if (emissionKind == EmitAs::LLVM_OPT) {
        *buf << module;
        std::move(output).emplace(buf.copy());
        return;
      }
      if (emissionKind == EmitAs::LLVM_OPT_BITCODE) {
        if (ErrorOrSuccess err = ObjectCompiler::emitBitcode(module, *buf)) {
          return std::move(output).setToError(
              AsyncRT::getMLIRDiagnostic(err.takeError(), loc));
        }
        std::move(output).emplace(buf.copy());
        return;
      }

      std::unique_ptr<llvm::MachineModuleInfo> machineModuleInfo;
      std::unique_ptr<llvm::MCContext> mcContext;

      // Primitives the backend hooks need, bound here so they can reach the
      // file-local runLlcPasses / createSharedObject helpers.
      auto runLlc = [&](llvm::Module &m, WriteableBuffer &out,
                        bool createObjectFile) -> ErrorOrSuccess {
        if (failed(runLlcPasses(
                m, options, tm, out, machineModuleInfo, mcContext,
                createObjectFile ? llvm::CodeGenFileType::ObjectFile
                                 : llvm::CodeGenFileType::AssemblyFile,
                /*stopBeforeAsmPrint=*/false, /*numFunctionsBase=*/0,
                /*sharedTargetMachine=*/nullptr))) {
          return Error(createObjectFile
                           ? "llc failed to codegen LLVM IR to object code"
                           : "llc failed to codegen LLVM IR to assembly");
        }
        return {};
      };
      auto linkObject = [&](BufferRef object,
                            StringRef moduleName) -> ErrorOr<BufferRef> {
        return createSharedObject(object, options, moduleName, linker, backend);
      };
      EmitContext backendCtx{options,
                             tm,
                             loc,
                             moduleIdx,
                             /*linker=*/nullptr,
                             /*linkerPath=*/linker,
                             runLlc,
                             linkObject};

      if (emissionKind == EmitAs::ASM) {
        ErrorOr<BufferRef> asmOr = backend.emitAssembly(module, backendCtx);
        if (asmOr.isError())
          return std::move(output).setToError(
              AsyncRT::getMLIRDiagnostic(asmOr.takeError(), loc));
        (*buf) << (*asmOr)->getBuffer();
        std::move(output).emplace(buf.copy());
        return;
      } else {
        ErrorOr<BufferRef> bufOr = backend.emitObject(module, backendCtx);
        if (bufOr.isError()) {
          return std::move(output).setToError(
              AsyncRT::getMLIRDiagnostic(bufOr.takeError(), loc));
        }
        (*buf) << (*bufOr)->getBuffer();

        std::move(output).emplace(buf.copy());
      }
    });
    return output;
  };

  auto onCacheHit = [&](BufferRef buf) { return buf.copy(); };
  return Cache::cachedTransform(
      AsyncRT::MLIRLocationDecoder::getEncodedLocation(loc),
      transformCache.copy(),
      AsyncRT::AsyncValueRef<Chain>::createReady(cpuDevice), keyBuf.copy(),
      std::move(runTransformation), onCacheHit);
}

static std::pair<AnyAsyncValueRef, AnyAsyncValueRef> lowerLLVMModuleToObject(
    llvm::unique_function<LLVMModuleAndContext()> produceModule, Location loc,
    RCRef<Cache::TransformCache> transformCache,
    std::optional<size_t> moduleIdx, AsyncRT::CPUDevice &cpuDevice,
    CompilationOptions options, bool isJIT,
    DenseMap<uint64_t, llvm::SmallSet<EmitAs, 4>> &kernelEmissionKinds,
    std::string &linker, SmallVector<std::pair<bool, Attribute>> &bitcodeLibs,
    const TargetBackend &backend) {
  auto resultBufs =
      AsyncRT::AsyncValueRef<DenseMap<EmitAs, BufferRef>>::allocate(cpuDevice);
  auto resultKernelId = AsyncRT::AsyncValueRef<uint64_t>::allocate(cpuDevice);

  cpuDevice.getWorkQueue()->addTask([resultBufs = resultBufs.copy(),
                                     resultKernelId = resultKernelId.copy(),
                                     produceModule = std::move(produceModule),
                                     loc, isJIT, options, &cpuDevice,
                                     transformCache = transformCache.copy(),
                                     &kernelEmissionKinds, &linker, &backend,
                                     &bitcodeLibs]() mutable {
    CompilerTimeTraceScope traceScope("lowerLLVMModuleToObjectKernels");

    // Materialize the module.
    LLVMModuleAndContext module = produceModule();

    ErrorOr<std::pair<uint64_t, llvm::Function *>> kernelIdFuncOr =
        getKernelIDFromLLVMModule(*module);
    if (kernelIdFuncOr) {
      std::move(resultBufs)
          .setToError(AsyncRT::getMLIRDiagnostic("Can't find kernelId", loc));
      std::move(resultKernelId)
          .setToError(
              AsyncRT::getMLIRDiagnostic(kernelIdFuncOr.takeError(), loc));
      return;
    }

    // Link bitcode libraries.
    ErrorOrSuccess linkResult =
        linkBitcodeLibraries(loc, *module, options, bitcodeLibs, backend);
    if (failed(linkResult)) {
      std::move(resultBufs)
          .setToError(AsyncRT::getMLIRDiagnostic(
              "failed to link bitcode libraries", loc));
      return;
    }

    uint64_t kernelId = (*kernelIdFuncOr).first;
    llvm::Function *kernelEntry = (*kernelIdFuncOr).second;
    backend.attachCodegenAttributes(kernelEntry);

    SmallVector<EmitAs> emissionKinds;
    SmallVector<AsyncRT::AnyAsyncValueRef> emissionResults;
    llvm::SmallSet<EmitAs, 4> &kinds = kernelEmissionKinds[kernelId];
    bool shouldDeserialize = kinds.size() > 1;
    bool shouldRunExtraAsm = !options.saveTempsPrefix.empty() &&
                             kinds.contains(EmitAs::OBJECT) &&
                             !kinds.contains(EmitAs::ASM);
    shouldDeserialize |= shouldRunExtraAsm;

    for (EmitAs kind : kinds) {
      emissionKinds.push_back(kind);
      emissionResults.push_back(lowerLLVMModuleToObject(
          *module, loc, transformCache, kernelId, cpuDevice, options, isJIT,
          shouldDeserialize, kind, linker, backend));
    }

    if (shouldRunExtraAsm) {
      // We need to run the llvm lowering again to saveTempsPrefix for
      // assembly if we are generating object. Since codegen has
      // side effect, we cannot reuse the same llvm module for assembly and
      // object file, we have to run the llvm lowering separately for each
      // codegen result.
      emissionResults.push_back(lowerLLVMModuleToObject(
          *module, loc, transformCache, kernelId, cpuDevice, options, isJIT,
          shouldDeserialize, EmitAs::ASM, linker, backend));
    }

    auto kernelBufs =
        AsyncRT::AsyncValueRef<DenseMap<EmitAs, BufferRef>>::allocate(
            cpuDevice);

    andThenSyncMoving(
        emissionResults,
        [emissionKinds, shouldRunExtraAsm, resultBufs = kernelBufs.copy()](
            MutableArrayRef<AnyAsyncValueRef> values) mutable {
          DenseMap<EmitAs, BufferRef> kernelResults;

          if (shouldRunExtraAsm) {
            // No need to process the last result as it's just for printing.
            values = values.drop_back(1);
          }

          for (auto [idx, result] : llvm::enumerate(values)) {
            if (result.isError())
              return std::move(resultBufs).setToError(result.takeDiagnostic());
            kernelResults.insert({emissionKinds[idx], result.get<BufferRef>()});
          }
          std::move(resultBufs).emplace(kernelResults);
        });
    await(kernelBufs);

    if (kernelBufs.isError())
      std::move(resultBufs).setToError(kernelBufs.takeDiagnostic());
    else
      std::move(resultBufs).emplace(kernelBufs.get());

    std::move(resultKernelId).emplace(kernelId);
  });

  return std::make_pair(std::move(resultBufs), std::move(resultKernelId));
}

// Emit offload kernels.
// The input module is a bundle of multiple offload kernels.
// Th output is a vector of compiled offload kernels with their corresponding
// kernel ids.
// This function does the following steps:
// - Split the input module into submodules for each kernel.
//   We don't do per function splitting for offload kernels since
//   some backends are inter-procedural.
// - Extract kernel ID for each split.
// - Run LLVM pipeline (opt + asmprint) to generate target-specific code for
//   each kernel.
ErrorOr<DenseMap<uint64_t, DenseMap<EmitAs, BufferRef>>>
ObjectCompiler::emitOffloadKernels(
    OwningOpRef<ModuleOp> module,
    llvm::DenseMap<uint64_t, llvm::SmallSet<EmitAs, 4>> kernelEmissionKinds) {
  LLVMTimingRegion timingRegion(options);
  CompilerTimeTraceScope traceScope("emitOffloadKernels");

  // Perform a cache aware transformation to translate the module to an
  // archive file.

  // Lower the module to LLVM.
  LLVMModuleAndContext llvmModule;
  Location moduleLoc = module->getLoc();

  // Save elaborated MLIR module to saveTempsPrefix.
  if (!options.saveTempsPrefix.empty()) {
    std::string str;
    llvm::raw_string_ostream ss(str);
    ss << *module;
    if (failed(writeBytesToTempWithHash(options.saveTempsPrefix, ".mlir", str)))
      return Error("failed to save mlir to saveTempPrefix");
  }

  if (auto err = llvmModule.create([&](llvm::LLVMContext &ctx) {
        return lowerAllFuncsToLLVM(ctx, *module);
      })) {
    return Error(
        Twine("failed to lower module to LLVM IR for archive compilation, ") +
        err.getError());
  }

  (void)writeTempModule(options.saveTempsPrefix, ".pre-split", *llvmModule);

  SmallVector<AsyncRT::AnyAsyncValueRef> cachedResults;
  auto handleSplit =
      [&](llvm::unique_function<LLVMModuleAndContext()> produceModule,
          std::optional<int64_t> idx, unsigned numFunctionsBase) {
        auto result = lowerLLVMModuleToObject(
            std::move(produceModule), moduleLoc, transformCache, idx, cpuDevice,
            options, isJIT, kernelEmissionKinds, linker, bitcodeLibs, backend);
        cachedResults.push_back(std::move(result.first));
        cachedResults.push_back(std::move(result.second));
      };

  splitPerExported(std::move(llvmModule), handleSplit);

  auto result = AsyncRT::AsyncValueRef<
      DenseMap<uint64_t, DenseMap<EmitAs, BufferRef>>>::allocate(cpuDevice);

  andThenSyncMoving(
      cachedResults, [result = result.copy(), &backend = backend](
                         MutableArrayRef<AnyAsyncValueRef> values) mutable {
        DenseMap<uint64_t, DenseMap<EmitAs, BufferRef>> results;

        for (size_t i = 0; i < values.size(); i += 2) {
          AnyAsyncValueRef &bufs = values[i];
          AnyAsyncValueRef &kernelId = values[i + 1];

          if (kernelId.isError()) {
            if (backend.isOffload())
              return std::move(result).setToError(kernelId.takeDiagnostic());
            else
              continue;
          }

          if (bufs.isError())
            return std::move(result).setToError(bufs.takeDiagnostic());

          results.insert({kernelId.get<uint64_t>(),
                          std::move(bufs.get<DenseMap<EmitAs, BufferRef>>())});
        }
        std::move(result).emplace(std::move(results));
      });

  await(result);

  if (result.isError())
    return {std::move(result.takeDiagnostic().getMessage())};

  return std::move(result.get());
}
