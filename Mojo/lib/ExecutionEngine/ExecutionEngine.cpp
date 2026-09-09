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

#include "Mojo/ExecutionEngine/ExecutionEngine.h"
#include "Mojo/ExecutionEngine/JIT/StaticArchiveLayer.h"
#include "Mojo/Support/Configuration.h"
#include "Support/ErrorOr.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/COFFPlatform.h"
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/ExecutionEngine/Orc/Debugging/DebugInfoSupport.h"
#include "llvm/ExecutionEngine/Orc/Debugging/DebuggerSupportPlugin.h"
#include "llvm/ExecutionEngine/Orc/Debugging/ELFDebugObjectPlugin.h"
#include "llvm/ExecutionEngine/Orc/Debugging/PerfSupportPlugin.h"
#include "llvm/ExecutionEngine/Orc/EPCDynamicLibrarySearchGenerator.h"
#include "llvm/ExecutionEngine/Orc/MapperJITLinkMemoryManager.h"
#include "llvm/ExecutionEngine/Orc/ObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderPerf.h"
#include "llvm/IR/Mangler.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Process.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"

using namespace M;
using namespace KGEN;

/// A standard name (that a user is unlikely to create) that we can use for a
/// JITDylib to define platform-specific symbols we want to be in the JIT'ed
/// address space.
static constexpr StringLiteral platformStdlibName = "$platform-stdlib";
static constexpr StringLiteral compilerRTlibName = "$compilerrt-lib";
static constexpr StringLiteral mlirclibName = "$mlirc-lib";

//===----------------------------------------------------------------------===//
// ExecutionEngine implementation
//===----------------------------------------------------------------------===//

/// Set up the ORC platform for the various different binary formats/platforms
/// we support. This requires that we have an ExecutionSession *and* an
/// ObjectLinkingLayer.
///
/// The main reason to use the platform like this is that it automatically sets
/// up the various symbols that complex code will need to execute on a target.
static ErrorOrSuccess setupPlatform(llvm::orc::JITDylib &platformStdlib,
                                    llvm::orc::ExecutionSession &session,
                                    llvm::orc::DylibManager &dylibMgr) {
  // Add the current process symbols in.
  // NOTE: COFF JIT currently doesn't support in process symbols, as it can
  // currently hit conflicts with symbols in the current COFF ORC runtime.
  auto generator = toModularErrorOr(
      llvm::orc::EPCDynamicLibrarySearchGenerator::GetForTargetProcess(
          session, dylibMgr));
  if (generator.isError())
    return generator.takeError();
  platformStdlib.addGenerator(std::move(*generator));

  return success();
}

/// Initialize the mlirc and CompilerRT dylib.
static ErrorOrSuccess
initializeCompilerRT(llvm::orc::ExecutionSession &session,
                     llvm::orc::DylibManager &dylibMgr, MojoConfig &cfg,
                     const llvm::DataLayout &layout,
                     const ExecutionEngineOptions &options) {
  std::error_code ec;

  // mlirc dylib. Grab the symbols from the current process.
  {
    auto *libJD = &session.createBareJITDylib(mlirclibName.str());
    libJD->addGenerator(llvm::cantFail(
        llvm::orc::EPCDynamicLibrarySearchGenerator::GetForTargetProcess(
            session, dylibMgr,
            [=](const llvm::orc::SymbolStringPtr &symbolStringPtr) {
              StringRef name = *symbolStringPtr;
              // On MachO, the symbol names start with `_`.
              return name.starts_with("mlir") || name.starts_with("_mlir");
            })));
  }

  // CompilerRT dylib.
  std::string compilerRTPath = cfg.getCompilerRTPath().str();
  if (!std::filesystem::exists(compilerRTPath, ec) || ec)
    return Error(std::string("unable to locate compiler_rt ") + compilerRTPath);

  auto *libJD = &session.createBareJITDylib(compilerRTlibName.str());

  SmallVector<StringRef> paths = options.libraryPaths;
  paths.push_back(compilerRTPath);

  for (StringRef libPath : paths) {
    auto generatorOr =
        toModularErrorOr(llvm::orc::EPCDynamicLibrarySearchGenerator::Load(
            session, dylibMgr, libPath.str().c_str()));
    if (generatorOr.isError()) {
      return Error(Twine("error '") + Twine(generatorOr.getError()) +
                   "' while loading compiler runtime library from '" +
                   libPath.str().c_str() + "'");
    }
    libJD->addGenerator(std::move(*generatorOr));
  }

  // Allow pulling in sanitizer methods from the current process, as we
  // currently can't activate any of these runtimes otherwise (they must
  // generally be loaded first in the host process).
  libJD->addGenerator(llvm::cantFail(
      llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          layout.getGlobalPrefix(),
          [=](const llvm::orc::SymbolStringPtr &symbolStringPtr) {
            return llvm::any_of(ArrayRef<StringRef>{"__asan", "__tsan"},
                                [&](StringRef prefix) {
                                  return (*symbolStringPtr).starts_with(prefix);
                                });
          })));

  return success();
}

M::ErrorOr<std::unique_ptr<ExecutionEngine>>
ExecutionEngine::create(ExecutionEngineOptions options,
                        const llvm::TargetMachine &tm) {
  // Create the data layout from the target machine.
  const llvm::DataLayout &layout = tm.createDataLayout();
  const llvm::Triple &tt = tm.getTargetTriple();

  // Construct the ExecutionSession. The user may have passed in an
  // ExecutorProcessControl that we need to use.
  std::unique_ptr<llvm::orc::ExecutorProcessControl> epc =
      std::move(options.epc);

  // The JIT-link memory manager is owned by the ObjectLinkingLayer so build it
  // up front.
  //
  // JIT mapper reservation granularity. Every fresh reservation by LLVM's
  // MapperJITLinkMemoryManager is rounded up to this size and mmapped
  // PROT_READ|PROT_WRITE, so it sets a floor on the JIT region's virtual
  // footprint — which counts against RLIMIT_AS on Linux. Large compiles
  // still work: the mapper reserves additional slabs on demand.
  //
  // Do not increase this unless you know what you are doing. In the past,
  // setting this to 1 GiB caused sporadic OoM crashes on memory-constrained
  // runners.
  size_t slabSize = size_t{64} * 1024 * 1024;
  auto managerOr =
      toModularErrorOr(llvm::orc::MapperJITLinkMemoryManager::CreateWithMapper<
                       llvm::orc::InProcessMemoryMapper>(slabSize));
  if (managerOr.isError())
    return managerOr.takeError();

  if (!epc) {
    auto pageSize = toModularErrorOr(llvm::sys::Process::getPageSize());
    if (pageSize.isError())
      return pageSize.takeError();

    epc = std::make_unique<llvm::orc::SelfExecutorProcessControl>(
        std::make_shared<llvm::orc::SymbolStringPool>(),
        std::make_unique<llvm::orc::InPlaceTaskDispatcher>(), tt, *pageSize);
  }
  auto sessionPtr =
      std::make_unique<llvm::orc::ExecutionSession>(std::move(epc));

  // Create a default DylibManager from the ExecutorProcessControl.
  auto dylibMgrOr = toModularErrorOr(
      sessionPtr->getExecutorProcessControl().createDefaultDylibMgr());
  if (dylibMgrOr.isError())
    return dylibMgrOr.takeError();

  // Now we can actually create the ExecutionEngine.
  auto ee = std::unique_ptr<ExecutionEngine>(
      new ExecutionEngine(std::move(sessionPtr), layout));
  ee->dylibMgr = std::move(*dylibMgrOr);

  // Open the config object so we can use it.
  auto cfgOr = MojoConfig::open();
  if (cfgOr.isError())
    return cfgOr.takeError();
  MojoConfig cfg = std::move(*cfgOr);

  // Construct the object linking layer; it takes ownership of the manager.
  ee->objectLayer = std::make_unique<llvm::orc::ObjectLinkingLayer>(
      *ee->executionSession, std::move(*managerOr));

  // Construct the platform stdlib - this way we don't have to worry about
  // whether or not we have it later on.
  llvm::orc::JITDylib &platformStdlib =
      ee->executionSession->createBareJITDylib(platformStdlibName.str());

  // If we have the platform support library, use it. This requires the
  // compilation target to be a subset of the host process, so disable it for
  // cross-compilation.
  if (!options.crossCompiling) {
    if (auto err =
            setupPlatform(platformStdlib, *ee->executionSession, *ee->dylibMgr))
      return err.takeError();
  }

  if (options.registerDebugPlugins) {
    llvm::orc::ExecutionSession &session = *ee->executionSession;

    // Get the registrar for the GDB JIT loader interface.
    if (tt.isOSBinFormatMachO()) {
      // Create and register the JIT DebugInfo plugin. The GDB alloc-action
      // symbol is resolved from the session's bootstrap JITDylib, which the
      // in-process executor populates automatically.
      auto plugin =
          toModularErrorOr(llvm::orc::GDBJITDebugInfoRegistrationPlugin::Create(
              session, session.getBootstrapJITDylib()));
      if (plugin.isError())
        return plugin.takeError();

      ee->objectLayer->addPlugin(std::move(*plugin));
    } else if (tt.isOSBinFormatELF()) {
      // Register the ELFDebugObjectPlugin.
      llvm::Error error = llvm::Error::success();
      auto plugin = std::make_unique<llvm::orc::ELFDebugObjectPlugin>(
          session, /*RequireDebugSections=*/true, error);
      if (auto errOr = toModularErrorOr(std::move(error)); failed(errOr))
        return errOr.takeError();
      ee->objectLayer->addPlugin(std::move(plugin));
    }
  }

  if (options.registerPerfPlugins) {
    auto debugInfo = llvm::orc::DebugInfoPreservationPlugin::Create();
    if (!debugInfo)
      return toModularError(debugInfo.takeError());
    ee->objectLayer->addPlugin(std::move(debugInfo.get()));
    auto perf = std::make_unique<llvm::orc::PerfSupportPlugin>(
        ee->objectLayer->getExecutionSession().getExecutorProcessControl(),
        llvm::orc::ExecutorAddr::fromPtr(&llvm_orc_registerJITLoaderPerfStart),
        llvm::orc::ExecutorAddr::fromPtr(&llvm_orc_registerJITLoaderPerfEnd),
        llvm::orc::ExecutorAddr::fromPtr(&llvm_orc_registerJITLoaderPerfImpl),
        true, true);
    ee->objectLayer->addPlugin(std::move(perf));
  }

  // Add the platform dylib to the search order.
  if (auto err = ee->addToSearchOrder(platformStdlibName, &platformStdlib))
    return err.takeError();

  // Prepare the CompilerRT dylib.
  if (auto err = initializeCompilerRT(*ee->executionSession, *ee->dylibMgr, cfg,
                                      layout, options))
    return err.takeError();

  return std::move(ee);
}

ErrorOr<std::unique_ptr<ExecutionEngine>>
ExecutionEngine::createWithStandardLayers(ExecutionEngineOptions options,
                                          const llvm::TargetMachine &tm) {
  auto engineOr = ExecutionEngine::create(std::move(options), tm);
  if (engineOr.isError())
    return engineOr.takeError();

  // Add the standard layers.
  (*engineOr)->addLayer<StaticArchiveLayer>((*engineOr)->getLinkingLayer());

  return std::move(*engineOr);
}

ExecutionEngine::ExecutionEngine(
    std::unique_ptr<llvm::orc::ExecutionSession> session,
    const llvm::DataLayout &dl)
    : executionSession(std::move(session)),
      // Parse the layout so that we own the underlying memory. DataLayout is a
      // bit weird, it seems like it has some internal data structures that
      // every instance shares.
      dataLayout(dl.getStringRepresentation()) {}

ExecutionEngine::~ExecutionEngine() {
  if (!executionSession)
    return;

  // If the execution engine has initialized the ORC runtime, the ELFNix and
  // COFF platform implementations need manual shutdown. The MachOPlatform
  // implementation is more sophisticated and performs shutdown automatically
  // through the JITLink LinkGraph allocation actions.
  const llvm::Triple &triple = executionSession->getTargetTriple();
  if (executionSession->getPlatform() &&
      // FIXME: On Windows, this complains about symbol not found. The Windows
      // build seems happy even without the shutdown, so disable it for now.
      (triple.isOSBinFormatELF() /*|| triple.isOSBinFormatCOFF()*/)) {
    ErrorOr<CompiledFunc> shutdown =
        lookup(triple.isOSBinFormatELF() ? "__orc_rt_elfnix_platform_shutdown"
                                         : "__orc_rt_coff_platform_shutdown");
    if (shutdown.isError()) {
      llvm::report_fatal_error(
          Twine("failed to find ELF/COFF platform shutdown function: ") +
          shutdown.takeError().get());
    }
    struct OrcRTCWrapperFunctionResult {
      char *data;
      size_t size;
    };
    shutdown->invoke<OrcRTCWrapperFunctionResult, char *, size_t>(nullptr, 0);
  }

  if (auto err = executionSession->endSession())
    executionSession->reportError(std::move(err));
}

ErrorOr<CompiledFunc> ExecutionEngine::lookup(StringRef symbol) {
  return lookupWithSearchOrder(searchOrder, symbol);
}

ErrorOr<CompiledFunc> ExecutionEngine::lookup(StringRef libName,
                                              StringRef symbol) {
  llvm::orc::JITDylib *dylib = executionSession->getJITDylibByName(libName);
  if (!dylib)
    return Error("could not find JITDylib with name: " + libName);

  return lookupWithSearchOrder(llvm::orc::makeJITDylibSearchOrder({dylib}),
                               symbol);
}

ErrorOrSuccess
ExecutionEngine::runProgram(StringRef libName, StringRef entryPoint,
                            function_ref<ErrorOrSuccess(void *)> runFn) {
  using namespace llvm::orc;
  // There is not global ctor/dtor in mojo.

  // Lookup the entry point symbol and directly invoke it rather than going
  // through the runtime.
  ErrorOr<CompiledFunc> mainFn = lookup(entryPoint);
  if (mainFn.isError())
    return mainFn.takeError();
  if (ErrorOrSuccess err = runFn(mainFn->getFunctionPointer()))
    return err.takeError();
  return success();
}

llvm::orc::SymbolStringPtr
KGEN::ExecutionEngine::mangleAndIntern(StringRef name) {
  std::string mangledName;
  llvm::raw_string_ostream mangledNameStream(mangledName);
  llvm::Mangler::getNameWithPrefix(mangledNameStream, name, dataLayout);
  return executionSession->intern(mangledName);
}

ErrorOrSuccess ExecutionEngine::addToSearchOrder(StringRef name,
                                                 llvm::orc::JITDylib *dylib) {
  [[maybe_unused]] auto [_, didInsert] = knownDylibs.insert(name);
  assert(didInsert && "must have uniquely-named dylibs");

  // If this isn't the platform stdlib, setup CompilerRT and mlirc.
  if (name != platformStdlibName) {
    dylib->addToLinkOrder(
        *executionSession->getJITDylibByName(compilerRTlibName));
    dylib->addToLinkOrder(*executionSession->getJITDylibByName(mlirclibName));
  }

  // Use higher preference for newer dylibs.
  searchOrder.insert(searchOrder.begin(),
                     {dylib, llvm::orc::JITDylibLookupFlags::MatchAllSymbols});
  return success();
}

ErrorOr<CompiledFunc> ExecutionEngine::lookupWithSearchOrder(
    const llvm::orc::JITDylibSearchOrder &order, llvm::StringRef symbol) {
  // Look up this symbol with the search order provided.
  llvm::Expected<llvm::orc::ExecutorSymbolDef> sym =
      executionSession->lookup(order, mangleAndIntern(symbol));
  if (sym)
    return CompiledFunc(sym->getAddress().toPtr<void *>());

  // Check to see if any of the layers have errors.
  auto found = llvm::find_if(
      layers, [](const auto &layer) { return layer->hasError(); });
  // If not, return the error returned by the ORC.
  if (found == layers.end())
    return toModularError(sym.takeError());

  // Add the additional context from the layer's error.
  return Error(llvm::toString(sym.takeError()) +
               " (from the layer: " + (*found)->takeError().get() + ")");
}
