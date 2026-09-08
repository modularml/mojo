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
// The kgen-opt driver implementation.
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/CompilerSupport/Context.h"
#include "AsyncRT/Runtime/CPUDevice.h"
#include "Init/Init.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Mojo/Support/ForceLinkMLIRC.h"
#include "Mojo/Support/MojoPrecompiledFile.h"
#include "Mojo/ToolCommon/CLOptions.h"
#include "Mojo/ToolCommon/Debug.h"
#include "Mojo/ToolCommon/InitAllDialects.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Support/Context.h"
#include "Support/DebugInfoDialect/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace M;

//===----------------------------------------------------------------------===//
// TestAlwaysFailPass
//===----------------------------------------------------------------------===//

namespace {
/// This is a pass that always fails for the purpose of debugging reproducers.
struct TestAlwaysFailPass
    : public mlir::PassWrapper<TestAlwaysFailPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestAlwaysFailPass)

  StringRef getArgument() const override { return "test-always-fail"; }

  void runOnOperation() override { return signalPassFailure(); }
};

LogicalResult runMlirOptMain(int argc, char **argv,
                             llvm::StringRef inputFilename,
                             llvm::StringRef outputFilename,
                             DialectRegistry &registry,
                             bool ignoreIncompatiblePrecompiledFileErrs) {
  llvm::InitLLVM y(argc, argv);
  mlir::MlirOptMainConfig config =
      mlir::MlirOptMainConfig::createFromCLOptions();

  // When reading from stdin and the input is a tty, it is often a user
  // mistake and the process "appears to be stuck". Print a message to let the
  // user know about it!
  if (inputFilename == "-" &&
      llvm::sys::Process::FileDescriptorIsDisplayed(fileno(stdin)))
    llvm::errs() << "(processing input from stdin now, hit ctrl-c/ctrl-d to "
                    "interrupt)\n";

  // Set up the input file.
  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  // If this is a Mojo precompiled file, verify the header and skip past it to
  // get to the MLIR within.
  llvm::MemoryBufferRef mlirBuffer = *file;
  std::unique_ptr<llvm::MemoryBuffer> decompressedPkgData;
  if (KGEN::isMojoPrecompiledFile(*file)) {
    auto mlirBufOrErr = M::KGEN::getMLIRBufferFromPrecompiledFile(
        *file, ignoreIncompatiblePrecompiledFileErrs);
    if (mlirBufOrErr.isError()) {
      llvm::errs() << mlirBufOrErr.takeError().get() << "\n";
      return failure();
    }
    mlirBuffer = mlirBufOrErr->buffer;
    decompressedPkgData = std::move(mlirBufOrErr->ownedData);
  }

  auto mlirBuff =
      llvm::MemoryBuffer::getMemBuffer(mlirBuffer,
                                       /*RequiresNullTerminator=*/true);

  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  if (failed(MlirOptMain(output->os(), std::move(mlirBuff), registry, config)))
    return failure();

  // Keep the output file if the invocation of MlirOptMain was successful.
  output->keep();
  return success();
}
} // namespace

int main(int argc, char **argv) {
  // Force linking of MLIR C symbols to JIT Mojo code relying on the mlir
  // bindings.
  KGEN::forceLinkMLIRC();

  // HACK: Read in the option early.
  bool asyncrtSingleThread = false;
  if (argc >= 2 && StringRef(argv[1]) == "--asyncrt-single-thread")
    asyncrtSingleThread = true;

  DialectRegistry registry;

  // Register all KGEN dialects.
  registerAllKGENDialects(registry);

  // Initialize all targets.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  // Initialize the host target.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmParser();
  llvm::InitializeNativeTargetAsmPrinter();

  // Initialize LLVM exporters.
  registerKGENToLLVMTranslation(registry);

  // Register test passes.
  mlir::PassRegistration<TestAlwaysFailPass>{};

  // Create our context.
  AsyncRT::CPUDeviceOptions asyncrtOpts;
  asyncrtOpts.withLeakCheckedAllocator();
  if (asyncrtSingleThread)
    asyncrtOpts.withSingleThreaded();
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "kgen-opt", Init::Options().withCPUDeviceOptions(asyncrtOpts));
  if (ctxOr.isError()) {
    llvm::errs() << "failed to create context: " << ctxOr.getError() << "\n";
    return 1;
  }
  if (asyncrtSingleThread) {
    // Defend against upstream errors.
    [[maybe_unused]] auto &cpuDevice = *(*ctxOr)->get<AsyncRT::CPUDevice>();
    assert(cpuDevice.getWorkQueue()->getParallelismLevel() == 1);
  }
  registerContext(registry, *ctxOr);

  // Register passes.
  KGEN::registerDefaultKGENPasses("kgen-opt");
  DebugInfo::registerTransformsPasses();

  // Register cl options.
  static llvm::cl::opt<bool> dummyOpt{"asyncrt-single-thread"};

  static llvm::cl::opt<bool> timeTrace{
      "time-trace",
      llvm::cl::desc("Turn on time profiler. Generates JSON file "
                     "called kgen.trace.json in the derived directory.")};

  static llvm::cl::opt<int> timeTraceGranularity{
      "time-trace-granularity",
      llvm::cl::desc("Minimum time granularity (in microseconds) "
                     "traced by time profiler."),
      llvm::cl::init(0)};

  static llvm::cl::opt<bool> ignoreIncompatiblePrecompiledFileErrs{
      "ignore-incompatible-precompiled-file-errors",
      llvm::cl::desc("Ignore errors encountered when loading incompatible Mojo "
                     "precompiled files."),
      llvm::cl::init(false)};

  KGEN::registerKGENCommandLineOptions();
  KGEN::initializeDebugOptions();
  KGEN::KGENPassCLOptions::registerOptions();

  // Register and parse command line options.
  std::string inputFilename, outputFilename;
  std::tie(inputFilename, outputFilename) =
      registerAndParseCLIOptions(argc, argv, "kgen optimizer driver", registry);
  if (KGEN::debugFlag)
    llvm::errs() << "WARNING: `kgen-debug-only` may work incorrectly with "
                    "multithreading enabled\n";

  KGEN::TraceProfiler tracer(timeTrace, timeTraceGranularity);

  return failed(runMlirOptMain(argc, argv, inputFilename, outputFilename,
                               registry,
                               ignoreIncompatiblePrecompiledFileErrs));
}
