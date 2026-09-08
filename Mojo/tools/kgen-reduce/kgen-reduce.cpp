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
#include "Helpers.h"
#include "Init/Init.h"
#include "Mojo/HLCFDialect/HLCFInterfaces.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/ToolCommon/CLOptions.h"
#include "Mojo/ToolCommon/InitAllDialects.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Mojo/TransformUtils/Walkers.h"
#include "PreOrderRegionIterator.h"
#include "Support/Context.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/ToolOutputFile.h"

#include <chrono>
#include <queue>

using namespace M;
using namespace KGEN;

namespace {
/// This struct tracks the current IR state of the reducer.
struct IRState {
  IRState(OwningOpRef<ModuleOp> ir) : ir(std::move(ir)) {}

  OwningOpRef<ModuleOp> ir;
};

/// This is the reducer class that keeps track of all state during reduction.
struct Reducer {
  llvm::cl::opt<std::string> inputFilename{llvm::cl::Positional,
                                           llvm::cl::desc("<input file>"),
                                           llvm::cl::init("-")};

  mlir::PassPipelineCLParser passPipeline{"", "pass pipeline to run"};
  mlir::PassReproducerOptions reproOptions;

  cl::opt<unsigned> numSnapshots{"num-snapshots",
                                 llvm::cl::desc("number of snapshots to keep"),
                                 llvm::cl::init(10)};

  cl::opt<unsigned> snapshotDelta{
      "snapshot-delta",
      llvm::cl::desc("delta between snapshots in milliseconds"),
      llvm::cl::init(2000)};

  cl::opt<unsigned> startAt{"start-at",
                            llvm::cl::desc("the reducer phase to start at"),
                            llvm::cl::init(0)};

  Reducer(MLIRContext *ctx) : ctx(ctx), reproPm(ctx), dcePm(ctx) {
    dcePm.addPass(createEliminateDeadSymbols());
  }

  ErrorOrSuccess run();
  ErrorOrSuccess reduceFunctionsBinary(IRState &curState);
  ErrorOrSuccess reduceFunctions(IRState &curState);
  ErrorOrSuccess reduceRegions(IRState &curState);
  ErrorOrSuccess reduceOps(IRState &curState);
  ErrorOrSuccess reduceDCE(IRState &curState);
  ErrorOr<bool> tryDCE(IRState &curState);

  /// Run the pass pipeline on a clone of the module and return the diagnostics
  /// if any were emitted.
  std::optional<std::string> attemptRepro(ModuleOp ir);
  /// Return true if the supplied module repros the error.
  bool doesRepro(ModuleOp ir);

  /// Maybe save a snapshot of the current module if enough time has passed
  /// since the last.
  ErrorOrSuccess maybeSnapshot(ModuleOp ir);

  /// The MLIR context.
  MLIRContext *ctx;

  /// The pass manager containing to run to generate the desired error.
  mlir::PassManager reproPm;
  /// The pass manager used to run symbol DCE.
  mlir::PassManager dcePm;
  /// The reproducer pipeline.
  std::string pipelineStr;

  /// The expected error to reproduce.
  std::string expectedDiag;

  /// The set of module snapshots to keep.
  std::queue<std::unique_ptr<llvm::ToolOutputFile>> snapshots;

  /// Time since the last snapshot.
  uint64_t lastSnapshotTime;

  /// Logging output.
  llvm::raw_ostream &log = llvm::outs();
};
} // namespace

ErrorOrSuccess Reducer::run() {
  mlir::ParserConfig parserConfig(ctx);
  if (!passPipeline.hasAnyOccurrences())
    reproOptions.attachResourceParser(parserConfig);

  OwningOpRef<ModuleOp> inputModule =
      mlir::parseSourceFile<ModuleOp>(inputFilename.getValue(), parserConfig);
  if (!inputModule)
    return Error("failed to parse input file: " + inputFilename.getValue());

  log << "[[===============================================================]]\n"
      << "[[======================== KGEN ⚜️ REDUCER =======================]]\n"
      << "[[===============================================================]]"
         "\n\n";

  log << "[kgen-reduce] " << inputFilename.getValue() << "\n";

  // Parse the pass pipeline.
  {
    std::string err;
    llvm::raw_string_ostream os(err);
    if (passPipeline.hasAnyOccurrences()) {
      if (failed(passPipeline.addToPipeline(reproPm, [&](const Twine &msg) {
            os << msg;
            return failure();
          })))
        return Error(err);
    } else if (failed(reproOptions.apply(reproPm))) {
      return Error("failed to read pass reproducer");
    }
  }

  llvm::raw_string_ostream os(pipelineStr);
  reproPm.printAsTextualPipeline(os);
  log << "[kgen-reduce] " << pipelineStr << "\n";

  std::optional<std::string> initDiag = attemptRepro(*inputModule);
  if (!initDiag)
    return Error("original input IR does not fail the provided pipeline");
  expectedDiag = std::move(*initDiag);

  log << "[kgen-reduce] expected diagnostic:\n" << expectedDiag << "\n\n";

  IRState curModule(std::move(inputModule));
  if (auto err = stashFile(*curModule.ir, "kgen-reduce.base", pipelineStr))
    return err;

  if (startAt <= 0) {
    if (auto err = reduceFunctionsBinary(curModule))
      return err;
    if (auto err = tryDCE(curModule))
      return err.takeError();
    if (auto err = stashFile(*curModule.ir, "kgen-reduce.functions.binary",
                             pipelineStr))
      return err;
  }

  if (startAt <= 1) {
    if (auto err = reduceFunctions(curModule))
      return err;
    if (auto err = tryDCE(curModule))
      return err.takeError();
    if (auto err =
            stashFile(*curModule.ir, "kgen-reduce.functions", pipelineStr))
      return err;
  }

  if (startAt <= 2) {
    if (auto err = reduceRegions(curModule))
      return err;
    if (auto err = tryDCE(curModule))
      return err.takeError();
    if (auto err = stashFile(*curModule.ir, "kgen-reduce.regions", pipelineStr))
      return err;
  }

  if (startAt <= 3) {
    if (auto err = reduceOps(curModule))
      return err;
    if (auto err = tryDCE(curModule))
      return err.takeError();
    if (auto err = stashFile(*curModule.ir, "kgen-reduce.ops", pipelineStr))
      return err;
  }

  if (startAt <= 4) {
    if (auto err = reduceDCE(curModule))
      return err;
    if (auto err = stashFile(*curModule.ir, "kgen-reduce.ops.dce", pipelineStr))
      return err;
  }

  return success();
}

std::optional<std::string> Reducer::attemptRepro(ModuleOp ir) {
  std::string diag;
  llvm::raw_string_ostream os(diag);
  mlir::ScopedDiagnosticHandler handler(
      ctx, [&](Diagnostic &diag) { diag.print(os); });
  OwningOpRef<ModuleOp> tmpModule = ir.clone();
  if (succeeded(reproPm.run(*tmpModule)))
    return {};
  return std::move(diag);
}

bool Reducer::doesRepro(ModuleOp ir) {
  std::optional<std::string> nextDiag = attemptRepro(ir);
  if (nextDiag && nextDiag == expectedDiag) {
    log << "[kgen-reduce] same failure 🛑\n\n";
    return true;
  }

  if (!nextDiag) {
    log << "[kgen-reduce] succeeded 🟢\n\n";
  } else {
    log << "[kgen-reduce] different failure 🟠:\n" << *nextDiag << "\n\n";
  }
  return false;
}

ErrorOrSuccess Reducer::maybeSnapshot(ModuleOp module) {
  uint64_t curTime = getCurTimeMs();
  if (curTime - lastSnapshotTime < snapshotDelta.getValue())
    return success();
  lastSnapshotTime = curTime;

  auto fileOr = getTempFile(module, getTempFileName(), pipelineStr);
  if (fileOr.isError())
    return fileOr.takeError();
  auto file = fileOr.takeValue();

  log << "[kgen-reduce] snapshotting IR to " << file->getFilename() << "\n";

  file->keep();
  llvm::sys::DontRemoveFileOnSignal(file->getFilename());
  snapshots.push(std::move(file));

  // Pop the oldest file off and unkeep it.
  if (snapshots.size() > numSnapshots.getValue()) {
    auto drop = std::move(snapshots.front());
    snapshots.pop();
    unkeepToolOutputFile(*drop);
  }

  return success();
}

/// This reduction function uses a bisecting search to determine the set of
/// functions in the module that can be stubbed while still reproducing the
/// original error. The algorithm is as follows:
///
/// We initialize a vector with all active (unstubbed) functions. The goal is to
/// find the minimum set of functions that need to be active for the error to
/// reproduce. We can stub and unstub subranges of functions in the vector at a
/// time and then test whether the error reproduces.
///
/// Given a subrange [low, high] of the vector where all functions are active
/// and the error is known to current reproduce, we find the lower bound [low,
/// ub) of functions that can be stubbed. This is done by iteratively stubbing
/// and unstubbing functions half of the subrange at a time.
///
/// Then we do the same to find the upper bound [lb, high). We know that the
/// functions at `ub` and `lb` (if they are within the original subrange) are
/// required to reproduce the error. We repeat this algorithm on the new
/// subrange (ub, lb).
///
/// Computing the lower bound is O(N + C*log(N)^2), where C is big (cost to
/// clone the module and run the pass pipeline times average size of a
/// function). Thus, the overall algorithm is approximately
/// O( (N + C*log(N)^2) * M ) where M is the number of functions in the minimum
/// reproducer and N is the number of functions in the module. You can see that
/// if the vector is sparse (M << N), then this runs in linear time, but what
/// actually matters is stomping the C factor as much as possible, because
/// cloning and running the pass pipeline is really, really slow compared to
/// stubbing functions (just moving regions around).
///
/// This algorithm assumes that the minimum reproducer is the same regardless of
/// the order in which it is found, which is almost certainly not the case.
/// That's why we still run a linear search after this one. The linear search is
/// O( N^2 + C*N*log(N) ). The C component is O(N*log(N)) because the module
/// gets smaller as we stub out more functions. From above, the C component
/// actually dominates. The C component is likely not constant either due to
/// memory shenanigans (pointer chasing/caching, memory allocator, etc.).
ErrorOrSuccess Reducer::reduceFunctionsBinary(IRState &curState) {
  std::vector<KGEN::FuncOp> funcs;
  bool anyKgenFunc = false;
  for (auto func : curState.ir->getOps<KGEN::FuncOp>()) {
    anyKgenFunc = true;
    // Ignore functions that are already stubbed.
    if (isStubbed(func.getBodyRegion()))
      continue;
    funcs.push_back(func);
  }
  if (!anyKgenFunc) {
    return Error("zero 'kgen.func' operations found, 'kgen-reduce' only works "
                 "on post-elaboration IR right now");
  }

  log << "[kgen-reduce] attempt to binary stub\n";

  /// We'll be stubbing multiple functions at once. Keep track of them all in
  /// this dandy data structure with helpers for stubbing and unstubbing.
  struct StubbedFunction {
    KGEN::FuncOp func;
    Region owner;
    /// This flag is used for correctness checking the algorithm.
    bool isStubbed = false;

    void stub() {
      assert(!isStubbed && "already stubbed?");
      stubRegion(func.getBodyRegion(), owner);
      isStubbed = true;
    }
    void unstub() {
      assert(isStubbed && "not stubbed?");
      func.getBodyRegion().takeBody(owner);
      isStubbed = false;
    }
  };

  std::vector<StubbedFunction> span(funcs.size());
  for (auto [stub, func] : llvm::zip(span, funcs))
    stub.func = func;

  // This functor stubs a span of functions in [low, high].
  auto stubSpan = [&span](int64_t low, int64_t high) {
    for (int64_t i = low; i <= high; ++i)
      span[i].stub();
  };
  // This functor unstubs a span of functions in [low, high].
  auto unstubSpan = [&span](int64_t low, int64_t high) {
    for (int64_t i = low; i <= high; ++i)
      span[i].unstub();
  };

  // This function computes the lower bound as described above with a bisection.
  // We start with a subrange and bisect by stubbing the lower have of the
  // current subrange and then moving to the upper half while the error still
  // reproduces. If at any point it stops reproducing, we bisect and unstub in
  // the opposite direction until the subrange converges to a single function.
  // Because of the flip-flopping, we have to specially handle the single
  // function case to avoid an infinite loop or indeterminate case.
  auto lowerBoundFail = [this, &curState, &stubSpan,
                         &unstubSpan](int64_t low, int64_t high) {
    [[maybe_unused]] int64_t low0 = low;
    bool repros = true;
    while (low < high) {
      // Repros in the current span. Stub the first half of the span.
      if (repros) {
        // Stub [low, mid].
        int64_t mid = low + (high - low) / 2;
        log << "[stubbing span [" << low << ", " << mid << "]]\n";
        stubSpan(low, mid);
        if ((repros = doesRepro(*curState.ir)))
          low = mid + 1;
        else
          high = mid;
      } else {
        // Unstub [mid, high].
        int64_t mid = low + llvm::divideCeil(high - low, 2);
        log << "[unstubbing span [" << mid << ", " << high << "]]\n";
        unstubSpan(mid, high);
        if ((repros = doesRepro(*curState.ir)))
          low = mid;
        else
          high = mid - 1;
      }
    }
    assert(low == high && "invalid exit condition");
    if (repros) {
      log << "[point testing " << low << "]\n";
      stubSpan(low, low);
      if (doesRepro(*curState.ir)) {
        return low + 1;
      }
      unstubSpan(low, low);
      return low;
    }
    log << "[double check " << low << "]\n";
    unstubSpan(low, low);
    if (doesRepro(*curState.ir)) {
      return low;
    }
    // This must be the initial state, and because we know we're in the initial
    // state, we know it must reproduce so this is invalid.
    assert(low == low0);
    llvm_unreachable("everything got unstubbed but not original repro?");
  };

  // This functor does the same but to find the upper bound. The index
  // computations are moved around a bit.
  auto upperBoundFail = [this, &curState, &stubSpan,
                         &unstubSpan](int64_t low, int64_t high) -> int64_t {
    [[maybe_unused]] int64_t high0 = high;
    bool repros = true;
    while (low < high) {
      // Repros in the current span. Stub the first half of the span.
      if (repros) {
        int64_t mid = low + llvm::divideCeil(high - low, 2);
        log << "[stubbing span [" << mid << ", " << high << "]]\n";
        stubSpan(mid, high);
        if ((repros = doesRepro(*curState.ir)))
          high = mid - 1;
        else
          low = mid;
      } else {
        int64_t mid = low + (high - low) / 2;
        log << "[unstubbing span [" << low << ", " << mid << "]]\n";
        unstubSpan(low, mid);
        if ((repros = doesRepro(*curState.ir)))
          high = mid;
        else
          low = mid + 1;
      }
    }
    assert(low == high && "invalid exit condition");
    if (repros) {
      log << "[point testing " << low << "]\n";
      stubSpan(low, low);
      if (doesRepro(*curState.ir)) {
        return low - 1;
      }
      unstubSpan(low, low);
      return low;
    }
    log << "[double check " << low << "]\n";
    unstubSpan(low, low);
    if (doesRepro(*curState.ir)) {
      return low;
    }
    assert(low == high0);
    llvm_unreachable("everything got unstubbed but not original repro?");
  };

  int64_t low = 0;
  int64_t high = span.size() - 1;

  while (low < high) {
    if (auto err = maybeSnapshot(*curState.ir))
      return err.takeError();

    int64_t ub = lowerBoundFail(low, high);
    log << "[" << low << ", " << high << "]\n";
    log << "ub: " << ub << "\n";
    if (ub >= high) {
      assert((ub == high) || (ub == (high + 1)));
      return success();
    }
    low = ub + 1;

    if (auto err = maybeSnapshot(*curState.ir))
      return err.takeError();

    int64_t lb = upperBoundFail(low, high);
    log << "[" << low << ", " << high << "]\n";
    log << "lb: " << lb << "\n";
    if (lb <= low) {
      assert((lb == low) || (lb == (low - 1)));
      return success();
    }
    high = lb - 1;
  }

  return success();
}

ErrorOrSuccess Reducer::reduceFunctions(IRState &curState) {
  std::vector<KGEN::FuncOp> funcs;
  bool anyKgenFunc = false;
  for (auto func : curState.ir->getOps<KGEN::FuncOp>()) {
    anyKgenFunc = true;
    // Ignore functions that are already stubbed.
    if (isStubbed(func.getBodyRegion()))
      continue;
    funcs.push_back(func);
  }
  if (!anyKgenFunc) {
    return Error("zero 'kgen.func' operations found, 'kgen-reduce' only works "
                 "on post-elaboration IR right now");
  }

  size_t funcNum = 0;
  const size_t totalNumFuncs = funcs.size();
  log << "[kgen-reduce] attempt to stub " << totalNumFuncs << " functions\n";

  // TODO: This is a linear search, which gets faster as more functions are
  // stubbed but it would be even faster to do a bisect search.
  //
  // Starting with N functions, stub out N/2. If repro, continue. Else, bisect
  // only the first half and then repeat until repro. Then repeat on remaining
  // functions.
  while (!funcs.empty()) {
    if (auto err = maybeSnapshot(*curState.ir))
      return err.takeError();

    KGEN::FuncOp func = funcs.back();
    funcs.pop_back();

    log << "[stubbing function " << (funcNum++) << "/" << totalNumFuncs << "] "
        << func.getSymName() << "\n";

    // Stub the function with an unreachable.
    Region owner;
    stubRegion(func.getBodyRegion(), owner);

    if (doesRepro(*curState.ir))
      continue;

    // Revert the transformation.
    func.getBodyRegion().takeBody(owner);
  }

  return success();
}

ErrorOrSuccess Reducer::reduceRegions(IRState &curState) {
  std::vector<KGEN::FuncOp> funcs;
  for (auto func : curState.ir->getOps<KGEN::FuncOp>()) {
    // Ignore stubbed functions.
    if (isStubbed(func.getBodyRegion()))
      continue;
    funcs.push_back(func);
  }

  size_t funcNum = 0;
  const size_t totalNumFuncs = funcs.size();
  log << "[kgen-reduce] reducing regions in " << totalNumFuncs
      << " functions\n";

  while (!funcs.empty()) {
    KGEN::FuncOp func = funcs.back();
    funcs.pop_back();

    log << "[reducing regions " << (funcNum++) << "/" << totalNumFuncs << "] "
        << func.getSymName();

    size_t regionNum = 0;
    auto it = PreOrderRegionIterator::begin(func);
    auto end = PreOrderRegionIterator::end(func);
    for (; it != end; ++it) {
      Region &region = *it;

      // Skip the region if we don't understand its semantics.
      if (!isa<HLCF::ControlFlowNode>(region.getParentOp()))
        continue;

      if (auto err = maybeSnapshot(*curState.ir))
        return err;

      log << "[region #" << regionNum++ << "]\n";

      Region owner;
      stubRegion(region, owner);

      if (doesRepro(*curState.ir))
        continue;
      region.takeBody(owner);
    }
  }

  return success();
}

ErrorOrSuccess Reducer::reduceOps(IRState &curState) {
  std::vector<KGEN::FuncOp> funcs;
  for (auto func : curState.ir->getOps<KGEN::FuncOp>()) {
    // Ignore stubbed functions.
    if (isStubbed(func.getBodyRegion()))
      continue;
    funcs.push_back(func);
  }

  size_t funcNum = 0;
  const size_t totalNumFuncs = funcs.size();
  log << "[kgen-reduce] reducing operations in " << totalNumFuncs
      << " functions\n";

  while (!funcs.empty()) {
    KGEN::FuncOp func = funcs.back();
    funcs.pop_back();

    log << "[reducing operations " << (funcNum++) << "/" << totalNumFuncs
        << "] " << func.getSymName();

    reversePostOrderWalk(func, [&](Operation *op) {
      if (op == func || op->hasTrait<OpTrait::IsTerminator>())
        return;
      Operation *next = op->getNextNode();
      Operation *stub = nullptr;
      assert(next && "unexpected terminator");
      if (!op->use_empty()) {
        OperationState state(op->getLoc(), "kgen-reduce.stub", {},
                             op->getResultTypes());
        OpBuilder b(op);
        stub = b.create(state);
        op->replaceAllUsesWith(stub->getResults());
      }
      op->remove();

      if (doesRepro(*curState.ir)) {
        op->erase();
        return;
      }

      OpBuilder(next).insert(op);
      if (stub) {
        stub->replaceAllUsesWith(op->getResults());
        stub->erase();
      }
    });
  }

  return success();
}

ErrorOrSuccess Reducer::reduceDCE(IRState &curState) {
  ErrorOr<bool> result = tryDCE(curState);
  if (failed(result))
    return result.takeError();
  if (*result)
    return success();

  std::vector<KGEN::FuncOp> funcs;
  for (auto func : curState.ir->getOps<KGEN::FuncOp>()) {
    // Ignore stubbed functions.
    if (isStubbed(func.getBodyRegion()))
      continue;
    funcs.push_back(func);
  }

  for (KGEN::FuncOp func : funcs) {
    func.setExported();
    ErrorOr<bool> result = tryDCE(curState);
    if (failed(result))
      return result.takeError();
    if (*result)
      return success();
    func.setNotExported();
  }

  return success();
}

ErrorOr<bool> Reducer::tryDCE(IRState &curState) {
  // Clone the module and attempt to run DCE.
  IRState nextState(curState.ir->clone());
  if (failed(dcePm.run(*nextState.ir)))
    return Error("DCE failed");

  // If the failure still reproduces after running DCE, swap the module.
  if (doesRepro(*nextState.ir)) {
    log << "[kgen-reduce] fails with DCE\n";
    curState.ir = std::move(nextState.ir);
    return true;
  }
  log << "[kgen-reduce] does not fail with DCE\n";

  return false;
}

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllKGENDialects(registry);
  MLIRContext mlirCtx{MLIRContext::Threading::DISABLED};
  mlirCtx.allowUnregisteredDialects();
  mlirCtx.appendDialectRegistry(registry);

  Reducer reducer(&mlirCtx);

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv);

  // Create our context.
  ErrorOr<ContextRef> ctxOr = Init::createContext(
      "kgen-reduce",
      Init::Options().withCPUDeviceOptions(
          AsyncRT::CPUDeviceOptions().withLeakCheckedAllocator()));
  if (ctxOr.isError()) {
    llvm::errs() << "failed to create context: " << ctxOr.getError() << "\n";
    return 1;
  }
  registerContext(mlirCtx, *ctxOr);

  KGEN::registerDefaultKGENPasses("kgen-reduce");

  if (auto err = reducer.run()) {
    llvm::errs() << "ERROR: " << err.getError() << "\n";
    return -1;
  }
}
