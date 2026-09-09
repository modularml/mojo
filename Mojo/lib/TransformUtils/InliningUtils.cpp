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

#include "Mojo/TransformUtils/InliningUtils.h"
#include "Mojo/HLCFDialect/HLCFDialect.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Mojo/ToolCommon/PassManagerConfigOptions.h"
#include "Support/Compiler/TimeProfilerTimingManager.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// inlineRegion
//===----------------------------------------------------------------------===//

std::pair<Operation *, bool> KGEN::inlineRegion(IRMapping &map,
                                                KGENCallOpInterface call,
                                                Region &region, bool takeBody) {
  IRRewriter b{OpBuilder(call)};
  return inlineRegion(b, map, call, call->getOperands(), region, takeBody);
}

std::pair<Operation *, bool> KGEN::inlineRegion(mlir::RewriterBase &b,
                                                IRMapping &map, Operation *call,
                                                ValueRange args, Region &region,
                                                bool takeBody) {
  // NOTE: All IR mutation must pass through the `RewriterBase`.
  // In-place mutation to `scope` is okay because it's a new operation.
  Operation *scope;
  std::function<void(Operation *)> handleReturn;

  // If the operation defines a call interface, use it to prepare inlining.
  if (auto itf = dyn_cast<KGENCallOpInterface>(call)) {
    FailureOr<InlineResult> result = itf.prepInline(b);
    if (failed(result)) {
      llvm::report_fatal_error("unexpected failure in inlining of call '" +
                               call->getName().getStringRef() +
                               "' -- please file a bug!");
    }
    std::tie(scope, handleReturn) = std::move(*result);
  } else {
    // Otherwise, assume this is inlining a direct call.
    StringAttr label = b.getStringAttr("inlined_cf_scope");
    scope = HLCF::LoopOp::create(b, call->getLoc(), call->getResultTypes(),
                                 ValueRange(), label);
    handleReturn = [label, &b](Operation *op) {
      b.replaceOpWithNewOp<HLCF::BreakOp>(op, op->getOperands(), label);
    };
  }

  // Update the location if the scope defines a subprogram.
  if (auto inlinedSubScoped =
          dyn_cast<DebugInfo::InlinedSubprogramScoped>(scope)) {
    inlinedSubScoped->setLoc(region.getParentOp()->getLoc());
    inlinedSubScoped.setCallLocAttr(call->getLoc());
  }

  Region &scopeBody = scope->getRegion(0);
  bool returnAtEnd = isa<ReturnOp>(region.front().getTerminator());
  if (takeBody) {
    b.inlineRegionBefore(region, scopeBody, scopeBody.end());
    for (auto [value, arg] : llvm::zip(args, scopeBody.getArguments()))
      b.replaceAllUsesWith(arg, value);
    scopeBody.front().eraseArguments(0, args.size());
  } else {
    Block *block = b.createBlock(&scopeBody);
    for (auto [value, arg] : llvm::zip(args, region.getArguments()))
      map.map(arg, value);
    for (BlockArgument trailing : region.getArguments().drop_front(args.size()))
      map.map(trailing,
              block->addArgument(trailing.getType(), trailing.getLoc()));
    for (Operation &op : region.getOps())
      b.clone(op, map);
  }

  unsigned numReturns = 0;
  scopeBody.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<ReturnOp>(op)) {
      b.setInsertionPoint(op);
      handleReturn(op);
      ++numReturns;
      return WalkResult::skip();
    }
    if (isa<FunctionLike>(op))
      return WalkResult::skip();

    if (auto sourceLocOp = dyn_cast<SourceLocOp>(op))
      processSourceLocOp(sourceLocOp, call->getLoc(), b);

    // If we inlined a "tail" call into a caller,  we need to clear the "tail"
    // marker.  This marker indicates that the callee doesn't access the
    // parent-caller's frame, but it might access the grand-caller's frame.
    TypeSwitch<Operation &>(*op).Case<CallOp, CallParamOp, CallIndirectOp>(
        [&](auto op) {
          if (op.getTailKind() == TailKind::Tail)
            op.setTailKind(TailKind::None);
        });

    return WalkResult::advance();
  });
  b.replaceOp(call, scope->getResults());
  return std::make_pair(scope, numReturns == 1 && returnAtEnd);
}

//===----------------------------------------------------------------------===//
// processSourceLocOp
//===----------------------------------------------------------------------===//

void KGEN::processSourceLocOp(SourceLocOp sourceLocOp, Location callLoc,
                              mlir::RewriterBase &b) {
  int64_t callLocInlineCount = 0;
  DebugInfo::walkLocation(callLoc, DebugInfo::LocWalkPolicy::CallerPriority,
                          [&](Location loc) -> WalkResult {
                            if (isa<mlir::CallSiteLoc>(loc))
                              return WalkResult::advance();
                            ++callLocInlineCount;
                            return WalkResult::skip();
                          });

  TypedAttr inlineCountParam = sourceLocOp.getInlineCount();
  // Decrement the inlineCount by the number of times the callsite was inlined.
  b.modifyOpInPlace(sourceLocOp, [&]() {
    sourceLocOp.setInlineCountAttr(ParamOperatorAttr::get(
        POC::Add, {inlineCountParam, b.getIndexAttr(-callLocInlineCount)}));
    // Op location must be updated immediately, as the op semantics is reliant
    // on an up to date location.
    DebugInfo::updateInlinedLoc(sourceLocOp, callLoc);
  });
}

//===----------------------------------------------------------------------===//
// foldTrivialLoop
//===----------------------------------------------------------------------===//

void KGEN::foldTrivialLoop(mlir::RewriterBase &b, Operation *op) {
  auto loop = dyn_cast<HLCF::LoopOp>(op);
  if (!loop)
    return;

  Block &body = loop.getBody().front();
  Operation *term = body.getTerminator();
  b.inlineBlockBefore(&body, loop);
  b.replaceOp(loop, term->getOperands());
  b.eraseOp(term);
}

//===----------------------------------------------------------------------===//
// updateScopeDebugInfo
//===----------------------------------------------------------------------===//

/// Get the IntegerAttr with the given name (if exists) and remove it from the
/// op. If the attr name is null, or the attr does not exist, returns null.
static IntegerAttr getAndRemoveTag(Operation *op, StringAttr updateAttrName) {
  IntegerAttr tag;
  if (updateAttrName)
    tag = dyn_cast_or_null<IntegerAttr>(op->removeAttr(updateAttrName));
  return tag;
}

static void updateBlockDebugInfo(Block &block, IntegerAttr tag,
                                 StringAttr updateAttrName,
                                 bool insideInlinedSubprogram,
                                 Location callLoc) {
  for (BlockArgument arg : block.getArguments())
    arg.setLoc(mlir::CallSiteLoc::get(arg.getLoc(), callLoc));

  for (Operation &op : llvm::make_early_inc_range(block)) {
    // Inline the location if not inside an inlined subprogram.
    // Skip SourceLocOp as its location is always up to date.
    if (!insideInlinedSubprogram && !isa<SourceLocOp>(op)) {
      DebugInfo::updateInlinedLoc(&op, callLoc);
    }

    // Don't recurse into nested functions.
    if (isa<FuncInterface>(op))
      continue;

    // Recurse into the body if needed and allowed.
    if (isa<DebugInfo::InlinedSubprogramScoped>(op)) {
      // Recurse inside if the inlined subprogram has a tag (deferred update).
      if (IntegerAttr tag = getAndRemoveTag(&op, updateAttrName))
        updateScopeDebugInfoFrom(&op, tag, updateAttrName);

      // Always skip walking directly into subprogram scopes.
      continue;
    } else if (updateAttrName && isa<HLCF::LoopOp>(op)) {
      if (IntegerAttr tag = getAndRemoveTag(&op, updateAttrName)) {
        updateScopeDebugInfoFrom(&op, tag, updateAttrName);
        continue;
      }
    }

    // Traverse into op.
    for (Region &region : op.getRegions())
      for (Block &block : region.getBlocks())
        updateBlockDebugInfo(block, tag, updateAttrName,
                             insideInlinedSubprogram, callLoc);
  }
}

void KGEN::updateScopeDebugInfoFrom(Operation *scope, IntegerAttr tag,
                                    StringAttr updateAttrName) {
  // Unpack the bits.
  auto value = static_cast<uint8_t>(tag.getInt());
  auto singleExit = static_cast<bool>(value & 1);

  // The scope operations contains the location of the call.
  Region &body = scope->getRegion(0);
  Location callLoc = scope->getLoc();

  bool insideInlinedSubprogram = isa<DebugInfo::InlinedSubprogramScoped>(scope);
  for (Block &block : body.getBlocks())
    updateBlockDebugInfo(block, tag, updateAttrName, insideInlinedSubprogram,
                         callLoc);

  // If this scope is a trivial control-flow scope, fold it away.
  if (singleExit) {
    IRRewriter b{OpBuilder(scope)};
    foldTrivialLoop(b, scope);
  }
}

void KGEN::updateScopeDebugInfo(FuncOp func, StringAttr updateAttrName) {
  VerboseCompilerTimeTraceScope updateScopeDebugInfo(
      "updateScopeDebugInfo", [&func] { return func.getSymName().str(); });
  func.getBody()->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (!isa<HLCF::LoopOp, FunctionLike>(op))
      return WalkResult::advance();
    IntegerAttr tag = getAndRemoveTag(op, updateAttrName);
    if (!tag)
      return WalkResult::advance();

    updateScopeDebugInfoFrom(op, tag, updateAttrName);
    return WalkResult::skip();
  });
}

void KGEN::maybeUpdateDebugInfo(Operation *scope,
                                std::optional<StringAttr> updateAttrName,
                                bool singleExit) {
  if (updateAttrName) {
    // We don't know where the op will end up, so tag it with an attribute.
    // Encode information {singleExit} as bits.
    IntegerAttr tag =
        OpBuilder(scope->getContext()).getI8IntegerAttr(singleExit);
    if (*updateAttrName) {
      // Deferred debuginfo update.
      scope->setAttr(*updateAttrName, tag);
    } else {
      // Immediate debuginfo update.
      // This will also foldTrivialLoops if applicable.
      updateScopeDebugInfoFrom(scope, tag, nullptr);
    }
  } else if (singleExit) {
    IRRewriter b{OpBuilder(scope)};
    foldTrivialLoop(b, scope);
  }
}

//===----------------------------------------------------------------------===//
// PerThreadPassManager
//===----------------------------------------------------------------------===//

/// This class manages a pass manager instance for each thread.
PerThreadPassManagers::PerThreadPassManagers(
    MLIRContext *ctx, function_ref<void(mlir::OpPassManager &)> buildFuncPasses)
    : ctx(ctx), buildFuncPasses(buildFuncPasses) {
  // Reserve the thread-local cache map so that it never resizes.
  pms.reserve(ctx->isMultithreadingEnabled()
                  ? ctx->getThreadPool().getMaxConcurrency()
                  : 1);
}

/// Get the pass manager for the current thread, initializing it if one does
/// not exist.
mlir::PassManager &PerThreadPassManagers::getPassManager() {
  int64_t threadId = llvm::get_threadid();
  {
    llvm::sys::SmartScopedReader<true> lock(mutex);
    if (auto it = pms.find(threadId); it != pms.end())
      return *it->second;
  }

  // Emplace a new pass manager for this thread.
  mutex.lock();
  mlir::PassManager &pm =
      *pms.try_emplace(threadId, std::make_unique<mlir::PassManager>(
                                     ctx, FuncOp::getOperationName()))
           .first->second;
  mutex.unlock();

  // Initialize the pass manager.
  buildFuncPasses(pm);
  pm.enableVerifier(false);
  PassManagerConfigOptions pmOptions;
  pmOptions.applyPassManagerCLOptions = true; // enable print options
  (void)pmOptions.configurePassManager(pm);
  // Enable time tracing on the nested pass manager.
  if constexpr (KGEN::kIsTracingEnabled)
    pm.enableTiming(std::make_unique<TimeProfilerTimingManager>());
  return pm;
}

uint64_t KGEN::getNumOperations(Operation *op) {
  if (!op)
    return 0;

  uint64_t result = 0;
  op->walk([&](Operation *op) {
    if (!isa_and_nonnull<DebugInfo::DebugInfoDialect>(op->getDialect()))
      ++result;
  });
  return result;
}
