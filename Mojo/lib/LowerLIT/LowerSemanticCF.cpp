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
// This pass lowers 'semantic' control flow into more structured control flow.
//
// This performs these lowerings:
//  1) It lowers statements like `lit.break` (which is not a terminator) into
//     `hlcf.break` (which is), and deletes unreachable code after it, and
//     diagnoses it if it is anything interesting.
//
//===----------------------------------------------------------------------===//

#include "Mojo/CODialect/COOps.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/HLCFDialect/HLCFUtils.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/LITDialect/LITAttrs.h"
#include "Mojo/LITDialect/LITOps.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Support/STLExtras.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Semantic control flow lowering.
//===----------------------------------------------------------------------===//

namespace {
/// Each function is lowered in a depth first walk through the region tree.
struct LowerSemanticCF {
  LIT::FnOp theFunc;
  SymbolRefAttr theFuncSymbol;

  // This is the current loop that a break or continue should exit from. This is
  // either an HLCF::LoopOp (for a LIT::LoopOp getting rewritten) or a
  // ParamForOp being lowering in place.
  Operation *currentLoop = nullptr;

  // This is the current loop that a lit.loop.break.else should exit from. This
  // is the lit.loop with the pre-lowered code in the else block.  Lowering a
  // lit.loop.break.else will take this IR.
  LIT::LoopOp currentBreakElseLoop = nullptr;

  // Each lowered hlcf.loop gets its own unique ID so we can break out of it if
  // needed.
  unsigned loopCounter = 0;

  // Each lowered lit.try gets its own unique ID so we can raise to it if
  // needed.
  unsigned tryCounter = 0;

  // True if we've emitted an error.
  bool hadError = false;

  // When lowering control flow, notice any recursive calls to diagnose infinite
  // recursion after the IR is validated and rewritten.
  bool hasRecursiveCalls = false;

  LowerSemanticCF(LIT::FnOp theFunc) : theFunc(theFunc) {
    if (!theFunc.isOptionalSymbol())
      theFuncSymbol = LIT::getFullyResolvedSymbolRef(theFunc);
  }
  void run();

private:
  void lowerBlock(Block &block, bool &doesRaise, bool &doesBreak,
                  bool &doesFallThrough);
  bool lowerLITLoop(LIT::LoopOp loopOp, bool &enclosingBlockDoesRaise,
                    bool &enclosingBlockDoesBreak);
  void lowerParamFor(ParamForOp paramFor, bool &enclosingBlockDoesRaise,
                     bool &enclosingBlockDoesBreak);
  void lowerElif(HLCF::ElifOp elifOp, bool &doesRaise, bool &doesBreak,
                 bool &doesFallThrough);
  bool checkSelfRecursion(Block &block, bool isConditional);
};
} // end anonymous namespace

/// Mangle a ParamDeclAttr or ParamDeclRefAttr during cloning of finally blocks.
/// This scheme postpends "f{cnt-1}" to the param name, which guarantees
/// uniqueness, since the parameters we started with must already be unique
/// within their scope. It also retains the demangling rule assumed by the stack
/// because removing everything after the backtick still yields the param name.
template <typename DeclOrRef>
static DeclOrRef mangle(DeclOrRef declOrRef, size_t cnt,
                        const SmallPtrSet<StringAttr, 4> &needToMangle,
                        DenseMap<Attribute, Attribute> &manglingCache) {
  StringAttr name = declOrRef.getName();
  // If counter is zero, we don't mangle to improve readability.
  if (cnt-- == 0 || !needToMangle.contains(name))
    return declOrRef;

  // Check the cache here instead of straightforward memoization so that we
  // limit memory footprint.
  if (Attribute cached = manglingCache.lookup(declOrRef))
    return cast<DeclOrRef>(cached);

  auto mangledName = StringAttr::get(name.getContext(),
                                     name.strref() + Twine("f") + Twine(cnt));
  auto mangledDecl = DeclOrRef::get(mangledName, declOrRef.getType());
  manglingCache.try_emplace(declOrRef, mangledDecl);
  return mangledDecl;
}

/// Insert 'finally' block logic on a `lit.try` operation by finding all
/// terminators that exit the try regions and pasting the finally clause before
/// it. Try operations must be processed post-order, so that the order in which
/// the finally clauses are pasted is correct.
static LIT::TryOp lowerTryFinally(LIT::TryOp tryOp) {
  Block &finallyBlock = tryOp.getFinallyRegion().front();

  // While cloning the finally block, we need to ensure parameter names are kept
  // unique. So we first collect parameter names that have to be mangled. Decls
  // nested within another decl scope can be ignored safely, but everything else
  // we need to remember.
  SmallPtrSet<StringAttr, 4> needToMangle;
  finallyBlock.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (auto paramOp = dyn_cast<ParamOpInterface>(op)) {
      paramOp.walkDeclarations(
          [&](ParamDeclAttr decl) { needToMangle.insert(decl.getName()); });
    }
    if (isa<DeclInterface>(op))
      return WalkResult::skip();
    return WalkResult::advance();
  });

  // We count how many times we cloned the block and use this for mangling.
  size_t finallyCount = 0;
  auto pasteFinally = [&](Operation *term) {
    OpBuilder b(term);
    IRMapping map;

    // Set up mangling utilities.
    DenseMap<Attribute, Attribute> manglingCache;
    mlir::AttrTypeReplacer replacer;
    replacer.addReplacement([&](ParamDeclAttr decl) {
      return mangle(decl, finallyCount, needToMangle, manglingCache);
    });
    replacer.addReplacement([&](ParamDeclRefAttr ref) {
      return mangle(ref, finallyCount, needToMangle, manglingCache);
    });

    for (Operation &op : finallyBlock.without_terminator()) {
      Operation *cloned = b.clone(op, map);

      // We mangle the parameter names in the op. This must happen recursively
      // for all ops, since references can be deeply nested.
      replacer.recursivelyReplaceElementsIn(cloned, /*replaceAttrs=*/true,
                                            /*replaceLocs=*/true,
                                            /*replaceTypes=*/true);
    }
    // If the finally block terminator exits, then the current terminator is
    // dead code.
    if (!isa<LIT::TryYieldOp>(finallyBlock.getTerminator())) {
      b.clone(*finallyBlock.getTerminator(), map);
      term->erase();
    }
    ++finallyCount;
  };

  // FIXME: Re-traversing `lit.try` operations is N^2. This could be computed
  // in one pass over the IR.
  auto checkRegion = [&](Operation *op) {
    // Control-flow will never cross nested functions.
    if (isa<LIT::FnOp>(op))
      return WalkResult::skip();

    // Check for a terminator that will branch past the enclosing try operation.
    auto term = dyn_cast<HLCF::ControlFlowTerminator>(op);
    if (!term)
      return WalkResult::advance();
    Operation *node = HLCF::getParentNode(term);
    if (node->isProperAncestor(tryOp))
      pasteFinally(term);
    return WalkResult::advance();
  };

  // Route exiting branches from the 'try', 'except', and 'else' regions through
  // the finally region. Nothing to do if the 'finally' region is trivial.
  if (!tryOp.hasTrivialFinally()) {
    tryOp.getTryRegion().walk(checkRegion);
    tryOp.getExceptRegion().walk(checkRegion);
    tryOp.getElseRegion().walk(checkRegion);
    // Paste the finally block at the exits of the else and except regions if
    // they are not terminated by an exit.
    if (auto yield = dyn_cast<LIT::TryYieldOp>(
            tryOp.getExceptRegion().front().getTerminator()))
      pasteFinally(yield);
    if (auto yield = dyn_cast<LIT::TryYieldOp>(
            tryOp.getElseRegion().front().getTerminator()))
      pasteFinally(yield);
  }

  // Clear the finally region by rebuilding the operation without it.
  OperationState state(tryOp.getLoc(), tryOp->getName());
  for (unsigned i = 0; i < 3; ++i) {
    state.regions.emplace_back(std::make_unique<Region>())
        ->takeBody(tryOp->getRegion(i));
  }
  OpBuilder b(tryOp);
  auto newTry = cast<LIT::TryOp>(b.create(state));
  newTry.setLabelAttr(tryOp.getLabelAttr());
  tryOp.erase();
  return newTry;
}

/// Erase operations to the end of the block after op.
static void eraseOpToEndOfBlock(Operation *op) {
  Block *block = op->getBlock();
  // Erase bottom up to avoid deleting an op while something uses its results.
  while (&block->back() != op)
    block->back().erase();
  op->erase();
}

/// Given a semantic terminator, diagnose and remove unreachable code, and
/// return a builder at the right spot to insert a replacement.
static ImplicitLocOpBuilder handleSemanticTerminatorOp(Operation &op,
                                                       StringRef stmtKind) {
  // Warn about dead code after the semantic terminator.
  Operation *nextOp = op.getNextNode();
  // We do report an error on `parameter if` since `parameter if` serves as a
  // if preprocessor in Mojo.
  if (!isa<ParamIfOp>(op) && !nextOp->hasTrait<OpTrait::IsTerminator>()) {
    // Don't complain if the location is the same as the enclosing function,
    // it is automatically synthesized.
    auto funcOp = nextOp->getParentOfType<LIT::FnOp>();
    if (!funcOp || funcOp->getLoc() != nextOp->getLoc())
      emitWarning(nextOp->getLoc(), "unreachable code after ") << stmtKind;
  }

  // Remove the unreachable code.
  eraseOpToEndOfBlock(nextOp);
  // Return a builder pointing to after "op".
  return ImplicitLocOpBuilder(op.getLoc(), op.getBlock(),
                              std::next(Block::iterator(&op)));
}

void LowerSemanticCF::lowerElif(HLCF::ElifOp elifOp, bool &doesRaise,
                                bool &doesBreak, bool &doesFallThrough) {
  bool elifFallsThrough = false;
  for (auto &region : elifOp->getRegions()) {
    if (region.empty())
      continue;
    // Additional condition regions (elifRegions even indices) always transfer
    // into a sibling then/else; they don't contribute to fallthrough of the
    // elif itself. Region layout: 0=then, 1=else, 2+=elifRegions.
    unsigned regionNumber = region.getRegionNumber();
    bool isAdditionalCond = regionNumber >= 2 && ((regionNumber - 2) % 2 == 0);

    bool blockRaises = false, blockBreaks = false, blockFallThroughs = false;
    lowerBlock(region.front(), blockRaises, blockBreaks, blockFallThroughs);
    doesRaise |= blockRaises;
    doesBreak |= blockBreaks;
    if (isAdditionalCond)
      continue;
    elifFallsThrough |= blockFallThroughs;
  }
  doesFallThrough = elifFallsThrough;
  if (!doesFallThrough) {
    auto b = handleSemanticTerminatorOp(
        *elifOp.getOperation(),
        "if statement with then/else that do not fall through");
    UnreachableOp::create(b, elifOp.getLoc());
  }
}

/// Lower a LIT::LoopOp to HLCF::LoopOp.  Return true if the lowering should
/// stop traversing the rest of the operations because this is an infinite loop
/// that doesn't fall through.
bool LowerSemanticCF::lowerLITLoop(LIT::LoopOp loopOp,
                                   bool &enclosingBlockDoesRaise,
                                   bool &enclosingBlockDoesBreak) {
  // Lower loop conditions.
  Block &bodyBlock = loopOp.getBodyRegion().front();
  Block &elseBlock = loopOp.getElseRegion().front();

  // Create the new HLCF::LoopOp.
  ImplicitLocOpBuilder builder(loopOp->getLoc(), loopOp);
  builder.setInsertionPointAfter(loopOp);
  auto newLoop = HLCF::LoopOp::create(builder);
  // Each loop gets a unique label.
  newLoop.setLabelAttr(builder.getStringAttr("_loop_" + Twine(loopCounter++)));

  // Move the loop's body to the HLCF::LoopOp's body.
  Block *newBody = builder.createBlock(&newLoop.getBody());
  newBody->getOperations().splice(newBody->end(), bodyBlock.getOperations());

  // Lower the body of the 'else' block before we move it over.  It is logically
  // NOT inside the loop even though it is nested under it in the HLCF AST. The
  // 'currentLoop' loop is set to the parent loop so any break or continue from
  // the 'else' logic will go to the right place.
  bool blockRaises = false, blockBreaks = false, elseBlockFallThroughs = false;
  lowerBlock(elseBlock, blockRaises, blockBreaks, elseBlockFallThroughs);
  enclosingBlockDoesRaise |= blockRaises;
  enclosingBlockDoesBreak |= blockBreaks;

  // Now that we know how the exit block works, we can look at its terminator.
  // If it fell through, it will end with lit.loop.yield: we replace it
  // with a break from this loop.  Other exits like return/break/continue in the
  // else block will already be rewritten if they are present.
  if (elseBlockFallThroughs) {
    assert(isa<LIT::LoopYieldOp>(elseBlock.getTerminator()));
    elseBlock.getTerminator()->erase();
    builder.setInsertionPointToEnd(&elseBlock);
    HLCF::BreakOp::create(builder, ValueRange{}, newLoop.getLabelAttr());
  }

  // Now that the else logic is set, lower the entire loop body to handle the
  // control flow in the body.  This is done with the loop set to the nested
  // loop so that breaks and continues get wired up to it.
  llvm::SaveAndRestore<Operation *> currentLoopSaver(currentLoop, newLoop);
  llvm::SaveAndRestore currentBreakElseLoopSaver(currentBreakElseLoop, loopOp);
  blockRaises = blockBreaks = false;
  bool blockFallThroughs = false;
  lowerBlock(*newBody, blockRaises, blockBreaks, blockFallThroughs);
  enclosingBlockDoesRaise |= blockRaises;

  // If the else block fell through, and if something actually used it (i.e. the
  // lit.loop.break.else was reachable) then the loop will exit with its break.
  if (elseBlockFallThroughs && elseBlock.empty())
    blockBreaks = true;

  // If the loop body never breaks, then the code after it is unreachable.
  if (!blockBreaks) {
    auto b = handleSemanticTerminatorOp(*newLoop, "infinite loop");
    UnreachableOp::create(b, loopOp.getLoc());
  }

  // Erase the lit.loop, and return true if it was an infinite loop.
  loopOp.erase();
  return !blockBreaks;
}

void LowerSemanticCF::lowerParamFor(ParamForOp paramFor,
                                    bool &enclosingBlockDoesRaise,
                                    bool &enclosingBlockDoesBreak) {
  // The 'else' region is not inside the loop. It is transparent to raises
  // and breaks.
  bool elseRaises = false, elseBreaks = false, elseFallsThrough = false;
  lowerBlock(paramFor.getElseRegion().front(), elseRaises, elseBreaks,
             elseFallsThrough);
  enclosingBlockDoesRaise |= elseRaises;
  enclosingBlockDoesBreak |= elseBreaks;

  // The loop is only transparent to raises.
  bool loopRaises = false, loopBreaks = false, loopFallsThrough = false;
  llvm::SaveAndRestore<Operation *> currentLoopSaver(currentLoop, paramFor);
  lowerBlock(paramFor.getBody().front(), loopRaises, loopBreaks,
             loopFallsThrough);
  enclosingBlockDoesRaise |= loopRaises;

  // Now that we've lowered the body and else logic to bind any nested breaks
  // or continues, we can re-parent the 'else' block into the body of the loop.
  // We do this transformation to make CheckLifetimes and the Elaborator's job
  // easier by not having to understand the 'else' logic.  We lower:
  //    kgen.param.for iter in stuff {
  //       kgen.param.if should_stop() {
  //          kgen.param.for.goto.else
  //       }
  //       body_that_uses_iter
  //    } else {
  //       cleanup_that_doesnt_happen_on_break_or_return
  //    }
  // Into:
  //    kgen.param.for i in stuff {
  //       kgen.param.if should_stop() {
  //          cleanup_that_doesnt_happen_on_break_or_return
  //          kgen.param.for.break
  //       }
  //       body_that_uses_i
  //    } else {
  //       kgen.unreachable
  //    }
  [[maybe_unused]] bool sawGotoElse = false;
  paramFor.getBody().front().walk<mlir::WalkOrder::PreOrder>(
      [&](Operation *op) {
        // Don't step into nested functions etc.
        if (isa<LIT::FnOp>(op))
          return WalkResult::skip();

        auto gotoElse = dyn_cast<ParamForGotoElseOp>(op);
        if (!gotoElse)
          return WalkResult::advance();
        assert(!sawGotoElse && "saw multiple goto else's");
        Block &elseBlock = paramFor.getElseRegion().front();
        Block &gotoElseBlock = *op->getBlock();

        // If the else block ended in a yield, then it should become a 'break',
        // otherwise it is a return or something else that we leave alone.
        if (auto elseTerm = dyn_cast<ParamYieldOp>(elseBlock.getTerminator())) {
          auto builder = OpBuilder(elseTerm);
          ParamForBreakOp::create(builder, elseTerm.getLoc());
          elseTerm.erase();
        }

        // Move the pre-lowered body of the 'else' block here, replacing the
        // kgen.param.for.goto.else.
        gotoElseBlock.getOperations().splice(Block::iterator(op),
                                             elseBlock.getOperations());

        // Fill the else block with an unreachable and remove this op.
        auto builder = OpBuilder::atBlockBegin(&elseBlock);
        UnreachableOp::create(builder, op->getLoc());
        op->erase();
        return WalkResult::skip();
      });
}

/// Emit the semantic control-flow IR corresponding to a raise statement.
static void emitRaise(ImplicitLocOpBuilder &b) {
  Operation *opForRaise = LIT::findOpProcessingRaise(b.getInsertionBlock());
  assert(opForRaise && "IR invalid, RaiseOp must only be in valid context");
  if (auto fnOp = dyn_cast<LIT::FnOp>(opForRaise)) {
    // If we are in a raising deinit method, and this call throws out of it,
    // then we should assume that we've deconstructed self enough.  Mark self
    // as destroyed so we don't get an error.
    for (auto [conv, arg] :
         llvm::zip(fnOp.getFuncTypeGenerator().getBody().getArgConventions(),
                   fnOp.getBody()->getArguments())) {
      if (conv == ArgConvention::DeinitMem)
        LIT::OwnershipMarkDestroyedOp::create(b, arg);
    }
    LIT::ErrorReturnOp::create(
        b, ParamConstantOp::create(
               b, SIMDAttr::getScalarBool(b.getContext(), true)));
  } else {
    LIT::TryRaiseOp::create(b, cast<LIT::TryOp>(opForRaise).getLabelAttr());
  }
}

/// This function adds the error branch regions to a call operation to a
/// throwing function. These are required by CheckLifetimes to understand
/// conditional initialization of the 'mut' results.
///
/// This returns the operation if nothing was added or returns the 'if' that is
/// left after the call if something was.
static Operation *addErrorRegions(Operation &op, FuncType sig,
                                  ValueRange operands) {
  // If the function throws Never, ignore the throwability entirely.
  if (sugarIsa<NeverType>(sig.getUserThrownType()))
    return &op;

  // Clone the op and add the error regions.
  ImplicitLocOpBuilder b(op.getLoc(), OpBuilder(&op));
  b.setInsertionPointAfter(&op);
  auto ifOp = HLCF::IfOp::create(b, op.getResult(0));

  // In the error region, mark the result has known consumed, then raise.
  b.createBlock(&ifOp.getThenRegion());
  LIT::OwnershipMarkConsumedOp::create(b, operands.back());
  emitRaise(b);

  // In the result region, mark the error as known consumed.
  b.createBlock(&ifOp.getElseRegion());
  LIT::OwnershipMarkConsumedOp::create(b, operands[operands.size() - 2]);
  HLCF::YieldOp::create(b);
  return ifOp;
}

/// This recursive function transforms the specified block:
///   1) It transforms any semantic CF ops like lit.break into terminators like
///      hlcf.break.
///   2) It removes dead code after that and reports errors.
///   3) It computes properties about the block and enclosing context.
void LowerSemanticCF::lowerBlock(Block &block, bool &doesRaise, bool &doesBreak,
                                 bool &doesFallThrough) {
  doesRaise = doesBreak = doesFallThrough = false;

  for (Operation &op : llvm::make_early_inc_range(block)) {
    // Look for semantic terminators and turn them into real terminators.
    if (auto returnOp = dyn_cast<LIT::ReturnOp>(op)) {
      auto b = handleSemanticTerminatorOp(op, "return statement");
      KGEN::ReturnOp::create(b, returnOp.getOperands());
      op.erase();
      doesFallThrough = false;
      return;
    }

    if (auto raiseOp = dyn_cast<LIT::RaiseOp>(op)) {
      doesRaise = true;
      auto b = handleSemanticTerminatorOp(op, "raise statement");
      emitRaise(b);
      op.erase();
      doesFallThrough = false;
      return;
    }

    if (isa<LIT::BreakOp>(op)) {
      if (!currentLoop) {
        emitError(op.getLoc(), "'break' is not inside a loop");
        hadError = true;
        op.erase();
        continue;
      }

      doesBreak = true;
      auto b = handleSemanticTerminatorOp(op, "break statement");
      if (auto hlcfLoop = dyn_cast<HLCF::LoopOp>(currentLoop))
        HLCF::BreakOp::create(b, ValueRange{}, hlcfLoop.getLabelAttr());
      else
        ParamForBreakOp::create(b);
      op.erase();
      return;
    }

    if (isa<LIT::ContinueOp>(op)) {
      if (!currentLoop) {
        emitError(op.getLoc(), "'continue' is not inside a loop");
        hadError = true;
        op.erase();
        continue;
      }

      auto b = handleSemanticTerminatorOp(op, "continue statement");
      if (auto hlcfLoop = dyn_cast<HLCF::LoopOp>(currentLoop))
        HLCF::ContinueOp::create(b, ValueRange{}, hlcfLoop.getLabelAttr());
      else
        ParamForContinueOp::create(b);
      op.erase();
      return;
    }

    // A lit.loop.continue is a terminator that continues the lit.loop.
    if (isa<LIT::LoopContinueOp>(op)) {
      OpBuilder b(&op);
      HLCF::ContinueOp::create(b, op.getLoc(), ValueRange{},
                               cast<HLCF::LoopOp>(currentLoop).getLabelAttr());
      op.erase();
      return;
    }

    // A lit.loop.break.else is a terminator that breaks out of a lit.loop to
    // its 'else' block.
    if (isa<LIT::LoopBreakElseOp>(op)) {
      // This operation is only created by the parser, there is no user error
      // that can trigger this.
      assert(currentBreakElseLoop &&
             "INTERNAL ERROR: malformed lit.loop.break.else "
             "not inside a lit.loop");
      // We currently assume there is only a single one of these per loop so
      // we can move the 'else' block over.  Allowing multiple would require us
      // to copy the IR.
      Block &elseBlock = currentBreakElseLoop.getElseRegion().front();
      assert(!elseBlock.empty() &&
             "ERROR: multiple lit.loop.break.else operations in one lit.loop?");
      // Lowering will already have lowered all the operations in the else block
      // of the loop.  We can just move them over now, including the terminator,
      // which will replace this one.
      op.getBlock()->getOperations().splice(Block::iterator(op),
                                            elseBlock.getOperations());
      op.erase();
      return;
    }

    // A kgen.param.for.goto.else is a terminator that jumps to the 'else' block
    // of a kgen.param.for.  It is generated syntactically by the parser so it
    // is always valid and does not fall through (though the else block can).
    if (isa<ParamForGotoElseOp>(op)) {
      doesBreak = true;
      return;
    }

    // Notice self-recursive calls so we can check them out later.
    if (auto call = dyn_cast<LIT::CallOp>(op))
      hasRecursiveCalls |= call.getDirectCallee() == theFuncSymbol;

    // Add error branches to calls to throwing functions.
    if (isa<LIT::CallOp, LIT::CallIndirectOp>(op)) {
      LIT::FnTypeGeneratorType calleeType = LIT::getFnTypeFromCall(op);
      assert(calleeType);
      Operation *opAfterCall = &op;
      if (calleeType.isThrows()) {
        opAfterCall = addErrorRegions(op, calleeType.getBody(),
                                      LIT::getCalleeArguments(&op));
        doesRaise |= (opAfterCall != &op);
      }

      // If the function returns NeverType, then it can never return.
      // treat it like a semantic terminator.
      if (sugarIsa<NeverType>(calleeType.getUserResultType())) {
        auto b = handleSemanticTerminatorOp(*opAfterCall,
                                            "function that never returns");
        UnreachableOp::create(b, op.getLoc(),
                              /*isAfterUnreachableCall=*/true);
        return;
      }

      continue;
    }

    // A kgen.param.assert with a statically false condition means the
    // assertion will fail at elaboration time; all code after it is dead.
    if (auto assertOp = dyn_cast<ParamAssertOp>(op)) {
      if (auto cond = sugarDynCast<SIMDAttr>(assertOp.getCond());
          cond && !cond.getAsBool()) {
        auto b =
            handleSemanticTerminatorOp(op, "compile-time assertion failure");
        UnreachableOp::create(b, op.getLoc(), /*isAfterUnreachableCall=*/false);
        return;
      }
      continue;
    }

    // Most ops don't have regions and are just fallthrough.
    // TODO: Add support for noreturn calls.
    if (!op.getNumRegions())
      continue;

    // Coroutine await regions are fallthrough only.
    if (auto await = dyn_cast<CO::SuspendOp>(op)) {
      bool awaitRaises = false, awaitBreaks = false, awaitFallsThrough = false;
      lowerBlock(await.getBody().front(), awaitRaises, awaitBreaks,
                 awaitFallsThrough);
      // The verifier will catch any invalid control-flow structure.
      continue;
    }

    // Ignore nested functions, they are handled (and lowered) separately by the
    // outer walker, which we are recursing within post-order.
    if (isa<LIT::FnOp>(op))
      continue;

    // Import gates carry a region but are resolution-only artifacts and play no
    // part in control flow, so skip them.
    if (isa<LIT::ImportOp>(op))
      continue;

    // Process a try op specially to identify dead code and warn.
    if (auto tryOp = dyn_cast<LIT::TryOp>(op)) {
      tryOp.setLabelAttr(
          StringAttr::get(tryOp->getContext(), "try" + Twine(tryCounter++)));

      bool tryBodyRaises = false, tryBodyBreaks = false,
           tryBodyFallsThrough = false;
      lowerBlock(tryOp.getTryRegion().front(), tryBodyRaises, tryBodyBreaks,
                 tryBodyFallsThrough);
      doesBreak |= tryBodyBreaks;

      // The try falls through if the except block is reachable and falls
      // through, or if the body falls through and so does the else.
      bool tryFallsThrough = false;

      // Diagnose unneeded code.
      if (!tryBodyRaises) {
        Operation &firstOpInExcept = tryOp.getExceptRegion().front().front();
        // If the finally region is not empty, then this could be a
        // try-finally pattern.
        if (!tryOp.getSuppressWarnings() && tryOp.hasTrivialFinally()) {
          if (!firstOpInExcept.hasTrait<OpTrait::IsTerminator>()) {
            emitWarning(firstOpInExcept.getLoc(),
                        "'except' logic is unreachable, try doesn't raise an "
                        "exception");

          } else {
            emitWarning(tryOp->getLoc(), "try body doesn't raise an exception");
          }
        }

        Operation *firstExceptOp = &tryOp.getExceptRegion().front().front();
        auto builder = OpBuilder(firstExceptOp);
        UnreachableOp::create(builder, firstExceptOp->getLoc());
        eraseOpToEndOfBlock(firstExceptOp);
      } else {
        // The except and else blocks execute without protection from the try.
        bool exceptRaises = false, exceptBreaks = false;
        lowerBlock(tryOp.getExceptRegion().front(), exceptRaises, exceptBreaks,
                   tryFallsThrough);
        doesRaise |= exceptRaises;
        doesBreak |= exceptBreaks;
      }

      // If there is an 'else' block that is unreachable, complain and remove
      // it, otherwise process it.
      if (!tryBodyFallsThrough) {
        Operation *firstElseOp = &tryOp.getElseRegion().front().front();
        auto builder = OpBuilder(firstElseOp);
        UnreachableOp::create(builder, firstElseOp->getLoc());
        if (!isa<LIT::TryYieldOp>(firstElseOp))
          emitWarning(firstElseOp->getLoc(),
                      "'else' logic in 'try' is unreachable");
        eraseOpToEndOfBlock(firstElseOp);
      } else {
        bool elseRaises = false, elseBreaks = false, elseFallsThrough = false;
        lowerBlock(tryOp.getElseRegion().front(), elseRaises, elseBreaks,
                   elseFallsThrough);
        doesRaise |= elseRaises;
        doesBreak |= elseBreaks;
        tryFallsThrough |= elseFallsThrough;
      }

      // The 'finally' block must fallthrough for the try to fallthrough. Also,
      // it is transparent to raises and breaks.
      bool finallyFallsThrough = false, finallyRaises = false,
           finallyBreaks = false;
      // Lower finally block in the context of the current tryOp s.t. the raise
      // label inside (if any) will be set correctly. The finally region will
      // then be pasted to other places with the correct label inferred.
      lowerBlock(tryOp.getFinallyRegion().front(), finallyRaises, finallyBreaks,
                 finallyFallsThrough);
      doesRaise |= finallyRaises;
      doesBreak |= finallyBreaks;
      tryFallsThrough &= finallyFallsThrough;

      // Modify the body of the try to implement 'finally' logic.
      tryOp->setOperands({});
      tryOp = lowerTryFinally(tryOp);

      // If the try doesn't fall through, diagnose unreachable code after it.
      if (!tryFallsThrough) {
        auto b = handleSemanticTerminatorOp(
            *tryOp, "try statement that doesn't fall through");
        UnreachableOp::create(b, tryOp.getLoc());
        return;
      }
      continue;
    }

    // Process a LIT::LoopOp.
    if (auto loopOp = dyn_cast<LIT::LoopOp>(op)) {
      if (lowerLITLoop(loopOp, doesRaise, doesBreak))
        return;
      continue;
    }

    // Process a ParamForOp
    if (auto paramFor = dyn_cast<ParamForOp>(op)) {
      lowerParamFor(paramFor, doesRaise, doesBreak);
      continue;
    }

    // Process a HLCF::ElifOp
    if (auto elifOp = dyn_cast<HLCF::ElifOp>(op)) {
      // If the first condition is a known constant, mark unreachable regions
      // so we don't consider them live.
      SIMDAttr elifCond;
      if (mlir::matchPattern(elifOp.getCond(), m_Constant(&elifCond))) {
        bool constantCondValue = elifCond.getAsBool();
        auto markDead = [&](Region &region) {
          if (region.empty())
            return;
          Block &deadBlock = region.front();
          Operation *firstDeadOp = &deadBlock.front();
          if (!firstDeadOp->hasTrait<OpTrait::IsTerminator>())
            emitWarning(firstDeadOp->getLoc(), "unreachable code after 'if ")
                << (constantCondValue ? "True'" : "False'");
          eraseOpToEndOfBlock(firstDeadOp);
          auto b = OpBuilder::atBlockBegin(&deadBlock);
          UnreachableOp::create(b, op.getLoc());
        };
        if (constantCondValue) {
          // First then is taken; else and additional arms are dead.
          markDead(elifOp.getElseRegion());
          for (Region &region : elifOp.getElifRegions())
            markDead(region);
        } else {
          // First then is dead; later arms / else remain live.
          markDead(elifOp.getThenRegion());
        }
      }

      bool elifFallsThrough = false;
      lowerElif(elifOp, doesRaise, doesBreak, elifFallsThrough);
      if (elifFallsThrough) {
        // Continue on and process the rest of the current containing `block`.
        // We don't assign doesFallThrough = true because this scope's
        // doesFallThrough is talking about what happens at the end of this
        // current containing `block`, and is only known when this
        // `LowerSemanticCF::lowerBlock` call returns.
        continue;
      } else {
        // The elif doesn't fall through, which means everything after here is
        // dead code, so return.
        doesFallThrough = false;
        return;
      }
    }

    // Otherwise we must have an if operation.
    assert((isa<HLCF::IfOp, ParamIfOp>(op)) &&
           "Unknown operation with regions");

    // If this is a dynamic `if False:` or comptime if on known condition,
    // mark the unreachable block as unreachable so we don't consider it live.
    Region *deadRegion = nullptr;
    bool constantCondValue = false;
    if (auto ifOp = dyn_cast<HLCF::IfOp>(op)) {
      SIMDAttr cond;
      if (mlir::matchPattern(ifOp.getCond(), m_Constant(&cond))) {
        constantCondValue = cond.getAsBool();
        deadRegion =
            &(constantCondValue ? ifOp.getElseRegion() : ifOp.getThenRegion());
      }
    } else if (auto ifOp = dyn_cast<ParamIfOp>(op)) {
      if (auto cond = sugarDynCast<SIMDAttr>(ifOp.getCond())) {
        constantCondValue = cond.getAsBool();
        deadRegion =
            &(constantCondValue ? ifOp.getElseRegion() : ifOp.getThenRegion());
      }
    }

    // If either branch of the if is unreachable, diagnose any live code there
    // as unreachable and replace it with a kgen.unreachable so we don't think
    // about it for liveness' sake.
    if (deadRegion) {
      Block &deadBlock = deadRegion->front();
      Operation *firstDeadOp = &deadBlock.front();
      // Warn about unreachable code in an 'if', but not in a 'comptime if'.
      // It serves the function of ifdef's, and conditions are often
      // known-statically true/false.
      if (!isa<ParamIfOp>(op) &&
          !firstDeadOp->hasTrait<OpTrait::IsTerminator>())
        emitWarning(firstDeadOp->getLoc(), "unreachable code after 'if ")
            << (constantCondValue ? "True'" : "False'");
      eraseOpToEndOfBlock(&deadBlock.front());
      auto builder = OpBuilder::atBlockBegin(&deadBlock);
      UnreachableOp::create(builder, op.getLoc());
    }

    bool ifOpFallsThrough = false;
    for (auto &region : op.getRegions()) {
      bool regionRaises = false, regionBreaks = false,
           regionFallsThrough = false;
      lowerBlock(region.front(), regionRaises, regionBreaks,
                 regionFallsThrough);
      doesRaise |= regionRaises;
      doesBreak |= regionBreaks;
      ifOpFallsThrough |= regionFallsThrough;
    }

    // If the operation doesn't fall through, cut off the code after it.
    if (!ifOpFallsThrough) {
      auto b = handleSemanticTerminatorOp(
          op, "if statement with then/else that do not fall through");
      UnreachableOp::create(b, op.getLoc());
      return;
    }
  }

  auto *terminator = &block.back();
  if (isa<HLCF::BreakOp, ParamForBreakOp>(terminator)) {
    doesBreak = true;
    return;
  }

  // These are not fallthroughs.
  if (isa<KGEN::ReturnOp, HLCF::ContinueOp, ParamForContinueOp,
          KGEN::UnreachableOp, LIT::ErrorReturnOp>(terminator))
    return;

  // If we fell off the bottom, then we have a fall-through terminator.
  assert((isa<HLCF::YieldOp, HLCF::ElifYieldOp, LIT::TryYieldOp, ParamYieldOp,
              LIT::EndFnOp, CO::SuspendEndOp, LIT::LoopYieldOp>(block.back())));
  doesFallThrough = true;
}

/// This function is called to check to see if the function has any
/// unconditional self-recursive calls.  Such a call will cause an infinite
/// loop, so we generate a warning.
///
/// This function is invoked on blocks after SemanticCF lowering is done on the
/// function. The "isConditional" argument indicates whether this is being
/// called in a conditional context (e.g. under an if check).  This returns
/// `true` if the block might early return out of the enclosing function with a
/// return or throw, `false` if it will fall through.
bool LowerSemanticCF::checkSelfRecursion(Block &block, bool isConditional) {
  for (Operation &op : llvm::make_early_inc_range(block)) {
    // Notice self-recursive calls so we can check them out later.  If we are
    // invoked in an unconditional area, we can emit the warning.
    if (auto call = dyn_cast<LIT::CallOp>(op);
        call && !isConditional && theFuncSymbol &&
        call.getDirectCallee() == theFuncSymbol) {
      emitWarning(call.getLoc(),
                  "self recursive call will cause an infinite loop");
      continue;
    }

    // If this is a return out of the function, notice this and we're done.
    // LIT::TryRaiseOp/break/continue/etc are used for transfers to an enclosing
    // try, which doesn't completely exit the function.
    if (isa<KGEN::ReturnOp, LIT::ErrorReturnOp>(op))
      return true;

    // Most ops don't have regions and are just fallthrough.
    if (!op.getNumRegions())
      continue;

    // Ignore nested functions, they are handled separately by the outer walker.
    if (isa<LIT::FnOp>(op))
      continue;

    // If we are already in conditional code, or if this is an 'if'-like
    // operation, then the subregions are executed conditionally.
    bool isSubregionConditional =
        isConditional || isa<HLCF::IfOp, ParamIfOp, HLCF::ElifOp>(op);
    // Handle things like if statements, HLCF::Loop, try, etc.
    for (auto &region : op.getRegions()) {
      if (checkSelfRecursion(region.front(), isSubregionConditional))
        return true;
    }
  }

  // If we made it this far then we didn't early return.
  return false;
}

/// Lower all lexical terminators in the function and remove dead code.
void LowerSemanticCF::run() {
  bool doesRaise = false, doesBreak = false, doesFallThrough = false;
  lowerBlock(*theFunc.getBody(), doesRaise, doesBreak, doesFallThrough);

  // If we had an error already, don't diagnose more semantic issues.
  if (hadError)
    return;

  // A return is required at the end of function, diagnose it if missing.  The
  // parser automatically inserts a `return None` in functions that return None.
  if (auto endFunc =
          dyn_cast<LIT::EndFnOp>(theFunc.getBody()->getTerminator())) {
    // If this is a signature resolved function (not body resolved) then don't
    // error.
    if (!endFunc.getUnresolved()) {
      emitError(endFunc->getLoc(),
                "return expected at end of function with results");
      hadError = true;
      return;
    }
  }

  // If everything looks good, check whether any self-recursive calls are
  // unconditionally executed.  If so, they are infinite recursion.  We need
  // control flow information to avoid diagnosing recursive calls in if
  // statements.
  if (hasRecursiveCalls)
    (void)checkSelfRecursion(*theFunc.getBody(), /*isConditional=*/false);
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERSEMANTICCF
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerSemanticCFPass : impl::LowerSemanticCFBase<LowerSemanticCFPass> {
  using LowerSemanticCFBase::LowerSemanticCFBase;

  void runOnOperation() override {
    // Walk all functions and update them.
    bool hadError = false;
    getOperation().walk<mlir::WalkOrder::PostOrder>([&](LIT::FnOp func) {
      // Skip external functions.
      if (func.isExternal())
        return;

      // Lower things like lit.break into hlcf.break which are terminators,
      // and diagnose unreachable code.
      LowerSemanticCF lowerer(func);
      lowerer.run();
      hadError |= lowerer.hadError;
    });
    if (hadError)
      return signalPassFailure();
  }
};
} // namespace
