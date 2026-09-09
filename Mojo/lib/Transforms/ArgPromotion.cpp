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
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Mojo/TransformUtils/MemoryUtils.h"
#include "Mojo/TransformUtils/SCCUtils.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace M;
using namespace KGEN;
using namespace POP;

namespace M::KGEN {
#define GEN_PASS_DEF_ARGPROMOTION
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct ArgPromotionPass : public impl::ArgPromotionBase<ArgPromotionPass> {
  using ArgPromotionBase::ArgPromotionBase;
  void runOnOperation() override;
};

enum class State { NoCapture, Capture };

struct Node : public SCCNode<Node, FuncOp, CallOp> {
  Node(Node &&other)
      : SCCNode(other.func), argStates(std::move(other.argStates)) {}

  Node(FuncOp func) : SCCNode(func) {
    // Check for the root node.
    if (!func)
      return;

    // Optimistically assume that in-memory arguments are not captured. However,
    // treat all other argument conventions as capturing if a pointer is passed
    // to them. This sets them to a fixed point, so we don't need to check them.
    bool exported = func.isExported();
    for (ArgConvention conv :
         func.getFuncTypeGenerator().getBody().getArgConventions()) {
      // We can't change the ABI of exported functions.
      argStates.push_back(!exported && hasAddress(conv) ? State::NoCapture
                                                        : State::Capture);
    }
  }

  /// Whether each in-memory argument of the function is known to be captured.
  SmallVector<State> argStates;

  /// If the function has references outside of direct calls, we cannot promote
  /// it. Keep a flag here to set during graph building.
  std::atomic<bool> canPromote = true;
};

struct Graph : public SCCGraph<Graph, Node> {
  Graph(const SymbolTable &symtab, TargetInfoAttr target,
        unsigned maxInlineSize)
      : symtab(symtab), target(target), maxInlineSize(maxInlineSize) {}

  /// Perform analysis on promotable arguments within a single node.
  bool doAnalysis(Node *node);
  State doAnalysis(BlockArgument arg);

  /// Promote arguments on a function.
  void doRewrite(const Node *node);

  /// Whether an argument should be promoted.
  bool shouldPromote(BlockArgument arg);

  /// Check non-direct-call operations for references to functions. These
  /// functions cannot be modified.
  void checkNonCallOp(Operation *op) {
    mlir::AttrTypeWalker walker;
    walker.addWalk([&](FlatSymbolRefAttr ref) {
      nodes.find(symtab.lookup<FuncOp>(ref.getAttr()))->second.canPromote =
          false;
    });
    for (const NamedAttribute &attr : op->getAttrs())
      walker.walk(attr.getValue());
    // Note: At this stage in the pipeline, there should be no function
    // references inside types or locations.
  }

  /// Symbol table for function lookup.
  const SymbolTable &symtab;
  /// Target model.
  TargetInfoAttr target;
  /// The largest argument size to promote.
  unsigned maxInlineSize;
};

struct UseIterator : public ProjectionUseIterator<UseIterator> {
  using ProjectionUseIterator::ProjectionUseIterator;

  /// Project through trivially aliasing operations.
  Value project(OpOperand &use) {
    Operation *op = use.getOwner();
    if (isa<StructGEPOp, ArrayGEPOp, OffsetOp, PointerBitcastOp,
            UnionBitcastOp>(op)) {
      // FIXME: We really need an interface for this.
      assert(op->getNumResults() == 1);
      return op->getResult(0);
    }
    // TODO: We could also project through `hlcf.yield` to `hlcf.if`, etc.
    return {};
  }
};
} // namespace

bool Graph::shouldPromote(BlockArgument arg) {
  Type argType = cast<PointerType>(arg.getType()).getElementType();
  // Promote if the natural size of the type is less than the configured max.
  std::optional<int64_t> size =
      DataLayoutInterface::getTypeStoreSize(target, argType);
  if (!size)
    return false;
  return static_cast<unsigned>(*size) <= maxInlineSize;
}

bool Graph::doAnalysis(Node *node) {
  FuncOp func = node->func;

  // If the function can't be promoted, set everything to a fixed point.
  if (!node->canPromote) {
    bool changed = false;
    for (State &state : node->argStates) {
      changed |= state != State::Capture;
      state = State::Capture;
    }
    return changed;
  }

  // An in-memory argument can be promoted if and only if the full projection of
  // the pointer does not escape. Specifically, we promote an argument if all
  // read and write effects are directly visible on the projection.
  bool changed = false;
  for (BlockArgument arg : func.getArguments()) {
    State &state = node->argStates[arg.getArgNumber()];
    // Short-circuit if the current argument state is at a fixed point.
    if (state == State::Capture)
      continue;

    State newState = doAnalysis(arg);
    changed |= newState != state;
    state = newState;
  }
  return changed;
}

State Graph::doAnalysis(BlockArgument arg) {
  // If we shouldn't promote the argument, then return the fixed point.
  if (!shouldPromote(arg))
    return State::Capture;

  for (UseIterator it(arg); !it.isAtEnd(); ++it) {
    OpOperand &use = *it;
    Operation *user = use.getOwner();

    // Direct load.
    if (auto load = dyn_cast<LoadOp>(user))
      if (!load.mightBeVolatile())
        continue;

    // Check stores.
    if (auto store = dyn_cast<StoreOp>(user)) {
      // Check if this is a direct capture.
      if (store.getArg() == use.get() || store.mightBeVolatile())
        return State::Capture;
      // Otherwise, this is a direct store.
      continue;
    }

    // Otherwise, if this isn't a call, assume it's a capture.
    auto call = dyn_cast<CallOp>(user);
    if (!call)
      return State::Capture;

    // Now check the call effects. If the argument convention is not in-memory,
    // we have already marked the state as capturing.
    auto func = symtab.lookup<FuncOp>(call.getCalleeSymbol().getAttr());
    const Node &node = nodes.find(func)->second;
    if (node.argStates[use.getOperandNumber()] == State::Capture)
      return State::Capture;

    // The call use does not capture!
  }

  // Okay, all the uses check out. We know this argument isn't captured.
  return State::NoCapture;
}

/// For a given argument convention, return whether it is an 'in' argument or
/// 'out' argument, respective. An 'in' argument is one that does not reflect
/// side-effects within the function back to callees, whereas 'out' arguments
/// are the opposite: they are uninitialized on entry and cannot be used to
/// observe the callee, but side-effects flow through to callees. 'mut'
/// arguments are both.
static std::pair<bool, bool> getInOutFlags(ArgConvention conv) {
  switch (conv) {
  case ArgConvention::ImmReg:
  case ArgConvention::OwnedReg:
    llvm_unreachable("these conventions should be treated as capturing");

  // 'read' and 'owned' arguments convey no side-effects to callees.
  case ArgConvention::ImmMem:
  case ArgConvention::OwnedMem:
  case ArgConvention::DeinitMem:
    return {true, false};

  // 'mut' can read and write. Pessimistically treat 'ref' as 'mut'.
  case ArgConvention::Mut:
  case ArgConvention::MutRef:
  case ArgConvention::Ref:
    return {true, true};

  // These argument conventions reflect values uninitialized on entry, and
  // thus cannot observe any callee state.
  case ArgConvention::ByRefResult:
  case ArgConvention::ByRefError:
    return {false, true};
  }
}

/// When converting an 'in' argument to be pass-by-value, return the
/// corresponding argument convention. To be pedantic, we preserve the
/// ownedness of the convention.
static ArgConvention getByValueConvention(ArgConvention conv) {
  assert(hasAddress(conv));
  return (conv == ArgConvention::OwnedMem || conv == ArgConvention::DeinitMem)
             ? ArgConvention::OwnedReg
             : ArgConvention::ImmReg;
}

void Graph::doRewrite(const Node *node) {
  FuncOp func = node->func;

  // Rewrite the nocapture function arguments on the current function. 'in'
  // arguments are passed in by value. 'out' arguments are dropped and returned
  // through SSA results.
  Block *body = func.getBody();
  FuncType signature = func.getFuncTypeGenerator().getBody();
  SmallVector<ArgConvention> convs;
  ImplicitLocOpBuilder b{func.getLoc(), OpBuilder::atBlockBegin(body)};
  SmallVector<mlir::TypedValue<PointerType>> outArgs;
  for (auto [i, conv, state] :
       llvm::enumerate(signature.getArgConventions(), node->argStates)) {
    BlockArgument arg = func.getArgument(i);

    // If the argument can't be rewritten, forward it as-is.
    if (state == State::Capture) {
      BlockArgument newArg = body->addArgument(arg.getType(), arg.getLoc());
      arg.replaceAllUsesWith(newArg);
      convs.push_back(conv);
      continue;
    }

    // We are going to be replacing this argument. Figure out how.
    auto [in, out] = getInOutFlags(conv);
    assert((in || out) && "expected some kind of rewrite prescription");

    // First, rewrite the argument as a local variable.
    auto type = cast<PointerType>(arg.getType());
    auto alloc = StackAllocationOp::create(b, type);
    arg.replaceAllUsesWith(alloc);

    // For 'in' arguments, take in the argument by value.
    if (in) {
      BlockArgument byval =
          body->addArgument(type.getElementType(), arg.getLoc());
      StoreOp::create(b, byval, alloc);
      convs.push_back(getByValueConvention(conv));
    }

    // For 'out' arguments, we need to add a new SSA result to the function and
    // return the value at every exit.
    if (out)
      outArgs.push_back(alloc);
  }
  // Erase the old set of arguments.
  body->eraseArguments(0, signature.getNumArguments());

  // Now find every return site and load and return each 'out' argument.
  SmallVector<Value> newOperands;
  if (!outArgs.empty()) {
    body->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
      // Visit return operations while walking over nested functions.
      if (isa<FunctionLike>(op))
        return WalkResult::skip();
      if (!isa<ReturnOp>(op))
        return WalkResult::advance();

      newOperands.clear(); // reuse memory

      ImplicitLocOpBuilder b{op->getLoc(), OpBuilder(op)};
      for (Value arg : outArgs)
        newOperands.push_back(LoadOp::create(b, arg));
      op->insertOperands(op->getNumOperands(), newOperands);
      return WalkResult::advance();
    });
  }

  // Now we can update the function signature.
  SmallVector<Type> newResultTypes = llvm::to_vector(signature.getResults());
  for (mlir::TypedValue<PointerType> arg : outArgs)
    newResultTypes.push_back(arg.getType().getElementType());
  auto functionType = FunctionType::get(
      func.getContext(), body->getArgumentTypes(), newResultTypes);
  signature = FuncType::get(functionType, convs, signature.getFnEffects());
  func.setFuncTypeGenerator(
      GeneratorType::get(/*inputParamTypes=*/{}, signature));

  // For the second part of the rewrite, we perform the corresponding rewrite of
  // calls in this function.
  for (auto [call, node] : node->callsites) {
    convs.clear(); // reuse memory
    newOperands.clear();
    outArgs.clear();

    FuncTypeGeneratorType sigGen = call.getCalleeSignature();
    FuncType sigBase = sigGen.getBody();
    ImplicitLocOpBuilder b{call.getLoc(), OpBuilder(call)};
    for (auto [arg, conv, state] :
         llvm::zip(call.getOperands(), sigBase.getArgConventions(),
                   node->argStates)) {

      // If the argument can't be rewritten, just forward it.
      if (state == State::Capture) {
        newOperands.push_back(arg);
        convs.push_back(conv);
        continue;
      }

      // Figure out how the argument needs to be rewritten.
      auto [in, out] = getInOutFlags(conv);
      assert((in || out) && "expected some kind of rewrite prescription");

      // For 'in' arguments, we load the current value and pass it in by value.
      if (in) {
        newOperands.push_back(LoadOp::create(b, arg));
        convs.push_back(getByValueConvention(conv));
      }

      // For 'out' arguments, we add a new result and store it. We will have to
      // bulk add the results, so just save the information here.
      if (out)
        outArgs.push_back(cast<mlir::TypedValue<PointerType>>(arg));
    }

    // Rewrite the operands.
    call->setOperands(newOperands);

    // Rewrite the results if necessary.
    if (!outArgs.empty()) {
      newResultTypes.clear();
      // Reconstruct the op with the new results added.
      llvm::append_range(newResultTypes, call->getResultTypes());
      for (mlir::TypedValue<PointerType> arg : outArgs)
        newResultTypes.push_back(arg.getType().getElementType());
      OperationState state(call.getLoc(), call->getName(), call->getOperands(),
                           newResultTypes);
      state.attributes = call->getAttrDictionary();
      auto newCall = cast<CallOp>(b.create(state));

      // Replace the old results with the slice of the new call's results.
      auto it = newCall->result_begin();
      auto e = std::next(it, call.getNumResults());
      call->replaceAllUsesWith(llvm::make_range(it, e));
      call.erase();
      call = newCall;

      // Now take the new results and write them back.
      b.setInsertionPointAfter(call);
      for (auto [arg, result] :
           llvm::zip(outArgs, llvm::make_range(e, newCall->result_end())))
        StoreOp::create(b, result, arg);
    }

    // Finally, update the callee signature.
    functionType = FunctionType::get(func.getContext(), call->getOperandTypes(),
                                     call->getResultTypes());
    FuncTypeGeneratorType calleeSignature = FuncTypeGeneratorType::get(
        {}, functionType, convs, sigBase.getFnEffects());
    call.setCalleeAttr(
        SymbolConstantAttr::get(call.getCalleeSymbol(), calleeSignature));
  }
}

void ArgPromotionPass::runOnOperation() {
  const SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
  AsyncRT::CPUDevice &cpuDevice =
      *loadContext(&getContext())->get<AsyncRT::CPUDevice>();

  // We need a target to run this pass.
  TargetInfoAttr target = lookupTargetInfo(getOperation());
  if (!target) {
    mlir::emitError(getOperation().getLoc(),
                    "could not find an enclosing target specification");
    return signalPassFailure();
  }

  Graph cg(symtab, target, maxInlineSize);
  cg.build(getOperation(), symtab);
  cg.run(cpuDevice);
}
