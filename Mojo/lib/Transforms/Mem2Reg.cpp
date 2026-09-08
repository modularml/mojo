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

#include "Mojo/HLCFDialect/Analysis/CFG.h"
#include "Mojo/HLCFDialect/HLCFOps.h"
#include "Mojo/Interpreter/InterpreterAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/TransformUtils/ControlFlowUtils.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace POP;

namespace M::KGEN {
#define GEN_PASS_DEF_MEM2REG
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
/// Statistics collected for this pass.
struct PassStats {
  unsigned numAllocsElided = 0;
  unsigned numLoadsElided = 0;
  unsigned numStoresElided = 0;
};

/// For each control-flow node, the promoted stack allocations that are stored
/// to somewhere inside it, in promotion order. Such an allocation is "variant"
/// across the node and has to be carried through its regions as an iteration
/// variable.
using NodeVariantMap = DenseMap<Operation *, SmallVector<StackAllocationOp>>;
} // namespace

/// Return the pointer element type of an allocation.
static Type getAllocType(StackAllocationOp alloc) {
  return alloc.getType().getElementType();
}

/// We can promote a stack allocation if all its uses are as the pointer to
/// loads and stores and no load or store crosses a region of an unknown
/// operation.
///
/// The walk to each store already visits exactly the control-flow nodes that
/// the store makes the allocation variant across, so on success `alloc` is
/// recorded in `nodeVariants` for each of them. This is why the promotability
/// check owns the variance bookkeeping: recomputing it per node would mean
/// re-testing every store against every node.
static bool canPromote(StackAllocationOp alloc, NodeVariantMap &nodeVariants) {
  // Only committed to `nodeVariants` once the allocation is known to be
  // promotable, since the checks below bail out on the first bad user.
  llvm::SmallSetVector<Operation *, 8> variantNodes;
  SmallVector<Operation *> enclosingNodes;

  for (Operation *user : alloc->getUsers()) {
    if (isa<StackAllocLifetimeStartOp, StackAllocLifetimeEndOp>(user))
      continue;
    if (auto load = dyn_cast<LoadOp>(user)) {
      if (userCrossesFunctionCFG(alloc, user) || load.mightBeVolatile())
        return false;
      continue;
    }

    if (isa<DebugInfo::ValueOp>(user)) {
      if (userCrossesFunctionCFG(alloc, user))
        return false;
      continue;
    }
    auto store = dyn_cast<StoreOp>(user);
    if (!store || store.getArg() == alloc || store.mightBeVolatile())
      return false;
    enclosingNodes.clear();
    if (userCrossesFunctionCFG(alloc, store, &enclosingNodes))
      return false;
    // Several stores can share an enclosing node; record each node once.
    variantNodes.insert_range(enclosingNodes);
  }

  for (Operation *node : variantNodes)
    nodeVariants[node].push_back(alloc);
  return true;
}

static ErrorOr<DebugInfo::DIExprAttr> mem2RegLeafConversion(Type irType) {
  auto ptrType = dyn_cast<PointerType>(irType);
  if (!ptrType)
    return Error("expected ir type to be a pointer type");

  Type elementType = ptrType.getElementType();
  auto newIrValue = DebugInfo::DIIRValueExprAttr::get(elementType);
  return DebugInfo::DIRefOfExprAttr::get(newIrValue, irType);
}

namespace {
/// Current state of a promoted stack allocation.
struct PromotedStackAlloc {
  /// The latest value that the alloc'd value has.
  Value currValue;
  /// The attributes that will be used to create DebugInfo ValueOps for the
  /// promoted StackAllocation. One alloc may be mapped to multiple source
  /// variables (each via a DebugInfo ValueOp) after inlining. This map allows
  /// tracking stores to them separately.
  struct DebugValue {
    DebugInfo::DILocalVariableAttr varInfo;
    DebugInfo::DIExprAttr conversionExpr;
  };
  DenseMap<DebugInfo::DISubprogramAttr, DebugValue> debugValues;

  /// The scope this binding belongs to. A binding inherited from an enclosing
  /// region has to be shadowed before it is mutated, rather than updated in
  /// place; see `PromotionState::forWrite`. This is purely an optimization, we
  /// could always insert a new binding, but that incurs additional allocations.
  llvm::ScopedHashTableScope<StackAllocationOp, PromotedStackAlloc> *owner =
      nullptr;

  /// Get the current value of this promoted stack allocation. If none exist
  /// yet, create an undef with the same type and return that.
  Value getCurrValueOrUndef(StackAllocationOp alloc, Operation *user) {
    if (LLVM_LIKELY(currValue))
      return currValue;
    // If the value is undefined, materialize an undef operation.
    OpBuilder builder(user);
    ParamConstantOp undefConst = ParamConstantOp::create(
        builder, user->getLoc(), UninitMemAttr::get(getAllocType(alloc)));
    // Create a DebugInfo ValueOp right after this undef.
    updateValue(undefConst, undefConst);
    return undefConst;
  }

  /// Register that a promoted StackAllocation needs DebugInfo.
  /// The caller pass in the existing DebugInfo ValueOp for the StackAllocation
  /// so that when future stores into this StackAllocation gets transformed, a
  /// DebugInfo ValueOp can be created at the previous store site.
  ErrorOrSuccess
  registerDebugValue(StackAllocationOp alloc, DebugInfo::ValueOp value,
                     DebugInfo::DIExprLeafReplacer &exprLeafReplacer) {
    ErrorOr<DebugInfo::DIExprAttr> newConversionExpr =
        exprLeafReplacer.apply(value.getConversionExprAttr());
    // Not enough source information available to track this transformation.
    // Cannot debug this local variable anymore.
    if (failed(newConversionExpr))
      return success();

    DebugInfo::DISubprogramAttr scope =
        extractPreInlineSubprogramScope(value->getLoc());
    if (!scope)
      return Error(
          "location of debug value does not contain a subprogram scope");
    debugValues[scope] = {value.getValueInfo(), newConversionExpr.get()};

    // Immediately create a ValueOp if currently not undef.
    if (currValue) {
      bool isUndef = false;
      if (auto cst = llvm::dyn_cast_if_present<ParamConstantOp>(
              currValue.getDefiningOp()))
        isUndef = isa<UninitMemAttr>(cst.getValue());

      if (!isUndef) {
        OpBuilder b(alloc->getContext());
        b.setInsertionPointAfter(value);
        DebugInfo::ValueOp::create(b, value->getLoc(), currValue,
                                   value.getValueInfo(),
                                   newConversionExpr.get());
      }
    }
    return success();
  }

  /// Update the current value of this promoted alloc.
  /// Also creates a new DebugInfo ValueOp with this new value if a DebugInfo
  /// ValueOp existed previously for this scope.
  void updateValue(Operation *mutator, Value newValue) {
    currValue = newValue;

    if (debugValues.empty())
      return;

    // Duplicate a DebugInfo::ValueOp for `newValue` if one existed before.
    // The new op is created after `op`.
    // Walk the series of inlined scopes of the mutator op from outermost caller
    // to innermost callee. For each scope where a variable is registered with
    // this value, create a DebugInfo::ValueOp for that variable.
    OpBuilder b(mutator->getContext());
    b.setInsertionPointAfter(mutator);

    // The current location corresponding to the depth of the walk.
    // Since the walk is from caller to callee, the CallSite tree created is
    // caller-side heavy.
    LocationAttr cumulativeLoc;
    // Update `cumulativeLoc` with a new callee location.
    auto appendCallee = [&cumulativeLoc](LocationAttr callee) {
      if (!cumulativeLoc) {
        cumulativeLoc = callee;
        return;
      }
      cumulativeLoc = mlir::CallSiteLoc::get(callee, cumulativeLoc);
    };

    DebugInfo::walkLocation(
        mutator->getLoc(), DebugInfo::LocWalkPolicy::CallerPriority,
        [&](Location loc) -> WalkResult {
          // Update the current cumulative location at each leaf location.
          if (isa<FileLineColLoc>(loc)) {
            appendCallee(loc);
          } else if (auto fused =
                         dyn_cast<mlir::FusedLocWith<DebugInfo::DIScopeAttr>>(
                             loc)) {
            appendCallee(loc);

            DebugInfo::DISubprogramAttr mutatorSubprogram =
                DebugInfo::getParentScopeOfType<DebugInfo::DISubprogramAttr>(
                    fused.getMetadata());
            if (auto it = debugValues.find(mutatorSubprogram);
                it != debugValues.end()) {
              DebugValue &dbgValue = it->second;
              DebugInfo::ValueOp::create(b, cumulativeLoc, newValue,
                                         dbgValue.varInfo,
                                         dbgValue.conversionExpr);
            }
            return WalkResult::skip();
          }
          return WalkResult::advance();
        });
  }

private:
  /// Get the innermost scope from a series of callsite locations.
  static DebugInfo::DISubprogramAttr
  extractPreInlineSubprogramScope(Location loc) {
    return DebugInfo::extractScopeFrom<DebugInfo::DISubprogramAttr>(
        loc, DebugInfo::LocWalkPolicy::CalleePriority);
  }
};
/// The promoted stack allocations visible at the current point of the walk,
/// along with their current values.
///
/// Each region opens a `Scope`, so a region inherits the values of the
/// allocations around it without copying them, and the bindings it adds
/// disappear when it closes.
class PromotionState {
  using Table = llvm::ScopedHashTable<StackAllocationOp, PromotedStackAlloc>;

public:
  /// Opens the state scope covering the walk of one region.
  ///
  /// The table cannot enumerate its keys, so the allocations promoted inside
  /// the region are also pushed onto a stack. Scopes close in reverse order of
  /// opening, which makes the entries above this scope's watermark exactly the
  /// allocations it promoted.
  class Scope {
  public:
    explicit Scope(PromotionState &state)
        : scope(state.table), state(state), watermark(state.promoted.size()) {}
    ~Scope() { state.promoted.truncate(watermark); }

    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;

    /// The allocations promoted within this scope.
    ArrayRef<StackAllocationOp> promotedAllocs() const {
      return ArrayRef<StackAllocationOp>(state.promoted).drop_front(watermark);
    }

  private:
    Table::ScopeTy scope;
    PromotionState &state;
    size_t watermark;
  };

  /// Return true if `alloc` has been promoted and is visible here.
  bool contains(StackAllocationOp alloc) const { return table.count(alloc); }

  /// Start tracking `alloc`, whose value is initially undefined.
  void promote(StackAllocationOp alloc) {
    insertOwned(alloc, PromotedStackAlloc());
    promoted.push_back(alloc);
  }

  /// Bind `alloc` to a value that only holds within the current scope, carrying
  /// over the debug values it has in the enclosing one.
  void bind(StackAllocationOp alloc, Value value) {
    PromotedStackAlloc bound;
    bound.currValue = value;
    bound.debugValues = find(alloc)->debugValues;
    insertOwned(alloc, bound);
  }

  /// Return the current value of `alloc`, materializing an undef if it does not
  /// have one yet.
  Value currValueOrUndef(StackAllocationOp alloc, Operation *user) {
    PromotedStackAlloc *entry = find(alloc);
    assert(entry && "expected the allocation to be promoted");
    // Reading an allocation that already has a value leaves the state alone, so
    // it does not need a binding of its own.
    if (LLVM_LIKELY(entry->currValue))
      return entry->currValue;
    return forWrite(alloc).getCurrValueOrUndef(alloc, user);
  }

  /// Return a binding of `alloc` that can be mutated, shadowing the one it was
  /// read from with a copy in the current scope.
  ///
  /// Mutations must not escape the region being walked: the undef constants and
  /// debug values they materialize are placed inside that region, so an
  /// enclosing region that picked them up would reference values that do not
  /// reach it. Shadowing leaves the enclosing region's value to be restored
  /// when the scope closes.
  PromotedStackAlloc &forWrite(StackAllocationOp alloc) {
    PromotedStackAlloc *entry = find(alloc);
    assert(entry && "expected the allocation to be promoted");
    // A binding this scope already owns can be mutated in place, which keeps
    // repeated writes to one allocation from each costing a binding.
    if (entry->owner == table.getCurScope())
      return *entry;
    return insertOwned(alloc, *entry);
  }

private:
  /// Return the innermost binding of `alloc`, or null if it is not promoted.
  PromotedStackAlloc *find(StackAllocationOp alloc) {
    Table::iterator it = table.begin(alloc);
    return it == table.end() ? nullptr : &*it;
  }

  /// Add a binding for `alloc` owned by the current scope, and return it.
  PromotedStackAlloc &insertOwned(StackAllocationOp alloc,
                                  PromotedStackAlloc value) {
    value.owner = table.getCurScope();
    table.insert(alloc, value);
    return *find(alloc);
  }

  Table table;
  SmallVector<StackAllocationOp> promoted;
};
} // namespace

static LogicalResult processRegion(
    Region &region, const HLCF::CFGAnalysis &cfg, PromotionState &state,
    DenseMap<HLCF::ControlFlowTerminator, ArrayRef<StackAllocationOp>>
        &termVariants,
    NodeVariantMap &nodeVariants,
    DebugInfo::DIExprLeafReplacer &exprLeafReplacer, PassStats &stats) {
  if (region.empty())
    return success();
  // This analysis only works on single-block regions.
  if (!llvm::hasSingleElement(region)) {
    return region.getParentOp()->emitError(
        "'mem-2-reg' can only be run on operations with all single block "
        "regions");
  }

  for (Operation &op : llvm::make_early_inc_range(region.front())) {
    if (auto alloc = dyn_cast<StackAllocationOp>(op)) {
      // If we can promote this stack allocation, initialize its state with an
      // undefined value.
      if (canPromote(alloc, nodeVariants))
        state.promote(alloc);
      continue;
    }
    if (auto load = dyn_cast<LoadOp>(op)) {
      if (load.mightBeVolatile())
        continue;
      // If we can elide this load, replace the result of the load with the last
      // value of the stack allocation.
      if (auto alloc = load.getPtr().getDefiningOp<StackAllocationOp>()) {
        if (state.contains(alloc)) {
          load.replaceAllUsesWith(state.currValueOrUndef(alloc, load));
          load.erase();
          ++stats.numLoadsElided;
        }
      }
      continue;
    }
    if (auto value = dyn_cast<DebugInfo::ValueOp>(op)) {
      // Delete stale debuginfo for the old stack allocation op.
      if (auto alloc = value.getValue().getDefiningOp<StackAllocationOp>()) {
        if (state.contains(alloc)) {
          auto newValue = state.forWrite(alloc).registerDebugValue(
              alloc, value, exprLeafReplacer);
          if (failed(newValue))
            return value.emitError() << newValue.getError();
          value.erase();
        }
      }
      continue;
    }
    if (isa<StackAllocLifetimeStartOp, StackAllocLifetimeEndOp>(op)) {
      llvm::BitVector eraseIndices(op.getNumOperands());
      for (auto [idx, value] : llvm::enumerate(op.getOperands())) {
        if (state.contains(value.getDefiningOp<StackAllocationOp>()))
          eraseIndices.set(idx);
      }
      op.eraseOperands(eraseIndices);
      continue;
    }
    if (auto store = dyn_cast<StoreOp>(op)) {
      // If we can elide this store, capture the last written value and erase
      // the operation.
      if (auto alloc = store.getPtr().getDefiningOp<StackAllocationOp>()) {
        if (state.contains(alloc)) {
          state.forWrite(alloc).updateValue(store, store.getArg());
          store.erase();
          ++stats.numStoresElided;
        }
      }
      continue;
    }
    if (auto term = dyn_cast<HLCF::ControlFlowTerminator>(op)) {
      // Look up the required variant values, if there are any.
      auto it = termVariants.find(term);
      if (it == termVariants.end() || it->second.empty())
        continue;
      // Bind the last values to the operands.
      SmallVector<Value> newOperands;
      for (StackAllocationOp alloc : it->second) {
        newOperands.push_back(state.currValueOrUndef(alloc, &op));
      }
      term.insertVariants(newOperands);
      continue;
    }

    // If this operation has regions, recurse into the regions.
    unsigned numRegions = op.getNumRegions();
    if (!numRegions)
      continue;

    auto node = dyn_cast<HLCF::ControlFlowNode>(op);
    if (!node) {
      // This is an unknown operation. Its regions share the current state, but
      // `canPromote` rejected any allocation used across an opaque boundary, so
      // only allocations defined inside them can be promoted there.
      for (Region &region : op.getRegions())
        if (failed(processRegion(region, cfg, state, termVariants, nodeVariants,
                                 exprLeafReplacer, stats)))
          return failure();
      continue;
    }

    // For control-flow operations, all current stack allocations are visible
    // within the regions. The variant ones were recorded when they were
    // promoted. Their values have to be carried through the regions using
    // iteration variables.
    //
    // The values move out of the map: `termVariants` holds a reference to them
    // across the recursion below, which inserts into `nodeVariants` and can
    // invalidate its storage. Dropping the entry also keeps this operation from
    // being found again once it is erased and its address is free to be reused.
    SmallVector<StackAllocationOp> variant;
    if (auto it = nodeVariants.find(&op); it != nodeVariants.end()) {
      variant = std::move(it->second);
      nodeVariants.erase(it);
    }

    // Map the required variant values to predecessor terminators of the end of
    // the operation and to each region.
    llvm::BitVector regionPreds(op.getNumRegions());
    llvm::BitVector parentPred(op.getNumRegions());
    if (!variant.empty()) {
      for (Operation *pred : cfg.predecessors.at({node, {}})) {
        if (auto term = dyn_cast<HLCF::ControlFlowTerminator>(pred))
          termVariants.try_emplace(term, variant);
      }

      for (Region &region : op.getRegions()) {
        ArrayRef<Operation *> preds =
            cfg.predecessors.at({node, region.getRegionNumber()});
        for (Operation *pred : preds) {
          if (auto term = dyn_cast<HLCF::ControlFlowTerminator>(pred)) {
            termVariants.try_emplace(term, variant);
            regionPreds.set(region.getRegionNumber());
          } else {
            assert(pred == &op);
            parentPred.set(region.getRegionNumber());
          }
        }
      }
    }

    // For each region with region predecessors (demarcated by a terminator)
    // and variant allocations, introduce block arguments.
    bool parentHasInit = false;
    for (Region &region : op.getRegions()) {
      // Determine if there are any region predecessors.
      bool hasRegionPreds =
          !variant.empty() && regionPreds[region.getRegionNumber()];

      // These operations mutate the node itself, so they run before the
      // region's scope opens: the initializer values are materialized in the
      // enclosing region and have to bind to its state, not the region's.
      SmallVector<Value> blockArgs;
      if (hasRegionPreds) {
        for (auto [i, alloc] : llvm::enumerate(variant)) {
          Type allocType = getAllocType(alloc);
          blockArgs.push_back(
              node.insertArgumentToRegion(op.getLoc(), allocType, i, region));
        }
        // If one of the predecessors is the parent operation, we need to
        // add initializer operands to it if this hasn't already been done.
        if (!parentHasInit && parentPred[region.getRegionNumber()]) {
          parentHasInit = true;
          SmallVector<Value> initOperands;
          for (StackAllocationOp alloc : variant) {
            initOperands.push_back(state.currValueOrUndef(alloc, &op));
          }

          node.insertVariants(initOperands);
        }
      }

      PromotionState::Scope regionScope(state);

      // Bind the block arguments to the values of the variant allocations.
      if (hasRegionPreds) {
        for (auto [alloc, blockArg] : llvm::zip(variant, blockArgs))
          state.bind(alloc, blockArg);
      }

      // Okay, now recurse into the region.
      if (failed(processRegion(region, cfg, state, termVariants, nodeVariants,
                               exprLeafReplacer, stats)))
        return failure();

      // Erase elided allocations in the nested region.
      for (StackAllocationOp alloc : regionScope.promotedAllocs()) {
        alloc.erase();
        ++stats.numAllocsElided;
      }
    }

    // After processing the regions, we need to add results to the operation
    // to merge the values of variant allocations, and then bind those as the
    // current values of those allocations.
    if (!variant.empty()) {
      if (!op.hasTrait<OpTrait::VariadicResults>()) {
        return op.emitOpError(
            "must have trailing variadic results to be used in 'mem-2-reg'");
      }
      SmallVector<Type> newTypes = llvm::to_vector(op.getResultTypes());
      for (StackAllocationOp alloc : variant)
        newTypes.push_back(getAllocType(alloc));
      Operation *newOp =
          Operation::create(op.getLoc(), op.getName(), newTypes,
                            op.getOperands(), op.getDiscardableAttrDictionary(),
                            op.getPropertiesStorage(), {}, op.getNumRegions());
      OpBuilder(&op).insert(newOp);
      for (unsigned i = 0, e = op.getNumRegions(); i != e; ++i)
        newOp->getRegion(i).takeBody(op.getRegion(i));
      unsigned iterStart = op.getNumResults();
      for (auto [i, alloc] : llvm::enumerate(variant)) {
        // Update currValue without creating a new debug value, since the
        // mutator inside the nested scope will have noted when the value was
        // updated.
        state.forWrite(alloc).currValue = newOp->getResult(iterStart + i);
      }
      op.replaceAllUsesWith(newOp->getResults().slice(0, iterStart));
      op.erase();
    }
  }

  return success();
}

namespace {
struct Mem2RegPass : public M::KGEN::impl::Mem2RegBase<Mem2RegPass> {
  void runOnOperation() override;
};
} // namespace

void Mem2RegPass::runOnOperation() {
  auto &cfg = getAnalysis<HLCF::CFGAnalysis>();
  PassStats stats;
  PromotionState state;
  DenseMap<HLCF::ControlFlowTerminator, ArrayRef<StackAllocationOp>>
      termVariants;
  NodeVariantMap nodeVariants;
  for (Region &region : getOperation()->getRegions()) {
    // Reuse the same memory for the maps each time.
    termVariants.clear();
    nodeVariants.clear();
    DebugInfo::DIExprLeafReplacer exprLeafReplacer(mem2RegLeafConversion);
    // The scope has to close before the state is reused for the next region.
    PromotionState::Scope entryScope(state);
    if (failed(processRegion(region, cfg, state, termVariants, nodeVariants,
                             exprLeafReplacer, stats)))
      return signalPassFailure();
    // Erase elided allocations.
    for (StackAllocationOp alloc : entryScope.promotedAllocs()) {
      alloc.erase();
      ++stats.numAllocsElided;
    }
  }

  numAllocsElided = stats.numAllocsElided;
  numLoadsElided = stats.numLoadsElided;
  numStoresElided = stats.numStoresElided;

  // Control-flow is not modified.
  markAnalysesPreserved<HLCF::CFGAnalysis>();
}
