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

#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Mojo/ToolCommon/CLOptions.h"
#include "Mojo/ToolCommon/Debug.h"
#include "Mojo/TransformUtils/ControlFlowUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"

#define KGEN_DEBUG_TYPE "stack-reuse"

using namespace M;
using namespace KGEN;
using namespace POP;

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_STACKREUSE
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct StackReuse : public impl::StackReuseBase<StackReuse> {
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// PotentialValue
//===----------------------------------------------------------------------===//

/// The state of a stack allocation is represented as a "potential value". That
/// is, the value of a stack allocation is either known to be equal to an SSA
/// value or some "opaque" value.
///
/// For example:
///
/// ```mlir
/// %s0 = pop.stack_allocation
/// pop.store %arg0, %s0
/// ```
///
/// The potential value of `%s0` will be `%arg0`. However, the value can be
/// opaque for two reasons:
///
/// 1. When a store to a view of the stack allocation occurs, or
/// 2. When entering a region where the stack allocation could be variant.
///
/// For example:
///
/// ```mlir
/// %s0 = pop.stack_allocation
/// %s1 = pop.stack_allocation
///
/// %gep = kgen.struct.gep %s0[2]
/// pop.store %arg0, %gep
///
/// %0 = pop.load %s0
/// pop.store %0, %s1
/// ```
///
/// In this example, we don't have an exact SSA value which we know the value
/// of `s0` is. However, we do know that `%s1` will have the same value of `%s0`
/// after the final store. This opaque but comparable value is represented using
/// an integer.
///
/// The other case can occur during control flow:
///
/// ```mlir
/// %s0 = pop.stack_allocation
/// %s1 = pop.stack_allocation
/// if %cond {
///   pop.store %arg0, %s0
/// } else {
///   pop.store %arg1, %s0
/// }
/// %0 = pop.load %s0
/// pop.store %0, %s1
/// ```
///
/// Once again, we don't know the exact value of `%s0` after the join of the
/// branch, but we do know that its value is equal to `%s1`.
///
/// Note that this pass is not at maximum strength because the effort has not
/// been made to implement additional analyses:
///
/// 1. Upon crossing any basic block, e.g. when entering a region or resuming
///    after the parent operation, all variant stack allocations in any region
///    of the operation are conservatively assigned a new opaque value.
/// 2. Storing to a view of a stack allocation is not field-sensitive. The
///    entire stack allocation is assumed to take on a whole new opaque value.
///
/// Each of these cases could be developed if necessary.
///
/// The analysis also performs copy elision by tracking the values of loads and
/// propagating them to other stack allocations via stores. This is seen in
/// previous examples where a `pop.load` followed by a `pop.store` into another
/// stack allocation propagates the value of the former stack allocation into
/// the latter.
using PotentialValue = SmartVariant<Value, unsigned>;

namespace llvm {
/// Allow `PotentialValue` to be used as a key in a hash map.
template <>
struct DenseMapInfo<PotentialValue> {
  static unsigned getHashValue(PotentialValue value) {
    if (isa<Value>(value))
      return DenseMapInfo<Value>::getHashValue(cast<Value>(value));
    return DenseMapInfo<unsigned>::getHashValue(cast<unsigned>(value));
  }
  static bool isEqual(PotentialValue lhs, PotentialValue rhs) {
    if (isa<Value>(lhs))
      return isa<Value>(rhs) && cast<Value>(lhs) == cast<Value>(rhs);
    return isa<unsigned>(rhs) && cast<unsigned>(lhs) == cast<unsigned>(rhs);
  }
};
} // namespace llvm

namespace {
struct PassInfo {
  mlir::DominanceInfo &domInfo;
  TargetInfoAttr target;
  unsigned numErasedOps = 0;
  unsigned numElidedVars = 0;
  unsigned numOptimizedConstants = 0;
};
} // namespace

/// The stack reuse optimization operates over single function regions. This is
/// the entry point into the optimization pass that should be invoked for every
/// function body.
static void runStackReuseOnRegion(Region &funcBody, PassInfo &pass);

/// This is an analysis pass over the function body that determines:
///
/// 1. The stack allocations that are eligible to be elided. These are stack
///    allocations whose projection (the stack-allocated and all views over it)
///    does not "escape".
/// 2. Alias analysis that maps pointers that represent a view of a stack
///    allocation back to that allocation.
/// 3. For each operation, all stack allocations that are variant within any
///    region of the operation.
///
static std::vector<StackAllocationOp>
runAnalysis(Region &top, PassInfo &pass,
            DenseMap<Value, StackAllocationOp> &aliases,
            DenseMap<Operation *, std::vector<StackAllocationOp>> &variant) {
  VerboseCompilerTimeTraceScope traceScope("runAnalysis");
  // These are the stack allocations that are eligible for elision by this pass.
  std::vector<StackAllocationOp> allocs;

  top.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    // Visit all stack allocations.
    auto alloc = dyn_cast<StackAllocationOp>(op);
    if (!alloc) {
      if (!op->getNumRegions() || isa<HLCF::ControlFlowNode>(op))
        return WalkResult::advance();
      // Skip over nested regions that aren't part of the CFG in this analysis,
      // but recurse the entire pass.
      for (Region &region : op->getRegions())
        runStackReuseOnRegion(region, pass);
      return WalkResult::skip();
    }

    // This set contains the parent operations of all users of the projection of
    // the stack allocation.
    DenseSet<Operation *> touchedParents;

    // Check the whole projection by iterating over the users.
    std::vector<Operation *> toCheck;
    toCheck.push_back(alloc);
    while (!toCheck.empty()) {
      Operation *op = toCheck.back();
      toCheck.pop_back();
      // Alias the projected pointer to the alloc, even for itself.
      Value ptr = op->getResult(0);
      aliases.try_emplace(ptr, alloc);

      for (Operation *user : ptr.getUsers()) {
        if (auto store = dyn_cast<StoreOp>(user)) {
          // If the pointer is the argument of a store or the store leaves the
          // CFG, then it escapes.
          if (store.getArg() == ptr || userCrossesFunctionCFG(op, user) ||
              store.mightBeVolatile()) {
            KGEN_DEBUG(0, {
              llvm::dbgs() << KGEN_DEBUG_TYPE << ": ";
              alloc.print(llvm::dbgs());
              llvm::dbgs() << ": escapes at ";
              user->print(llvm::dbgs());
              llvm::dbgs() << '\n';
            });
            return WalkResult::advance();
          }
          // Indicate that this stack allocation was modified in this parent
          // operation.
          touchedParents.insert(user->getParentOp());
          // Stores are terminals.
          continue;
        }
        if (auto loadOp = dyn_cast<LoadOp>(user)) {
          // If the load leaves the CFG, then it escapes.
          if (userCrossesFunctionCFG(op, user) || loadOp.mightBeVolatile()) {
            KGEN_DEBUG(0, {
              llvm::dbgs() << KGEN_DEBUG_TYPE << ": ";
              alloc.print(llvm::dbgs());
              llvm::dbgs() << ": escapes at ";
              user->print(llvm::dbgs());
              llvm::dbgs() << '\n';
            });
            return WalkResult::advance();
          }
          // Loads are terminals.
          continue;
        }
        if (isa<ArrayGEPOp, StructGEPOp, OffsetOp, PointerBitcastOp>(user)) {
          // Recurse on the view.
          toCheck.push_back(user);
          continue;
        }
        if (isa<StackAllocLifetimeStartOp, StackAllocLifetimeEndOp>(user)) {
          // Both operations don't make the pointer escape and the proper
          // handling of validity of the next optimizations should be done
          // later.
          continue;
        }
        // Any other using operation conservatively is an escape.
        KGEN_DEBUG(0, {
          llvm::dbgs() << KGEN_DEBUG_TYPE << ": ";
          alloc.print(llvm::dbgs());
          llvm::dbgs() << ": escapes at ";
          user->print(llvm::dbgs());
          llvm::dbgs() << '\n';
        });
        return WalkResult::advance();
      }
    }
    KGEN_DEBUG(0, {
      llvm::dbgs() << KGEN_DEBUG_TYPE << ": ";
      alloc.print(llvm::dbgs());
      llvm::dbgs() << " doesn't escape\n";
    });
    // The projection of the pointer does not escape.
    allocs.push_back(alloc);

    // Mark the allocation as variant in all ancestor operations to all its
    // users besides its own parent operation.
    DenseSet<Operation *> allParentOps;
    Operation *ownParentOp = alloc->getParentOp();
    for (Operation *parent : touchedParents)
      for (; parent != ownParentOp; parent = parent->getParentOp())
        allParentOps.insert(parent);
    // The stack allocation is marked as variant in the union of all these ops.
    for (Operation *parent : allParentOps)
      variant[parent].push_back(alloc);

    return WalkResult::advance();
  });

  return allocs;
}

/// Given a stack allocation and a set of stack allocations that contain the
/// same value (including itself), return the stack allocation that most
/// dominates the current one, which may be itself.
static StackAllocationOp
mostDominatingAlloc(mlir::DominanceInfo &domInfo, StackAllocationOp alloc,
                    const DenseSet<StackAllocationOp> &others) {
  VerboseCompilerTimeTraceScope traceScope("mostDominatingAlloc");
  for (StackAllocationOp other : others) {
    if (domInfo.properlyDominates(&*other, alloc)) {
      auto lifetimeEnd = find_if(other->getUsers(), [&](Operation *user) {
        return isa<StackAllocLifetimeEndOp>(user);
      });
      // if the scope of the 'other' allocation ends before 'alloc' scope
      // starts, then 'other' is assumed to be non-dominating.
      if (lifetimeEnd != other->user_end() &&
          domInfo.properlyDominates(*lifetimeEnd, alloc))
        continue;
      alloc = other;
    }
  }
  return alloc;
}

/// Give each of the variant stack allocations a new opaque potential value.
static void markVariantPvsOpaque(
    ArrayRef<StackAllocationOp> variant,
    DenseMap<StackAllocationOp, PotentialValue> &pvs,
    DenseMap<PotentialValue, DenseSet<StackAllocationOp>> &rmap,
    unsigned &opaqueCounter) {
  VerboseCompilerTimeTraceScope traceScope("markAllPvsOpaque");
  for (StackAllocationOp alloc : variant) {
    auto it = pvs.find(alloc);
    rmap[it->second].erase(alloc);
    it->second = ++opaqueCounter;
    rmap[it->second].insert(alloc);
  }
}

/// Determine which stack allocations can be elided. These are stack allocations
/// for which at all points the projection of the pointer are accessed, it
/// contains a value possesed by a more dominating stack allocation.
///
/// FIXME: That's a lot of hash maps! There has got to be a more efficient way
/// to capture the information.
static void scanRegion(
    Region &region, DenseMap<StackAllocationOp, PotentialValue> &pvs,
    DenseMap<LoadOp, PotentialValue> &loadValues,
    DenseMap<PotentialValue, DenseSet<StackAllocationOp>> &rmap,
    DenseMap<Value, StackAllocationOp> &aliases,
    DenseMap<Operation *, std::vector<StackAllocationOp>> &regionVariants,
    DenseMap<StackAllocationOp, SmallPtrSet<Operation *, 1>> &canElide,
    mlir::DominanceInfo &domInfo, unsigned &opaqueCounter) {
  // Visit the operations in program order.
  for (Operation &op : llvm::make_early_inc_range(region.front())) {
    // For loads, check if it is reading a projection of a stack allocation, and
    // if so, whether at this point there is a more dominating stack allocation
    // with the same value. If not, then the stack allocation in question cannot
    // be elided.
    if (auto load = dyn_cast<LoadOp>(op)) {
      auto it = aliases.find(load.getPtr());
      if (it == aliases.end()) // load from an untracked pointer
        continue;

      // Retrieve the current value of the projected pointer.
      StackAllocationOp alloc = it->second;
      auto pvIt = pvs.find(alloc);
      if (pvIt == pvs.end()) // load from escaped stack allocation
        continue;

      // If no other stack allocation has the same value at this point, it
      // cannot be elided.
      PotentialValue value = pvIt->second;
      StackAllocationOp maybeReuse =
          mostDominatingAlloc(domInfo, alloc, rmap.at(value));
      // Alloc could previously be removed from the set
      if (maybeReuse == alloc || !canElide.contains(alloc))
        canElide.erase(alloc);
      else
        canElide.find(alloc)->second.insert(maybeReuse);

      // Save the value of the load for copy elision.
      // TODO(#22921): Support copy elision on aliases.
      if (alloc.getResult() == load.getPtr())
        loadValues.try_emplace(load, value);
      continue;
    }

    // For stores, check if it stores to a projection of a stack allocation. If
    // so, update the new potential value of the stack allocation. If storing
    // directly to the stack allocation, then the whole SSA value can be used.
    // If the store argument is the result of a load, then also attempt to
    // see through copies by propagating a known value.
    if (auto store = dyn_cast<StoreOp>(op)) {
      auto it = aliases.find(store.getPtr());
      if (it == aliases.end()) // store to an untracked pointer
        continue;

      StackAllocationOp alloc = it->second;
      auto pvIt = pvs.find(alloc);
      if (pvIt == pvs.end()) // store to an escaped stack allocation
        continue;

      // The value of the stack allocation is about to change. Remove it from
      // the set for its current value.
      rmap[pvIt->second].erase(alloc);

      // If this is a store to a view of a stack allocation, it takes on a new
      // opaque value.
      if (store.getPtr() != alloc.getResult()) {
        pvIt->second = ++opaqueCounter;
      } else if (auto load = store.getArg().getDefiningOp<LoadOp>()) {
        // Check for a copy. See through it by mapping the originally loaded
        // value.
        auto loadIt = loadValues.find(load);
        if (loadIt != loadValues.end())
          pvIt->second = loadIt->second;
        else
          pvIt->second = store.getArg();
      } else {
        pvIt->second = store.getArg();
      }

      // Map the allocation to its new value.
      rmap[pvIt->second].insert(alloc);
      continue;
    }

    // If the operation has regions, it must be a control-flow operation.
    if (!op.getNumRegions() || !isa<HLCF::ControlFlowNode>(op))
      continue;

    // Don't do anything too fancy here. Mark all stack allocations variant in
    // the operation as having new opaque values when recursing into the
    // regions. Stronger control-flow analysis can be added if necessary.
    ArrayRef<StackAllocationOp> variant = regionVariants[&op];
    for (Region &region : op.getRegions()) {
      markVariantPvsOpaque(variant, pvs, rmap, opaqueCounter);
      scanRegion(region, pvs, loadValues, rmap, aliases, regionVariants,
                 canElide, domInfo, opaqueCounter);
    }
    // Do it again coming out of the operation.
    markVariantPvsOpaque(variant, pvs, rmap, opaqueCounter);
  }
}

/// Run the same analysis again, except this time we know which allocations can
/// be elided from the start. Elide loads as they appear.
static void processRegion(
    Region &region, DenseMap<StackAllocationOp, PotentialValue> &pvs,
    DenseMap<LoadOp, PotentialValue> &loadValues,
    DenseMap<PotentialValue, DenseSet<StackAllocationOp>> &rmap,
    DenseMap<Value, StackAllocationOp> &aliases,
    DenseMap<Operation *, std::vector<StackAllocationOp>> &regionVariants,
    DenseMap<StackAllocationOp, SmallPtrSet<Operation *, 1>> &canElide,
    mlir::DominanceInfo &domInfo, unsigned &opaqueCounter) {
  for (Operation &op : llvm::make_early_inc_range(region.front())) {
    if (auto load = dyn_cast<LoadOp>(op)) {
      auto it = aliases.find(load.getPtr());
      if (it == aliases.end()) // load from an untracked pointer
        continue;

      // Retrieve the current value of the projected pointer.
      StackAllocationOp alloc = it->second;
      auto pvIt = pvs.find(alloc);
      if (pvIt == pvs.end()) // load from escaped stack allocation
        continue;
      PotentialValue value = pvIt->second;

      // If the alloc cannot be elided, move on.
      if (!canElide.contains(alloc)) {
        // Save the value of the load for copy elision.
        // TODO(#22921): Support copy elision on aliases.
        if (alloc.getResult() == load.getPtr())
          loadValues.try_emplace(load, value);
        continue;
      }

      // We can definitely elide the load this time around.
      StackAllocationOp reuse =
          mostDominatingAlloc(domInfo, alloc, rmap.at(value));
      assert(reuse != alloc && "was supposed to be elidable");

      // Elide the load. Generate the accesses up until this point. We can't
      // just replace the use of the stack allocation because the point at
      // which the GEP/offset occurs could have a different value for the
      // alloc than at the load.
      ImplicitLocOpBuilder b(load.getLoc(), OpBuilder(load));
      OpOperand *operand = &load->getOpOperand(0);
      for (Operation *defOp = operand->get().getDefiningOp(); defOp != alloc;
           defOp = operand->get().getDefiningOp()) {
        Operation *newOp = b.clone(*defOp);
        // Back up one op.
        b.setInsertionPoint(newOp);
        operand->set(newOp->getResult(0));
        operand = &newOp->getOpOperand(0);
      }
      // Okay, we have cloned the chain of accesses from the original stack
      // allocation op. Replace the base pointer.
      operand->set(reuse);

      // Save the value of the load for copy elision.
      // TODO(#22921): Support copy elision on aliases.
      if (alloc.getResult() == load.getPtr())
        loadValues.try_emplace(load, value);
      continue;
    }

    if (auto store = dyn_cast<StoreOp>(op)) {
      auto it = aliases.find(store.getPtr());
      if (it == aliases.end()) // store to an untracked pointer
        continue;

      StackAllocationOp alloc = it->second;
      auto pvIt = pvs.find(alloc);
      if (pvIt == pvs.end()) // store to an escaped stack allocation
        continue;

      // The value of the stack allocation is about to change.
      rmap[pvIt->second].erase(alloc);

      // If this is a store to a view of a stack allocation, it takes on a new
      // opaque value.
      if (store.getPtr() != alloc.getResult()) {
        pvIt->second = ++opaqueCounter;
      } else if (auto load = store.getArg().getDefiningOp<LoadOp>()) {
        // Check for a copy. See through it by mapping the originally loaded
        // value.
        auto loadIt = loadValues.find(load);
        if (loadIt != loadValues.end())
          pvIt->second = loadIt->second;
        else
          pvIt->second = store.getArg();
      } else {
        pvIt->second = store.getArg();
      }

      // Map the allocation to its new value.
      rmap[pvIt->second].insert(alloc);
      continue;
    }

    // If the operation has regions, it must be a control-flow operation.
    if (!op.getNumRegions() || !isa<HLCF::ControlFlowNode>(op))
      continue;

    // Don't do anything too fancy here. Mark all stack allocations as having
    // new opaque values when recursing into the regions. Stronger control-flow
    // analysis can be added if necessary.
    ArrayRef<StackAllocationOp> variant = regionVariants[&op];
    for (Region &region : op.getRegions()) {
      markVariantPvsOpaque(variant, pvs, rmap, opaqueCounter);
      processRegion(region, pvs, loadValues, rmap, aliases, regionVariants,
                    canElide, domInfo, opaqueCounter);
    }
    // Do it again coming out of the operation.
    markVariantPvsOpaque(variant, pvs, rmap, opaqueCounter);
  }
}

// Extra peephole optimization to promote read-only stack allocations (with a
// single write from a KGEN::ParamConstantOp) to a global constant
//
// TODO: Even though LLVM has StackColoring that combines many same stack
// allocations into single one, it makes sense to move it here to reduce code
// size sent to LLVM.
//
// TODO: We can also have just one stack allocation:
//  - If stack allocations are not read-only, but execution can only reach
//    one region
//  - If stack allocations are not read-only, but store into the same index
//  - If all, but post-dominate stack allocation are read-only
static void optimizeReadOnlyMemory(Region &funcBody, PassInfo &pass) {
  // Since now we use target to get size of a constant, exit Early if it cannot
  // be identified.
  if (!pass.target)
    return;

  // Return true if type can be lowered to pop.global_constant
  std::function<bool(Type)> isSupportedType = [&isSupportedType](Type type) {
    if (!type)
      return false;
    if (isa<IntegerType, FloatType>(type))
      return true;
    if (auto simd = dyn_cast<SIMDType>(type))
      return simd.isScalar();
    if (auto structTy = dyn_cast<StructType>(type)) {
      if (!structTy.getIsParamPack()) {
        auto elementTypes = structTy.getElementTypes();
        return elementTypes && llvm::all_of(*elementTypes, isSupportedType);
      }
    }
    return false;
  };

  DenseMap<Value, StackAllocationOp> aliases;
  DenseMap<Operation *, std::vector<StackAllocationOp>> regionVariants;
  std::vector<StackAllocationOp> allocs =
      runAnalysis(funcBody, pass, aliases, regionVariants);
  DenseMap<KGEN::ParamConstantOp, llvm::SetVector<StackAllocationOp>>
      candidates;

  // Collect all constants that can be optimized
  funcBody.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    auto constant = dyn_cast<KGEN::ParamConstantOp>(op);
    if (!constant)
      return WalkResult::advance();

    auto popArrayType = dyn_cast<POP::ArrayType>(constant.getType());
    if (!popArrayType)
      return WalkResult::advance();

    std::optional<int64_t> constantSize = popArrayType.getTypeSize(pass.target);
    // Don't try to optimize the memory if constant's size is smaller than
    // expected threshold.
    if (!constantSize ||
        (size_t)*constantSize <
            *KGENPassCLOptions::stackReusePromoteToGlobalThreshold()) {
      KGEN_DEBUG(0, {
        llvm::dbgs() << KGEN_DEBUG_TYPE << ": ";
        constant.print(llvm::dbgs());
        llvm::dbgs()
            << " constant is too small to be optimized to global constant\n";
      });
      return WalkResult::advance();
    }

    // Non-simple type, such as string, requires to allocate memory for each
    // element and use their pointers to access them, which GlobalConstantOp
    // does not support.
    auto eltType = popArrayType.getElementType();
    if (!eltType || !isSupportedType(eltType)) {
      KGEN_DEBUG(0, {
        llvm::dbgs() << KGEN_DEBUG_TYPE << ": ";
        constant.print(llvm::dbgs());
        llvm::dbgs() << " constant's element type is not supported (size = "
                     << *constantSize << ")\n";
      });
      return WalkResult::advance();
    }

    for (Operation *user : constant->getUsers()) {
      if (auto store = dyn_cast<StoreOp>(user)) {
        assert(store.getArg() == constant &&
               "Unexpected use of a constant in StoreOp");
        auto alloc = store.getPtr().getDefiningOp<StackAllocationOp>();
        // Store must be done into a non-escaping stack allocation.
        if (!alloc || !aliases.contains(alloc->getResult(0))) {
          KGEN_DEBUG(
              0, {
                llvm::dbgs() << KGEN_DEBUG_TYPE << ": ";
                constant.print(llvm::dbgs());
                llvm::dbgs()
                    << " constant is stored into escaping stack allocation ";
                alloc.print(llvm::dbgs());
                llvm::dbgs() << '\n';
              });
          return WalkResult::advance();
        }

        // No other store should be performed into that stack allocation, except
        // for the store of the constant.
        if (any_of(alloc->getUsers(), [&](Operation *user) {
              return isa<StoreOp>(user) && user != store;
            })) {
          KGEN_DEBUG(0, {
            llvm::dbgs() << KGEN_DEBUG_TYPE << ": ";
            constant.print(llvm::dbgs());
            llvm::dbgs() << " constant is stored into stack allocation with "
                            "other stores: ";
            alloc.print(llvm::dbgs());
            llvm::dbgs() << '\n';
          });
          return WalkResult::advance();
        }

        KGEN_DEBUG(0, {
          llvm::dbgs() << KGEN_DEBUG_TYPE << ": ";
          constant.print(llvm::dbgs());
          llvm::dbgs() << " constant can be optimized to global constant\n";
        });
        candidates[constant].insert(cast<StackAllocationOp>(alloc));
      }
    }
    return WalkResult::advance();
  });

  DenseSet<Operation *> toErase;
  // Optimize all found constants:
  //  - replace the constant with the global constant followed by the load from
  //    it
  //  - remove all stack allocation lifemarks, stores and replace uses of the
  //    stack allocation with the load.
  for (auto &[constant, allocs] : candidates) {
    pass.numOptimizedConstants += allocs.size();

    ImplicitLocOpBuilder b(constant.getLoc(), OpBuilder(constant));
    b.setInsertionPointAfter(constant);
    auto ptr = GlobalConstantOp::create(b, constant.getValue());

    for (StackAllocationOp alloc : allocs) {
      for (Operation *user : alloc->getUsers()) {
        // Need to drop lifetime markers as they are no longer relevant.
        if (isa<StackAllocLifetimeStartOp, StackAllocLifetimeEndOp>(user)) {
          toErase.insert(user);
          continue;
        }
        // Store of the constant can be dropped as it's done after new
        // allocation.
        if (auto store = dyn_cast<StoreOp>(user);
            store && store.getArg() == constant) {
          toErase.insert(user);
          continue;
        }
      }
      alloc->replaceAllUsesWith(ptr);
      toErase.insert(alloc);
    }
    toErase.insert(constant);
  }

  for (Operation *op : toErase) {
    op->dropAllUses();
    op->erase();
  }
}

static void runStackReuseOnRegion(Region &funcBody, PassInfo &pass) {
  VerboseCompilerTimeTraceScope traceScope("runStackReuseOnRegion");

  // Determine the eligible stack allocations, the region variant allocations,
  // and the aliases.
  DenseMap<Value, StackAllocationOp> aliases;
  DenseMap<Operation *, std::vector<StackAllocationOp>> regionVariants;
  std::vector<StackAllocationOp> allocs =
      runAnalysis(funcBody, pass, aliases, regionVariants);

  DenseMap<StackAllocationOp, PotentialValue> pvs;
  DenseMap<PotentialValue, DenseSet<StackAllocationOp>> rmap;
  DenseMap<LoadOp, PotentialValue> loadValues;
  DenseMap<StackAllocationOp, SmallPtrSet<Operation *, 1>> canElide;
  unsigned opaqueCounter = 0;

  // Run the first pass.
  {
    VerboseCompilerTimeTraceScope traceScope("scanRegion");
    for (StackAllocationOp alloc : allocs) {
      // Set the current value to uninitialized.
      pvs.try_emplace(alloc, Value());
      // This allocation is now one that has the uninitialized value.
      rmap[Value()].insert(alloc);
      // Alias the stack allocation result to itself.
      aliases.try_emplace(alloc.getResult(), alloc);
      // Assume it can be elided.
      canElide.insert({alloc, {}});
    }
    scanRegion(funcBody, pvs, loadValues, rmap, aliases, regionVariants,
               canElide, pass.domInfo, opaqueCounter);
  }

  {
    VerboseCompilerTimeTraceScope traceScope("cantElide");
    // `canElide` now contains the set of stack allocations that *could* be
    // elided map to the stack allocations that would be used to elide it. This
    // forms a kind of DAG where only the roots can actually be elided. Because
    // the most dominated stack allocations are used, removing a stack
    // allocation will not cause more to be removable -- the max depth of the
    // DAG is 2 and we don't need to iterate.
    DenseSet<Operation *> cantElide;
    for (auto &[alloc, used] : canElide)
      cantElide.insert(used.begin(), used.end());
    for (Operation *alloc : cantElide)
      canElide.erase(cast<StackAllocationOp>(alloc));
  }

  // Reset state.
  pvs.clear();
  loadValues.clear();
  rmap.clear();
  opaqueCounter = 0;

  // Now process the regions.
  {
    VerboseCompilerTimeTraceScope traceScope("processRegion");
    for (StackAllocationOp alloc : allocs) {
      // Set the current value to uninitialized.
      pvs.try_emplace(alloc, Value());
      // This allocation is now one that has the uninitialized value.
      rmap[Value()].insert(alloc);
    }
    processRegion(funcBody, pvs, loadValues, rmap, aliases, regionVariants,
                  canElide, pass.domInfo, opaqueCounter);
  }

  // Now start deleting ops.
  VerboseCompilerTimeTraceScope deleteScope("deleteOps");
  std::vector<Operation *> worklist;
  llvm::append_range(worklist, llvm::make_first_range(canElide));
  pass.numElidedVars += canElide.size();
  unsigned numErasedOps = 0;
  while (!worklist.empty()) {
    Operation *op = worklist.back();
    worklist.pop_back();
    // The users of this operation must be unique. None of the ops this pass
    // covers can use the same pointer more than once, so appending is safe.
    llvm::append_range(worklist, op->getUsers());
    op->dropAllUses();
    op->erase();
    ++numErasedOps;
  }
  pass.numErasedOps += numErasedOps;

  optimizeReadOnlyMemory(funcBody, pass);
}

void StackReuse::runOnOperation() {
  VerboseCompilerTimeTraceScope traceScope("StackReuse::runOnOperation");

  FuncOp func = getOperation();
  auto &domInfo = getAnalysis<mlir::DominanceInfo>();
  PassInfo info{domInfo, lookupTargetInfo(func)};

  runStackReuseOnRegion(func.getBodyRegion(), info);
}
