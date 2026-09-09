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
// It is common for functions to end up with a lot of parameters that are unused
// in the function body.  This can happen when they are defined on a highly
// parameterized struct, for example, because the function will get all of the
// parameters from that struct.
//
// This pass scans the IR and removes these unused parameters (and also function
// arguments) to reduce burden on the elaborator.  This reduces the amount of
// function clones produced, which reduces compile time and code size.
//
//===----------------------------------------------------------------------===//

#include "Mojo/ToolCommon/KGENPasses.h"

#include "Mojo/HLCFDialect/Analysis/CFG.h"
#include "Mojo/Interpreter/InterpreterAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/KGENDialect/ParameterEvaluator.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"

using namespace M;
using namespace KGEN;
using namespace POP;

namespace M::KGEN {
#define GEN_PASS_DEF_REMOVEUNUSEDPARAMS
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
class RemoveUnusedParams
    : public impl::RemoveUnusedParamsBase<RemoveUnusedParams> {
public:
  using RemoveUnusedParamsBase::RemoveUnusedParamsBase;
  void runOnOperation() override;

private:
  std::optional<SymTabEvaluationContext> evaluationContext;

  // Memory cache of data structures we can reuse without reallocating.
  llvm::BitVector unusedArgs, unusedParamsIndex;
  llvm::SmallPtrSet<StringAttr, 8> unusedParamsAttr;

  SmallVector<ParamDeclAttr> inputParams;
  SmallVector<TypedAttr> newParams;
  SmallVector<Value> newOperands;
  SmallVector<Type> inputTypes;
  SmallVector<ArgConvention> conventions;

  void clearState() {
    unusedArgs.clear();
    unusedParamsIndex.clear();
    unusedParamsAttr.clear();
    inputParams.clear();
    inputTypes.clear();
    conventions.clear();
  }

  void identifyUnusedArguments(GeneratorOp oldFunction, StringAttr oldSymbol,
                               bool isRecursive) {
    FuncType oldBaseSig = oldFunction.getFuncTypeGenerator().getBody();

    unusedArgs = llvm::BitVector(oldFunction.getNumArguments(), true);

    // Identify unused parameters.
    for (auto [idx, arg, argConvention] : llvm::enumerate(
             oldFunction.getArguments(), oldBaseSig.getArgConventions())) {
      // If the function is recursive check that this argument has users beyond
      // the recursive call.
      if (isRecursive) {
        bool allCallsToSelf = true;
        for (Operation *op : arg.getUsers()) {
          CallOp call = dyn_cast<CallOp>(op);
          if (!call) {
            allCallsToSelf = false;
            break;
          }
          if (oldSymbol !=
              cast<FlatSymbolRefAttr>(call.getCalleeSymbol()).getValue()) {
            allCallsToSelf = false;
            break;
          }
          if (op->getOperand(idx) != arg) {
            // If argument is passed to a different argument then the argument
            // cannot be removed.
            // TODO: Properly resolve this case.
            allCallsToSelf = false;
            break;
          }
        }

        if (allCallsToSelf)
          continue;
      }

      SmallPtrSet<Operation *, 2> debugValues;
      bool hasNonDebugUser = false;
      for (Operation *user : arg.getUsers()) {
        if (auto value = dyn_cast<DebugInfo::ValueOp>(user)) {
          debugValues.insert(value);
        } else {
          hasNonDebugUser = true;
          break;
        }
      }

      if (hasNonDebugUser) {
        unusedArgs[idx] = false;
        inputTypes.push_back(arg.getType());
        conventions.push_back(argConvention);
      } else if (!debugValues.empty()) {
        // If the only users are debug values, replace them with a kill.
        auto firstValue = cast<DebugInfo::ValueOp>(*debugValues.begin());
        auto builder = OpBuilder::atBlockBegin(oldFunction.getBody());
        DebugInfo::KillOp::create(builder, firstValue.getLoc(),
                                  firstValue.getValueInfo());
        for (Operation *value : debugValues)
          value->erase();
      }
    }
  }

  void identifyUnusedParameters(GeneratorOp oldFunction, StringAttr oldSymbol,
                                bool isRecursive) {
    // Start with all parameters.
    for (ParamDeclAttr decl : oldFunction.getInputParams())
      unusedParamsAttr.insert(decl.getName());

    // Size to the declared param list, not the unique-name set: duplicate
    // names would otherwise undersize the bitvector and OOB on index access.
    unusedParamsIndex =
        llvm::BitVector(oldFunction.getInputParams().size(), true);
    // Walk over all parameter uses.
    mlir::AttrTypeWalker walker;
    walker.addWalk(
        [&](ParamDeclRefAttr ref) { unusedParamsAttr.erase(ref.getName()); });

    // We can not remove result type dependent parameters.
    for (auto res : oldFunction.getResultTypes())
      walker.walk(res);

    oldFunction.walk([&](Operation *op) {
      if (unusedParamsAttr.empty())
        return WalkResult::interrupt();

      // Don't include the parameters on the recursive calls.
      if (isRecursive) {
        if (CallOp call = dyn_cast<CallOp>(op)) {
          if (oldSymbol ==
              cast<FlatSymbolRefAttr>(call.getCalleeSymbol()).getValue()) {

            // Exclude direct uses but not uses within expressions. e.g.
            // foo<x+step, step>
            for (auto [param, decl] :
                 llvm::zip(call.getCallee().getParamValues(),
                           oldFunction.getInputParams())) {
              auto asRef = dyn_cast<ParamDeclRefAttr>(param);
              // A direct pass through of the same parameter doesn't count.
              if (!asRef || asRef.getName() != decl.getName() ||
                  asRef.getType() != decl.getType())
                walker.walk(param);
            }
            return WalkResult::advance();
          }
        }
      }

      // Don't scan the input parameters on the function since they obviously
      // will include a hit.
      if (op == oldFunction) {
        walker.walk(oldFunction.getLLVMMetadataArray());

        // The linkage name might contain parameter references.
        if (auto linkageName = oldFunction.getLinkageNameAttr())
          walker.walk(linkageName);
        // Most function attrs will only contain false positives, types on the
        // arguments / results are the real source of truth.
        walker.walk(oldFunction.getDecoratorsAttr());
        for (const NamedAttribute &attr : op->getDiscardableAttrs())
          walker.walk(attr.getValue());
        // Exclude arguments only used on unused inputs.
        for (auto [idx, arg] : llvm::enumerate(oldFunction.getArguments())) {
          if (!unusedArgs[idx])
            walker.walk(arg.getType());
        }
      } else {
        // Anything else just scan normally.
        for (const NamedAttribute &attr : op->getAttrs())
          walker.walk(attr.getValue());
        for (Type type : op->getOperandTypes())
          walker.walk(type);
        for (Region &region : op->getRegions()) {
          for (Type type : region.getArgumentTypes())
            walker.walk(type);
        }
      }

      for (Type type : op->getResultTypes())
        walker.walk(type);
      return WalkResult::advance();
    });

    // Map from the actual parameters we know are unused onto their index in the
    // function parameter list.
    for (auto [idx, decl] : llvm::enumerate(oldFunction.getInputParams())) {
      if (!unusedParamsAttr.contains(decl.getName())) {
        unusedParamsIndex[idx] = false;
        inputParams.push_back(decl);
      }
    }
  }

  void replaceCall(OpBuilder &builder, CallOp oldCall, GeneratorOp newFunc,
                   FlatSymbolRefAttr flatSym) {
    builder.setInsertionPoint(oldCall);
    newParams.clear();
    newOperands.clear();

    for (auto [idx, inParam] :
         llvm::enumerate(oldCall.getCallee().getParamValues())) {
      if (!unusedParamsIndex[idx])
        newParams.push_back(inParam);
    }

    for (auto [idx, operand] : llvm::enumerate(oldCall.getOperands())) {
      if (!unusedArgs[idx]) {
        assert(operand && "Operand cannot be nullptr");
        newOperands.push_back(operand);
      }
    }
    auto symbol = SymbolConstantAttr::get(
        flatSym,
        newFunc.getFuncTypeGenerator().getSpecializedGenerator(
            newParams, &*evaluationContext, oldCall.getLoc()),
        newParams);

    auto newCall =
        CallOp::create(builder, oldCall.getLoc(), symbol, newOperands);

    oldCall.replaceAllUsesWith(newCall);
    oldCall.erase();
  }
};

// Tracker of all the calls to a function and the set of function those calls
// are contained within.
struct CallSites {
  llvm::SetVector<GeneratorOp> callers;
  SmallVector<CallOp> calls;
};

// This worklist helper tracks which ops are still to be scheduled.
struct WorklistHelper {
  WorklistHelper(size_t numFuncs)
      : shouldSchedule(numFuncs, false), allOps(numFuncs), numProcessed(0) {
    funcToIndex.reserve(numFuncs);
  }

  // 1:1 mapping between generator and a bool which marks it as ready to
  // schedule. std::vector to make use of the bool optimization.
  std::vector<bool> shouldSchedule;

  // All the ops so we can traverse cheaply.
  SmallVector<GeneratorOp> allOps;

  // Fast lookup for generators in the above.
  DenseMap<GeneratorOp, size_t> funcToIndex;

  // The number of ops which have been processed. Same as
  // std::all_of(shouldSchedule)
  size_t numProcessed;

  // We include the index in the worklist so we have an O(1) way to update the
  // schedule.
  SmallVector<std::pair<GeneratorOp, size_t>> worklist;

  // Preserved temporaries to find cycles
  DenseSet<GeneratorOp> cycleSeenFuncs;
  SmallVector<GeneratorOp> cycleFinderWorklist;

  void addToSchedule(GeneratorOp gen, size_t index) {
    allOps[index] = gen;
    shouldSchedule[index] = true;
    funcToIndex[gen] = index;
  }

  void markProcessed(std::pair<GeneratorOp, size_t> pair) {
    shouldSchedule[pair.second] = false;
    ++numProcessed;
  }

  std::pair<GeneratorOp, size_t>
  pop(DenseMap<GeneratorOp, CallSites> &funcUsers,
      DenseMap<GeneratorOp, SmallVector<GeneratorOp>> &genToCallees,
      DenseMap<GeneratorOp, size_t> &numCallersFromFunc, bool &brokeCycle) {
    // If the worklist isn't empty obviously we can just pop from the back right
    // away.
    if (!worklist.empty())
      return worklist.pop_back_val();

    // Otherwise look and see if there's anything that can be obviously
    // scheduled.
    for (auto [idx, gen] : llvm::enumerate(allOps)) {
      // Skip ops which should not be scheduled.
      if (!shouldSchedule[idx])
        continue;

      // Remove any with no calls.
      CallSites &callSites = funcUsers[gen];
      if (callSites.calls.empty()) {
        shouldSchedule[idx] = false;
        ++numProcessed;
      } else if (numCallersFromFunc[gen] == 0) {
        // Add any functions with no dependencies.
        worklist.push_back({gen, idx});
      }
    }

    if (!worklist.empty())
      return pop_back();

    // Finally if we couldn't find anything it must mean there is a cycle in the
    // graph.
    for (auto [idx, gen] : llvm::enumerate(allOps)) {
      // Skip ops which should not be scheduled.
      if (!shouldSchedule[idx])
        continue;

      cycleSeenFuncs.clear();
      cycleFinderWorklist.clear();
      cycleFinderWorklist.push_back(gen);

      while (!cycleFinderWorklist.empty()) {
        GeneratorOp head = cycleFinderWorklist.back();
        cycleFinderWorklist.pop_back();
        if (!shouldSchedule[funcToIndex[head]])
          continue;
        cycleSeenFuncs.insert(head);

        SmallVector<GeneratorOp> &uniqueCallees = genToCallees[head];
        for (GeneratorOp called : uniqueCallees) {
          if (!shouldSchedule[funcToIndex[called]])
            continue;
          // Found the cycle, break it.
          if (cycleSeenFuncs.contains(called)) {
            // By returning the function with the cycle we will schedule it
            // immediately. This however means we need to remove any pending
            // dependencies on it.
            brokeCycle = true;
            return {head, funcToIndex[head]};
          }

          cycleFinderWorklist.push_back(called);
        }
      }
    }

    // If all possible cycles have been eliminated and we still couldn't find
    // something legal then there's no more we can do and there's no more
    // functions to schedule.
    return {nullptr, 0};
  }

  bool empty() { return numProcessed == allOps.size(); }

private:
  std::pair<GeneratorOp, size_t> pop_back() {
    auto pair = worklist.back();
    worklist.pop_back();
    return pair;
  }
};

} // namespace

void RemoveUnusedParams::runOnOperation() {
  ModuleOp mod = getOperation();
  MLIRContext *ctx = mod.getContext();
  OpBuilder builder{mod->getContext()};
  auto &analysis = getAnalysis<mlir::SymbolTableAnalysis>();
  mlir::LockedSymbolTableCollection symTabCollection(
      analysis.getSymbolTables());
  SymbolTable &symTab = analysis.getTopLevelSymbolTable();
  evaluationContext.emplace(mod.getOperation(), symTabCollection);
  auto optimizedOutDIType = builder.getType<DebugInfo::DIUnspecifiedType>(
      builder.getStringAttr("optimized out"));

  // Count number of functions cloned to tell how much memory to alloc.
  size_t numFuncs = 0;
  for (auto _ : mod.getOps<GeneratorOp>()) {
    (void)_;
    ++numFuncs;
  }

  DenseMap<GeneratorOp, CallSites> funcUsers;
  funcUsers.reserve(numFuncs);

  // Maintain a map of all the functions called by this function.
  DenseMap<GeneratorOp, llvm::SmallVector<GeneratorOp>> genToCallees;
  genToCallees.reserve(numFuncs);

  // Maintaining a separate dict of the number of unique callers avoids the need
  // to resize the above vector which is needed for deterministic traversal.
  DenseMap<GeneratorOp, size_t> numCallersFromFunc;
  numCallersFromFunc.reserve(numFuncs);

  // A set of the functions which are immediately recursive, i.e directly call
  // themselves. This is excludes indirect recursion through calls etc. They
  // will need to manually remap their calls.
  DenseSet<GeneratorOp> recursiveFuncs;

  // Maintain a separate list of operations we are going to process but aren't
  // ready yet. The active worklist of functions which are ready for us to
  // remove their arguments from them.
  WorklistHelper toSchedule(numFuncs);

  DenseSet<GeneratorOp> seenCallees;

  // Build call graph.
  for (auto [idx, generator] : llvm::enumerate(mod.getOps<GeneratorOp>())) {
    seenCallees.clear();
    SmallVector<GeneratorOp> callees;

    numCallersFromFunc[generator] = 0;

    // Gen needs explicit capture to make structured bindings capture legal.
    generator.walk([&, gen = generator](CallOp call) {
      // We can also call external generators and other things so it's not safe
      // to assume it's a generator op.
      auto calledFunc = dyn_cast_or_null<GeneratorOp>(symTab.lookup(
          cast<FlatSymbolRefAttr>(call.getCalleeSymbol()).getValue()));
      if (!calledFunc || calledFunc.isExternal())
        return;

      // Track the caller of this generator, but only one entry per call.
      if (!seenCallees.contains(calledFunc)) {
        callees.push_back(calledFunc);
        seenCallees.insert(calledFunc);
        ++numCallersFromFunc[gen];
      }

      // Track recursive functions separately.
      if (calledFunc == gen) {
        recursiveFuncs.insert(gen);
      } else {
        // Get or allocate.
        CallSites &entry = funcUsers[calledFunc];
        // Then insert without iteration invalidation in case of allocate.
        entry.callers.insert(gen);
        entry.calls.push_back(call);
      }
    });

    toSchedule.addToSchedule(generator, idx);

    // Start from the leaf nodes, i.e the functions which don't call anything.
    if (callees.empty())
      toSchedule.worklist.push_back({generator, idx});

    genToCallees.try_emplace(generator, callees);
  }

  // The actual algorithm which will traverse the calls, identify unused
  // parameters and a rewrite them.
  while (!toSchedule.empty()) {
    bool brokeCycle = false;
    std::pair<GeneratorOp, size_t> pair =
        toSchedule.pop(funcUsers, genToCallees, numCallersFromFunc, brokeCycle);
    GeneratorOp oldFunction = pair.first;

    // Will return null if there's nothing left to schedule.
    if (!oldFunction)
      break;

    // To optimize memory allocations we reuse the allocations of previous steps
    // by just clearing the data structures.
    clearState();

    bool isRecursive = recursiveFuncs.contains(oldFunction);
    StringAttr oldSymbol = oldFunction.getSymNameAttr();
    FuncTypeGeneratorType oldSigGen = oldFunction.getFuncTypeGenerator();
    FuncType oldBaseSig = oldSigGen.getBody();

    // Collate information about unused parameters + arguments in the shared
    // state.
    identifyUnusedArguments(oldFunction, oldSymbol, isRecursive);
    identifyUnusedParameters(oldFunction, oldSymbol, isRecursive);

    CallSites &callSites = funcUsers[oldFunction];

    // If nothing is unused we need to clear the calls and treat this as
    // having already been processed so the rest of the call graph can
    // progress.
    if (callSites.calls.empty() ||
        (unusedArgs.none() && unusedParamsIndex.none())) {
      callSites.calls.clear();
      for (GeneratorOp caller : callSites.callers)
        --numCallersFromFunc[caller];

      // We don't need to process this anymore.
      toSchedule.markProcessed(pair);
      continue;
    }

    GeneratorOp newFunc = oldFunction.clone();

    // Recursive functions may still have one reference so we need to drop it.
    if (isRecursive) {
      for (size_t i = 0, e = unusedArgs.size(); i != e; ++i) {
        if (unusedArgs[i])
          newFunc.getArgument(i).dropAllUses();
      }
    }

    auto eraseResult = newFunc.eraseArguments(unusedArgs);
    if (failed(eraseResult)) {
      newFunc.emitError() << "Failed to erase unused arguments";
      return;
    }

    auto functionType = FunctionType::get(
        ctx, inputTypes, oldFunction.getFunctionType().getResults());

    // Update the sig to partially specialize on those function types.
    newFunc.setFuncTypeGenerator(
        FuncTypeGeneratorType::remapToFuncTypeGenerator(
            inputParams, functionType,
            /*argConventions=*/conventions,
            /*fnEffects=*/oldBaseSig.getFnEffects(),
            /*fnMetadata=*/oldBaseSig.getMetadata(),
            /*genMetadata=*/oldSigGen.getParamListAttrs(), [&] {
              llvm_unreachable("Failed to remap generator signature.");
              return oldFunction.emitError(
                  "Failed to remap generator signature.");
            }));
    newFunc.setFunctionType(functionType);
    newFunc.setInputParams(inputParams);

    // Will either be an internal func or a cloned copy of an external func.
    newFunc.setNotExported();
    // The linkage name applies to the exported original, not this internal
    // optimized variant.
    newFunc.removeLinkageNameAttr();

    // Update the name so the ABI's don't clash. I.E so this doesn't name match
    // with another package which didn't run this optimization.
    // TODO: shouldn't need to do this but we need to do it while internal
    // functions are being linked across packages by the package include.
    newFunc.setSymName((Twine(newFunc.getSymName()) + "_REMOVED_ARG").str());
    symTab.insert(newFunc);

    // Update LLVM per-arg metadata.
    if (ArrayRef<Attribute> oldLLVMArgMetadata =
            newFunc.getLLVMArgMetadataArray().getValue();
        !oldLLVMArgMetadata.empty()) {
      SmallVector<Attribute> llvmArgMetadata;
      for (unsigned i = 0, e = oldLLVMArgMetadata.size(); i < e; ++i)
        if (!unusedArgs[i])
          llvmArgMetadata.emplace_back(oldLLVMArgMetadata[i]);
      newFunc.setLLVMArgMetadataArrayAttr(
          mlir::ArrayAttr::get(ctx, llvmArgMetadata));
    }

    // The DISubroutineType of the function may reference unused parameters.
    // This just means this function has a shared implementation across all
    // possible instantiations of this parameter. Concretize unused parameters
    // into UninitMemAttr  for now.
    // TODO (MOCO-900): Represent templated DISubroutineType and concretize
    // unused parameters to some special type (e.g. DIUnspecifiedType).
    if (DebugInfo::DISubprogramAttr oldScope =
            oldFunction.getSubprogramScope()) {
      // In line-tables mode the debuginfo-strip pass has already emptied the
      // subroutine type, so there are no argument types to update.
      auto cu = oldScope.getCompileUnit();
      if (!cu ||
          cu.getEmissionKind() != DebugInfo::EmissionKind::LineTablesOnly) {
        ParameterEvaluator evaluator;
        ArrayRef<ParamDeclAttr> inputParams(oldFunction.getInputParams());
        for (size_t index : unusedParamsIndex.set_bits()) {
          ParamDeclAttr decl = inputParams[index];
          evaluator.setDeclBinding(decl, UninitMemAttr::get(decl.getType()));
        }

        auto subroutineType =
            cast<DebugInfo::DISubroutineType>(oldScope.getType());
        SmallVector<DebugInfo::DIType> argTypes(
            subroutineType.getArgumentTypes());
        for (size_t index : unusedArgs.set_bits())
          argTypes[index] = optimizedOutDIType;

        auto newType =
            cast<DebugInfo::DISubroutineType>(evaluator.getReboundType(
                builder.getType<DebugInfo::DISubroutineType>(
                    subroutineType.getCallingConvention(), argTypes,
                    subroutineType.getResultTypes())));
        if (newType != subroutineType) {
          mlir::AttrTypeReplacer replacer;
          // Replace occurrences of the current subprogram with a new type &
          // name.
          auto newScope = oldScope.cloneWith(oldScope.getSourceName(),
                                             newFunc.getSymNameAttr(), newType);
          replacer.addReplacement([=](DebugInfo::DISubprogramAttr scope) {
            if (scope == oldScope)
              return std::make_pair(newScope, WalkResult::skip());
            return std::make_pair(scope, WalkResult::advance());
          });
          // Replace subroutine types of other subprograms (inlined scopes).
          replacer.addReplacement([&](DebugInfo::DISubroutineType subroutine) {
            return evaluator.getReboundType(subroutine);
          });

          DebugInfo::DebugInfoDialect *diDialect =
              getContext().getLoadedDialect<DebugInfo::DebugInfoDialect>();
          newFunc->walk([&](Operation *op) {
            // Need to replace attrs for debuginfo ops (to update scopes inside
            // variable info). For all other ops, only need to update locs.
            bool replaceAttrs = op->getDialect() == diDialect;
            replacer.replaceElementsIn(op, replaceAttrs,
                                       /*replaceLocs=*/true,
                                       /*replaceTypes=*/false);
          });
        }
      }
    }

    auto flatSym = FlatSymbolRefAttr::get(ctx, newFunc.getSymName());

    // Update all the callers to call the new function, dropping the unused
    // parameters on their side.
    for (CallOp oldCall : llvm::make_early_inc_range(callSites.calls))
      replaceCall(builder, oldCall, newFunc, flatSym);
    callSites.calls.clear();

    // Manually rewrite any recursive calls to self.
    if (isRecursive) {
      newFunc.walk([&](CallOp oldCall) {
        if (oldSymbol ==
            cast<FlatSymbolRefAttr>(oldCall.getCalleeSymbol()).getValue())
          replaceCall(builder, oldCall, newFunc, flatSym);
      });
    }

    toSchedule.markProcessed(pair);

    // If we broke a cycle we need to add the calls we have which have not been
    // updated yet to the call graph so they will be updated later as well.
    if (brokeCycle && newFunc != oldFunction) {
      newFunc.walk([&](CallOp call) {
        auto calledFunc = dyn_cast_or_null<GeneratorOp>(symTab.lookup(
            cast<FlatSymbolRefAttr>(call.getCalleeSymbol()).getValue()));
        if (calledFunc && calledFunc != newFunc) {
          // Update the calls of any called function if they are still to be
          // scheduled.
          auto itr = funcUsers.find(calledFunc);
          if (itr != funcUsers.end() && !itr->second.calls.empty())
            itr->second.calls.push_back(call);
        }
      });
    }

    // Remove this function as a dependency from any callers.
    for (GeneratorOp caller : callSites.callers)
      --numCallersFromFunc[caller];
  }

  // Control-flow is not modified.
  markAnalysesPreserved<HLCF::CFGAnalysis>();
}
