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

#include "Mojo/Compiler/LLVMIRUtils.h"
#include "Mojo/Compiler/Target/TargetBackend.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Support/Buffer.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/GlobalStatus.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace M;
using namespace KGEN;

#define DEBUG_TYPE "llvm-module-split"

//===----------------------------------------------------------------------===//
// LLVMModuleAndContext
//===----------------------------------------------------------------------===//

ErrorOrSuccess LLVMModuleAndContext::create(
    function_ref<ErrorOr<std::unique_ptr<llvm::Module>>(llvm::LLVMContext &)>
        createModule) {
  assert(!module && "already have a module");
  auto moduleOr = createModule(*ctx);
  if (moduleOr.isError())
    return moduleOr.takeError();
  module = moduleOr.takeValue();
  return success();
}

void LLVMModuleAndContext::reset() {
  module.reset();
  ctx.reset();
}

//===----------------------------------------------------------------------===//
// StringConstantTable
//===----------------------------------------------------------------------===//

namespace {
/// Large strings are very inefficiently encoded in LLVM bitcode (each `char` is
/// encoded as a `uint64_t`). The LLVM bitcode reader is also very inefficiently
/// reads strings back, performing 3 ultimate copies of the data. This is made
/// worse by the fact the `getLazyBitcodeModule` does not lazily parse constants
/// from the LLVM bitcode. Thus, when per-function splitting a module with N
/// functions and M large string constants, we form 3*M*N copies of the large
/// strings.
///
/// This class is part of a workaround of this inefficiency. When processing a
/// module for splitting, we track any string global constants and their indices
/// in this table. If a module is going to be roundtripped through bitcode to be
/// lazily loaded, we externalize the strings by setting the corresponding
/// constants to `zeroinitializer` in the module before it is written to
/// bitcode. As we materialize constants on the other side, we check for a
/// materialized global variable that matches an entry in the string table and
/// directly copy the data over into the new LLVM context.
///
/// We can generalize this optimization to other large data types as necessary.
///
/// This class is used in an `RCRef` to be shared across multiple threads.
class StringConstantTable : public ReferenceCounted<StringConstantTable> {
  /// An entry in the string table consists of a global variable, its module
  /// index, and the a reference to the string data. Because the string data is
  /// owned by the original LLVM context, we have to ensure it stays alive.
  struct Entry {
    unsigned idx;
    const llvm::GlobalVariable *var;
    StringRef value;
  };

public:
  /// If `value` denotes a string constant, record the data at index `gvIdx`.
  void recordIfStringConstant(unsigned gvIdx, const llvm::GlobalValue &value) {
    auto var = dyn_cast<llvm::GlobalVariable>(&value);
    if (var && var->isConstant() && var->hasInternalLinkage()) {
      auto *init =
          dyn_cast<llvm::ConstantDataSequential>(var->getInitializer());
      if (init && init->isCString())
        stringConstants.push_back(Entry{gvIdx, var, init->getAsString()});
    }
  }

  /// Before writing the main module to bitcode, externalize large string
  /// constants by stubbing out their values. Take ownership of the main module
  /// so the string data stays alive.
  llvm::Module &externalizeStrings(LLVMModuleAndContext &&module) {
    mainModule = std::move(module);
    // Stub the initializers. The global variable is an internal constant, so it
    // must have an initializer.
    for (Entry &e : stringConstants) {
      auto *stub =
          llvm::Constant::getNullValue(e.var->getInitializer()->getType());
      // `const_cast` is OK because we own the module now.
      const_cast<llvm::GlobalVariable *>(e.var)->setInitializer(stub);
    }
    return *mainModule;
  }

  /// This is an iterator over the entries in the string table.
  class Injector {
    using const_iterator = std::vector<Entry>::const_iterator;

  public:
    /// Given a global variable in a materialized module and its index, if it is
    /// a string constant found in the table, copy the data over into the new
    /// LLVM context and set the initializer.
    void materializeIfStringConstant(unsigned gvIdx,
                                     llvm::GlobalVariable &var) {
      while (it != e && it->idx < gvIdx)
        ++it;
      if (it == e || it->idx != gvIdx)
        return;
      var.setInitializer(llvm::ConstantDataArray::getString(
          var.getType()->getContext(), it->value, /*AddNull=*/false));
    }

  private:
    explicit Injector(const_iterator it, const_iterator e) : it(it), e(e) {}

    const_iterator it, e;

    friend class StringConstantTable;
  };

  Injector begin() const {
    return Injector(stringConstants.begin(), stringConstants.end());
  }

private:
  std::vector<Entry> stringConstants;
  LLVMModuleAndContext mainModule;
};

//===----------------------------------------------------------------------===//
// Module Splitter
//===----------------------------------------------------------------------===//

class LLVMModuleSplitterImpl {
public:
  explicit LLVMModuleSplitterImpl(LLVMModuleAndContext module)
      : mainModule(std::move(module)) {}

  /// Split the LLVM module into multiple modules using the provided process
  /// function.
  void split(LLVMSplitProcessFn processFn);

private:
  struct ValueInfo {
    /// The immediate global value dependencies of a value.
    SmallVector<const llvm::GlobalValue *> dependencies;
    /// Map each global value to its index in the module. We will use this to
    /// materialize global values from bitcode.
    unsigned gvIdx;
  };

  struct TransitiveDeps {
    /// The transitive dependencies.
    llvm::MapVector<const llvm::GlobalValue *, unsigned> deps;
    /// True if computation is complete.
    bool complete = false;
    /// The assigned module index.
    std::optional<unsigned> mutIdx;
  };

  /// Collect the immediate global value dependencies of `value`. `orig` is the
  /// original transitive value, which is not equal to `value` when it is used
  /// in a constant.
  void collectImmediateDependencies(const llvm::Value *value,
                                    const llvm::GlobalValue *orig);

  /// The main LLVM module being split.
  LLVMModuleAndContext mainModule;

  /// The value info for each global value in the module.
  llvm::DenseMap<const llvm::Value *, ValueInfo> infos;

  /// The transitive dependencies of each global value.
  llvm::MapVector<const llvm::GlobalValue *, TransitiveDeps> transitiveDeps;

  /// Users of split "anchors". These are global values where we don't want
  /// their users to be split into different modules because it will cause the
  /// symbol to be duplicated.
  llvm::MapVector<const llvm::GlobalValue *, llvm::SetVector<TransitiveDeps *>>
      splitAnchorUsers;
};
} // namespace

static LLVMModuleAndContext readAndMaterializeDependencies(
    BufferRef buf,
    const llvm::MapVector<const llvm::GlobalValue *, unsigned> &set,
    const StringConstantTable &strtab) {

  // First, create a lazy module with an internal bitcode materializer.
  // TODO: Not sure how to make lazy loading metadata work.
  LLVMModuleAndContext result;
  {
    CompilerTimeTraceScope traceScope("createLLVMContext");
    (void)result.create([&](llvm::LLVMContext &ctx) {
      CompilerTimeTraceScope traceScope("getLazyBitcodeModule");
      return llvm::cantFail(llvm::getLazyBitcodeModule(
          llvm::MemoryBufferRef(buf->getBuffer(), "<split-module>"), ctx,
          /*ShouldLazyLoadMetadata=*/false));
    });
    result->removeModuleInlineAsm();
  }

  SmallVector<unsigned> sortIndices =
      llvm::to_vector(llvm::make_second_range(set));
  llvm::sort(sortIndices, std::less<unsigned>());
  auto idxIt = sortIndices.begin();
  auto idxEnd = sortIndices.end();

  // The global value indices go from globals, functions, then aliases. This
  // mirrors the order in which global values are deleted by LLVM's GlobalDCE.
  unsigned curIdx = 0;
  StringConstantTable::Injector it = strtab.begin();
  for (llvm::GlobalVariable &global : result->globals()) {
    if (idxIt != idxEnd && curIdx == *idxIt) {
      ++idxIt;
      llvm::cantFail(global.materialize());
      it.materializeIfStringConstant(curIdx, global);
    } else {
      global.setInitializer(nullptr);
      global.setComdat(nullptr);
      global.setLinkage(llvm::GlobalValue::ExternalLinkage);
      // External link should not be DSOLocal anymore,
      // otherwise position independent code generates
      // `R_X86_64_PC32` instead of `R_X86_64_REX_GOTPCRELX`
      // for these symbols and building shared library from
      // a static archive of this module will error with an `fPIC` confusion.
      global.setDSOLocal(false);
    }
    ++curIdx;
  }
  for (llvm::Function &func : result->functions()) {
    if (idxIt != idxEnd && curIdx == *idxIt) {
      ++idxIt;
      llvm::cantFail(func.materialize());
    } else {
      func.deleteBody();
      func.setComdat(nullptr);
      func.setLinkage(llvm::GlobalValue::ExternalLinkage);
      // External link should not be DSOLocal anymore,
      // otherwise position independent code generates
      // `R_X86_64_PC32` instead of `R_X86_64_REX_GOTPCRELX`
      // for these symbols and building shared library from
      // a static archive of this module will error with an `fPIC` confusion.
      // External link should not be DSOLocal anymore,
      // otherwise position independent code generation get confused.
      func.setDSOLocal(false);
    }
    ++curIdx;
  }

  // Finalize materialization of the module.
  llvm::cantFail(result->materializeAll());

  // Now that the module is materialized, we can start deleting stuff. Just
  // delete declarations with no uses.
  for (llvm::GlobalVariable &global :
       llvm::make_early_inc_range(result->globals())) {
    if (global.isDeclaration() && global.use_empty())
      global.eraseFromParent();
  }
  for (llvm::Function &func : llvm::make_early_inc_range(result->functions())) {
    if (func.isDeclaration() && func.use_empty())
      func.eraseFromParent();
  }
  return result;
}

/// support for splitting an LLVM module into multiple parts using exported
/// functions as anchors, and pull in all dependency on the call stack into one
/// module.
void KGEN::splitPerExported(LLVMModuleAndContext module,
                            LLVMSplitProcessFn processFn) {

  CompilerTimeTraceScope traceScope("splitPerExported");
  LLVMModuleSplitterImpl impl(std::move(module));
  impl.split(processFn);
}

void LLVMModuleSplitterImpl::split(LLVMSplitProcessFn processFn) {
  // The use-def list is sparse. Use it to build a sparse dependency graph
  // between global values.
  auto strtab = RCRef<StringConstantTable>::create();
  unsigned gvIdx = 0;
  auto computeDeps = [&](const llvm::GlobalValue &value) {
    strtab->recordIfStringConstant(gvIdx, value);
    infos[&value].gvIdx = gvIdx++;
    collectImmediateDependencies(&value, &value);
  };

  // The target backend, used to recognize shared-memory globals (which aren't
  // split anchors). Null for unregistered targets, in which case nothing is
  // treated as shared memory.
  ErrorOr<const TargetBackend *> backendOr =
      TargetBackendRegistry::get().lookup(mainModule->getTargetTriple());
  const TargetBackend *backend = backendOr.isError() ? nullptr : *backendOr;

  // NOTE: The visitation of globals then functions has to line up with
  // `readAndMaterializeDependencies`.
  for (const llvm::GlobalVariable &global : mainModule->globals()) {
    computeDeps(global);
    if (!global.hasInternalLinkage() && !global.hasPrivateLinkage() &&
        !(backend && backend->isSharedMemoryGlobal(global)))
      transitiveDeps[&global];
  }
  for (const llvm::Function &fn : mainModule->functions()) {
    computeDeps(fn);
    if (!fn.isDeclaration() && (fn.hasExternalLinkage() || fn.hasWeakLinkage()))
      transitiveDeps[&fn];
  }

  // If there is only one (or fewer) exported functions, forward the main
  // module.
  if (transitiveDeps.size() <= 1)
    return processFn(forwardModule(std::move(mainModule)), std::nullopt,
                     /*numFunctionBase=*/0);

  // Now for each export'd global value, compute the transitive set of
  // dependencies using DFS.
  SmallVector<const llvm::GlobalValue *> worklist;
  for (auto &[value, deps] : transitiveDeps) {
    worklist.clear();
    worklist.push_back(value);
    while (!worklist.empty()) {
      const llvm::GlobalValue *it = worklist.pop_back_val();

      auto [iter, inserted] = deps.deps.insert({it, -1});
      if (!inserted) {
        // Already visited.
        continue;
      }
      // Pay the cost of the name lookup only on a miss.
      const ValueInfo &info = infos.at(it);
      iter->second = info.gvIdx;

      // If this value depends on another value that is going to be split, we
      // don't want to duplicate the symbol. Keep all the users together.
      if (it != value) {
        if (auto depIt = transitiveDeps.find(it);
            depIt != transitiveDeps.end()) {
          auto &users = splitAnchorUsers[it];
          users.insert(&deps);
          // Make sure to include the other value in its own user list.
          users.insert(&depIt->second);
          // We don't have to recurse since the subgraph will get processed.
          continue;
        }
      }

      // If this value depends on a mutable global, keep track of it. We have to
      // put all users of a mutable global in the same module.
      if (auto *global = dyn_cast<llvm::GlobalVariable>(it);
          global && !global->isConstant()) {
        // Globals in shared memory are not splitting anchors (see
        // TargetBackend::isSharedMemoryGlobal).
        if (!(backend && backend->isSharedMemoryGlobal(*global))) {
          splitAnchorUsers[global].insert(&deps);
        }
      }

      // Recursive on dependencies.
      llvm::append_range(worklist, info.dependencies);
    }

    deps.complete = true;
  }

  // For each mutable global, grab all the transitive users and put them in one
  // module. If global A has user set A* and global B has user set B* where
  // A* and B* have an empty intersection, all values in A* will be assigned 0
  // and all values in B* will be assigned 1. If global C has user set C* that
  // overlaps both A* and B*, it will overwrite both to 2.
  SmallVector<SmallVector<TransitiveDeps *>> bucketing(splitAnchorUsers.size());
  for (auto [curMutIdx, bucket, users] :
       llvm::enumerate(bucketing, llvm::make_second_range(splitAnchorUsers))) {
    for (TransitiveDeps *deps : users) {
      if (deps->mutIdx && *deps->mutIdx != curMutIdx) {
        auto &otherBucket = bucketing[*deps->mutIdx];
        for (TransitiveDeps *other : otherBucket) {
          bucket.push_back(other);
          other->mutIdx = curMutIdx;
        }
        otherBucket.clear();
        assert(*deps->mutIdx == curMutIdx);
      } else {
        bucket.push_back(deps);
        deps->mutIdx = curMutIdx;
      }
    }
  }

  // Now that we have assigned buckets to each value, merge the transitive
  // dependency sets of all values belonging to the same set.
  SmallVector<llvm::MapVector<const llvm::GlobalValue *, unsigned>> buckets(
      bucketing.size());
  for (auto [deps, bucket] : llvm::zip(bucketing, buckets)) {
    for (TransitiveDeps *dep : deps) {
      for (auto &namedValue : dep->deps)
        bucket.insert(namedValue);
    }
  }

  SmallVector<llvm::MapVector<const llvm::GlobalValue *, unsigned> *>
      setsToProcess;
  setsToProcess.reserve(buckets.size() + transitiveDeps.size());

  // Clone each mutable global bucket into its own module.
  for (auto &bucket : buckets) {
    if (bucket.empty())
      continue;
    setsToProcess.push_back(&bucket);
  }

  for (auto &[root, deps] : transitiveDeps) {
    // Skip values included in another transitive dependency set and values
    // included in mutable global sets.
    if (!deps.mutIdx)
      setsToProcess.push_back(&deps.deps);
  }

  if (setsToProcess.size() <= 1)
    return processFn(forwardModule(std::move(mainModule)), std::nullopt,
                     /*numFunctionBase=*/0);

  // Sort the sets by to schedule the larger modules first.
  llvm::sort(setsToProcess,
             [](auto *lhs, auto *rhs) { return lhs->size() > rhs->size(); });

  // Prepare to materialize slices of the module by first writing the main
  // module as bitcode to a shared buffer.
  auto buf = WriteableBuffer::get();
  {
    CompilerTimeTraceScope traceScope("writeMainModuleBitcode");
    llvm::Module &module = strtab->externalizeStrings(std::move(mainModule));
    llvm::WriteBitcodeToFile(module, *buf);
  }

  unsigned numFunctions = 0;
  for (auto [idx, set] : llvm::enumerate(setsToProcess)) {
    unsigned next = numFunctions + set->size();
    auto makeModule = [set = std::move(*set), buf = BufferRef(buf.copy()),
                       strtab = strtab.copy()]() mutable {
      return readAndMaterializeDependencies(std::move(buf), set, *strtab);
    };
    processFn(std::move(makeModule), idx, numFunctions);
    numFunctions = next;
  }
}

void LLVMModuleSplitterImpl::collectImmediateDependencies(
    const llvm::Value *value, const llvm::GlobalValue *orig) {
  for (const llvm::Value *user : value->users()) {
    // Recurse into pure constant users.
    if (isa<llvm::Constant>(user) && !isa<llvm::GlobalValue>(user)) {
      collectImmediateDependencies(user, orig);
      continue;
    }

    if (auto *inst = dyn_cast<llvm::Instruction>(user)) {
      const llvm::Function *func = inst->getParent()->getParent();
      infos[func].dependencies.push_back(orig);
    } else if (auto *globalVal = dyn_cast<llvm::GlobalValue>(user)) {
      infos[globalVal].dependencies.push_back(orig);
    } else {
      llvm_unreachable("unexpected user of global value");
    }
  }
}

namespace {
/// This class provides support for splitting an LLVM module into multiple
/// parts.
/// TODO: Clean up the splitters here (some code duplication) when we can move
/// to per function llvm compilation.
class LLVMModulePerFunctionSplitterImpl {
public:
  LLVMModulePerFunctionSplitterImpl(LLVMModuleAndContext module)
      : mainModule(std::move(module)) {}

  /// Split the LLVM module into multiple modules using the provided process
  /// function.
  void
  split(LLVMSplitProcessFn processFn,
        llvm::StringMap<llvm::GlobalValue::LinkageTypes> &symbolLinkageTypes,
        int64_t moduleIdx, unsigned numFunctionBase);

private:
  struct ValueInfo {
    const llvm::Value *value = nullptr;
    bool canBeSplit = true;
    llvm::DenseSet<const llvm::GlobalValue *> dependencies;
    llvm::DenseSet<const llvm::GlobalValue *> users;
    /// Map each global value to its index in the module. We will use this to
    /// materialize global values from bitcode.
    unsigned gvIdx;
    bool userEmpty = true;
  };

  /// Collect all of the immediate global value users of `value`.
  void collectValueUsers(const llvm::GlobalValue *value);

  /// Propagate use information through the module.
  void propagateUseInfo();

  /// The main LLVM module being split.
  LLVMModuleAndContext mainModule;

  /// The value info for each global value in the module.
  llvm::MapVector<const llvm::GlobalValue *, ValueInfo> valueInfos;
};
} // namespace

static void
removeDuplicates(llvm::MapVector<const llvm::GlobalValue *, unsigned> &set,
                 llvm::StringSet<> &seenFns, llvm::StringSet<> &dupFns) {
  set.remove_if([&](std::pair<const llvm::GlobalValue *, unsigned int> &kv) {
    if (auto fn = dyn_cast<llvm::Function>(kv.first)) {
      StringRef name = fn->getName();
      return !seenFns.insert(name).second || dupFns.contains(name);
    }
    return false;
  });
}

static uint64_t getNumFunctionsInSet(
    llvm::MapVector<const llvm::GlobalValue *, unsigned> &set) {
  uint64_t result = 0;
  for (auto [global, _] : set) {
    if (auto f = dyn_cast<llvm::Function>(global)) {
      if (f->isDeclaration())
        continue;
      ++result;
    }
  }
  return result;
}

/// support for splitting an LLVM module into multiple parts with each part
/// contains only one function (with exception for coroutine related functions.)
void KGEN::splitPerFunction(
    LLVMModuleAndContext module, LLVMSplitProcessFn processFn,
    llvm::StringMap<llvm::GlobalValue::LinkageTypes> &symbolLinkageTypes,
    int64_t inputModuleIdx, unsigned numFunctionBase) {
  CompilerTimeTraceScope traceScope("splitPerFunction");
  LLVMModulePerFunctionSplitterImpl impl(std::move(module));
  impl.split(processFn, symbolLinkageTypes, inputModuleIdx, numFunctionBase);
}

/// Split the LLVM module into multiple modules using the provided process
/// function.
void LLVMModulePerFunctionSplitterImpl::split(
    LLVMSplitProcessFn processFn,
    llvm::StringMap<llvm::GlobalValue::LinkageTypes> &symbolLinkageTypes,
    int64_t inputModuleIdx, unsigned numFunctionBase) {
  // Compute the value info for each global in the module.
  // NOTE: The visitation of globals then functions has to line up with
  // `readAndMaterializeDependencies`.
  auto strtab = RCRef<StringConstantTable>::create();
  unsigned gvIdx = 0;
  auto computeUsers = [&](const llvm::GlobalValue &value) {
    strtab->recordIfStringConstant(gvIdx, value);
    valueInfos[&value].gvIdx = gvIdx++;
    collectValueUsers(&value);
  };
  llvm::for_each(mainModule->globals(), computeUsers);
  llvm::for_each(mainModule->functions(), computeUsers);

  // With use information collected, propagate it to the dependencies.
  propagateUseInfo();

  // Now we can split the module.
  // We split the module per function and cloning any necessary dependencies:
  // - For function dependencies, only clone the declaration unless its
  //   coroutine related.
  // - For other internal values, clone as is.
  // This is much fine-grained splitting, which enables significantly higher
  // levels of parallelism (and smaller generated artifacts).
  // LLVM LTO style optimization may suffer a bit here since we don't have
  // the full callstack present anymore in each cloned module.
  llvm::DenseSet<const llvm::Value *> splitValues;
  SmallVector<llvm::MapVector<const llvm::GlobalValue *, unsigned>>
      setsToProcess;

  // Hoist these collections to re-use memory allocations.
  llvm::ValueToValueMapTy valueMap;
  SmallPtrSet<const llvm::Value *, 4> splitDeps;
  auto splitValue = [&](const llvm::GlobalValue *root) {
    // If the function is already split, e.g. if it was a dependency of
    // another function, skip it.
    if (splitValues.count(root))
      return;

    auto &valueInfo = valueInfos[root];
    valueMap.clear();
    splitDeps.clear();
    auto shouldSplit = [&](const llvm::GlobalValue *globalVal,
                           const ValueInfo &info) {
      // Only clone root and the declaration of its dependencies.
      if (globalVal == root) {
        splitDeps.insert(globalVal);
        return true;
      }

      if ((info.canBeSplit || info.userEmpty) &&
          isa_and_nonnull<llvm::Function>(globalVal))
        return false;

      if (valueInfo.dependencies.contains(globalVal)) {
        splitDeps.insert(globalVal);
        return true;
      }

      return false;
    };

    auto &set = setsToProcess.emplace_back();
    for (auto &[globalVal, info] : valueInfos) {
      if (shouldSplit(globalVal, info))
        set.insert({globalVal, info.gvIdx});
    }
    if (set.empty())
      setsToProcess.pop_back();

    // Record the split values.
    splitValues.insert(splitDeps.begin(), splitDeps.end());
  };

  // This is a set of special private symbol name that need to be renamed (e.g.,
  // those inserted by llvm optimizer) in order to avoid symbol override during
  // linking. Since each split goes through its own LLVM optimization, private
  // symbol with the same name (but different values) could be inserted by LLVM
  // (e.g., when detecting the same optimization pattern the transformation
  // leads to an insertion of global).
  // We have to do the renaming now before codegen because we need to ensure
  // name consistency between MCSymbols and global symbol names.
  //
  // Explanation on the entries:
  // - ".memset_pattern": MacOS specific, introduced by `LoopIdiomRecognize`.
  const llvm::StringSet<> namesToRename{".memset_pattern"};

  [[maybe_unused]] int64_t count = 0;
  SmallVector<const llvm::GlobalValue *> toSplit;
  unsigned unnamedGlobal = numFunctionBase;
  for (auto &global : mainModule->globals()) {
    if (global.hasInternalLinkage() || global.hasPrivateLinkage()) {
      if (!global.hasName()) {
        // Give unnamed GlobalVariable a unique name so that MCLink will not get
        // confused to name them while generating linked code since the IR
        // values can be different in each splits (for X86 backend.)
        // asan build inserts these unnamed GlobalVariables.
        global.setName("__mojo_unnamed" + Twine(unnamedGlobal++));
      }

      for (StringRef prefix : namesToRename.keys()) {
        if (global.getName().starts_with(prefix)) {
          global.setName(global.getName() + "_" + Twine(inputModuleIdx));
          break;
        }
      }

      symbolLinkageTypes.insert({global.getName().str(), global.getLinkage()});
      global.setLinkage(llvm::GlobalValue::WeakAnyLinkage);
      continue;
    }

    if (global.hasExternalLinkage())
      continue;

    // TODO: Add special handling for `llvm.global_ctors` and
    // `llvm.global_dtors`, because otherwise they end up tying almost all
    // symbols into the same split.
    LLVM_DEBUG(llvm::dbgs()
                   << (count++) << ": split global: " << global << "\n";);
    toSplit.emplace_back(&global);
  }

  for (auto &fn : mainModule->functions()) {
    if (fn.isDeclaration())
      continue;

    ValueInfo &info = valueInfos[&fn];
    if (fn.hasInternalLinkage() || fn.hasPrivateLinkage()) {
      // Avoid renaming when linking in MCLink.
      symbolLinkageTypes.insert({fn.getName().str(), fn.getLinkage()});
      fn.setLinkage(llvm::Function::LinkageTypes::WeakAnyLinkage);
    }

    if (info.canBeSplit || info.userEmpty) {
      LLVM_DEBUG(llvm::dbgs()
                     << (count++) << ": split fn: " << fn.getName() << "\n";);
      toSplit.emplace_back(&fn);
    }
  }

  // Run this now since we just changed the linkages.
  for (const llvm::GlobalValue *value : toSplit)
    splitValue(value);

  if (setsToProcess.size() <= 1)
    return processFn(forwardModule(std::move(mainModule)), std::nullopt,
                     numFunctionBase);

  auto duplicatedFns = std::move(mainModule.duplicatedFns);

  // Prepare to materialize slices of the module by first writing the main
  // module as bitcode to a shared buffer.
  auto buf = WriteableBuffer::get();
  {
    CompilerTimeTraceScope traceScope("writeMainModuleBitcode");
    llvm::Module &module = strtab->externalizeStrings(std::move(mainModule));
    llvm::WriteBitcodeToFile(module, *buf);
  }

  unsigned numFunctions = numFunctionBase;
  llvm::StringSet<> seenFns;

  for (auto [idx, set] : llvm::enumerate(setsToProcess)) {
    // Giving each function a unique ID across all splits for proper MC level
    // linking and codegen into one object file where duplicated functions
    // in each split will be deduplicated (with the linking).
    removeDuplicates(set, seenFns, duplicatedFns);
    if (set.empty())
      continue;

    unsigned next = numFunctions + getNumFunctionsInSet(set);

    LLVM_DEBUG(llvm::dbgs() << "split function base id: " << numFunctions
                            << " set size: " << set.size() << "\n";);

    auto makeModule = [set = std::move(set), buf = BufferRef(buf.copy()),
                       strtab = strtab.copy()]() mutable {
      return readAndMaterializeDependencies(std::move(buf), set, *strtab);
    };
    processFn(std::move(makeModule), idx, numFunctions);
    numFunctions = next;
  }
}

/// Collect all of the immediate global value users of `value`.
void LLVMModulePerFunctionSplitterImpl::collectValueUsers(
    const llvm::GlobalValue *value) {
  SmallVector<const llvm::User *> worklist(value->users());

  while (!worklist.empty()) {
    const llvm::User *userIt = worklist.pop_back_val();

    // Recurse into pure constant users.
    if (isa<llvm::Constant>(userIt) && !isa<llvm::GlobalValue>(userIt)) {
      worklist.append(userIt->user_begin(), userIt->user_end());
      continue;
    }

    if (const auto *inst = dyn_cast<llvm::Instruction>(userIt)) {
      const llvm::Function *func = inst->getParent()->getParent();
      valueInfos[value].users.insert(func);
      valueInfos[func];
    } else if (const auto *globalVal = dyn_cast<llvm::GlobalValue>(userIt)) {
      valueInfos[value].users.insert(globalVal);
      valueInfos[globalVal];
    } else {
      llvm_unreachable("unexpected user of global value");
    }
  }

  // If the current value is a mutable global variable, then it can't be
  // split.
  // [MOCO-1680] Also don't split private symbols as linking might not work if
  // there's a use of the symbol in a module where the symbol is external. That
  // is, if the symbol becomes external in a module, code generation may emit
  // code assuming the symbol is in GOT. However, private symbols are not
  // getting entries in GOT, therefore final linkage will fail with "undefined
  // symbol".
  //
  // NOTE: It's possible to support private symbols by switching them
  // to internal linkage (see https://llvm.org/docs/LangRef.html#linkage-types):
  // - internal: Similar to private, but the value shows as a local symbol
  //            (STB_LOCAL in the case of ELF) in the object file.
  // However, it cannot be done here, because it's not yet known how
  // many modules will use it.
  if (auto *global = dyn_cast<llvm::GlobalVariable>(value)) {
    valueInfos[value].canBeSplit =
        global->isConstant() && !global->hasPrivateLinkage();
    LLVM_DEBUG({
      llvm::dbgs() << "global " << global->getName()
                   << (valueInfos[value].canBeSplit ? " can be split"
                                                    : " cannot be split");
      llvm::dbgs() << '\n';
    });
  }
}

/// Propagate use information through the module.
void LLVMModulePerFunctionSplitterImpl::propagateUseInfo() {
  const size_t numValues = valueInfos.size();
  SmallVector<ValueInfo *> worklist;
  SmallVector<llvm::BitVector> deps(numValues, llvm::BitVector(numValues));
  SmallVector<const llvm::GlobalValue *> gvIdxValueMap(numValues);

  // Each value depends on itself. Seed the iteration with that.
  for (auto &[value, info] : valueInfos) {
    if (auto func = llvm::dyn_cast<llvm::Function>(value)) {
      if (func->isDeclaration())
        continue;
    }

    deps[info.gvIdx].set(info.gvIdx);
    gvIdxValueMap[info.gvIdx] = value;

    info.dependencies.insert(value);
    info.value = value;
    worklist.push_back(&info);
    if (!info.canBeSplit) {
      // If a value cannot be split, its users are also its dependencies.
      llvm::set_union(info.dependencies, info.users);
    }
  }

  // deps[idx1] := deps[idx1] u deps[idx2]
  auto unite = [&](size_t idx1, size_t idx2) {
    if (idx1 == idx2 || deps[idx1] == deps[idx2])
      return false;
    deps[idx1] |= deps[idx2];
    return true;
  };

  while (!worklist.empty()) {
    ValueInfo *info = worklist.pop_back_val();

    // Propagate the dependencies of this value to its users.
    for (const llvm::GlobalValue *user : info->users) {
      ValueInfo &userInfo = valueInfos.find(user)->second;
      if (info == &userInfo)
        continue;
      bool changed = false;

      // Merge dependency to user if current value is not a function that will
      // be split into a separate module.
      bool mergeToUserDep = true;
      if (llvm::isa_and_nonnull<llvm::Function>(info->value)) {
        mergeToUserDep = !info->canBeSplit;
      }

      // If there is a change, add the user info to the worklist.
      if (mergeToUserDep) {
        if (unite(userInfo.gvIdx, info->gvIdx))
          changed = true;
      }

      // If the value cannot be split, its users cannot be split either.
      if (!info->canBeSplit && userInfo.canBeSplit) {
        userInfo.canBeSplit = false;
        changed = true;
        // If a value cannot be split, its users are also its dependencies.
        for (const llvm::GlobalValue *uu : userInfo.users) {
          const ValueInfo &uuInfo = valueInfos.find(uu)->second;
          deps[userInfo.gvIdx].set(uuInfo.gvIdx);
        }
      }

      if (changed) {
        userInfo.value = user;
        worklist.push_back(&userInfo);
      }
    }

    // To save time spent in combining dependencies, skip non-immediate
    // dependencies (users) of the global value that cannot be split.
    if (info->canBeSplit || !isa_and_nonnull<llvm::GlobalValue>(info->value))
      continue;

    // If a value cannot be split, propagate its dependencies up to its
    // dependencies.
    for (auto depIter = deps[info->gvIdx].set_bits_begin(),
              depEnd = deps[info->gvIdx].set_bits_end();
         depIter != depEnd; ++depIter) {
      unite(info->gvIdx, *depIter);
    }
  }

  // Reconstruct dependencies from the bit vector
  for (size_t i = 0; i < numValues; ++i) {
    ValueInfo &info = valueInfos.find(gvIdxValueMap[i])->second;
    for (auto depIter = deps[i].set_bits_begin(),
              depEnd = deps[i].set_bits_end();
         depIter != depEnd; ++depIter) {
      info.dependencies.insert(gvIdxValueMap[*depIter]);
    }
  }

  for (auto &[value, info] : valueInfos) {
    info.userEmpty = info.users.empty() ||
                     (info.users.size() == 1 && info.users.contains(value));
  }
}
