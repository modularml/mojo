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
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Mojo/TransformUtils/SCCUtils.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Support/ThreadLocalCache.h"

using namespace M;
using namespace KGEN;
using namespace POP;

namespace {
// Forward declarations.
struct RegionState;

/// The "bottom" state is the uninitialized, overconstrained state of a value
/// (lattice element). No lattice element should be at this state at the end of
/// the analysis: this state, when joined with any other state, takes on the
/// value of the other state. It is thus used as the initial value of the state
/// of all lattice elements.
struct BottomState {
  /// This state is a singleton.
  bool operator==(const BottomState &o) const { return true; }
};

/// This is an intermediate state that represents a value that directly aliases
/// some allocation. It tracks the memory value being aliases and whether this
/// is a subelementalias.
struct AliasState {
  AliasState(size_t memoryIdx, bool isSubelement)
      : memoryIdx(memoryIdx), isSubelementFlag(isSubelement) {}

  bool operator==(const AliasState &o) const {
    return memoryIdx == o.memoryIdx && isSubelementFlag == o.isSubelementFlag;
  }

  bool isSubelement() const { return isSubelementFlag; }
  size_t getIdx() const { return memoryIdx; }

private:
  /// The index of the memory value contained inside RegionState that this
  /// references.
  size_t memoryIdx;
  /// When true, this is an alias to some unknown subelement of the original
  /// allocation.
  /// TODO: Properly track struct, array, and bitcast subelements.
  bool isSubelementFlag;
};

/// The "top" state is the unknown state of a lattice. This state indicates that
/// nothing is known about the state of a lattice element and we have to make
/// wholly pessimistic assumptions about the value: we don't know its constraint
/// value, its value range, and that it may alias anything.
struct TopState {
  /// This is a singleton state.
  bool operator==(const TopState &o) const { return true; }
};

/// This class represents the state of a lattice element. It is implemented as a
/// variant between the individual possible states. The current state hierarchy:
///
/// bottom -> alias -> top
class State {
public:
  /// Default the state to bottom. This is useful when defaulting constructing
  /// values from a `DenseMap`, like `state[value].join(rhs)`.
  State() : impl(BottomState{}) {}

  /// Forwarding constructor to the underlying variant. This allows the state to
  /// be constructed from the variant element types.
  template <typename T>
  State(T &&t) : impl(std::forward<T>(t)) {}

  /// Perform the join function between two states.
  bool join(const State &o, RegionState &rs);

  /// This function is called to indicate that a reference to a value of a state
  /// has been lost.
  bool drop(RegionState &rs);

  void dump();

  SmartVariant<BottomState, AliasState, TopState> impl;

private:
  bool operator==(const State &o) const { return impl == o.impl; }

  /// Assign the value of the state. This assumes the current value gets
  /// dropped.
  void assign(State o, RegionState &rs) {
    drop(rs);
    impl = std::move(o.impl);
  }
};

/// Effects on memory. We track whether a memory allocation is read, written, or
/// captured.
enum Effects {
  NONE = 0,
  READ = 1,
  WRITE = 2,
  CAPTURE = 4,
};

/// This represents the state of a piece of memory.
struct Memory {
  Memory() {}

  bool join(int effect) {
    int before = effects;
    effects |= effect;
    return before != effects;
  }

  Effects getEffects() const { return (Effects)effects; }

private:
  int effects = Effects::NONE;
};

/// This struct captures the dense state in an IR region. This class implements
/// a poor man's functional data structure to optimize the dense state: a row is
/// only created for each operation in the region that can modify the state
/// (e.g. operation with the write effect), where all other operations will
/// query the last row in a "sequence" of state transitions.
struct RegionState {
  RegionState() { sequence.emplace_back(); }

  /// Access a memory state in read mode.
  State &getReadable(size_t seqIdx, size_t memoryIdx) {
    assert(seqIdx < sequence.size());
    ArrayRef<State *> seq = sequence[seqIdx];
    assert(memoryIdx < seq.size());
    return *seq[memoryIdx];
  }
  State &getReadable(size_t seqIdx, const AliasState *alias) {
    return getReadable(seqIdx, alias->getIdx());
  }

  /// Access a memory state in write mode. This will default-initialize the
  /// state and add a new row if necessary.
  State &getWriteable(size_t seqIdx, size_t memoryIdx) {
    if (seqIdx < sequence.size()) {
      MutableArrayRef<State *> seq = sequence[seqIdx];
      assert(memoryIdx < seq.size());
      return *seq[memoryIdx];
    }
    MutableArrayRef<State *> seq = sequence.emplace_back(sequence.back());
    assert(memoryIdx < seq.size());
    State *state = allocator.Allocate<State>();
    ::new (state) State();
    seq[memoryIdx] = state;
    return *state;
  }
  State &getWriteable(size_t seqIdx, const AliasState *alias) {
    return getWriteable(seqIdx, alias->getIdx());
  }

  /// Get the memory state at the given index. The memory state is not attached
  /// to the IR and is accumulated by the analysis over the whole function.
  Memory &getMemory(size_t memoryIdx) { return memory[memoryIdx]; }
  const Memory &getMemory(size_t memoryIdx) const { return memory[memoryIdx]; }
  Memory &getMemory(const AliasState *alias) {
    return getMemory(alias->getIdx());
  }
  const Memory &getMemory(const AliasState *alias) const {
    return getMemory(alias->getIdx());
  }

  /// Add a new memory entry. This is used when processing ops that can allocate
  /// or function arguments.
  size_t addMemory(size_t seqIdx, State init) {
    memory.push_back(Memory{});
    std::vector<State *> &seq = sequence[seqIdx];
    size_t memoryIdx = seq.size();
    State *state = allocator.Allocate<State>();
    ::new (state) State(std::move(init));
    seq.push_back(state);
    return memoryIdx;
  }

  /// Apply a read effect on all current memory.
  bool readAll(size_t seqIdx) {
    bool changed = false;
    for (Memory &mem : memory)
      changed |= mem.join(Effects::READ);
    for (size_t i = 0, e = memory.size(); i != e; ++i)
      changed |= getReadable(seqIdx, i).drop(*this);
    return changed;
  }

  /// Apply a write effect on all current memory.
  bool writeAll(size_t seqIdx, RegionState &rs) {
    if (memory.empty()) {
      if (seqIdx >= sequence.size())
        sequence.emplace_back();
      return false;
    }

    bool changed = false;
    for (Memory &mem : memory)
      changed |= mem.join(Effects::WRITE);
    for (size_t i = 0, e = memory.size(); i != e; ++i) {
      changed |= getWriteable(seqIdx, i).join(TopState{}, rs);
      changed |= getReadable(seqIdx - 1, i).drop(rs);
    }
    return changed;
  }

private:
  std::vector<Memory> memory;
  /// Implement a poor man's functional data structure to represent the dense
  /// state: we organize the entries by "rows" in a sequence, adding a row only
  /// for ops that can apply the write effect to memory. As well, reduce memory
  /// and copy constructor pressure by storing references states: we only
  /// allocate new states on modification. This means that every store in the
  /// program will add a new row, and copy the pointers of all other states, but
  /// only allocate one new state. This becomes important as the size of `State`
  /// grows.
  std::vector<std::vector<State *>> sequence;
  llvm::BumpPtrAllocator allocator;
};

/// This is the analysis state of a whole function.
struct FunctionState {
  const State &lookup(Value value) const {
    const State &s = values.at(value);
    assert(!isa<BottomState>(s) && "lookup yielded overconstrained state");
    return s;
  }
  State &modify(Value value) { return values[value]; }

  /// The states of all SSA values.
  DenseMap<Value, State> values;
  /// The dense states of each region.
  DenseMap<Region *, RegionState> regions;
};
} // namespace

namespace llvm {
template <>
struct ValueIsPresent<State> {
  static bool isPresent(const State &t) { return true; }
  static decltype(auto) unwrapValue(State &t) { return t; }
};

template <typename To>
struct CastInfo<To, const State> {
  using From = const State;

  static bool isPossible(From &f) { return isa<To>(f.impl); }
  static const To *doCast(From &f) { return &cast<To>(f.impl); }
  static const To *doCastIfPossible(From &f) {
    if (!isPossible(f))
      return castFailed();
    return doCast(f);
  }
  static To *castFailed() { return nullptr; }
};

template <typename To>
struct CastInfo<To, State> {
  using From = State;

  static bool isPossible(From &f) { return isa<To>(f.impl); }
  static const To *doCast(From &f) { return &cast<To>(f.impl); }
  static const To *doCastIfPossible(From &f) {
    if (!isPossible(f))
      return castFailed();
    return doCast(f);
  }
  static To *castFailed() { return nullptr; }
};
} // namespace llvm

bool State::join(const State &o, RegionState &rs) {
  // x U x = x
  if (*this == o)
    return false;

  // top U x = top
  if (isa<TopState>(*this))
    return false;
  // x U top = top
  if (isa<TopState>(o)) {
    assign(o, rs);
    return true;
  }

  // bot U x = x
  if (isa<BottomState>(*this)) {
    assign(o, rs);
    return true;
  }
  // x U bot = x
  if (isa<BottomState>(o))
    return false;

  // We know the two states are different.
  assign(TopState{}, rs);
  return true;
}

bool State::drop(RegionState &rs) {
  if (auto *alias = dyn_cast<AliasState>(*this))
    return rs.getMemory(alias).join(Effects::CAPTURE);
  return false;
}

[[maybe_unused]] void State::dump() {
  if (isa<BottomState>(*this)) {
    llvm::dbgs() << "bottom\n";
  } else if (isa<TopState>(*this)) {
    llvm::dbgs() << "top\n";
  } else if (auto *alias = dyn_cast<AliasState>(*this)) {
    llvm::dbgs() << "alias[" << alias->getIdx()
                 << ", isSub=" << alias->isSubelement() << "]\n";
  }
}

namespace {
struct Node : public SCCNode<Node, FuncOp, KGENCallOpInterface> {
  using SCCNode::SCCNode;

  FunctionState fs;
};

struct Graph : public SCCGraph<Graph, Node> {
  bool doAnalysis(Region &region, RegionState &rs, FunctionState &fs,
                  size_t &seqIdx);
  bool doAnalysis(Node *node);
  void doRewrite(const Node *node);
};
} // namespace

bool Graph::doAnalysis(Region &region, RegionState &rs, FunctionState &fs,
                       size_t &seqIdx) {
  bool changed = false;

  for (Operation &op : region.front()) {
    if (op.getNumRegions()) {
      // TODO: handle regions
      llvm_unreachable("not handled yet");
    }

    if (isa<ArrayGEPOp, StructGEPOp, OffsetOp, PointerBitcastOp>(op)) {
      Value ptr = op.getOperand(0);
      assert(isa<PointerType>(ptr.getType()) && op.getNumResults() == 1);
      // If the pointer is tracked, the result is an alias to it.
      if (auto *alias = dyn_cast<AliasState>(fs.lookup(ptr))) {
        changed |=
            fs.modify(op.getResult(0))
                .join(AliasState(alias->getIdx(), /*isSubelement=*/true), rs);
      } else {
        // Otherwise, the state is unknown.
        changed |= fs.modify(op.getResult(0)).join(TopState{}, rs);
      }
      continue;
    }

    if (auto load = dyn_cast<LoadOp>(op)) {
      if (auto *alias = dyn_cast<AliasState>(fs.lookup(load.getPtr()))) {
        changed |= rs.getMemory(alias).join(Effects::READ);
        // If this is a load from a subelement alias, we can't read it properly.
        State &pointee = rs.getReadable(seqIdx, alias);
        if (alias->isSubelement()) {
          changed |= fs.modify(load.getResult()).join(TopState{}, rs);
          // We've lost track of the pointee.
          changed |= pointee.drop(rs);
        } else {
          // Otherwise, join the pointee state with the result state.
          changed |= fs.modify(load.getResult()).join(pointee, rs);
        }
      } else {
        // We don't have alias analysis, so this could be a load from anything.
        changed |= rs.readAll(seqIdx);
        // Otherwise, the result state is unknown.
        changed |= fs.modify(load.getResult()).join(TopState{}, rs);
      }
      continue;
    }

    if (auto store = dyn_cast<StoreOp>(op)) {
      ++seqIdx;
      if (auto *alias = dyn_cast<AliasState>(fs.lookup(store.getPtr()))) {
        changed |= rs.getMemory(alias).join(Effects::WRITE);
        State &pointee = rs.getWriteable(seqIdx, alias);
        // If this is a store to a subelement pointer, we can't track it.
        if (alias->isSubelement()) {
          changed |= pointee.join(TopState{}, rs);
          // We lost track of the value.
          changed |= fs.modify(store.getArg()).drop(rs);
          // A partial store to a memory location causes the original value to
          // be lost.
          changed |= rs.getReadable(seqIdx - 1, alias).drop(rs);
        } else {
          // Otherwise, join the value state into the pointee state.
          changed |= pointee.join(fs.lookup(store.getArg()), rs);
        }
      } else {
        // We don't have alias analysis, so this could be a store to anything.
        // We need to invalidate all our memory.
        changed |= rs.writeAll(seqIdx, rs);
        // We lost track of the value.
        changed |= fs.modify(store.getArg()).drop(rs);
      }
      continue;
    }

    if (auto alloc = dyn_cast<StackAllocationOp>(op)) {
      // If the SSA result state is BottomState, we know we haven't added the
      // memory, because we know it should always be AliasState.
      State &result = fs.modify(alloc.getResult());
      if (isa<BottomState>(result)) {
        changed |= result.join(AliasState(rs.addMemory(seqIdx, TopState{}),
                                          /*isSubelement=*/false),
                               rs);
      } else {
        assert(isa<AliasState>(result));
      }
      continue;
    }

    // TODO: handle calls
    // TODO: handle aligned_alloc + aligned_free

    // Handle unknown operations.
    assert(op.getNumRegions() == 0);

    // Handle fine-grain memory effects on unknown operations. Collect all
    // specific reads and writes, while tracking if we need to pessimistically
    // treat everything as read or written.
    bool readAll = false, writeAll = false;
    SmallVector<const AliasState *> reads, writes;
    if (auto itf = dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
      using namespace mlir::MemoryEffects;
      SmallVector<EffectInstance> effects;
      itf.getEffects(effects);
      for (EffectInstance &e : effects) {
        if (isa<Read>(e.getEffect())) {
          if (Value v = e.getValue()) {
            if (auto *alias = dyn_cast<AliasState>(fs.lookup(v))) {
              reads.push_back(alias);
              continue;
            }
          }
          readAll = true;
          continue;
        }
        if (isa<Write>(e.getEffect())) {
          if (Value v = e.getValue()) {
            if (auto *alias = dyn_cast<AliasState>(fs.lookup(v))) {
              writes.push_back(alias);
              continue;
            }
          }
          writeAll = true;
          continue;
        }
      }
    } else {
      // If the op doesn't have finer-grain memory effects, assume this op reads
      // and writes to all its operands.
      writeAll = true;
      readAll = true;
    }

    // Apply read effect to all operands.
    if (readAll) {
      readAll = false;
      for (Value v : op.getOperands()) {
        if (auto *alias = dyn_cast<AliasState>(fs.lookup(v)))
          reads.push_back(alias);
        else
          readAll = true;
      }
    }

    // Apply write effect to all operands.
    if (writeAll) {
      writeAll = false;
      for (Value v : op.getOperands()) {
        if (auto *alias = dyn_cast<AliasState>(fs.lookup(v)))
          writes.push_back(alias);
        else
          writeAll = true;
      }
    }

    // Now apply read effects first. If we only had specific reads, apply them.
    // Otherwise, apply a read on all active memory.
    if (!readAll) {
      for (const AliasState *alias : reads) {
        changed |= rs.getMemory(alias).join(Effects::READ);
        // We've lost track of the pointee.
        changed |= rs.getReadable(seqIdx, alias).drop(rs);
      }
    } else {
      changed |= rs.readAll(seqIdx);
    }
    if (!writeAll) {
      seqIdx += !writes.empty();
      for (const AliasState *alias : writes) {
        changed |= rs.getMemory(alias).join(Effects::WRITE);
        changed |= rs.getWriteable(seqIdx, alias).join(TopState{}, rs);
        if (alias->isSubelement())
          changed |= rs.getReadable(seqIdx - 1, alias).drop(rs);
      }
    } else {
      ++seqIdx;
      rs.writeAll(seqIdx, rs);
    }

    // We've lost track of all operands.
    for (Value v : op.getOperands())
      changed |= fs.modify(v).drop(rs);
    // Map all results to unknown.
    for (Value v : op.getResults())
      changed |= fs.modify(v).join(TopState{}, rs);
  }

  return changed;
}

bool Graph::doAnalysis(Node *node) {
  Region &body = node->func.getBodyRegion();
  RegionState &rs = node->fs.regions[&body];
  bool changed = false;
  for (Value v : body.getArguments()) {
    if (isa<PointerType>(v.getType())) {
      changed |= node->fs.modify(v).join(
          AliasState(rs.addMemory(/*seqIdx=*/0, TopState{}),
                     /*isSubelement=*/false),
          rs);
    } else {
      changed |= node->fs.modify(v).join(TopState{}, rs);
    }
  }
  size_t seqIdx = 0;
  changed |= doAnalysis(body, rs, node->fs, seqIdx);
  // A pointer stored to any argument pointer is considered captured.
  for (Value v : body.getArguments()) {
    if (!isa<PointerType>(v.getType()))
      continue;
    auto *alias = cast<AliasState>(node->fs.lookup(v));
    changed |= rs.getReadable(seqIdx, alias).drop(rs);
  }
  return changed;
}

void Graph::doRewrite(const Node *node) {
  FuncOp func = node->func;
  SmallVector<Attribute> attrs;
  Builder b(func.getContext());
  const RegionState &rs = node->fs.regions.at(&func.getBodyRegion());
  auto stringify = [](Effects effects) -> std::string {
    SmallVector<StringRef> strs;
    if (effects == Effects::NONE)
      strs.push_back("none");
    if (effects & Effects::READ)
      strs.push_back("read");
    if (effects & Effects::WRITE)
      strs.push_back("write");
    if (effects & Effects::CAPTURE)
      strs.push_back("cap");
    std::string str;
    llvm::raw_string_ostream os(str);
    llvm::interleave(strs, os, ",");
    return os.str();
  };
  for (Value v : func.getArguments()) {
    if (auto *alias = dyn_cast<AliasState>(node->fs.lookup(v))) {
      attrs.push_back(
          b.getStringAttr(stringify(rs.getMemory(alias).getEffects())));
    } else {
      attrs.push_back(b.getStringAttr(""));
    }
  }
  func->setAttr("ipdf", b.getArrayAttr(attrs));
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_IPDF
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct IPDF : impl::IPDFBase<IPDF> {
  using IPDFBase::IPDFBase;

  void runOnOperation() override;
};
} // namespace

void IPDF::runOnOperation() {
  const SymbolTable &symtab =
      getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();
  AsyncRT::CPUDevice &cpuDevice =
      *loadContext(&getContext())->get<AsyncRT::CPUDevice>();

  Graph g;
  g.build(getOperation(), symtab);
  g.run(cpuDevice);
}
