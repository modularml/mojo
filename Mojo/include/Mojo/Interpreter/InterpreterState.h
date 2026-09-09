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

#ifndef SUPPORT_INTERPRETER_INTERPRETERSTATE_H
#define SUPPORT_INTERPRETER_INTERPRETERSTATE_H

#include "Mojo/Interpreter/InterpreterAttrs.h"
#include "Support/ADT/SmartVariant.h"
#include "Support/Compiler/ErrorTree.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/Operation.h"

namespace M {

//===----------------------------------------------------------------------===//
// ReallocatingTrivialVector
//===----------------------------------------------------------------------===//

/// This data structure is a resizable vector that contains only trivial types:
/// types which have trivial destructors, constructors, and copy constructors.
/// This vector is useful for carrying a varying number of such elements,
/// resizing with minimal memory pressure.
///
/// The vector doubles in capacity each time it increases in size.
template <typename T>
class ReallocatingTrivialVector {
public:
  static_assert(std::is_trivially_copy_constructible_v<T> &&
                    std::is_trivially_move_constructible_v<T> &&
                    std::is_trivially_destructible_v<T>,
                "T must be trivial");

  /// Initialize the vector with an initial size.
  explicit ReallocatingTrivialVector(unsigned initialSize)
      : ptr((T *)malloc(sizeof(T) * initialSize)), size(initialSize) {}

  ReallocatingTrivialVector() : ptr(nullptr), size(0) {}
  ReallocatingTrivialVector(ReallocatingTrivialVector<T> &&other)
      : ptr(other.ptr), size(other.size) {
    other.ptr = nullptr;
    other.size = 0;
  }
  ~ReallocatingTrivialVector() { free(ptr); }

  /// Access the element at `i`.
  T &operator[](unsigned i) { return ptr[i]; }
  const T &operator[](unsigned i) const { return ptr[i]; }

  /// Ensure the vector can contain at least this many elements. This
  /// aggressively reserves up to the next power of 2 elements.
  void reserve(unsigned requiredSize) {
    requiredSize = llvm::NextPowerOf2(requiredSize);
    if (size >= requiredSize)
      return;
    // `realloc` copies the data over for us.
    ptr = (T *)realloc(ptr, sizeof(T) * requiredSize);
    size = requiredSize;
  }

  /// Directly access the underlying data.
  T *data() { return ptr; }
  const T *data() const { return ptr; }

private:
  T *ptr;
  unsigned size;
};

//===----------------------------------------------------------------------===//
// SoftPopStack
//===----------------------------------------------------------------------===//

/// This class is a stack data structure that does not deallocate elements when
/// `pop` is called. Instead, it keeps them alive so they can be re-used
/// later on `push`. By virtually growing and shrinking the stack, we reduce
/// memory pressure.
template <typename T>
class SoftPopStack {
public:
  size_t size() const { return idx; }
  bool empty() const { return !idx; }
  void clear() { idx = 0; }
  void reserve(size_t capacity) { stack.reserve(capacity); }

  T &back() {
    assert(!empty() && "empty stack");
    return stack[idx - 1];
  }

  T &operator[](size_t idx) {
    assert(idx < size() && "index out of bounds");
    return stack[idx];
  }

  void pop() { --idx; }

  T &push() {
    T &result = idx == stack.size() ? stack.emplace_back() : stack[idx];
    ++idx;
    return result;
  }

  ArrayRef<T> getArrayRef() const { return ArrayRef(stack).take_front(idx); }

private:
  SmallVector<T> stack;
  size_t idx = 0;
};

/// Whether `attr` may be bound to a value of type `type`; an untyped attribute
/// carries no constraint. Interpreted pointer arithmetic takes the address from
/// the attribute and the stride from the static type, so a disagreement
/// silently computes wrong offsets.
inline bool isAttributeTypeCompatible(Attribute attr, Type type) {
  auto typed = llvm::dyn_cast_if_present<mlir::TypedAttr>(attr);
  return !typed || typed.getType() == type;
}

//===----------------------------------------------------------------------===//
// InterpreterState
//===----------------------------------------------------------------------===//

/// If a region in blob of memory requires a mapping before it can be used by
/// the interpreter then we must mark that region. There are two cases where we
/// mark blob regions: (1) Pointer Region: This region contains the address of
/// another region and (2) Symbol Region: This region refers to a symbol. A
/// Symbol is not represented in the blob.
enum class RegionMark { None = 0, Pointer, Symbol };

class InterpreterState {
public:
  InterpreterState(MLIRContext *ctx, TargetInfoAttr target = nullptr);
  InterpreterState(TargetInfoAttr target);

  InterpreterState(const InterpreterState &other) = delete;
  InterpreterState(InterpreterState &&other) = default;

  virtual ~InterpreterState() = default;

  MLIRContext *getContext() const { return ctx; }

  //===--------------------------------------------------------------------===//
  // Interpreter Global State

  /// Get the interpreter target.
  TargetInfoAttr getTarget() const { return target; }

  /// Lookup the body of the referenced function. This method is made virtual so
  /// that implementors that don't have a monolithic module available can
  /// implement it differently than a symbol table lookup.
  virtual ErrorOr<Region *> lookupFunctionBody(SymbolRefAttr symbol) = 0;

  /// Lookup a type definition. This is used by operations that require an
  /// out-of-line definition of a type to interpret.
  virtual Operation *lookupTypeDefinition(SymbolRefAttr symbol) { return {}; }

  //===--------------------------------------------------------------------===//
  // Interpreter Memory Management

  /// Allocate stack memory of the request size and alignment on the current
  /// stack frame.
  ErrorOr<int64_t> allocateStackMemory(size_t size, size_t align);

  /// Allocate internal interpreter heap memory of a requested size and
  /// alignment.
  ErrorOr<int64_t> allocateHeapMemory(size_t size, size_t align);

  /// Allocate internal const global memoryof the request size and alignment,
  /// and write the attr.
  ErrorOr<int64_t> writeAttributeToConstantGlobalMemory(TypedAttr attr,
                                                        size_t size,
                                                        size_t align);

  /// Allocate internal interpreter memory for a persistent object.
  ErrorOr<int64_t> allocatePersistentMemory(size_t size, size_t align,
                                            unsigned addressSpace = 0);

  /// Free heap-allocated memory from the interpreter.
  ErrorOrSuccess freeHeapMemory(int64_t addr);

  /// Map a constant global memory handle into the interpreter if it hasn't been
  /// added yet and return the address to the start of the blob.
  ErrorOr<int64_t> mapConstGlobalMemory(MemoryHandleAttr hdl);

  /// Get writable memory for the given address to interpreter memory.
  ///
  /// The `writePointer` flag must be set if the memory is being written to as a
  /// pointer value. Pointers are special because they are the only fundamental
  /// type whose in-memory representation differs between compile-time and
  /// run-time. The interpreter implements special handling for pointer values
  /// to ensure they are mapped in to and out of the interpreter memory space
  /// when required.
  ///
  /// Clobbering a pointer region is invalid. This can either be partially
  /// overwriting a pointer region with a non-pointer value or a pointer value.
  ErrorOr<void *> getWritableMemory(int64_t addr, size_t size,
                                    RegionMark regionMark = RegionMark::None,
                                    int32_t addrSpace = 0);

  /// Get readable memory for the given address to interpreter memory.
  ErrorOr<const void *> getReadableMemory(int64_t addr, size_t size);

  /// Copy pointer and symbol region markings from [srcAddr, srcAddr+len) to
  /// [dstAddr, dstAddr+len). Must be called after a raw memcpy so that
  /// pointer-region tracking stays in sync with the copied bytes.
  ErrorOrSuccess copyMarkedRegions(int64_t srcAddr, int64_t dstAddr,
                                   size_t len);

  /// Write an attribute value of a given type to the provided chunk of memory.
  ErrorOrSuccess writeAttributeToMemory(int64_t addr, TypedAttr value);

  /// Read an attribute value of the given type from the provided chunk of
  /// memory.
  ErrorOr<TypedAttr> readAttributeFromMemory(int64_t addr, Type type);

  /// Allocate a slot in the symbol table and store the specified symbol.
  uint64_t addSymbolToSymbolTable(TypedAttr symbol);

  /// Lookup a symbol in the memory slot.
  ErrorOr<TypedAttr &> getSymbol(uint64_t slot);

  /// Read an attribute value of a given type from a PointerAttr.
  ErrorOr<Attribute> readAttributeFromPointer(Attribute pointer,
                                              Type elementType);

  /// Exchange memory references for interpreter memory references upon entering
  /// the interpreter.
  ErrorOrSuccess internalizeMemory(MutableArrayRef<Attribute> args);

  /// Exchange raw pointers to interpreter memory to dialect resource references
  /// upon exit from the interpreter.
  ///
  /// `selfStorageAddr`, when given, is the address the results were read out
  /// of. Pointers into that storage are references to the value's own memory,
  /// and are recorded on the memory model so materialization can point them at
  /// the value's runtime home instead of rebuilding the storage.
  ErrorOrSuccess
  externalizeMemory(MutableArrayRef<Attribute> results,
                    std::optional<int64_t> selfStorageAddr = std::nullopt);

  /// Load a single attribute from memory from a memref.
  virtual ErrorOr<TypedAttr> loadAttributeFromMemRef(MemRefAttr memref,
                                                     Type type);

  //===--------------------------------------------------------------------===//
  // Interpreter Control Flow

  /// Run the interpreter starting from the first operation in the entry block
  /// of the provided region given the constant values of the region arguments.
  ErrorTreeOr<SmallVector<Attribute>>
  executeRegion(Region &region, ArrayRef<Attribute> arguments);

  /// Run the interpreter starting from the provided region using a result slot
  /// calling convention. The result of the function will be the materialized
  /// memory for the result slot. The caller is required to provide the type of
  /// the result slot.
  ErrorTreeOr<TypedAttr>
  executeRegionWithResultSlot(Region &region, ArrayRef<Attribute> arguments,
                              SmartVariant<Type, TypedAttr> result);

  /// Transfer control flow to the given operation. If the operation is null,
  /// this is indicating that the interpreter should exit. Otherwise, the
  /// current return values are taken as the results of the target operation.
  virtual ErrorTreeOrSuccess
  transferControlFlowTo(Operation *target, ArrayRef<Attribute> values) = 0;

  /// Transfer control flow to the beginning of the given region with the
  /// constant values of the block arguments. The target region may belong to
  /// un-substituted IR, so its argument types can still be symbolic
  /// (`!kgen.param<T>`) while the incoming values are already concrete;
  /// implementations therefore validate the argument count but not the types.
  virtual ErrorTreeOrSuccess
  transferControlFlowTo(Region &target, ArrayRef<Attribute> arguments) = 0;

  //===--------------------------------------------------------------------===//
  // Interpreter Stack Management

  /// This function executes a function call in the interpreter, where the
  /// current operation is treated as the calling operation that the interpreter
  /// should return to when the callee returns, `body` is the region of the
  /// callee, and `arguments` are the call argument values.
  virtual ErrorTreeOrSuccess
  callFunctionBody(Region &body, ArrayRef<Attribute> arguments) = 0;

  /// Return from the current function back to the caller using `returnValues`
  /// as the return values of the function.
  virtual ErrorTreeOrSuccess
  returnFromFunction(ArrayRef<Attribute> returnValues) = 0;

  /// Return the origin operation of the frame at the given depth in the stack.
  /// If the stack is not deep enough, return null.
  virtual Operation *getOrigin(size_t depth) = 0;

  /// Set the value of a named global.
  void setNamedGlobal(StringAttr name, Attribute value) {
    namedGlobals[name] = value;
  }

  /// Get the value of a named global.
  Attribute getNamedGlobal(StringAttr name) {
    return namedGlobals.lookup(name);
  }

  //===--------------------------------------------------------------------===//
  // Interpreter Value Management

  /// Map the results of the current operation.
  virtual ErrorTreeOrSuccess mapResults(ArrayRef<Attribute> results) = 0;

  virtual ErrorTreeOrSuccess
  interpretOpWithFolder(Operation *op, ArrayRef<Attribute> operands) = 0;

  void reset() {
    resetExecutor();
    for (MemoryTable &table : memory)
      table.reset();
    symbols.clear();
  }

private:
  /// The MLIR context.
  MLIRContext *ctx;

  /// The interpreter target configuration.
  TargetInfoAttr target;

  //===--------------------------------------------------------------------===//
  // Interpreter Memory Model

  /// This struct represents a piece of memory in the interpreter.
  struct MemoryBlob {
    using MemoryT = SmartVariant<void *, MemoryHandleAttr, std::nullptr_t>;

    /// Create a memory blob. If `hdl` is null, an owned blob will be created.
    explicit MemoryBlob(llvm::BumpPtrAllocator &allocator, int64_t baseAddr,
                        size_t size, size_t align, unsigned addressSpace,
                        MemoryHandleAttr hdl, size_t refCount = 1);

    /// Mark or unmark the given region of the blob as a pointer value.
    ErrorOrSuccess setMarkedRegion(int64_t offset, int64_t regionSize,
                                   int64_t pointerSize, RegionMark regionMark);

    /// Return true if the memory is owned by the interpreter.
    bool isOwned() const { return isa<void *>(memory); }

    /// Get the handle to external memory.
    MemoryHandleAttr getHandle() const {
      return cast<MemoryHandleAttr>(memory);
    }

    /// Get the pointer to owned memory.
    void *getOwned() const { return cast<void *>(memory); }

    /// Get the pointer to the memory.
    void *getMemory() const {
      if (isOwned())
        return getOwned();
      return const_cast<void *>(
          reinterpret_cast<const void *>(getHandle().getData()));
    }

    /// Return true if the memory has been freed.
    bool isFreed() const { return isa<std::nullptr_t>(memory); }

    /// Free the owned memory.
    void free() {
      --refCount;
      if (refCount == 0)
        memory = nullptr;
    }

    /// The base address of the blob.
    int64_t baseAddr;
    /// The size of the blob.
    size_t size;
    /// The alignment of the blob.
    size_t align;
    /// The address space the blob lives in.
    unsigned addressSpace;
    /// The actual memory managed by the interpreter.
    MemoryT memory;

    /// Ref count of the blob. Multiple Memref can point to the same blob
    /// due to mlir::attr uniqueing. We need a ref count to free the blob
    /// correctly.
    size_t refCount;
    /// A bit is set for each offset value where pointer regions begin. The
    /// vector is lazily-initialized to save memory.
    std::optional<llvm::BitVector> pointerRegions;
    /// A bit is set for each offset value where a symbol region begins.
    std::optional<llvm::BitVector> symbolRegions;
  };

protected:
  /// A memory table is just a vector of blobs organized by ascending address.
  struct MemoryTable {
    MemoryTable(MemoryKind kind, int64_t minAddr, int64_t maxAddr)
        : kind(kind), minAddr(minAddr), maxAddr(maxAddr) {}

    /// Get the memory blob corresponding to the address.
    ErrorOr<MemoryBlob &> getBlob(int64_t addr);

    /// Allocate a new memory blob .
    ErrorOr<MemoryBlob &> addBlob(llvm::BumpPtrAllocator &allocator,
                                  size_t size, size_t align,
                                  unsigned addressSpace = 0,
                                  MemoryHandleAttr hdl = {},
                                  bool resetRefCount = false);

    /// Return true if the table contains the address.
    bool contains(int64_t addr) { return addr >= minAddr && addr < maxAddr; }

    /// Reset the table.
    void reset() {
      blobs.clear();
      handleToAddr.clear();
    }

    /// The kind of contained memory.
    MemoryKind kind;
    /// The base address of the table (inclusive).
    int64_t minAddr;
    /// The maximum address of the table (exclusive).
    int64_t maxAddr;
    /// The memory blobs in the table.
    std::vector<MemoryBlob> blobs;
    /// The base address of each handle-backed blob, keyed by its handle.
    /// Handle-backed blobs are never freed individually, so entries stay
    /// valid until reset().
    DenseMap<MemoryHandleAttr, int64_t> handleToAddr;
  };

private:
  /// Get the memory blob for the given address. Check that the access is
  /// in-bounds and then compute the offset into the blob.
  ErrorOr<std::pair<MemoryBlob &, int64_t>> getMemory(int64_t addr,
                                                      size_t size);

  /// Get the memory table for the memory kind.
  MemoryTable &getTable(MemoryKind kind) {
    return memory[static_cast<unsigned>(kind)];
  }

  /// This function is called to tell the interpreter executor model that an
  /// allocation is to be made on the current function frame, if there is one.
  /// This allocation needs to be freed when the function exits.
  virtual void notifyAllocationOnFrame() = 0;

protected:
  ErrorOr<PointerAttr> allocateInternalStackFor(Type type, Type ptrType);

  /// When the interpreter executor returns from a function, it needs to notify
  /// the memory model how many stack allocations need to get popped.
  void notifyReturnFromFrame(size_t numStackAllocs) {
    auto &vec = getTable(MemoryKind::Stack).blobs;
    vec.erase(vec.end() - numStackAllocs, vec.end());
  }

  // TODO: mark const once all dependencies have const versions.
  void dump();

protected:
  /// All interpreter memory tables, containing stack, heap, persistent, and
  /// constant global memory.
  MemoryTable memory[4];

  /// A structure to store values with no runtime representation.
  SmallVector<TypedAttr, 0> symbols;

private:
  /// A bump pointer allocator for fast memory allocations.
  /// TODO: Use an ArrayRecycler if memory usage is too high.
  llvm::BumpPtrAllocator allocator;

  //===--------------------------------------------------------------------===//
  // Interpreter Execution

  /// Reset the executor state.
  virtual void resetExecutor() = 0;

protected:
  virtual ErrorTreeOr<SmallVector<Attribute>>
  interpretFunction(Region &body, ArrayRef<Attribute> arguments) = 0;

  /// When the interpreter hits an error, construct an error tree given the
  /// current stack frame.
  virtual ErrorTree addStackTrace(ErrorTree error) = 0;

private:
  /// This map implements named global values. Named global values represent a
  /// mechanism for storing SSA value captures at compile time.
  DenseMap<StringAttr, Attribute> namedGlobals;
};

//===----------------------------------------------------------------------===//
// IRInterpreter
//===----------------------------------------------------------------------===//

class IRInterpreter : public InterpreterState {
public:
  using InterpreterState::InterpreterState;

  ErrorTreeOrSuccess callFunctionBody(Region &body,
                                      ArrayRef<Attribute> arguments) override {
    // Function regions are isolated from above, so push a new stack frame.
    // Then, transfer control flow to the beginning of the function body.
    pushFrame(pc.isValid() ? &*pc : nullptr, body.getParentOp());
    return transferControlFlowTo(body, arguments);
  }

  ErrorTreeOrSuccess
  returnFromFunction(ArrayRef<Attribute> returnValues) override {
    // Pop the current frame and transfer control flow back to the call
    // operation, using the operands of the return as the results of the call.
    Operation *call = popFrame();
    return transferControlFlowTo(call, returnValues);
  }

  ErrorTreeOrSuccess
  transferControlFlowTo(Operation *target,
                        ArrayRef<Attribute> values) override {
    if (target) {
      block = target->getBlock();
      pc = target->getIterator();
      return mapResults(values);
    }
    block = nullptr;
    pc = Block::iterator();
    exitValues = llvm::to_vector(values);
    return success();
  }

  ErrorTreeOrSuccess
  transferControlFlowTo(Region &target,
                        ArrayRef<Attribute> arguments) override {
    if (target.getNumArguments() != arguments.size())
      return ErrorTree(target.getLoc(),
                       "internal error: region argument count does not match "
                       "the values passed to it");
    // Argument types may still be symbolic in the un-substituted callee body,
    // so only the argument count is checked.
    for (auto [arg, value] : llvm::zip(target.getArguments(), arguments))
      mapOrOverwrite(arg, value);
    block = &target.front();
    pc = Block::iterator();
    return success();
  }

  ErrorTreeOrSuccess mapResults(ArrayRef<Attribute> results) override {
    assert(pc->getNumResults() == results.size());
    for (auto [result, value] : llvm::zip(pc->getResults(), results)) {
      if (!isAttributeTypeCompatible(value, result.getType()))
        return ErrorTree(pc->getLoc(),
                         "internal error: attribute type does not match the "
                         "result type it is bound to");
      mapOrOverwrite(result, value);
    }
    return success();
  }

private:
  ErrorTree addStackTrace(ErrorTree err) override;

  Operation *getOrigin(size_t depth) override;

  void resetExecutor() override {
    block = nullptr;
    pc = Block::iterator();
    stack.clear();
  }

  void notifyAllocationOnFrame() override {
    if (!stack.empty())
      ++getCurrentFrame().numStackAllocs;
  }

  ErrorTreeOr<SmallVector<Attribute>>
  interpretFunction(Region &body, ArrayRef<Attribute> arguments) override;

  /// A call stack frame contains the caller operation, the number of stack
  /// allocations live in the current function frame, and the value map of the
  /// function. The entry frame has a null origin. It also keeps the function
  /// operation the stack frame is for so that if an error occurs, we can emit a
  /// nice stacktrace.
  struct StackFrame {
    StackFrame() {}

    /// The operation that created the frame and invoked the function.
    Operation *origin;
    /// The corresponding function to the frame.
    Operation *func;
    /// The number of memory blobs allocated on the stack. This many blobs
    /// are popped off stack memory when the function returns.
    size_t numStackAllocs;
    /// The map of SSA values to constant values in the current frame.
    DenseMap<Value, Attribute> values;
  };

  /// Interpret a generic operation by trying to use its operation folder.
  ErrorTreeOrSuccess
  interpretOpWithFolder(Operation *op, ArrayRef<Attribute> operands) override;

  /// Push a new stack frame.
  void pushFrame(Operation *origin, Operation *func) {
    StackFrame &frame = stack.push();
    // Initialize or re-initialize the frame. This avoids unnecessary memory
    // pressure from freeing and allocating the contained DenseMap.
    frame.origin = origin;
    frame.func = func;
    frame.numStackAllocs = 0;
  }

  /// Pop the current stack frame, returning the origin operation.
  Operation *popFrame() {
    // Drop all stack memory on the current frame.
    StackFrame &frame = getCurrentFrame();
    notifyReturnFromFrame(frame.numStackAllocs);

    Operation *origin = frame.origin;
    stack.pop();
    return origin;
  }

  StackFrame &getCurrentFrame() { return stack.back(); }

  /// Map a value to a constant value, overwriting the previous value if there
  /// was one.
  void mapOrOverwrite(Value value, Attribute attr) {
    getCurrentFrame().values[value] = attr;
  }

  /// Lookup a constant value for the value.
  Attribute lookupValue(Value value) {
    Attribute attr = getCurrentFrame().values[value];
    assert(attr && "value was not mapped");
    return attr;
  }

  /// The current block being interpreter. The interpreter exits when the block
  /// is null. in which case the required invariant is that the stack frame is
  /// empty.
  Block *block = nullptr;
  /// The operation within the current block being interpreter. If the iterator
  /// is not valid, then this refers to the beginning of the block.
  Block::iterator pc;

  /// A call stack. The values in the current frame are available to the
  /// operation being interpreted.
  SoftPopStack<StackFrame> stack;

  /// Reused vector for operation operand values.
  SmallVector<Attribute> operands;

  /// Reused vector for operation payloads.
  ReallocatingTrivialVector<uint8_t> payload;

  /// The return values of the top frame of the interpreter. These are set when
  /// the interpreter is exiting from the entry function.
  SmallVector<Attribute> exitValues;
};

} // namespace M

#endif //  SUPPORT_INTERPRETER_INTERPRETERSTATE_H
