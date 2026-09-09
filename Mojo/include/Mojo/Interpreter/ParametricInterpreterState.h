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

#ifndef KGEN_INTERPRETER_PARAMETRICINTERPRETERSTATE_H
#define KGEN_INTERPRETER_PARAMETRICINTERPRETERSTATE_H

#include "Mojo/Interpreter/InterpreterState.h"

namespace M {

class ParametricInterpreterState : public InterpreterState {
public:
  ParametricInterpreterState(MLIRContext *ctx, unsigned maxDepth,
                             TargetInfoAttr target = nullptr);
  ParametricInterpreterState(unsigned maxDepth, TargetInfoAttr target);

  ParametricInterpreterState(const ParametricInterpreterState &other) = delete;
  ParametricInterpreterState(ParametricInterpreterState &&other) = default;

  virtual ~ParametricInterpreterState() = default;

  virtual ErrorOr<std::pair<Region *, Operation *>>
  lookupParametricFunctionBody(SymbolRefAttr symbol) = 0;

  virtual ErrorOr<Type> lookupFuncTypeGenerator(SymbolRefAttr symbol) = 0;

  /// Run the interpreter starting from the provided region using a result slot
  /// calling convention. The result of the function will be the materialized
  /// memory for the result slot. The caller is required to provide the type of
  /// the result slot.
  ErrorTreeOr<TypedAttr>
  executeRegionWithResultSlot(Region &region, ArrayRef<Attribute> arguments,
                              SmartVariant<Type, TypedAttr> result,
                              Type resultPtrType);

  virtual ErrorTreeOrSuccess
  transferControlFlowToParent(Operation *target,
                              ArrayRef<Attribute> values) = 0;

  virtual ErrorOr<TypedAttr> loadAttributeFromMemRef(MemRefAttr memref,
                                                     Type type) override;

  //===--------------------------------------------------------------------===//
  // APIs for interpret parametrically

  /// Get the specified type with any nested parameter expressions rewritten.
  virtual Type getReboundType(Type type) = 0;
  virtual Type getReboundTypeAlways(Type type) = 0;

  /// Get the specified attribute with any nested parameter expressions
  /// rewritten.
  virtual Attribute getReboundAttribute(Attribute attr) = 0;

  /// Get the specified attribute with any nested parameter expressions
  /// rewritten.
  virtual TypedAttr getReboundAttribute(TypedAttr attr) = 0;

  virtual TypedAttr getFailableReboundAttribute(TypedAttr attr) = 0;

  virtual void setDeclBinding(Attribute decl, TypedAttr value,
                              bool overwrite = false) = 0;

  virtual bool overwriteDeclBinding(Attribute decl, TypedAttr value) = 0;

  virtual ErrorTreeOr<TypedAttr>
  interpretGenerator(Attribute calleeAttr,
                     llvm::ArrayRef<TypedAttr> paramValues,
                     ArrayRef<Attribute> arguments, Location loc) = 0;

  virtual ErrorTreeOr<TypedAttr> interpretGeneratorWithResultSlot(
      Attribute calleeAttr, llvm::ArrayRef<TypedAttr> paramValues,
      ArrayRef<Attribute> arguments, Location loc) = 0;

  virtual void
  setDeclBindings(const DenseMap<StringAttr, TypedAttr> &values) = 0;
  virtual void setDeclBindings(Operation *gen, ArrayRef<TypedAttr> values) = 0;

  virtual void clearParameterCache() = 0;
  virtual void pushEvalFrame(Operation *op, Region *region,
                             llvm::ArrayRef<TypedAttr> paramValues, int id) = 0;
  virtual void popEvalFrame() = 0;
  virtual void popEvalFrame(size_t size) = 0;
  virtual void pushParamValues(llvm::ArrayRef<TypedAttr> values, bool pushFrame,
                               Operation *op = nullptr) = 0;
  virtual void appendParamValues(llvm::ArrayRef<TypedAttr> values, int id,
                                 Operation *op) = 0;
  virtual void popParamValues(bool popFrame, Operation *op,
                              Operation *tillOp = nullptr) = 0;

  virtual void dumpParams() = 0;
  virtual void *currentEvaluator() = 0;
  virtual void *currentFrame() = 0;
  virtual size_t numParamEvals() = 0;
  virtual void setRewritten(
      const DenseMap<std::pair<size_t, const void *>, const void *> &) = 0;

  // Some ops like ParamFor has states that's not directly
  // reflected in the IR. Keep track of those here.
  struct OpSideEffectState {
    Attribute iterator;
    SmallVector<Attribute> operands;
    Attribute nextIterator;
  };

  virtual DenseMap<Operation *, OpSideEffectState> &currOpSideEffectState() = 0;
  virtual DenseSet<Operation *> *getParamOps(Operation *op,
                                             std::string &name) = 0;
  virtual void setIsCurrOpParam(Operation *op) = 0;

  void resetMemory() {
    for (MemoryTable &table : memory)
      table.reset();
    symbols.clear();
  }

  // max depth for the stack frame.
  unsigned maxDepth = std::numeric_limits<unsigned>::max();

  bool isCurrOpParam = true;
};

//===----------------------------------------------------------------------===//
// ParametricIRInterpreter
//===----------------------------------------------------------------------===//

class ParametricIRInterpreter : public ParametricInterpreterState {
public:
  using ParametricInterpreterState::ParametricInterpreterState;

  ErrorTreeOrSuccess callFunctionBody(Region &body,
                                      ArrayRef<Attribute> arguments) override {
    // Function regions are isolated from above, so push a new stack frame.
    // Then, transfer control flow to the beginning of the function body.
    if (stack.size() + nestedStackDepth > maxDepth) {
      return ErrorTree(body.getLoc(), "interpreter is " + Twine(maxDepth + 1) +
                                          " levels deep - infinite recursion?");
    }

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
  transferControlFlowToParent(Operation *target,
                              ArrayRef<Attribute> values) override {
    block = target->getBlock();
    pc = target->getIterator();
    if (ErrorTreeOrSuccess err = mapResults(values); err.isError())
      return err;
    pc--;
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
    // No type check here: this interpreter runs on un-substituted IR, where a
    // result type can still be parametric while the value is already concrete.
    for (auto [result, value] : llvm::zip(pc->getResults(), results))
      mapOrOverwrite(result, value);
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

    ///// The map of parameter values;
    void *paramEvaluator;
    size_t numParamEvals;

    DenseMap<Operation *, OpSideEffectState> opSideEffectState;

    DenseSet<Operation *> *paramOps = nullptr;
    std::string genName;
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
    frame.paramEvaluator = currentEvaluator();
    frame.numParamEvals = numParamEvals();
    frame.paramOps = getParamOps(func, frame.genName);
  }

  /// Pop the current stack frame, returning the origin operation.
  Operation *popFrame() {
    //  Drop all stack memory on the current frame.
    StackFrame &frame = getCurrentFrame();
    notifyReturnFromFrame(frame.numStackAllocs);
    popEvalFrame(frame.numParamEvals);
    popParamValues(true, frame.origin);
    stack.pop();
    return frame.origin;
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

protected:
  /// A call stack. The values in the current frame are available to the
  /// operation being interpreted.
  SoftPopStack<StackFrame> stack;

  unsigned nestedStackDepth = 0;

private:
  /// Reused vector for operation operand values.
  SmallVector<Attribute> operands;

  /// Reused vector for operation payloads.
  ReallocatingTrivialVector<uint8_t> payload;

  /// The return values of the top frame of the interpreter. These are set when
  /// the interpreter is exiting from the entry function.
  SmallVector<Attribute> exitValues;
};

} // namespace M

#endif // KGEN_INTERPRETER_PARAMETRICINTERPRETERSTATE_H
