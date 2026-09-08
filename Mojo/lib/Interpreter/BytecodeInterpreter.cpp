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

#include "Mojo/Interpreter/BytecodeInterpreter.h"
#include "Mojo/Interpreter/InterpreterInterface.h"
#include "Mojo/Interpreter/Utils.h"
#include "Mojo/Support/CompilerProfiling.h"
#include "Support/AlignedAlloc.h"
#include "llvm/Support/MathExtras.h"

using namespace M;

namespace {
/// A bytecode operation operand. Operands are the only uses of SSA values. The
/// index is a reference to the SSA value being used.
struct alignas(4) BCOperand final {
  uint32_t idx;
};
/// Arguments define SSA values. This indicates indicates the SSA value the
/// argument values should be mapped to.
struct alignas(4) BCArgument final {
  uint32_t idx;
};
/// Results define SSA values. This indicates the SSA value the operation
/// results should be mapped to.
struct alignas(4) BCResult final {
  uint32_t idx;
};

/// The interpreter bytecode for a function is encoded as a series of operations
/// separated by region start markers. Each region contains a list of block
/// arguments and the bytecode offset to the first operation in the region.
///
/// KGEN operations only have single-block regions, so don't bother modeling
/// CFGs in the bytecode.
class alignas(4) BCRegion final
    : public llvm::TrailingObjects<BCRegion, BCArgument> {
public:
  BCRegion(unsigned numArgs, uint32_t firstOpOffset)
      : numArgs(numArgs), firstOpOffset(firstOpOffset) {}

  BCArgument *getArgument(unsigned i) { return getTrailingObjects() + i; }
  const BCArgument *getArgument(unsigned i) const {
    return getTrailingObjects() + i;
  }

  /// The number of region arguments.
  uint32_t numArgs;
  /// The absolute bytecode offset of the first operation in the region. Regions
  /// can never be empty.
  uint32_t firstOpOffset;

private:
  friend class llvm::TrailingObjects<BCRegion, BCArgument>;

  size_t numTrailingObjects(OverloadToken<BCArgument>) const { return numArgs; }
};

/// A preprocessed operation bytecode entry. All bytecode objects are
/// represented in-line with trailing objects to maximize cache efficiency.
///
/// The generated operation bytecode contains preprocessed information required
/// to execute the operation.
class alignas(8) BCOperation final
    : public llvm::TrailingObjects<BCOperation, BCOperand, BCResult> {
public:
  BCOperation(Operation *op, InterpretHook interpret,
              ParametricInterpretHook parametric_interpret,
              unsigned numOperands, unsigned numResults, unsigned payloadOffset)
      : op(op), interpret(interpret),
        parametric_interpret(parametric_interpret), numOperands(numOperands),
        numResults(numResults), payloadOffset(payloadOffset), nextOffset(-1) {}

  BCOperand *getOperand(unsigned i) {
    return getTrailingObjects<BCOperand>() + i;
  }
  BCResult *getResult(unsigned i) { return getTrailingObjects<BCResult>() + i; }

  const BCOperand *getOperand(unsigned i) const {
    return getTrailingObjects<BCOperand>() + i;
  }
  const BCResult *getResult(unsigned i) const {
    return getTrailingObjects<BCResult>() + i;
  }

  void *getPayload() { return (uint8_t *)this + payloadOffset; }
  const void *getPayload() const {
    return (const uint8_t *)this + payloadOffset;
  }

  void setNextOffset(uint32_t nextOffset) { this->nextOffset = nextOffset; }

  /// Backreference to the IR of the operation. This is used for error reporting
  /// and calling the folder.
  Operation *op;
  /// A precomputed interpreter hook, which is used to interpret the operation.
  /// If null, then the folder is used.
  InterpretHook interpret;

  ParametricInterpretHook parametric_interpret;

  uint32_t numOperands;
  uint32_t numResults;

  /// The relative offset from the start of the operation of the payload.
  uint32_t payloadOffset;

  /// This is the absolute bytecode offset to the next operation in the current
  /// basic block.
  uint32_t nextOffset;

private:
  friend class llvm::TrailingObjects<BCOperation, BCOperand, BCResult>;

  size_t numTrailingObjects(OverloadToken<BCOperand>) const {
    return numOperands;
  }
  size_t numTrailingObjects(OverloadToken<BCResult>) const {
    return numResults;
  }
};

/// This is a helper class for writing bytecode.
class BytecodeStream {
  /// Allocate in multiples of 32KB to reduce memory pressure at the cost of
  /// over-allocating.
  static constexpr size_t chunkSize = 256 * 128;
  /// Meet the strictest alignment requirements of bytecode objects.
  static constexpr size_t chunkAlign =
      std::max(alignof(BCOperation), alignof(BCRegion));

public:
  BytecodeStream()
      : data(alignedAlloc(chunkAlign, chunkSize)), capacity(chunkSize),
        offset(0) {}

  ~BytecodeStream() { assert(!data && "must be consumed"); }

  /// Return the offset and pointer to append a bytecode object of the given
  /// type and size to the stream.
  template <typename T>
  std::pair<T *, size_t> next(size_t size) {
    // Align all writes.
    offset = llvm::alignTo<chunkAlign>(offset);
    ensureCapacity(size);
    void *result = (uint8_t *)data + offset;
    size_t idx = offset;
    offset += size;
    return {(T *)result, idx};
  }

  /// Lookup the bytecode object at the given index. This is needed because
  /// appending objects may resize the blob.
  template <typename T>
  T *at(size_t idx) {
    return (T *)((uint8_t *)data + idx);
  }

  /// Consume the stream and take its blob.
  void *take() && {
    void *result = data;
    data = nullptr;
    return result;
  }

  /// Get the offset the next object would be written at.
  size_t getNextOffset() { return next<char>(0).second; }

private:
  void ensureCapacity(size_t size) {
    size_t end = offset + size;
    if (end <= capacity)
      return;

    // Keep doubling the chunk size until it fits the end offset. Use
    // `NextPowerOf2` instead of `PowerOf2Ceil` to bias towards over-allocating.
    size_t newCapacity =
        chunkSize * llvm::NextPowerOf2(llvm::divideCeil(end, chunkSize));
    void *newData = alignedAlloc(chunkAlign, newCapacity);
    memcpy(newData, data, capacity);
    alignedFree(data);
    data = newData;
    capacity = newCapacity;
  }

  /// The blob being written to.
  void *data;
  /// The current size of the blob allocation.
  size_t capacity;
  /// The current offset being written to.
  size_t offset;
};

/// Helper class for writing bytecode recursively.
class BytecodeBuilder {
public:
  explicit BytecodeBuilder(TargetInfoAttr target) : target(target) {}

  /// Write an operation.
  ErrorTreeOrSuccess writeOperation(Operation *op);
  /// Write a region.
  ErrorTreeOrSuccess writeRegion(Region &region);

  auto take() && {
    return std::make_tuple(valueCounter, std::move(stream).take(),
                           std::move(cfgIndices));
  }

private:
  TargetInfoAttr target;

  BytecodeStream stream;
  DenseMap<Value, unsigned> valueMap;
  unsigned valueCounter = 0;
  DenseMap<const void *, uint32_t> cfgIndices;
};
}; // namespace

ErrorTreeOrSuccess BytecodeBuilder::writeOperation(Operation *op) {
  // Compute the inline size of the operation
  unsigned numOperands = op->getNumOperands();
  unsigned numResults = op->getNumResults();
  unsigned numRegions = op->getNumRegions();
  size_t totalSize = BCOperation::totalSizeToAlloc<BCOperand, BCResult>(
      numOperands, numResults);

  // Get the interpret hook for the operation. If it does not implement the
  // interpreter interface, use the operation folder.
  OpBytecodeGenerator generator{.payloadSize = 0,
                                .genBytecode = nullptr,
                                .interpret = nullptr,
                                .genParametricBytecode = nullptr,
                                .parametric_interpret = nullptr};

  if (auto itf = dyn_cast<BytecodeInterpreterOpInterface>(op))
    generator = itf.getBytecodeGenerator();

  if (generator.genBytecode)
    totalSize = llvm::alignTo(totalSize, generator.payloadAlignment);

  // Create the operation first. This saves the computed properties of the
  // operation required by the interpreter.
  auto [bc, bcIdx] =
      stream.next<BCOperation>(totalSize + generator.payloadSize);
  ::new (bc)
      BCOperation(op, generator.interpret, generator.parametric_interpret,
                  numOperands, numResults, totalSize);

  // Now create the trailing objects.
  for (unsigned i = 0; i != numOperands; ++i) {
    Value operand = op->getOperand(i);
    ::new (bc->getOperand(i)) BCOperand{valueMap.at(operand)};
  }
  for (unsigned i = 0; i != numResults; ++i) {
    Value result = op->getResult(i);
    // Allocate a spot for the result.
    uint32_t idx = valueCounter++;
    valueMap.try_emplace(result, idx);
    ::new (bc->getResult(i)) BCResult{idx};
  }

  // Generate the payload if necessary.
  if (GenBytecodeHook genBytecode = generator.genBytecode)
    if (auto err = genBytecode(op, bc->getPayload(), target))
      return ErrorTree(op->getLoc(), err.takeError());

  // Write the regions first.
  for (unsigned i = 0; i != numRegions; ++i) {
    Region &region = op->getRegion(i);
    if (auto err = writeRegion(region))
      return err.takeError();
  }

  // If the op has regions, we have to save it in case an op branches back.
  if (numRegions)
    cfgIndices.try_emplace(op, bcIdx);

  // Save the offset of where the next thing will be written.
  stream.at<BCOperation>(bcIdx)->setNextOffset(stream.getNextOffset());

  return success();
}

ErrorTreeOrSuccess BytecodeBuilder::writeRegion(Region &region) {
  unsigned numArgs = region.getNumArguments();
  size_t totalSize = BCRegion::totalSizeToAlloc<BCArgument>(numArgs);

  auto [bc, bcIdx] = stream.next<BCRegion>(totalSize);
  ::new (bc) BCRegion(numArgs, stream.getNextOffset());

  for (unsigned i = 0; i != numArgs; ++i) {
    Value arg = region.getArgument(i);
    // Allocate a spot for the argument.
    uint32_t idx = valueCounter++;
    valueMap.try_emplace(arg, idx);
    ::new (bc->getArgument(i)) BCArgument{idx};
  }

  assert(llvm::hasSingleElement(region));
  for (Operation &op : region.front())
    if (auto err = writeOperation(&op))
      return err.takeError();

  cfgIndices.try_emplace(&region, bcIdx);

  return success();
}

//===----------------------------------------------------------------------===//
// FunctionIRBytecode
//===----------------------------------------------------------------------===//

FunctionIRBytecode::~FunctionIRBytecode() {
  if (data)
    alignedFree(data);
}

ErrorTreeOr<FunctionIRBytecode>
FunctionIRBytecode::compile(Region &entry, TargetInfoAttr target) {
  BytecodeBuilder builder(target);
  if (auto err = builder.writeRegion(entry)) {
    (void)std::move(builder).take();
    return err.takeError();
  }

  auto [numValues, data, cfgIndices] = std::move(builder).take();
  return FunctionIRBytecode(numValues, data, std::move(cfgIndices));
}

//===----------------------------------------------------------------------===//
// BytecodeInterpreter
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess
BytecodeInterpreter::callFunctionBody(Region &body,
                                      ArrayRef<Attribute> arguments) {
  if constexpr (KGEN::kIsTracingEnabled) {
    KGEN::InterpreterProfilerEntry::createAndPush(
        "Interpret", [&]() -> std::string {
          std::string detail;
          llvm::raw_string_ostream os(detail);
          if (auto sym = dyn_cast<mlir::SymbolOpInterface>(body.getParentOp()))
            os << sym.getName();
          if (!arguments.empty()) {
            os << "\nargs:";
            for (Attribute arg : arguments)
              os << "\n  " << arg;
          }
          return detail;
        });
  }

  // Push a new frame.
  StackFrame &newFrame = stack.push();
  newFrame.func = body.getParentOp();
  newFrame.numStackAllocs = 0;
  newFrame.numSymbolicAllocs = 0;

  // Save the current offset and function bytecode reference.
  newFrame.origin = pc;
  newFrame.originBc = bc;

  // Now request bytecode for the callee.
  ErrorTreeOr<const FunctionIRBytecode *> newBcOr =
      bcCompiler->compileBytecode(body);
  if (newBcOr.isError())
    return newBcOr.takeError();
  const FunctionIRBytecode *newBc = newBcOr.takeValue();

  // Pre-allocate enough space to hold all the intermediate values.
  newFrame.values.reserve(newBc->numValues);

  // Enter the callee region and map the arguments in.
  bc = newBc;
  values = newFrame.values.data();
  auto *bcRegion = bc->at<BCRegion>(0);
  for (unsigned i = 0, e = bcRegion->numArgs; i != e; ++i) {
    uint32_t idx = bcRegion->getArgument(i)->idx;
    values[idx] = arguments[i];
  }

  pc = bcRegion->firstOpOffset;
  didTransfer = true;

  return success();
}

ErrorTreeOrSuccess
BytecodeInterpreter::returnFromFunction(ArrayRef<Attribute> returnValues) {
  if constexpr (KGEN::kIsTracingEnabled)
    KGEN::InterpreterProfilerEntry::endAndPop();

  StackFrame &frame = getCurrentFrame();
  notifyReturnFromFrame(frame.numStackAllocs);

  // Restore the program counter and bytecode.
  pc = frame.origin;
  didTransfer = true;
  bc = frame.originBc;
  stack.pop();

  // If `bc` is null, this is return from the entry function.
  if (!bc) {
    assert(pc == (uint32_t)-1);
#ifndef NDEBUG
    values = nullptr;
#endif
    // Set the return values as the interpreter exit values.
    exitValues = returnValues;
    return success();
  }

  // Restore the values vector and map the call results.
  values = getCurrentFrame().values.data();
  auto *op = bc->at<BCOperation>(pc);
  for (unsigned i = 0, e = op->numResults; i != e; ++i) {
    uint32_t idx = op->getResult(i)->idx;
    if (!isAttributeTypeCompatible(returnValues[i],
                                   op->op->getResult(i).getType()))
      return ErrorTree(op->op->getLoc(),
                       "internal error: returned attribute type does not match "
                       "the call result type it is bound to");
    values[idx] = returnValues[i];
  }

  // Advance to the next instruction.
  pc = op->nextOffset;
  return success();
}

ErrorTreeOrSuccess
BytecodeInterpreter::transferControlFlowTo(Operation *target,
                                           ArrayRef<Attribute> results) {
  assert(bc && "expected an active function frame");

  // FIXME: This is the only hashmap lookup in the bytecode interpreter. Is
  // there some way we can remove ths?
  auto it = bc->cfgIndices.find(target);
  assert(it != bc->cfgIndices.end() &&
         "target operation does not have any regions");
  uint32_t parentOffset = it->second;

  // Map the results of the parent operation.
  auto *op = bc->at<BCOperation>(parentOffset);
  assert(op->numResults == results.size() && "result count mismatch");

  for (unsigned i = 0, e = op->numResults; i != e; ++i) {
    uint32_t idx = op->getResult(i)->idx;
    if (!isAttributeTypeCompatible(results[i], op->op->getResult(i).getType()))
      return ErrorTree(op->op->getLoc(),
                       "internal error: attribute type does not match the "
                       "result type it is bound to");
    values[idx] = results[i];
  }

  // Advance to the next operation.
  pc = op->nextOffset;
  didTransfer = true;
  return success();
}

ErrorTreeOrSuccess
BytecodeInterpreter::transferControlFlowTo(Region &target,
                                           ArrayRef<Attribute> arguments) {
  assert(bc && "expected an active function frame");

  // FIXME: This is the only hashmap lookup in the bytecode interpreter. Is
  // there some way we can remove this?
  auto it = bc->cfgIndices.find(&target);
  assert(it != bc->cfgIndices.end() && "region key missing");
  uint32_t offset = it->second;

  // Map the entry arguments of the region.
  auto *bcRegion = bc->at<BCRegion>(offset);
  assert(bcRegion->numArgs == arguments.size() && "arg count mismatch");
  // Argument types may still be symbolic in the un-substituted callee body, so
  // only the argument count is checked.
  for (unsigned i = 0, e = bcRegion->numArgs; i != e; ++i) {
    uint32_t idx = bcRegion->getArgument(i)->idx;
    values[idx] = arguments[i];
  }

  // Advance to the first operation.
  pc = bcRegion->firstOpOffset;
  didTransfer = true;
  return success();
}

ErrorTreeOrSuccess
BytecodeInterpreter::mapResults(ArrayRef<Attribute> results) {
  assert(bc && "expected an active function frame");

  auto *op = bc->at<BCOperation>(pc);
  assert(op->numResults == results.size() && "result count mismatch");

  for (uint32_t i = 0, e = op->numResults; i != e; ++i) {
    uint32_t idx = op->getResult(i)->idx;
    if (!isAttributeTypeCompatible(results[i], op->op->getResult(i).getType()))
      return ErrorTree(op->op->getLoc(),
                       "internal error: attribute type does not match the "
                       "result type it is bound to");
    values[idx] = results[i];
  }
  return success();
}

ErrorTree BytecodeInterpreter::addStackTrace(ErrorTree err) {
  return addStackTraceImpl(
      std::move(err), stack.getArrayRef(),
      [](const StackFrame &frame) -> Operation * {
        if (frame.originBc)
          return frame.originBc->at<BCOperation>(frame.origin)->op;
        return nullptr;
      });
}

Operation *BytecodeInterpreter::getOrigin(size_t depth) {
  // Lookup the callee at `depth` and return it.
  if (depth >= stack.size())
    return nullptr;
  StackFrame &frame = stack[stack.size() - 1 - depth];
  if (!frame.originBc)
    return nullptr;
  return frame.originBc->at<BCOperation>(frame.origin)->op;
}

void BytecodeInterpreter::resetExecutor() {
  stack.clear();
  pc = -1;
  bc = nullptr;
  values = nullptr;
}

void BytecodeInterpreter::notifyAllocationOnFrame() {
  ++getCurrentFrame().numStackAllocs;
}

ErrorTreeOr<SmallVector<Attribute>>
BytecodeInterpreter::interpretFunction(Region &body,
                                       ArrayRef<Attribute> arguments) {
  // Enter the entry function.
  if (auto err = callFunctionBody(body, arguments))
    return err.takeError();
  didTransfer = false;

  while (bc) {
    // Offset to the current instruction.
    auto *op = bc->at<BCOperation>(pc);

    // Load the required operands.
    uint32_t numOperands = op->numOperands;
    operands.reserve(numOperands);
    for (uint32_t i = 0; i != numOperands; ++i) {
      uint32_t idx = op->getOperand(i)->idx;
      operands[i] = values[idx];
    }

    ArrayRef<Attribute> operandsRef(operands.data(), numOperands);
    // Use the interpreter interface if one was found.
    if (InterpretHook interpret = op->interpret) {
      ErrorTreeOrSuccess err =
          interpret(op->op, operandsRef, op->getPayload(), *this);
      if (LLVM_UNLIKELY(err.isError())) {
        return reportFoldError(op->op, operandsRef,
                               "failed to interpret operation ")
            .addCause(err.takeError());
      }
    } else {
      results.clear();
      // Otherwise, use the folder.
      Operation *mlirOp = op->op;
      if (LLVM_UNLIKELY(failed(mlirOp->fold(operandsRef, results)))) {
        return reportFoldError(mlirOp, operandsRef,
                               "failed to fold operation ");
      }

      for (unsigned i = 0, e = op->numResults; i != e; ++i) {
        uint32_t idx = op->getResult(i)->idx;
        auto value = dyn_cast<Attribute>(results[i]);
        // The bytecode interpreter doesn't support arbitrary value returns.
        // Enforce that the folder returned an attribute!
        if (LLVM_UNLIKELY(!value)) {
          llvm::report_fatal_error(
              "INTERNAL ERROR: operation '" + mlirOp->getName().getStringRef() +
              "' folder returned a relative value in the interpreter");
        }
        values[idx] = value;
      }
    }

    // Advance to the next operation if the op didn't change control flow
    // itself.
    if (!didTransfer)
      pc = op->nextOffset;
    didTransfer = false;
  }

  if (LLVM_UNLIKELY(!stack.empty())) {
    llvm::report_fatal_error(
        "exiting interpreter with remaining stack frames " +
        Twine(stack.size()));
  }
  return llvm::to_vector(exitValues);
}

ErrorTreeOrSuccess
BytecodeInterpreter::interpretOpWithFolder(Operation *op,
                                           ArrayRef<Attribute> operands) {
  SmallVector<OpFoldResult> results;
  // Otherwise, use the folder.
  if (LLVM_UNLIKELY(failed(op->fold(operands, results)))) {
    return reportFoldError(op, operands, "failed to fold operation ");
  }
  SmallVector<Attribute> resultAttrs;
  resultAttrs.reserve(results.size());

  for (auto result : results) {
    auto value = dyn_cast<Attribute>(result);
    // The bytecode interpreter doesn't support arbitrary value returns.
    // Enforce that the folder returned an attribute!
    if (LLVM_UNLIKELY(!value)) {
      llvm::report_fatal_error(
          "INTERNAL ERROR: operation '" + op->getName().getStringRef() +
          "' folder returned a relative value in the interpreter");
    }
    resultAttrs.push_back(value);
  }
  return mapResults(resultAttrs);
}
