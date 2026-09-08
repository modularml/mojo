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

#include "Mojo/Interpreter/ParametricInterpreterState.h"
#include "Mojo/Interpreter/InterpreterInterface.h"
#include "Mojo/Interpreter/Utils.h"
#include "llvm/ADT/ScopeExit.h"

using namespace M;

ParametricInterpreterState::ParametricInterpreterState(MLIRContext *ctx,
                                                       unsigned maxDepth,
                                                       TargetInfoAttr target)
    : InterpreterState(ctx, target), maxDepth(maxDepth) {}
ParametricInterpreterState::ParametricInterpreterState(unsigned maxDepth,
                                                       TargetInfoAttr target)
    : InterpreterState(target), maxDepth(maxDepth) {}

/// Execute a region that has a ByRefResult argument.
ErrorTreeOr<TypedAttr> ParametricInterpreterState::executeRegionWithResultSlot(
    Region &region, ArrayRef<Attribute> arguments,
    SmartVariant<Type, TypedAttr> resultValue, Type resultPtrType) {

  Location loc = region.getLoc();
  if (region.getArguments().empty())
    return ErrorTree(loc, "internal error: region has no arguments");
  if (!getTarget())
    return ErrorTree(loc, Error("call into memory requires a target model"));

  ErrorOr<PointerAttr> resultSlotAttrOr =
      allocateInternalStackFor(cast<Type>(resultValue), resultPtrType);
  if (resultSlotAttrOr.isError()) {
    return ErrorTree(loc, resultSlotAttrOr.takeError());
  }
  TypedAttr resultSlotAttr = resultSlotAttrOr.takeValue();

  SmallVector<Attribute> allArgs;
  llvm::append_range(allArgs, arguments);
  allArgs.push_back(resultSlotAttr);

  // Internalize memory inside function arguments.
  if (ErrorOrSuccess err = internalizeMemory(allArgs); err.isError())
    return ErrorTree(region.getLoc(), err.takeError());

  // Reset the interpret to a clean state.
  auto resetState = llvm::scope_exit([&] { reset(); });

  // Run the interpreter.
  ErrorTreeOr<SmallVector<Attribute>> result =
      interpretFunction(region, allArgs);

  // The interpreter ran into an error. Report an error using a stacktrace.
  if (result.isError()) {
    return addStackTrace(result.takeError());
  }

  int64_t resultSlotAddr = cast<PointerAttr>(resultSlotAttr).getAddr();
  ErrorOr<TypedAttr> resultOr =
      readAttributeFromMemory(resultSlotAddr, cast<Type>(resultValue));
  if (resultOr.isError())
    return ErrorTree(loc, resultOr.takeError());
  TypedAttr value = resultOr.takeValue();

  if (ErrorOrSuccess err = externalizeMemory(value, resultSlotAddr);
      err.isError())
    return ErrorTree(region.getLoc(), err.takeError());
  return value;
}

ErrorOr<TypedAttr>
ParametricInterpreterState::loadAttributeFromMemRef(MemRefAttr memref,
                                                    Type type) {
  Attribute attr = memref;
  if (ErrorOrSuccess err = internalizeMemory(attr))
    return err.takeError();
  ErrorOr<TypedAttr> attrOr =
      readAttributeFromMemory(cast<PointerAttr>(attr).getAddr(), type);
  if (attrOr.isError())
    return attrOr.takeError();
  SmallVector<Attribute> results;
  results.push_back(attrOr.takeValue());
  if (ErrorOrSuccess errorMaybe = externalizeMemory(results))
    return errorMaybe.takeError();
  Attribute result = results.front();
  return ::cast<TypedAttr>(result);
}

//===----------------------------------------------------------------------===//
// ParametricIRInterpreter
//===----------------------------------------------------------------------===//

ErrorTree ParametricIRInterpreter::addStackTrace(ErrorTree error) {
  return addStackTraceImpl(
      std::move(error), stack.getArrayRef(),
      [](const StackFrame &frame) { return frame.origin; });
}

ErrorTreeOrSuccess
ParametricIRInterpreter::interpretOpWithFolder(Operation *op,
                                               ArrayRef<Attribute> operands) {
  SmallVector<OpFoldResult> results;
  if (failed(op->fold(operands, results)))
    return reportFoldError(op, operands, "failed to fold operation ");
  for (auto [result, output] : llvm::zip(results, op->getResults())) {
    if (auto value = llvm::dyn_cast<Attribute>(result))
      mapOrOverwrite(output, value);
    else
      mapOrOverwrite(output, lookupValue(cast<Value>(result)));
  }
  return success();
}

ErrorTreeOr<SmallVector<Attribute>>
ParametricIRInterpreter::interpretFunction(Region &body,
                                           ArrayRef<Attribute> arguments) {
  if (auto err = callFunctionBody(body, arguments))
    return err.takeError();

  while (block) {
    // Advance the iterator.
    if (pc.isValid())
      ++pc;
    else
      pc = block->begin();

    operands.clear();
    // Lookup the operands of the current operation.
    for (Value operand : pc->getOperands())
      operands.push_back(lookupValue(operand));

    // Check for an interpreter interface implementation.
    Operation &op = *pc;
    setIsCurrOpParam(&op);

    if (auto interpItf = dyn_cast<BytecodeInterpreterOpInterface>(op)) {
      OpBytecodeGenerator gen = interpItf.getBytecodeGenerator();

      if (ParametricGenBytecodeHook genBytecode = gen.genParametricBytecode) {
        payload.reserve(gen.payloadSize);
        if (auto err =
                genBytecode(&op, payload.data(), getTarget(), operands, *this))
          return ErrorTree(op.getLoc(), err.takeError());
      }
      ErrorTreeOrSuccess err =
          interpItf.getBytecodeGenerator().parametric_interpret(
              &op, operands, payload.data(), *this);
      if (err.isError())
        return reportFoldError(&*pc, operands, "failed to interpret operation ")
            .addCause(err.takeError());

      // Otherwise, try to use the operation folder.
    } else if (ErrorTreeOrSuccess result =
                   interpretOpWithFolder(&*pc, operands)) {
      return result.takeError();
    }
  }

  // The stack frame must be empty.
  if (LLVM_UNLIKELY(!stack.empty())) {
    llvm::report_fatal_error(
        "exiting interpreter with remaining stack frames " +
        Twine(stack.size()));
  }
  return std::move(exitValues);
}

Operation *ParametricIRInterpreter::getOrigin(size_t depth) {
  if (depth >= stack.size())
    return nullptr;
  return stack[stack.size() - 1 - depth].origin;
}
