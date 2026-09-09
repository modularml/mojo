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

#ifndef SUPPORT_INTERPRETER_INTERPRETERINTERFACE_H
#define SUPPORT_INTERPRETER_INTERPRETERINTERFACE_H

#include "Support/Compiler/ErrorTree.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/OpDefinition.h"

namespace M {
class InterpreterState;
class ParametricInterpreterState;
class TargetInfoAttr;

using InterpretHook = ErrorTreeOrSuccess (*)(Operation *, ArrayRef<Attribute>,
                                             const void *, InterpreterState &);
using GenBytecodeHook = ErrorOrSuccess (*)(Operation *, void *, TargetInfoAttr);

using ParametricInterpretHook =
    ErrorTreeOrSuccess (*)(Operation *, ArrayRef<Attribute>, const void *,
                           ParametricInterpreterState &);
using ParametricGenBytecodeHook =
    ErrorOrSuccess (*)(Operation *, void *, TargetInfoAttr, ArrayRef<Attribute>,
                       ParametricInterpreterState &);

struct OpBytecodeGenerator {
  uint32_t payloadSize;
  GenBytecodeHook genBytecode;
  InterpretHook interpret;

  ParametricGenBytecodeHook genParametricBytecode;
  ParametricInterpretHook parametric_interpret;

  // The alignment of the payload. Has to be a maximum of allowed alignment of
  // all Payloads in KGEN Dialect.
  // Ideally we want to use `alignof(KGEN*::Payload)` here, but that will
  // introduce extra dependency to interpreter.
  static constexpr uint64_t payloadAlignment = 8;
};

} // namespace M

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "Mojo/Interpreter/InterpreterOpInterface.h.inc"
#include "Mojo/Interpreter/MemoryableTypeInterface.h.inc"

//===----------------------------------------------------------------------===//
// Delegate Interface Declarations
//===----------------------------------------------------------------------===//

namespace M::detail {
class InterpreterDelegateOpInterface;
class BytecodeDelegateOpInterface;

/// This class defines a delegate op interface to
/// `BytecodeInterpreterOpInterface` for operations that define a simple
/// `interpret` method with no additional bytecode payload. This is poor man's
/// interface inheritance. Most of the code here is boilerplate.
struct InterpreterDelegateOpInterfaceTraits
    : public BytecodeInterpreterOpInterfaceInterfaceTraits {
  template <typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = InterpreterDelegateOpInterface;
    Model() : Concept{getInterpretHook} {}

    /// This method defines the delegate `interpret` hook to call into the
    /// concrete operation's `interpret` method.
    static inline OpBytecodeGenerator getInterpretHook() {
      return {0, nullptr,
              +[](Operation *op, ArrayRef<Attribute> operands,
                  const void *payload, InterpreterState &state) {
                return cast<ConcreteOp>(op).interpret(operands, state);
              },
              nullptr,
              +[](Operation *op, ArrayRef<Attribute> operands,
                  const void *payload, ParametricInterpreterState &state) {
                return cast<ConcreteOp>(op).parametric_interpret(operands,
                                                                 state);
              }};
    }
  };
};

struct BytecodeDelegateOpInterfaceTraits
    : public BytecodeInterpreterOpInterfaceInterfaceTraits {
  template <typename ConcreteOp>
  class Model : public Concept {
  public:
    using Interface = InterpreterDelegateOpInterface;
    Model() : Concept{.getBytecodeGenerator = getInterpretHook} {}

    static inline OpBytecodeGenerator getInterpretHook() {
      using Payload = typename ConcreteOp::Payload;
      return {
          sizeof(Payload),
          +[](Operation *op, void *payload, TargetInfoAttr target) {
            assert(llvm::isAddrAligned(
                       llvm::Align(OpBytecodeGenerator::payloadAlignment),
                       payload) &&
                   "payload is not properly aligned");
            return cast<ConcreteOp>(op).compile(*(Payload *)payload, target);
          },
          +[](Operation *op, ArrayRef<Attribute> operands, const void *payload,
              InterpreterState &state) {
            assert(llvm::isAddrAligned(
                       llvm::Align(OpBytecodeGenerator::payloadAlignment),
                       payload) &&
                   "payload is not properly aligned");
            return cast<ConcreteOp>(op).interpret(
                operands, *(const Payload *)payload, state);
          },
          +[](Operation *op, void *payload, TargetInfoAttr target,
              ArrayRef<Attribute> operands, ParametricInterpreterState &state) {
            assert(llvm::isAddrAligned(
                       llvm::Align(OpBytecodeGenerator::payloadAlignment),
                       payload) &&
                   "payload is not properly aligned");
            return cast<ConcreteOp>(op).parametric_compile(
                *(Payload *)payload, target, operands, state);
          },

          +[](Operation *op, ArrayRef<Attribute> operands, const void *payload,
              ParametricInterpreterState &state) {
            assert(llvm::isAddrAligned(
                       llvm::Align(OpBytecodeGenerator::payloadAlignment),
                       payload) &&
                   "payload is not properly aligned");
            return cast<ConcreteOp>(op).parametric_interpret(
                operands, *(const Payload *)payload, state);
          }};
    }
  };
};

template <typename ConcreteOp>
struct InterpreterDelegateOpInterfaceTrait;
template <typename ConcreteOp>
struct BytecodeDelegateOpInterfaceTrait;

class InterpreterDelegateOpInterface : public BytecodeInterpreterOpInterface {
public:
  template <typename ConcreteOp>
  struct Trait : public InterpreterDelegateOpInterfaceTrait<ConcreteOp> {};
};
class BytecodeDelegateOpInterface : public BytecodeInterpreterOpInterface {
public:
  template <typename ConcreteOp>
  struct Trait : public BytecodeDelegateOpInterfaceTrait<ConcreteOp> {};
};

template <typename ConcreteOp>
struct InterpreterDelegateOpInterfaceTrait
    : public mlir::OpInterface<
          InterpreterDelegateOpInterface,
          InterpreterDelegateOpInterfaceTraits>::Trait<ConcreteOp> {};
template <typename ConcreteOp>
struct BytecodeDelegateOpInterfaceTrait
    : public mlir::OpInterface<
          BytecodeDelegateOpInterface,
          BytecodeDelegateOpInterfaceTraits>::Trait<ConcreteOp> {};

} // namespace M::detail

#endif // SUPPORT_INTERPRETER_INTERPRETERINTERFACE_H
