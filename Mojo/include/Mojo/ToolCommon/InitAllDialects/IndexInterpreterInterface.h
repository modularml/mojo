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
// This file registers all the dialects in the KGEN library.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_TOOLCOMMON_INITALLDIALECTS_INDEXINTERPRETERINTERFACE_H
#define KGEN_TOOLCOMMON_INITALLDIALECTS_INDEXINTERPRETERINTERFACE_H

#include "Mojo/Interpreter/InterpreterDialect.h"
#include "Mojo/Interpreter/InterpreterInterface.h"
#include "Mojo/Interpreter/ParametricInterpreterState.h"
#include "Mojo/Interpreter/Utils.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"

namespace M::KGEN {

template <typename Interface, typename Concrete>
struct IndexOpInterpretInterface
    : public BytecodeInterpreterOpInterface::ExternalModel<
          IndexOpInterpretInterface<Interface, Concrete>, Concrete> {

  static OpBytecodeGenerator getBytecodeGenerator() {
    return {
        0, nullptr,
        +[](Operation *op, ArrayRef<Attribute> operands, const void *payload,
            InterpreterState &state) -> ErrorTreeOrSuccess {
          if (!state.getTarget()) {
            SmallVector<OpFoldResult> foldResults;
            if (LLVM_UNLIKELY(failed(op->fold(operands, foldResults))))
              return reportFoldError(op, operands, "failed to fold operation ");

            SmallVector<Attribute> results =
                llvm::map_to_vector(foldResults, [](OpFoldResult foldResult) {
                  return cast<Attribute>(foldResult);
                });
            return state.mapResults(results);
          }
          Concrete concrete = cast<Concrete>(op);
          return Interface::interpret(concrete, operands, state);
        },
        nullptr,
        +[](Operation *op, ArrayRef<Attribute> operands, const void *payload,
            ParametricInterpreterState &state) -> ErrorTreeOrSuccess {
          if (!state.getTarget()) {
            SmallVector<OpFoldResult> foldResults;
            if (LLVM_UNLIKELY(failed(op->fold(operands, foldResults))))
              return reportFoldError(op, operands, "failed to fold operation ");

            SmallVector<Attribute> results =
                llvm::map_to_vector(foldResults, [](OpFoldResult foldResult) {
                  return cast<Attribute>(foldResult);
                });
            return state.mapResults(results);
          }
          Concrete concrete = cast<Concrete>(op);
          return Interface::parametric_interpret(concrete, operands, state);
        }};
  }
};

//===----------------------------------------------------------------------===//
// IndexOpInterpretInterface Implementations
//===----------------------------------------------------------------------===//

template <typename IndexOpT>
struct IndexOpInterpretInterfaceImplementation
    : public IndexOpInterpretInterface<
          IndexOpInterpretInterfaceImplementation<IndexOpT>, IndexOpT> {
  static ErrorTreeOrSuccess interpret(IndexOpT op, ArrayRef<Attribute> operands,
                                      InterpreterState &state);

  static ErrorTreeOrSuccess
  parametric_interpret(IndexOpT op, ArrayRef<Attribute> operands,
                       ParametricInterpreterState &state) {
    return interpret(op, operands, state);
  }
};

using CmpOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::CmpOp>;

using SubOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::SubOp>;

using ShlOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::ShlOp>;

using ShrSOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::ShrSOp>;

using ShrUOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::ShrUOp>;

using AndOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::AndOp>;

using CeilDivUOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::CeilDivUOp>;

using CeilDivSOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::CeilDivSOp>;

using DivUOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::DivUOp>;

using DivSOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::DivSOp>;

using MaxUOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::MaxUOp>;

using MaxSOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::MaxSOp>;

using MinUOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::MinUOp>;

using MinSOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::MinSOp>;

using MulOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::MulOp>;

using OrOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::OrOp>;

using RemSOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::RemSOp>;

using RemUOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::RemUOp>;

using XOrOpInterpretInterface =
    IndexOpInterpretInterfaceImplementation<mlir::index::XOrOp>;
} // namespace M::KGEN

#endif // KGEN_TOOLCOMMON_INITALLDIALECTS_INDEXINTERPRETERINTERFACE_H
