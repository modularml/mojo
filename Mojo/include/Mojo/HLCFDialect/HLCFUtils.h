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

#ifndef KGEN_HLCFDIALECT_HLCFUTILS_H
#define KGEN_HLCFDIALECT_HLCFUTILS_H

#include "Mojo/HLCFDialect/HLCFOps.h"
#include "mlir/IR/OpImplementation.h"

namespace M::HLCF {
/// Return true if the operation is a loop and has a matching label.
bool isMatchingLoop(Operation *op, StringAttr label);

/// Return the nearest enclosing matching loop. This runs on valid IR, so it
/// must find a matching loop.
LoopOp getParentLoop(Operation *op, StringAttr label);

/// Check if the child loop is nested in the parentToCheck loop.
bool isParentLoop(LoopOp child, LoopOp parentToCheck);

/// Get the parent operation of a terminator.
Operation *getParentNode(HLCF::ControlFlowTerminator term);

/// Given an elif op, transform into multiple IfOps. Return top IfOp.
IfOp replaceElifWithIfOps(ElifOp elifOp);

ParseResult parseLoop(OpAsmParser &p,
                      SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
                      SmallVectorImpl<Type> &operandTypes,
                      SmallVectorImpl<Type> &resultTypes, Region &body);
void printLoop(OpAsmPrinter &p, Operation *op, ValueRange operands,
               TypeRange operandTypes, TypeRange resultTypes, Region &body);

/// Custom assembly parser for `hlcf.for` bounds (`lb to ub step s`).
/// An optional `: type` annotation after `s` sets the type for all three
/// bounds; when absent, `index` is assumed (backward-compatible with
/// existing IR that always used `index` bounds).
ParseResult parseForBoundsWithOptionalType(
    OpAsmParser &parser, OpAsmParser::UnresolvedOperand &lowerBound,
    Type &lowerBoundType, OpAsmParser::UnresolvedOperand &upperBound,
    Type &upperBoundType, OpAsmParser::UnresolvedOperand &step, Type &stepType);
void printForBoundsWithOptionalType(OpAsmPrinter &printer, Operation *op,
                                    Value lowerBound, Type lowerBoundType,
                                    Value upperBound, Type upperBoundType,
                                    Value step, Type stepType);

} // namespace M::HLCF

#endif // KGEN_HLCFDIALECT_HLCFUTILS_H
