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

#include "Mojo/CODialect/COOps.h"
#include "Mojo/CODialect/CODialect.h"
#include "Mojo/CODialect/COUtils.h"
#include "Mojo/HLCFDialect/HLCFUtils.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "Mojo/TransformUtils/InliningUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

using namespace M;
using namespace KGEN;
using namespace CO;

//===----------------------------------------------------------------------===//
// HandleOp
//===----------------------------------------------------------------------===//

LogicalResult HandleOp::verify() {
  if (auto func = (*this)->getParentOfType<FuncOp>()) {
    if (func.getNumResults() != 1) {
      return emitOpError("surrounding function must have 1 result")
                 .attachNote(func.getLoc())
             << "see function here";
    }
    Type resultType = func.getResultTypes().front();
    if (resultType != getType()) {
      return emitOpError("surrounding function result type does not match "
                         "coroutine handle type")
                 .attachNote(func.getLoc())
             << "surrounding function returns " << resultType;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SuspendOp
//===----------------------------------------------------------------------===//

static ParseResult parseSuspendBody(OpAsmParser &p, Region &body) {
  SmallVector<OpAsmParser::Argument, 1> args;
  if (succeeded(p.parseOptionalLParen())) {
    if (p.parseArgument(args.emplace_back()) || p.parseRParen())
      return failure();
    args.back().type = CoroutineType::get(p.getContext());
  }
  return p.parseRegion(body, args);
}

static void printSuspendBody(OpAsmPrinter &p, Operation *op, Region &body) {
  if (body.getNumArguments()) {
    p << '(';
    p.printRegionArgument(body.getArgument(0), /*argAttrs=*/{},
                          /*omitType=*/true);
    p << ") ";
  }
  p.printRegion(body, /*printEntryBlockArgs=*/false);
}

void SuspendOp::getAsmBlockArgumentNames(
    Region &region, llvm::function_ref<void(Value, StringRef)> setNameFn) {
  // Sugar the SSA value name
  if (region.getNumArguments())
    setNameFn(region.getArgument(0), "hdl");
}

LogicalResult SuspendOp::verify() {
  Region &body = getBody();
  if (body.getNumArguments() == 0)
    return success();
  if (body.getNumArguments() == 1 &&
      isa<CoroutineType>(body.getArgument(0).getType()))
    return success();
  return emitOpError("expected its body region to have a "
                     "single `!co.routine` type argument or no arguments");
}

//===----------------------------------------------------------------------===//
// InvokeOp
//===----------------------------------------------------------------------===//

static ParseResult parseAsyncParametricCallee(
    OpAsmParser &p, TypedAttr &callee,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &operandTypes) {
  if (failed(parseParametricCallee(p, callee)))
    return failure();
  // Operands match signature arguments with the exception that byref error and
  // byref result are omitted.
  FuncType signature = cast<FuncTypeGeneratorType>(callee.getType()).getBody();
  ArrayRef<Type> argumentTypes =
      signature.getArguments().drop_back(signature.getNumAsyncReturnSlots());
  llvm::append_range(operandTypes, argumentTypes);

  // Parse the untyped operands.
  if (failed(p.parseCommaSeparatedList(
          AsmParser::Delimiter::Paren, [&]() -> ParseResult {
            return p.parseOperand(operands.emplace_back());
          })))
    return failure();
  return success();
}

static void printAsyncParametricCallee(OpAsmPrinter &p, Operation *op,
                                       TypedAttr callee, ValueRange operands,
                                       TypeRange operandTypes) {
  printParametricCallee(p, op, callee);
  p << "(";
  p.printOperands(operands);
  p << ")";
}

LogicalResult InvokeOp::verify() {
  FuncType signature =
      cast<FuncTypeGeneratorType>(getCallee().getType()).getBody();
  if (!signature.isAsync())
    return emitOpError("callable must be 'async'");
  return verifyCallOperands(*this, getOperands(), signature,
                            /*ignoreByRef=*/true);
}

FailureOr<InlineResult> InvokeOp::prepInline(mlir::RewriterBase &b) {
  auto op =
      ExecuteOp::create(b, getLoc(), getCalleeType().getBody().getResults());
  return {{op, [](Operation *) {}}};
}

//===----------------------------------------------------------------------===//
// HotInvokeOp
//===----------------------------------------------------------------------===//

static ParseResult parseHotAsyncParametricCallee(
    OpAsmParser &p, TypedAttr &callee,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &operandTypes, SmallVectorImpl<Type> &resultTypes) {
  if (failed(parseParametricCallee(p, callee)))
    return failure();

  // Parse async function operands.
  FuncType signature = cast<FuncTypeGeneratorType>(callee.getType()).getBody();
  llvm::append_range(operandTypes, signature.getArguments());
  if (failed(p.parseCommaSeparatedList(
          mlir::AsmParser::Delimiter::Paren, [&]() -> ParseResult {
            return p.parseOperand(operands.emplace_back());
          })))
    return failure();
  llvm::append_range(resultTypes, signature.getResults());
  return success();
}

static void printHotAsyncParametricCallee(OpAsmPrinter &p, Operation *op,
                                          TypedAttr callee, ValueRange operands,
                                          TypeRange operandTypes,
                                          TypeRange resultTypes) {
  printParametricCallee(p, op, callee);
  p << "(";
  p.printOperands(operands);
  p << ")";
}

LogicalResult HotInvokeOp::verify() {
  FuncType signature =
      cast<FuncTypeGeneratorType>(getCallee().getType()).getBody();
  if (!signature.isAsync())
    return emitOpError("callable must be 'async'");
  return verifyCallOperands(*this, getOperands(), signature);
}

// inlining a hot invoke is equivalent to awaiting an execute op
FailureOr<InlineResult> HotInvokeOp::prepInline(mlir::RewriterBase &b) {
  StringAttr label = b.getStringAttr("inlined_cf_scope");
  auto op =
      HLCF::LoopOp::create(b, getLoc(), getResultTypes(), ValueRange(), label);
  return {{op, [label, &b](Operation *op) {
             b.replaceOpWithNewOp<HLCF::BreakOp>(op, op->getOperands(), label);
           }}};
}

//===----------------------------------------------------------------------===//
// ExecuteOp
//===----------------------------------------------------------------------===//

static ParseResult parseExecuteBody(OpAsmParser &p, Region &body) {
  SmallVector<OpAsmParser::Argument, 2> args;
  if (succeeded(p.parseOptionalLParen())) {
    if (p.parseArgument(args.emplace_back(), /*allowType=*/true))
      return failure();
    if (succeeded(p.parseOptionalKeyword("byref_error"))) {
      if (p.parseComma() ||
          p.parseArgument(args.emplace_back(), /*allowType=*/true))
        return failure();
    }
    if (p.parseKeyword("byref_result") || p.parseRParen())
      return failure();
  }
  return p.parseRegion(body, args);
}

static void printExecuteBody(OpAsmPrinter &p, Operation *op, Region &body) {
  if (body.getNumArguments()) {
    p << '(';
    p.printRegionArgument(body.getArgument(0));
    // Include the "argument conventions" in the assembly syntax for
    // familiarity. They're not strictly necessary.
    if (body.getNumArguments() == 2) {
      p << " byref_error, ";
      p.printRegionArgument(body.getArgument(1));
    }
    p << " byref_result) ";
  }
  p.printRegion(body, /*printEntryBlockArgs*/ false);
}

LogicalResult ExecuteOp::verify() {
  if (getBody()->getNumArguments() <= 2)
    return success();
  return emitOpError("body expected at most 2 arguments");
}

ArrayRef<Type> ExecuteOp::getResultTypes() { return getTypes(); }

//===----------------------------------------------------------------------===//
// AwaitOp
//===----------------------------------------------------------------------===//

LogicalResult AwaitOp::canonicalize(AwaitOp op, PatternRewriter &b) {
  // `co.await(co.execute) -> inlined region`.
  if (auto execute = op.getCoroutine().getDefiningOp<ExecuteOp>()) {
    SmallVector<Value, 2> args(llvm::reverse(op.getSlots()));
    // If the block argument types don't match the provided result slots, then
    // this operation is UB.
    if (execute.getBody()->getArgumentTypes() != ValueRange(args).getType())
      return b.notifyMatchFailure(op.getLoc(), "result slot types don't match");
    // This should be the only user of the coroutine.
    if (!op.getCoroutine().hasOneUse())
      return b.notifyMatchFailure(op.getLoc(), "coroutine has other uses");
    IRMapping map;
    auto [scope, singleExit] = KGEN::inlineRegion(
        b, map, op, args, execute.getBodyRegion(), /*takeBody=*/true);
    if (singleExit)
      foldTrivialLoop(b, scope);
    b.eraseOp(execute);
    return success();
  } else if (auto invoke = op.getCoroutine().getDefiningOp<InvokeOp>()) {
    if (!invoke.getCoroutine().hasOneUse())
      return b.notifyMatchFailure(op.getLoc(), "coroutine has other uses");
    SmallVector<Value> args;
    llvm::append_range(args, invoke.getOperands());
    llvm::append_range(args, llvm::reverse(op.getSlots()));
    b.replaceOpWithNewOp<HotInvokeOp>(op, op->getResultTypes(),
                                      invoke.getCallee(), args);
    b.eraseOp(invoke);
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// CODialect
//===----------------------------------------------------------------------===//

void CODialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "Mojo/CODialect/CO.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Mojo/CODialect/CO.cpp.inc"
