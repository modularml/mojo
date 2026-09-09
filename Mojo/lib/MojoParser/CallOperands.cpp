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
// This file contains code pertaining to manipulation and diagnostics for
// operand/parameter list processing.
//
//===----------------------------------------------------------------------===//

#include "Mojo/MojoParser/CallOperands.h"
#include "Mojo/LITDialect/LITAttrs.h"
#include "Mojo/LITDialect/LITUtils.h"
#include "Mojo/MojoParser/ExprNode.h"
#include "Mojo/MojoParser/MojoDiags.h"
#include "MojoUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Twine.h"
using namespace M;
using namespace KGEN;
using namespace LIT;

//===----------------------------------------------------------------------===//
// CallSyntax
//===----------------------------------------------------------------------===//

StringRef LIT::stringifyCallSyntax(CallSyntax val) {
  switch (val) {
  case CallSyntax::kParamBindings:
    return "param_bindings";
  case CallSyntax::kDirectCall:
    return "direct_call";
  case CallSyntax::kIndirectCall:
    return "indirect_call";
  case CallSyntax::kMethodCall:
    return "method_call";
  case CallSyntax::kTypeCall:
    return "type_call";
  case CallSyntax::kOperator:
    return "operator";
  case CallSyntax::kReversedOperator:
    return "reversed_operator";
  case CallSyntax::kSubscript:
    return "subscript";
  case CallSyntax::kAttribute:
    return "attribute";
  case CallSyntax::kImplicitConvert:
    return "implicit_convert";
  case CallSyntax::kImplicitCopyCtor:
    return "implicit_copy";
  case CallSyntax::kImplicitMoveCtor:
    return "implicit_move";
  case CallSyntax::kDestructor:
    return "destructor";
  case CallSyntax::kTupleGetItem:
    return "tuple_get_item";
  case CallSyntax::kMethodCallSynthetic:
    return "method_call_synthetic";
  }
  llvm_unreachable("unknown CallSyntax");
  return "";
}

raw_ostream &LIT::operator<<(raw_ostream &os, CallSyntax val) {
  return os << stringifyCallSyntax(val);
}

//===----------------------------------------------------------------------===//
// CallOperands
//===----------------------------------------------------------------------===//

llvm::SMLoc CallOperands::getExprLoc() const { return callExpr->getLoc(); }

void CallOperands::dump() const { llvm::errs() << *this << '\n'; }

raw_ostream &M::KGEN::LIT::operator<<(raw_ostream &os,
                                      const CallOperands &operands) {
  os << "CallOperands{ " << operands.getNumPositional() << " pos args, "
     << operands.getNumKwOperands() << " kw args";
  if (operands.hasSelfOperand)
    os << " <HAS SELF OPERAND>";
  os << '\n';

  for (auto &operand : operands.values) {
    os << "  ";
    if (operand.keyword)
      os << operand.keyword.getValue() << ": ";
    os << operand.ir << "\n";
  }
  return os << '}';
}

/// Validate the operand list against the signature indicated by
/// pogListAttr, emitting an error with "getDiag" if invalid.
///
/// This populates "pogAssignment" with information about the mapping of
/// operands to POG entries.
LogicalResult CallOperands::assignToPogs(
    PogListAttr pogListAttr, bool isParameterList, PogAssignment &pogAssignment,
    llvm::function_ref<MojoInflightDiag &(llvm::SMLoc)> getDiag) const {

  // The specified operand is being passed to a non-variadic arg/param.  If it
  // is unpacked with `*arg` emit an error.
  auto checkSingleValue = [&](size_t opIdx) -> LogicalResult {
    if (values[opIdx].unpackStyle == ArgUnpackStyle::kPositional ||
        values[opIdx].unpackStyle == ArgUnpackStyle::kKeyword)
      return success();
    getDiag(values[opIdx].expr->getLoc())
        << "unpack is only supported when the callee accepts a "
           "variadic; to forward a runtime pack to a fixed-arity callee, "
           "route the call through a dispatcher whose argument is itself a "
           "variadic pack (e.g. "
           "`def shim[Ts: TypeList[Trait=AnyType, ...], //, callee: "
           "def(*args: *Ts) thin](...): callee(*pack)`)"
        << values[opIdx].expr->getRange();
    return failure();
  };

  // Scan each pog and figure out which operands contribute to it. We know that
  // the POG list is validated, so positional arguments will precede keyword
  // arguments, but keyword arguments can be specified in call operands in any
  // order:   foo(kw=42, 7).
  SmallVector<size_t> skippedKW;

  size_t opIdx = 0, numOperands = values.size();
  for (auto [idx, pog] : llvm::enumerate(pogListAttr.getPogs())) {
    PassingKind passingKind = pog.getPassingKind();

    // Ignore implicit operands like out parameters: byref_result and error.
    if (passingKind == PassingKind::Implicit) {
      pogAssignment.operandIdxs.push_back(PogAssignment::kPA_Unspecified);
      continue;
    }

    // Skip over any provided keyword operands when matching things up, we
    // handle them separately below.
    while (opIdx < numOperands &&
           (values[opIdx].unpackStyle == ArgUnpackStyle::kKeyword ||
            values[opIdx].unpackStyle == ArgUnpackStyle::kStarStar)) {
      skippedKW.push_back(opIdx);
      ++opIdx;
    }

    // For positional variadics and packs, consume any positional operands.
    if (pog.isPosVarArg() || pog.isPack()) {
      // Note that the contents will be captured in the posVariadicIdxs list.
      // Note this captures individual values as well as unpacks. A '**'
      // unpack carries no keyword name but binds only to '**kwargs'.
      pogAssignment.operandIdxs.push_back(PogAssignment::kPA_Variadic);
      while (opIdx != numOperands) {
        if (!values[opIdx].keyword &&
            values[opIdx].unpackStyle != ArgUnpackStyle::kStarStar)
          pogAssignment.posVariadicIdxs.push_back(opIdx);
        else
          skippedKW.push_back(opIdx);
        ++opIdx;
      }
      continue;
    }

    // KW variadics eat up any remaining keyword operands, and accept **kwargs.
    if (pog.isKwVarArg()) {
      pogAssignment.operandIdxs.push_back(PogAssignment::kPA_Variadic);

      // Start with any skipped kw operands.
      pogAssignment.kwVariadicIdxs = std::move(skippedKW);

      // Then eat up any remaining keyword operands.
      for (; opIdx < numOperands; ++opIdx) {
        if (values[opIdx].unpackStyle == ArgUnpackStyle::kKeyword ||
            values[opIdx].unpackStyle == ArgUnpackStyle::kStarStar)
          pogAssignment.kwVariadicIdxs.push_back(opIdx);
        else // Unassigned positional operands are errors.
          break;
      }

      // Can only forward a single '**', if there are additional operands, emit
      // an error.
      if (pogAssignment.kwVariadicIdxs.size() != 1) {
        for (auto operandIdx : pogAssignment.kwVariadicIdxs) {
          if (values[operandIdx].unpackStyle != ArgUnpackStyle::kStarStar)
            continue;
          auto &diag = getDiag(values[operandIdx].expr->getLoc());
          diag << "combining a '**' unpack with other keyword arguments for "
                  "'**kwargs' is not supported"
               << values[operandIdx].expr->getRange();
          for (auto otherIdx : pogAssignment.kwVariadicIdxs) {
            if (otherIdx != operandIdx) {
              diag.attachNote(values[otherIdx].expr->getLoc())
                  << "other keyword argument specified here";
              break;
            }
          }
          return failure();
        }
      }
      continue;
    }

    // If we have a non-kw value and non-kw POG, bind the operand.
    if (opIdx < numOperands && (passingKind == PassingKind::PosOrKw ||
                                passingKind == PassingKind::PosOnly)) {
      if (failed(checkSingleValue(opIdx)))
        return failure();

      pogAssignment.operandIdxs.push_back(opIdx);
      ++opIdx;
      continue;
    }

    // If this POG allows a keyword operand and we have one, bind it.
    if (passingKind != PassingKind::PosOnly &&
        passingKind != PassingKind::Implicit) {
      if (const OperandValue *operand = findKwArg(pog.getName())) {
        size_t operandIdx = operand - values.begin();
        pogAssignment.operandIdxs.push_back(operandIdx);
        if (failed(checkSingleValue(operandIdx)))
          return failure();

        auto it = std::find(skippedKW.begin(), skippedKW.end(), operandIdx);
        if (it != skippedKW.end())
          skippedKW.erase(it);
        continue;
      }
    }

    // Parameters can be inferred for many calls, return them inferred.
    if (isParameterList) {
      pogAssignment.operandIdxs.push_back(PogAssignment::kPA_Unspecified);
      continue;
    }

    // If there is a default value use it.
    if (auto defaultVal = pog.getDefaultValue()) {
      pogAssignment.operandIdxs.push_back(PogAssignment::kPA_Default);
      continue;
    }

    // Otherwise, this is missing. If there were an kwargs unpack, emit a
    // specific error.
    for (auto &value : values) {
      if (value.unpackStyle == ArgUnpackStyle::kStarStar) {
        getDiag(value.expr->getLoc())
            << "kwargs unpack can only satisfy another kwargs argument, it "
               "can't fulfill "
            << pog.getName() << value.expr->getRange();
        return failure();
      }
    }

    const char *kindStr = isParameterList ? "parameter" : "argument";
    getDiag(getExprLoc()) << "missing required " << kindStr << ": "
                          << pog.getName();
    return failure();
  }

  // Now that we assigned operands to all POG entries, make sure we don't have
  // any excess operands.

  // If there are no positional variadics, we can check for too many operands.
  if (opIdx != numOperands || !skippedKW.empty()) {
    if (opIdx == numOperands)
      opIdx = skippedKW.front();

    auto &diag = getDiag(values[opIdx].expr->getLoc()) << "unexpected ";
    if (values[opIdx].keyword)
      diag << "keyword ";
    diag << (isParameterList ? "parameter" : "argument");
    if (values[opIdx].keyword)
      diag << " " << values[opIdx].keyword;
    diag << values[opIdx].expr->getRange();
    return failure();
  }
  return success();
}
