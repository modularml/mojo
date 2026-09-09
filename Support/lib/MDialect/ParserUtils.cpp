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

#include "Support/MDialect/ParserUtils.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/MDialect/MAttrs.h"

#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SMLoc.h"
#include <cassert>
#include <cctype>
#include <cstddef>

using namespace M;

static bool allDistinct(ValueRange values) {
  DenseSet<Value> unique;
  for (const auto &value : llvm::enumerate(values)) {
    if (unique.size() != value.index()) {
      return false; // Early termination.
    }
    unique.insert(value.value());
  }
  return unique.size() == values.size();
}

ParseResult
M::parseRegion(OpAsmParser &parser,
               const SmallVector<OpAsmParser::Argument> &argumentInfo,
               Region &region, bool isIsolatedFromAbove) {
  return parser.parseRegion(region, argumentInfo,
                            /*enableNameShadowing=*/isIsolatedFromAbove);
}

void M::printRegion(OpAsmPrinter &printer, const OperandRange &operands,
                    Region &region, bool isIsolatedFromAbove) {
  assert(operands.size() == region.getNumArguments());
  if (isIsolatedFromAbove && allDistinct(operands)) {
    printer.shadowRegionArgs(region, operands);
    printer.printRegion(region, /*printEntryBlockArgs=*/false);
  } else {
    printer.printRegion(region, /*printEntryBlockArgs=*/false);
  }
}

ParseResult M::parseParenOperandListWithShadowing(
    OpAsmParser &parser, OperationState &state,
    SmallVectorImpl<OpAsmParser::Argument> &argumentInfo,
    bool isIsolatedFromAbove) {
  llvm::SMLoc loc = parser.getCurrentLocation();

  auto parseOperandFn = [&]() -> ParseResult {
    // The operand to capture as part of the ops inputs.
    OpAsmParser::UnresolvedOperand unresolvedOperand;
    // The corresponding argument to use for the op's nested blocks.
    OpAsmParser::Argument arg;

    // Parse the input operand.
    if (parser.parseOperand(unresolvedOperand,
                            /*allowResultNumber=*/true))
      return failure();

    // Parse an 'as' clause to name the nested block arguments. This can be
    // used to given nested blocks a unique argument name even if the same
    // SSA value is repeated as on operand.
    if (!isIsolatedFromAbove) {
      if (parser.parseKeyword("as") ||
          parser.parseOperand(arg.ssaName, /*allowResultNumber=*/false))
        return failure();
    } else if (succeeded(parser.parseOptionalKeyword("as"))) {
      if (parser.parseOperand(arg.ssaName, /*allowResultNumber=*/false))
        return failure();
    } else {
      // The nested blocks will 'pun' the operand name, presumably without
      // ambiguity.
      arg.ssaName = unresolvedOperand;
    }

    // Parse type annotation
    if (parser.parseColon() || parser.parseType(arg.type))
      return failure();

    // No block argument attributes.
    NamedAttrList attrs;
    arg.attrs = attrs.getDictionary(parser.getContext());

    argumentInfo.push_back(arg);

    // Resolve the input operand into the operation state.
    if (parser.resolveOperand(unresolvedOperand, arg.type, state.operands))
      return failure();

    return success();
  };

  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                     parseOperandFn, "in operand list"))
    return failure();

  llvm::StringSet<> argumentNames;
  for (auto arg : argumentInfo) {
    argumentNames.insert(
        (Twine(arg.ssaName.name) + "#" + Twine(arg.ssaName.number)).str());
  }
  if (argumentNames.size() != argumentInfo.size())
    return parser.emitError(
        loc, "has duplicate SSA values in its operand list which have not been "
             "renamed apart by 'as' clauses.");

  return success();
}

void M::printParenOperandListWithShadowing(
    OpAsmPrinter &printer, const OperandRange &operands,
    const Block::BlockArgListType &arguments, bool isIsolatedFromAbove) {
  assert(operands.size() == arguments.size());
  bool needAs = !isIsolatedFromAbove || !allDistinct(operands);
  printer << "(";
  bool first = true;
  for (auto [operand, arg] : llvm::zip(operands, arguments)) {
    if (first)
      first = false;
    else
      printer << ", ";
    printer << operand;
    if (needAs)
      printer << " as " << arg;
    printer << ": " << operand.getType();
  }
  printer << ")";
}

ParseResult M::parseParenOperandListWithDefaultType(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &unresolvedOperands,
    SmallVectorImpl<Type> &operandTypes, Type defaultType) {
  auto parseOperandFn = [&]() -> ParseResult {
    // Parse the unresolved operand.
    OpAsmParser::UnresolvedOperand &unresolvedOperand =
        unresolvedOperands.emplace_back();

    if (parser.parseOperand(unresolvedOperand))
      return failure();

    // Parse optional type annotation
    Type type = defaultType;
    if (succeeded(parser.parseOptionalColon())) {
      if (parser.parseType(type))
        return failure();
    }
    operandTypes.push_back(type);

    return success();
  };

  return parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                        parseOperandFn, "in operand list");
}

ParseResult M::parseParenOperandListWithDefaultType(
    OpAsmParser &parser, SmallVectorImpl<Value> &operands,
    SmallVectorImpl<Type> &operandTypes, Type defaultType) {
  auto parseOperandFn = [&]() -> ParseResult {
    // The operand to capture as part of the ops inputs.
    OpAsmParser::UnresolvedOperand unresolvedOperand;

    // Parse the input operand.
    if (parser.parseOperand(unresolvedOperand))
      return failure();

    // Parse optional type annotation
    Type type = defaultType;
    if (succeeded(parser.parseOptionalColon())) {
      if (parser.parseType(type))
        return failure();
    }

    operandTypes.push_back(type);

    // Resolve the input operand into the operation state.
    if (parser.resolveOperand(unresolvedOperand, type, operands))
      return failure();

    return success();
  };

  return parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                        parseOperandFn, "in operand list");
}

ParseResult M::parseParenOperandListWithDefaultType(OpAsmParser &parser,
                                                    OperationState &state,
                                                    Type defaultType) {
  SmallVector<Type> operandTypes;
  return parseParenOperandListWithDefaultType(parser, state.operands,
                                              operandTypes, defaultType);
}

void M::printParenOperandListWithDefaultType(OpAsmPrinter &printer,
                                             const OperandRange &operands,
                                             Type defaultType) {
  auto printArg = [&](Value arg) {
    printer << arg;
    if (arg.getType() != defaultType)
      printer << ": " << arg.getType();
  };
  printer << '(';
  llvm::interleaveComma(operands, printer, printArg);
  printer << ')';
}

ParseResult M::parseInOutArgsSignature(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &args,
    SmallVectorImpl<Type> &argTypes, InOutSignatureAttr &inOutSignatureAttr) {
  SmallVector<InOutSignatureAttr::InOutSemantics> semantics;
  auto parseOperandFn = [&]() -> ParseResult {
    llvm::SMLoc loc;

    // Parse the in/out/mut keyword.
    StringRef inOutMut;
    if (parser.getCurrentLocation(&loc) || parser.parseKeyword(&inOutMut))
      return failure();
    if (inOutMut == "in")
      semantics.emplace_back(InOutSignatureAttr::kIn);
    else if (inOutMut == "out")
      semantics.emplace_back(InOutSignatureAttr::kOut);
    else if (inOutMut == "mut")
      semantics.emplace_back(InOutSignatureAttr::kMut);
    else
      return parser.emitError(loc) << "expecting 'in', 'out' or 'mut' keyword";

    // Parse the operand proper.
    OpAsmParser::UnresolvedOperand &unresolvedOperand = args.emplace_back();
    if (parser.parseOperand(unresolvedOperand))
      return failure();

    // Parse the type annotation.
    Type &bufferType = argTypes.emplace_back();
    if (parser.parseColon() || parser.parseType(bufferType))
      return failure();

    return success();
  };

  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren,
                                     parseOperandFn,
                                     "in in/out signature list"))
    return failure();

  inOutSignatureAttr = InOutSignatureAttr::get(parser.getContext(), semantics);

  return success();
}

void M::printInOutArgsSignature(OpAsmPrinter &printer,
                                const Operation *opIgnored, ValueRange args,
                                TypeRange argTypes,
                                InOutSignatureAttr inOutSignatureAttr) {
  size_t arity = inOutSignatureAttr.size();
  assert(args.size() == arity);
  assert(argTypes.size() == arity);
  printer << "(";
  for (size_t i = 0; i < arity; ++i) {
    if (i > 0)
      printer << ", ";
    switch (inOutSignatureAttr[i]) {
    case InOutSignatureAttr::kNone:
      llvm::llvm_unreachable_internal(
          "unexpected kNone buffer semantics in signature");
      break;
    case InOutSignatureAttr::kIn:
      printer << "in ";
      break;
    case InOutSignatureAttr::kOut:
      printer << "out ";
      break;
    case InOutSignatureAttr::kMut:
      printer << "mut ";
      break;
    }
    printer << args[i];
    printer << " : ";
    printer << argTypes[i];
  }
  printer << ")";
}

/// Returns true if the given string can be represented as a bare identifier
/// compatible with the MLIR lexer.
static bool isBareIdentifier(StringRef name) {
  if (name.empty() || (!isalpha(name[0]) && name[0] != '_'))
    return false;
  return llvm::all_of(name.drop_front(), [](unsigned char c) {
    return isalnum(c) || c == '_' || c == '$' || c == '.';
  });
}

void StreamAsmPrinter::printString(StringRef string) {
  os << "\"";
  llvm::printEscapedString(string, os);
  os << '"';
}

void StreamAsmPrinter::printKeywordOrString(StringRef keyword) {
  if (isBareIdentifier(keyword)) {
    os << keyword;
    return;
  }
  os << "\"";
  llvm::printEscapedString(keyword, os);
  os << '"';
}

void StreamAsmPrinter::printSymbolName(StringRef symbolRef) {
  os << '@';
  printKeywordOrString(symbolRef);
}

void StreamAsmPrinter::printResourceHandle(
    const mlir::AsmDialectResourceHandle &resource) {
  auto *interface = cast<OpAsmDialectInterface>(resource.getDialect());
  os << interface->getResourceKey(resource);
}

void StreamAsmPrinter::printFloat(const APFloat &value) {
  if (!value.isInfinity() && !value.isNaN()) {
    SmallString<128> strValue;
    value.toString(strValue, /*FormatPrecision=*/6, /*FormatMaxPadding=*/0,
                   /*TruncateZero=*/false);
    if (APFloat(value.getSemantics(), strValue).bitwiseIsEqual(value)) {
      os << strValue;
      return;
    }
    strValue.clear();
    value.toString(strValue);
    if (strValue.str().contains('.')) {
      os << strValue;
      return;
    }
  }
  SmallVector<char, 16> str;
  APInt apInt = value.bitcastToAPInt();
  apInt.toString(str, /*Radix=*/16, /*Signed=*/false,
                 /*formatAsCLiteral=*/true);
  os << str;
}
