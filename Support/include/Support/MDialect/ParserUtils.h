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

#ifndef SUPPORT_COMPILER_COMPILERUTILS_H
#define SUPPORT_COMPILER_COMPILERUTILS_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/LogicalResult.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/Support/SMLoc.h"
#include <optional>

namespace M {

/// Parses a region, using argumentInfo as the block arguments. If
/// isIsolatedFromAbove is true then it is legal for argumentInfo SSA values
/// to shadow those already bound in an outer scope.
ParseResult parseRegion(OpAsmParser &parser,
                        const SmallVector<OpAsmParser::Argument> &argumentInfo,
                        Region &region, bool isIsolatedFromAbove);

/// Prints a region. If isIsolatedFromAbove is true then it is legal for the
/// region's argument to shadow the given operands.
void printRegion(OpAsmPrinter &printer, const OperandRange &operands,
                 Region &region, bool isIsolatedFromAbove);

/// Parses a parenthesized operand list with optional or required 'as' clauses.
/// Each operand contributes both to the operation state's operand list and to
/// the argumentInfo array to be used by the op's nested blocks.
///
/// If isIsolatedFromAbove is true then the 'as' clauses are aptional.
/// Otherwise the op must spell out the mapping from 'outer' bound SSA values
/// to 'inner' argument SSA values.
///
/// Eg: Basic form:
///   (%a: i32, %b: f32)
/// Eg: With corresponding block argument name distinct from the operand name
///   (%a: i32, %a as %b: f32)
ParseResult parseParenOperandListWithShadowing(
    OpAsmParser &parser, OperationState &state,
    SmallVectorImpl<OpAsmParser::Argument> &argumentInfo,
    bool isIsolatedFromAbove);

/// Prints a parenthesized operand list, matching the syntax parsed by
/// parseParenOperandListWithShadowing.
///
/// Intended to pair with printRegionWithShadowing for nested regions.
void printParenOperandListWithShadowing(
    OpAsmPrinter &printer, const OperandRange &operands,
    const Block::BlockArgListType &arguments, bool isIsolatedFromAbove);

/// Parse a list of operands that have optional types. If an operand in the list
/// does not have a specified type, the default type is assigned to the operand.
///
/// Basic form is (%a, %b: f32). %a will be assigned the default type, and %b
/// will have type f32.
ParseResult parseParenOperandListWithDefaultType(
    OpAsmParser &parser,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
    SmallVectorImpl<Type> &operandTypes, Type defaultType);

/// Parses a parenthesized operand list with optional type annotations.
/// The given optional type is used if no type annotation is given.
/// Additionally, populates operandList with the types of the operands.
///
/// Eg: Basic form:
///   (%a: i32, %b: f32)
/// Eg: With default type taken from defaultType:
///   (%a, %b: f32)
ParseResult parseParenOperandListWithDefaultType(
    OpAsmParser &parser, SmallVectorImpl<Value> &operands,
    SmallVectorImpl<Type> &operandTypes, Type defaultType);

/// Parses a parenthesized operand list with optional type annotations.
/// The operands are gotten from the OperationState argument.
/// The given optional type is used if no type annotation is given.
///
/// Eg: Basic form:
///   (%a: i32, %b: f32)
/// Eg: With default type taken from defaultType:
///   (%a, %b: f32)
ParseResult parseParenOperandListWithDefaultType(OpAsmParser &parser,
                                                 OperationState &state,
                                                 Type defaultType);

/// Prints a parenthesized operand list, matching the syntax parsed by
/// parseParenOperandListWithDefaultType.
void printParenOperandListWithDefaultType(OpAsmPrinter &printer,
                                          const OperandRange &operands,
                                          Type defaultType);

/// Parses an 'in/out signature' of the form:
///    ( in|out|mut %x : type, ... )
/// The SSA values will be added to args, their types to argTypes,
/// and the inOutSignatureAttr will have matching arity and capture the
/// in/out/mut keywords.
///
/// Note that the types are unconstrained and need not be any particular
/// type. However generally they are pointer-like for the in/out/mut keyword
/// to have any meaning.
ParseResult parseInOutArgsSignature(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &args,
    SmallVectorImpl<Type> &argTypes, InOutSignatureAttr &inOutSignatureAttr);

/// Prints an 'in/out signature', matching the syntax parsed by
/// parseInOutArgsSignature.
void printInOutArgsSignature(OpAsmPrinter &printer, const Operation *opIgnored,
                             ValueRange buffers, TypeRange bufferTypes,
                             InOutSignatureAttr inOutSignatureAttr);

/// This is an AsmPrinter implementation that just outputs to an external output
/// stream.
class StreamAsmPrinter : public AsmPrinter {
public:
  explicit StreamAsmPrinter(raw_ostream &os) : os(os) {}

  /// Implement all the virtual hooks.

  raw_ostream &getStream() const override { return os; }

  /// Trivial hooks

  void printType(Type type) override { os << type; }
  void printAttribute(Attribute attr) override { os << attr; }
  void printAttributeWithoutType(Attribute attr) override {
    attr.print(os, /*elideType=*/true);
  }
  LogicalResult printAlias(Attribute attr) override { return failure(); }
  LogicalResult printAlias(Type type) override { return failure(); }

  /// Less trivial hooks.

  void printString(StringRef string) override;
  void printKeywordOrString(StringRef keyword) override;
  void printSymbolName(StringRef symbolRef) override;
  void
  printResourceHandle(const mlir::AsmDialectResourceHandle &resource) override;

  /// Print floats like MLIR does.
  void printFloat(const APFloat &value) override;

private:
  /// The stream to output to.
  raw_ostream &os;
};

/// Optionally parse an enum attribute.
template <typename EnumAttrT,
          typename EnumT = decltype(std::declval<EnumAttrT>().getValue)>
ParseResult parseOptionalEnum(AsmParser &p, EnumAttrT &result,
                              std::optional<EnumT> (*symbolize)(StringRef)) {
  EnumT value = EnumT();
  llvm::SMLoc loc = p.getCurrentLocation();
  StringRef kw;
  if (succeeded(p.parseOptionalKeyword(&kw))) {
    std::optional<EnumT> symbol = symbolize(kw);
    if (!symbol)
      return p.emitError(loc, "failed to symbolize enum");
    value = *symbol;
  }
  result = EnumAttrT::get(p.getContext(), value);
  return success();
}

/// Optionally print an enum attribute.
template <typename EnumAttrT,
          typename EnumT = decltype(std::declval<EnumAttrT>().getValue)>
void printOptionalEnum(AsmPrinter &p, Operation *, EnumAttrT attr,
                       std::optional<EnumT> (*)(StringRef)) {
  if (attr.getValue() == EnumT())
    return;
  p << ' ' << stringifyEnum(attr.getValue());
}

} // namespace M

#endif // SUPPORT_COMPILER_COMPILERUTILS_H
