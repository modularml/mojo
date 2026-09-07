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
// This file provides declarations for various expression nodes when in
// syntactic (not yet type checked) form.
//
// Expressions are parsed with a two-phase approach.  The first phase pulls out
// the syntactic structure of the expression, whereas the second pass does type
// checking and IR generation.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_EXPRNODES_H
#define KGEN_MOJOPARSER_EXPRNODES_H

#include "Mojo/MojoParser/ExprNode.h"
#include "Mojo/MojoParser/MojoDiags.h"

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/LITDialect/LITAttrs.h"
#include "Mojo/LITDialect/LITOps.h"
#include "llvm/ADT/StringExtras.h"

namespace M::KGEN {
class FuncTypeGeneratorType;
} // namespace M::KGEN

namespace M::KGEN::LIT {
struct ParsedArgument;
struct ParsedConstraint;
enum class CaptureConvention : uint8_t;
class SRValue;

/// The IREmitter depends on ExprNode to provide a location and emit IR for
/// its value. In the case of synthetic code, there is a source sequence that
/// triggered the generation but not necessarily a value associated with the
/// synthetic code. To solve this, we use the synthetic node which will vend a
/// location to the emitter but has no value.
struct SyntheticNode final : public ExprNode {
  // If SyntheticNode is created with a value, then emitIR will produce that
  // value.  If not, emitIR will abort.
  SyntheticNode(SMLoc loc, AnyValue irValue = AnyValue())
      : ExprNode(kSynthetic), location(loc), irValue(irValue) {}

  const SMLoc location;

  // If null, emitIR will explode, otherwise it will produce this.
  AnyValue irValue;

  static bool classof(const ExprNode *node) { return node->kind == kSynthetic; }
  SMLoc getLoc() const override { return location; }
  SourceRange getRange() const override { return {getLoc(), getLoc()}; }
  AnyValue emitIR(ExprDest &dest, IREmitter &emitter) const override;
  void print(mlir::raw_indented_ostream &os) const override;
};

/// This returns an SMLoc from a StringRef that points into the source buffer.
inline SMLoc getSMLocFromStringRef(StringRef bufferRef,
                                   uint8_t startOffset = 0) {
  return SMLoc::getFromPointer(bufferRef.data() - startOffset);
}

struct IntLiteralNode final : public ExprNode {
  IntLiteralNode(StringRef spelling)
      : ExprNode(kIntLiteral), spelling(spelling) {}

  const StringRef spelling;

  static bool classof(const ExprNode *node) {
    return node->kind == kIntLiteral;
  }
  SMLoc getLoc() const override { return getSMLocFromStringRef(spelling); }
  SourceRange getRange() const override { return {getLoc(), getLoc()}; }

  AnyValue emitIR(ExprDest &dest, IREmitter &emitter) const override;
  CValue emitMatch(IREmitter &emitter, CValue subject,
                   PatternDeclKind patternKind) const override;
  void print(mlir::raw_indented_ostream &os) const override;
};

struct FloatLiteralNode final : public ExprNode {
  FloatLiteralNode(StringRef spelling)
      : ExprNode(kFloatLiteral), spelling(spelling) {}

  const StringRef spelling;

  static bool classof(const ExprNode *node) {
    return node->kind == kFloatLiteral;
  }
  SMLoc getLoc() const override { return getSMLocFromStringRef(spelling); }
  SourceRange getRange() const override { return {getLoc(), getLoc()}; }
  AnyValue emitIR(ExprDest &dest, IREmitter &emitter) const override;
  CValue emitMatch(IREmitter &emitter, CValue subject,
                   PatternDeclKind patternKind) const override;
  void print(mlir::raw_indented_ostream &os) const override;
};

struct BoolLiteralNode final : public ExprNode {
  BoolLiteralNode(SMLoc loc, bool value)
      : ExprNode(kBoolLiteral), loc(loc), value(value) {}

  const SMLoc loc;
  const bool value;

  static bool classof(const ExprNode *node) {
    return node->kind == kBoolLiteral;
  }

  SMLoc getLoc() const override { return loc; }
  SourceRange getRange() const override { return {getLoc(), getLoc()}; }
  AnyValue emitIR(ExprDest &dest, IREmitter &emitter) const override;
  CValue emitMatch(IREmitter &emitter, CValue subject,
                   PatternDeclKind patternKind) const override;
  void print(mlir::raw_indented_ostream &os) const override;
};

// This node is used for things like 'Self', '_', 'None' expressions etc.
struct SimpleLiteralNode final : public ExprNode {
  SimpleLiteralNode(Kind kind, SMLoc loc) : ExprNode(kind), loc(loc) {
    assert(classof(kind) && "invalid expr kind for this node");
  }

  const SMLoc loc;

  static bool classof(const ExprNode *node) { return classof(node->kind); }

  static bool classof(Kind kind) {
    return kind == kSelfLiteral || kind == kNoneLiteral ||
           kind == kDiscardLiteral || kind == kEllipsisLiteral;
  }

  SMLoc getLoc() const override { return loc; }
  SourceRange getRange() const override { return {getLoc(), getLoc()}; }
  AnyValue emitIR(ExprDest &dest, IREmitter &emitter) const override;
  LogicalResult emitDestructuringPValue(PValue value,
                                        IREmitter &emitter) const override;
  CValue emitMatch(IREmitter &emitter, CValue subject,
                   PatternDeclKind patternKind) const override;
  void print(mlir::raw_indented_ostream &os) const override;
};

/// String literal nodes like "foo".  String literals support implicit
/// concatenation, so `"foo" "bar"` is treated as one expression node.
struct StringLiteralNode final : public ExprNode {
  StringLiteralNode(ArrayRef<StringRef> spellings)
      : ExprNode(kStringLiteral), spellings(spellings) {}

  const ArrayRef<StringRef> spellings;

  /// Return the contents of the string without the quotes and after
  /// concatenation.
  std::string getValue() const;

  /// Emit a constructor call for a string literal with the specified data, that
  /// does not include enclosing quotes.  The specified expression specifies the
  /// location but need not by a StringLiteral.
  static CValue emitCtorCall(StringRef bytes, const ExprNode *expr,
                             ExprDest &dest, IREmitter &emitter);

  static bool classof(const ExprNode *node) {
    return node->kind == kStringLiteral;
  }
  SMLoc getLoc() const override {
    return getSMLocFromStringRef(spellings.front());
  }
  SourceRange getRange() const override {
    return {getLoc(), getSMLocFromStringRef(spellings.back())};
  }
  AnyValue emitIR(ExprDest &dest, IREmitter &emitter) const override;
  CValue emitMatch(IREmitter &emitter, CValue subject,
                   PatternDeclKind patternKind) const override;
  void print(mlir::raw_indented_ostream &os) const override;
};

/// T-string literal nodes like t"hello {name}".
/// For Phase 1, we support basic {expr} interpolation.
struct TStringExprNode final : public ExprNode {
  struct LiteralPart {
    StringRef text;
    bool isRaw = false;
  };

  struct InterpolationPart {
    ExprNode *expr;
  };

  using Part = std::variant<LiteralPart, InterpolationPart>;

  TStringExprNode(SMLoc startLoc, SMLoc endLoc, ArrayRef<Part> parts)
      : ExprNode(kTStringLiteral), startLoc(startLoc), endLoc(endLoc),
        parts(parts) {}

  const SMLoc startLoc;
  const SMLoc endLoc;
  const ArrayRef<Part> parts;

  static bool classof(const ExprNode *node) {
    return node->kind == kTStringLiteral;
  }

  SMLoc getLoc() const override { return startLoc; }
  SourceRange getRange() const override;
  AnyValue emitIR(ExprDest &dest, IREmitter &emitter) const override;
  void print(mlir::raw_indented_ostream &os) const override;
};

struct Identifier {
  Identifier(StringRef spelling, bool isEscaped)
      : spelling(spelling), isEscaped(isEscaped) {}

  const StringRef spelling;
  /// Needed to emit correct location if an identifier was escaped.
  bool isEscaped;

  /// Return the identifier's location with the offset taken into account.
  SMLoc getIdentifierLoc() const {
    return getSMLocFromStringRef(spelling, /*startOffset=*/isEscaped);
  }

  /// Return the identifier's range with the offset taken into account.
  SourceRange getIdentifierRange() const {
    SMLoc start = getIdentifierLoc();
    if (!isEscaped)
      return {start, start};
    auto end = SMLoc::getFromPointer(start.getPointer() + spelling.size() + 2);
    return SourceRange::getByteLevel(start, end);
  }
};

struct DeclRefNode final : public LValueCapableExprNode, Identifier {
  DeclRefNode(StringRef spelling, bool isEscapedIdentifier = false)
      : LValueCapableExprNode(kDeclRef),
        Identifier(spelling, isEscapedIdentifier) {}

  static bool classof(const ExprNode *node) { return node->kind == kDeclRef; }
  SMLoc getLoc() const override { return getIdentifierLoc(); }
  SourceRange getRange() const override { return getIdentifierRange(); }

  /// This performs a lookup of the specified identifier in the specified lookup
  /// scope, which might be different than the emitter's current scope. This is
  /// refactored out from emitLCVIR because we use it for qualified lookup `x.y`
  /// when `x` is a package.
  static ELVIITResult emitUnqualLookup(StringRef spelling, const ExprNode *expr,
                                       ASTDecl &lookupScope, ExprDest &dest,
                                       IREmitter &emitter, bool isSpeculative);
  LogicalResult emitDestructuringPValue(PValue value,
                                        IREmitter &emitter) const override;
  ELVIITResult emitLCVIR(ExprDest &dest, IREmitter &emitter,
                         bool isSpeculative) const override;
  CValue emitMatch(IREmitter &emitter, CValue subject,
                   PatternDeclKind patternKind) const override;
  void print(mlir::raw_indented_ostream &os) const override;
};

struct AttributeRefNode final : public LValueCapableExprNode, Identifier {
  AttributeRefNode(ExprNode *base, SMLoc dotLoc, StringRef spelling,
                   bool isEscapedIdentifier = false)
      : LValueCapableExprNode(kAttributeRef),
        Identifier(spelling, isEscapedIdentifier), base(base), dotLoc(dotLoc) {}

  ExprNode *const base;
  const SMLoc dotLoc;

  static bool classof(const ExprNode *node) {
    return node->kind == kAttributeRef;
  }
  SMLoc getLoc() const override { return dotLoc; }
  SourceRange getAttributeNameRange() const { return getIdentifierRange(); }
  SourceRange getRange() const override {
    return {base->getRangeStart(), getIdentifierLoc()};
  }
  ELVIITResult emitLCVIR(ExprDest &dest, IREmitter &emitter,
                         bool isSpeculative) const override;
  CValue emitMatch(IREmitter &emitter, CValue subject,
                   PatternDeclKind patternKind) const override;
  void print(mlir::raw_indented_ostream &os) const override;

  /// Emit a reference to a stored field with a base that is known not to be a
  /// dynamic lvalue.
  static CValue emitStoredFieldRef(ASTExprAnd<CValue> base,
                                   StructFieldOp fieldOp, const ExprNode *expr,
                                   ExprDest &dest, IREmitter &emitter);
};

/// An attribute reference with no explicit base, e.g. `.f64` in `foo(.f64)`.
/// The base type is inferred from context (like Swift's leading-dot syntax).
struct InferredAttributeRefNode final : public LValueCapableExprNode,
                                        Identifier {
  InferredAttributeRefNode(SMLoc dotLoc, StringRef spelling,
                           bool isEscapedIdentifier = false)
      : LValueCapableExprNode(kInferredAttributeRef),
        Identifier(spelling, isEscapedIdentifier), dotLoc(dotLoc) {}

  const SMLoc dotLoc;

  static bool classof(const ExprNode *node) {
    return node->kind == kInferredAttributeRef;
  }
  SMLoc getLoc() const override { return dotLoc; }
  SourceRange getAttributeNameRange() const { return getIdentifierRange(); }
  SourceRange getRange() const override { return {dotLoc, getIdentifierLoc()}; }
  ELVIITResult emitLCVIR(ExprDest &dest, IREmitter &emitter,
                         bool isSpeculative) const override;
  CValue emitMatch(IREmitter &emitter, CValue subject,
                   PatternDeclKind patternKind) const override;
  void print(mlir::raw_indented_ostream &os) const override;
};

/// Struct to represent a parsed expression passed as a parameter or argument
/// operand, along with metadata to help overload resolution and call emission.
struct Operand {
  Operand(ExprNode *expr, SMLoc startLoc, ArgUnpackStyle unpackStyle,
          StringAttr name = StringAttr())
      : expr(expr), startLoc(startLoc), unpackStyle(unpackStyle), name(name) {
    assert(unpackStyle != ArgUnpackStyle::kKeyword || name);
  }

  /// This is the expression for the operand value.
  ExprNode *expr;

  /// The location where the keyword (if given) or the value starts.
  const SMLoc startLoc;

  // This indicates whether the operand is a keyword argument or a positional
  // argument and if it is unpacked.
  const ArgUnpackStyle unpackStyle;

  /// This is the name of a keyword operand when kind=kKeyword, else null.
  const StringAttr name;

  SMLoc getLoc() const { return startLoc; }

  /// Return true if this is a positional operand.
  bool isPositional() const {
    return unpackStyle == ArgUnpackStyle::kPositional;
  }

  /// Return true if this is a keyword operand.
  bool isKeyword() const { return unpackStyle == ArgUnpackStyle::kKeyword; }

  /// Return true if this is a positional operand with a string literal
  /// containing the specified string.
  bool isPositionalStringLiteral(StringRef str) const;

  /// Print the operand for debugging.
  void print(mlir::raw_indented_ostream &os) const;
  LLVM_DUMP_METHOD void dump() const;
};

struct CallNode final : public ExprNode {
  CallNode(const ExprNode *callee, SMLoc lparenLoc, ArrayRef<Operand> operands,
           SMLoc rparenLoc)
      : ExprNode(kCall), callee(callee), lparenLoc(lparenLoc),
        operands(operands), rparenLoc(rparenLoc) {}

  const ExprNode *const callee;
  const SMLoc lparenLoc;
  const ArrayRef<Operand> operands;
  const SMLoc rparenLoc;

  static bool classof(const ExprNode *node) { return node->kind == kCall; }
  SMLoc getLoc() const override { return lparenLoc; }
  SourceRange getRange() const override {
    return {callee->getRangeStart(), rparenLoc};
  }
  SourceRange getParenRange() const { return {lparenLoc, rparenLoc}; }
  AnyValue emitIR(ExprDest &dest, IREmitter &emitter) const override;
  CValue emitMatch(IREmitter &emitter, CValue subject,
                   PatternDeclKind patternKind) const override;
  void print(mlir::raw_indented_ostream &os) const override;
};

/// This represents `A[i,j]`.  In the case of slices (e.g. `A[i, ::]`), the
/// slice will be represented with a subexpression.
struct SubscriptNode final : public LValueCapableExprNode {
  SubscriptNode(const ExprNode *base, SMLoc lsquareLoc,
                ArrayRef<Operand> operands, SMLoc rsquareLoc)
      : LValueCapableExprNode(kSubscript), base(base), lsquareLoc(lsquareLoc),
        operands(operands), rsquareLoc(rsquareLoc) {}

  const ExprNode *const base;
  const SMLoc lsquareLoc;
  const ArrayRef<Operand> operands;
  const SMLoc rsquareLoc;

  static bool classof(const ExprNode *node) { return node->kind == kSubscript; }
  SMLoc getLoc() const override { return lsquareLoc; }
  SourceRange getRange() const override {
    return {base->getRangeStart(), rsquareLoc};
  }
  /// Return a source range from '[' to ']'.
  SourceRange getIndexRange() const { return {lsquareLoc, rsquareLoc}; }

  ELVIITResult emitLCVIR(ExprDest &dest, IREmitter &emitter,
                         bool isSpeculative) const override;
  void print(mlir::raw_indented_ostream &os) const override;
};

/// This is an expression that produces a slice value in a SubscriptNode index
/// expression.  These have at least one colon in them, and one, two, or three
/// expressions, e.g. `:`, `: :`, `a:b`, `a::b` etc.
///
/// All the elements of the syntax are optional (and thus may be null!) except
/// for the first colon.
struct SliceLiteralNode final : public ExprNode {
  SliceLiteralNode(ExprNode *lower, SMLoc colon1Loc, ExprNode *upper,
                   SMLoc colon2Loc, ExprNode *stride)
      : ExprNode(kSliceLiteral), lower(lower), colon1Loc(colon1Loc),
        upper(upper), colon2Loc(colon2Loc), stride(stride) {}

  ExprNode *const lower;
  SMLoc colon1Loc;
  ExprNode *const upper;
  SMLoc colon2Loc;
  ExprNode *const stride;

  static bool classof(const ExprNode *node) {
    return node->kind == kSliceLiteral;
  }
  SMLoc getLoc() const override { return colon1Loc; }

  SourceRange getRange() const override {
    auto startLoc = lower ? lower->getRangeStart() : colon1Loc;
    if (stride)
      return {startLoc, stride->getRangeEnd()};
    if (colon2Loc.isValid())
      return {startLoc, colon2Loc};
    if (upper)
      return {startLoc, upper->getRangeEnd()};
    return {startLoc, colon1Loc};
  }

  AnyValue emitIR(ExprDest &dest, IREmitter &emitter) const override;
  void print(mlir::raw_indented_ostream &os) const override;
};

struct ParenNode final : public ExprNode {
  ParenNode(SMLoc lparenLoc, ExprNode *subExpr, SMLoc rparenLoc)
      : ExprNode(kParen), lparenLoc(lparenLoc), subExpr(subExpr),
        rparenLoc(rparenLoc) {}

  const SMLoc lparenLoc;
  ExprNode *const subExpr;
  const SMLoc rparenLoc;

  static bool classof(const ExprNode *node) { return node->kind == kParen; }
  SMLoc getLoc() const override { return lparenLoc; }
  SourceRange getRange() const override { return {lparenLoc, rparenLoc}; }
  AnyValue emitIR(ExprDest &dest, IREmitter &emitter) const override;
  LogicalResult emitDestructuringPValue(PValue value,
                                        IREmitter &emitter) const override;
  CValue emitMatch(IREmitter &emitter, CValue subject,
                   PatternDeclKind patternKind) const override;
  ELVIITResult
  emitLValueIfImplicitlyTyped(IREmitter &emitter, PatternDeclKind kind,
                              bool hasInferrableRHS) const override {
    return subExpr->emitLValueIfImplicitlyTyped(emitter, kind,
                                                hasInferrableRHS);
  }

  void print(mlir::raw_indented_ostream &os) const override;
};

/// `a, b, c` and `a,`.  TupleNode does not carry parens, but is often nested
/// in a ParenNode.
///
/// Note that an empty tuple `()` is represented as a TupleNode no exprs,
/// and the firstCommaLoc is at the `(`.  It is then wrapped with a ParenNode.
struct TupleNode final : public LValueCapableExprNode {
  TupleNode(SMLoc firstCommaLoc, ArrayRef<ExprNode *> exprs)
      : LValueCapableExprNode(kTuple), firstCommaLoc(firstCommaLoc),
        exprs(exprs) {}

  const SMLoc firstCommaLoc;
  ArrayRef<ExprNode *> exprs;

  static bool classof(const ExprNode *node) { return node->kind == kTuple; }
  SMLoc getLoc() const override { return firstCommaLoc; }
  SourceRange getRange() const override {
    if (exprs.empty())
      return {firstCommaLoc, firstCommaLoc};
    return {exprs.front()->getRangeStart(), exprs.back()->getRangeEnd()};
  }
  ELVIITResult emitLCVIR(ExprDest &dest, IREmitter &emitter,
                         bool isSpeculative) const override;
  LogicalResult emitDestructuringPValue(PValue value,
                                        IREmitter &emitter) const override;
  CValue emitMatch(IREmitter &emitter, CValue subject,
                   PatternDeclKind patternKind) const override;
  void print(mlir::raw_indented_ostream &os) const override;
};

/// [a, b, c]
struct ListLiteralNode final : public LValueCapableExprNode {
  ListLiteralNode(SMLoc lsquareLoc, ArrayRef<ExprNode *> exprs,
                  SMLoc rsquareLoc)
      : LValueCapableExprNode(kListLiteral), lsquareLoc(lsquareLoc),
        exprs(exprs), rsquareLoc(rsquareLoc) {}

  const SMLoc lsquareLoc;
  ArrayRef<ExprNode *> exprs;
  const SMLoc rsquareLoc;

  static bool classof(const ExprNode *node) {
    return node->kind == kListLiteral;
  }
  SMLoc getLoc() const override { return lsquareLoc; }
  SourceRange getRange() const override { return {lsquareLoc, rsquareLoc}; }
  ELVIITResult emitLCVIR(ExprDest &dest, IREmitter &emitter,
                         bool isSpeculative) const override;
  LogicalResult emitDestructuringPValue(PValue value,
                                        IREmitter &emitter) const override;
  void print(mlir::raw_indented_ostream &os) const override;
};

/// This struct represents a parsed list comprehension clause. This has two
/// forms:
///   for pattern in expr
///   if cond
/// These can be chained, e.g. [i for i in range(10) if i % 2 == 0]
///
struct ComprehensionClause {
  SMLoc kwLoc; // Location of 'for' or 'if'
  enum Kind {
    kFor,
    kIf,
  } kind;
  ExprNode *const forPattern; // pattern for a for.
  ExprNode *const expr;       // cond for an if, range for 'for'
};

/// [a    for a in range    if cond]
/// {a    for a in range    if cond}
/// {a: a * a    for a in range    if cond}
struct ComprehensionNode final : public ExprNode {
  ComprehensionNode(ExprNode::Kind kind, SMLoc lsquareLoc, ExprNode *expr,
                    ExprNode *valueExpr, ArrayRef<ComprehensionClause> clauses,
                    SMLoc rsquareLoc)
      : ExprNode(kind), lsquareLoc(lsquareLoc), expr(expr),
        valueExpr(valueExpr), clauses(clauses), rsquareLoc(rsquareLoc) {}

  const SMLoc lsquareLoc;
  ExprNode *const expr;
  ExprNode *const valueExpr; // Used for dict comprehension.
  ArrayRef<ComprehensionClause> clauses;
  const SMLoc rsquareLoc;

  static bool classof(const ExprNode *node) {
    return node->kind == kListComprehension ||
           node->kind == kSetComprehension || node->kind == kDictComprehension;
  }
  SMLoc getLoc() const override { return lsquareLoc; }
  SourceRange getRange() const override { return {lsquareLoc, rsquareLoc}; }
  AnyValue emitIR(ExprDest &dest, IREmitter &emitter) const override;
  void print(mlir::raw_indented_ostream &os) const override;
};

/// This represents `{key1: value1, key2: value2, **dictunpack}` expressions.
/// The dictionary unpacking syntax is represented with a null key and with the
/// unpack expression as the value.
struct DictLiteralNode final : public ExprNode {
  DictLiteralNode(SMLoc lbraceLoc,
                  ArrayRef<std::pair<ExprNode *, ExprNode *>> values,
                  SMLoc rbraceLoc)
      : ExprNode(kDictLiteral), lbraceLoc(lbraceLoc), values(values),
        rbraceLoc(rbraceLoc) {}

  const SMLoc lbraceLoc;
  const ArrayRef<std::pair<ExprNode *, ExprNode *>> values;
  const SMLoc rbraceLoc;

  static bool classof(const ExprNode *node) {
    return node->kind == kDictLiteral;
  }
  SMLoc getLoc() const override { return lbraceLoc; }
  SourceRange getRange() const override { return {lbraceLoc, rbraceLoc}; }

  AnyValue emitIR(ExprDest &dest, IREmitter &emitter) const override;
  void print(mlir::raw_indented_ostream &os) const override;
};

/// This represents `{v1, v2}` and `{v1, arg=v2}` expressions which are either
/// set literals or initializer lists.
struct SetInitLiteralNode final : public ExprNode {
  SetInitLiteralNode(SMLoc lbraceLoc,
                     ArrayRef<std::pair<ExprNode *, ExprNode *>> values,
                     SMLoc rbraceLoc)
      : ExprNode(kSetInitLiteral), lbraceLoc(lbraceLoc), values(values),
        rbraceLoc(rbraceLoc) {}

  const SMLoc lbraceLoc;
  const ArrayRef<std::pair<ExprNode *, ExprNode *>> values;
  const SMLoc rbraceLoc;

  static bool classof(const ExprNode *node) {
    return node->kind == kSetInitLiteral;
  }
  SMLoc getLoc() const override { return lbraceLoc; }
  SourceRange getRange() const override { return {lbraceLoc, rbraceLoc}; }

  AnyValue emitIR(ExprDest &dest, IREmitter &emitter) const override;
  void print(mlir::raw_indented_ostream &os) const override;
};

// trueExpr 'if' condition 'else' falseExpr
struct IfElseOpNode final : public ExprNode {
  IfElseOpNode(ExprNode *trueExpr, SMLoc ifLoc, ExprNode *condExpr,
               SMLoc elseLoc, ExprNode *falseExpr)
      : ExprNode(kIfElse), trueExpr(trueExpr), ifLoc(ifLoc), condExpr(condExpr),
        elseLoc(elseLoc), falseExpr(falseExpr) {}

  ExprNode *const trueExpr;
  const SMLoc ifLoc;
  ExprNode *const condExpr;
  const SMLoc elseLoc;
  ExprNode *const falseExpr;

  static bool classof(const ExprNode *node) { return node->kind == kIfElse; }

  SMLoc getLoc() const override { return ifLoc; }
  SourceRange getRange() const override {
    return {trueExpr->getRangeStart(), falseExpr->getRangeEnd()};
  }
  AnyValue emitIR(ExprDest &dest, IREmitter &emitter) const override;
  void print(mlir::raw_indented_ostream &os) const override;
};

struct BinOpNode final : public ExprNode {
  BinOpNode(Kind kind, ExprNode *lhs, SMLoc opLoc, ExprNode *rhs)
      : ExprNode(kind), lhs(lhs), opLoc(opLoc), rhs(rhs) {}

  ExprNode *const lhs;
  const SMLoc opLoc;
  ExprNode *const rhs;

  static bool classof(const ExprNode *node) {
    return node->kind >= kFirstBinOp && node->kind <= kLastBinOp;
  }

  /// Return true if this is an "assignment stmt" node like =, +=, or *=.
  bool isAssignmentStmt() const {
    return kind >= kFirstAssignStmt && kind <= kLastAssignStmt;
  }

  SMLoc getLoc() const override { return opLoc; }
  SourceRange getRange() const override {
    return {lhs->getRangeStart(), rhs->getRangeEnd()};
  }
  ELVIITResult
  emitLValueIfImplicitlyTyped(IREmitter &emitter, PatternDeclKind kind,
                              bool hasInferrableRHS) const override;
  AnyValue emitIR(ExprDest &dest, IREmitter &emitter) const override;
  CValue emitMatch(IREmitter &emitter, CValue subject,
                   PatternDeclKind patternKind) const override;
  void print(mlir::raw_indented_ostream &os) const override;

private:
  AnyValue emitAndOr(ExprDest &dest, IREmitter &emitter) const;
  AnyValue emitAssign(ExprDest &dest, IREmitter &emitter) const;
  AnyValue emitWalrus(ExprDest &dest, IREmitter &emitter) const;
  AnyValue emitInplace(ExprDest &dest, IREmitter &emitter) const;
  CValue emitAsMatch(IREmitter &emitter, CValue subject,
                     PatternDeclKind patternKind) const;
  CValue emitOrMatch(IREmitter &emitter, CValue subject,
                     PatternDeclKind patternKind) const;
};

struct UnaryOpNode final : public ExprNode {
  UnaryOpNode(Kind kind, SMLoc opLoc, ExprNode *subExpr)
      : ExprNode(kind), opLoc(opLoc), subExpr(subExpr) {}

  const SMLoc opLoc;
  ExprNode *const subExpr;

  static bool classof(const ExprNode *node) {
    return node->kind >= kFirstUnaryOp && node->kind <= kLastUnaryOp;
  }
  bool isPostfix() const { return kind == kTransfer; }
  SMLoc getLoc() const override { return opLoc; }
  SourceRange getRange() const override {
    return isPostfix() ? SourceRange(subExpr->getRangeStart(), opLoc)
                       : SourceRange(opLoc, subExpr->getRangeEnd());
  }
  AnyValue emitIR(ExprDest &dest, IREmitter &emitter) const override;
  CValue emitMatch(IREmitter &emitter, CValue subject,
                   PatternDeclKind patternKind) const override;
  ELVIITResult
  emitLValueIfImplicitlyTyped(IREmitter &emitter, PatternDeclKind kind,
                              bool hasInferrableRHS) const override;
  void print(mlir::raw_indented_ostream &os) const override;
  AnyValue emitTransfer(AnyValue argValue, ExprDest &dest,
                        IREmitter &emitter) const;
  AnyValue emitComptime(ExprDest &dest, IREmitter &emitter) const;

  /// Emit a unary arithmetic operation.
  static AnyValue emitArith(Kind kind, const ExprNode *expr,
                            ASTExprAnd<AnyValue> value, ExprDest &dest,
                            IREmitter &emitter);
};

/// This represents a chained comparison expression (ex. a < b <= c).
/// exprs stores all the expressions in the comparison (ex. a, b, c), while
/// ops stores the ops in between pairs of expressions (ex. <, <=).
/// Chained expressions are evaluated left to right and each expression is
/// valuated at most once: a < b <= c is equivalent to a < b and b <= c, but
/// b is evaluated only once.
struct ChainedCmpOpNode final : public ExprNode {
  ChainedCmpOpNode(ArrayRef<ExprNode *> exprs, ArrayRef<ExprNode::Kind> ops,
                   SMLoc opLoc)
      : ExprNode(ExprNode::Kind::kChainedCmp), exprs(exprs), ops(ops),
        opLoc(opLoc) {}

  const ArrayRef<ExprNode *> exprs;
  const ArrayRef<ExprNode::Kind> ops;
  const SMLoc opLoc;

  SMLoc getLoc() const override { return opLoc; }
  SourceRange getRange() const override {
    return {exprs.front()->getRangeStart(), exprs.back()->getRangeEnd()};
  }

  AnyValue emitIR(ExprDest &dest, IREmitter &emitter) const override;
  void print(mlir::raw_indented_ostream &os) const override;
  RValue emitNextCmp(IREmitter &emitter, size_t opIdx, RValue lastCmp,
                     AnyValue lastExpr, bool hasPrevIfOp, ExprDest &dest) const;
};

struct FunctionTypeNode final : public ExprNode {
  FunctionTypeNode(SMLoc baseLoc, ArrayRef<ParsedArgument> parsedParams,
                   ArrayRef<ParsedArgument> parsedArgs,
                   const ParsedArgument &resultArg, FnEffects effects,
                   bool isThin, bool isExperimentalParamTrait,
                   const ExprNode *thrownTypeExpr, const ExprNode *originExpr,

                   ArrayRef<ParsedConstraint> parsedConstraints, SMLoc endLoc)
      : ExprNode(kFunctionType), baseLoc(baseLoc), parsedParams(parsedParams),
        parsedArgs(parsedArgs), resultArg(resultArg), effects(effects),
        isThin(isThin), isExperimentalParamTrait(isExperimentalParamTrait),
        thrownTypeExpr(thrownTypeExpr), originExpr(originExpr),
        parsedConstraints(parsedConstraints), endLoc(endLoc) {}

  SMLoc baseLoc;
  ArrayRef<ParsedArgument> parsedParams; // Parameter list
  ArrayRef<ParsedArgument> parsedArgs;   // Argument list
  const ParsedArgument &resultArg;       // Result argument
  FnEffects effects;
  bool isThin;
  bool isExperimentalParamTrait;
  const ExprNode *thrownTypeExpr;
  const ExprNode *originExpr;
  ArrayRef<ParsedConstraint> parsedConstraints; // Trailing body constraints
  SMLoc endLoc;

  static bool classof(const ExprNode *node) {
    return node->kind == kFunctionType;
  }
  SMLoc getLoc() const override { return baseLoc; }
  SourceRange getRange() const override { return {baseLoc, endLoc}; }
  AnyValue emitIR(ExprDest &dest, IREmitter &emitter) const override;
  void print(mlir::raw_indented_ostream &os) const override;
};

/// `__generator_type[Idx: Int] ToT` — a `!lit.generator<<params> body>` type
/// literal. Parameters are optional (`__generator_type Int` for no params).
struct GeneratorTypeNode final : public ExprNode {
  GeneratorTypeNode(SMLoc baseLoc, ArrayRef<ParsedArgument> parsedParams,
                    const ExprNode *bodyTypeExpr, SMLoc endLoc)
      : ExprNode(kGeneratorType), baseLoc(baseLoc), parsedParams(parsedParams),
        bodyTypeExpr(bodyTypeExpr), endLoc(endLoc) {}

  SMLoc baseLoc;
  ArrayRef<ParsedArgument> parsedParams;
  const ExprNode *bodyTypeExpr;
  SMLoc endLoc;

  static bool classof(const ExprNode *node) {
    return node->kind == kGeneratorType;
  }
  SMLoc getLoc() const override { return baseLoc; }
  SourceRange getRange() const override { return {baseLoc, endLoc}; }
  AnyValue emitIR(ExprDest &dest, IREmitter &emitter) const override;
  void print(mlir::raw_indented_ostream &os) const override;
};

/// A `lambda` expression: a closure literal whose body is a single expression.
/// Parsed syntactically here; the closure is constructed at emit time. Field
/// order mirrors the parsed grammar, which follows nested-`def` closures:
///
/// lambda [parsedParams] (parsedArgs) <effects> {captures} -> resultArg : body
struct LambdaNode final : public ExprNode {
  LambdaNode(SMLoc baseLoc, ArrayRef<ParsedArgument> parsedParams,
             ArrayRef<ParsedArgument> parsedArgs,
             const ParsedArgument &resultArg, FnEffects effects,
             const ExprNode *thrownTypeExpr,
             ArrayRef<std::tuple<StringRef, CaptureConvention, SMLoc>> captures,
             bool hasExplicitCaptureList,
             std::optional<CaptureConvention> captureAllByConvention,
             const ExprNode *body, SMLoc endLoc)
      : ExprNode(kLambda), baseLoc(baseLoc), parsedParams(parsedParams),
        parsedArgs(parsedArgs), resultArg(resultArg), effects(effects),
        thrownTypeExpr(thrownTypeExpr), captures(captures),
        hasExplicitCaptureList(hasExplicitCaptureList),
        captureAllByConvention(captureAllByConvention), body(body),
        endLoc(endLoc) {}

  SMLoc baseLoc;
  ArrayRef<ParsedArgument> parsedParams; // [parameter_list]
  ArrayRef<ParsedArgument> parsedArgs;   // (argument_list)
  const ParsedArgument &resultArg;       // -> result_type (default if elided)
  FnEffects effects;                     // raises, async, etc.
  const ExprNode *thrownTypeExpr;        // explicit `raises T`, else null
  // The capture list `{...}` decomposes into two parts that are NOT mutually
  // exclusive and may both be set by a single list (e.g. `{mut, x}`):
  //
  //   `captures`              — explicitly named captures, one (name,
  //                             convention, loc) per entry (`{imm x, mut y}`;
  //                             a bare name defaults to `imm`, so `{y}` is
  //                             `{imm y}`).
  //   `captureAllByConvention` — the default convention from a bare convention
  //                             with no name (`{mut}`/`{imm}`), applied to all
  //                             *implicitly* captured variables.
  //
  // When both are present, the named entries act as per-variable overrides on
  // top of the default (`{mut, x}` = capture everything `mut`, but `x` `imm`).
  ArrayRef<std::tuple<StringRef, CaptureConvention, SMLoc>> captures;
  bool hasExplicitCaptureList; // whether a `{...}` was written at all
  std::optional<CaptureConvention> captureAllByConvention;
  const ExprNode *body; // the single body expression
  SMLoc endLoc;

  static bool classof(const ExprNode *node) { return node->kind == kLambda; }
  SMLoc getLoc() const override { return baseLoc; }
  SourceRange getRange() const override { return {baseLoc, endLoc}; }
  AnyValue emitIR(ExprDest &dest, IREmitter &emitter) const override;
  void print(mlir::raw_indented_ostream &os) const override;
};

/// __get_value_from_rvalue(some_ref)      # returns LValue or BValue
/// __get_address_as_owned_value(some_ptr) # returns RValue
/// origin_of(decl)                        # returns !lit.origin<mut>
struct MagicFunctionNode final : public ExprNode {
  MagicFunctionNode(ExprNode::Kind kind, SMLoc baseLoc, StringRef spelling,
                    ArrayRef<ExprNode *> subExprs, SMLoc rparenLoc)
      : ExprNode(kind), baseLoc(baseLoc), spelling(spelling),
        subExprs(subExprs), rparenLoc(rparenLoc) {
    assert(classof(this) && "Kind is wrong");
  }

  const SMLoc baseLoc;
  const StringRef spelling; // The magic function keyword as written in source.
  const ArrayRef<ExprNode *> subExprs;
  const SMLoc rparenLoc;

  static bool classof(const ExprNode *node) {
    return node->kind >= kFirstMagicFunction &&
           node->kind <= kLastMagicFunction;
  }
  SMLoc getLoc() const override { return baseLoc; }
  SourceRange getRange() const override { return {baseLoc, rparenLoc}; }
  AnyValue emitIR(ExprDest &dest, IREmitter &emitter) const override;
  void print(mlir::raw_indented_ostream &os) const override;

  AnyValue emitOriginOf(ExprDest &dest, IREmitter &emitter) const;
  AnyValue emitTypeOf(ExprDest &dest, IREmitter &emitter) const;
  AnyValue emitFunctionsInModule(ExprDest &dest, IREmitter &emitter) const;
  AnyValue emitConformsTo(ExprDest &dest, IREmitter &emitter) const;
  AnyValue emitGetCurrentFunctionName(ExprDest &dest, IREmitter &emitter) const;
  AnyValue emitStructFieldRef(ExprDest &dest, IREmitter &emitter) const;
  AnyValue emitIsRunInComptimeInterpreter(ExprDest &dest,
                                          IREmitter &emitter) const;

private:
  /// Helper to validate and extract the type argument for struct field
  /// reflection magic functions. Returns the PValue if valid, or emits an
  /// error and returns std::nullopt.
  std::optional<PValue>
  getValidatedStructTypeArg(IREmitter &emitter, StringRef publicApiName) const;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_EXPRNODES_H
