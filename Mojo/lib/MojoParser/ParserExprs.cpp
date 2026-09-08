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
// Expression parsing in Mojo is done in with a 2-phase approach where we
// parse one or more expressions into an AST-like representation in a first
// pass, then type check and generate operations or a type for it in a second
// pass.  This enables a number of features:
//
//   1) Non-lexical variable references: `[x.strip().upper() for x in flags]`.
//      Where we can only type check the expression after the 'for x' is type
//      checked and resolved.
//   2) Weird order of evaluations: `foo() if cond() else bar()`
//   3) Parser ambiguity of the LHS of an assignment, which we don't know if it
//      is a target until we see the equals: `x[foo()] = bar()`
//
// We handle this by having an expression parser distinct from the main parser
// that builds this tree.
//
//===----------------------------------------------------------------------===//

#include "Signatures.h"

#include "ExprNodes.h"
#include "Mojo/MojoParser/Lexer.h"
#include "ParserBase.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace M::KGEN::LIT;
using namespace M;

//===----------------------------------------------------------------------===//
// Expression Parsing
//===----------------------------------------------------------------------===//

using Precedence = ParserBase::Precedence;

namespace M::KGEN::LIT {
/// This class implements the ExprParser interface, implemented with the pImpl
/// idiom.
class ExprParser : public ParserBase {
public:
  ExprParser(SharedState &shared, Lexer &lexer,
             std::optional<size_t> stmtIndent)
      : ParserBase(shared, lexer), stmtIndent(stmtIndent) {}

  ~ExprParser() = default;

  // Expressions.
  ParseResult parseStarredItemList(SmallVectorImpl<ExprNode *> &results,
                                   ArrayRef<Token::Kind> terminators,
                                   bool allowAssign,
                                   SMLoc *firstCommaLoc = nullptr);
  ParseResult parseStarredExprListAsTuple(ExprNode *&result,
                                          ArrayRef<Token::Kind> terminators,
                                          bool allowAssign,
                                          bool allowVarRef = false);

  ParseResult parseExpression(ExprNode *&result,
                              Precedence minPrec = Precedence::kExpression);
  ParseResult parseStarredItem(ExprNode *&result, bool allowAssign);

  template <typename T, typename... Args>
  T *alloc(Args &&...args) {
    return shared.allocPersistent<T>(std::forward<Args>(args)...);
  }

  template <typename T>
  ArrayRef<T> copyArrayRef(ArrayRef<T> elements) {
    return shared.getPersistentCopy(elements);
  }

  /// Return true if the current token is part of the current statement, false
  /// if it is the start of a new one.
  bool isTokenInCurrentStatement() const {
    return ParserBase::isTokenInCurrentStatement(stmtIndent);
  }

private:
  ParseResult parsePrimaryExpr(ExprNode *&result);
  ParseResult parsePrefixLParen(ExprNode *&result, SMLoc lparenLoc);
  ParseResult parsePrefixLSquare(ExprNode *&result, SMLoc lsquareLoc);
  ParseResult parsePrefixLBrace(ExprNode *&result, SMLoc lbraceLoc);
  ParseResult parseAttributeRefSuffix(ExprNode *&result, SMLoc dotLoc);
  FailureOr<Operand> parseOperand(
      function_ref<ParseResult(ExprNode *&, Precedence)> parseOperandValue);
  ParseResult parseCallSuffix(ExprNode *&result, SMLoc lparenLoc);
  ParseResult parseExprOrSlice(ExprNode *&result);
  ParseResult parseSubscriptSuffix(ExprNode *&result, SMLoc lsquareLoc);
  ParseResult parseComparisonExpr(ExprNode *&result, ExprNode *rhs,
                                  ExprNode::Kind kind, SMLoc loc);
  ParseResult parseFunctionType(ExprNode *&result);
  ParseResult parseGeneratorType(ExprNode *&result);
  ParseResult parseLambda(ExprNode *&result);
  ParseResult parseMagicFunction(ExprNode *&result);
  ParseResult
  parseTStringFromSpelling(const Token &tstrTok,
                           SmallVectorImpl<TStringExprNode::Part> &parts);

  ParseResult parseComprehension(ExprNode *&result, ExprNode::Kind kind,
                                 SMLoc startLoc, ExprNode *expr,
                                 ExprNode *value);

  /// Check if the given operands (e.g. in a `(...)` call or `[...]` subscript)
  /// adhere to the Python grammar. Positional operands cannot appear after
  /// keyword operands, and duplicate keyword operands are not allowed. If the
  /// `isArgument` flag is true, operands are checked as if dynamic runtime
  /// operands (and explicitly unbound packs, i.e. `*_` are not allowed).
  /// Otherwise, operands are considered parameters.
  ParseResult checkOperands(ArrayRef<Operand>, bool isArgument);

  /// This specifies the indentation level of the start of the statement that
  /// contains this expression if the expression can exist at the end of the
  /// line.  This allows the expression parser to know when to keep parsing the
  /// expression on the next line - when it is more indented than the start of
  /// the current statement.  This is None when there is a trailing punctuator
  /// that naturally terminates the expression.
  std::optional<size_t> stmtIndent;
};
} // namespace M::KGEN::LIT

//===----------------------------------------------------------------------===//
// Parsing rules
//===----------------------------------------------------------------------===//

/// starred_item_list  ::=  starred_item ("," starred_item)* [","]
ParseResult
ExprParser::parseStarredItemList(SmallVectorImpl<ExprNode *> &results,
                                 ArrayRef<Token::Kind> terminators,
                                 bool allowAssign, SMLoc *firstCommaLoc) {
  auto parseItem = [&]() -> ParseResult {
    return parseStarredItem(results.emplace_back(nullptr), allowAssign);
  };

  return parseCommaSeparatedList(parseItem, terminators, stmtIndent,
                                 firstCommaLoc);
}

/// Parse a starred_list, forming a single TupleExpr if a comma is present.
/// Note that starred_item includes assignments, starred_expressions do not.
///     starred_expression  ::=  ["*"] or_expr
/// When `allowVarRef` is true, a leading `var`/`ref` takes this whole list
/// (`var x, y` → `var (x, y)`). Expression clients leave it false so prefix
/// `var`/`ref` bind a single subexpression instead.
ParseResult
ExprParser::parseStarredExprListAsTuple(ExprNode *&result,
                                        ArrayRef<Token::Kind> terminators,
                                        bool allowAssign, bool allowVarRef) {
  // Statement-level `var x, y` / `ref x, y` take a starred list. Prefix
  // `var`/`ref` in expressions bind a single subexpression instead.
  if (allowVarRef && getToken().isAny(Token::kw_var, Token::kw_ref)) {
    auto kind =
        getToken().is(Token::kw_var) ? ExprNode::kVarPat : ExprNode::kRefPat;
    SMLoc opLoc = consumeToken().getLoc();
    ExprNode *inner = nullptr;
    if (parseStarredExprListAsTuple(inner, terminators, /*allowAssign=*/false))
      return failure();
    result = alloc<UnaryOpNode>(kind, opLoc, inner);
    return success();
  }

  SmallVector<ExprNode *> exprs;
  SMLoc firstCommaLoc;
  if (parseStarredItemList(exprs, terminators, allowAssign, &firstCommaLoc))
    return failure();

  // If there was a tuple inside the parens, form it.
  if (firstCommaLoc.isValid())
    result = alloc<TupleNode>(firstCommaLoc, copyArrayRef<ExprNode *>(exprs));
  else {
    assert(exprs.size() == 1);
    result = exprs[0];
  }
  return success();
}

namespace {
/// This struct bundles up information related to infix binary operations.
struct InfixInfo {
  Precedence precedence;
  ExprNode::Kind nodeKind;

  // True when this operator is right associative:
  //   https://en.wikipedia.org/wiki/Operator_associativity
  // This matters when operators are at the same precedence level.  Consider
  // 7 op 4 op 2. The result could be either `(7 op 4) op 2` or `7 op (4 op 2)`.
  // The former result corresponds to the case the operators are
  // left-associative, the latter to when they are right-associative.
  //
  // Almost all operators in Python/Mojo are left associative.  Exceptions are
  // the power operator and assignment operator `=`.
  bool isRightAssociative;

  /// Classify a token for an infix operator.
  static InfixInfo get(Token::Kind tokKind, ParserBase &p) {
    // Helper to reduce boilerplate with isRightAssociative.
    auto get = [](ParserBase::Precedence precedence, ExprNode::Kind nodeKind,
                  bool isRightAssociative = false) -> InfixInfo {
      return {precedence, nodeKind, isRightAssociative};
    };

    switch (tokKind) {
    default:
      return get(Precedence::kInvalid, ExprNode::kLastBinOp);
    case Token::plus:
      return get(Precedence::kSum, ExprNode::kAdd);
    case Token::minus:
      return get(Precedence::kSum, ExprNode::kSub);
    case Token::star:
      return get(Precedence::kTerm, ExprNode::kMul);
    case Token::at:
      return get(Precedence::kTerm, ExprNode::kMatMul);
    case Token::slash:
      return get(Precedence::kTerm, ExprNode::kTrueDiv);
    case Token::slash_slash:
      return get(Precedence::kTerm, ExprNode::kFloorDiv);
    case Token::percent:
      return get(Precedence::kTerm, ExprNode::kMod);
    case Token::kw_or:
      return get(Precedence::kBoolOr, ExprNode::kBoolOr);
    case Token::kw_and:
      return get(Precedence::kBoolAnd, ExprNode::kBoolAnd);
    case Token::kw_not: {
      LexerCursor c;
      std::ignore = p.getCursor(c);
      p.consumeToken();
      Token next = p.getToken();
      c.restore(p.lexer);
      if (next.getKind() == Token::kw_in)
        return get(Precedence::kComparison, ExprNode::kCmpNotIn);
      // `not` by itself is not an infix operator, it is prefix.
      return get(Precedence::kInvalid, ExprNode::kLastBinOp);
    }
    case Token::kw_in:
      return get(Precedence::kComparison, ExprNode::kCmpIn);
    case Token::kw_is:
      return get(Precedence::kComparison, ExprNode::kCmpIs);
    case Token::less:
      return get(Precedence::kComparison, ExprNode::kCmpLT);
    case Token::less_equal:
      return get(Precedence::kComparison, ExprNode::kCmpLE);
    case Token::greater:
      return get(Precedence::kComparison, ExprNode::kCmpGT);
    case Token::greater_equal:
      return get(Precedence::kComparison, ExprNode::kCmpGE);
    case Token::exclaim_equal:
      return get(Precedence::kComparison, ExprNode::kCmpNE);
    case Token::equal_equal:
      return get(Precedence::kComparison, ExprNode::kCmpEQ);
    case Token::pipe:
      return get(Precedence::kOr, ExprNode::kOr);
    case Token::caret:
      return get(Precedence::kXor, ExprNode::kXor);
    case Token::amp:
      return get(Precedence::kAnd, ExprNode::kAnd);
    case Token::less_less:
      return get(Precedence::kShift, ExprNode::kLShift);
    case Token::right_right:
      return get(Precedence::kShift, ExprNode::kRShift);
    case Token::kw_if:
      return get(Precedence::kIfElse, ExprNode::kIfElse,
                 /*isRightAssociative=*/true);
    case Token::kw_var:
      return get(Precedence::kVarRefPat, ExprNode::kVarPat);
    case Token::kw_ref:
      return get(Precedence::kVarRefPat, ExprNode::kRefPat);
    case Token::star_star:
      return get(Precedence::kPower, ExprNode::kPow,
                 /*isRightAssociative=*/true);
    case Token::colon_equal:
      return get(Precedence::kAssignExpr, ExprNode::kWalrus);
    case Token::kw_as:
      return get(Precedence::kAsPat, ExprNode::kAsPat);
    }
  }
};
} // namespace

/// Parse an expression using top-down operator precedence parsing.  minPrec
/// specifies the minimum precedence that binary sub-expression must have to be
/// included.  Anything looser than the specified precedence is left for a
/// parent expression to parse.
ParseResult ExprParser::parseExpression(ExprNode *&result, Precedence minPrec) {
  // Parse any prefix expression like -1.
  if (parsePrimaryExpr(result))
    return failure();

  // Consume infix tokens until we meet a token whose tokPrecedence is equal or
  // lower than minPrec. This means that it collects all tokens that bind
  // together before returning to the operator that called it.
  InfixInfo infixInfo = InfixInfo::get(getToken().getKind(), *this);
  while (isTokenInCurrentStatement() && minPrec <= infixInfo.precedence) {
    Token::Kind tokKind = getToken().getKind();
    auto binOpLoc = consumeToken().getLoc();

    ExprNode *ifElseCond;
    SMLoc elseLoc;
    if (tokKind == Token::Kind::kw_if && minPrec <= Precedence::kIfElse) {
      // Conditional if - else expression.
      // trueExpr 'if' condition 'else' falseExpr.
      // If/else operator needs special handling because it has an expression in
      // the middle of what can otherwise be parsed like a binary operator.
      if (parseExpression(ifElseCond, Precedence(int(Precedence::kIfElse) + 1)))
        return failure();
      elseLoc = getToken().getLoc();
      if (parseToken(Token::Kind::kw_else, "expected 'else' clause in ternary; "
                                           "add 'else' and the false branch"))
        return failure();
    }

    // rhs 'is' 'not' lhs -> a is not True.
    if (tokKind == Token::Kind::kw_is && consumeIf(Token::Kind::kw_not))
      infixInfo.nodeKind = ExprNode::Kind::kCmpIsNot;
    // rhs 'not' 'in' lhs -> a not in {1, 2}.
    else if (tokKind == Token::Kind::kw_not && consumeIf(Token::Kind::kw_in)) {
      infixInfo.nodeKind = ExprNode::Kind::kCmpNotIn;
      infixInfo.precedence = Precedence::kComparison;
    }

    // Right associative operations can parse anything at the current operator
    // level on the right side, but left associative operators consume RHS that
    // binds more tightly than the current operator.
    Precedence subExprPrec = infixInfo.precedence;
    if (!infixInfo.isRightAssociative)
      subExprPrec = Precedence(unsigned(infixInfo.precedence) + 1);

    ExprNode *rhs = nullptr;
    if (parseExpression(rhs, subExprPrec))
      return failure();

    if (infixInfo.precedence == Precedence::kComparison) {
      // Comparison operators get special handling to treat 'a < b < c' as a
      // ChainedCmpOpNode.
      if (parseComparisonExpr(result, rhs, infixInfo.nodeKind, binOpLoc))
        return failure();
    } else if (tokKind == Token::Kind::kw_if) {
      result = alloc<IfElseOpNode>(result, binOpLoc, ifElseCond, elseLoc, rhs);
    } else
      result = alloc<BinOpNode>(infixInfo.nodeKind, result, binOpLoc, rhs);
    infixInfo = InfixInfo::get(getToken().getKind(), *this);
  }
  return success();
}

/// If allowAssign is true then we use:
///    starred_item       ::= assignment_expression | "*" or_expr
/// otherwise:
///    starred_item       ::= ["*"] or_expr
ParseResult ExprParser::parseStarredItem(ExprNode *&result, bool allowAssign) {
  // If allowAssign is true then we use a much more permissive subexpression
  // precedence, matching things like "in". If false, we allow limited things
  // that is more like a pattern to avoid interfering with "for x in y".
  auto subPrec = allowAssign ? Precedence::kAssignExpr : Precedence::kVarRefPat;

  SMLoc starLoc;
  if (consumeIf(Token::star, &starLoc))
    subPrec = Precedence::kOr; // Star always forces 'or' precedence.

  if (parseExpression(result, subPrec))
    return failure();
  if (starLoc.isValid())
    result = alloc<UnaryOpNode>(ExprNode::kUnpack, starLoc, result);
  return success();
}

/// Parse a chained comparison expression (ex. a < b < c) starting from the
/// first comparison given as input:
/// expr is the lhs and kind specifies the type of comparison, ex. kCmpLT.
/// This function returns in expr a ChainedCmpOpNode on success.
ParseResult ExprParser::parseComparisonExpr(ExprNode *&result, ExprNode *rhs,
                                            ExprNode::Kind kind, SMLoc loc) {
  SmallVector<ExprNode *> exprs;
  SmallVector<ExprNode::Kind> ops;
  exprs.push_back(result);
  exprs.push_back(rhs);
  ops.push_back(kind);
  InfixInfo infixInfo = InfixInfo::get(getToken().getKind(), *this);
  while (isTokenInCurrentStatement() &&
         infixInfo.precedence == Precedence::kComparison) {
    consumeToken();
    ExprNode *cmpOperand;
    if (parseExpression(cmpOperand, Precedence::kOr))
      return failure();
    exprs.push_back(cmpOperand);
    ops.push_back(infixInfo.nodeKind);
    infixInfo = InfixInfo::get(getToken().getKind(), *this);
  }
  result = alloc<ChainedCmpOpNode>(copyArrayRef<ExprNode *>(exprs),
                                   copyArrayRef<ExprNode::Kind>(ops), loc);
  return success();
}

/// Return true if the specified token kind is the start of a primary
/// expression.
bool ParserBase::isPrimaryExprStart(Token::Kind tokKind) {
  switch (tokKind) {
  case Token::plus:
  case Token::minus:
  case Token::tilde:
  case Token::star:
  case Token::kw_await:
  case Token::kw_not:
  case Token::kw_var:
  case Token::kw_ref:
  case Token::kw_comptime:
  case Token::identifier:
  case Token::escaped_identifier:
  case Token::integer:
  case Token::kw_False:
  case Token::kw_True:
  case Token::kw_Self:
  case Token::kw__:
  case Token::dot_dot_dot:
  case Token::dot:
  case Token::float_num:
  case Token::string:
  case Token::t_string:
  case Token::kw_None:
  case Token::l_paren:
  case Token::l_square:
  case Token::l_brace:
  case Token::kw_async:
  case Token::kw_def:
  case Token::kw_lambda:
  case Token::kw_fn:
  case Token::kw___generator_type:
  case Token::kw___get_mvalue_as_litref:
  case Token::kw___get_litref_as_mvalue:
  case Token::kw___get_address_as_owned_value:
  case Token::kw___get_address_as_uninit_lvalue:
  case Token::kw_conforms_to:
  case Token::kw_origin_of:
  case Token::kw_type_of:
  case Token::kw___functions_in_module:
  case Token::kw___get_current_function_name:
  case Token::kw___struct_field_ref:
  case Token::kw___is_run_in_comptime_interpreter:
#ifndef MODULAR_PRODUCTION
  case Token::kw___mojo_crash:
#endif // MODULAR_PRODUCTION
    return true;
  default:
    return false;
  }
}

/// Given a token for a unary operator like await or ~, return the ExprNode
/// code to use along with the precedence of the subexpression we should parse.
static std::pair<ExprNode::Kind, Precedence>
getUnaryOpInfo(Token::Kind tokKind) {
  switch (tokKind) {
  default:
    llvm_unreachable("invalid unary token");
  case Token::kw_await:
    return {ExprNode::kAwait, Precedence::kPrimary};
  case Token::kw_not:
    return {ExprNode::kBoolNot, Precedence::kBoolNot};
  case Token::kw_var:
    return {ExprNode::kVarPat, Precedence::kVarRefPat};
  case Token::kw_ref:
    return {ExprNode::kRefPat, Precedence::kVarRefPat};
  case Token::kw_comptime:
    return {ExprNode::kComptime, Precedence::kUnpack};
  case Token::plus:
    return {ExprNode::kPos, Precedence::kFactor};
  case Token::minus:
    return {ExprNode::kNeg, Precedence::kFactor};
  case Token::tilde:
    return {ExprNode::kInvert, Precedence::kFactor};
  case Token::star:
    return {ExprNode::kUnpack, Precedence::kUnpack};
  }
}

/// Parse the expression identified by the current token and provided
/// `precedence`.  Store the resulting expression in `expr`.
/// Prefix expressions supported are:
///
/// primary ::=  atom | attributeref | subscription | slicing | call
///
/// atom    ::= identifier | literal | enclosure [TODO]
/// call    ::=  primary "(" [argument_list [","] | comprehension] ")"
///
/// enclosure ::= parenth_form | list_display | dict_display | set_display
///             | generator_expression | yield_atom
/// literal ::=
///     stringliteral | bytesliteral | integer | floatnumber | imagnumber
///
/// u_expr ::=  power | "-" u_expr | "+" u_expr | "~" u_expr
///
ParseResult ExprParser::parsePrimaryExpr(ExprNode *&result) {
  Token startTok = getToken();
  switch (startTok.getKind()) {
  case Token::plus:
  case Token::star:
  case Token::minus:
  case Token::tilde:
  case Token::kw_await:
  case Token::kw_var:
  case Token::kw_ref:
  case Token::kw_comptime:
  case Token::kw_not: { // u_expr
    consumeToken();
    // Get the kind enum and the precedence of the subexpression.
    auto [unaryKind, subExprPrec] = getUnaryOpInfo(startTok.getKind());

    // We limit comptime subexpression to have parens after it.  If we relax
    // this in the future, this special case would be eliminated.
    if (unaryKind == ExprNode::kComptime) {
      SMLoc lparenLoc;
      if (!consumeIf(Token::l_paren, &lparenLoc)) {
        emitTokenError("expected '(' after 'comptime'");
        return failure();
      }
      if (parsePrefixLParen(result, lparenLoc))
        return failure();
      result =
          alloc<UnaryOpNode>(ExprNode::kComptime, startTok.getLoc(), result);
      break;
    }

    ExprNode *expr = nullptr;
    if (parseExpression(expr, subExprPrec))
      return failure();
    result = alloc<UnaryOpNode>(unaryKind, startTok.getLoc(), expr);
    break;
  }
  case Token::identifier: // primary -> atom -> identifier
  case Token::escaped_identifier:
    consumeIdentifier();
    result = alloc<DeclRefNode>(startTok.getSpelling(),
                                startTok.is(Token::escaped_identifier));
    break;
  case Token::integer: // primary -> literal -> integer
    consumeToken(Token::integer);
    result = alloc<IntLiteralNode>(startTok.getSpelling());
    break;
  case Token::kw_False:
    consumeToken(Token::kw_False);
    result = alloc<BoolLiteralNode>(startTok.getLoc(), false);
    break;
  case Token::kw_True:
    consumeToken(Token::kw_True);
    result = alloc<BoolLiteralNode>(startTok.getLoc(), true);
    break;
  case Token::kw_Self:
    consumeToken(Token::kw_Self);
    result =
        alloc<SimpleLiteralNode>(ExprNode::kSelfLiteral, startTok.getLoc());
    break;
  case Token::kw__:
    consumeToken(Token::kw__);
    result =
        alloc<SimpleLiteralNode>(ExprNode::kDiscardLiteral, startTok.getLoc());
    break;
  case Token::dot: {
    // Inferred attribute reference: `.member` (base type inferred from
    // context), e.g. `foo(.f64)`.
    SMLoc dotLoc = startTok.getLoc();
    consumeToken(Token::dot);
    Token nameTok = getToken();
    StringRef spelling = nameTok.getSpelling();
    if (parseIdentifier("expected name in inferred attribute reference",
                        nullptr, /*forbidStartOfLine=*/false,
                        /*allowKeyword=*/true)) {
      // If we didn't get an identifier, recover by using an empty string.
      // Reuse the spelling buffer to preserve the expected location of the
      // identifier.
      spelling = StringRef(spelling.data(), 0);
    }
    result = alloc<InferredAttributeRefNode>(
        dotLoc, spelling, nameTok.is(Token::escaped_identifier));
    break;
  }
  case Token::dot_dot_dot:
    consumeToken(Token::dot_dot_dot);
    result =
        alloc<SimpleLiteralNode>(ExprNode::kEllipsisLiteral, startTok.getLoc());
    break;
  case Token::float_num: // primary -> literal -> floatnumber
    consumeToken(Token::float_num);
    result = alloc<FloatLiteralNode>(startTok.getSpelling());
    break;
  case Token::string: { // primary -> literal -> stringliteral
    SmallVector<StringRef> spellings;
    do {
      spellings.push_back(getToken().getSpelling());
      consumeToken(Token::string);
      // Python supports string literal concatenation
    } while (getToken().is(Token::string) && isTokenInCurrentStatement());
    result = alloc<StringLiteralNode>(copyArrayRef<StringRef>(spellings));
    break;
  }
  case Token::t_string: { // primary -> literal -> t-string
    SmallVector<TStringExprNode::Part> allParts;
    SMLoc tstringStart = startTok.getLoc();
    SMLoc tstringEnd = startTok.getLoc();

    // T-string concatenation: collect consecutive t-strings (like string
    // literals)
    do {
      Token tstrTok = consumeToken(Token::t_string);
      tstringEnd = tstrTok.getEndLoc();
      if (parseTStringFromSpelling(tstrTok, allParts))
        return failure();
    } while (getToken().is(Token::t_string) && isTokenInCurrentStatement());

    result =
        alloc<TStringExprNode>(tstringStart, tstringEnd,
                               copyArrayRef<TStringExprNode::Part>(allParts));
    break;
  }
  case Token::kw_None:
    consumeToken(Token::kw_None);
    result =
        alloc<SimpleLiteralNode>(ExprNode::kNoneLiteral, startTok.getLoc());
    break;
  case Token::l_paren: // primary -> atom -> enclosure -> parenth_form
    consumeToken(Token::l_paren);
    if (parsePrefixLParen(result, startTok.getLoc()))
      return failure();
    break;
  case Token::l_square: // list_display
    consumeToken(Token::l_square);
    if (parsePrefixLSquare(result, startTok.getLoc()))
      return failure();
    break;
  case Token::l_brace: { // dict_display
    consumeToken(Token::l_brace);
    if (parsePrefixLBrace(result, startTok.getLoc()))
      return failure();
    break;
  }

  case Token::kw_async:
  case Token::kw_def:
  case Token::kw_fn:
    if (failed(parseFunctionType(result)))
      return failure();
    break;

  case Token::kw___generator_type:
    if (failed(parseGeneratorType(result)))
      return failure();
    break;

  case Token::kw_lambda:
    // We parse lambda as part of primary expressions to simplify the grammar.
    // They end on an 'expression' production though, and thus should not / can
    // not / will not consume any postfix attachments - the expression suffix
    // will have handled that, so we can return here.
    return parseLambda(result);

  case Token::kw___get_mvalue_as_litref:
  case Token::kw___get_litref_as_mvalue:
  case Token::kw___get_address_as_owned_value:
  case Token::kw___get_address_as_uninit_lvalue:
  case Token::kw_origin_of:
  case Token::kw_type_of:
  case Token::kw_conforms_to:
  case Token::kw___functions_in_module:
  case Token::kw___get_current_function_name:
  case Token::kw___struct_field_ref:
    if (failed(parseMagicFunction(result)))
      return failure();
    break;

  case Token::kw___is_run_in_comptime_interpreter: {
    Token tok = consumeToken();
    result = alloc<MagicFunctionNode>(ExprNode::kIsRunInComptimeInterpreter,
                                      tok.getLoc(), tok.getSpelling(),
                                      ArrayRef<ExprNode *>{}, tok.getLoc());
    break;
  }

#ifndef MODULAR_PRODUCTION
  case Token::kw___mojo_crash:
    llvm::errs() << "__mojo_crash encountered\n";
    std::abort();
    break;
#endif // MODULAR_PRODUCTION

  default:
    emitTokenError("unexpected token in expression");
    result = nullptr;
    return failure();
  }

  // Check isPrimaryExprStart agrees with the cases above.
  assert(ParserBase::isPrimaryExprStart(startTok.getKind()) &&
         "isPrimaryExprStart out of sync with grammar above");

  // Parse postfix productions so long as they aren't the start of the next
  // statement.
  while (isTokenInCurrentStatement()) {
    auto loc = getToken().getLoc();

    // Handle "attributeref": x.y
    if (consumeIf(Token::dot)) {
      if (parseAttributeRefSuffix(result, loc))
        return failure();
      continue;
    }

    // Handle calls.
    if (consumeIf(Token::l_paren)) {
      if (parseCallSuffix(result, loc))
        return failure();
      continue;
    }

    // Handle "subscription" and "slicing" array subscripts and slicing.
    if (consumeIf(Token::l_square)) {
      if (parseSubscriptSuffix(result, loc))
        return failure();
      continue;
    }

    // Handle postfix ^.  This is a bit tricky because ^ is also an infix
    // expression.  We handle this by consuming it and backtracking if needed.
    if (getToken().is(Token::caret)) {
      auto cursor = lexer.getCursor();
      auto loc = consumeToken(Token::caret).getLoc();

      // We know this is a binary ^ if there is a primary expression after it.
      // Leading-dot inferred attribute refs (`.member`) must not count:
      // `a^.b` is postfix transfer followed by attribute access, not
      // `a ^ (.b)`.
      Token::Kind nextKind = getToken().getKind();
      if (nextKind != Token::dot && ParserBase::isPrimaryExprStart(nextKind) &&
          isTokenInCurrentStatement()) {
        cursor.restore(lexer);
        break;
      }

      result = alloc<UnaryOpNode>(ExprNode::kTransfer, loc, result);
      continue;
    }

    break;
  }

  return success();
}

/// Walk the spelling of a single t-string token and extract literal parts
/// and expression interpolation parts. For each `{...}` expression region,
/// the lexer cursor is temporarily repositioned into the source buffer to
/// parse the expression, then restored.
ParseResult ExprParser::parseTStringFromSpelling(
    const Token &tstrTok, SmallVectorImpl<TStringExprNode::Part> &parts) {
  StringRef spelling = tstrTok.getSpelling();
  const char *ptr = spelling.data();
  const char *end = ptr + spelling.size();

  // Detect and skip (raw) t-string prefix.
  bool isRaw = Lexer::skipTStringPrefix(ptr);

  auto peekNextIs = [&](char c) -> bool {
    return ptr + 1 < end && ptr[1] == c;
  };

  // Determine quote style and skip opening quote(s).
  char quoteChar = *ptr;
  bool isTriple = Lexer::isTripleQuote(ptr, end, quoteChar);
  ptr += isTriple ? 3 : 1;

  // Calculate end of content (before closing quote(s)).
  const char *contentEnd = end - (isTriple ? 3 : 1);
  assert(contentEnd >= ptr && "tstring end is before the beginning");
  const char *literalStart = ptr;

  while (ptr < contentEnd) {
    // Check for escaped braces {{ and }}.
    if ((*ptr == '{' && peekNextIs('{')) || (*ptr == '}' && peekNextIs('}'))) {
      ptr += 2;
      continue;
    }

    // Skip escaped backslash (\\) before checking for \u/\U, so that
    // \\u and \\U are not misidentified as unicode escapes.
    if (*ptr == '\\' && ptr + 1 < contentEnd && ptr[1] == '\\') {
      ptr += 2;
      continue;
    }

    // Validate \u/\U unicode escapes in literal segments (not in raw
    // t-strings).
    if (!isRaw && *ptr == '\\' && ptr + 1 < contentEnd &&
        (ptr[1] == 'u' || ptr[1] == 'U')) {
      const char *escPtr = ptr + 1; // points at 'u'/'U'; caller consumed '\\'
      auto res = Lexer::consumeUnicodeEscape(escPtr, contentEnd);
      if (auto *err =
              std::get_if<Lexer::ConsumeStringResult::ErrorAt>(&res.result)) {
        emitError(SMLoc::getFromPointer(err->errorLoc), err->errorMsg);
        return failure();
      }
      ptr = escPtr;
      continue;
    }

    // Start of interpolation expression.
    if (*ptr == '{') {
      // Emit pending literal if any.
      if (ptr > literalStart) {
        parts.push_back(TStringExprNode::LiteralPart{
            .text = StringRef(literalStart, ptr - literalStart),
            .isRaw = isRaw,
        });
      }

      const char *exprStart = ptr + 1; // one past the opening '{'
      // Find the matching '}' by skipping nested strings/t-strings.
      if (!Lexer::findTStringInterpolationEnd(ptr, contentEnd)) {
        // The lexer should have caught this, but be defensive.
        emitError(SMLoc::getFromPointer(exprStart - 1),
                  "unmatched '{' in t-string; add '}' to close the expression "
                  "or use '{{' for a literal brace");
        return failure();
      }

      const char *exprEnd = ptr - 1; // points to '}'

      // Check for empty braces.
      if (exprStart == exprEnd) {
        emitError(SMLoc::getFromPointer(exprStart - 1),
                  "t-string expression must not be empty; add an expression or "
                  "remove these braces");
        return failure();
      }

      // Save the current lexer state (which is past the t_string token).
      auto savedCursor = lexer.getCursor();

      // Make a lexer at the expression content within the source buffer.
      // Swap the ExprLexer state into our lexer so parseExpression works.
      Lexer tStringExprLexer(shared.diags, lexer.getBuffer(), exprStart);
      LexerCursor(tStringExprLexer).restore(lexer);
      auto restoreLexer = llvm::scope_exit([&] { savedCursor.restore(lexer); });

      // Parse the expression.
      ExprNode *expr = nullptr;
      if (parseExpression(expr)) {
        return failure();
      }

      // Check if user attempted to use format spec (not yet supported).
      if (getToken().is(Token::colon)) {
        emitError(getToken().getLoc(),
                  "format specifiers are not supported in t-strings; format "
                  "the value manually before interpolating");
        return failure();
      }

      // Verify we stopped at the closing '}'.
      if (!getToken().is(Token::r_brace)) {
        emitError(getToken().getLoc(),
                  "expected '}' to close t-string expression");
        return failure();
      }

      parts.push_back(TStringExprNode::InterpolationPart{
          .expr = expr,
      });

      literalStart = ptr;
      continue;
    }

    ptr++;
  }

  // Emit trailing literal if any.
  if (ptr > literalStart && literalStart < contentEnd) {
    parts.push_back(TStringExprNode::LiteralPart{
        .text = StringRef(literalStart, contentEnd - literalStart),
        .isRaw = isRaw,
    });
  }

  return success();
}

/// parenth_form ::= "(" [starred_expression] ")"
///
/// If the list contains at least one comma, it yields a tuple.
ParseResult ExprParser::parsePrefixLParen(ExprNode *&result, SMLoc lparenLoc) {
  SMLoc rparenLoc;

  ExprNode *element = nullptr;

  // Empty parens is a tuple.
  if (consumeIf(Token::r_paren, &rparenLoc)) {
    // Empty tuples are represented as ParenNode(TupleNode()) where the tuple
    // has no subexpressions.
    element = alloc<TupleNode>(lparenLoc, ArrayRef<ExprNode *>());
  } else if (parseStarredExprListAsTuple(element, Token::r_paren,
                                         /*allowAssign=*/true) ||
             parseToken(Token::r_paren,
                        "expected ')' in parenthesized expression", &rparenLoc))
    return failure();

  result = alloc<ParenNode>(lparenLoc, element, rparenLoc);
  return success();
}

/// Parse a collection literal comprehension, a sequence of for/if clauses. This
/// kicks in after the first element expression is parsed.
///
/// This is used for list, set, and dict comprehensions.
ParseResult ExprParser::parseComprehension(ExprNode *&result,
                                           ExprNode::Kind kind, SMLoc startLoc,
                                           ExprNode *expr, ExprNode *value) {
  // Parse a list of for/if clauses.
  SmallVector<ComprehensionClause> clauses;
  while (getToken().isAny(Token::kw_for, Token::kw_if)) {
    auto kwLoc = getToken().getLoc();
    ComprehensionClause::Kind kind;
    ExprNode *forPattern;
    ExprNode *expr;
    if (consumeIf(Token::kw_if)) {
      forPattern = nullptr;
      kind = ComprehensionClause::kIf;
    } else {
      consumeToken(Token::kw_for);
      if (parseTargetListExpr(forPattern, /*curIndent*/ {}) ||
          parseToken(Token::kw_in, "expected 'in' after target for 'for'"))
        return failure();
      kind = ComprehensionClause::kFor;
    }
    // Avoid 'if' exprs.
    if (parseExpression(expr, Precedence(int(Precedence::kIfElse) + 1)))
      return failure();
    clauses.push_back({kwLoc, kind, forPattern, expr});
  }

  SMLoc endLoc;

  // Parse the end token.
  switch (kind) {
  case ExprNode::kListComprehension:
    if (parseToken(Token::r_square, "expected ']' in list comprehension",
                   &endLoc))
      return failure();
    break;
  case ExprNode::kSetComprehension:
    if (parseToken(Token::r_brace, "expected '}' in set comprehension",
                   &endLoc))
      return failure();
    break;
  case ExprNode::kDictComprehension:
    if (parseToken(Token::r_brace, "expected '}' in dict comprehension",
                   &endLoc))
      return failure();
    break;
  default:
    llvm_unreachable("not a comprehension");
  }

  // Create the comprehension node.
  result = alloc<ComprehensionNode>(kind, startLoc, expr, value,
                                    copyArrayRef<ComprehensionClause>(clauses),
                                    endLoc);
  return success();
}

/// list_display ::=  "[" [starred_list | comprehension "]"
ParseResult ExprParser::parsePrefixLSquare(ExprNode *&result,
                                           SMLoc lsquareLoc) {
  SMLoc rsquareLoc;
  SmallVector<ExprNode *> exprs;
  // Handle empty list: []
  if (consumeIf(Token::r_square, &rsquareLoc)) {
    result = alloc<ListLiteralNode>(lsquareLoc, exprs, rsquareLoc);
    return success();
  }
  // Parse the items in the list.
  if (parseStarredItemList(exprs, Token::r_square, /*allowAssign=*/true))
    return failure();

  // Handle a normal list.
  if (consumeIf(Token::r_square, &rsquareLoc)) {
    result = alloc<ListLiteralNode>(lsquareLoc, copyArrayRef<ExprNode *>(exprs),
                                    rsquareLoc);
    return success();
  }

  // Otherwise we might have a comprehension. List comprehensions always start
  // with a 'for', but may be followed by an arbitrary number of "if" and "for"
  // clauses.
  if (getToken().is(Token::kw_for)) {
    // Only a single expression is allowed in a comprehension.
    if (exprs.size() != 1)
      emitError(exprs[1]->getLoc(),
                "list comprehension must have a single expression before "
                "'for'; remove extra expressions")
          << exprs[1]->getRange();

    return parseComprehension(result, ExprNode::kListComprehension, lsquareLoc,
                              exprs[0], nullptr);
  }

  emitTokenError("expected ']' in list expression");
  return failure();
}

/// dict_display       ::=  "{" [key_datum_list | dict_comprehension] "}"
/// key_datum_list     ::=  key_datum ("," key_datum)* [","]
/// key_datum          ::=  expression ":" expression | "**" or_expr
/// dict_comprehension ::=  expression ":" expression comp_for
/// set_display ::= "{" (flexible_expression_list | comprehension) "}"
///
/// This function handles parsing of dictionary literals as well as set and
/// initializer lists.
ParseResult ExprParser::parsePrefixLBrace(ExprNode *&result, SMLoc lbraceLoc) {
  SMLoc rbraceLoc;
  SmallVector<std::pair<ExprNode *, ExprNode *>> elements;

  // As a mojo extension, we support parsing set initializer lists that have
  // keyword arguments in them.  This will be treated as an initializer list
  // for a value, e.g. `{a, kwarg=42}` will be interpreted as `T(a, kwarg=42)`
  // when the inferred type is `T`.  We need to keep track of whether we're
  // parsing a dict or set/init, which we determine on the first element parsed.
  bool isDict = false;

  // Parse all the comma separated elements.
  while (elements.empty() || consumeIf(Token::comma)) {
    // Allow empty initializers and trailing comma in the initializer.
    if (getToken().is(Token::r_brace))
      break;

    ExprNode *key = nullptr, *value = nullptr;

    // Handle normal key:value and dictionary unpacking. The latter has a null
    // key in the DictLiteralNode representation.
    bool isDictEntry = false;
    auto loc = getToken().getLoc();
    if (consumeIf(Token::star_star)) { // **x is an unpack.
      // Sets and init lists don't support unpacking so this is a dict entry.
      isDictEntry = true;
      if (parseExpression(value))
        return failure();
    } else {
      if (parseExpression(key))
        return failure();

      // If we have an equal sign or colon, then we have an additional value.
      isDictEntry = consumeIf(Token::colon);
      if (isDictEntry || consumeIf(Token::equal)) {
        if (parseExpression(value))
          return failure();
      } else {
        // Otherwise we have a set, and the first expression we parsed is the
        // value.
        value = key;
        key = nullptr;
      }

      // Make sure all the elements are consistent if this is an additional
      // element.
      if (elements.empty()) {
        isDict = isDictEntry;
      } else if (isDict != isDictEntry) {
        if (isDict)
          emitError(loc, "expected 'key: value' in dictionary expression");
        else
          emitError(loc, "cannot have a 'key: value' pair in set initializer");
        // Maintain invariant by not adding this element, but keep parsing.
        continue;
      }
    }
    elements.push_back({key, value});
  }

  // Handle dict_comprehension if present
  if (getToken().is(Token::kw_for)) {
    if (elements.size() != 1)
      emitError(elements[1].second->getLoc(),
                "comprehension must have a single expression before 'for'; "
                "remove extra expressions")
          << elements[1].second->getRange();
    else if (!isDict && elements[0].first)
      emitError(elements[0].first->getLoc(),
                "cannot use keyword argument in set comprehension")
          << elements[0].first->getRange();
    auto kind =
        isDict ? ExprNode::kDictComprehension : ExprNode::kSetComprehension;

    auto key = isDict ? elements[0].first : elements[0].second;
    auto value = isDict ? elements[0].second : nullptr;
    return parseComprehension(result, kind, lbraceLoc, key, value);
  }

  // Otherwise we must be out of elements.
  if (parseToken(Token::r_brace,
                 isDict ? "expected '}' at end of dictionary"
                        : "expected '}' at end of set",
                 &rbraceLoc))
    return failure();

  auto stableElts = copyArrayRef<std::pair<ExprNode *, ExprNode *>>(elements);
  if (isDict)
    result = alloc<DictLiteralNode>(lbraceLoc, stableElts, rbraceLoc);
  else
    result = alloc<SetInitLiteralNode>(lbraceLoc, stableElts, rbraceLoc);
  return success();
}

/// attributeref ::=  primary "." identifier
ParseResult ExprParser::parseAttributeRefSuffix(ExprNode *&result,
                                                SMLoc dotLoc) {
  Token token = getToken();
  StringRef spelling = token.getSpelling();
  if (parseIdentifier("expected name in attribute reference", nullptr,
                      /*forbidStartOfLine=*/false,
                      /*allowKeyword=*/true)) {
    // If we didn't get an identifier, recover by using an empty string.
    // Reuse the spelling buffer to preserve the expected location of the
    // identifier.
    spelling = StringRef(spelling.data(), 0);
  }

  result = alloc<AttributeRefNode>(result, dotLoc, spelling,
                                   token.is(Token::escaped_identifier));
  return success();
}

/// Parses a (subscript or call) operand expression with optional keyword. The
/// given callback is used to parse the operand value expression.
FailureOr<Operand> ExprParser::parseOperand(
    function_ref<ParseResult(ExprNode *&, Precedence)> parseOperandValue) {
  ExprNode *value;
  SMLoc startLoc = getToken().getLoc();
  if (getToken().is(Token::star)) {
    if (failed(parseStarredItem(value, /*allowAssign=*/false)))
      return failure();
    return Operand(value, startLoc, ArgUnpackStyle::kStar);
  }
  if (consumeIf(Token::star_star)) {
    if (failed(parseExpression(value)))
      return failure();
    return Operand(value, startLoc, ArgUnpackStyle::kStarStar);
  }

  // Check for a keyword argument.  We need look-ahead to determine whether
  // the token after the identifier is an equal sign.
  if (getToken().isIdentifier()) {
    auto cursor = lexer.getCursor();
    StringAttr name;
    (void)parseIdentifier(name, "<<already know this is identifier>>");
    if (consumeIf(Token::equal)) {
      if (failed(parseOperandValue(value, Precedence::kExpression)))
        return failure();
      return Operand(value, startLoc, ArgUnpackStyle::kKeyword, name);
    }
    // Otherwise, we consumed the base expression, just pop it back off.
    cursor.restore(lexer);
  }

  // Parse this as an assignment_expression, allowing := operator.
  if (failed(parseOperandValue(value, Precedence::kAssignExpr)))
    return failure();
  return Operand(value, startLoc, ArgUnpackStyle::kPositional);
}

/// call ::=  primary "(" [argument_list [","] | comprehension] ")"
/// argument_list ::= argument ("," argument)*
/// argument      ::= assignment_expression
/// argument      ::= "*" expression
/// argument      ::= identifier "=" expression
/// argument      ::= "**" expression
///
/// The official Python grammar is super complicated, but the constraint is
/// just that you can't have position arguments after keyword arguments. This
/// is easier to enforce imperatively than with BNF.
ParseResult ExprParser::parseCallSuffix(ExprNode *&result, SMLoc lparenLoc) {
  SmallVector<Operand> operands;
  SMLoc rparenLoc;
  if (!consumeIf(Token::r_paren, &rparenLoc)) {
    // Expressions continue maximally because we are within ()'s.
    llvm::SaveAndRestore<std::optional<size_t>> x(stmtIndent, std::nullopt);

    // Parse an argument.
    auto parseCallOperand = [&]() -> ParseResult {
      auto parseOperandValue = [&](ExprNode *&result, Precedence minPrec) {
        return parseExpression(result, minPrec);
      };
      FailureOr<Operand> operandOr = parseOperand(parseOperandValue);
      if (failed(operandOr))
        return failure();
      operands.emplace_back(std::move(*operandOr));
      return success();
    };

    // TODO: Handle comprehension argument.
    if (parseCommaSeparatedList(parseCallOperand, Token::r_paren) ||
        parseToken(Token::r_paren, "expected ')' in call argument list",
                   &rparenLoc)) {
      return failure();
    }
  }

  if (checkOperands(operands, /*isArgument=*/true))
    return failure();

  // Otherwise we're good to go.
  result = alloc<CallNode>(result, lparenLoc, copyArrayRef<Operand>(operands),
                           rparenLoc);
  return success();
}

/// Parses a slice or an ordinary expression
ParseResult ExprParser::parseExprOrSlice(ExprNode *&result) {
  // If this has a leading expr it could be an expr only or could be the first
  // (optional) part of a slice.
  if (getToken().isNot(Token::colon)) {
    if (parseExpression(result))
      return failure();
    // If we had an expr with no trailing colon, then we are done with the
    // expr case.
    if (getToken().isNot(Token::colon, Token::equal))
      return success();
  } else {
    // If it starts with a colon, this is a slice without a lower bound.
    result = nullptr;
  }

  /// Consume either a colon or an equal sign.  If we have an equal sign,
  /// diagnose it as a typo error.
  auto consumeColonOrEqual = [&]() -> SMLoc {
    assert(getToken().isAny(Token::colon, Token::equal));
    auto loc = getToken().getLoc();
    if (getToken().is(Token::equal))
      emitTokenError("expected ':' in subscript slice, not '='")
          << FixIt::replaceToken(loc, ":");
    consumeToken();
    return loc;
  };

  // Okay we have at least one colon, so we have a slice.
  SMLoc colon1Loc = consumeColonOrEqual(), colon2Loc;
  ExprNode *secondExpr = nullptr, *thirdExpr = nullptr;

  // Parse the second expr if present.
  if (getToken().isNot(Token::colon, Token::equal, Token::comma,
                       Token::r_square))
    if (parseExpression(secondExpr))
      return failure();

  // Parse a second colon if present and stride expression.
  if (getToken().isAny(Token::colon, Token::equal)) {
    colon2Loc = consumeColonOrEqual();
    if (getToken().isNot(Token::comma, Token::r_square))
      if (parseExpression(thirdExpr))
        return failure();
  }

  result = alloc<SliceLiteralNode>(result, colon1Loc, secondExpr, colon2Loc,
                                   thirdExpr);
  return success();
}

ParseResult ExprParser::checkOperands(ArrayRef<Operand> operands,
                                      bool isArgument) {
  std::string argOrParam = isArgument ? "argument" : "parameter";
  // We keep a map of "name -> operand" so that we can emit better diagnostics.
  llvm::SmallDenseMap<StringAttr, const Operand *> kwOperandMap;
  bool hasUnpackedKw = false;
  const Operand *unpackedPosOperand = nullptr;
  for (const Operand &operand : operands) {
    SMLoc loc = operand.getLoc();
    hasUnpackedKw |= operand.unpackStyle == ArgUnpackStyle::kStarStar;

    if (isArgument && operand.unpackStyle == ArgUnpackStyle::kStar) {
      if (unpackedPosOperand) {
        auto diag =
            emitError(loc, "unpack markers (*name syntax) must not appear more "
                           "than once in a call; remove the second unpack");
        diag.attachNote(unpackedPosOperand->getLoc())
            << "previous unpacked positional argument specified here";
        return std::move(diag);
      }
      if (hasUnpackedKw) {
        return emitError(loc,
                         "positional unpack must not follow keyword unpacking; "
                         "move it before the '**' unpack");
      }
      unpackedPosOperand = &operand;
    } else if (isArgument && unpackedPosOperand && !operand.isKeyword() &&
               operand.unpackStyle != ArgUnpackStyle::kStarStar) {
      auto diag =
          emitError(loc, "positional argument must not follow an unpack; move "
                         "it before or convert to a keyword argument");
      diag.attachNote(unpackedPosOperand->getLoc())
          << "unpacked positional argument specified here";
      return std::move(diag);
    }

    if (operand.unpackStyle == ArgUnpackStyle::kPositional) {
      if (isArgument && !kwOperandMap.empty()) {
        // Parameter operands allow keywords before positional operands because
        // inferred parameters can be passed with keyword syntax before
        // positional operands. Avoid checking parameter operand ordering here.
        // It will be checked later when verifying bindings.
        return emitError(
            loc, "positional argument must not follow a keyword argument; move "
                 "it before or convert to a keyword argument");
      }
      if (hasUnpackedKw) {
        return emitError(loc, "positional ")
               << argOrParam
               << " must not follow keyword unpacking; move it before or "
                  "convert to a keyword argument";
      }
    }
    if (operand.unpackStyle == ArgUnpackStyle::kKeyword) {
      auto [it, addedNew] = kwOperandMap.try_emplace(operand.name, &operand);
      if (!addedNew) {
        auto diag = emitError(loc, "keyword ")
                    << argOrParam << " " << operand.name
                    << " was already used; remove the duplicate";
        diag.attachNote(it->getSecond()->getLoc())
            << "previously specified here";
        return std::move(diag);
      }
    }
  }
  return success();
}

/// subscription ::=  primary "[" expression_list "]"
///
/// slicing      ::=  primary "[" slice_list "]"  [TODO]
/// slice_list   ::=  slice_item ("," slice_item)* [","]
/// slice_item   ::=  expression | proper_slice
/// proper_slice ::=  [lower_bound] ":" [upper_bound] [ ":" [stride] ]
/// lower_bound  ::=  expression
/// upper_bound  ::=  expression
/// stride       ::=  expression
ParseResult ExprParser::parseSubscriptSuffix(ExprNode *&result,
                                             SMLoc lsquareLoc) {
  // Expressions continue maximally because we are within []'s.
  llvm::SaveAndRestore<std::optional<size_t>> x(stmtIndent, std::nullopt);

  // If we have an empty parameter list, we return immediately.
  SMLoc rsquareLoc;
  if (consumeIf(Token::r_square, &rsquareLoc)) {
    result = alloc<SubscriptNode>(result, lsquareLoc, ArrayRef<Operand>(),
                                  rsquareLoc);
    return success();
  }

  // Helper to parse an input or a result parameter.
  auto parseSubscriptOperand =
      [&](SmallVectorImpl<Operand> &parsed) -> ParseResult {
    auto parseOperandValue = [&](ExprNode *&result, Precedence minPrec) {
      // Precedence is ignored here on purpose; we don't allow walrus here.
      return parseExprOrSlice(result);
    };
    FailureOr<Operand> operandOr = parseOperand(parseOperandValue);
    if (failed(operandOr))
      return failure();
    parsed.emplace_back(std::move(*operandOr));
    return success();
  };

  SmallVector<Operand> operands;
  if (parseCommaSeparatedList([&]() { return parseSubscriptOperand(operands); },
                              {Token::r_square, Token::minus_greater}) ||
      getLocation(rsquareLoc))
    return failure();

  if (checkOperands(operands, /*isArgument=*/false))
    return failure();

  if (parseToken(Token::r_square, "expected ']' in call argument list"))
    return failure();
  result = alloc<SubscriptNode>(result, lsquareLoc,
                                copyArrayRef<Operand>(operands), rsquareLoc);
  return success();
}

/// function_type ::= ["async"] "def" [parameter_list] argument_list [effects]
///                   [origin_set] ["->" result_type] [constraint_clauses]
ParseResult ExprParser::parseFunctionType(ExprNode *&result) {
  SMLoc baseLoc = getToken().getLoc();
  ParsedParamList paramList;
  ParsedArgumentList fnSignature;

  // Parse the function effects from the leading keyword.
  fnSignature.effects.setAsync(consumeIf(Token::kw_async));
  // TODO(26.5): Remove support for 'fn' entirely.
  if (getToken().is(Token::kw_fn)) {
    emitError(getToken().getLoc(), "'fn' has been removed; use 'def' instead")
        << FixIt::replaceToken(getToken().getLoc(), "def");
  }
  consumeToken();

  // Parameter signature, argument list and the function effects next.
  if (paramList.parseParametersIfPresent(*this,
                                         ArgListKind::kFnTypeParamList) ||
      fnSignature.parseArgumentListAndEffects(*this,
                                              ArgListKind::kFnTypeArgList))
    return failure();

  // Parse the capture origin set if present.
  ExprNode *originExpr = nullptr;
  if (consumeIf(Token::l_square)) {
    if (ParserBase::parseExpression(originExpr, stmtIndent) ||
        parseToken(Token::r_square, "expected ']' in function origins"))
      return failure();
  }

  // Parse the result type.
  SMLoc endLoc = getToken().getEndLoc();

  // Parse the result type if present.
  fnSignature.parseResultIfPresent(*this, stmtIndent);

  // Parse trailing body constraints if present (only supported for thin
  // functions for now).
  if (paramList.parseTrailingConstraintsIfPresent(*this))
    return failure();

  if (!paramList.bodyConstraints.empty() && !fnSignature.isThin) {
    emitError(getToken().getLoc(),
              "trailing constraints are only supported for thin functions");
    return failure();
  }

  result = alloc<FunctionTypeNode>(
      baseLoc, copyArrayRef<ParsedArgument>(paramList.params),
      copyArrayRef<ParsedArgument>(fnSignature.parsedArgs),
      copyArrayRef<ParsedArgument>(fnSignature.resultArg)[0],
      fnSignature.effects, fnSignature.isThin,
      fnSignature.isExperimentalParamTrait, fnSignature.thrownTypeExpr,
      originExpr, copyArrayRef<ParsedConstraint>(paramList.bodyConstraints),
      endLoc);
  return success();
}

/// generator_type ::= "__generator_type" [parameter_list] type_expression
///
/// Produces a `!lit.generator<<params> body>` type. The parameter list is
/// optional; when present it uses the same grammar as function/struct
/// parameter lists.
ParseResult ExprParser::parseGeneratorType(ExprNode *&result) {
  SMLoc baseLoc = consumeToken(Token::kw___generator_type).getLoc();

  ParsedParamList paramList;
  if (paramList.parseParametersIfPresent(*this, ArgListKind::kFnTypeParamList))
    return failure();

  ExprNode *bodyTypeExpr = nullptr;
  if (ParserBase::parseExpression(bodyTypeExpr, stmtIndent))
    return failure();

  result = alloc<GeneratorTypeNode>(
      baseLoc, copyArrayRef<ParsedArgument>(paramList.params), bodyTypeExpr,
      bodyTypeExpr->getRangeEnd());
  return success();
}

/// lambda_expr ::= "lambda" [parameter_list] [argument_list] [effects]
///                 [capture_list] ["->" result_type] ":" expression
///
/// The signature is parsed in the same order as a nested-`def` closure:
/// parameters, then the argument list with its trailing effects, then the
/// capture list, then the result type. This produces a LambdaNode; the closure
/// itself is constructed at emit time (LambdaNode::emitIR).
ParseResult ExprParser::parseLambda(ExprNode *&result) {
  SMLoc lambdaLoc = consumeToken(Token::kw_lambda).getLoc();

  ParsedParamList paramList;
  ParsedArgumentList fnSignature;
  ParsedCaptureList captureList;

  // Parameter list `[...]`, if present.
  if (paramList.parseParametersIfPresent(*this, ArgListKind::kParamList))
    return failure();

  // Argument list `(...)` plus trailing effects. Python-style bare
  // (unparenthesized) arguments are rejected with guidance below: lambda
  // arguments must be typed, and a type annotation is only unambiguous inside
  // parentheses (otherwise the ":" introducing the body would be ambiguous).
  // A lambda may also take no arguments at all (e.g. `lambda: expr`), in which
  // case there is nothing to parse here.
  if (getToken().is(Token::l_paren)) {
    if (fnSignature.parseArgumentListAndEffects(*this, ArgListKind::kArgList))
      return failure();
  } else if (getToken().is(Token::identifier)) {
    // Python-style unparenthesized args (`lambda x, y: ...`) are not supported.
    emitError(getToken().getLoc(),
              "unparenthesized lambda arguments are not supported; write them "
              "in parentheses, with types, e.g. "
              "`lambda (x: Int, y: Int) -> Int: x + y`");
    return failure();
  }

  // Capture list `{...}`, if present (no-op otherwise).
  if (captureList.parseCaptureList(*this))
    return failure();

  // Result type `-> T`, if present (no-op otherwise).
  fnSignature.parseResultIfPresent(*this, stmtIndent);

  // Body: a single expression after `:`.
  SMLoc endLoc = getToken().getEndLoc();
  ExprNode *bodyExpr = nullptr;
  if (parseToken(Token::colon, "expected ':' in lambda expression") ||
      ParserBase::parseExpression(bodyExpr, stmtIndent))
    return failure();

  result = alloc<LambdaNode>(
      lambdaLoc, copyArrayRef<ParsedArgument>(paramList.params),
      copyArrayRef<ParsedArgument>(fnSignature.parsedArgs),
      copyArrayRef<ParsedArgument>(fnSignature.resultArg)[0],
      fnSignature.effects, fnSignature.thrownTypeExpr,
      copyArrayRef<std::tuple<StringRef, CaptureConvention, SMLoc>>(
          captureList.parsedCaptures),
      captureList.hasExplicitCaptureList, captureList.captureAllByConvention,
      bodyExpr, endLoc);
  return success();
}

ParseResult ExprParser::parseMagicFunction(ExprNode *&result) {
  ExprNode::Kind nodeKind;
  switch (getToken().getKind()) {
  default:
    llvm_unreachable("bad token");
  case Token::kw___get_address_as_uninit_lvalue:
    nodeKind = ExprNode::kGetAddressAsUninitLValue;
    break;
  case Token::kw___get_mvalue_as_litref:
    nodeKind = ExprNode::kGetMValueAsLitRef;
    break;
  case Token::kw___get_litref_as_mvalue:
    nodeKind = ExprNode::kGetLitRefAsMValue;
    break;
  case Token::kw___get_address_as_owned_value:
    nodeKind = ExprNode::kGetAddressAsOwned;
    break;
  case Token::kw_conforms_to:
    nodeKind = ExprNode::kConformsTo;
    break;
  case Token::kw_origin_of:
    nodeKind = ExprNode::kOriginOf;
    break;
  case Token::kw_type_of:
    nodeKind = ExprNode::kTypeOf;
    break;
  case Token::kw___functions_in_module:
    nodeKind = ExprNode::kFunctionsInModule;
    break;
  case Token::kw___get_current_function_name:
    nodeKind = ExprNode::kGetCurrentFunctionName;
    break;
  case Token::kw___struct_field_ref:
    nodeKind = ExprNode::kStructFieldRef;
    break;
  }

  Token tok = consumeToken();
  SMLoc baseLoc = tok.getLoc();
  StringRef spelling = tok.getSpelling();

  SmallVector<ExprNode *> subExprs;
  SMLoc rpLoc;
  // All "magic" functions take an argument.
  if (parseToken(Token::l_paren, "expected '('"))
    return failure();
  if (!consumeIf(Token::r_paren, &rpLoc)) {
    if (parseCommaSeparatedList(
            [&] { return parseExpression(subExprs.emplace_back()); },
            Token::r_paren) ||
        parseToken(Token::r_paren, "expected ')'", &rpLoc))
      return failure();
  }

  result = alloc<MagicFunctionNode>(nodeKind, baseLoc, spelling,
                                    copyArrayRef<ExprNode *>(subExprs), rpLoc);
  return success();
}

//===----------------------------------------------------------------------===//
// ExprParser implementation
//===----------------------------------------------------------------------===//

/// Parse an expression_list production, returning a single expression or a
/// tuple expression if there are commas.
ParseResult ParserBase::parseExpressionList(ExprNode *&result,
                                            std::optional<size_t> stmtIndent,
                                            ArrayRef<Token::Kind> terminators) {
  ExprParser parser(shared, getLexer(), stmtIndent);
  SmallVector<ExprNode *> exprs;
  auto parseItem = [&]() -> ParseResult {
    return parser.parseExpression(exprs.emplace_back(nullptr),
                                  Precedence::kExpression);
  };

  SMLoc firstCommaLoc;
  if (parser.parseCommaSeparatedList(parseItem, terminators, stmtIndent,
                                     &firstCommaLoc))
    return failure();

  // If we parsed multiple items or have a comma, then this is actually a tuple.
  // If there was a tuple inside the parens, form it.
  if (exprs.size() != 1 || firstCommaLoc.isValid())
    result = parser.alloc<TupleNode>(firstCommaLoc,
                                     parser.copyArrayRef<ExprNode *>(exprs));
  else
    result = exprs[0];
  return success();
}

ParseResult ParserBase::parseOptionalIdentifier(StringAttr &result,
                                                SMLoc *loc) {
  LexerCursor cursor(lexer);
  result = StringAttr::get(getContext(), getToken().getSpelling());
  if (consumeIf(Token::identifier, loc)) {
    if (loc)
      *loc = getToken().getLoc();
    return success();
  }
  cursor.restore(lexer);
  return failure();
}

ParseResult ParserBase::parseOptionalIdentifier(StringAttr &result,
                                                Token::Kind delimiter,
                                                SMLoc *loc) {
  LexerCursor cursor(lexer);
  if (succeeded(parseOptionalIdentifier(result, loc))) {
    if (getToken().is(delimiter))
      return success();
  }
  cursor.restore(lexer);
  return failure();
}

/// Expression parsing.  Each of these take a `stmtIndent` specifier that
/// indicates the indentation level of the start of the statement that
/// contains this expression if the expression can exist at the end of the
/// line.  This allows the expression parser to know when to keep parsing the
/// expression on the next line - when it is more indented than the start of
/// the current statement.  This can be passed in as None when there is a
/// trailing punctuator that naturally terminates the expression.
ParseResult ParserBase::parseExpression(ExprNode *&result,
                                        std::optional<size_t> stmtIndent,
                                        Precedence minPrec) {
  return ExprParser(shared, getLexer(), stmtIndent)
      .parseExpression(result, minPrec);
}

ParseResult ParserBase::parseStarredItem(ExprNode *&result) {
  return ExprParser(shared, getLexer(), std::nullopt)
      .parseStarredItem(result, /*allowAssign*/ true);
}

/// This parses a superset of the "target_list" production, which is used as the
/// pattern in a "for" loop and comprehensions. This notably does not include
/// "in" expressions.
ParseResult ParserBase::parseTargetListExpr(ExprNode *&result,
                                            std::optional<size_t> stmtIndent) {
  return ExprParser(shared, getLexer(), stmtIndent)
      .parseStarredExprListAsTuple(result, /*terminators=*/{},
                                   /*allowAssign=*/false, /*allowVarRef=*/true);
}

ParseResult ParserBase::parseVarInitExpression(ExprNode *&result,
                                               size_t stmtIndent) {
  return ExprParser(shared, getLexer(), stmtIndent)
      .parseStarredExprListAsTuple(result, /*terminators=*/{},
                                   /*allowAssign=*/true);
}

/// If the specified token is an '=' or '+=' sort of token, return the
/// expression kind, otherwise return null.
static std::optional<ExprNode::Kind> getAssignmentKind(Token::Kind tokenKind) {
  switch (tokenKind) {
  default:
    return std::nullopt;
  case Token::equal:
    return ExprNode::kAssign;
  case Token::plus_equal:
    return ExprNode::kIAdd;
  case Token::minus_equal:
    return ExprNode::kISub;
  case Token::star_equal:
    return ExprNode::kIMul;
  case Token::at_equal:
    return ExprNode::kIMatMul;
  case Token::slash_equal:
    return ExprNode::kITrueDiv;
  case Token::percent_equal:
    return ExprNode::kIMod;
  case Token::amp_equal:
    return ExprNode::kIAnd;
  case Token::pipe_equal:
    return ExprNode::kIOr;
  case Token::caret_equal:
    return ExprNode::kIXor;
  case Token::less_less_equal:
    return ExprNode::kILShift;
  case Token::right_right_equal:
    return ExprNode::kIRShift;
  case Token::star_star_equal:
    return ExprNode::kIPow;
  case Token::slash_slash_equal:
    return ExprNode::kIFloorDiv;
  }
}

/// Parse a simple_stmt production containing an expression, including
/// expression_stmt and {augmented_|annotated_|}assignment_stmt.
///
/// expression_stmt ::= starred_expression
/// assignment_stmt ::=
///              (expression_list "=")+ (starred_expression | yield_expression)
/// augmented_assignment_stmt ::=
///                        expression "+=" (expression_list | yield_expression)
///
/// NOTE: we do not handle this as part of binary operator parsing because the
/// grammar is so weird and different with yield expressions, expression_list,
/// and starred expression.
ParseResult ParserBase::parseSimpleStmtExprs(ExprNode *&result,
                                             size_t stmtIndent) {
  ExprParser p(shared, getLexer(), stmtIndent);

  // We have three very different grammar productions that all start with an
  // expression, starred_expression, or assignment_expression plus the target
  // stuff in various mixes.  This is all the Python grammar trying to enforce
  // semantic considerations in the grammar, which is unpleasant.  Implement
  // this by parsing the most general thing and sorting out what is valid later.
  ExprNode *expr = nullptr;
  // TODO: Handle yield_expression.
  if (p.parseStarredExprListAsTuple(
          expr, /*terminators=*/{Token::colon, Token::equal},
          /*allowAssign=*/true, /*allowVarRef=*/true))
    return failure();

  // Check for type pattern. In Python this is the:
  //   annotated_assignment_stmt ::= augtarget ":" expression
  // Part of the grammar.
  SMLoc colonLoc;
  if (p.consumeIf(Token::colon, &colonLoc)) {
    ExprNode *type = nullptr;
    if (p.parseExpression(type))
      return failure();
    expr = p.alloc<BinOpNode>(ExprNode::kTypePattern, expr, colonLoc, type);
  }

  // If that was it, just return the expression.
  std::optional<ExprNode::Kind> assignKind =
      getAssignmentKind(p.getToken().getKind());
  if (!p.isTokenInCurrentStatement() || !assignKind.has_value()) {
    result = expr;
    return success();
  }
  SMLoc assignLoc = p.consumeToken().getLoc();

  // Make sure the next token is in the current statement, not something like
  //   x =
  // y.foo()
  if (!p.isTokenInCurrentStatement()) {
    emitError(assignLoc,
              "'=' must be followed by an expression on the same line");
    return failure();
  }

  // If we have an = or += operator, parse the rest of the statement pieces;
  // assignments are right associative, so we just recurse to handle this.
  ExprNode *rhsExpr = nullptr;
  if (parseSimpleStmtExprs(rhsExpr, stmtIndent))
    return failure();

  result = p.alloc<BinOpNode>(assignKind.value(), expr, assignLoc, rhsExpr);
  return success();
}
