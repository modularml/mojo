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
// This file provides the base class for Mojo parsers that is common between
// expression and statement parsing.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_MOJOPARSER_PARSERBASE_H
#define KGEN_MOJOPARSER_PARSERBASE_H

#include "Mojo/MojoParser/Lexer.h"
#include "Mojo/MojoParser/SharedState.h"
#include "mlir/IR/Diagnostics.h"

namespace M::KGEN::LIT {
class ExprNode;
class ASTDecl;

//===----------------------------------------------------------------------===//
// ParserBase
//===----------------------------------------------------------------------===//

/// This class implements logic that is common to many parts of the parser, but
/// which is independent of the concrete grammar.
class ParserBase : public SharedStateUser {
public:
  ParserBase(SharedState &shared, Lexer &lexer)
      : SharedStateUser(shared), lexer(lexer) {}
  Lexer &getLexer() { return lexer; }

  /// Return the current token the parser is inspecting.
  const Token &getToken() const { return lexer.getToken(); }
  StringRef getTokenSpelling() const { return getToken().getSpelling(); }

  //===--------------------------------------------------------------------===//
  // Error Handling
  //===--------------------------------------------------------------------===//

  using SharedStateUser::emitError;

  /// Emit an error at a specific lexer location.
  MojoInflightDiag emitError(llvm::SMLoc loc, const Twine &message = {});

  /// Emit an error at the current token.
  MojoInflightDiag emitTokenError(const Twine &message = {}) {
    return emitError(getToken().getLoc(), message);
  }

  //===--------------------------------------------------------------------===//
  // Location Handling
  //===--------------------------------------------------------------------===//

  /// Capture the location of the current token in a convenient way that can be
  /// used in parsing pipelines.
  ParseResult getLocation(SMLoc &result) {
    result = getToken().getLoc();
    return success();
  }

  /// This returns the current lexer cursor and succeeds, so it can be used in a
  /// parser pipeline.
  ParseResult getCursor(LexerCursor &cursor) const {
    cursor = lexer.getCursor();
    return success();
  }

  /// Return the location of the current token, or a location at the end of the
  /// previous line if it is on a new line.  This is used when there was a
  /// problem with the previous token to make sure we report the error on that
  /// line.
  SMLoc getTokenLocOrEndOfPreviousLineIfOnNewLine() const {
    SMLoc loc = getToken().getLoc();
    if (getToken().isStartOfLine())
      return lexer.findEndOfPreviousLine(loc);
    return loc;
  }

  //===--------------------------------------------------------------------===//
  // Token Parsing
  //===--------------------------------------------------------------------===//

  /// If the current token has the specified kind, consume it and return true.
  /// If not, return false.  If 'tokLoc' is non-null, it is filled in with the
  /// location of the consumed token (on success).
  bool consumeIf(Token::Kind kind, llvm::SMLoc *tokLoc = nullptr) {
    if (getToken().isNot(kind))
      return false;
    if (tokLoc)
      *tokLoc = getToken().getLoc();
    consumeToken(kind);
    return true;
  }

  /// Advance the current lexer onto the next token.
  ///
  /// This returns the consumed token.
  Token consumeToken() {
    Token consumedToken = getToken();
    assert(consumedToken.isNot(Token::eof) && "shouldn't advance past EOF");
    lexer.lexToken();
    return consumedToken;
  }

  /// Advance the current lexer onto the next token, asserting what the expected
  /// current token is.  This is preferred to the above method because it leads
  /// to more self-documenting code with better checking.
  ///
  /// This returns the consumed token.
  Token consumeToken(Token::Kind kind) {
    Token consumedToken = getToken();
    assert(consumedToken.is(kind) && "consumed an unexpected token");
    consumeToken();
    return consumedToken;
  }

  Token consumeIdentifier() {
    Token consumedToken = getToken();
    assert(consumedToken.isIdentifier() && "consumed an unexpected token");
    consumeToken();
    return consumedToken;
  }

  /// If the current token is an identifier with the current spelling, consume
  /// it and return true, otherwise return false.
  bool consumeIfSoftIdentifier(StringRef identifier) {
    Token tok = getToken();
    if (!tok.isIdentifier() || tok.getSpelling() != identifier)
      return false;
    consumeIdentifier();
    return true;
  }

  /// Consume the specified token if present and return success.  On failure,
  /// output a diagnostic and return failure. If `loc` is set, it is populated
  /// with the source location of the token.
  ParseResult parseToken(Token::Kind expectedToken, const Twine &message,
                         SMLoc *loc = nullptr);

  /// Consume an identifier token. If `loc` is set, it is populated with the
  /// source location of the token. If `allowKeyword` is true, keywords are
  /// allowed, not just identifiers.
  ParseResult parseIdentifier(const Twine &message, SMLoc *loc = nullptr,
                              bool forbidStartOfLine = false,
                              bool allowKeyword = false);

  /// Consume an identifier token, binding its name into the specified result
  /// string attribute. If `loc` is set, it is populated with the source
  /// location of the token. If `allowKeyword` is true, keywords are
  /// allowed, not just identifiers.
  ParseResult parseIdentifier(StringAttr &result, const Twine &message,
                              SMLoc *loc = nullptr,
                              bool forbidStartOfLine = false,
                              bool allowKeyword = false);

  /// Consume an identifier token regardless of the delimiter. The location
  /// is set to the delimiter (the delimiter is not consumed)
  ParseResult parseOptionalIdentifier(StringAttr &result, SMLoc *loc = nullptr);
  /// Consume an identifier token if delimiter is the next token and the parsed
  /// identifier is not a keyword. The location is set to the delimiter (the
  /// delimiter is not consumed)
  ParseResult parseOptionalIdentifier(StringAttr &result, Token::Kind delimiter,
                                      SMLoc *loc = nullptr);

  /// Parse a list of elements continued with commas.  If a set of terminators
  /// are specified, then the list ends when one is encountered (but it is not
  /// consumed).  If no terminators are specified, the list ends at end of the
  /// current statement.
  ///
  /// firstCommaLoc (if non-null) is set to the location of the first comma that
  /// is parsed, even if it is a trailing comma.
  ///
  /// separated_list ::= (element (',' element)* [','] TERMINATOR
  ///
  ParseResult
  parseCommaSeparatedList(const function_ref<ParseResult()> &parseElement,
                          ArrayRef<Token::Kind> terminators,
                          std::optional<size_t> stmtIndent = std::nullopt,
                          SMLoc *firstCommaLoc = nullptr);

  /// Return true if the current token is part of the current statement, false
  /// if it is the start of a new one.  When stmtIndent is null, the token is
  /// always considered part of the current statement (returns true).
  ///
  /// When allowSameIndent=true, a matching indent level is considered part of
  /// this statement.  An 'else' is part of an 'if' when it is at the same
  /// indent level as the 'if', it doesn't need to be indented extra.
  bool isTokenInCurrentStatement(std::optional<size_t> stmtIndent,
                                 bool allowSameIndent = false) const {
    // If the current token is on the same line as the last or if we should
    // always eat tokens (e.g. because within parens), then keep going.
    auto tokIndent = getToken().getIndentation();
    if (!tokIndent.has_value() || !stmtIndent.has_value())
      return true;

    // If this token is on its own line is indented less or the same as the
    // current statement, then it is the start of a new statement.
    if (allowSameIndent)
      return tokIndent >= stmtIndent;
    return tokIndent > stmtIndent;
  }

  /// Skip tokens until we get to a token at start of line that has indentation
  /// that is equal or less than the specified indentation.  This is used for
  /// multiphase parsing.
  ///
  /// When stopOnSemicolon is true this will stop at the first semicolon seen.
  /// This should only be used for statements that can share a line with other
  /// statements with ; separation.
  ///
  /// customStopPredicate (if non-null) is called on each token that isn't
  /// surrounded by brackets, allowing the scan to stop early by returning true.
  void
  skipUntilIndentation(size_t minIndent, bool stopOnSemicolon = false,
                       llvm::unique_function<bool()> customStopPredicate = {});

  ParseResult rejectTokenAtStartOfLine(const Twine &what) const;
  ParseResult rejectTokenAtStartOfLine(Token tok, const Twine &what) const;

  /// Skip over tokens that make up a function signature starting at the
  /// current token and consume everything up to *and including* the ':' that
  /// terminates the signature.  After this returns, the lexer will be
  /// positioned on the first token of the function body (or at EOF if the
  /// file ended unexpectedly).

  //===--------------------------------------------------------------------===//
  // Integration with parsers for subsets of the grammar.
  //===--------------------------------------------------------------------===//

  /// Parse the follow-on doc string for the given decl if it is present.
  void parseDocString(StringRef &docString);
  void parseDocString(ASTDecl &decl);

  /// Parse and return a set of decorators for the specified declaration or
  /// statement at the specified indentation level.
  SmallVector<std::pair<ExprNode *, LexerCursor>>
  parseDecorators(ASTDecl &decl);
  SmallVector<std::pair<ExprNode *, LexerCursor>>
  parseDecorators(ssize_t indentation);

  // See
  // https://docs.python.org/3/reference/expressions.html#operator-precedence
  enum class Precedence {
    kInvalid, // No precedence

    kUnpack,     // prefix: * or **
    kAssignExpr, // infix: := (walrus)
    kAsPat,      // infix: as (match patterns; looser than if so `with x as y`
                 // still parses the `as` as a statement suffix)
    kIfElse,     // infix: if - else
    kBoolOr,     // infix: or
    kBoolAnd,    // infix: and
    kBoolNot,    // prefix: not
    kComparison, // infix: in, not in, is, is not, <, <=, >, >=, !=, ==
    kVarRefPat,  // prefix: var, ref
    kOr,         // infix: |
    kXor,        // infix: ^
    kAnd,        // infix: &
    kShift,      // infix: <<, >>
    kSum,        // infix: +, -
    kTerm,       // infix: *, @, /, //, %
    kFactor,     // prefix: +, -, ~
    kPower,      // infix: **
    kAwait,      // prefix: await
    kPrimary,    // prefix: foo, "123", 123, 1.23, True, False, foo(1),
                 //         foo.bar, foo[bar], lambda

    kExpression = kIfElse, // "expression" precedence is if/else.
    kHighest = kPrimary
  };

  /// Return true if `tokKind` can start a primary expression (i.e. the set of
  /// tokens `parsePrimaryExpr` accepts as its leading token).  This lets
  /// callers decide whether the upcoming token begins an expression without
  /// committing to actually parsing one.
  static bool isPrimaryExprStart(Token::Kind tokKind);

  /// Expression parsing.  Each of these take a `stmtIndent` specifier that
  /// indicates the indentation level of the start of the statement that
  /// contains this expression if the expression can exist at the end of the
  /// line.  This allows the expression parser to know when to keep parsing the
  /// expression on the next line - when it is more indented than the start of
  /// the current statement.  This can be passed in as None when there is a
  /// trailing punctuator that naturally terminates the expression.
  ParseResult parseExpression(ExprNode *&result,
                              std::optional<size_t> stmtIndent = std::nullopt,
                              Precedence minPrec = Precedence::kExpression);
  ParseResult parseVarInitExpression(ExprNode *&result, size_t stmtIndent);
  ParseResult parseTargetListExpr(ExprNode *&result,
                                  std::optional<size_t> stmtIndent);

  /// Parse an expression_list production, returning a single expression or a
  /// tuple expression if there are commas.
  ParseResult parseExpressionList(ExprNode *&result,
                                  std::optional<size_t> stmtIndent,
                                  ArrayRef<Token::Kind> terminators = {});

  ParseResult parseStarredItem(ExprNode *&result);

  /// Parse a simple_stmt production containing an expression, including
  /// expression_stmt and {augmented_|annotated_|}assignment_stmt.
  ParseResult parseSimpleStmtExprs(ExprNode *&result, size_t stmtIndent);

  /// Parse a 'suite' production into the declaration specified by `decl`.
  ParseResult parseSuite(ASTDecl &decl);

  /// Parse a 'ref [exprlist]' production into expr, with the expression set to
  /// the exprlist if specified, otherwise set to null if absent.  This returns
  /// failure on a parse error.
  ParseResult parseRefSpecifier(ExprNode *&expr);

public:
  Lexer &lexer;

  ParserBase(const ParserBase &) = delete;
  void operator=(const ParserBase &) = delete;
};

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_PARSERBASE_H
