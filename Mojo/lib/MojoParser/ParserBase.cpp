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
// This implements the base class for Mojo file parsers, logic that is shared
// between expression and statement parsing in particular.
//
//===----------------------------------------------------------------------===//

#include "ParserBase.h"

using namespace M::KGEN::LIT;
using namespace M;

MojoInflightDiag ParserBase::emitError(SMLoc loc, const Twine &message) {
  auto diag = shared.emitError(loc, message);

  // If we hit a parse error in response to a lexer error, then the lexer
  // already reported the error.
  if (getToken().is(Token::error))
    diag.abandon();
  return diag;
}

ParseResult ParserBase::rejectTokenAtStartOfLine(const Twine &what) const {
  return rejectTokenAtStartOfLine(getToken(), what);
}

ParseResult ParserBase::rejectTokenAtStartOfLine(Token tok,
                                                 const Twine &what) const {
  if (!tok.isStartOfLine())
    return success();
  emitError(tok.getLoc(), what + " may not appear at the start of the line");
  return failure();
}

/// Consume the specified token if present and return success.  On failure,
/// output a diagnostic and return failure.
ParseResult ParserBase::parseToken(Token::Kind expectedToken,
                                   const Twine &message, SMLoc *loc) {
  if (loc)
    *loc = getToken().getLoc();
  if (consumeIf(expectedToken))
    return success();

  // If the current token is on a new line, report the error on the end of the
  // previous line, this is probably where the punctuation was omitted.
  auto diagLoc = getTokenLocOrEndOfPreviousLineIfOnNewLine();

  // Report the error.
  auto diag = emitError(diagLoc, message);

  // Customize the error if an identifier was expected but a keyword was found.
  if (expectedToken == Token::identifier && getToken().isKeyword())
    diag.attachNote(diagLoc) << "escape keyword '" << getToken().getSpelling()
                             << "' with backticks to use it as an identifier";

  return failure();
}

/// Consume an identifier token if present, or emit an error if not. If `loc` is
/// set, it is populated with the source location of the token.  If
/// `allowKeyword` is true, keywords are allowed, not just identifiers.
ParseResult ParserBase::parseIdentifier(const Twine &message, SMLoc *loc,
                                        bool forbidStartOfLine,
                                        bool allowKeyword) {
  if (loc)
    *loc = getToken().getLoc();

  // Enforce the start-of-line restriction only against a token that would
  // otherwise be accepted as a name, so that e.g. a dangling comma at the end
  // of an import list still reports the caller's more specific "expected ..."
  // diagnostic. The condition is ordered so the common (unrestricted) path
  // short-circuits before doing any extra work: parseIdentifier is hot.
  if (forbidStartOfLine &&
      (getToken().isIdentifier() || (allowKeyword && getToken().isKeyword())) &&
      rejectTokenAtStartOfLine("identifier")) {
    return failure();
  }

  if (consumeIf(Token::escaped_identifier))
    return success();

  // If the client allows a keyword and we have one, consume it.
  if (allowKeyword && getToken().isKeyword()) {
    consumeToken();
    return success();
  }
  // Otherwise we require an identifier, let parseToken handle the error.
  return parseToken(Token::identifier, message);
}

/// Consume an identifier token, binding its name into the specified result
/// string attribute. If `loc` is set, it is populated with the source location
/// of the token.  If `allowKeyword` is true, keywords are allowed, not just
/// identifiers.
ParseResult ParserBase::parseIdentifier(StringAttr &result,
                                        const Twine &message, SMLoc *loc,
                                        bool forbidStartOfLine,
                                        bool allowKeyword) {
  result = StringAttr::get(getContext(), getToken().getSpelling());
  return parseIdentifier(message, loc, forbidStartOfLine, allowKeyword);
}

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
ParseResult ParserBase::parseCommaSeparatedList(
    const function_ref<ParseResult()> &parseElement,
    ArrayRef<Token::Kind> terminators, std::optional<size_t> stmtIndent,
    SMLoc *firstCommaLoc) {
  if (firstCommaLoc)
    *firstCommaLoc = SMLoc();

  while (true) {
    if (parseElement())
      return failure();

    if (!consumeIf(Token::comma, firstCommaLoc))
      break;
    // Get the location of the first comma, not subsequent ones.
    firstCommaLoc = nullptr;

    // Mojo/Python supports trailing commas in lists, e.g. for tuples without
    // parens.
    if (!terminators.empty() && getToken().isAny(terminators))
      break;
    // Treat this as a terminating comma when the next token is at the same or
    // higher indentation, e.g.:
    //    _ = 1,          # terminating comma.
    //    def thing(): ... # not part of a tuple.
    if (!isTokenInCurrentStatement(stmtIndent))
      break;
  }
  return success();
}

/// Skip tokens until we get to a token at start of line that has indentation
/// that is equal or less than the specified indentation.  This is used for
/// multiphase parsing.
///
/// When stopOnSemicolon is true this will stop at the first semicolon seen.
/// This should only be used for statements that can share a line with other
/// statements with ; separation.
void ParserBase::skipUntilIndentation(
    size_t minIndent, bool stopOnSemicolon,
    llvm::unique_function<bool()> customStopPredicate) {
  // This keeps track of open brackets we are inside of.
  SmallVector<Token> openBrackets;

  auto handleCloseBracket = [&](Token::Kind leftBracket) {
    // If we see the correct closing bracket for the structure we're in, then
    // just pop out of that context and keep going.
    if (!openBrackets.empty() && openBrackets.back().getKind() == leftBracket) {
      openBrackets.pop_back();
      return;
    }

    // Otherwise, we have a parse error: don't diagnose it though, because the
    // non-skipping parse will.  We don't really know how best to recover so we
    // just nuke our scope which will cause us to stop skipping at the
    // indentation level requested.
    openBrackets.clear();
  };

  // We scan until we find the specified indentation at the same expression
  // level as the current token.
  while (getToken().isNot(Token::eof)) {
    // If we are outside a bracketed expression, check indentation and stop
    // predicate.
    if (openBrackets.empty()) {
      if (auto indent = getToken().getIndentation())
        if (*indent <= minIndent)
          return;
      if (customStopPredicate && customStopPredicate())
        return;
    }

    // Check to see if this is a bracket that needs special handling.
    switch (getToken().getKind()) {
    default:
      break;
    case Token::l_paren:
    case Token::l_square:
    case Token::l_brace:
      // Remember that we're nested.
      openBrackets.push_back(getToken());
      break;

      // Handle closing brackets.
    case Token::r_paren:
      handleCloseBracket(Token::l_paren);
      break;
    case Token::r_square:
      handleCloseBracket(Token::l_square);
      break;
    case Token::r_brace:
      handleCloseBracket(Token::l_brace);
      break;

      // Stop on semicolons when outside a bracket expression if requested.
    case Token::semi:
      if (stopOnSemicolon && openBrackets.empty())
        return;
      break;
    }

    // Otherwise, keep eating.
    consumeToken();
  }
}

/// Parse a ref '[exprlist]' production into expr, with the expression set to
/// the exprlist if specified, otherwise set to null if absent.  This returns
/// failure on a parse error.
ParseResult ParserBase::parseRefSpecifier(ExprNode *&expr) {
  expr = nullptr;

  // Parse the [a, b, c] specification if present.
  if (consumeIf(Token::l_square)) {
    if (parseExpressionList(expr, /*stmtIndent=*/{}, Token::r_square) ||
        parseToken(Token::r_square, "expected ']' in ref specifier"))
      return failure();
  }

  return success();
}
