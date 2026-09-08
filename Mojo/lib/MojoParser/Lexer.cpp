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
// Defines the a Lexer and Token interface for .mojo files.
//
//===----------------------------------------------------------------------===//

#include "Mojo/MojoParser/Lexer.h"
#include "Mojo/MojoParser/ASTType.h"
#include "Mojo/MojoParser/SharedState.h"
#include "Support/IPRational.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace M::KGEN::LIT;

using llvm::SMLoc;
using llvm::SMRange;
using llvm::SourceMgr;

// These C macros are often inefficient due to attempt to support unicode, use
// the llvm::isAlpha methods instead.
#define isalpha(x) DO_NOT_USE_SLOW_CTYPE_FUNCTIONS
#define isdigit(x) DO_NOT_USE_SLOW_CTYPE_FUNCTIONS

//===----------------------------------------------------------------------===//
// Token
//===----------------------------------------------------------------------===//

SMLoc Token::getLoc() const { return SMLoc::getFromPointer(spelling.data()); }

SMLoc Token::getEndLoc() const {
  return SMLoc::getFromPointer(spelling.data() + spelling.size());
}

SMRange Token::getLocRange() const { return SMRange(getLoc(), getEndLoc()); }

/// Return true if this is one of the keyword token kinds (e.g. kw_pass).
bool Token::isKeyword() const {
  switch (kind) {
  default:
    return false;
#define TOK_KEYWORD(SPELLING)                                                  \
  case kw_##SPELLING:                                                          \
    return true;
#include "Mojo/MojoParser/TokenKinds.def"
  }
}

bool Token::isStatementKeyword() const {
  switch (kind) {
  default:
    return false;
  // Control flow statements
  case kw_elif:
  case kw_else:
  case kw_while:
  case kw_pass:
  case kw_break:
  case kw_continue:
  case kw_return:
  case kw_raise:
  case kw_assert:
  case kw_try:
  case kw_except:
  case kw_finally:
  case kw_with:
  case kw_del:
  case kw_yield:
  case kw_match:
  case kw_case:
  case kw___match:
  case kw_async:
  case kw_await:
  case kw_global:
  case kw_nonlocal:
    return true;
  }
}

bool Token::isDeclKeyword() const {
  switch (kind) {
  default:
    return false;
  // Declaration keywords
  case kw_class:
  case kw_def:
  case kw_fn:
  case kw_struct:
  case kw_trait:
  case kw_var:
    return true;
  }
}

bool Token::isIdentifier() const {
  return isAny(identifier, escaped_identifier);
}

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

Lexer::Lexer(MojoDiags &diags, StringRef curBuffer, const char *curPtr)
    : diags(diags), curBuffer(curBuffer), curPtr(curPtr),
      curToken(Token::eof, StringRef(), 0), lastLineStart(nullptr),
      lastLineIndent(0) {
  lexToken();
}

Lexer::Lexer(MojoDiags &diags, const llvm::MemoryBuffer *buffer)
    : diags(diags), curBuffer(buffer->getBuffer()), curPtr(curBuffer.begin()),
      curToken(Token::eof, StringRef(), 0), lastLineStart(nullptr),
      lastLineIndent(0) {

  // Prime the first token.
  lexToken();
}

static StringRef findBuffer(llvm::SourceMgr &sourceMgr,
                            const LexerCursor &cursor) {
  unsigned cursorBufferId =
      sourceMgr.FindBufferContainingLoc(cursor.getToken().getLoc());
  assert(cursorBufferId && "invalid cursor!");
  const auto *buffer = sourceMgr.getMemoryBuffer(cursorBufferId);
  return buffer->getBuffer();
}

Lexer::Lexer(MojoDiags &diags, const LexerCursor &cursor)
    : diags(diags), curBuffer(findBuffer(diags.sourceMgr, cursor)),
      curToken(Token::eof, {}, 0) {
  cursor.restore(*this);
}

/// Emit an error message and return a Token::error token.
MojoInflightDiag Lexer::emitErrorAt(const char *loc, const Twine &message) {
  auto diag = diags.emitError(SMLoc::getFromPointer(loc), message);
  formToken(Token::error, loc, -1);
  return diag;
}

/// This function point is the funnel point for all tokens that are lexed.  This
/// updates curToken and does other final checking.
///
/// The tokenStartOffset field is used to indicate tokens whose spelling is
/// artificially shifted from the start of the token, notably things like
/// `x y` are given a spelling of "x y" and don't include the `.
void Lexer::formToken(Token::Kind kind, StringRef spelling, ssize_t indentation,
                      size_t tokenStartOffset) {
  // We're about to form a token.  If the token is at the start of line, make
  // sure the leading indentation of this token and the previous start of line
  // match in spelling, then update our current start-of-line marker.
  if (indentation != -1) {
    // Check that the leading indentation of these two tokens match.
    const char *thisLineStart =
        spelling.data() - indentation - tokenStartOffset;
    if (lastLineStart && memcmp(lastLineStart, thisLineStart,
                                std::min(indentation, lastLineIndent))) {
      diags.emitError(
          SMLoc::getFromPointer(spelling.data()),
          "indentation must not mix tabs and spaces; select one style");
    }

    lastLineStart = thisLineStart;
    lastLineIndent = indentation;
  }

  curToken = Token(kind, spelling, indentation);
}

//===----------------------------------------------------------------------===//
// Lexer Implementation Methods
//===----------------------------------------------------------------------===//

static bool isQuoteChar(char c) { return c == '\'' || c == '"'; }
static bool isTChar(char c) { return c == 't' || c == 'T'; }
static bool isRChar(char c) { return c == 'r' || c == 'R'; }

void Lexer::lexToken() {
  // This keeps track of the indentation of the current token from the start of
  // the line.  The first byte of the file starts with an indentation of zero,
  // but subsequent tokens always start out by following an existing token, so
  // they aren't at the start of line.
  ssize_t indentation = curPtr == curBuffer.begin() ? 0 : -1;
  const char *tokStart;
  // This is a helper lambda for forming tokens with tokStart and indentation,
  // and optionally incrementing `curPtr` to make some of the conditionals below
  // ergonomic.
  auto formToken = [&](Token::Kind kind, size_t incr = 0) {
    curPtr += incr;
    this->formToken(kind, tokStart, indentation);
  };

  while (true) {
    // This loop is set up so a "continue" can be used to ignore a whitespace
    // character.  Always reset 'tokStart'.
    tokStart = curPtr;
    switch (*curPtr++) {
    case 0:
      // This may either be a nul character in the source file or may be the EOF
      // marker that MemoryBuffer guarantees will be there.
      if (curPtr - 1 == curBuffer.end())
        return this->formToken(Token::eof, tokStart, 0);

      [[fallthrough]]; // Treat as whitespace.

      // Horizontal whitespace increases the indentation if current token is at
      // start of line.
    case ' ':
    case '\t':
      if (indentation != -1)
        ++indentation;
      continue;

      // Vertical whitespace resets the indentation to zero since anything that
      // comes after it is at the start of the line.
    case '\n':
    case '\r':
    case '\f':
    case '\v':
      indentation = 0;
      continue;

      // Handle \ at end of line by treating it as whitespace instead of
      // tracking the next token as start of line.
    case '\\': {
      // Check that there is only horizontal whitespace before the \n.
      while (*curPtr == ' ' || *curPtr == '\t')
        ++curPtr;
      if (*curPtr == '\n' || *curPtr == '\r') {
        if (*curPtr == '\r' && curPtr[1] == '\n') // Windows new line
          ++curPtr;
        ++curPtr;
        indentation = -1;
        continue;
      }
      emitErrorAt(tokStart, "encountered unexpected '\\'; use line "
                            "continuation at the end of lines");
      return;
    }

    default:
      // Handle identifiers.
      if (llvm::isAlpha(curPtr[-1])) {
        // Raw t-string: rt"..."
        if (isRChar(curPtr[-1]) && isTChar(*curPtr) &&
            curPtr + 1 < curBuffer.end() && isQuoteChar(curPtr[1]))
          return lexTString(tokStart, indentation);
        // T-string: t"..." — or raw t-string: tr"..."
        if (isTChar(curPtr[-1])) {
          bool nextIsQuote = isQuoteChar(*curPtr);
          bool nextIsRawPrefix = isRChar(*curPtr) &&
                                 curPtr + 1 < curBuffer.end() &&
                                 isQuoteChar(curPtr[1]);
          if (nextIsQuote || nextIsRawPrefix)
            return lexTString(tokStart, indentation);
        }
        // Raw string: r"..."
        if (isRChar(curPtr[-1]) && isQuoteChar(*curPtr))
          return lexString(tokStart, indentation);
        return lexIdentifierOrKeyword(tokStart, indentation);
      }

      // Unknown character, emit an error.
      emitErrorAt(tokStart, "unexpected character");
      return;

    case '_':
      // Handle identifiers.
      return lexIdentifierOrKeyword(tokStart, indentation);
    case '`':
      return lexBacktickIdentifier(tokStart, indentation);
    case '"':
    case '\'':
      return lexString(tokStart, indentation);
    case '%':
      if (*curPtr == '=')
        return formToken(Token::percent_equal, 1);
      return formToken(Token::percent);
    case '&':
      if (*curPtr == '=')
        return formToken(Token::amp_equal, 1);
      return formToken(Token::amp);
    case '(':
      return formToken(Token::l_paren);
    case ')':
      return formToken(Token::r_paren);
    case '*':
      if (*curPtr == '=')
        return formToken(Token::star_equal, 1);
      if (*curPtr == '*') {
        if (curPtr[1] == '=')
          return formToken(Token::star_star_equal, 2);
        return formToken(Token::star_star, 1);
      }
      return formToken(Token::star);
    case '+':
      if (*curPtr == '=')
        return formToken(Token::plus_equal, 1);
      return formToken(Token::plus);
    case ',':
      return formToken(Token::comma);
    case '-':
      if (*curPtr == '=')
        return formToken(Token::minus_equal, 1);
      if (*curPtr == '>')
        return formToken(Token::minus_greater, 1);
      return formToken(Token::minus);
    case '.':
      if (llvm::isDigit(*curPtr))
        return lexFloat(tokStart, indentation);
      if (*curPtr == '.' && curPtr[1] == '.')
        return formToken(Token::dot_dot_dot, 2);
      return formToken(Token::dot);
    case '/':
      if (*curPtr == '=')
        return formToken(Token::slash_equal, 1);
      if (*curPtr == '/') {
        if (curPtr[1] == '=')
          return formToken(Token::slash_slash_equal, 2);
        return formToken(Token::slash_slash, 1);
      }
      return formToken(Token::slash);
    case ':':
      // TODO: Python keeps track of nesting level in the lexer to report
      // mismatched tokens here.  How does that affect error recovery?
      if (*curPtr == '=')
        return formToken(Token::colon_equal, 1);
      return formToken(Token::colon);
    case ';':
      return formToken(Token::semi);
    case '<':
      switch (*curPtr) {
      case '<':
        if (curPtr[1] == '=')
          return formToken(Token::less_less_equal, 2);
        return formToken(Token::less_less, 1);
      case '=':
        return formToken(Token::less_equal, 1);
      case '>':
        return formToken(Token::less_greater, 1);
      }
      return formToken(Token::less);
    case '=':
      if (*curPtr == '=')
        return formToken(Token::equal_equal, 1);
      return formToken(Token::equal);
    case '>':
      switch (*curPtr) {
      case '=':
        return formToken(Token::greater_equal, 1);
      case '>':
        if (curPtr[1] == '=')
          return formToken(Token::right_right_equal, 2);
        return formToken(Token::right_right, 1);
      }
      return formToken(Token::greater);
    case '@':
      if (*curPtr == '=')
        return formToken(Token::at_equal, 1);
      return formToken(Token::at);
    case '[':
      return formToken(Token::l_square);
    case ']':
      return formToken(Token::r_square);
    case '^':
      if (*curPtr == '=')
        return formToken(Token::caret_equal, 1);
      return formToken(Token::caret);
    case '{':
      return formToken(Token::l_brace);
    case '|':
      if (*curPtr == '=')
        return formToken(Token::pipe_equal, 1);
      return formToken(Token::pipe);
    case '}':
      return formToken(Token::r_brace);
    case '~':
      return formToken(Token::tilde);
    case '!':
      if (*curPtr == '=')
        return formToken(Token::exclaim_equal, 1);
      emitErrorAt(tokStart, "unexpected character");
      return;

    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      return lexInteger(tokStart, indentation);

    case '#':
      skipComment();
      indentation = 0; // skipComment eats the \n.
      continue;
    }
  }
}

/// Lex an identifier or keyword that starts with a letter.
///
/// TODO: Python supports unicode in is_potential_identifier_start etc.
///
void Lexer::lexIdentifierOrKeyword(const char *tokStart, ssize_t indentation) {
  // Match the rest of the identifier regex: [0-9a-zA-Z_$]*
  while (llvm::isAlpha(*curPtr) || llvm::isDigit(*curPtr) || *curPtr == '_' ||
         *curPtr == '$')
    ++curPtr;

  StringRef spelling(tokStart, curPtr - tokStart);

  // Check to see if this identifier is a keyword.
  Token::Kind kind = llvm::StringSwitch<Token::Kind>(spelling)
#define TOK_KEYWORD(SPELLING) .Case(#SPELLING, Token::kw_##SPELLING)
#include "Mojo/MojoParser/TokenKinds.def"
                         .Default(Token::identifier);

  formToken(kind, tokStart, indentation);
}

/// Lex an identifier with backtick syntax, e.g. `ide nt if ier` or `fn`.  These
/// may contain any character other than vertical whitespace and `'s in them and
/// are otherwise interpreted verbatim as an identifier.
void Lexer::lexBacktickIdentifier(const char *tokStart, ssize_t indentation) {
  assert(curPtr[-1] == '`');
  while (true) {
    switch (*curPtr++) {
    case '`':
      // Found the end character.
      if (curPtr - tokStart - 2 == 0)
        emitErrorAt(tokStart, "backtick identifier must not be empty; add "
                              "content between the backticks");

      formToken(Token::escaped_identifier,
                StringRef(tokStart + 1, curPtr - tokStart - 2), indentation,
                /*tokenOffset*/ 1);
      return;
    case '\n':
    case '\r':
    case '\v':
    case '\f':
      // Vertical whitespace within a ` is invalid is the end of the comment.
      emitErrorAt(tokStart, "unterminated backtick identifier");
      return;
    case 0:
      // If this is the end of the buffer, end the comment.
      if (curPtr - 1 == curBuffer.end()) {
        --curPtr;
        emitErrorAt(tokStart, "unterminated backtick identifier");
        return;
      }
      [[fallthrough]];
    default:
      // Skip over other characters.
      break;
    }
  }
}

/// Skip a comment line, starting with a '#' and going to end of line.
void Lexer::skipComment() {
  while (true) {
    switch (*curPtr++) {
    case '\n':
    case '\r':
    case '\v':
    case '\f':
      // Vertical whitespaces is the end of the comment.
      return;
    case 0:
      // If this is the end of the buffer, end the comment.
      if (curPtr - 1 == curBuffer.end()) {
        --curPtr;
        return;
      }
      [[fallthrough]];
    default:
      // Skip over other characters.
      break;
    }
  }
}

/// Checks if character \p C is one of the 8 octal digits.
static bool isOctalDigit(char C) { return C >= '0' && C <= '7'; }

// === Static helpers for string/t-string skipping (shared by lexer & parser)

bool Lexer::skipTStringPrefix(const char *&ptr) {
  if (isRChar(ptr[0]) && isTChar(ptr[1])) {
    ptr += 2;
    return true;
  }
  if (isTChar(ptr[0]) && isRChar(ptr[1])) {
    ptr += 2;
    return true;
  }
  ptr++; // plain t/T
  return false;
}

bool Lexer::isTripleQuote(const char *ptr, const char *end, char quoteChar) {
  return ptr + 2 < end && ptr[0] == quoteChar && ptr[1] == quoteChar &&
         ptr[2] == quoteChar;
}

void Lexer::skipStringBody(const char *&ptr, const char *end, char quoteChar,
                           bool isTriple) {
  while (ptr < end) {
    if (*ptr == '\\') {
      ptr++;
      if (ptr < end)
        ptr++;
      continue;
    }
    if (*ptr == quoteChar) {
      if (!isTriple) {
        ptr++;
        return;
      }
      if (isTripleQuote(ptr, end, quoteChar)) {
        ptr += 3;
        return;
      }
    }
    ptr++;
  }
}

// === Instance methods

bool Lexer::isTripleQuoteAt(char quoteChar) const {
  return isTripleQuote(curPtr, curBuffer.end(), quoteChar);
}

bool Lexer::consumeQuoteDelimiter(char quoteChar, bool isTriple) {
  if (!isTriple) {
    ++curPtr;
    return true;
  }

  if (isTripleQuoteAt(quoteChar)) {
    curPtr += 3;
    return true;
  }
  return false;
}

bool Lexer::consumeQuoteOpening(char quoteChar) {
  bool isTriple = isTripleQuoteAt(quoteChar);
  consumeQuoteDelimiter(quoteChar, isTriple);
  return isTriple;
}

/// Consume and validate a \u (4-digit) or \U (8-digit) unicode escape sequence.
/// `curPtr` must point at the 'u'/'U' character on entry; on return it points
/// past the last consumed digit (or stops early if digits are missing).
/// Returns Success on success, or ErrorAt on failure.
Lexer::ConsumeStringResult Lexer::consumeUnicodeEscape(const char *&curPtr,
                                                       const char *bufEnd) {
  char escChar = *curPtr;
  const char *escStart = curPtr - 1; // backslash
  ++curPtr;
  int numDigits = (escChar == 'u') ? 4 : 8;
  uint32_t cp = 0;
  int i = 0;
  while (curPtr != bufEnd && llvm::isHexDigit(*curPtr) && i < numDigits) {
    cp = (cp << 4) | llvm::hexDigitValue(*curPtr);
    ++curPtr;
    ++i;
  }
  if (i != numDigits) {
    return {.result = Lexer::ConsumeStringResult::ErrorAt{
                .errorLoc = escStart,
                .errorMsg = (escChar == 'u')
                                ? "invalid unicode escape sequence: \\u "
                                  "requires exactly four hex digits"
                                : "invalid unicode escape sequence: \\U "
                                  "requires exactly eight hex digits"}};
  }
  if (cp > 0x10FFFF)
    return {.result = Lexer::ConsumeStringResult::ErrorAt{
                .errorLoc = escStart,
                .errorMsg = "unicode escape sequence out of range: value must "
                            "not exceed U+10FFFF"}};
  if (cp >= 0xD800 && cp <= 0xDFFF)
    return {.result = Lexer::ConsumeStringResult::ErrorAt{
                .errorLoc = escStart,
                .errorMsg = "unicode escape sequences do not support surrogate "
                            "code points (U+D800 to U+DFFF); use '\\U' with "
                            "the full code point (not a UTF-16 surrogate "
                            "pair)"}};
  return {.result = Lexer::ConsumeStringResult::Success{}};
}

/// Scan for a string's closing quote, handling escape sequences.
/// Returns Success, InvalidEscape, or Unterminated.
/// Updates curPtr to point after the closing quote if found.
/// On Unterminated, curPtr is left at the problematic position.
Lexer::ConsumeStringResult
Lexer::consumeStringBody(char quoteChar, bool isTripleQuote, bool isRaw) {
  auto startPtr = curPtr;
  auto result = ConsumeStringResult{.result = ConsumeStringResult::Success{}};
  while (curPtr != curBuffer.end()) {
    switch (*curPtr) {
    case '\\':
      if (isRaw) {
        ++curPtr;
        if (curPtr == curBuffer.end())
          return {.result = ConsumeStringResult::Unterminated{}};
        // Handle trailing windows style newline.
        if (*curPtr == '\r' && curPtr + 1 < curBuffer.end() &&
            curPtr[1] == '\n')
          ++curPtr;
        ++curPtr;
        break;
      }

      // Handle escape sequences
      ++curPtr;
      if (curPtr == curBuffer.end())
        return {.result = ConsumeStringResult::Unterminated{}};

      if (isOctalDigit(*curPtr)) {
        // at most 3 octal digits.
        size_t i = 0;
        while (curPtr != curBuffer.end() && isOctalDigit(*curPtr) && i < 3) {
          ++curPtr;
          ++i;
        }
      } else if (*curPtr == 'x') {
        ++curPtr;
        // exactly 2 hex digits.
        size_t i = 0;
        while (curPtr != curBuffer.end() && llvm::isHexDigit(*curPtr) &&
               i < 2) {
          ++curPtr;
          ++i;
        }
        if (i != 2) {
          result.result = ConsumeStringResult::ErrorAt{
              .errorLoc = startPtr,
              .errorMsg =
                  "invalid hex escape sequence: exactly two hex digits needed"};
        }
      } else if (*curPtr == 'u' || *curPtr == 'U') {
        auto res = consumeUnicodeEscape(curPtr, curBuffer.end());
        if (auto *err = std::get_if<ConsumeStringResult::ErrorAt>(&res.result))
          result.result = *err;
      } else if (!llvm::is_contained({'\\', '"', '\'', '\n', '\r', 'a', 'b',
                                      'f', 'n', 'r', 't', 'v'},
                                     *curPtr)) {
        result.result = ConsumeStringResult::ErrorAt{
            .errorLoc = curPtr - 1,
            .errorMsg = "invalid escape sequence",
        };
        ++curPtr;
      } else {
        if (*curPtr == '\r' && curPtr + 1 < curBuffer.end() &&
            curPtr[1] == '\n') // Windows newline
          ++curPtr;
        ++curPtr;
      }
      break;
    case '\'':
    case '"':
      // Check if this is the closing quote
      if (*curPtr == quoteChar) {
        if (consumeQuoteDelimiter(quoteChar, isTripleQuote)) {
          // Successfully found closing quote
          return result;
        }
      }
      ++curPtr;
      break;
    case '\n':
    case '\r':
      // newline isn't allowed in a short string.
      if (!isTripleQuote)
        return {.result = ConsumeStringResult::Unterminated{}};
      ++curPtr;
      break;
    default:
      ++curPtr;
      break;
    }
  }

  return {.result = ConsumeStringResult::Unterminated{}};
}

/// Lex a string literal.
///
/// stringliteral   ::=  [stringprefix](shortstring | longstring)
/// stringprefix    ::=  "r" | "R"
/// shortstring     ::=  "'" shortstringitem* "'" | '"' shortstringitem* '"'
/// longstring      ::=  "'''" longstringitem* "'''" |
///                      '"""' longstringitem* '"""'
/// shortstringitem ::=  shortstringchar | stringescapeseq
/// longstringitem  ::=  longstringchar | stringescapeseq
/// shortstringchar ::=  <any source character except "\" or newline or the
///                      quote>
/// longstringchar  ::=  <any source character except "\">
/// stringescapeseq ::=  "\" <any source character>
void Lexer::lexString(const char *tokStart, ssize_t indentation) {
  curPtr = tokStart;
  bool isRaw = false;
  if (*curPtr == 'r' || *curPtr == 'R') {
    isRaw = true;
    ++curPtr;
  }

  if (*curPtr != '\'' && *curPtr != '"') {
    emitErrorAt(tokStart,
                "expected a quote to open the string; use ''' or '\"'");
    return;
  }
  char quoteChar = *curPtr;
  bool isTripleQuote = consumeQuoteOpening(quoteChar);

  auto result = consumeStringBody(quoteChar, isTripleQuote, isRaw);
  std::visit(Overloaded{
                 [&](ConsumeStringResult::Success) {
                   formToken(Token::string, tokStart, indentation);
                 },
                 [&](ConsumeStringResult::ErrorAt errorAt) {
                   emitErrorAt(errorAt.errorLoc, errorAt.errorMsg);
                 },
                 [&](ConsumeStringResult::Unterminated) {
                   emitErrorAt(tokStart, "unterminated string");
                 },
             },
             result.result);
}

bool Lexer::findTStringInterpolationEnd(const char *&ptr, const char *end,
                                        size_t depth) {
  assert(depth <= 20 && "t-string nesting depth (20) exceeded");
  assert((ptr && *ptr == '{') && "expected '{' at beginning of interpolation");

  struct QuotedString {
    char quote;
    bool triple;
  };

  auto tryConsumeStringOpening =
      [&](auto... charPrefix) -> std::optional<QuotedString> {
    constexpr auto prefixLen = sizeof...(charPrefix);
    if (ptr + prefixLen < end) {
      size_t index = 1;
      if ((true && ... && charPrefix(ptr[index++]))) {
        ptr += prefixLen;
        char quote = *ptr;
        bool triple = isTripleQuote(ptr, end, quote);
        ptr += triple ? 3 : 1;
        return {{.quote = quote, .triple = triple}};
      }
    }
    return {};
  };

  ptr++; // skip '{'
  int braceDepth = 1;
  while (ptr < end && braceDepth > 0) {
    switch (*ptr) {
    case '{':
      braceDepth++;
      ptr++;
      break;
    case '}':
      braceDepth--;
      if (braceDepth > 0)
        ptr++;
      break;
    case '\\':
      ptr++;
      if (ptr < end)
        ptr++;
      break;
    case '\'':
    case '"':
      if (auto result = tryConsumeStringOpening(); result)
        skipStringBody(ptr, end, result->quote, result->triple);
      break;
    case 'r':
    case 'R':
      // rt"..." / rT"..." — raw t-string.
      if (auto result = tryConsumeStringOpening(isTChar, isQuoteChar); result) {
        skipTStringBody(ptr, end, result->quote, result->triple, depth + 1);
        break;
      }
      // r"..." — raw string.
      if (auto result = tryConsumeStringOpening(isQuoteChar); result) {
        skipStringBody(ptr, end, result->quote, result->triple);
        break;
      }
      ptr++;
      break;
    case 't':
    case 'T':
      // tr"..." / tR"..." — raw t-string.
      if (auto result = tryConsumeStringOpening(isRChar, isQuoteChar); result) {
        skipTStringBody(ptr, end, result->quote, result->triple, depth + 1);
        break;
      }
      // t"..." — plain t-string.
      if (auto result = tryConsumeStringOpening(isQuoteChar); result) {
        skipTStringBody(ptr, end, result->quote, result->triple, depth + 1);
        break;
      }
      ptr++;
      break;
    default:
      ptr++;
      break;
    }
  }
  if (braceDepth == 0) {
    ptr++; // skip '}'
    return true;
  }
  return false;
}

bool Lexer::skipTStringBody(const char *&ptr, const char *end, char quoteChar,
                            bool isTriple, size_t depth) {
  while (ptr < end) {
    switch (*ptr) {
    case '{':
      // Escaped brace {{ stays in the literal region.
      if (ptr + 1 < end && ptr[1] == '{') {
        ptr += 2;
        continue;
      }
      if (!findTStringInterpolationEnd(ptr, end, depth))
        return false;
      continue;
    case '}':
      // Escaped brace }} stays in the literal region.
      if (ptr + 1 < end && ptr[1] == '}') {
        ptr += 2;
        continue;
      }
      // Lone '}' at depth 0 — just advance past it.
      ptr++;
      continue;
    case '\\':
      ptr++;
      if (ptr < end)
        ptr++;
      continue;
    case '\'':
    case '"':
      if (*ptr == quoteChar) {
        if (!isTriple) {
          ptr++;
          return true;
        }
        if (isTripleQuote(ptr, end, quoteChar)) {
          ptr += 3;
          return true;
        }
      }
      ptr++;
      continue;
    case '\n':
    case '\r':
      if (!isTriple)
        return false; // newline in single-quoted literal region
      [[fallthrough]];
    default:
      ptr++;
      continue;
    }
  }
  return false; // unterminated
}

/// Lex a t-string literal as a single token, consuming the entire body from the
/// opening t" to the closing ". Also handles raw t-string prefixes (rt, tr).
/// The parser will later walk the spelling to extract literal parts and
/// expression regions.
void Lexer::lexTString(const char *tokStart, ssize_t indentation) {
  curPtr = tokStart;
  skipTStringPrefix(curPtr);
  assert(isQuoteChar(*curPtr) && "lexTString expected a quote character");

  char quoteChar = *curPtr;
  bool isTriple = consumeQuoteOpening(quoteChar);

  if (skipTStringBody(curPtr, curBuffer.end(), quoteChar, isTriple))
    return formToken(Token::t_string, tokStart, indentation);

  // Distinguish newline-in-single-quoted from truly unterminated.
  // If we stopped at a newline character, give the more specific error.
  if (curPtr != curBuffer.end() && (*curPtr == '\n' || *curPtr == '\r'))
    emitErrorAt(tokStart,
                "t-string cannot contain unescaped newline (use triple "
                "quotes or escape as \\n)");
  else
    emitErrorAt(tokStart, "unterminated t-string (missing closing quote)");
}

/// Lex a integer number literal.
///
/// integer      ::=  decinteger | bininteger | octinteger | hexinteger
/// decinteger   ::=  nonzerodigit ("_" | digit)* | "0"+ ("_" | "0")*
/// bininteger   ::=  "0" ("b" | "B") (["_"] bindigit)+
/// octinteger   ::=  "0" ("o" | "O") (["_"] octdigit)+
/// hexinteger   ::=  "0" ("x" | "X") (["_"] hexdigit)+
/// nonzerodigit ::=  "1"..."9"
/// digit        ::=  "0"..."9"
/// bindigit     ::=  "0" | "1"
/// octdigit     ::=  "0"..."7"
/// hexdigit     ::=  digit | "a"..."f" | "A"..."F"
///
/// DIFFERENCES with Python:
/// - Python uses the following more restrictive productions, which
///   disallows `1__9_` for example:
///   decinteger   ::=  nonzerodigit (["_"] digit)* | "0"+ (["_"] "0")*
///   same thing for  bininteger, octinteger and hexinteger
/// - Python warns if the numeric literal is immediately followed by
///   other keyword or identifier.
void Lexer::lexInteger(const char *tokStart, ssize_t indentation) {
  assert(llvm::isDigit(curPtr[-1]));

  if (curPtr[-1] == '0') {
    if (*curPtr == 'b' || *curPtr == 'B') {
      ++curPtr;
      bool hasDigits = false;
      while (*curPtr == '0' || *curPtr == '1' || *curPtr == '_') {
        hasDigits |= *curPtr != '_';
        ++curPtr;
      }
      if (!hasDigits) {
        emitErrorAt(curPtr, "no digits specified for binary literal");
        return;
      }
    } else if (*curPtr == 'o' || *curPtr == 'O') {
      ++curPtr;
      bool hasDigits = false;
      while (isOctalDigit(*curPtr) || *curPtr == '_') {
        hasDigits |= *curPtr != '_';
        ++curPtr;
      }
      if (!hasDigits) {
        emitErrorAt(curPtr, "no digits specified for octal literal");
        return;
      }
    } else if (*curPtr == 'x' || *curPtr == 'X') {
      ++curPtr;
      bool hasDigits = false;
      while (llvm::isHexDigit(*curPtr) || *curPtr == '_') {
        hasDigits |= *curPtr != '_';
        ++curPtr;
      }
      if (!hasDigits) {
        emitErrorAt(curPtr, "no digits specified for hex literal");
        return;
      }
    } else if (*curPtr == '.' || *curPtr == 'e' || *curPtr == 'E' ||
               *curPtr == 'j' || *curPtr == 'J') {
      return lexFloat(tokStart, indentation);
    } else if (*curPtr == '0' || *curPtr == '_') {
      // Literal zero, ex. 00, 00_0, 0_0_0__0
      // Superset of Python's grammar, we allow consecutive and trailing `_`
      // ex. 0__0_
      do
        ++curPtr;
      while (*curPtr == '0' || *curPtr == '_');
    } else if (llvm::isDigit(*curPtr)) {
      // ex. 0123
      emitErrorAt(curPtr,
                  "decimal integer literals must not use leading zeros; "
                  "add '0o' for octal literals");
      return;
    }
  } else {
    // nonzerodigit
    // Superset of Python's grammar, we allow consecutive and trailing `_`
    // ex. 1__9_
    while (llvm::isDigit(*curPtr) || *curPtr == '_')
      ++curPtr;
  }
  if (*curPtr == '.' || *curPtr == 'e' || *curPtr == 'E' || *curPtr == 'j' ||
      *curPtr == 'J')
    return lexFloat(tokStart, indentation);
  formToken(Token::integer, tokStart, indentation);
}

/// Lex a float number literal.
/// When the function is called tokStart points to "." or a digit.
/// floatnumber   ::=  pointfloat | exponentfloat
/// pointfloat    ::=  [digitpart] fraction | digitpart "."
/// exponentfloat ::=  (digitpart | pointfloat) exponent
/// digitpart     ::=  digit ("_" | digit)*
/// fraction      ::=  "." digitpart
/// exponent      ::=  ("e" | "E") ["+" | "-"] digitpart
///
/// DIFFERENCES with Python:
/// - Python uses the following more restrictive productions, which
///   disallows `1__9_` for example:
///   digitpart     ::=  digit (["_"] digit)*
void Lexer::lexFloat(const char *tokStart, ssize_t indentation) {
  assert(*tokStart == '.' || llvm::isDigit(*tokStart));
  // lexFloat could have been called from lexInteger so reset curPtr to undo
  // previous increments done by lexInteger
  curPtr = tokStart;
  if (llvm::isDigit(*curPtr)) {
    do
      ++curPtr;
    while (llvm::isDigit(*curPtr) || *curPtr == '_');
  }
  if (*curPtr == '.')
    ++curPtr;
  if (llvm::isDigit(*curPtr)) {
    do
      ++curPtr;
    while (llvm::isDigit(*curPtr) || *curPtr == '_');
  }
  if (*curPtr == 'e' || *curPtr == 'E') {
    ++curPtr;
    if (*curPtr == '+' || *curPtr == '-')
      ++curPtr;
    if (!llvm::isDigit(*curPtr)) {
      emitErrorAt(curPtr, "expected a digit after the exponent");
      return;
    }
    while (llvm::isDigit(*curPtr) || *curPtr == '_')
      ++curPtr;
  }
  formToken(Token::float_num, tokStart, indentation);
}

static std::string filterUnderscores(StringRef spelling) {
  std::string digits;
  digits.reserve(spelling.size());
  for (auto c : spelling) {
    if (c != '_')
      digits.push_back(c);
  }
  return digits;
}

/// Return the a value for the specified string, which is known to have been
/// lexed as a float literal token.
IPRational Lexer::getFloatLiteralValue(StringRef spelling) {
  std::string digits = filterUnderscores(spelling);
  spelling = StringRef(digits);
  IPInt numerator(0);

  size_t digitsIndex = 0;
  bool pastDecimal = false;
  bool foundE = false;
  size_t denominatorCounter = 0;
  while (digitsIndex < digits.size()) {
    char digit = digits[digitsIndex];
    if (digit >= '0' && digit <= '9') {
      char decimalValue = digit - '0';
      numerator = numerator * IPInt(10) + IPInt(decimalValue);
      if (pastDecimal)
        ++denominatorCounter;
    } else if (digit == '.' && !pastDecimal) {
      pastDecimal = true;
    } else if (digit == 'e' || digit == 'E') {
      foundE = true;
      ++digitsIndex;
      break;
    } else {
      assert(false && "bad float literal");
    }
    ++digitsIndex;
  }
  IPInt denominator(IPInt(10).exponentiate(denominatorCounter));

  if (foundE) {
    IPInt exponent = 0;
    bool negativeSign = false;
    if (digits[digitsIndex] == '-') {
      negativeSign = true;
      ++digitsIndex;
    } else if (digits[digitsIndex] == '+') {
      ++digitsIndex;
    }
    while (digitsIndex < digits.size()) {
      char digit = digits[digitsIndex];
      if (digit >= '0' && digit <= '9') {
        char decimalValue = digit - '0';
        exponent = exponent * 10;
        exponent = exponent + IPInt(decimalValue);
      }
      ++digitsIndex;
    }
    IPInt exponentMulValue = IPInt(10).exponentiate(exponent);
    if (negativeSign)
      denominator = denominator * exponentMulValue;
    else
      numerator = numerator * exponentMulValue;
  }

  return IPRational(numerator, denominator);
}

/// Helper function to process escape sequences in string content.
// Encode a Unicode code point as UTF-8 and append the bytes to `out`.
static void appendUtf8(uint32_t cp, std::string &out) {
  if (cp <= 0x7F) {
    out.push_back(static_cast<char>(cp));
  } else if (cp <= 0x7FF) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp <= 0xFFFF) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

// Advance `src` past one escape unit, appending produced bytes to `out`.
// Returns true when bytes were appended, false for line continuations.
// When `isRaw` is true, backslash is not treated as an escape leader.
// `end` is the one-past-the-end pointer of the source buffer; it is used
// to bounds-check multi-byte escape sequences (`\xNN`, `\uNNNN`, octal).
static bool advanceEscapeUnit(const char *&src, const char *end, bool isRaw,
                              std::string &out) {
  char c = *src++;
  if (c != '\\' || isRaw) {
    out.push_back(c); // regular char passes through as-is
    return true;
  }

  // Escape sequence: `c` was '\\', consume the specifier.
  assert(src < end && "invalid string should be caught by lexer");
  char c1 = *src++;
  switch (c1) {
  case '\\':
  case '"':
  case '\'':
    out.push_back(c1);
    return true;
  case '\n':
    return false; // line continuation (\ + LF)
  case '\r':      // line continuation (\ + CR[LF])
    if (src < end && *src == '\n')
      ++src;
    return false;
  case 'a':
    out.push_back('\a');
    return true;
  case 'b':
    out.push_back('\b');
    return true;
  case 'f':
    out.push_back('\f');
    return true;
  case 'n':
    out.push_back('\n');
    return true;
  case 'r':
    out.push_back('\r');
    return true;
  case 't':
    out.push_back('\t');
    return true;
  case 'v':
    out.push_back('\v');
    return true;
  case 'x': {
    assert(src + 2 <= end && "invalid \\x escape should be caught by lexer");
    char hex0 = src[0], hex1 = src[1];
    assert(llvm::isHexDigit(hex0) && llvm::isHexDigit(hex1) &&
           "invalid escape");
    src += 2;
    // `\xhh` encodes the Unicode code point U+00hh, matching Python `str`
    // semantics. Values >= 0x80 are emitted as two UTF-8 bytes so the
    // resulting string literal is well-formed UTF-8.
    uint32_t cp = (llvm::hexDigitValue(hex0) << 4) | llvm::hexDigitValue(hex1);
    appendUtf8(cp, out);
    return true;
  }
  case 'u':
  case 'U': {
    int numDigits = (c1 == 'u') ? 4 : 8;
    assert(src + numDigits <= end &&
           "invalid unicode escape should be caught by lexer");
    uint32_t cp = 0;
    for (int i = 0; i < numDigits; ++i) {
      assert(llvm::isHexDigit(src[i]) && "invalid escape");
      cp = (cp << 4) | llvm::hexDigitValue(src[i]);
    }
    src += numDigits;
    appendUtf8(cp, out);
    return true;
  }
  case '0':
  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7': {
    // c1 is the first (mandatory) octal digit. Up to two more may follow;
    // each is guarded by `src < end` before the dereference.
    assert(isOctalDigit(c1) && "switch case guarantees c1 is an octal digit");
    unsigned num = c1 - '0';
    for (int i = 0; i < 2 && src < end && isOctalDigit(*src); ++i, ++src)
      num = (num << 3) | (*src - '0');
    // `\ooo` encodes the Unicode code point with the given octal value,
    // matching Python `str` semantics and `\x` / `\u` handling above.
    appendUtf8(num, out);
    return true;
  }
  default:
    out.push_back(c1); // invalid escape — already diagnosed at lex time
    return true;
  }
}

static std::string processEscapeSequences(StringRef bytes, bool isRaw) {
  std::string result;
  result.reserve(bytes.size());
  const char *src = bytes.data();
  const char *end = bytes.data() + bytes.size();
  while (src < end)
    advanceEscapeUnit(src, end, isRaw, result);
  return result;
}

/// Return the a string value of `spelling` after the escape sequences are
/// handled. `spelling` is known to have been lexed as a string literal token.
std::string Lexer::getStringLiteralValue(StringRef bytes) {
  bool isRaw = false;
  if (bytes[0] == 'r' || bytes[0] == 'R') {
    isRaw = true;
    bytes = bytes.drop_front();
  }

  // Drop quotes and triple quotes.
  if (bytes.size() >= 6 &&
      (bytes.starts_with("\"\"\"") || bytes.starts_with("'''")))
    bytes = bytes.drop_front(3).drop_back(3);
  else
    bytes = bytes.drop_front().drop_back();

  return processEscapeSequences(bytes, isRaw);
}

std::string Lexer::getTStringLiteralValue(StringRef bytes, bool isRaw) {
  return processEscapeSequences(bytes, isRaw);
}

SmallVector<unsigned> Lexer::buildProcessedToSourceOffsets(const char *srcStart,
                                                           const char *srcEnd,
                                                           size_t procCount) {
  // Each entry records the source byte offset (from srcStart) of the escape
  // unit that produced processed byte i. For escape sequences (e.g. `\t`),
  // this is the offset of the leading backslash.
  SmallVector<unsigned> offsets;
  offsets.reserve(procCount + 1);
  const char *src = srcStart;
  std::string tmp;
  for (size_t produced = 0; produced < procCount;) {
    unsigned srcOffset = src - srcStart;
    tmp.clear();
    // Docstrings are not raw literals
    if (advanceEscapeUnit(src, srcEnd, /*isRaw=*/false, tmp)) {
      // Record the source offset once per produced byte (unicode escapes may
      // produce multiple UTF-8 bytes from a single escape sequence).
      for (size_t i = 0; i < tmp.size() && produced < procCount;
           ++i, ++produced)
        offsets.push_back(srcOffset);
    }
  }
  offsets.push_back(src - srcStart); // sentinel: source offset past last byte
  return offsets;
}

SMLoc Lexer::getStringLiteralStartLoc(StringRef spelling) {
  size_t stringStartOffset = 1;
  if (spelling[0] == 'r' || spelling[0] == 'R')
    ++stringStartOffset;
  // Handle triple quoted strings.
  if (spelling.size() >= 6 &&
      (spelling.starts_with("\"\"\"") || spelling.starts_with("'''")))
    stringStartOffset += 2;
  return SMLoc::getFromPointer(spelling.data() + stringStartOffset);
}

/// Return the a value for the specified string, which is known to have been
/// lexed as an integer literal token.
APInt Lexer::getIntegerLiteralValue(StringRef spelling) {
  APInt result;
  unsigned base = 10;
  if (spelling[0] == '0' && spelling.size() > 2) {
    switch (spelling[1]) {
    case 'b':
    case 'B':
      base = 2;
      spelling = spelling.drop_front(2);
      break;
    case 'o':
    case 'O':
      base = 8;
      spelling = spelling.drop_front(2);
      break;
    case 'x':
    case 'X':
      base = 16;
      spelling = spelling.drop_front(2);
      break;
    }
  }
  std::string digits = filterUnderscores(spelling);
  spelling = StringRef(digits);
  bool failed = spelling.getAsInteger(base, result);
  assert(!failed && "we know this should always work because we lexed it");
  (void)failed;
  return result;
}

//===----------------------------------------------------------------------===//
// Support methods
//===----------------------------------------------------------------------===//

/// Given a location that is at the start of a line, scan backwards to find
/// the end of the last line that contains a token, or start of the source
/// buffer if there is none.
SMLoc Lexer::findEndOfPreviousLine(SMLoc loc) const {
  // To find the end of the previous line, we repeatedly segment the buffer into
  // chunks from the current position to the start of the current line and scan
  // it to see if it contains any tokens.  If not, we keep going, if so we use
  // the end of the last token.
  auto locOffset = size_t(loc.getPointer() - curBuffer.data());
  assert(locOffset <= curBuffer.size() && "loc not in current buffer!");
  // Truncate whole buffer to this segment.
  StringRef buffer(curBuffer.data(), locOffset);

  while (1) {
    auto nextNewLine = buffer.find_last_of("\n\r");
    // If we ran out of lines to check, we must be at the start of the buffer.
    // Give up.
    if (nextNewLine == StringRef::npos) {
      // Make sure to consider tokens on the first line of the file.
      if (buffer.empty())
        return loc;
      nextNewLine = 0;
    }

    // Scan from the start of the line to the current position.
    auto *lineStart = curBuffer.data() + nextNewLine;
    Lexer tmpLexer(diags, curBuffer, lineStart);

    // If the token is on this line, then there was at least one token on this
    // line.  Report the error on the last token of this line.
    if (tmpLexer.getToken().getLoc().getPointer() < buffer.end()) {
      // The current line might end with a comment.  Re-lex it to find the last
      // token and complain right after it.
      Token lastToken = tmpLexer.getToken();
      do {
        lastToken = tmpLexer.getToken();
        tmpLexer.lexToken();
      } while (tmpLexer.getToken().getLoc().getPointer() < buffer.end());

      // Error location is at the end of the last token on this line.
      return lastToken.getEndLoc();
    }

    // Otherwise, drop the newline and anything after it and try again.
    buffer = buffer.take_front(nextNewLine);
  }
}

//===----------------------------------------------------------------------===//
// CrashReporter
//===----------------------------------------------------------------------===//

void CrashReporter::print(raw_ostream &os) const {
  // Ignore invalid source locations, so clients can deal with conditional
  // locations cleanly.
  if (!loc.isValid())
    return;

  auto &diags = shared.diags;

  os << "Crash " << message << " at " << diags.translateLocation(loc) << '\n';

  // TODO get buffer from SourceMgr;
  auto &sourceMgr = shared.getSourceMgr();
  unsigned bufferID = sourceMgr.FindBufferContainingLoc(loc);
  if (!bufferID)
    return; // Don't crash if we fail to find the source buffer for some reason.
  const llvm::MemoryBuffer &memBuffer = *sourceMgr.getMemoryBuffer(bufferID);

  // We know where the location is, though it may not be the
  // first token on the line.  We know the current lexer position which is the
  // first token we haven't processed (generally the next statement, but might
  // be in the middle of a statement.
  StringRef buffer = memBuffer.getBuffer();
  Lexer lexer(diags, buffer, loc.getPointer());

  // Figure out where the current location is: if it is on something that is the
  // start of a line, back it up to the end of the previous line.
  const char *curTokenPtr = loc.getPointer();
  auto curLineWithoutWhitespace =
      buffer.drop_back(buffer.end() - curTokenPtr).rtrim(" \t");
  if (curLineWithoutWhitespace.rtrim("\n\r") != curLineWithoutWhitespace)
    curTokenPtr =
        lexer.findEndOfPreviousLine(lexer.getToken().getLoc()).getPointer();

  // This helper prints a line of the source buffer with highlighting to keep
  // track of where things are.
  auto printSourceLine = [&](StringRef sourceLine) {
    os << "    >> " << sourceLine << '\n';
    // Print out ^'s at the start and current token pointer if they exist in the
    // line.
    if (loc.getPointer() < sourceLine.begin() && curTokenPtr > sourceLine.end())
      return; // Don't print fully "." lines.

    os << "       ";
    for (const char &c : sourceLine) {
      char charToPrint;
      if (&c == loc.getPointer())
        charToPrint = '^';
      else if (&c == curTokenPtr)
        charToPrint = '<';
      else if (&c > loc.getPointer() && &c < curTokenPtr)
        charToPrint = '.';
      else if (c == '\t')
        charToPrint = '\t';
      else
        charToPrint = ' ';
      os << charToPrint;
    }
    // The next token pointer is typically at the \n of the current line.
    if (sourceLine.end() == curTokenPtr)
      os << '<';
    os << '\n';
  };

  // Start by printing the first line of code
  // that we started on, being careful to stay in the source file.
  size_t stmtStartOffset = loc.getPointer() - buffer.data();

  size_t prevNewLine = buffer.find_last_of("\n\r", stmtStartOffset);
  prevNewLine = prevNewLine == StringRef::npos ? 0 : prevNewLine + 1;
  size_t nextNewLine = buffer.find_first_of("\n\r", stmtStartOffset);
  nextNewLine = nextNewLine == StringRef::npos ? buffer.size() : nextNewLine;

  StringRef sourceLine =
      StringRef(buffer.data() + prevNewLine, nextNewLine - prevNewLine);
  printSourceLine(sourceLine);

  // If the current token position isn't in the first line, then print a few
  // more lines of context just in case.
  size_t numLinesPrinted = 0;
  while (curTokenPtr > sourceLine.end() && numLinesPrinted++ < 4 &&
         nextNewLine != buffer.size()) {
    size_t nextNextNewLine = buffer.find_first_of("\n\r", nextNewLine + 1);
    nextNextNewLine =
        nextNextNewLine == StringRef::npos ? buffer.size() : nextNextNewLine;
    sourceLine = StringRef(buffer.data() + nextNewLine + 1,
                           nextNextNewLine - (nextNewLine + 1));
    printSourceLine(sourceLine);
    nextNewLine = nextNextNewLine;
  }
}
