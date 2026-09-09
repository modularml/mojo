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

#ifndef KGEN_MOJOPARSER_LEXER_H
#define KGEN_MOJOPARSER_LEXER_H

#include "Mojo/MojoParser/MojoDiags.h"

#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include <optional>

namespace M {
class IPRational;
} // namespace M

namespace M::KGEN::LIT {
class LexerCursor;
using llvm::SMLoc;

/// This represents a specific token for .mojo files.
class Token {
public:
  enum Kind {
#define TOK_MARKER(NAME) NAME,
#define TOK_IDENTIFIER(NAME) NAME,
#define TOK_LITERAL(NAME) NAME,
#define TOK_PUNCTUATION(NAME, SPELLING) NAME,
#define TOK_KEYWORD(SPELLING) kw_##SPELLING,
#include "TokenKinds.def"
  };

  Token(Kind kind, StringRef spelling, ssize_t indentation)
      : kind(kind), spelling(spelling), indentation(indentation) {}

  /// Return the bytes that make up this token in the original source buffer.
  StringRef getSpelling() const { return spelling; }

  /// Return the indentation of this token.
  std::optional<size_t> getIndentation() const {
    if (indentation == -1)
      return std::nullopt;
    return size_t(indentation);
  }

  /// Return true if the token is the first on a line.
  bool isStartOfLine() const { return indentation != -1; }

  // Token classification.
  Kind getKind() const { return kind; }
  bool is(Kind K) const { return kind == K; }

  bool isAny(Kind k1, Kind k2) const { return is(k1) || is(k2); }

  /// Return true if this token is one of the specified kinds.
  template <typename... T>
  bool isAny(Kind k1, Kind k2, Kind k3, T... others) const {
    if (is(k1))
      return true;
    return isAny(k2, k3, others...);
  }

  /// Return true if this token is any one of the specified token kinds.
  bool isAny(ArrayRef<Kind> kinds) const {
    for (auto k : kinds)
      if (kind == k)
        return true;
    return false;
  }

  bool isNot(Kind k) const { return kind != k; }
  bool isNot(ArrayRef<Kind> kinds) const { return !isAny(kinds); }

  /// Return true if this token isn't one of the specified kinds.
  template <typename... T>
  bool isNot(Kind k1, Kind k2, T... others) const {
    return !isAny(k1, k2, others...);
  }

  /// Return true if this is one of the keyword token kinds (e.g. kw_pass).
  bool isKeyword() const;

  /// Return true if this is a statement keyword (e.g. while, try, break).
  bool isStatementKeyword() const;

  /// Return true if this is a declaration keyword (e.g. class, def, fn).
  bool isDeclKeyword() const;

  /// Return true if the kind is either `identifier` or `escaped_identifier`.
  bool isIdentifier() const;

  // Location processing.
  SMLoc getLoc() const;
  SMLoc getEndLoc() const;
  llvm::SMRange getLocRange() const;

private:
  /// Discriminator that indicates the sort of token this is.
  Kind kind;

  /// A reference to the entire token contents; this is always a pointer into
  /// a memory buffer owned by the source manager.
  StringRef spelling;

  /// If this token is at the start of a logical source line, then this
  /// specifies the number of bytes the character is indented by.  If the token
  /// is not at the start of line (or follows a \ on the previous line), then
  /// this contains -1.
  ssize_t indentation;
};

/// This implements a lexer for .mojo files.
class Lexer {
public:
  Lexer(MojoDiags &diags, StringRef curBuffer, const char *curPtr);
  Lexer(MojoDiags &diags, const llvm::MemoryBuffer *buffer);
  Lexer(MojoDiags &diags, const LexerCursor &cursor);

  /// Move to the next valid token.
  void lexToken();

  const Token &getToken() const { return curToken; }

  /// Get an opaque pointer into the lexer state that can be restored later.
  LexerCursor getCursor() const;

  /// Return the a value for the specified string, which is known to have been
  /// lexed as an integer literal token.
  static APInt getIntegerLiteralValue(StringRef spelling);
  /// Return the a value for the specified string, which is known to have been
  /// lexed as a float literal token.
  static IPRational getFloatLiteralValue(StringRef spelling);
  /// Return the a string value of `spelling` after the escape sequences are
  /// handled. `spelling` is known to have been lexed as a string literal token.
  static std::string getStringLiteralValue(StringRef spelling);
  /// Return the string value of a t-string literal part after processing
  /// escape sequences. Unlike getStringLiteralValue, this expects raw content
  /// without quotes. When \p isRaw is true, backslashes are treated as literal
  /// characters (for raw t-strings like rt"..." or tr"...").
  static std::string getTStringLiteralValue(StringRef bytes,
                                            bool isRaw = false);
  /// Return a location to the start of the given string, after stripping the
  /// wrapping quotes.
  static SMLoc getStringLiteralStartLoc(StringRef spelling);

  /// Build a table mapping processed-string byte offsets to source byte
  /// offsets. \p srcStart points to the first byte of the source content
  /// (i.e., the character immediately after the opening quotes, as returned
  /// by getStringLiteralStartLoc). \p procCount is the number of bytes in
  /// the processed string (i.e., the result of getStringLiteralValue).
  /// Returns a vector of \p procCount + 1 entries: offsets[i] is the source
  /// byte offset (from srcStart) for processed byte i; offsets[procCount] is
  /// the source offset one past the last consumed byte (a sentinel).
  /// Docstrings are never raw strings, so isRaw is not exposed here.
  static SmallVector<unsigned>
  buildProcessedToSourceOffsets(const char *srcStart, const char *srcEnd,
                                size_t procCount);

  /// Detect and skip a t-string prefix at \p ptr. Handles t, T, rt, rT, Rt,
  /// RT, tr, tR, Tr, TR. Advances \p ptr past the prefix. Returns true if
  /// the prefix includes 'r'/'R' (i.e. this is a raw t-string).
  /// Caller must ensure \p ptr starts at a valid t-string prefix character.
  struct ConsumeStringResult {
    struct ErrorAt {
      const char *errorLoc = nullptr;
      const char *errorMsg = nullptr;
    };
    struct Unterminated {};
    struct Success {};

    std::variant<Success, Unterminated, ErrorAt> result{};
  };

  static bool skipTStringPrefix(const char *&ptr);

  /// Check if \p ptr points to a triple-quote sequence of \p quoteChar
  /// within the bounds [ptr, end). Used by both lexer and parser.
  static bool isTripleQuote(const char *ptr, const char *end, char quoteChar);

  /// Skip past a string literal body starting after the opening quote(s).
  /// Advances \p ptr past the closing quote delimiter. Handles escape
  /// sequences by simply skipping backslash + next character (no validation).
  /// Used by both the lexer and parser to skip over nested strings.
  static void skipStringBody(const char *&ptr, const char *end, char quoteChar,
                             bool isTriple);

  /// Starting after an opening '{' in a t-string interpolation, advance \p ptr
  /// to the matching '}' while correctly skipping nested strings, t-strings,
  /// and tracking brace depth. Returns true if matching '}' was found, false
  /// if \p end was reached without finding it. Used by both the lexer and
  /// parser.
  static bool findTStringInterpolationEnd(const char *&ptr, const char *end,
                                          size_t depth = 0);

  /// Skip past a t-string body (brace-tracking + nested string/t-string
  /// skipping). Advances \p ptr past the closing quote delimiter. Returns
  /// true if the closing quote was found, false if unterminated.
  /// Used by both the lexer and parser.
  static bool skipTStringBody(const char *&ptr, const char *end, char quoteChar,
                              bool isTriple, size_t depth = 0);

  /// Consume and validate a \\u (4-digit) or \\U (8-digit) unicode escape.
  /// \p curPtr must point at the 'u'/'U' on entry (the caller has consumed
  /// the leading '\\'); it is advanced past the consumed digits on return.
  /// Returns a \c ConsumeStringResult with \c Success on success, or
  /// \c ErrorAt on failure.
  static ConsumeStringResult consumeUnicodeEscape(const char *&curPtr,
                                                  const char *bufEnd);

  MojoInflightDiag emitTokenError(const Twine &message) {
    return emitErrorAt(getToken().getSpelling().data(), message);
  }

  /// Given a location that is at the start of a line, scan backwards to find
  /// the end of the last line that contains a token, or start of the source
  /// buffer if there is none.
  SMLoc findEndOfPreviousLine(SMLoc loc) const;

  /// Return the current buffer we are lexing from.
  StringRef getBuffer() const { return curBuffer; }

private:
  void formToken(Token::Kind kind, const char *tokStart, ssize_t indentation,
                 size_t tokenStartOffset = 0) {
    formToken(kind, StringRef(tokStart, curPtr - tokStart), indentation,
              tokenStartOffset);
  }
  void formToken(Token::Kind kind, StringRef spelling, ssize_t indentation,
                 size_t tokenStartOffset = 0);
  MojoInflightDiag emitErrorAt(const char *loc, const Twine &message);

  // Lexer implementation methods.
  void lexIdentifierOrKeyword(const char *tokStart, ssize_t indentation);
  void lexBacktickIdentifier(const char *tokStart, ssize_t indentation);
  void lexInteger(const char *tokStart, ssize_t indentation);
  void lexFloat(const char *tokStart, ssize_t indentation);
  void lexString(const char *tokStart, ssize_t indentation);
  void lexTString(const char *tokStart, ssize_t indentation);
  void skipComment();

  /// Check if curPtr points to a triple-quote sequence. Delegates to the
  /// static isTripleQuote.
  bool isTripleQuoteAt(char quoteChar) const;

  /// Consume a quote delimiter (single or triple). Returns true on success.
  bool consumeQuoteDelimiter(char quoteChar, bool isTriple);

  /// Consume the opening quote(s) of a string. Returns true if triple-quoted.
  bool consumeQuoteOpening(char quoteChar);

  /// Consume a string body up to its closing quote with full escape-sequence
  /// validation and error reporting. Used by lexString for actual string
  /// lexing (not for skipping nested strings — use the static skipStringBody
  /// for that).
  ConsumeStringResult consumeStringBody(char quoteChar, bool isTripleQuote,
                                        bool isRaw);

private:
  /// This the source file diagnostic manager to use.
  MojoDiags &diags;
  /// This is the overall memory buffer that we are lexing from.
  StringRef curBuffer;
  /// This the start of the next byte to lex.
  const char *curPtr;
  /// This is the next token that hasn't been consumed yet.
  Token curToken;

  // This is the start of the last token that was at a beginning of line, and
  // the indentation (in bytes) of that token.
  const char *lastLineStart;
  ssize_t lastLineIndent;

  Lexer(const Lexer &) = delete;
  void operator=(const Lexer &) = delete;
  friend class LexerCursor;
};

/// This is the state captured for a lexer cursor.
class LexerCursor {
public:
  LexerCursor()
      : curPtr(0), curToken(Token(Token::eof, StringRef(), 0)),
        lastLineStart(nullptr), lastLineIndent(0) {}
  LexerCursor(const Lexer &lexer)
      : curPtr(lexer.curPtr), curToken(lexer.getToken()),
        lastLineStart(lexer.lastLineStart),
        lastLineIndent(lexer.lastLineIndent) {}
  LexerCursor(const LexerCursor &cursor) = default;
  LexerCursor &operator=(const LexerCursor &cursor) = default;

  /// Get a cursor that indicates the end of file.  This isn't for continued
  /// lexing, it is for comparisons.
  static LexerCursor getEOF(const llvm::MemoryBuffer *buffer) {
    LexerCursor result;
    result.curPtr = buffer->getBufferEnd() + 1;
    result.curToken = Token(Token::eof, StringRef(result.curPtr, 0), 0);
    return result;
  }

  void restore(Lexer &lexer) const {
    lexer.curPtr = curPtr;
    lexer.curToken = curToken;
    lexer.lastLineStart = lastLineStart;
    lexer.lastLineIndent = lastLineIndent;
  }

  /// Return true if this cursor is default constructed, not valid for lexing.
  bool isInvalid() const { return curPtr == nullptr; }

  /// Return an internal pointer that represents the cursor state without the
  /// current token.
  const char *getState() const { return curPtr; }
  const Token &getToken() const { return curToken; }

  bool operator==(const LexerCursor &rhs) const { return curPtr == rhs.curPtr; }
  bool operator!=(const LexerCursor &rhs) const { return !(*this == rhs); }

private:
  const char *curPtr;
  Token curToken;
  const char *lastLineStart;
  ssize_t lastLineIndent;
};

inline LexerCursor Lexer::getCursor() const { return LexerCursor(*this); }

} // namespace M::KGEN::LIT

#endif // KGEN_MOJOPARSER_LEXER_H
