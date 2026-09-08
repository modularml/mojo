# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
#
# File originates from:
#   Repo:   git@github.com:psf/black.git
#   Commit: d4a85643a465f5fae2113d07d22d021d4af4795a
#   Path:   src/mblib2to3/pgen2/tokenize.py
#
# ===----------------------------------------------------------------------=== #

# Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006 Python Software Foundation.
# All rights reserved.

# mypy: allow-untyped-defs, allow-untyped-calls

"""Tokenization help for Python programs.

generate_tokens(readline) is a generator that breaks a stream of
text into Python tokens.  It accepts a readline-like method which is called
repeatedly to get the next line of input (or "" for EOF).  It generates
5-tuples with these members:

    the token type (see token.py)
    the token (a string)
    the starting (row, column) indices of the token (a 2-tuple of ints)
    the ending (row, column) indices of the token (a 2-tuple of ints)
    the original line (string)

It is designed to match the working of the Python tokenizer exactly, except
that it produces COMMENT tokens for comments and gives type OP for all
operators

Older entry points
    tokenize_loop(readline, tokeneater)
    tokenize(readline, tokeneater=printtoken)
are the same, except instead of generating tokens, tokeneater is a callback
function to which the 5 fields described above are passed as 5 arguments,
each time a new token is found."""

from collections.abc import Callable, Iterable, Iterator
from re import Pattern
from typing import (
    Final,
    cast,
)

from mblib2to3.pgen2.grammar import Grammar
from mblib2to3.pgen2.token import *

__author__ = "Ka-Ping Yee <ping@lfw.org>"
__credits__ = "GvR, ESR, Tim Peters, Thomas Wouters, Fred Drake, Skip Montanaro"

import re
from codecs import BOM_UTF8, lookup

from mblib2to3.pgen2.token import *

from . import token

__all__ = [x for x in dir(token) if x[0] != "_"] + [
    "tokenize",
    "generate_tokens",
    "untokenize",
]
del token


def group(*choices):  # noqa: ANN202
    return "(" + "|".join(choices) + ")"


def any(*choices):  # noqa: ANN202
    return group(*choices) + "*"


def maybe(*choices):  # noqa: ANN202
    return group(*choices) + "?"


def _combinations(*l):  # noqa: ANN202
    return {x + y for x in l for y in l + ("",) if x.casefold() != y.casefold()}


Whitespace = r"[ \f\t]*"
Comment = r"#[^\r\n]*"
Ignore = Whitespace + any(r"\\\r?\n" + Whitespace) + maybe(Comment)
Name = (  # this is invalid but it's fine because Name comes after Number in all groups
    r"[^\s#\(\)\[\]\{\}+\-*/!@$%^&=|;:'\",\.<>/?~\\]+"
)

# Unlike Python, Mojo accepts trailing underscores in numeric literals (e.g.
# `2_000_000_`, `0xFF_`, `1.5_e3`). Each digit run therefore ends with a
# trailing `_*`. Underscores still may not lead a number, follow `.` or the
# exponent marker directly, or stand alone after a base prefix. The bare-zero
# decimal form likewise allows interior zeros/underscores (`00`, `0_0`), all of
# which Mojo reads as zero.
Binnumber = r"0[bB]_*[01]+(?:_+[01]+)*_*"
Hexnumber = r"0[xX]_*[\da-fA-F]+(?:_+[\da-fA-F]+)*_*[lL]?"
Octnumber = r"0[oO]?_*[0-7]+(?:_+[0-7]+)*_*[lL]?"
Decnumber = group(r"[1-9]\d*(?:_+\d+)*_*[lL]?", "0[0_]*[lL]?")
Intnumber = group(Binnumber, Hexnumber, Octnumber, Decnumber)
Exponent = r"[eE][-+]?\d+(?:_+\d+)*_*"
Pointfloat = group(
    r"\d+(?:_+\d+)*_*\.(?:\d+(?:_+\d+)*_*)?", r"\.\d+(?:_+\d+)*_*"
) + maybe(Exponent)
Expfloat = r"\d+(?:_+\d+)*_*" + Exponent
Floatnumber = group(Pointfloat, Expfloat)
Imagnumber = group(r"\d+(?:_\d+)*[jJ]", Floatnumber + r"[jJ]")
Number = group(Imagnumber, Floatnumber, Intnumber)

# Tail end of ' string.
Single = r"[^'\\]*(?:\\.[^'\\]*)*'"
# Tail end of " string.
Double = r'[^"\\]*(?:\\.[^"\\]*)*"'
# Tail end of ` expr.
Backtick = r"[^`\\]*(?:\\.[^`\\]*)*`"
# Tail end of ''' string.
Single3 = r"[^'\\]*(?:(?:\\.|'(?!''))[^'\\]*)*'''"
# Tail end of """ string.
Double3 = r'[^"\\]*(?:(?:\\.|"(?!""))[^"\\]*)*"""'

# F-string/T-string prefixes for detection
_FSTRING_SINGLE_PREFIXES = (
    'f"',
    "f'",
    't"',
    "t'",
)
_FSTRING_TRIPLE_PREFIXES = (
    'f"""',
    "f'''",
    't"""',
    "t'''",
)


def _is_fstring_or_tstring(token: str, triple_quoted: bool = False) -> bool:
    token_lower = token.lower()
    prefixes = (
        _FSTRING_TRIPLE_PREFIXES if triple_quoted else _FSTRING_SINGLE_PREFIXES
    )
    return token_lower.startswith(prefixes)


def _get_fstring_quote(token: str, triple_quoted: bool = False) -> str:
    """Parse f-string/t-string token to extract quote characters.

    Args:
        token: The token string (e.g., f", t''', etc.)
        triple_quoted: Whether this is a triple-quoted string

    Returns:
        The quote character(s) - single or triple quotes
    """
    # Prefix is always 1 character (f or t)
    if triple_quoted:
        return token[1:4]  # """ or '''
    else:
        return token[1]  # " or '


def scan_fstring_content(s: str, start: int, quote: str) -> int:
    """Scan f-string content handling nested braces and return end position.

    This function properly tracks brace depth to handle nested f-strings
    with same-quote delimiters, e.g., f"{f"{x}"}".

    Args:
        s: The source string to scan
        start: Starting position (after opening quote)
        quote: The quote character(s) to match (single or triple quotes)

    Returns:
        Position after closing quote, or -1 if not found
    """
    i = start
    brace_depth = 0
    quote_len = len(quote)

    while i < len(s):
        # Check for closing quote (only when not inside braces)
        if brace_depth == 0 and s[i : i + quote_len] == quote:
            return i + quote_len

        # Handle escaped characters
        if s[i] == "\\" and i + 1 < len(s):
            i += 2
            continue

        # Track brace depth
        if s[i] == "{":
            # Only treat {{ as escaped when outside expressions (brace_depth == 0)
            if brace_depth == 0 and i + 1 < len(s) and s[i + 1] == "{":
                i += 2  # Escaped {{
                continue
            brace_depth += 1
            i += 1
            continue

        if s[i] == "}":
            if brace_depth > 0:
                brace_depth -= 1
                i += 1
                continue
            # Only treat }} as escaped when outside expressions (brace_depth == 0)
            if i + 1 < len(s) and s[i + 1] == "}":
                i += 2  # Escaped }}
                continue
            i += 1
            continue

        # Inside braces, skip over string literals to avoid false brace matches
        if brace_depth > 0 and s[i] in ('"', "'", "`"):
            # Detect triple-quoted strings
            delim = s[i : i + 3] if s[i : i + 3] in ('"""', "'''") else s[i]
            i += len(delim)
            # Skip to end of string literal
            while i < len(s):
                if s[i : i + len(delim)] == delim:
                    i += len(delim)
                    break
                if s[i] == "\\" and i + 1 < len(s):
                    i += 2
                else:
                    i += 1
            continue

        i += 1

    return -1  # Not found


def _process_fstring_or_tstring(
    token: str,
    line: str,
    start: int,
    triple_quoted: bool,
) -> tuple[str, int, str | None] | None:
    """Process f-string or t-string token and return (token, pos, continuation_quote).

    Returns:
        Tuple of (token, new_pos, continuation_quote) or None
    """
    if not _is_fstring_or_tstring(token, triple_quoted):
        return None

    quote_chars = _get_fstring_quote(token, triple_quoted)
    quote_len = 3 if triple_quoted else 1
    # Prefix is always 1 character (f or t)
    content_start = start + 1 + quote_len

    new_end = scan_fstring_content(line, content_start, quote_chars)

    if new_end > 0:
        # Found on same line
        return (line[start:new_end], new_end, None)
    else:
        # Multiline - needs continuation
        return (line[start:], len(line), quote_chars)


_litprefix = r"(?:[uUrRbBfFtT]|[rR][fFbBtT]|[fFbBuUtT][rR])?"
_fprefix = r"(?:[fFtT]|[fFtT][rR]|[rR][fFtT])"
_non_fprefix = r"(?:[uUrRbB]|[rR][bB]|[bB][rR])?"
Triple = group(_litprefix + "'''", _litprefix + '"""')
# Single-line ' or " string or ` expr.
# F-strings need special handling to allow nested quotes inside {...}
String = group(
    # F-string patterns (must come first for longest match)
    _fprefix + r"'[^\n'\\{]*(?:(?:\\.|(?:\{[^\}]*\}))[^\n'\\{]*)*'",
    _fprefix + r'"[^\n"\\{]*(?:(?:\\.|(?:\{[^\}]*\}))[^\n"\\{]*)*"',
    # Regular string patterns
    _non_fprefix + r"'[^\n'\\]*(?:\\.[^\n'\\]*)*'",
    _non_fprefix + r'"[^\n"\\]*(?:\\.[^\n"\\]*)*"',
    _litprefix + r"`[^\n`\\]*(?:\\.[^\n`\\]*)*`",
)

# Because of leftmost-then-longest match semantics, be sure to put the
# longest operators first (e.g., if = came before ==, == would get
# recognized as two instances of =).
Operator = group(
    r"\*\*=?",
    r">>=?",
    r"<<=?",
    r"<>",
    r"!=",
    r"//=?",
    r"->",
    r"[+\-*/%&@|^=<>:]=?",
    r"~",
)

Bracket = "[][(){}]"
Special = group(r"\r?\n", r"[:;.,@]")
Funny = group(Operator, Bracket, Special)

# First (or only) line of ' or " string.
# F-strings need special patterns to allow nested quotes inside {...}
ContStr = group(
    # F-string patterns
    _fprefix
    + r"'[^\n'\\{]*(?:(?:\\.|(?:\{[^\}]*\}))[^\n'\\{]*)*"
    + group("'", r"\\\r?\n"),
    _fprefix
    + r'"[^\n"\\{]*(?:(?:\\.|(?:\{[^\}]*\}))[^\n"\\{]*)*'
    + group('"', r"\\\r?\n"),
    # Regular string patterns
    _non_fprefix + r"'[^\n'\\]*(?:\\.[^\n'\\]*)*" + group("'", r"\\\r?\n"),
    _non_fprefix + r'"[^\n"\\]*(?:\\.[^\n"\\]*)*' + group('"', r"\\\r?\n"),
    _litprefix + r"`[^\n`\\]*(?:\\.[^\n`\\]*)*" + group("`", r"\\\r?\n"),
)
PseudoExtras = group(r"\\\r?\n", Comment, Triple)
PseudoToken = Whitespace + group(PseudoExtras, Number, Funny, ContStr, Name)

pseudoprog: Final = re.compile(PseudoToken, re.UNICODE)
single3prog = re.compile(Single3)
double3prog = re.compile(Double3)

_strprefixes = (
    _combinations("r", "R", "f", "F")
    | _combinations("r", "R", "t", "T")
    | _combinations("r", "R", "b", "B")
    | {"u", "U", "ur", "uR", "Ur", "UR"}
)

endprogs: Final = {
    "'": re.compile(Single),
    '"': re.compile(Double),
    "`": re.compile(Backtick),
    "'''": single3prog,
    '"""': double3prog,
    # All string prefixes that aren't f/t-strings use regular regex
    **{f"{prefix}'''": single3prog for prefix in _strprefixes},
    **{f'{prefix}"""': double3prog for prefix in _strprefixes},
    **{prefix: None for prefix in _strprefixes},
}

triple_quoted: Final = (
    {"'''", '"""'}
    | {f"{prefix}'''" for prefix in _strprefixes}
    | {f'{prefix}"""' for prefix in _strprefixes}
)
single_quoted: Final = (
    {"'", '"'}
    | {f"{prefix}'" for prefix in _strprefixes}
    | {f'{prefix}"' for prefix in _strprefixes}
)

tabsize = 8


class TokenError(Exception):
    pass


class StopTokenizing(Exception):
    pass


def printtoken(
    type,  # noqa: ANN001
    token,  # noqa: ANN001
    xxx_todo_changeme,  # noqa: ANN001
    xxx_todo_changeme1,  # noqa: ANN001
    line,  # noqa: ANN001
) -> None:  # for testing
    (srow, scol) = xxx_todo_changeme
    (erow, ecol) = xxx_todo_changeme1
    print(
        "%d,%d-%d,%d:\t%s\t%s"  # noqa: UP031
        % (srow, scol, erow, ecol, tok_name[type], repr(token))
    )


Coord = tuple[int, int]
TokenEater = Callable[[int, str, Coord, Coord, str], None]


def tokenize(
    readline: Callable[[], str], tokeneater: TokenEater = printtoken
) -> None:
    """
    The tokenize() function accepts two parameters: one representing the
    input stream, and one providing an output mechanism for tokenize().

    The first parameter, readline, must be a callable object which provides
    the same interface as the readline() method of built-in file objects.
    Each call to the function should return one line of input as a string.

    The second parameter, tokeneater, must also be a callable object. It is
    called once for each token, with five arguments, corresponding to the
    tuples generated by generate_tokens().
    """
    try:
        tokenize_loop(readline, tokeneater)
    except StopTokenizing:
        pass


# backwards compatible interface
def tokenize_loop(readline, tokeneater) -> None:  # noqa: ANN001
    for token_info in generate_tokens(readline):
        tokeneater(*token_info)


GoodTokenInfo = tuple[int, str, Coord, Coord, str]
TokenInfo = tuple[int, str] | GoodTokenInfo


class Untokenizer:
    tokens: list[str]
    prev_row: int
    prev_col: int

    def __init__(self) -> None:
        self.tokens = []
        self.prev_row = 1
        self.prev_col = 0

    def add_whitespace(self, start: Coord) -> None:
        row, col = start
        assert row <= self.prev_row
        col_offset = col - self.prev_col
        if col_offset:
            self.tokens.append(" " * col_offset)

    def untokenize(self, iterable: Iterable[TokenInfo]) -> str:
        for t in iterable:
            if len(t) == 2:
                self.compat(cast(tuple[int, str], t), iterable)
                break
            tok_type, token, start, end, _line = cast(
                tuple[int, str, Coord, Coord, str], t
            )
            self.add_whitespace(start)
            self.tokens.append(token)
            self.prev_row, self.prev_col = end
            if tok_type in (NEWLINE, NL):
                self.prev_row += 1
                self.prev_col = 0
        return "".join(self.tokens)

    def compat(
        self, token: tuple[int, str], iterable: Iterable[TokenInfo]
    ) -> None:
        startline = False
        indents = []
        toks_append = self.tokens.append
        toknum, tokval = token
        if toknum in (NAME, NUMBER):
            tokval += " "
        if toknum in (NEWLINE, NL):
            startline = True
        for tok in iterable:
            toknum, tokval = tok[:2]

            if toknum in (NAME, NUMBER, ASYNC, AWAIT):
                tokval += " "

            if toknum == INDENT:
                indents.append(tokval)
                continue
            elif toknum == DEDENT:
                indents.pop()
                continue
            elif toknum in (NEWLINE, NL):
                startline = True
            elif startline and indents:
                toks_append(indents[-1])
                startline = False
            toks_append(tokval)


cookie_re = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)", re.ASCII)
blank_re = re.compile(rb"^[ \t\f]*(?:[#\r\n]|$)", re.ASCII)


def _get_normal_name(orig_enc: str) -> str:
    """Imitates get_normal_name in tokenizer.c."""
    # Only care about the first 12 characters.
    enc = orig_enc[:12].lower().replace("_", "-")
    if enc == "utf-8" or enc.startswith("utf-8-"):
        return "utf-8"
    if enc in ("latin-1", "iso-8859-1", "iso-latin-1") or enc.startswith(
        ("latin-1-", "iso-8859-1-", "iso-latin-1-")
    ):
        return "iso-8859-1"
    return orig_enc


def detect_encoding(readline: Callable[[], bytes]) -> tuple[str, list[bytes]]:
    """
    The detect_encoding() function is used to detect the encoding that should
    be used to decode a Python source file. It requires one argument, readline,
    in the same way as the tokenize() generator.

    It will call readline a maximum of twice, and return the encoding used
    (as a string) and a list of any lines (left as bytes) it has read
    in.

    It detects the encoding from the presence of a utf-8 bom or an encoding
    cookie as specified in pep-0263. If both a bom and a cookie are present, but
    disagree, a SyntaxError will be raised. If the encoding cookie is an invalid
    charset, raise a SyntaxError.  Note that if a utf-8 bom is found,
    'utf-8-sig' is returned.

    If no encoding is specified, then the default of 'utf-8' will be returned.
    """
    bom_found = False
    encoding = None
    default = "utf-8"

    def read_or_stop() -> bytes:
        try:
            return readline()
        except StopIteration:
            return b""

    def find_cookie(line: bytes) -> str | None:
        try:
            line_string = line.decode("ascii")
        except UnicodeDecodeError:
            return None
        match = cookie_re.match(line_string)
        if not match:
            return None
        encoding = _get_normal_name(match.group(1))
        try:
            codec = lookup(encoding)
        except LookupError:
            # This behaviour mimics the Python interpreter
            raise SyntaxError("unknown encoding: " + encoding)  # noqa: B904

        if bom_found:
            if codec.name != "utf-8":
                # This behaviour mimics the Python interpreter
                raise SyntaxError("encoding problem: utf-8")
            encoding += "-sig"
        return encoding

    first = read_or_stop()
    if first.startswith(BOM_UTF8):
        bom_found = True
        first = first[3:]
        default = "utf-8-sig"
    if not first:
        return default, []

    encoding = find_cookie(first)
    if encoding:
        return encoding, [first]
    if not blank_re.match(first):
        return default, [first]

    second = read_or_stop()
    if not second:
        return default, [first]

    encoding = find_cookie(second)
    if encoding:
        return encoding, [first, second]

    return default, [first, second]


def untokenize(iterable: Iterable[TokenInfo]) -> str:
    """Transform tokens back into Python source code.

    Each element returned by the iterable must be a token sequence
    with at least two elements, a token number and token value.  If
    only two tokens are passed, the resulting output is poor.

    Round-trip invariant for full input:
        Untokenized source will match input source exactly

    Round-trip invariant for limited input:
        # Output text will tokenize the back to the input
        t1 = [tok[:2] for tok in generate_tokens(f.readline)]
        newcode = untokenize(t1)
        readline = iter(newcode.splitlines(1)).next
        t2 = [tok[:2] for tokin generate_tokens(readline)]
        assert t1 == t2
    """
    ut = Untokenizer()
    return ut.untokenize(iterable)


def generate_tokens(
    readline: Callable[[], str], grammar: Grammar | None = None
) -> Iterator[GoodTokenInfo]:
    """
    The generate_tokens() generator requires one argument, readline, which
    must be a callable object which provides the same interface as the
    readline() method of built-in file objects. Each call to the function
    should return one line of input as a string.  Alternately, readline
    can be a callable function terminating with StopIteration:
        readline = open(myfile).next    # Example of alternate readline

    The generator produces 5-tuples with these members: the token type; the
    token string; a 2-tuple (srow, scol) of ints specifying the row and
    column where the token begins in the source; a 2-tuple (erow, ecol) of
    ints specifying the row and column where the token ends in the source;
    and the line on which the token was found. The line passed is the
    logical line; continuation lines are included.
    """
    lnum = parenlev = continued = 0
    numchars: Final = "0123456789"
    contstr, needcont = "", 0
    contline: str | None = None
    contstr_fstring_quote: str | None = (
        None  # Track f-string quote for continuation
    )
    indents = [0]

    # If we know we're parsing 3.7+, we can unconditionally parse `async` and
    # `await` as keywords.
    async_keywords = False if grammar is None else grammar.async_keywords
    # 'stashed' and 'async_*' are used for async/await parsing
    stashed: GoodTokenInfo | None = None
    async_def = False
    async_def_indent = 0
    async_def_nl = False
    # If we know we're parsing Mojo, we can unconditionally parse various
    # identifiers, like `struct`, as keywords.
    has_mojo_keywords = False if grammar is None else grammar.mojo_keywords
    # Tracks the value of the last meaningful (non-whitespace) token emitted,
    # used to decide whether a Mojo keyword token should instead be treated as
    # an ordinary NAME (e.g. `def struct(...)` where `struct` is the function name).
    prev_token_value = None
    def_keywords = ("def", "__mlir_region") if has_mojo_keywords else ("def",)
    # NOTE: If extending also update any lists of keywords in the testsuite.
    mojo_keyword_tokens = {
        "struct": STRUCT,
        "comptime": COMPTIME,
        "var": VAR,
        "__mlir_region": MLIR_REGION,
        "__generator_type": GENERATOR_TYPE,
        "read": READ,
        "imm": IMM,
        "mut": MUT,
        "out": OUT,
        "deinit": DEINIT,
        "trait": TRAIT,
        "ref": REF,
        "where": WHERE,
        "__extension": EXTENSION,
    }

    strstart: tuple[int, int]
    endprog: Pattern[str]

    while 1:  # loop over lines in stream
        try:
            line = readline()
        except StopIteration:
            line = ""
        lnum += 1
        pos, max = 0, len(line)

        if contstr:  # continued string
            assert contline is not None
            if not line:
                raise TokenError("EOF in multi-line string", strstart)

            # Check if this is a continued f-string that needs stateful scanning
            if contstr_fstring_quote is not None:
                # Use stateful scanner for f-string continuation
                end = scan_fstring_content(line, 0, contstr_fstring_quote)
                if end > 0:
                    # Found the end
                    pos = end
                    yield (
                        STRING,
                        contstr + line[:end],
                        strstart,
                        (lnum, end),
                        contline + line,
                    )
                    contstr, needcont = "", 0
                    contline = None
                    contstr_fstring_quote = None
                else:
                    # Still continuing
                    contstr = contstr + line
                    contline = contline + line
                    continue
            else:
                # Regular string continuation (not f-string)
                endmatch = endprog.match(line)
                if endmatch:
                    pos = end = endmatch.end(0)
                    yield (
                        STRING,
                        contstr + line[:end],
                        strstart,
                        (lnum, end),
                        contline + line,
                    )
                    contstr, needcont = "", 0
                    contline = None
                elif needcont and line[-2:] != "\\\n" and line[-3:] != "\\\r\n":
                    yield (
                        ERRORTOKEN,
                        contstr + line,
                        strstart,
                        (lnum, len(line)),
                        contline,
                    )
                    contstr = ""
                    contline = None
                    continue
                else:
                    contstr = contstr + line
                    contline = contline + line
                    continue

        elif parenlev == 0 and not continued:  # new statement
            if not line:
                break
            column = 0
            while pos < max:  # measure leading whitespace
                if line[pos] == " ":
                    column += 1
                elif line[pos] == "\t":
                    column = (column // tabsize + 1) * tabsize
                elif line[pos] == "\f":
                    column = 0
                else:
                    break
                pos += 1
            if pos == max:
                break

            if stashed:
                yield stashed
                stashed = None

            if line[pos] in "\r\n":  # skip blank lines
                yield (NL, line[pos:], (lnum, pos), (lnum, len(line)), line)
                continue

            if line[pos] == "#":  # skip comments
                comment_token = line[pos:].rstrip("\r\n")
                nl_pos = pos + len(comment_token)
                yield (
                    COMMENT,
                    comment_token,
                    (lnum, pos),
                    (lnum, nl_pos),
                    line,
                )
                yield (
                    NL,
                    line[nl_pos:],
                    (lnum, nl_pos),
                    (lnum, len(line)),
                    line,
                )
                continue

            if column > indents[-1]:  # count indents
                indents.append(column)
                yield (INDENT, line[:pos], (lnum, 0), (lnum, pos), line)

            while column < indents[-1]:  # count dedents
                if column not in indents:
                    raise IndentationError(
                        "unindent does not match any outer indentation level",
                        ("<tokenize>", lnum, pos, line),
                    )
                indents = indents[:-1]

                if async_def and async_def_indent >= indents[-1]:
                    async_def = False
                    async_def_nl = False
                    async_def_indent = 0

                yield (DEDENT, "", (lnum, pos), (lnum, pos), line)

            if async_def and async_def_nl and async_def_indent >= indents[-1]:
                async_def = False
                async_def_nl = False
                async_def_indent = 0

        else:  # continued statement
            if not line:
                raise TokenError("EOF in multi-line statement", (lnum, 0))
            continued = 0

        # Given an identifier matching a Mojo token, do extra context sensitive
        # checks for validity.  This is because we can't get things like 'out'
        # handled properly as soft tokens.  This returns true if this can be
        # handled as a normal Mojo token.
        def check_mojo_token(token_value: str, token_end: int):  # noqa: ANN202
            assert grammar is not None
            # In Mojo, keywords can be used as function/struct names.
            # After a def-like keyword or '.', treat the token as a
            # plain NAME so `def fn(...)` or `x.struct` parse correctly.
            if prev_token_value in grammar.declaration_keywords + ["."]:  # noqa: B023
                return False
            # Context sensitive arg conventions are only a keyword if followed
            # by an identifier letter or a variadic.
            if token_value not in ["out", "read", "imm", "mut", "deinit"]:
                return True
            next_token = line[token_end:].lstrip()  # noqa: B023
            if token_value == "out" and next_token.startswith("["):
                bracket_depth = 0
                for idx, char in enumerate(next_token):
                    if char == "[":
                        bracket_depth += 1
                    elif char == "]":
                        bracket_depth -= 1
                        if bracket_depth == 0:
                            after_bracket = next_token[idx + 1 :].lstrip()
                            return (
                                bool(after_bracket)
                                and after_bracket[0].isidentifier()
                            )
                return False
            return next_token and (
                next_token[0].isidentifier() or next_token[0] == "*"
            )

        while pos < max:
            pseudomatch = pseudoprog.match(line, pos)
            if pseudomatch:  # scan for tokens
                start, end = pseudomatch.span(1)
                spos, epos, pos = (lnum, start), (lnum, end), end
                token, initial = line[start:end], line[start]

                if initial in numchars or (
                    initial == "." and token != "."
                ):  # ordinary number
                    yield (NUMBER, token, spos, epos, line)
                elif initial in "\r\n":
                    newline = NEWLINE
                    if parenlev > 0:
                        newline = NL
                    elif async_def:
                        async_def_nl = True
                    if stashed:
                        yield stashed
                        stashed = None
                    if newline == NEWLINE:
                        prev_token_value = None
                    yield (newline, token, spos, epos, line)

                elif initial == "#":
                    assert not token.endswith("\n")
                    if stashed:
                        yield stashed
                        stashed = None
                    yield (COMMENT, token, spos, epos, line)
                elif token in triple_quoted:
                    # Try processing as f-string/t-string
                    result = _process_fstring_or_tstring(
                        token, line, start, triple_quoted=True
                    )
                    if result:
                        token, pos, quote_chars = result
                        if quote_chars is None:
                            # Found on same line
                            if stashed:
                                yield stashed
                                stashed = None
                            yield (STRING, token, spos, (lnum, pos), line)
                        else:
                            # Multi-line f-string/t-string
                            strstart = (lnum, start)
                            contstr = token
                            contline = line
                            contstr_fstring_quote = quote_chars
                            break
                    else:
                        # Regular triple-quoted string (not f-string/t-string)
                        endprog = endprogs[token]  # type: ignore
                        endmatch = endprog.match(line, pos)
                        if endmatch:  # all on one line
                            pos = endmatch.end(0)
                            token = line[start:pos]
                            if stashed:
                                yield stashed
                                stashed = None
                            yield (STRING, token, spos, (lnum, pos), line)
                        else:
                            strstart = (lnum, start)  # multiple lines
                            contstr = line[start:]
                            contline = line
                            break
                elif (
                    initial in single_quoted
                    or token[:2] in single_quoted
                    or token[:3] in single_quoted
                ):
                    # Try processing as f-string/t-string
                    if token[-1] != "\n":
                        result = _process_fstring_or_tstring(
                            token, line, start, triple_quoted=False
                        )
                        if result:
                            token, pos, quote_char = result
                            if quote_char is None:
                                # Found on same line
                                if stashed:
                                    yield stashed
                                    stashed = None
                                yield (STRING, token, spos, (lnum, pos), line)
                                continue
                            else:
                                # Multi-line f-string/t-string
                                strstart = (lnum, start)
                                contstr = token
                                contline = line
                                contstr_fstring_quote = quote_char
                                break

                    if token[-1] == "\n":  # continued string
                        strstart = (lnum, start)
                        endprog = (
                            endprogs[initial]  # type: ignore
                            or endprogs[token[1]]
                            or endprogs[token[2]]
                        )
                        contstr, needcont = line[start:], 1
                        contline = line
                        break
                    else:  # ordinary string
                        if stashed:
                            yield stashed
                            stashed = None
                        yield (STRING, token, spos, epos, line)
                elif token.startswith("`"):
                    endprog = endprogs["`"]  # type: ignore
                    endmatch = endprog.match(line, pos)
                    yield (NAME, token, spos, epos, line)
                elif initial.isidentifier():  # ordinary name
                    if (
                        has_mojo_keywords
                        and token in mojo_keyword_tokens
                        and check_mojo_token(token, end)
                    ):
                        tok_type = mojo_keyword_tokens[token]
                        # comptime followed by '(' is the expression form
                        # comptime(expr) — emit as NAME so it parses as a
                        # regular function call.
                        if tok_type == COMPTIME:
                            next_chars = line[end:].lstrip()
                            if next_chars and next_chars[0] == "(":
                                tok_type = NAME
                        prev_token_value = token
                        yield (tok_type, token, spos, epos, line)
                        continue

                    if token in ("async", "await"):
                        if async_keywords or async_def:
                            yield (
                                ASYNC if token == "async" else AWAIT,
                                token,
                                spos,
                                epos,
                                line,
                            )
                            continue

                    tok = (NAME, token, spos, epos, line)
                    if token == "async" and not stashed:
                        stashed = tok
                        continue

                    if token == "for" or token in def_keywords:
                        if (
                            stashed
                            and stashed[0] == NAME
                            and stashed[1] == "async"
                        ):
                            if token in def_keywords:
                                async_def = True
                                async_def_indent = indents[-1]

                            yield (
                                ASYNC,
                                stashed[1],
                                stashed[2],
                                stashed[3],
                                stashed[4],
                            )
                            stashed = None

                    if stashed:
                        prev_token_value = stashed[1]
                        yield stashed
                        stashed = None

                    prev_token_value = token
                    yield tok
                elif initial == "\\":  # continued stmt
                    # This yield is new; needed for better idempotency:
                    if stashed:
                        yield stashed
                        stashed = None
                    yield (NL, token, spos, (lnum, pos), line)
                    continued = 1
                else:
                    if initial in "([{":
                        parenlev += 1
                    elif initial in ")]}":
                        parenlev -= 1
                    if stashed:
                        prev_token_value = stashed[1]
                        yield stashed
                        stashed = None
                    prev_token_value = token
                    yield (OP, token, spos, epos, line)
            else:
                yield (
                    ERRORTOKEN,
                    line[pos],
                    (lnum, pos),
                    (lnum, pos + 1),
                    line,
                )
                pos += 1

    if stashed:
        yield stashed
        stashed = None

    for _indent in indents[1:]:  # pop remaining indent levels
        yield (DEDENT, "", (lnum, 0), (lnum, 0), "")
    yield (ENDMARKER, "", (lnum, 0), (lnum, 0), "")


if __name__ == "__main__":  # testing
    import sys

    if len(sys.argv) > 1:
        tokenize(open(sys.argv[1]).readline)
    else:
        tokenize(sys.stdin.readline)
