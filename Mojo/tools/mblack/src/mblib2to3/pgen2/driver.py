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
#   Path:   src/mblib2to3/pgen2/driver.py
#
# ===----------------------------------------------------------------------=== #

# Copyright 2004-2005 Elemental Security, Inc. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

# Modifications:
# Copyright 2006 Google, Inc. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Parser driver.

This provides a high-level interface to parse a file into a syntax tree.

"""

__author__ = "Guido van Rossum <guido@python.org>"

__all__ = ["Driver", "load_grammar"]

# Python imports
import io
import logging
import os
import pkgutil
import re
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import Logger
from typing import (
    IO,
    Any,
    Union,
    cast,
)

from mblib2to3.pgen2.grammar import Grammar
from mblib2to3.pgen2.tokenize import GoodTokenInfo
from mblib2to3.pytree import NL

# Pgen imports
from . import grammar, parse, pgen, token, tokenize

Path = Union[str, "os.PathLike[str]"]

_COLON_ONLY_LINE = re.compile(r"^\s+:\s*$")


def _join_colon_next_line(prev: str, line: str, head_indent: int) -> str | None:
    """Move a standalone colon from its own indented line to the previous line.

    *head_indent* is unused here (a lone colon needs no indent test) but is
    accepted so every normalizer shares one signature.

    In Mojo, this is valid syntax::

        def foo()  # comment
            :
            pass

    but the lib2to3-based parser requires the colon on the same line as
    the signature.
    """
    if _COLON_ONLY_LINE.match(line):
        return prev + ":"
    return None


def _starts_with_keyword(line: str, keyword: str) -> bool:
    """Check that *line* starts with *keyword* as a whole word, not a prefix
    of a longer identifier (e.g. ``if`` but not ``ifvar``)."""
    if not line.startswith(keyword):
        return False
    rest = line[len(keyword) :]
    return bool(rest) and not (rest[0].isalnum() or rest[0] == "_")


def _join_multiline_ternary(
    prev: str, line: str, head_indent: int
) -> str | None:
    """Join ternary continuations that appear on an indented next line.

    In Mojo, these are valid syntax::

        var a = 10
           if b % 2 else 100

        var a = 10 if True
           else 100

    but the lib2to3-based parser requires the ternary on the same logical
    line.
    """
    stripped = line.lstrip()
    # Distinguish ternary ``if`` from a statement ``if``: a statement
    # always ends with ``:`` (ignoring trailing comments).
    code, _ = _split_trailing_comment(stripped)
    is_continuation = (
        (_starts_with_keyword(stripped, "if") and not code.endswith(":"))
        or (_starts_with_keyword(stripped, "else") and not code.endswith(":"))
        or stripped == "else"
    )
    if not is_continuation:
        return None
    cur_indent = len(line) - len(line.lstrip())
    if (
        prev
        and not prev.endswith(":")
        and not prev.endswith("\\")
        and not prev.lstrip().startswith("#")
        and cur_indent > head_indent
    ):
        return prev + " " + stripped
    return None


# Matches a line that ends with a binary or assignment operator.  Every
# multi-char symbol operator (``==``, ``+=``, ``>>=``, …) ends with one of
# these characters, so matching the last character is sufficient.
_TRAILING_OP_RE = re.compile(r"[-+*/%&|^<>=]\s*$|\b(?:and|or)\s*$")


def _join_trailing_operator(
    prev: str, line: str, head_indent: int
) -> str | None:
    """Join a continuation line when the previous line ends with an operator.

    In Mojo, implicit line continuation is allowed when a line ends with
    a binary or assignment operator::

        var result = a == b and
                     c == d
        x[0] =
            x[0] + 1

    The lib2to3-based parser requires these on the same logical line.
    """
    cur_indent = len(line) - len(line.lstrip())
    if cur_indent > head_indent and _TRAILING_OP_RE.search(prev):
        return prev + " " + line.lstrip()
    return None


# Matches a line whose first non-whitespace character starts a string literal.
_LEADING_STRING = re.compile(r"""^\s+["']""")


def _join_string_continuation(
    prev: str, line: str, head_indent: int
) -> str | None:
    """Join an implicit string concatenation from a continuation line.

    In Mojo, adjacent string literals on a further-indented continuation
    line are concatenated::

        var s = "hello"
                 " world"

    The lib2to3-based parser requires them on the same logical line.
    Only joins when the previous line ends with a closing quote — this
    avoids swallowing docstrings or standalone string literals.
    """
    if not _LEADING_STRING.match(line):
        return None
    if not prev.rstrip() or prev.rstrip()[-1] not in ('"', "'"):
        return None
    if len(line) - len(line.lstrip()) > head_indent:
        return prev + " " + line.lstrip()
    return None


# Matches a line whose first non-whitespace token is a dot followed by an
# identifier (method chain).  The identifier requirement excludes ``...``
# (Mojo's ellipsis / pass equivalent).
_LEADING_DOT = re.compile(r"^\s+\.[A-Za-z_]")


def _join_dot_continuation(
    prev: str, line: str, head_indent: int
) -> str | None:
    """Join a dot-prefixed method call from a continuation line.

    In Mojo, method chains can start on a further-indented continuation
    line::

        var text = String("hello")
               .upper()

    The lib2to3-based parser requires them on the same logical line.
    """
    if not _LEADING_DOT.match(line):
        return None
    cur_indent = len(line) - len(line.lstrip())
    if (
        prev
        and cur_indent > head_indent
        and not prev.endswith(":")
        and not prev.endswith("\\")
        and not prev.lstrip().startswith("#")
    ):
        return prev + line.lstrip()
    return None


# Normalizers are applied in order; the *first* one that returns a
# non-None replacement wins and the rest are skipped.  Each receives the
# previous result line (with any trailing comment already stripped), the
# current raw line, and the indent of the current logical statement's head
# (so continuations compare against the head, not an interior line such as
# the closing bracket of a multi-line operand).
_MOJO_LINE_NORMALIZERS = [
    _join_colon_next_line,
    _join_multiline_ternary,
    _join_trailing_operator,
    _join_string_continuation,
    _join_dot_continuation,
]


def _split_trailing_comment(line: str) -> tuple[str, str]:
    """Split *line* into ``(code, comment)`` around the first ``#`` outside a
    string literal.

    *comment* includes any whitespace between the code and ``#``.  String
    literals are skipped whole (see `_skip_string`), so a ``#`` inside one
    never starts a comment.  Returns ``(line, "")`` when there is none.
    """
    if "#" not in line:  # fast path: nothing to split
        return line, ""
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if c == "#":
            code = line[:i].rstrip()
            return code, line[len(code) :]
        elif c in "\"'`":
            i = _skip_string_at(line, i)
            if i == -1:
                return line, ""
        else:
            i += 1
    return line, ""


def _try_merge_line(result: list[str], line: str, head_indent: int) -> bool:
    """Try to merge *line* into the last element of *result*.

    Strips the trailing comment from the previous result line before
    calling each normalizer, then reattaches it so normalizers only
    see code.  *head_indent* is the indent of the current logical
    statement's head.  Returns ``True`` if any normalizer matched.
    """
    prev_code, comment = _split_trailing_comment(result[-1])
    merged = False
    for normalizer in _MOJO_LINE_NORMALIZERS:
        replacement = normalizer(prev_code, line, head_indent)
        if replacement is not None:
            prev_code = replacement
            merged = True
            break
    if merged:
        result[-1] = prev_code + comment
    return merged


def _opens_interp(line: str, end: int) -> bool:
    """Whether the string opening at *end* is an interpolating f/t-string.

    Only a valid prefix counts (<=2 of ``r``/``f``/``t``), so a keyword left
    flush against a quote (``if"..."``) is not read as an f/t-string.
    """
    start = end
    while start > 0 and line[start - 1].isalpha():
        start -= 1
    prefix = line[start:end]
    return (
        len(prefix) <= 2
        and all(c in "rftRFT" for c in prefix)
        and any(c in "ftFT" for c in prefix)
    )


def _quote_at(line: str, i: int) -> str:
    """The string delimiter opening at *i* -- a triple or single quote, or a
    backtick.

    Mojo backticks quote raw identifiers / MLIR attributes, e.g.
    ``__mlir_attr.`[#llvm.foo]```. Treating them as (non-interpolating) strings
    keeps a ``#`` or bracket inside from desyncing the bracket depth.
    """
    return line[i : i + 3] if line[i : i + 3] in ('"""', "'''") else line[i]


def _skip_string_at(line: str, i: int) -> int:
    """Skip the string literal whose opening quote is at *i*. Return the
    index past the close, or ``-1`` if it does not close on this line."""
    q = _quote_at(line, i)
    return _skip_string(line, i + len(q), q, _opens_interp(line, i))


def _skip_string(line: str, i: int, quote: str, is_interp: bool) -> int:
    """Skip a string; *i* starts just past the opening *quote*. Return the
    index past the close, or ``-1`` if it does not close on this line.

    In an f/t string, ``{...}`` interpolations are skipped as code (see
    `_skip_interp`) so a nested same-quote string is not read as the close.
    """
    n = len(line)
    while i < n:
        if line[i] == "\\":
            i += 2  # escaped char, a \" never counts toward the close
        elif line.startswith(quote, i):
            return i + len(quote)
        elif is_interp and (
            line.startswith("{{", i) or line.startswith("}}", i)
        ):
            i += 2  # escaped brace, not interpolation
        elif is_interp and line[i] == "{":
            i = _skip_interp(line, i + 1)
            if i == -1:
                return -1
        else:
            i += 1
    return -1


def _skip_interp(line: str, i: int) -> int:
    """Skip a ``{...}`` interpolation (code body); *i* starts just past the
    ``{``. Return the index past the matching ``}``, else ``-1``.

    Nested strings and ``{...}`` (dict/set literals) are skipped recursively.
    """
    n = len(line)
    while i < n:
        c = line[i]
        if c == "}":
            return i + 1
        elif c in "\"'`":
            i = _skip_string_at(line, i)
            if i == -1:
                return -1
        elif c == "{":
            i = _skip_interp(line, i + 1)  # nested dict/set literal
            if i == -1:
                return -1
        else:
            i += 1
    return -1


def _scan_code_brackets(line: str, quote: str | None) -> tuple[int, str | None]:
    """Return ``(bracket_delta, quote)`` for *line*, counting ``()[]{}`` in
    code only.

    String literals contribute nothing, each is skipped whole so a bracket or
    a nested same-quote string like ``t"{"("}"`` never desyncs depth. *quote*
    is an open triple-quoted string carried in from the previous line, or
    ``None``. The return carries a still-open one forward. Interpolation state
    is not carried across lines.
    """
    delta = 0
    i, n = 0, len(line)

    # Finish a triple-quoted string carried from the previous line.
    if quote:
        i = _skip_string(line, 0, quote, False)
        if i == -1:
            return delta, quote

    while i < n:
        c = line[i]
        if c == "#":
            break  # comment to end of line
        elif c in "([{":
            delta += 1
            i += 1
        elif c in ")]}":
            delta -= 1
            i += 1
        elif c in "\"'`":
            end = _skip_string_at(line, i)
            if end == -1:
                # Unterminated: only a triple-quoted string carries over.
                q = _quote_at(line, i)
                return delta, (q if len(q) == 3 else None)
            i = end
        else:
            i += 1

    return delta, None


def _normalize_mojo_source(text: str) -> str:
    """Apply Mojo-specific line normalizations before tokenization.

    Iterates over the lines once, skipping triple-quoted string regions,
    ``# fmt: off`` regions, and lines inside open brackets, then gives
    each normalizer a chance to merge the current line into the previous
    one.

    In an ideal architecture normalization would not be required. In
    reality the grammar strongly assumes that new lines end statements.
    """
    from mblack.comments import FMT_OFF, FMT_ON

    lines = text.split("\n")
    result: list[str] = []
    ml_quote: str | None = None
    in_fmt_off = False
    bracket_depth = 0
    head_indent = 0

    for line in lines:
        stripped = line.lstrip()

        # Track ``# fmt: off`` / ``# fmt: on`` regions.
        if stripped in FMT_OFF:
            in_fmt_off = True
        elif stripped in FMT_ON:
            in_fmt_off = False

        at_statement_level = (
            bracket_depth == 0 and ml_quote is None and not in_fmt_off
        )
        # Try to merge this line into the previous one. Needs a previous
        # line to merge into.
        can_normalize = at_statement_level and result
        if not (can_normalize and _try_merge_line(result, line, head_indent)):
            result.append(line)
            # A line appended at bracket depth 0 (outside any string or
            # ``# fmt: off`` region) begins a new logical statement; record
            # its indent so later continuation lines compare against the
            # statement head rather than an interior line such as the closing
            # bracket of a multi-line operand.
            if at_statement_level:
                head_indent = len(line) - len(stripped)

        # Update the code-bracket depth for the next line. Brackets inside
        # strings and comments are ignored, so prose never disables rejoining.
        delta, ml_quote = _scan_code_brackets(line, ml_quote)
        bracket_depth += delta

    return "\n".join(result)


@dataclass
class ReleaseRange:
    start: int
    end: int | None = None
    tokens: list[Any] = field(default_factory=list)

    def lock(self) -> None:
        total_eaten = len(self.tokens)
        self.end = self.start + total_eaten


class TokenProxy:
    def __init__(self, generator: Any) -> None:
        self._tokens = generator
        self._counter = 0
        self._release_ranges: list[ReleaseRange] = []

    @contextmanager
    def release(self) -> Iterator["TokenProxy"]:
        release_range = ReleaseRange(self._counter)
        self._release_ranges.append(release_range)
        try:
            yield self
        finally:
            # Lock the last release range to the final position that
            # has been eaten.
            release_range.lock()

    def eat(self, point: int) -> Any:
        eaten_tokens = self._release_ranges[-1].tokens
        if point < len(eaten_tokens):
            return eaten_tokens[point]
        else:
            while point >= len(eaten_tokens):
                token = next(self._tokens)
                eaten_tokens.append(token)
            return token

    def __iter__(self) -> "TokenProxy":
        return self

    def __next__(self) -> Any:
        # If the current position is already compromised (looked up)
        # return the eaten token, if not just go further on the given
        # token producer.
        for release_range in self._release_ranges:
            assert release_range.end is not None

            start, end = release_range.start, release_range.end
            if start <= self._counter < end:
                token = release_range.tokens[self._counter - start]
                break
        else:
            token = next(self._tokens)
        self._counter += 1
        return token

    def can_advance(self, to: int) -> bool:
        # Try to eat, fail if it can't. The eat operation is cached
        # so there wont be any additional cost of eating here
        try:
            self.eat(to)
        except StopIteration:
            return False
        else:
            return True


class Driver:
    def __init__(self, grammar: Grammar, logger: Logger | None = None) -> None:
        self.grammar = grammar
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def parse_tokens(
        self, tokens: Iterable[GoodTokenInfo], debug: bool = False
    ) -> NL:
        """Parse a series of tokens and return the syntax tree."""
        # XXX Move the prefix computation into a wrapper around tokenize.
        proxy = TokenProxy(tokens)

        p = parse.Parser(self.grammar)
        p.setup(proxy=proxy)

        lineno = 1
        column = 0
        indent_columns: list[int] = []
        type = value = start = end = line_text = None
        prefix = ""

        for quintuple in proxy:
            type, value, start, end, line_text = quintuple
            if start != (lineno, column):
                assert (lineno, column) <= start, ((lineno, column), start)
                s_lineno, s_column = start
                if lineno < s_lineno:
                    prefix += "\n" * (s_lineno - lineno)
                    lineno = s_lineno
                    column = 0
                if column < s_column:
                    prefix += line_text[column:s_column]
                    column = s_column
            if type in (tokenize.COMMENT, tokenize.NL):
                prefix += value
                lineno, column = end
                if value.endswith("\n"):
                    lineno += 1
                    column = 0
                continue
            if type == token.OP:
                type = grammar.opmap[value]
            if debug:
                assert type is not None
                self.logger.debug(
                    "%s %r (prefix=%r)", token.tok_name[type], value, prefix
                )
            if type == token.INDENT:
                indent_columns.append(len(value))
                _prefix = prefix + value
                prefix = ""
                value = ""
            elif type == token.DEDENT:
                _indent_col = indent_columns.pop()
                prefix, _prefix = self._partially_consume_prefix(
                    prefix, _indent_col
                )
            if p.addtoken(cast(int, type), value, (prefix, start)):
                if debug:
                    self.logger.debug("Stop.")
                break
            prefix = ""
            if type in {token.INDENT, token.DEDENT}:
                prefix = _prefix
            lineno, column = end
            if value.endswith("\n"):
                lineno += 1
                column = 0
        else:
            # We never broke out -- EOF is too soon (how can this happen???)
            assert start is not None
            raise parse.ParseError(
                "incomplete input", type, value, (prefix, start)
            )
        assert p.rootnode is not None
        return p.rootnode

    def parse_stream_raw(self, stream: IO[str], debug: bool = False) -> NL:
        """Parse a stream and return the syntax tree."""
        tokens = tokenize.generate_tokens(stream.readline, grammar=self.grammar)
        return self.parse_tokens(tokens, debug)

    def parse_stream(self, stream: IO[str], debug: bool = False) -> NL:
        """Parse a stream and return the syntax tree."""
        return self.parse_stream_raw(stream, debug)

    def parse_file(
        self,
        filename: Path,
        encoding: str | None = None,
        debug: bool = False,
    ) -> NL:
        """Parse a file and return the syntax tree."""
        with open(filename, encoding=encoding) as stream:
            return self.parse_stream(stream, debug)

    def parse_string(self, text: str, debug: bool = False) -> NL:
        """Parse a string and return the syntax tree."""
        if self.grammar.mojo_keywords:
            text = _normalize_mojo_source(text)
        tokens = tokenize.generate_tokens(
            io.StringIO(text).readline, grammar=self.grammar
        )
        return self.parse_tokens(tokens, debug)

    def _partially_consume_prefix(
        self, prefix: str, column: int
    ) -> tuple[str, str]:
        lines: list[str] = []
        current_line = ""
        current_column = 0
        wait_for_nl = False
        for char in prefix:
            current_line += char
            if wait_for_nl:
                if char == "\n":
                    if current_line.strip() and current_column < column:
                        res = "".join(lines)
                        return res, prefix[len(res) :]

                    lines.append(current_line)
                    current_line = ""
                    current_column = 0
                    wait_for_nl = False
            elif char in " \t":
                current_column += 1
            elif char == "\n":
                # unexpected empty line
                current_column = 0
            else:
                # indent is finished
                wait_for_nl = True
        return "".join(lines), current_line


def _generate_pickle_name(gt: Path, cache_dir: Path | None = None) -> str:
    head, tail = os.path.splitext(gt)
    if tail == ".txt":
        tail = ""
    name = head + tail + ".".join(map(str, sys.version_info)) + ".pickle"
    if cache_dir:
        return os.path.join(cache_dir, os.path.basename(name))
    else:
        return name


def load_grammar(
    gt: str = "Grammar.txt",
    gp: str | None = None,
    save: bool = True,
    force: bool = False,
    logger: Logger | None = None,
) -> Grammar:
    """Load the grammar (maybe from a pickle)."""
    if logger is None:
        logger = logging.getLogger(__name__)
    gp = _generate_pickle_name(gt) if gp is None else gp
    if force or not _newer(gp, gt):
        g: grammar.Grammar = pgen.generate_grammar(gt)
        if save:
            try:
                g.dump(gp)
            except OSError:
                # Ignore error, caching is not vital.
                pass
    else:
        g = grammar.Grammar()
        g.load(gp)
    return g


def _newer(a: str, b: str) -> bool:
    """Inquire whether file a was written since file b.

    Uses max(mtime, ctime) to handle the case where package installers
    preserve the original build-time mtime when extracting files. The
    ctime (inode change time) reflects the actual installation time.
    """
    if not os.path.exists(a):
        return False
    if not os.path.exists(b):
        return True
    a_time = max(os.path.getmtime(a), os.path.getctime(a))
    b_time = max(os.path.getmtime(b), os.path.getctime(b))
    return a_time > b_time


def load_packaged_grammar(
    package: str, grammar_source: str, cache_dir: Path | None = None
) -> grammar.Grammar:
    """Normally, loads a pickled grammar by doing
        pkgutil.get_data(package, pickled_grammar)
    where *pickled_grammar* is computed from *grammar_source* by adding the
    Python version and using a ``.pickle`` extension.

    However, if *grammar_source* is an extant file, load_grammar(grammar_source)
    is called instead. This facilitates using a packaged grammar file when needed
    but preserves load_grammar's automatic regeneration behavior when possible.

    """
    if os.path.isfile(grammar_source):
        gp = (
            _generate_pickle_name(grammar_source, cache_dir)
            if cache_dir
            else None
        )
        # Fix MOTO-1264: Force grammar regeneration to prevent stale cache issues
        # Always force regeneration when no cache directory is specified (development mode)
        force_regen = gp is None

        # Clean up any existing pickle files in source directory to force regeneration
        if gp is None:
            default_pickle = _generate_pickle_name(grammar_source)
            if os.path.exists(default_pickle):
                try:
                    os.remove(default_pickle)
                except OSError:
                    pass  # Ignore errors if we can't remove it

        return load_grammar(grammar_source, gp=gp, force=force_regen)
    pickled_name = _generate_pickle_name(
        os.path.basename(grammar_source), cache_dir
    )
    data = pkgutil.get_data(package, pickled_name)
    assert data is not None
    g = grammar.Grammar()
    g.loads(data)
    return g


def main(*args: str) -> bool:
    """Main program, when run as a script: produce grammar pickle files.

    Calls load_grammar for each argument, a path to a grammar text file.
    """
    if not args:
        args = tuple(sys.argv[1:])
    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout, format="%(message)s"
    )
    for gt in args:
        load_grammar(gt, save=True, force=True)
    return True


if __name__ == "__main__":
    sys.exit(int(not main()))
