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
#   Path:   src/black/parsing.py
#
# ===----------------------------------------------------------------------=== #

"""
Parse Python code and perform AST validation.
"""

import ast
import sys
from collections.abc import Iterable, Iterator
from typing import Final

from mblib2to3 import pygram
from mblib2to3.pgen2 import driver, token
from mblib2to3.pgen2.grammar import Grammar
from mblib2to3.pgen2.parse import ParseError
from mblib2to3.pgen2.tokenize import TokenError
from mblib2to3.pytree import Leaf, Node

from mblack.mode import Feature, TargetVersion, supports_feature
from mblack.nodes import syms

PY2_HINT: Final = "Python 2 support was removed in version 22.0."


class InvalidInput(ValueError):
    """Raised when input source code fails all parse attempts."""


def get_grammars(target_versions: set[TargetVersion]) -> list[Grammar]:
    if not target_versions:
        # No target_version specified, so try all grammars.
        return [
            # Python 3.7+
            pygram.python_grammar_no_print_statement_no_exec_statement_async_keywords,
            # Python 3.0-3.6
            pygram.python_grammar_no_print_statement_no_exec_statement,
            # Python 3.10+
            pygram.python_grammar_soft_keywords,
        ]
    if TargetVersion.MOJO in target_versions:
        return [pygram.mojo_grammar]

    grammars = []
    # If we have to parse both, try to parse async as a keyword first
    if not supports_feature(
        target_versions, Feature.ASYNC_IDENTIFIERS
    ) and not supports_feature(target_versions, Feature.PATTERN_MATCHING):
        # Python 3.7-3.9
        grammars.append(
            pygram.python_grammar_no_print_statement_no_exec_statement_async_keywords
        )
    if not supports_feature(target_versions, Feature.ASYNC_KEYWORDS):
        # Python 3.0-3.6
        grammars.append(
            pygram.python_grammar_no_print_statement_no_exec_statement
        )
    if supports_feature(target_versions, Feature.PATTERN_MATCHING):
        # Python 3.10+
        grammars.append(pygram.python_grammar_soft_keywords)

    # At least one of the above branches must have been taken, because every Python
    # version has exactly one of the two 'ASYNC_*' flags
    return grammars


def _leading_spaces(line: str) -> int:
    """Return the number of leading spaces in a line."""
    return len(line) - len(line.lstrip(" "))


def _parse_error_hint(pe: ParseError, lines: list[str], lineno: int) -> str:
    """Return a hint string for common parse errors, or empty string."""
    if pe.type == token.INDENT:
        cur_indent = (
            _leading_spaces(lines[lineno - 1]) if lineno <= len(lines) else 0
        )
        # Walk backwards to find the previous non-blank line's indentation.
        prev_indent: int | None = None
        for i in range(lineno - 2, -1, -1):
            stripped = lines[i].strip()
            if stripped and not stripped.startswith("#"):
                prev_indent = _leading_spaces(lines[i])
                break
        if prev_indent is not None:
            return (
                f"Unexpected indent: previous line has {prev_indent} spaces"
                f" but this line has {cur_indent} spaces"
            )
        return "Unexpected indent"
    if pe.type == token.DEDENT:
        return "Unexpected dedent (unindent does not match any outer indentation level)"
    return ""


def lib2to3_parse(
    src_txt: str, target_versions: Iterable[TargetVersion] = ()
) -> Node:
    """Given a string with source, return the lib2to3 Node."""
    if not src_txt.endswith("\n"):
        src_txt += "\n"

    grammars = get_grammars(set(target_versions))
    errors = {}
    for grammar in grammars:
        drv = driver.Driver(grammar)
        try:
            result = drv.parse_string(src_txt, True)
            break

        except ParseError as pe:
            lineno, column = pe.context[1]
            lines = src_txt.splitlines()
            try:
                faulty_line = lines[lineno - 1]
            except IndexError:
                faulty_line = "<line number missing in source>"
            hint = _parse_error_hint(pe, lines, lineno)
            msg = f"Cannot parse: {lineno}:{column}: {faulty_line}"
            if hint:
                msg += f"\n{hint}"
            errors[grammar.version] = InvalidInput(msg)

        except TokenError as te:
            # In edge cases these are raised; and typically don't have a "faulty_line".
            lineno, column = te.args[1]
            errors[grammar.version] = InvalidInput(
                f"Cannot parse: {lineno}:{column}: {te.args[0]}"
            )

    else:
        # Choose the latest version when raising the actual parsing error.
        assert len(errors) >= 1
        exc = errors[max(errors)]

        if matches_grammar(src_txt, pygram.python_grammar) or matches_grammar(
            src_txt, pygram.python_grammar_no_print_statement
        ):
            original_msg = exc.args[0]
            raise InvalidInput(f"{original_msg}\n{PY2_HINT}") from None

        raise exc from None

    if isinstance(result, Leaf):
        result = Node(syms.file_input, [result])
    return result


def matches_grammar(src_txt: str, grammar: Grammar) -> bool:
    drv = driver.Driver(grammar)
    try:
        drv.parse_string(src_txt, True)
    except (ParseError, TokenError, IndentationError):
        return False
    else:
        return True


def lib2to3_unparse(node: Node) -> str:
    """Given a lib2to3 node, return its string representation."""
    code = str(node)
    return code


def parse_single_version(src: str, version: tuple[int, int]) -> ast.AST:
    filename = "<unknown>"
    return ast.parse(src, filename, feature_version=version, type_comments=True)


def parse_ast(src: str) -> ast.AST:
    # TODO: support Python 4+ ;)
    versions = [(3, minor) for minor in range(3, sys.version_info[1] + 1)]

    first_error = ""
    for version in sorted(versions, reverse=True):
        try:
            return parse_single_version(src, version)
        except SyntaxError as e:
            if not first_error:
                first_error = str(e)

    raise SyntaxError(first_error)


ast3_AST: Final[type[ast.AST]] = ast.AST


def _normalize(lineend: str, value: str) -> str:
    # To normalize, we strip any leading and trailing space from
    # each line...
    stripped: list[str] = [i.strip() for i in value.splitlines()]
    normalized = lineend.join(stripped)
    # ...and remove any blank lines at the beginning and end of
    # the whole string
    return normalized.strip()


def stringify_ast(node: ast.AST, depth: int = 0) -> Iterator[str]:
    """Simple visitor generating strings to compare ASTs by content."""

    yield f"{'  ' * depth}{node.__class__.__name__}("

    for field in sorted(node._fields):
        # TypeIgnore has only one field 'lineno' which breaks this comparison
        if isinstance(node, ast.TypeIgnore):
            break

        try:
            value: object = getattr(node, field)
        except AttributeError:
            continue

        yield f"{'  ' * (depth + 1)}{field}="

        if isinstance(value, list):
            for item in value:
                # Ignore nested tuples within del statements, because we may insert
                # parentheses and they change the AST.
                if (
                    field == "targets"
                    and isinstance(node, ast.Delete)
                    and isinstance(item, ast.Tuple)
                ):
                    for elt in item.elts:
                        yield from stringify_ast(elt, depth + 2)

                elif isinstance(item, ast.AST):
                    yield from stringify_ast(item, depth + 2)

        # Note that we are referencing the typed-ast ASTs via global variables and not
        # direct module attribute accesses because that breaks mypyc. It's probably
        # something to do with the ast3 variables being marked as Any leading
        # mypy to think this branch is always taken, leaving the rest of the code
        # unanalyzed. Tighting up the types for the typed-ast AST types avoids the
        # mypyc crash.
        elif isinstance(value, (ast.AST, ast3_AST)):
            yield from stringify_ast(value, depth + 2)

        else:
            normalized: object
            # Constant strings may be indented across newlines, if they are
            # docstrings; fold spaces after newlines when comparing. Similarly,
            # trailing and leading space may be removed.
            if (
                isinstance(node, ast.Constant)
                and field == "value"
                and isinstance(value, str)
            ):
                normalized = _normalize("\n", value)
            else:
                normalized = value
            yield (
                f"{'  ' * (depth + 2)}{normalized!r},  # {value.__class__.__name__}"
            )

    yield f"{'  ' * depth})  # /{node.__class__.__name__}"
