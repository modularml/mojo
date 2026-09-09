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
"""Lint Python docstrings for style guide violations unaccounted for by rumdl.

This linter enforces the following rules from docs/internal/PythonDocstringStyleGuide.md:

1. DOC001: Classes using msgspec.Struct should document fields with a
   "Configuration:" section, not "Args:" (per DOCS-990).

2. DOC002: Argument lists must NOT include type names in parentheses.
   Types are auto-generated from signatures; manual types interfere with
   doc rendering and can go stale.
   Bad:  `name (str): Description`
   Good: `name: Description`

3. DOC003: Never use "Attributes:" in class docstrings. Writing attributes in
   the class docstring generates duplicate attribute docs, and can go stale. For
   regular classes, dataclasses, and NamedTuples, use inline docstrings on
   attribute definitions. For Enums, use #: doc comments above each enum value.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from lint_helpers import get_all_files, get_changed_files, is_fast


@dataclass
class LintError:
    """A single lint error."""

    file: Path
    line: int
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}: {self.code} {self.message}"


# Regex to detect type annotations in parentheses in docstring arg lists.
# Matches lines like:
#   name (str): description
#   name (list[int]): description
#   name (Optional[str]): description
#   name (torch.Tensor): description
# But NOT:
#   name: (some parenthetical note) description
#   name: description (with parens)
#
# The pattern requires the parentheses to come immediately after the arg name
# and before the colon.
TYPE_IN_PARENS_PATTERN = re.compile(
    r"^\s+(\w+)\s+\([A-Za-z_][\w\[\], |.]*\)\s*:", re.MULTILINE
)

# Pattern to find section headers in Google-style docstrings
SECTION_HEADER_PATTERN = re.compile(
    r"^\s*(Args|Arguments|Configuration|Attributes|Returns|Yields|Raises):\s*$",
    re.MULTILINE,
)

# Pattern specifically for Attributes: section
ATTRIBUTES_SECTION_PATTERN = re.compile(r"^\s*Attributes:\s*$", re.MULTILINE)

# Pattern specifically for Args: or Arguments: section
ARGS_SECTION_PATTERN = re.compile(r"^\s*(Args|Arguments):\s*$", re.MULTILINE)


def get_docstring_and_line(node: ast.AST) -> tuple[str | None, int | None]:
    """Extract the docstring and its line number from an AST node."""
    if not isinstance(
        node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
    ):
        return None, None

    if not node.body:
        return None, None

    first_stmt = node.body[0]
    if not isinstance(first_stmt, ast.Expr):
        return None, None

    if not isinstance(first_stmt.value, ast.Constant):
        return None, None

    if not isinstance(first_stmt.value.value, str):
        return None, None

    return first_stmt.value.value, first_stmt.lineno


def is_enum(node: ast.ClassDef) -> bool:
    """Check if a class inherits from enum.Enum or Enum."""
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id in (
            "Enum",
            "IntEnum",
            "StrEnum",
        ):
            return True
        if isinstance(base, ast.Attribute):
            # Handle enum.Enum, enum.IntEnum, etc.
            if (
                isinstance(base.value, ast.Name)
                and base.value.id == "enum"
                and base.attr
                in ("Enum", "IntEnum", "StrEnum", "Flag", "IntFlag")
            ):
                return True
    return False


def is_msgspec_struct(node: ast.ClassDef) -> bool:
    """Check if a class inherits from msgspec.Struct."""
    for base in node.bases:
        # Handle simple name: Struct
        if isinstance(base, ast.Name) and base.id == "Struct":
            return True
        # Handle attribute access: msgspec.Struct
        if isinstance(base, ast.Attribute):
            if base.attr == "Struct":
                if (
                    isinstance(base.value, ast.Name)
                    and base.value.id == "msgspec"
                ):
                    return True
        # Handle call with bases: msgspec.Struct(tag=..., kw_only=True)
        if isinstance(base, ast.Call):
            func = base.func
            if isinstance(func, ast.Name) and func.id == "Struct":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "Struct":
                if (
                    isinstance(func.value, ast.Name)
                    and func.value.id == "msgspec"
                ):
                    return True
    return False


def check_type_in_parens(
    docstring: str, base_line: int
) -> Iterator[tuple[int, str]]:
    """Check for type annotations in parentheses in docstring sections.

    Yields (line_offset, matched_text) for each violation.
    """
    lines = docstring.split("\n")
    in_args_section = False

    for i, line in enumerate(lines):
        # Check if we're entering/leaving a section
        section_match = SECTION_HEADER_PATTERN.match(line)
        if section_match:
            section_name = section_match.group(1)
            # Only check in sections that list parameters
            in_args_section = section_name in (
                "Args",
                "Arguments",
                "Configuration",
                "Returns",
                "Yields",
                "Raises",
            )
            continue

        # Check for type in parens only in relevant sections
        if in_args_section:
            match = TYPE_IN_PARENS_PATTERN.match(line)
            if match:
                yield i, match.group(0).strip()


def check_attributes_section(docstring: str) -> bool:
    """Check if docstring contains an Attributes: section."""
    return bool(ATTRIBUTES_SECTION_PATTERN.search(docstring))


def check_args_in_struct(docstring: str) -> bool:
    """Check if docstring contains an Args: or Arguments: section."""
    return bool(ARGS_SECTION_PATTERN.search(docstring))


def lint_file(filepath: Path) -> Iterator[LintError]:
    """Lint a single Python file for docstring violations."""
    try:
        source = filepath.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
        return

    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError as e:
        print(f"Warning: Could not parse {filepath}: {e}", file=sys.stderr)
        return

    for node in ast.walk(tree):
        # Check class docstrings
        if isinstance(node, ast.ClassDef):
            docstring, doc_line = get_docstring_and_line(node)
            if docstring and doc_line:
                # DOC003: No Attributes: section in class docstrings
                if check_attributes_section(docstring):
                    if is_enum(node):
                        yield LintError(
                            file=filepath,
                            line=doc_line,
                            code="DOC003",
                            message=(
                                f"Enum '{node.name}' uses 'Attributes:' section in docstring. "
                                "Move these descriptions to #: doc comments above each enum value "
                                "(e.g., `#: Description\\n    VALUE = 'x'`). "
                                "See docs/internal/PythonDocstringStyleGuide.md"
                            ),
                        )
                    else:
                        yield LintError(
                            file=filepath,
                            line=doc_line,
                            code="DOC003",
                            message=(
                                f"Class '{node.name}' uses 'Attributes:' section in docstring. "
                                "Move these descriptions to inline docstrings on attribute definitions "
                                '(e.g., `name: str\\n    """Description."""`). '
                                "See docs/internal/PythonDocstringStyleGuide.md#dataclass-attributes"
                            ),
                        )

                # DOC001: msgspec.Struct should use Configuration: not Args:
                if is_msgspec_struct(node):
                    if check_args_in_struct(docstring):
                        yield LintError(
                            file=filepath,
                            line=doc_line,
                            code="DOC001",
                            message=(
                                f"msgspec.Struct class '{node.name}' uses 'Args:' section. "
                                "Use 'Configuration:' instead for struct fields "
                                "(e.g., `Configuration:\\n        field_name: Description.`). "
                                "See DOCS-990 and docs/internal/PythonDocstringStyleGuide.md"
                            ),
                        )

                # DOC002: No type in parentheses
                for line_offset, matched in check_type_in_parens(
                    docstring, doc_line
                ):
                    yield LintError(
                        file=filepath,
                        line=doc_line + line_offset,
                        code="DOC002",
                        message=(
                            f"Type annotation in parentheses: '{matched}'. "
                            "Remove the type; use `name: Description` not `name (type): Description`. "
                            "See docs/internal/PythonDocstringStyleGuide.md"
                        ),
                    )

        # Check function/method docstrings for DOC002
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring, doc_line = get_docstring_and_line(node)
            if docstring and doc_line:
                for line_offset, matched in check_type_in_parens(
                    docstring, doc_line
                ):
                    yield LintError(
                        file=filepath,
                        line=doc_line + line_offset,
                        code="DOC002",
                        message=(
                            f"Type annotation in parentheses: '{matched}'. "
                            "Remove the type; use `name: Description` not `name (type): Description`. "
                            "See docs/internal/PythonDocstringStyleGuide.md"
                        ),
                    )


def _linter_changed(changed_files: set[str]) -> bool:
    """Return True if the linter script itself is among the changed files.

    When the linter changes, fast mode falls back to scanning all tracked
    Python files so that new or updated rules are applied to the full codebase.
    """
    return "oss/modular/bazel/lint/lint_docstrings.py" in changed_files


def get_python_files() -> list[Path]:
    """Get Python files to lint, respecting the FAST env-var convention.

    When FAST=1 (the default), only files changed relative to origin/main are
    returned — unless the linter script itself changed, in which case all
    tracked Python files are returned so new rules apply to the full codebase.
    When FAST=0, every tracked Python file is always returned. Both paths
    exclude files matched by :func:`should_skip_file`.
    """
    if not is_fast():
        # FAST=0: Always return all files, no need to call get_changed_files()
        all_files = get_all_files()
    else:
        # FAST=1: Only check changed files, unless linter itself changed
        changed = get_changed_files()
        if _linter_changed(changed):
            all_files = get_all_files()
        else:
            all_files = changed
    return [Path(f) for f in all_files if Path(f).suffix == ".py"]


def should_skip_file(filepath: Path) -> bool:
    """Check if a file should be skipped based on exclude patterns."""
    path_str = str(filepath)

    # Skip third-party code
    if "third-party/" in path_str:
        return True

    # Skip generated files
    if path_str.endswith(("_pb2.py", "_pb2.pyi")):
        return True

    # Skip virtual environments and derived directories
    if "/.derived/" in path_str or "/venv/" in path_str:
        return True

    # Skip test data files that might have intentionally bad docstrings
    return "/testdata/" in path_str


def main() -> int:
    """Main entry point."""
    files = get_python_files()
    errors: list[LintError] = []

    for filepath in files:
        if should_skip_file(filepath):
            continue
        errors.extend(lint_file(filepath))

    if errors:
        print(f"Found {len(errors)} docstring lint error(s):\n")
        for error in sorted(errors, key=lambda e: (str(e.file), e.line)):
            print(error)
        print(
            "\nFor more information, see docs/internal/PythonDocstringStyleGuide.md"
        )
        return 1

    return 0


if __name__ == "__main__":
    if path := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(path)
    sys.exit(main())
