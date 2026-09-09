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
"""Sybil collection for ``max/python/max`` docstring examples."""

from __future__ import annotations

import ast
import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from sybil import Sybil
from sybil.document import DOCSTRING_PUNCTUATION, PythonDocStringDocument
from sybil.evaluators.skip import Skipper
from sybil.example import Example, NotEvaluated
from sybil.integration.pytest import SybilItem
from sybil.parsers.rest import PythonCodeBlockParser, SkipParser
from sybil.text import LineNumberOffsets

#: Paths under ``max/python/max`` that are never collected.
COLLECT_EXCLUDES = (
    "serve/schemas/",  # generated
    "graph/weights/load_gguf.py",  # hard top-level ``import gguf``
    # benchmark/benchmark_shared/server_metrics.py has a module-level import
    # cycle, and Sybil imports each document's module directly; the benchmark
    # tree's examples are illustrative, so skip the whole directory.
    "benchmark/",
)


def is_private_name(name: str) -> bool:
    """True for a single-underscore name; dunders are public."""
    stem = name.removesuffix(".py")
    return stem.startswith("_") and not (
        stem.startswith("__") and stem.endswith("__")
    )


def _node_char_span(
    offsets: LineNumberOffsets, node: ast.stmt
) -> tuple[int, int] | None:
    if node.end_lineno is None or node.end_col_offset is None:
        return None
    start = offsets.get(node.lineno - 1, node.col_offset)
    end = offsets.get(node.end_lineno - 1, node.end_col_offset)
    return start, end


def _private_spans(
    tree: ast.AST, offsets: LineNumberOffsets
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ) and is_private_name(node.name):
            span = _node_char_span(offsets, node)
            if span is not None:
                spans.append(span)
    return spans


def _is_string_constant(node: ast.expr) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _string_inner_span(
    source: str, offsets: LineNumberOffsets, node: ast.Constant
) -> tuple[int, int] | None:
    # Mirror Sybil's own docstring extraction (sybil.document): strip the
    # opening quote and prefix, then an equal-length closing quote. A docstring
    # is always a single string literal, so open and close lengths match.
    # Return None (skip) rather than crash if the literal uses a prefix Sybil's
    # regex doesn't recognize (e.g. ``u`` or ``rf``); such docstrings are rare.
    end_lineno = node.end_lineno or node.lineno
    node_start = offsets.get(node.lineno - 1, node.col_offset)
    node_end = offsets.get(end_lineno - 1, node.end_col_offset or 0)
    punc = DOCSTRING_PUNCTUATION.match(source, node_start, node_end)
    if punc is None:
        return None
    return punc.end(), node_end - len(punc.group(1))


def _extra_docstrings(
    source: str, tree: ast.AST, offsets: LineNumberOffsets
) -> Iterator[tuple[int, int, str]]:
    """Yield ``__doc__`` assignments and attribute docstrings Sybil misses."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr == "__doc__"
            and _is_string_constant(node.value)
        ):
            assert isinstance(node.value, ast.Constant)
            text = node.value.value
            assert isinstance(text, str)
            span = _string_inner_span(source, offsets, node.value)
            if span is not None:
                yield span[0], span[1], text
        if isinstance(node, (ast.Module, ast.ClassDef)):
            for prev, stmt in zip(node.body, node.body[1:], strict=False):
                if not (
                    isinstance(prev, (ast.Assign, ast.AnnAssign))
                    and isinstance(stmt, ast.Expr)
                    and _is_string_constant(stmt.value)
                ):
                    continue
                assert isinstance(stmt.value, ast.Constant)
                text = stmt.value.value
                assert isinstance(text, str)
                span = _string_inner_span(source, offsets, stmt.value)
                if span is not None:
                    yield span[0], span[1], text


class _PublicDocStringDocument(PythonDocStringDocument):
    @staticmethod
    def extract_docstrings(source: str) -> Iterator[tuple[int, int, str]]:
        tree = ast.parse(source)
        offsets = LineNumberOffsets(source)
        private = _private_spans(tree, offsets)
        docstrings = list(PythonDocStringDocument.extract_docstrings(source))
        docstrings.extend(_extra_docstrings(source, tree, offsets))
        docstrings.sort()
        for start, end, text in docstrings:
            if not any(lo <= start < hi for lo, hi in private):
                yield start, end, text

    def import_document(self, example: Example) -> None:
        # Run examples in isolation. Skip Sybil's default import-region
        # evaluation and do not seed the module's namespace, so each example
        # must import the names it uses. A non-self-contained example then
        # fails here instead of silently borrowing the module's imports, which
        # is what keeps the rendered examples copy-paste runnable.
        self.pop_evaluator(self.import_document)
        raise NotEvaluated()


#: The scanned package root. Sybil anchors ``path`` at the directory of the
#: file that constructs it -- this file's directory, not the scanned tree --
#: and matches collected files by exact path prefix (``Path.relative_to``).
#: Normalize the anchor lexically: bazel runfiles are symlink forests, so
#: ``resolve()`` would escape the sandbox and never match pytest's paths,
#: while a literal ``..`` segment in the anchor never matches anything.
_SCAN_ROOT = Path(os.path.normpath(Path(__file__).parent / "../../python/max"))

_sybil = Sybil(
    parsers=[
        PythonCodeBlockParser(),
        SkipParser(),
    ],
    path=str(_SCAN_ROOT),
    patterns=["*.py"],
    # Directory excludes need a glob suffix for Sybil's matcher.
    excludes=[
        exclude + "*" if exclude.endswith("/") else exclude
        for exclude in COLLECT_EXCLUDES
    ],
    document_types={".py": _PublicDocStringDocument},
)
_collect_file = _sybil.pytest()


def _should_collect(path: Path) -> bool:
    """True if ``path`` is a public source Sybil should scan for examples."""
    return (
        path.suffix == ".py"
        and not is_private_name(path.name)
        and _sybil.should_parse(path)
    )


def pytest_ignore_collect(collection_path: Path, config: Any) -> bool | None:
    """Skip files the collector won't handle so pytest never imports them.

    A target's sources are passed as explicit path arguments, and an explicit
    ``.py`` path bypasses pytest's ``python_files`` filter: pytest would import
    each declined file (private, excluded, or out of scan root) as a test
    module and fail on any optional top-level dependency it carries. Ignoring
    the path here happens before any module import.
    """
    if collection_path.suffix == ".py" and not _should_collect(collection_path):
        return True
    return None


def pytest_collect_file(file_path: Path, parent: Any) -> Any:
    """Collect docstring examples from a public module under max/python/max."""
    return _collect_file(file_path, parent)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Accept the shard options ``pytest_runner`` derives from bazel's
    shard env vars.

    Not the stock pytest-shard plugin: its round-robin split separates an
    example from the ``invisible-code-block`` that checks it.
    """
    group = parser.getgroup("sybil_collect")
    group.addoption("--shard-id", type=int, default=0)
    group.addoption("--num-shards", type=int, default=1)


#: Directives that begin a new example group; the blocks that follow one
#: (``invisible-code-block`` checks, ``skip`` guards) belong to its group.
_GROUP_STARTING_DIRECTIVES = frozenset({"code", "code-block", "sourcecode"})


def example_group_keys(entries: Sequence[tuple[str, str]]) -> list[str]:
    """Map ordered ``(path, directive)`` items to shardable group keys.

    A group is one visible code block plus its trailing invisible checks;
    the authoring contract makes each visible example self-contained, so
    groups can run in separate shards. ``skip`` directives guard the example
    that follows them and therefore attach forward to the next group.
    """
    counters: dict[str, int] = {}
    pending_skips: dict[str, list[int]] = {}
    keys: list[str] = []
    for path, directive in entries:
        index = len(keys)
        if directive in _GROUP_STARTING_DIRECTIVES:
            counters[path] = counters.get(path, -1) + 1
            key = f"{path}::{counters[path]}"
            for skip_index in pending_skips.pop(path, []):
                keys[skip_index] = key
            keys.append(key)
        elif directive == "skip":
            pending_skips.setdefault(path, []).append(index)
            keys.append("")
        else:
            keys.append(f"{path}::{counters.get(path, 0)}")
    for path, indices in pending_skips.items():
        for skip_index in indices:
            keys[skip_index] = f"{path}::{counters.get(path, 0)}"
    return keys


def shard_assignments(
    counts: Mapping[str, int], num_shards: int
) -> dict[str, int]:
    """Assign each group to a shard, balancing by weight.

    Greedy, heaviest first; deterministic so every shard computes the same
    mapping.
    """
    loads = [0] * num_shards
    assignment: dict[str, int] = {}
    for key, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        shard = min(range(num_shards), key=lambda i: loads[i])
        assignment[key] = shard
        loads[shard] += count
    return assignment


_COLLECTED_ANY = pytest.StashKey[bool]()


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Keep only this shard's example groups when bazel sharding is active."""
    config.stash[_COLLECTED_ANY] = any(
        isinstance(item, SybilItem) for item in items
    )
    num_shards = config.getoption("--num-shards")
    if num_shards <= 1:
        return
    entries: list[tuple[str, str]] = []
    for item in items:
        example = getattr(item, "example", None)
        region = example.region if example else None
        directive = (getattr(region, "lexemes", None) or {}).get(
            "directive", ""
        )
        # Skip regions carry no directive lexeme; identify them by evaluator.
        if not directive and isinstance(
            getattr(region, "evaluator", None), Skipper
        ):
            directive = "skip"
        entries.append((str(item.path), directive))
    keys = example_group_keys(entries)
    # Weight groups equally: cost is dominated by the one graph compile a
    # visible example performs, not by its number of check blocks.
    assignment = shard_assignments({key: 1 for key in keys}, num_shards)
    shard_id = config.getoption("--shard-id")
    keep: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item, key in zip(items, keys, strict=False):
        (keep if assignment[key] == shard_id else deselected).append(item)
    items[:] = keep
    if deselected:
        config.hook.pytest_deselected(items=deselected)


def pytest_collection_finish(session: pytest.Session) -> None:
    """Fail fast if an opted-in target collected zero docstring examples.

    Checks the pre-shard collection, so a shard that legitimately received
    no files does not trip the guard. Runs only in sessions that load this
    plugin via ``-p sybil_collect``.
    """
    if session.config.stash.get(_COLLECTED_ANY, False):
        return
    raise pytest.UsageError(
        "test_docstring_examples is enabled but no docstring examples were "
        "collected; check the target's sources and COLLECT_EXCLUDES."
    )
