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

from __future__ import annotations

import ast

import pytest
from sybil.text import LineNumberOffsets
from sybil_collect import (
    _extra_docstrings,
    _private_spans,
    _string_inner_span,
    example_group_keys,
    is_private_name,
    shard_assignments,
)


@pytest.mark.parametrize(
    "prefix,extracts",
    [("", True), ("r", True), ("f", True), ("u", False), ("rf", False)],
)
def test_string_prefixes_do_not_crash(prefix: str, extracts: bool) -> None:
    # Synthetic Constant so the prefix still exercises _string_inner_span.
    # Sybil's DOCSTRING_PUNCTUATION recognizes r/f; an unrecognized prefix is
    # skipped (None) rather than crashing collection.
    source = f'{prefix}"""hello"""'
    node = ast.Constant(
        value="hello",
        lineno=1,
        col_offset=0,
        end_lineno=1,
        end_col_offset=len(source),
    )
    span = _string_inner_span(source, LineNumberOffsets(source), node)
    if extracts:
        assert span is not None
        assert source[span[0] : span[1]] == "hello"
    else:
        assert span is None


def test_private_module_names_are_detected() -> None:
    assert is_private_name("_internal.py")
    assert is_private_name("_internal")
    assert not is_private_name("elementwise.py")
    assert not is_private_name("__init__.py")


def test_extra_docstrings_finds_dunder_doc_assignment() -> None:
    source = 'class C:\n    pass\nC.__doc__ = "set later"\n'
    tree = ast.parse(source)
    results = list(_extra_docstrings(source, tree, LineNumberOffsets(source)))
    assert [text for _, _, text in results] == ["set later"]


def test_extra_docstrings_finds_attribute_docstring() -> None:
    source = 'X = 1\n"""doc for X"""\n'
    tree = ast.parse(source)
    results = list(_extra_docstrings(source, tree, LineNumberOffsets(source)))
    assert [text for _, _, text in results] == ["doc for X"]


def test_private_spans_cover_private_defs_only() -> None:
    source = "def _hidden():\n    pass\n\ndef shown():\n    pass\n"
    tree = ast.parse(source)
    spans = _private_spans(tree, LineNumberOffsets(source))
    assert len(spans) == 1
    shown_start = source.index("def shown")
    assert not any(lo <= shown_start < hi for lo, hi in spans)


def test_string_inner_span_strips_quotes_and_prefix() -> None:
    source = 'r"""abc"""'
    expr = ast.parse(source).body[0]
    assert isinstance(expr, ast.Expr)
    node = expr.value
    assert isinstance(node, ast.Constant)
    span = _string_inner_span(source, LineNumberOffsets(source), node)
    assert span is not None
    assert source[span[0] : span[1]] == "abc"


def test_shard_assignments_balances_and_covers_all_shards() -> None:
    counts = {"big.py": 90, "mid.py": 30, "small.py": 10, "tiny.py": 5}
    assignment = shard_assignments(counts, 3)
    assert set(assignment) == set(counts)
    assert set(assignment.values()) == {0, 1, 2}
    big_shard = assignment["big.py"]
    assert [f for f, s in assignment.items() if s == big_shard] == ["big.py"]


def test_example_group_keys_bind_checks_and_skips_to_their_example() -> None:
    entries = [
        ("a.py", "skip"),  # guards the example after it
        ("a.py", "code-block"),
        ("a.py", "invisible-code-block"),
        ("a.py", "code-block"),
        ("b.py", "code-block"),
    ]
    keys = example_group_keys(entries)
    assert keys[0] == keys[1] == keys[2] == "a.py::0"
    assert keys[3] == "a.py::1"
    assert keys[4] == "b.py::0"


def test_shard_assignments_is_deterministic() -> None:
    counts = {f"file{i}.py": (i * 7) % 13 + 1 for i in range(20)}
    assert shard_assignments(counts, 4) == shard_assignments(
        dict(reversed(counts.items())), 4
    )
