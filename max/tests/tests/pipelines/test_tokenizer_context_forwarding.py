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
"""Validates that every tokenizer forwards the request's dKV cache hint.

:attr:`TextContext.dkv_cache_hint` is the only carrier from the request body to
the dKV connector's ``load``, and each tokenizer that builds its own context
passes it by hand. A tokenizer that omits it serves every hinted request as an
unhinted one: a silent loss of remote cache hits, with nothing failing.

This test parses files with :mod:`ast` (no imports) so it runs without pulling
in the per-architecture tokenizer dependency graph, and so a newly added
architecture is covered without a new test.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest


def _pipelines_root() -> Path:
    """Locates the pipelines source tree without importing any of it.

    Under bazel the sources arrive as ``data`` in the runfiles tree, whose root
    ``TEST_SRCDIR`` names; the workspace segment below it is globbed rather
    than hardcoded. Outside bazel, walk up from this file.
    """
    srcdir = os.environ.get("TEST_SRCDIR")
    if srcdir:
        roots = sorted(Path(srcdir).glob("*/max/python/max/pipelines"))
        if roots:
            return roots[0]
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "max" / "python" / "max" / "pipelines"
        if candidate.is_dir():
            return candidate
    raise AssertionError(f"could not locate the pipelines tree from {here}")


_PIPELINES_ROOT = _pipelines_root()
_ARCH_ROOT = _PIPELINES_ROOT / "architectures"

# Carried on the context purely so the dKV connector can parse it in Rust; see
# max/python/max/pipelines/lib/tokenizer.py and dkv/docs/cache-hint.md.
_REQUIRED_KWARG = "dkv_cache_hint"

# The root of the context hierarchy that has a ``dkv_cache_hint`` field.
# ``PixelContext`` and ``AudioContext`` are siblings rather than subclasses and
# have no KV cache behind them, so they carry no hint.
_ROOT_CONTEXT = "TextContext"


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return None


def _context_modules() -> list[Path]:
    return [
        p
        for p in [
            _PIPELINES_ROOT / "context" / "context.py",
            *sorted(_ARCH_ROOT.glob("*/context.py")),
        ]
        if p.exists()
    ]


def _base_names(node: ast.ClassDef) -> set[str]:
    names = set()
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.add(base.id)
        elif isinstance(base, ast.Attribute):
            names.add(base.attr)
    return names


def _text_context_classes() -> set[str]:
    """Every context class that inherits ``TextContext``, transitively.

    Resolved from source rather than by importing, so this stays cheap; the
    hierarchy spans the context package and the per-architecture ``context.py``
    files, so the fixed point is iterated until it stops growing.
    """
    classes: list[ast.ClassDef] = []
    for path in _context_modules():
        tree = _parse(path)
        if tree is None:
            continue
        classes += [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

    resolved = {_ROOT_CONTEXT}
    while True:
        grown = {
            node.name
            for node in classes
            if _base_names(node) & resolved and node.name not in resolved
        }
        if not grown:
            return resolved
        resolved |= grown


def _context_sites(
    path: Path, context_classes: set[str]
) -> list[tuple[str, set[str]]]:
    """Returns ``(context class, kwarg names)`` per text context built here.

    Deliberately not scoped to ``new_context``: qwen2_5vl builds its context in
    a ``_create_context`` helper, and keying on the caller's name would leave
    that (and any future indirection) silently uncovered.
    """
    tree = _parse(path)
    if tree is None:
        return []

    sites = []
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        name = (
            func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        )
        if name in context_classes:
            sites.append((name, {k.arg for k in call.keywords if k.arg}))
    return sites


def _tokenizer_files() -> list[Path]:
    files = [
        _PIPELINES_ROOT / "lib" / "tokenizer.py",
        *sorted(_ARCH_ROOT.glob("*/tokenizer.py")),
    ]
    return [p for p in files if p.exists()]


def _forwarding_files() -> list[Path]:
    classes = _text_context_classes()
    return [p for p in _tokenizer_files() if _context_sites(p, classes)]


def test_only_text_contexts_are_in_scope() -> None:
    """The hierarchy resolves to the contexts that actually carry a hint."""
    classes = _text_context_classes()

    assert {"TextContext", "TextAndVisionContext"} <= classes
    # Sibling hierarchies with no KV cache must not be pulled in, or the
    # diffusion and audio tokenizers would be asked for a hint they have no
    # field for.
    assert not ({"PixelContext", "AudioContext"} & classes)


def test_discovery_finds_the_known_tokenizers() -> None:
    """Guards the parametrization from silently collapsing to nothing.

    Without this, a rename that breaks discovery turns every case below into a
    vacuous pass rather than a failure.
    """
    found = {p.parent.name for p in _forwarding_files()}

    assert "lib" in found, f"lib/tokenizer.py not discovered (found {found})"
    # The architectures that build their own text contexts today. Listed so a
    # tokenizer dropping out of discovery is a failure rather than a silent
    # loss of coverage; extend it when an architecture is added.
    assert {
        "gemma4",
        "idefics3",
        "kimik2_5",
        "kimik2_5_modulev3",
        "qwen2_5vl",
        "qwen3vl_moe",
    } <= found, f"missing from discovery: {found}"


@pytest.mark.parametrize(
    "path", _forwarding_files(), ids=lambda p: p.parent.name
)
def test_tokenizer_forwards_the_dkv_cache_hint(path: Path) -> None:
    classes = _text_context_classes()
    missing = [
        context
        for context, kwargs in _context_sites(path, classes)
        if _REQUIRED_KWARG not in kwargs
    ]

    assert not missing, (
        f"{path.relative_to(_PIPELINES_ROOT)} builds {missing} without "
        f"{_REQUIRED_KWARG}=..., so a hinted request served by it loses every "
        f"remote dKV cache hit. Forward it with "
        f"{_REQUIRED_KWARG}=encode_dkv_cache_hint(request.dkv_cache_hint)."
    )
