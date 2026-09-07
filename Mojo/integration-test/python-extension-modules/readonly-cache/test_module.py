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

"""End-to-end tests for importing a Mojo module from a read-only source tree.

These reproduce the failure mode where a Mojo extension lives in a read-only
``site-packages`` directory: the importer cannot create an in-tree
``__mojocache__`` directory, so it must redirect the compiled artifact to the
Modular cache folder (which is configurable via ``modular.cfg``'s ``cache_dir``
key or the ``MODULAR_CACHE_DIR`` environment variable).
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import mojo.importer
import pytest

# A minimal Mojo extension module. The ``PyInit_<name>`` symbol must match the
# name the module is imported under, so the source is templated per test.
_MOJO_SOURCE_TEMPLATE = """\
from std.os import abort

from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder


@export
def PyInit_{name}() -> PythonObject:
    try:
        var b = PythonModuleBuilder("{name}")
        b.def_function[get_answer]("get_answer")
        return b.finalize()
    except e:
        abort(String("failed to create Python module: ", e))


def get_answer() raises -> PythonObject:
    return 42
"""


def _make_read_only_module(parent: Path, name: str) -> Path:
    """Writes a single-file Mojo module under `parent` and marks it read-only.

    Returns the read-only directory that should be added to `sys.path`.
    """
    src_dir = parent / f"{name}_src"
    src_dir.mkdir()
    (src_dir / f"{name}.mojo").write_text(
        _MOJO_SOURCE_TEMPLATE.format(name=name)
    )
    # Mark the source directory read-only so the in-tree cache cannot be used.
    os.chmod(src_dir, 0o555)
    return src_dir


def _skip_if_root() -> None:
    if os.geteuid() == 0:
        pytest.skip(
            "running as root: read-only directory detection is bypassed"
        )


def test_read_only_tree_redirects_to_configured_cache_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`MODULAR_CACHE_DIR` redirects compilation out of a read-only tree."""
    _skip_if_root()

    name = "readonly_configured_mod"
    src_dir = _make_read_only_module(tmp_path, name)
    cache_dir = tmp_path / "configured_cache"

    monkeypatch.setenv("MODULAR_CACHE_DIR", str(cache_dir))
    monkeypatch.syspath_prepend(str(src_dir))

    try:
        module = importlib.import_module(name)
        assert module.get_answer() == 42
    finally:
        os.chmod(src_dir, 0o755)
        sys.modules.pop(name, None)

    # Nothing was written next to the read-only source.
    assert not (src_dir / "__mojocache__").exists()
    # The compiled artifact landed under the configured cache directory.
    compiled = list((cache_dir / ".mojo_cache").rglob(f"{name}.*.so"))
    assert compiled, (
        f"expected a compiled .so under {cache_dir / '.mojo_cache'}"
    )


def test_read_only_tree_auto_redirects_without_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no config, a read-only tree still imports via the default cache."""
    _skip_if_root()

    name = "readonly_default_mod"
    src_dir = _make_read_only_module(tmp_path, name)

    monkeypatch.delenv("MODULAR_CACHE_DIR", raising=False)
    monkeypatch.syspath_prepend(str(src_dir))

    # Resolve the default cache root for this (unconfigured) environment so we
    # can assert the artifact actually landed there, rather than merely that it
    # avoided the read-only source tree.
    cache_root = mojo.importer._modular_cache_root()
    assert cache_root is not None

    try:
        module = importlib.import_module(name)
        assert module.get_answer() == 42
    finally:
        os.chmod(src_dir, 0o755)
        sys.modules.pop(name, None)

    # Nothing was written next to the read-only source.
    assert not (src_dir / "__mojocache__").exists()
    # The compiled artifact landed in the default Modular cache folder,
    # namespaced by the module name.
    compiled = list(
        (cache_root / "python_extensions" / name).glob(f"{name}.*.so")
    )
    assert compiled, (
        f"expected a compiled .so under {cache_root / 'python_extensions' / name}"
    )
