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
"""Tests for ConfigFileModel's config-file + CLI/env merge."""

from __future__ import annotations

import json
from pathlib import Path

from max.config import ConfigFileModel
from pydantic import Field


class _Backend(ConfigFileModel):
    model: str = ""
    tensor_parallel_size: int = 1


class _Load(ConfigFileModel):
    max_concurrency: int = 1
    num_prompts: int = 0


class _Top(ConfigFileModel):
    backend: _Backend = Field(default_factory=_Backend)
    load: _Load = Field(default_factory=_Load)


def _write(tmp_path: Path, payload: dict[str, object]) -> str:
    # JSON is valid YAML, so yaml.safe_load handles it.
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(payload))
    return str(p)


def test_config_file_loads_without_overrides(tmp_path: Path) -> None:
    """A config file with no CLI overrides populates the whole model."""
    cfg = _write(
        tmp_path, {"backend": {"model": "m", "tensor_parallel_size": 8}}
    )
    top = _Top.model_validate({"config_file": cfg})
    assert top.backend.model == "m"
    assert top.backend.tensor_parallel_size == 8


def test_partial_override_keeps_config_file_siblings(tmp_path: Path) -> None:
    """A partial CLI override deep in a subtree must not wipe sibling values.

    Regression for the shallow ``base | override`` merge: passing only
    ``load.max_concurrency`` used to drop the whole config-file ``backend``
    object (e.g. ``backend.model``).
    """
    cfg = _write(
        tmp_path,
        {
            "backend": {"model": "m", "tensor_parallel_size": 8},
            "load": {"max_concurrency": 8, "num_prompts": 100},
        },
    )
    # Mimics what cyclopts feeds the before-validator for a single CLI flag
    # (`--load.max-concurrency 16`): a sparse, partial nested dict.
    top = _Top.model_validate(
        {"config_file": cfg, "load": {"max_concurrency": 16}}
    )
    # Sibling subtree from the file is preserved.
    assert top.backend.model == "m"
    assert top.backend.tensor_parallel_size == 8
    # Sibling leaf within the overridden subtree is preserved.
    assert top.load.num_prompts == 100
    # The explicitly-overridden leaf wins.
    assert top.load.max_concurrency == 16


def test_cli_override_wins_over_config_file(tmp_path: Path) -> None:
    """An explicit override of a leaf the file also set takes precedence."""
    cfg = _write(tmp_path, {"backend": {"model": "from-file"}})
    top = _Top.model_validate(
        {"config_file": cfg, "backend": {"model": "from-cli"}}
    )
    assert top.backend.model == "from-cli"
