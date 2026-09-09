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

"""Tests for the shared interleaved-RoPE-weights derivation."""

from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest
from max.pipelines.lib.config.model_config import _interleaved_rope_weights

if TYPE_CHECKING:
    from max.pipelines.lib import MAXModelConfig


def _config(weight_path: str, rope_type: str | None) -> "MAXModelConfig":
    config = SimpleNamespace(
        weight_path=[Path(weight_path)], rope_type=rope_type
    )
    return cast("MAXModelConfig", config)


@pytest.mark.parametrize(
    ("rope_type", "expected"),
    [
        # Unset rope_type means the model default ("normal"), which keeps
        # the GGUF interleaved layout.
        (None, True),
        ("normal", True),
        ("neox", False),
        ("none", False),
        ("yarn", False),
    ],
)
def test_gguf_interleaving_by_rope_type(
    rope_type: str | None, expected: bool
) -> None:
    config = _config("model.gguf", rope_type)
    assert _interleaved_rope_weights(config) is expected


@pytest.mark.parametrize("rope_type", [None, "normal", "neox"])
def test_safetensors_never_interleaved(rope_type: str | None) -> None:
    config = _config("model.safetensors", rope_type)
    assert _interleaved_rope_weights(config) is False
