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
"""Draft-width checks for the unified DFlash Kimi K2.5 config.

Validation never writes back to the caller's ``SpeculativeConfig``; the
trained width is resolved as a plain int through the shared
:func:`resolve_dflash_num_speculative_tokens` helper and threaded by the
model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import pytest
from max.graph import DeviceRef
from max.pipelines.architectures.deepseekV3.model_config import (
    DeepseekV3Config,
)
from max.pipelines.architectures.dflash_kimi_k25 import (
    DFlashKimiK25DraftConfig,
)
from max.pipelines.architectures.unified_dflash_kimi_k25.model_config import (
    UnifiedDflashKimiK25Config,
    resolve_dflash_num_speculative_tokens,
)
from max.pipelines.architectures.unified_dflash_llama3.model_config import (
    resolve_dflash_num_speculative_tokens as _llama3_resolve,
)
from max.pipelines.lib import SpeculativeConfig

BLOCK_SIZE = 7


def test_resolve_helper_is_the_shared_dflash_one() -> None:
    assert resolve_dflash_num_speculative_tokens is _llama3_resolve


@dataclass
class _FakeConfig:
    devices: list[DeviceRef]
    vocab_size: int = 1000
    num_hidden_layers: int = 2


def _make_arch_config(
    speculative_config: SpeculativeConfig,
) -> UnifiedDflashKimiK25Config:
    return UnifiedDflashKimiK25Config(
        target=cast(DeepseekV3Config, _FakeConfig([DeviceRef.GPU()])),
        draft=cast(DFlashKimiK25DraftConfig, _FakeConfig([DeviceRef.GPU()])),
        speculative_config=speculative_config,
        target_layer_ids=[10, 20],
        mask_token_id=3,
        block_size=BLOCK_SIZE,
    )


def test_validate_never_mutates_speculative_config(
    caplog: pytest.LogCaptureFixture,
) -> None:
    unset = SpeculativeConfig(speculative_method="dflash")
    config = _make_arch_config(unset)
    config.validate_dflash_fields()
    assert unset.num_speculative_tokens is None

    mismatched = SpeculativeConfig(
        speculative_method="dflash", num_speculative_tokens=4
    )
    config = _make_arch_config(mismatched)
    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        config.validate_dflash_fields()
    assert "overridden from 4 to 6" in caplog.text
    assert mismatched.num_speculative_tokens == 4

    # Module construction reads the trained width off the arch config
    # regardless of the CLI value.
    assert config.resolve_block_size() == BLOCK_SIZE
