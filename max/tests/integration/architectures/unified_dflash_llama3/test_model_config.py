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
"""Draft-width resolution for the unified DFlash Llama3 config.

The drafter's trained width (``block_size - 1``) is resolved as a plain
int and threaded to KV sizing and module construction without ever
writing back to (or copying) the caller's config, so a shared or frozen
pipeline config is never mutated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import cast

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.architectures.llama3.model_config import Llama3Config
from max.pipelines.architectures.unified_dflash_llama3.model import (
    UnifiedDflashLlama3Model,
)
from max.pipelines.architectures.unified_dflash_llama3.model_config import (
    UnifiedDflashLlama3Config,
    resolve_dflash_num_speculative_tokens,
)
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    PipelineConfig,
    SpeculativeConfig,
)
from max.pipelines.lib.model_manifest import ModelManifest

BLOCK_SIZE = 7
TRAINED_WIDTH = BLOCK_SIZE - 1


def _make_pipeline_config(
    num_speculative_tokens: int | None,
    *,
    draft_block_size: int | None = BLOCK_SIZE,
) -> PipelineConfig:
    """Builds a minimal two-model PipelineConfig without full validation."""
    dflash_config: dict[str, object] = {
        "mask_token_id": 3,
        "target_layer_ids": [10, 20],
    }
    if draft_block_size is not None:
        dflash_config["block_size"] = draft_block_size

    model_config = MAXModelConfig.model_construct(model_path="fake/target")
    draft_config = MAXModelConfig.model_construct(model_path="fake/draft")
    draft_config._huggingface_config = SimpleNamespace(
        dflash_config=dflash_config
    )
    return PipelineConfig.model_construct(
        models=ModelManifest({"main": model_config, "draft": draft_config}),
        speculative=SpeculativeConfig(
            speculative_method="dflash",
            num_speculative_tokens=num_speculative_tokens,
        ),
    )


def test_resolve_returns_trained_width_for_unset() -> None:
    pipeline_config = _make_pipeline_config(None)
    original_speculative = pipeline_config.speculative

    resolved = resolve_dflash_num_speculative_tokens(pipeline_config)

    assert resolved == TRAINED_WIDTH
    # The caller's config objects are never written to.
    assert pipeline_config.speculative is original_speculative
    assert original_speculative is not None
    assert original_speculative.num_speculative_tokens is None


def test_resolve_keeps_matching_value_unchanged() -> None:
    pipeline_config = _make_pipeline_config(TRAINED_WIDTH)
    assert (
        resolve_dflash_num_speculative_tokens(pipeline_config) == TRAINED_WIDTH
    )
    assert pipeline_config.speculative is not None
    assert pipeline_config.speculative.num_speculative_tokens == TRAINED_WIDTH


def test_resolve_overrides_mismatch_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline_config = _make_pipeline_config(9)

    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        resolved = resolve_dflash_num_speculative_tokens(pipeline_config)

    assert "overridden from 9 to 6" in caplog.text
    assert resolved == TRAINED_WIDTH
    assert pipeline_config.speculative is not None
    assert pipeline_config.speculative.num_speculative_tokens == 9


def test_resolve_without_block_size_requires_explicit_value() -> None:
    with pytest.raises(ValueError, match="declares no block_size"):
        resolve_dflash_num_speculative_tokens(
            _make_pipeline_config(None, draft_block_size=None)
        )

    explicit = _make_pipeline_config(5, draft_block_size=None)
    assert resolve_dflash_num_speculative_tokens(explicit) == 5
    assert explicit.speculative is not None
    assert explicit.speculative.num_speculative_tokens == 5


def test_resolved_width_flows_to_kv_params() -> None:
    """``get_kv_params`` bakes the width the config was built with."""
    pipeline_config = _make_pipeline_config(TRAINED_WIDTH)

    huggingface_config = SimpleNamespace(
        num_key_value_heads=2,
        num_hidden_layers=2,
        head_dim=16,
    )
    kv_params = UnifiedDflashLlama3Model.get_kv_params(
        huggingface_config,
        pipeline_config,
        [DeviceRef.CPU()],
        KVCacheConfig(),
        DType.bfloat16,
    )
    assert kv_params.num_draft_tokens == TRAINED_WIDTH


@dataclass
class _FakeLlama3Config:
    devices: list[DeviceRef]
    vocab_size: int = 1000
    num_hidden_layers: int = 2
    return_logits: object = None
    return_hidden_states: object = None
    target_layer_ids: list[int] | None = None


def _make_arch_config(
    speculative_config: SpeculativeConfig,
) -> UnifiedDflashLlama3Config:
    return UnifiedDflashLlama3Config(
        target=cast(Llama3Config, _FakeLlama3Config([DeviceRef.GPU()])),
        draft=cast(Llama3Config, _FakeLlama3Config([DeviceRef.GPU()])),
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
