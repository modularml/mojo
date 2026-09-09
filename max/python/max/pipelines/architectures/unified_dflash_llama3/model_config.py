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
"""Config for DFlash Llama3 unified pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import ClassVar

from max.nn import ReturnHiddenStates
from max.nn.kv_cache import KVCacheParamInterface, MultiKVCacheParams
from max.nn.transformer import ReturnLogits
from max.pipelines.lib.config import (
    MAXModelConfig,
    PipelineConfig,
    SpeculativeConfig,
)
from max.pipelines.modeling.config_enums import SupportedEncoding
from max.pipelines.speculative._dflash import parse_dflash_draft_hf_config
from transformers import AutoConfig
from typing_extensions import Self

from ..llama3.model_config import ArchConfigWithKVCache, Llama3Config

logger = logging.getLogger("max.pipelines")


def resolve_dflash_num_speculative_tokens(
    pipeline_config: PipelineConfig,
    *,
    warn: bool = True,
) -> int:
    """Returns the resolved DFlash draft width.

    The DFlash drafter's behavior is only defined at its trained
    ``block_size``, so the width is ``block_size - 1``; a mismatching
    explicit ``num_speculative_tokens`` is overridden with a warning. When
    the draft checkpoint declares no ``block_size``, an explicit
    ``--num-speculative-tokens`` is required and returned as-is. Pure: the
    caller's config is never mutated or copied.
    """
    assert pipeline_config.speculative is not None
    assert pipeline_config.draft_model is not None
    return parse_dflash_draft_hf_config(
        pipeline_config.draft_model.huggingface_config
    ).draft_width(pipeline_config.speculative, warn=warn)


@dataclass(kw_only=True)
class UnifiedDflashLlama3Config(ArchConfigWithKVCache):
    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float32",
    }

    target: Llama3Config
    draft: Llama3Config
    speculative_config: SpeculativeConfig
    target_layer_ids: list[int] = field(default_factory=list)
    mask_token_id: int = 0
    block_size: int = 0
    quantization_encoding: SupportedEncoding | None = None
    resolved_num_speculative_tokens: int | None = None
    """Per-step draft count: explicit value if set, else the trained width."""

    def __post_init__(self) -> None:
        self.target.return_logits = ReturnLogits.VARIABLE
        self.target.return_hidden_states = ReturnHiddenStates.SELECTED_LAYERS
        self.target.target_layer_ids = list(self.target_layer_ids)
        self.draft.return_hidden_states = ReturnHiddenStates.LAST

        if len(self.target.devices) != len(self.draft.devices):
            raise ValueError(
                "Target and draft must have the same number of devices."
                f" Got target={len(self.target.devices)}"
                f" draft={len(self.draft.devices)}."
            )
        if len(self.target.devices) != 1:
            raise ValueError(
                "DFlash currently supports a single device only. Got"
                f" {len(self.target.devices)} devices."
            )

    def validate_dflash_fields(self) -> None:
        """Strict validation run from ``UnifiedDflashLlama3Model.load_model``
        once the DFlash-specific fields have been populated from the draft
        HF config — ``__post_init__`` accepts the empty-placeholder config
        produced by :meth:`initialize` so we can't enforce these there.
        """
        if not self.target_layer_ids:
            raise ValueError(
                "DFlash requires non-empty target_layer_ids (one per draft"
                " hidden layer)."
            )
        if len(self.target_layer_ids) != self.draft.num_hidden_layers:
            raise ValueError(
                "DFlash invariant: len(target_layer_ids) must equal the"
                " draft's num_hidden_layers."
                f" Got len(target_layer_ids)={len(self.target_layer_ids)}"
                f" draft.num_hidden_layers={self.draft.num_hidden_layers}."
            )
        if not 0 <= self.mask_token_id < self.target.vocab_size:
            raise ValueError(
                "DFlash mask_token_id must be in [0, target.vocab_size)."
                f" Got mask_token_id={self.mask_token_id}"
                f" target.vocab_size={self.target.vocab_size}."
            )
        if self.block_size > 0:
            expected_spec = self.block_size - 1
            actual_spec = self.speculative_config.num_speculative_tokens
            if actual_spec is not None and actual_spec != expected_spec:
                # Check only, never written back: the trained width is
                # resolved as a plain int by
                # :func:`resolve_dflash_num_speculative_tokens` and
                # threaded by the model.
                logger.warning(
                    "DFlash draft was trained at block_size=%d, so"
                    " num_speculative_tokens is being overridden from %d to"
                    " %d. The DFlash draft's behavior is only defined at"
                    " its trained block_size.",
                    self.block_size,
                    actual_spec,
                    expected_spec,
                )

    def resolve_block_size(self, *, default: int | None = None) -> int:
        if self.block_size > 0:
            return self.block_size
        if default is not None:
            return default
        num_spec = (
            self.resolved_num_speculative_tokens
            if self.resolved_num_speculative_tokens is not None
            else self.speculative_config.num_speculative_tokens
        )
        if num_spec is None:
            raise ValueError(
                "The DFlash draft checkpoint declares no block_size; set"
                " --num-speculative-tokens explicitly."
            )
        return num_spec + 1

    def get_kv_params(self) -> KVCacheParamInterface:
        target_kv_params = self.target.get_kv_params()
        draft_kv_params = self.draft.get_kv_params()
        return MultiKVCacheParams.from_params(
            {"target": target_kv_params, "draft": draft_kv_params}
        )

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        assert model_config.huggingface_config is not None
        assert pipeline_config.speculative is not None

        speculative_config = pipeline_config.speculative
        explicit = speculative_config.num_speculative_tokens
        if explicit is None:
            # Unset resolves to the drafter's trained width.
            resolved = resolve_dflash_num_speculative_tokens(pipeline_config)
        else:
            resolved = explicit

        assert pipeline_config.draft_model is not None
        assert pipeline_config.draft_model.huggingface_config is not None

        target_config = Llama3Config.initialize_from_config(
            pipeline_config,
            model_config.huggingface_config,
            model_config,
            max_seq_len=max_seq_len,
        )
        # Resolved at construction by PipelineConfig._resolve_max_length.
        draft_max_length = pipeline_config.draft_model.max_length
        assert draft_max_length is not None
        draft_config = Llama3Config.initialize_from_config(
            pipeline_config,
            pipeline_config.draft_model.huggingface_config,
            pipeline_config.draft_model,
            max_seq_len=draft_max_length,
        )
        if explicit is None:
            # ``initialize_from_config`` derives KV from the raw pipeline
            # config, where the unset width would bake num_draft_tokens=0.
            target_config.kv_params = replace(
                target_config.kv_params, num_draft_tokens=resolved
            )
            draft_config.kv_params = replace(
                draft_config.kv_params, num_draft_tokens=resolved
            )

        # Empty placeholder values for the DFlash-specific fields;
        # ``UnifiedDflashLlama3Model.load_model`` parses the draft HF config
        # and constructs the real values, then re-instantiates the config.
        return cls(
            target=target_config,
            draft=draft_config,
            speculative_config=speculative_config,
            resolved_num_speculative_tokens=resolved,
            target_layer_ids=[],
            mask_token_id=0,
        )

    def get_max_seq_len(self) -> int:
        return self.target.get_max_seq_len()

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        return Llama3Config.calculate_max_seq_len(
            huggingface_config, model_config
        )
