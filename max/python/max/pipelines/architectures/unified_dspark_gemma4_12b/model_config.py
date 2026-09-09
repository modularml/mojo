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
"""Config for the unified DSpark Gemma4 pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheParams,
    MultiKVCacheParams,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib.config import (
    MAXModelConfig,
    PipelineConfig,
    SpeculativeConfig,
)
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithKVCache,
)
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers import AutoConfig
from typing_extensions import Self

from ..gemma4.model_config import Gemma4ForConditionalGenerationConfig
from .dspark_gemma4 import DSparkGemma4DraftConfig, _get

logger = logging.getLogger("max.pipelines")


def dspark_draft_width(
    speculative: SpeculativeConfig,
    draft_huggingface_config: Any,
    *,
    warn: bool = True,
) -> int:
    """Returns the resolved DSpark draft width from the draft HF config."""
    raw_block_size = _get(draft_huggingface_config, "block_size", None)
    block_size = int(raw_block_size) if raw_block_size is not None else 0
    if block_size <= 0:
        if speculative.num_speculative_tokens is None:
            raise ValueError(
                "The DSpark draft config declares no block_size; set"
                " --num-speculative-tokens explicitly."
            )
        return speculative.num_speculative_tokens
    actual_spec = speculative.num_speculative_tokens
    if warn and actual_spec is not None and actual_spec != block_size:
        logger.warning(
            "DSpark draft was trained at block_size=%d and drafts at"
            " every block position, so num_speculative_tokens is"
            " being overridden from %d to %d.",
            block_size,
            actual_spec,
            block_size,
        )
    return block_size


def resolve_dspark_num_speculative_tokens(
    pipeline_config: PipelineConfig,
    *,
    warn: bool = True,
) -> int:
    """Returns the resolved DSpark draft width.

    DSpark drafts at every block position, so the width is the drafter's
    trained ``block_size`` itself (no ``- 1``); a mismatching explicit
    ``num_speculative_tokens`` is overridden with a warning. When the draft
    config declares no ``block_size``, an explicit
    ``--num-speculative-tokens`` is required and returned as-is. Pure: the
    caller's config is never mutated or copied.
    """
    assert pipeline_config.speculative is not None
    assert pipeline_config.draft_model is not None
    return dspark_draft_width(
        pipeline_config.speculative,
        pipeline_config.draft_model.huggingface_config,
        warn=warn,
    )


def construct_draft_kv_params(
    pipeline_config: PipelineConfig,
    draft_config: DSparkGemma4DraftConfig,
    devices: list[DeviceRef],
    *,
    num_draft_tokens: int,
) -> KVCacheParams:
    """Builds the draft KV leaf from the DSpark drafter's own geometry."""
    assert pipeline_config.speculative is not None
    return pipeline_config.model.kv_cache.to_params(
        dtype=DType.bfloat16,
        n_kv_heads=draft_config.num_key_value_heads,
        head_dim=draft_config.head_dim,
        num_layers=draft_config.num_hidden_layers,
        devices=devices,
        data_parallel_degree=pipeline_config.model.data_parallel_degree,
        speculative_method=pipeline_config.speculative.speculative_method,
        num_draft_tokens=num_draft_tokens,
    )


@dataclass
class Gemma4DSparkDraftArchConfig:
    """Thin ArchConfig for the standalone ``Gemma4DSparkModel`` registration.

    A DSpark checkpoint is only ever served as ``--draft-model`` next to a
    Gemma4 target (the registry rewrites the pair to
    ``UnifiedDSparkGemma4_12BForCausalLM``), so the registry needs just the
    draft-side max-sequence-length clamp from this config.
    """

    max_position_embeddings: int = 262144

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Gemma4DSparkDraftArchConfig:
        del pipeline_config
        assert model_config is not None
        huggingface_config = model_config.huggingface_config
        assert huggingface_config is not None
        max_pos = getattr(huggingface_config, "max_position_embeddings", None)
        if isinstance(huggingface_config, dict):
            max_pos = huggingface_config.get("max_position_embeddings")
        assert max_pos is not None, (
            "DSpark draft config is missing max_position_embeddings."
        )
        return cls(max_position_embeddings=int(max_pos))

    def get_max_seq_len(self) -> int:
        return self.max_position_embeddings

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        del model_config
        max_pos = getattr(huggingface_config, "max_position_embeddings", None)
        if isinstance(huggingface_config, dict):
            max_pos = huggingface_config.get("max_position_embeddings")
        assert max_pos is not None, (
            "DSpark draft config is missing max_position_embeddings."
        )
        return int(max_pos)


@dataclass(kw_only=True)
class UnifiedDSparkGemma4_12BConfig(ArchConfigWithKVCache):
    # Mirrors the `SupportedArchitecture` registration; generic consumers
    # (`PipelineModel._resolved_encoding`, `ArchConfig.initialize`) resolve a
    # config with no explicit `quantization_encoding` through these.
    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"bfloat16"}

    target: Gemma4ForConditionalGenerationConfig
    draft: DSparkGemma4DraftConfig
    draft_kv_params: KVCacheParams
    speculative_config: SpeculativeConfig
    target_layer_ids: list[int] = field(default_factory=list)
    mask_token_id: int = 0
    block_size: int = 0

    def __post_init__(self) -> None:
        self.target.text_config.return_logits = ReturnLogits.VARIABLE
        self.target.text_config.return_hidden_states = (
            ReturnHiddenStates.SELECTED_LAYERS
        )
        self.target.text_config.target_layer_ids = list(self.target_layer_ids)

        if len(self.target.devices) != 1:
            raise ValueError(
                "DSpark currently supports a single device only. Got"
                f" {len(self.target.devices)} devices."
            )

    def validate_dspark_fields(self) -> None:
        """Strict validation of the DSpark fields parsed from the draft HF
        config. Unlike DFlash, ALL ``block_size`` positions produce drafts,
        so ``num_speculative_tokens`` must equal ``block_size`` (no ``- 1``).
        """
        if not self.target_layer_ids:
            raise ValueError(
                "DSpark requires non-empty target_layer_ids (taps consumed"
                " by the drafter's fc projection)."
            )
        n_target_layers = self.target.text_config.num_hidden_layers
        if any(not 0 <= i < n_target_layers for i in self.target_layer_ids):
            raise ValueError(
                "DSpark target_layer_ids must be valid target layer indices."
                f" Got {self.target_layer_ids} for a {n_target_layers}-layer"
                " target."
            )
        if not 0 <= self.mask_token_id < self.draft.vocab_size:
            raise ValueError(
                "DSpark mask_token_id must be in [0, vocab_size). Got"
                f" mask_token_id={self.mask_token_id}"
                f" vocab_size={self.draft.vocab_size}."
            )
        if self.target.text_config.vocab_size != self.draft.vocab_size:
            raise ValueError(
                "DSpark draft vocab must match the target's (embed/lm_head"
                " are shared)."
                f" Got draft={self.draft.vocab_size}"
                f" target={self.target.text_config.vocab_size}."
            )
        if self.block_size > 0:
            expected_spec = self.block_size
            actual_spec = self.speculative_config.num_speculative_tokens
            if actual_spec is not None and actual_spec != expected_spec:
                # Check only, never written back: the trained width is
                # resolved as a plain int by
                # :func:`resolve_dspark_num_speculative_tokens` and
                # threaded by the model.
                logger.warning(
                    "DSpark draft was trained at block_size=%d and drafts at"
                    " every block position, so num_speculative_tokens is"
                    " being overridden from %d to %d.",
                    self.block_size,
                    actual_spec,
                    expected_spec,
                )

    def resolve_block_size(self, *, default: int | None = None) -> int:
        if self.block_size > 0:
            return self.block_size
        if default is not None:
            return default
        num_spec = self.speculative_config.num_speculative_tokens
        if num_spec is None:
            raise ValueError(
                "The DSpark draft config declares no block_size; set"
                " --num-speculative-tokens explicitly."
            )
        return num_spec

    @property
    def devices(self) -> list[DeviceRef]:
        """Devices the unified model runs on (the memory-planner protocol)."""
        return list(self.target.devices)

    def get_kv_params(self) -> KVCacheParamInterface:
        return MultiKVCacheParams.from_params(
            {
                "target": self.target.get_kv_params(),
                "draft": self.draft_kv_params,
            }
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
        assert pipeline_config.draft_model is not None
        draft_hf_config = pipeline_config.draft_model.huggingface_config
        assert draft_hf_config is not None
        assert pipeline_config.speculative is not None

        target_config = (
            Gemma4ForConditionalGenerationConfig.initialize_from_config(
                pipeline_config,
                model_config.huggingface_config,
                max_seq_len=max_seq_len,
            )
        )
        draft_config = DSparkGemma4DraftConfig.from_huggingface_config(
            draft_hf_config
        )
        # Keep the whole {target, draft} tree on one num_draft_tokens at
        # initialize time (the target leaves carry the CLI value, 0 while
        # it is still unset);
        # ``UnifiedDSparkGemma4_12BModel._create_model_config`` re-derives all
        # leaves from the draft's trained block_size before the KV manager
        # is built.
        draft_kv_params = construct_draft_kv_params(
            pipeline_config,
            draft_config,
            list(target_config.devices),
            num_draft_tokens=(
                pipeline_config.speculative.num_speculative_tokens or 0
            ),
        )

        return cls(
            target=target_config,
            draft=draft_config,
            draft_kv_params=draft_kv_params,
            speculative_config=pipeline_config.speculative,
            target_layer_ids=list(draft_config.target_layer_ids),
            mask_token_id=draft_config.mask_token_id,
            block_size=draft_config.block_size,
        )

    def get_max_seq_len(self) -> int:
        return self.target.get_max_seq_len()

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        return Gemma4ForConditionalGenerationConfig.calculate_max_seq_len(
            huggingface_config, model_config
        )


def gemma4_dspark_12b_width(
    speculative: SpeculativeConfig,
    target_huggingface_config: Any,
    draft_huggingface_config: Any,
) -> int:
    """Returns the block size: on the 12B draft every slot drafts a token."""
    del target_huggingface_config
    if draft_huggingface_config is None:
        raise ValueError("DSpark requires a draft model.")
    return dspark_draft_width(speculative, draft_huggingface_config)
