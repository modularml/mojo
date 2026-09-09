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
"""Config for the unified DSpark Gemma4 31B pipeline.

The draft checkpoint (e.g. ``RedHatAI/gemma-4-31B-it-speculator.dspark``) is
in the vLLM *speculators* format; its parsing lives in the shared
``speculators_common`` package. This module only binds the parsed draft
config to the Gemma4 target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

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
from ..speculators_common.draft_config import (
    DSparkSpeculatorsDraftConfig,
    construct_draft_kv_params,
)


@dataclass(kw_only=True)
class UnifiedDSparkGemma4_31BConfig(ArchConfigWithKVCache):
    # Mirrors the `SupportedArchitecture` registration; generic consumers
    # (`PipelineModel._resolved_encoding`, `ArchConfig.initialize`) resolve a
    # config with no explicit `quantization_encoding` through these. The
    # target may be an NVFP4 checkpoint (auto-detected from the repo's packed
    # uint8 tensors); the DSpark drafter stays bfloat16 either way, since it
    # is loaded from its own checkpoint under its own encoding.
    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float4_e2m1fnx2",
    }

    target: Gemma4ForConditionalGenerationConfig
    draft: DSparkSpeculatorsDraftConfig
    draft_kv_params: KVCacheParams
    speculative_config: SpeculativeConfig
    target_layer_ids: list[int] = field(default_factory=list)
    mask_token_id: int = 0
    block_size: int = 0
    resolved_num_speculative_tokens: int | None = None
    """Per-step draft count: explicit value if set, else the trained width."""

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

        self.resolved_num_speculative_tokens = (
            self.draft.checkpoint_draft_width(self.speculative_config)
        )

    def validate_dspark_fields(self) -> None:
        """Strict validation of the draft config against the target.

        fc in-features and the d2t / lm_head / markov_w2 row counts are
        enforced against the actual tensors at weight-load time.
        """
        n_target_layers = self.target.text_config.num_hidden_layers
        if any(
            not 0 <= i < n_target_layers for i in self.draft.target_layer_ids
        ):
            raise ValueError(
                "DSpark aux_hidden_state_layer_ids must be in"
                f" [1, {n_target_layers}] (vLLM eagle convention). Got"
                f" {list(self.draft.aux_hidden_state_layer_ids)} for a"
                f" {n_target_layers}-layer target."
            )
        if self.draft.hidden_size != self.target.text_config.hidden_size:
            raise ValueError(
                "DSpark draft hidden_size must match the target's (the"
                " fc / block-embedding / lm_head contract). Got"
                f" draft={self.draft.hidden_size}"
                f" target={self.target.text_config.hidden_size}."
            )
        if self.draft.vocab_size != self.target.text_config.vocab_size:
            raise ValueError(
                "DSpark draft embedding vocab must match the target's."
                f" Got draft={self.draft.vocab_size}"
                f" target={self.target.text_config.vocab_size}."
            )
        if self.draft.draft_vocab_size > self.target.text_config.vocab_size:
            raise ValueError(
                "DSpark draft_vocab_size must not exceed the target vocab."
                f" Got draft_vocab_size={self.draft.draft_vocab_size}"
                f" target={self.target.text_config.vocab_size}."
            )
        if not 0 <= self.mask_token_id < self.target.text_config.vocab_size:
            raise ValueError(
                "DSpark mask_token_id must be in [0, target vocab_size)."
                f" Got mask_token_id={self.mask_token_id}"
                f" vocab_size={self.target.text_config.vocab_size}."
            )

    @property
    def effective_block_size(self) -> int:
        """Anchor slot plus the resolved per-step draft count.

        The draft block is causal and width-generic: fewer mask slots
        than trained is prefix-stable (the drafted positions come out
        identical to the trained-width block's), and more than trained
        runs the extra slots as extrapolation.
        """
        num_spec = self.resolved_num_speculative_tokens
        assert num_spec is not None, (
            "num_speculative_tokens is resolved at construction"
        )
        return num_spec + 1

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
        draft_config = DSparkSpeculatorsDraftConfig.from_huggingface_config(
            draft_hf_config
        )
        # Keep the whole {target, draft} tree on one num_draft_tokens at
        # initialize time (the target leaves carry the CLI value, 0 while
        # it is still unset); the pipeline model re-derives all leaves from
        # the resolved effective block width before the KV manager is
        # built.
        draft_kv_params = construct_draft_kv_params(
            pipeline_config,
            draft_config,
            list(target_config.devices),
            num_draft_tokens=(
                pipeline_config.speculative.num_speculative_tokens or 0
            ),
        )

        config = cls(
            target=target_config,
            draft=draft_config,
            draft_kv_params=draft_kv_params,
            speculative_config=pipeline_config.speculative,
            target_layer_ids=list(draft_config.target_layer_ids),
            mask_token_id=draft_config.mask_token_id,
            block_size=draft_config.block_size,
        )
        config.validate_dspark_fields()
        return config

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
