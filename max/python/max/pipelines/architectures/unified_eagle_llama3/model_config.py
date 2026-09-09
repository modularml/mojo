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
"""Config for EAGLE Llama3 draft models."""

from __future__ import annotations

from dataclasses import dataclass
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
from transformers import AutoConfig
from typing_extensions import Self

from ..llama3.model_config import ArchConfigWithKVCache, Llama3Config


@dataclass(kw_only=True)
class UnifiedEagleLlama3Config(ArchConfigWithKVCache):
    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float32",
    }

    target: Llama3Config
    draft: Llama3Config
    speculative_config: SpeculativeConfig
    enable_structured_output: bool = False
    """When True, the graph accepts a bitmask input for grammar-constrained decoding."""
    quantization_encoding: SupportedEncoding | None = None

    def __post_init__(self) -> None:
        self.target.return_logits = ReturnLogits.VARIABLE
        self.target.return_hidden_states = ReturnHiddenStates.ALL_NORMALIZED
        self.draft.return_hidden_states = ReturnHiddenStates.LAST

        if len(self.target.devices) != len(self.draft.devices):
            raise ValueError(
                f"Target and draft must have the same number of devices. Found {len(self.target.devices)} and {len(self.draft.devices)} respectively."
            )

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
        assert pipeline_config.speculative is not None

        return cls(
            target=target_config,
            draft=draft_config,
            speculative_config=pipeline_config.speculative,
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
