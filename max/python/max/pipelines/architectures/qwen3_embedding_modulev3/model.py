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
"""Qwen3 Embedding pipeline model without KV caching (V3 eager API)."""

from __future__ import annotations

import functools
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, cast

from max.driver import Buffer, Device
from max.engine import InferenceSession
from max.experimental.nn.common_layers.mlp import MLP
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.nn.embedding import Embedding
from max.experimental.nn.norm import RMSNorm
from max.graph.weights import Weights, WeightsAdapter
from max.nn.transformer import ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    ModuleV3PipelineModel,
    PipelineConfig,
)
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lib.utils import parse_state_dict_from_weights
from typing_extensions import override

from .batch_processor import Qwen3EmbeddingModuleV3BatchProcessor
from .layers import (
    Qwen3AttentionNoCache,
    Qwen3Embedding,
    Qwen3EmbeddingTransformer,
    Qwen3EmbeddingTransformerBlock,
)
from .model_config import Qwen3EmbeddingConfig

logger = logging.getLogger("max.pipelines")


@dataclass
class Qwen3EmbeddingInputs(ModelInputs):
    """Input structure for Qwen3 embedding models."""

    tokens: Buffer
    """Input token IDs [total_seq_len]"""

    input_row_offsets: Buffer
    """Row offsets for ragged tensors [batch_size + 1]"""

    return_n_logits: Buffer
    """Number of logits to return (kept for interface compatibility)"""


class Qwen3EmbeddingModel(ModuleV3PipelineModel[TextContext]):
    """Qwen3 embedding pipeline model without KV caching (V3 eager API).

    Optimized for embedding generation with:
    - No KV cache overhead
    - Single-pass forward computation
    - Flash attention without cache operations
    - Last token pooling with L2 normalization
    """

    model_config_cls: ClassVar[type[Any]] = Qwen3EmbeddingConfig
    batch_processor_cls: ClassVar[
        type[Qwen3EmbeddingModuleV3BatchProcessor]
    ] = Qwen3EmbeddingModuleV3BatchProcessor

    model: Callable[..., Any]
    """Compiled model callable."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        *,
        memory_plan: MemoryPlan,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.ALL,
        max_batch_size: int = 1,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter=adapter,
            return_logits=return_logits,
            memory_plan=memory_plan,
        )
        self.model = self.load_model()

    @override
    def _load_state_dict(self) -> dict[str, Any]:
        return parse_state_dict_from_weights(
            self.pipeline_config,
            self.weights,
            self.adapter,
            hf_config=self._hf_config_for_weights(),
        )

    def _create_model_config(self, state_dict: dict[str, Any]) -> None:
        del state_dict

    def _prepare_state_dict(
        self, state_dict: dict[str, Any], model_config: Any
    ) -> dict[str, Any]:
        del model_config
        # Remove lm_head weights — embedding model doesn't use them
        state_dict = {k: v for k, v in state_dict.items() if "lm_head" not in k}

        # Qwen3-Embedding checkpoints store weights without a "model." prefix
        # (e.g. "layers.0.self_attn.q_proj.weight" instead of
        # "model.layers.0.self_attn.q_proj.weight"). The llama3_modulev3
        # adapter maps "model." → "language_model.", but when there is no
        # "model." prefix to replace, keys pass through unchanged.
        # Ensure every key carries the "language_model." prefix that the
        # compiled module tree expects.
        return {
            k if k.startswith("language_model.") else f"language_model.{k}": v
            for k, v in state_dict.items()
        }

    def _instantiate_module(self, model_config: Any) -> Qwen3Embedding:
        del model_config
        huggingface_config = self.huggingface_config

        head_dim = huggingface_config.head_dim
        norm_eps = getattr(huggingface_config, "rms_norm_eps", 1e-6)
        attention_multiplier = getattr(
            huggingface_config,
            "attention_multiplier",
            1.0 / math.sqrt(float(head_dim)),
        )

        rope = RotaryEmbedding(
            dim=huggingface_config.hidden_size,
            n_heads=huggingface_config.num_attention_heads,
            theta=huggingface_config.rope_theta,
            max_seq_len=self.max_seq_len,
            device=self.devices[0],
            head_dim=head_dim,
            interleaved=False,
        )

        create_norm = functools.partial(
            RMSNorm,
            huggingface_config.hidden_size,
            eps=norm_eps,
        )

        layers = []
        for _layer_idx in range(huggingface_config.num_hidden_layers):
            attention = Qwen3AttentionNoCache(
                rope=rope,
                num_attention_heads=huggingface_config.num_attention_heads,
                num_key_value_heads=huggingface_config.num_key_value_heads,
                hidden_size=huggingface_config.hidden_size,
                head_dim=head_dim,
                scale=attention_multiplier,
                qk_norm_eps=norm_eps,
            )

            mlp = MLP(
                hidden_dim=huggingface_config.hidden_size,
                feed_forward_length=huggingface_config.intermediate_size,
                bias=False,
            )

            block = Qwen3EmbeddingTransformerBlock(
                attention=attention,
                mlp=mlp,
                attention_norm=create_norm(),
                mlp_norm=create_norm(),
                residual_multiplier=1.0,
            )
            layers.append(block)

        embedding = Embedding(
            huggingface_config.vocab_size,
            dim=huggingface_config.hidden_size,
        )

        transformer = Qwen3EmbeddingTransformer(
            layers=layers,
            norm=create_norm(),
            embedding=embedding,
            pool_embeddings=self.pipeline_config.model.pool_embeddings,
            embedding_multiplier=1.0,
        )

        nn_model = Qwen3Embedding(transformer)
        nn_model.to(self.devices[0])
        return nn_model

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Execute the model."""
        assert isinstance(model_inputs, Qwen3EmbeddingInputs)

        model_outputs = self.model(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
        )

        return ModelOutputs(logits=cast(Buffer, model_outputs[0].driver_tensor))
