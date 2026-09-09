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
"""Qwen3 Embedding pipeline model without KV caching."""

from __future__ import annotations

import functools
import logging
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, ops
from max.graph.weights import Weights, WeightsAdapter
from max.nn.embedding import Embedding
from max.nn.linear import MLP, Linear
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import (
    Llama3RopeScalingParams,
    Llama3RotaryEmbedding,
)
from max.nn.transformer import ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    GraphPipelineModel,
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
)
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from max.pipelines.lib.utils import parse_state_dict_from_weights
from typing_extensions import override

from .batch_processor import Qwen3EmbeddingBatchProcessor
from .layers import (
    Qwen3AttentionNoCache,
    Qwen3EmbeddingTransformer,
    Qwen3EmbeddingTransformerBlock,
    last_token_pool,
    normalize_embeddings,
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


class Qwen3EmbeddingModel(GraphPipelineModel[TextContext]):
    """Qwen3 embedding pipeline model without KV caching.

    This model is optimized for embedding generation with:
    - No KV cache overhead
    - Single-pass forward computation
    - Flash attention without cache operations
    - Last token pooling with L2 normalization
    """

    model_config_cls: ClassVar[type[Qwen3EmbeddingConfig]] = (
        Qwen3EmbeddingConfig
    )
    batch_processor_cls: ClassVar[type[Qwen3EmbeddingBatchProcessor]] = (
        Qwen3EmbeddingBatchProcessor
    )

    model: Model
    """Compiled and initialized model."""

    norm_method: Literal["rms_norm", "layer_norm"] = "rms_norm"
    """Normalization method."""

    attention_bias: bool = False
    """Whether to use attention bias."""

    state_dict: dict[str, Any]
    """Model weights."""

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
        """Initialize the Qwen3 embedding pipeline model.

        Args:
            pipeline_config: Pipeline configuration
            session: Inference session
            devices: List of devices
            kv_cache_config: KV cache configuration
            weights: Model weights
            adapter: Optional weight adapter
            return_logits: Return logits mode
        """
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter=adapter,
            return_logits=return_logits,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )
        self.model = self.load_model(session)

    @override
    def _load_state_dict(self) -> dict[str, Any]:
        return parse_state_dict_from_weights(
            self.pipeline_config,
            self.weights,
            self.adapter,
            hf_config=self._hf_config_for_weights(),
        )

    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: Any,
    ) -> tuple[Graph, dict[str, Any]]:
        del session, model_config
        dtype = self.dtype
        device_refs = [DeviceRef.from_device(d) for d in self.devices]

        # Create RoPE
        head_dim = self.huggingface_config.head_dim
        rope_scaling_params: Llama3RopeScalingParams | None = None
        rope_scaling = getattr(self.huggingface_config, "rope_scaling", None)
        if rope_scaling is not None:
            rope_type = rope_scaling.get("type") or rope_scaling.get(
                "rope_type"
            )
            if rope_type == "llama3":
                rope_scaling_params = Llama3RopeScalingParams(
                    factor=rope_scaling["factor"],
                    low_freq_factor=rope_scaling["low_freq_factor"],
                    high_freq_factor=rope_scaling["high_freq_factor"],
                    orig_max_position=rope_scaling[
                        "original_max_position_embeddings"
                    ],
                )
        rope = Llama3RotaryEmbedding(
            dim=self.huggingface_config.hidden_size,
            n_heads=self.huggingface_config.num_attention_heads,
            theta=get_rope_theta(self.huggingface_config),
            max_seq_len=self.max_seq_len,
            head_dim=head_dim,
            interleaved=False,  # Qwen3 uses non-interleaved RoPE
            scaling_params=rope_scaling_params,
        )

        # Calculate Qwen3-specific attention multiplier
        attention_multiplier = getattr(
            self.huggingface_config,
            "attention_multiplier",
            1.0 / math.sqrt(float(head_dim)),
        )

        # Create normalization layer
        norm_eps = getattr(self.huggingface_config, "rms_norm_eps", 1e-6)
        create_norm = functools.partial(
            RMSNorm,
            self.huggingface_config.hidden_size,
            dtype=dtype,
            eps=norm_eps,
            multiply_before_cast=False,
        )

        # Create transformer layers
        layers = []
        for _layer_idx in range(self.huggingface_config.num_hidden_layers):
            # Create attention layer
            attention = Qwen3AttentionNoCache(
                rope=rope,
                num_attention_heads=self.huggingface_config.num_attention_heads,
                num_key_value_heads=self.huggingface_config.num_key_value_heads,
                hidden_size=self.huggingface_config.hidden_size,
                head_dim=head_dim,
                dtype=dtype,
                devices=device_refs,
                scale=attention_multiplier,
                qk_norm_eps=norm_eps,
            )

            # Create MLP
            mlp = MLP(
                dtype=dtype,
                quantization_encoding=None,
                hidden_dim=self.huggingface_config.hidden_size,
                feed_forward_length=self.huggingface_config.intermediate_size,
                devices=device_refs,
            )

            # Create transformer block
            block = Qwen3EmbeddingTransformerBlock(
                attention=attention,
                mlp=mlp,
                attention_norm=create_norm(),
                mlp_norm=create_norm(),
                residual_multiplier=1.0,
            )
            layers.append(block)

        # Create embedding layer
        embed_weight = state_dict.get("embed_tokens.weight")
        embedding_quantization = (
            embed_weight.quantization_encoding if embed_weight else None
        )
        embedding_dtype = dtype if not embedding_quantization else DType.uint8

        embedding_layer = Embedding(
            self.huggingface_config.vocab_size,
            self.huggingface_config.hidden_size,
            embedding_dtype,
            device_refs[0],
            quantization_encoding=embedding_quantization,
        )

        # Create output layer (for weight sharing with embedding)
        output = Linear(
            self.huggingface_config.hidden_size,
            self.huggingface_config.vocab_size,
            embedding_dtype,
            device_refs[0],
            quantization_encoding=embedding_quantization,
        )

        # Share weights if configured
        if getattr(self.huggingface_config, "tie_word_embeddings", False):
            output.set_shared_weight("weight", embedding_layer.weight)

        # Create transformer
        nn_model = Qwen3EmbeddingTransformer(
            dim=self.huggingface_config.hidden_size,
            n_heads=self.huggingface_config.num_attention_heads,
            layers=layers,
            norm=create_norm(),
            output=output,
            embedding=embedding_layer,
            rope=rope,
            embedding_multiplier=1.0,
            device=device_refs[0],
        )

        # Load weights into model
        nn_model.load_state_dict(
            state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=getattr(
                self.huggingface_config, "tie_word_embeddings", False
            ),
        )

        weights_registry = nn_model.state_dict()

        # Build graph
        graph_inputs = nn_model.input_types()

        with Graph("qwen3_embedding", input_types=graph_inputs) as graph:
            tokens, input_row_offsets, return_n_logits = graph.inputs

            # Forward pass - returns hidden_states
            hidden_states = nn_model(
                tokens.tensor,
                input_row_offsets.tensor,
                return_n_logits.tensor,
            )

            if self.pipeline_config.model.pool_embeddings:
                # Apply last token pooling
                embeddings = last_token_pool(
                    hidden_states, input_row_offsets.tensor
                )

                # Apply L2 normalization
                embeddings_normalized = normalize_embeddings(embeddings)

                # Output pooled and normalized embeddings [batch_size, hidden_size]
                graph.output(embeddings_normalized)
            else:
                # Return raw hidden states [total_seq_len, hidden_size]
                hidden_states_f32 = ops.cast(hidden_states, DType.float32)
                graph.output(hidden_states_f32)

        return graph, weights_registry

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        """Execute the model.

        Args:
            model_inputs: Model inputs

        Returns:
            Model outputs with embeddings in the logits field
        """
        assert isinstance(model_inputs, Qwen3EmbeddingInputs)

        # Execute model
        model_outputs = self.model.execute(
            model_inputs.tokens,
            model_inputs.input_row_offsets,
            model_inputs.return_n_logits,
        )

        # Return embeddings in logits field for pipeline compatibility
        assert isinstance(model_outputs[0], Buffer)
        return ModelOutputs(logits=model_outputs[0])
