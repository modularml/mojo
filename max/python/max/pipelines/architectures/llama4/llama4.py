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
"""Implements the Llama4 (text-only) model."""

from __future__ import annotations

import functools

from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.embedding import Embedding
from max.nn.kv_cache import KVCacheParamInterface
from max.nn.layer import Module
from max.nn.linear import MLP, Linear
from max.nn.moe import StackedMoE
from max.nn.moe.stacked_moe import GateUpFormat
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.nn.transformer import Transformer, TransformerBlock

from .layers import Llama4TextAttention, Llama4TextMoEGate
from .model_config import Llama4Config


def _build_feed_forward(config: Llama4Config, is_moe_layer: bool) -> Module:
    """Builds the per-layer feed-forward: MoE on MoE layers, dense MLP else.

    When ``config.quant_config`` is set (FP8), the routed experts and the shared
    expert are quantized: ``config.dtype`` is the FP8 storage dtype
    (``float8_e4m3fn``) and :class:`StackedMoE` builds FP8 expert weights with
    per-channel weight scales + dynamic per-token activation quant. The router
    gate stays bf16 (forced inside :class:`StackedMoE`). The bf16 path is
    unchanged (``quant_config`` is ``None``).
    """
    if is_moe_layer:
        return StackedMoE(
            devices=config.devices,
            hidden_dim=config.hidden_size,
            num_experts=config.num_local_experts,
            num_experts_per_token=config.num_experts_per_tok,
            moe_dim=config.intermediate_size,
            gate_cls=Llama4TextMoEGate,
            dtype=config.dtype,
            gate_up_format=GateUpFormat.CONCATENATED,
            has_bias=False,
            has_shared_experts=True,
            shared_experts_dim=config.intermediate_size,
            apply_router_weight_first=True,
            quant_config=config.quant_config,
        )
    return MLP(
        dtype=config.dtype,
        quantization_encoding=None,
        hidden_dim=config.hidden_size,
        feed_forward_length=config.intermediate_size_mlp,
        devices=config.devices,
        activation_function="silu",
        quant_config=config.quant_config,
    )


class Llama4(Transformer):
    """The Llama4 text model.

    Reuses the standard :class:`~max.nn.transformer.Transformer` for embedding,
    final norm, language-model head, and logits post-processing. Each decoder
    layer is a :class:`~max.nn.transformer.TransformerBlock` whose attention is
    the Llama4-specific :class:`Llama4TextAttention` and whose feed-forward is
    either a shared-expert :class:`~max.nn.moe.StackedMoE` (MoE layers) or a
    dense :class:`~max.nn.linear.MLP`.
    """

    def __init__(self, config: Llama4Config) -> None:
        assert len(config.devices) == 1, "Llama4 currently supports one device."
        self.config = config
        device = config.devices[0]
        # When FP8, ``config.dtype`` is the FP8 weight-storage dtype used by the
        # quantized MoE experts; everything outside the feed-forward experts
        # (attention, embedding, lm_head, norms) computes in bf16.
        compute_dtype = (
            DType.bfloat16
            if config.dtype == DType.float8_e4m3fn
            else config.dtype
        )
        norm_dtype = config.norm_dtype or compute_dtype

        # Llama4 uses the interleaved (complex) RoPE variant; the llama3-style
        # frequency scaling (when present) is applied on top.
        rope = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            head_dim=config.head_dim,
            interleaved=True,
            scaling_params=config.rope_scaling_params,
        )

        create_norm = functools.partial(
            RMSNorm,
            config.hidden_size,
            norm_dtype,
            config.rms_norm_eps,
            multiply_before_cast=False,
        )

        moe_layers = set(config.moe_layers)
        layers = [
            TransformerBlock(
                attention=Llama4TextAttention(
                    rope=rope,
                    num_attention_heads=config.num_attention_heads,
                    num_key_value_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    head_dim=config.head_dim,
                    kv_params=config.kv_params,
                    layer_idx=i,
                    dtype=compute_dtype,
                    devices=config.devices,
                    scale=config.attention_multiplier,
                    use_rope=bool(config.no_rope_layers[i]),
                    use_qk_norm=config.use_qk_norm,
                    attn_temperature_tuning=config.attn_temperature_tuning,
                    floor_scale=config.floor_scale,
                    attn_scale=config.attn_scale,
                    attention_chunk_size=config.attention_chunk_size,
                    rms_norm_eps=config.rms_norm_eps,
                    attention_bias=config.attention_bias,
                ),
                mlp=_build_feed_forward(config, i in moe_layers),
                attention_norm=create_norm(),
                mlp_norm=create_norm(),
            )
            for i in range(config.num_hidden_layers)
        ]

        embedding_layer = Embedding(
            config.vocab_size,
            config.hidden_size,
            compute_dtype,
            device,
        )
        output = Linear(
            config.hidden_size,
            config.vocab_size,
            compute_dtype,
            device,
        )
        if config.tie_word_embeddings:
            output.set_shared_weight("weight", embedding_layer.weight)

        super().__init__(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            layers=layers,
            norm=create_norm(),
            output=output,
            embedding=embedding_layer,
            kv_params=config.kv_params,
            rope=rope,
            return_logits=config.return_logits,
            return_hidden_states=config.return_hidden_states,
        )

    def input_types(
        self, kv_params: KVCacheParamInterface
    ) -> tuple[TensorType | BufferType, ...]:
        device_ref = self.config.devices[0]
        return (
            TensorType(DType.int64, shape=["total_seq_len"], device=device_ref),
            TensorType(
                DType.uint32,
                shape=["input_row_offsets_len"],
                device=device_ref,
            ),
            TensorType(
                DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
            ),
            *kv_params.get_symbolic_inputs().flatten(),
        )
