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

"""UMT5 encoder implementation aligned with Hugging Face UMT5 behavior.

Uses module_v2 (nn.layer.Module / ops) API exclusively.
"""

from __future__ import annotations

import copy
import math

from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorValue, Weight, ops
from max.nn.embedding import Embedding
from max.nn.kernels import masked_flash_attention_gpu
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear

from .model_config import UMT5ConfigBase


class UMT5LayerNorm(Module):
    """T5-style RMSNorm (no bias, no mean subtraction).

    Uses the fused ``ops.rms_norm`` kernel (Llama-style: normalize in
    f32, cast to output dtype, then multiply by weight).
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        dtype: DType = DType.float32,
        device: DeviceRef = DeviceRef.GPU(),  # noqa: B008
    ) -> None:
        super().__init__()
        self.weight = Weight("weight", dtype, [hidden_size], device)
        self.variance_epsilon = eps

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        return ops.rms_norm(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            weight_offset=0.0,
            multiply_before_cast=False,
        )


class UMT5DenseActDense(Module):
    def __init__(
        self,
        config: UMT5ConfigBase,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.wi = Linear(
            in_dim=config.d_model,
            out_dim=config.d_ff,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.wo = Linear(
            in_dim=config.d_ff,
            out_dim=config.d_model,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        act_name = config.dense_act_fn
        if act_name == "relu":
            self._act = "relu"
        elif act_name in ("gelu_new", "gelu"):
            self._act = act_name
        else:
            raise ValueError(f"Unsupported UMT5 dense_act_fn: {act_name}")

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        hidden_states = self.wi(hidden_states)
        if self._act == "relu":
            hidden_states = ops.relu(hidden_states)
        elif self._act == "gelu_new":
            hidden_states = ops.gelu(hidden_states, approximate="tanh")
        else:
            hidden_states = ops.gelu(hidden_states)
        return self.wo(hidden_states)


class UMT5DenseGatedActDense(Module):
    def __init__(
        self,
        config: UMT5ConfigBase,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.wi_0 = Linear(
            in_dim=config.d_model,
            out_dim=config.d_ff,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.wi_1 = Linear(
            in_dim=config.d_model,
            out_dim=config.d_ff,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.wo = Linear(
            in_dim=config.d_ff,
            out_dim=config.d_model,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        act_name = config.dense_act_fn
        if act_name == "relu":
            self._act = "relu"
        elif act_name in ("gelu_new", "gelu"):
            self._act = act_name
        else:
            raise ValueError(f"Unsupported UMT5 dense_act_fn: {act_name}")

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        gated = self.wi_0(hidden_states)
        if self._act == "relu":
            gated = ops.relu(gated)
        elif self._act == "gelu_new":
            gated = ops.gelu(gated, approximate="tanh")
        else:
            gated = ops.gelu(gated)
        linear = self.wi_1(hidden_states)
        return self.wo(gated * linear)


class UMT5LayerFF(Module):
    def __init__(
        self,
        config: UMT5ConfigBase,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        if config.is_gated_act:
            self.DenseReluDense: UMT5DenseGatedActDense | UMT5DenseActDense = (
                UMT5DenseGatedActDense(config, dtype=dtype, device=device)
            )
        else:
            self.DenseReluDense = UMT5DenseActDense(
                config, dtype=dtype, device=device
            )
        self.layer_norm = UMT5LayerNorm(
            config.d_model,
            eps=config.layer_norm_epsilon,
            dtype=config.dtype,
            device=device,
        )

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        forwarded = self.layer_norm(hidden_states)
        forwarded = self.DenseReluDense(forwarded)
        return hidden_states + forwarded


class UMT5Attention(Module):
    def __init__(
        self,
        config: UMT5ConfigBase,
        has_relative_attention_bias: bool = False,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = (
            config.relative_attention_num_buckets
        )
        self.relative_attention_max_distance = (
            config.relative_attention_max_distance
        )
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self._dtype = config.dtype
        self._device = device

        self.q = Linear(
            in_dim=self.d_model,
            out_dim=self.inner_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.k = Linear(
            in_dim=self.d_model,
            out_dim=self.inner_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.v = Linear(
            in_dim=self.d_model,
            out_dim=self.inner_dim,
            dtype=dtype,
            device=device,
            has_bias=False,
        )
        self.o = Linear(
            in_dim=self.inner_dim,
            out_dim=self.d_model,
            dtype=dtype,
            device=device,
            has_bias=False,
        )

        if self.has_relative_attention_bias:
            self.relative_attention_bias = Embedding(
                vocab_size=self.relative_attention_num_buckets,
                hidden_dim=self.n_heads,
                dtype=dtype,
                device=device,
            )

    def _relative_position_bucket(
        self,
        relative_position: TensorValue,
    ) -> TensorValue:
        num_buckets = self.relative_attention_num_buckets
        max_distance = self.relative_attention_max_distance

        dev = self._device
        if not self.is_decoder:
            num_buckets = num_buckets // 2
            is_positive = ops.greater(relative_position, 0)
            relative_buckets = ops.cast(is_positive, DType.int32) * num_buckets
            relative_position = ops.abs(relative_position)
        else:
            relative_buckets = ops.broadcast_to(
                ops.constant(0, dtype=DType.int32, device=dev),
                relative_position.shape,
            )
            relative_position = -ops.min(relative_position, 0)

        max_exact = num_buckets // 2
        is_small = ops.greater(
            ops.constant(max_exact, dtype=DType.int32, device=dev).broadcast_to(
                relative_position.shape
            ),
            relative_position,
        )

        scale = (num_buckets - max_exact) / math.log(max_distance / max_exact)
        rel_pos_float = ops.cast(relative_position, DType.float32)
        val_log = ops.log(rel_pos_float / float(max_exact))
        relative_position_if_large = (
            ops.cast(val_log * scale, DType.int32) + max_exact
        )
        max_val = ops.constant(
            num_buckets - 1, dtype=DType.int32, device=dev
        ).broadcast_to(relative_position_if_large.shape)
        relative_position_if_large = ops.where(
            ops.greater(relative_position_if_large, max_val),
            max_val,
            relative_position_if_large,
        )

        return relative_buckets + ops.where(
            is_small, relative_position, relative_position_if_large
        )

    def _compute_bias(
        self,
        query_length: int | Dim,
        key_length: int | Dim,
    ) -> TensorValue:
        context_position = ops.range(
            0,
            query_length,
            1,
            dtype=DType.int32,
            device=self._device,
        )
        context_position = ops.unsqueeze(context_position, 1)

        memory_position = ops.range(
            0,
            key_length,
            1,
            dtype=DType.int32,
            device=self._device,
        )
        memory_position = ops.unsqueeze(memory_position, 0)
        relative_position = memory_position - context_position

        relative_position_bucket = self._relative_position_bucket(
            relative_position
        )
        values = self.relative_attention_bias(relative_position_bucket)
        # values: [query_length, key_length, n_heads]
        values = ops.permute(values, [2, 0, 1])
        values = ops.unsqueeze(values, 0)
        # values: [1, n_heads, query_length, key_length]
        return values

    def __call__(
        self,
        hidden_states: TensorValue,
        attention_mask: TensorValue | None = None,
    ) -> TensorValue:
        batch_size = hidden_states.shape[0]
        seq_length = hidden_states.shape[1]

        # Project Q, K, V directly to BSHD. masked_flash_attention_gpu
        # takes BSHD as-is, so no permute to BHSD is needed.
        query_states = ops.reshape(
            self.q(hidden_states),
            [batch_size, seq_length, self.n_heads, self.key_value_proj_dim],
        )
        key_states = ops.reshape(
            self.k(hidden_states),
            [batch_size, seq_length, self.n_heads, self.key_value_proj_dim],
        )
        value_states = ops.reshape(
            self.v(hidden_states),
            [batch_size, seq_length, self.n_heads, self.key_value_proj_dim],
        )

        # Build a combined additive mask: position_bias + attention_mask.
        # position_bias is [1, H, S, S]; attention_mask is [B, 1, 1, S]
        # in the model dtype. Broadcasting produces [B, H, S, S].
        if self.has_relative_attention_bias:
            mask = self._compute_bias(seq_length, seq_length)
        else:
            # Defensive fallback: every UMT5 encoder layer has
            # has_relative_attention_bias=True in practice.
            mask = ops.broadcast_to(
                ops.constant(0, dtype=self._dtype, device=self._device),
                [batch_size, self.n_heads, seq_length, seq_length],
            )
        if attention_mask is not None:
            mask = mask + attention_mask

        # UMT5/T5 does NOT apply 1/sqrt(d) scaling, so scale=1.0.
        attn_output = masked_flash_attention_gpu(
            query_states, key_states, value_states, mask, scale=1.0
        )

        # [B, S, H, D] -> [B, S, H*D]
        attn_output = ops.reshape(
            attn_output, [batch_size, seq_length, self.inner_dim]
        )
        return self.o(attn_output)


class UMT5LayerSelfAttention(Module):
    def __init__(
        self,
        config: UMT5ConfigBase,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.SelfAttention = UMT5Attention(
            config,
            has_relative_attention_bias=True,
            dtype=dtype,
            device=device,
        )
        self.layer_norm = UMT5LayerNorm(
            config.d_model,
            eps=config.layer_norm_epsilon,
            dtype=config.dtype,
            device=device,
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        attention_mask: TensorValue | None = None,
    ) -> TensorValue:
        normed = self.layer_norm(hidden_states)
        attn_output = self.SelfAttention(normed, attention_mask)
        return hidden_states + attn_output


class UMT5Block(Module):
    def __init__(
        self,
        config: UMT5ConfigBase,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        # Use LayerList with index 0 = self-attn, index 1 = FF
        # to match HF weight key paths: block.{i}.layer.0 / block.{i}.layer.1
        self.layer = LayerList(
            [
                UMT5LayerSelfAttention(config, dtype=dtype, device=device),
                UMT5LayerFF(config, dtype=dtype, device=device),
            ]
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        attention_mask: TensorValue | None = None,
    ) -> TensorValue:
        hidden_states = self.layer[0](hidden_states, attention_mask)
        hidden_states = self.layer[1](hidden_states)
        return hidden_states


class UMT5Stack(Module):
    def __init__(
        self,
        config: UMT5ConfigBase,
        embed_tokens: Embedding,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.embed_tokens = embed_tokens
        self.block = LayerList(
            [
                UMT5Block(config, dtype=dtype, device=device)
                for _ in range(config.num_layers)
            ]
        )
        self.final_layer_norm = UMT5LayerNorm(
            config.d_model,
            eps=config.layer_norm_epsilon,
            dtype=config.dtype,
            device=device,
        )
        self._dtype = config.dtype

    def __call__(
        self,
        input_ids: TensorValue,
        attention_mask: TensorValue | None = None,
    ) -> TensorValue:
        hidden_states = self.embed_tokens(input_ids)

        # Build causal mask from attention_mask [B, S]
        causal_mask: TensorValue | None = None
        if attention_mask is not None:
            # (1 - mask) * dtype_min → masked positions get large negative
            _DTYPE_MIN: dict[DType, float] = {
                DType.float16: -65504.0,
                DType.bfloat16: -3.3895314e38,
                DType.float32: -3.4028235e38,
            }
            mask_min = _DTYPE_MIN.get(self._dtype, -3.4028235e38)
            mask_float = ops.cast(attention_mask, hidden_states.dtype)
            causal_mask = (1.0 - mask_float) * mask_min
            # [B, S] -> [B, 1, 1, S] for broadcasting with [B, H, S, S]
            causal_mask = ops.unsqueeze(ops.unsqueeze(causal_mask, 1), 1)

        for block in self.block:
            hidden_states = block(hidden_states, causal_mask)

        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class UMT5EncoderModel(Module):
    """UMT5 encoder using module_v2 (ops-based) API."""

    def __init__(
        self,
        config: UMT5ConfigBase,
        *,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        # Parse feed_forward config
        act_info = config.feed_forward_proj.split("-")
        config.dense_act_fn = act_info[-1]
        config.is_gated_act = act_info[0] == "gated"
        if (len(act_info) > 1 and act_info[0] != "gated") or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {config.feed_forward_proj} is not valid."
            )
        if config.feed_forward_proj == "gated-gelu":
            config.dense_act_fn = "gelu_new"

        self.shared = Embedding(
            vocab_size=config.vocab_size,
            hidden_dim=config.d_model,
            dtype=dtype,
            device=device,
        )

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = UMT5Stack(
            encoder_config, self.shared, dtype=dtype, device=device
        )

    def __call__(
        self,
        input_ids: TensorValue,
        attention_mask: TensorValue | None = None,
    ) -> TensorValue:
        return self.encoder(input_ids, attention_mask)


__all__ = ["UMT5EncoderModel"]
