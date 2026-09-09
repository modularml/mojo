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
"""Custom layers for the Llama4 (text-only) model."""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import (
    flash_attention_ragged,
    rope_split_store_ragged,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.moe import MoEGate
from max.nn.rotary_embedding import RotaryEmbedding
from max.nn.stacked_linear import StackedLinear


def l2_norm(x: TensorValue, eps: float) -> TensorValue:
    """L2 normalizes ``x`` over its last axis (no learnable weight).

    Matches ``Llama4TextL2Norm``: ``x * rsqrt(mean(x^2) + eps)`` computed in
    float32 then cast back to the input dtype.
    """
    xf = x.cast(DType.float32)
    mean_sq = ops.mean(xf * xf, axis=-1)
    normed = xf * ops.rsqrt(
        mean_sq + ops.constant(eps, DType.float32, device=x.device)
    )
    return normed.cast(x.dtype)


class Llama4TextMoEGate(MoEGate):
    """Router gate for Llama4: top-k selection then per-expert sigmoid.

    Unlike a softmax gate, each selected expert's weight is an independent
    ``sigmoid`` of its router logit (matches ``Llama4Router``).
    """

    def __call__(
        self, hidden_state: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        scores = self.gate_score(hidden_state)
        topk_scores, topk_indices = ops.top_k(
            scores, k=self.num_experts_per_token, axis=-1
        )
        topk_scores = ops.sigmoid(topk_scores.cast(DType.float32)).cast(
            topk_scores.dtype
        )
        return topk_indices, topk_scores


class Llama4TextAttention(Module):
    """Llama4 attention with iRoPE, L2 QK-norm, and temperature tuning.

    A single layer is either a RoPE layer or a NoPE layer:

    - RoPE layers apply rotary embeddings (and optional L2 QK-norm) and use a
      chunked-causal mask of width ``attention_chunk_size``.
    - NoPE layers skip rotary embeddings, optionally scale the query by a
      position-dependent attention temperature, and use a full causal mask.
    """

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        head_dim: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        dtype: DType,
        devices: list[DeviceRef],
        scale: float,
        use_rope: bool,
        use_qk_norm: bool,
        attn_temperature_tuning: bool,
        floor_scale: float,
        attn_scale: float,
        attention_chunk_size: int,
        rms_norm_eps: float,
        attention_bias: bool = False,
    ) -> None:
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.n_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.kv_params = kv_params
        self.devices = devices
        self.scale = scale
        self.use_rope = use_rope
        self.use_qk_norm = use_qk_norm and use_rope
        self.attn_temperature_tuning = attn_temperature_tuning and not use_rope
        self.floor_scale = floor_scale
        self.attn_scale = attn_scale
        self.attention_chunk_size = attention_chunk_size
        self.rms_norm_eps = rms_norm_eps

        self.q_dim = head_dim * num_attention_heads
        self.kv_dim = head_dim * num_key_value_heads

        self.qkv_proj = StackedLinear(
            in_dim=hidden_size,
            out_dims=[self.q_dim, self.kv_dim, self.kv_dim],
            names=["q_proj", "k_proj", "v_proj"],
            dtype=dtype,
            device=devices[0],
            stacked=False,
            has_bias=attention_bias,
        )
        self.o_proj = Linear(
            in_dim=self.q_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=devices[0],
            has_bias=attention_bias,
        )

    def _attention_temperature(
        self,
        total_seq_len,  # noqa: ANN001
        input_row_offsets: TensorValue,
        cache_lengths: TensorValue,
        device,  # noqa: ANN001
    ) -> TensorValue:
        """Per-token query scale for NoPE layers.

        ``log(floor((pos + 1) / floor_scale) + 1) * attn_scale + 1`` over each
        token's *absolute* position. Matching the HF reference, a token's
        absolute position is its index within its sequence plus the number of
        tokens already cached for that sequence
        (``positions = arange(seq_len) + past_seen_tokens``). For the ragged
        batch this is::

            position[t] = global_idx[t]
                          + (cache_lengths[seq(t)] - row_offsets[seq(t)])

        which restarts per sequence and accounts for the KV-cache offset on
        decode steps. For a single-sequence prefill (``cache_lengths == 0``,
        one sequence) it reduces to ``[0, total_seq_len)`` exactly, matching
        the original prefill behavior.
        """
        cpu = DeviceRef.CPU()
        # ``repeat_interleave`` runs on CPU; do the integer position math there,
        # then move the final scales to the compute device (as before).
        row_offsets = input_row_offsets.to(cpu).cast(DType.int64)
        starts = row_offsets[:-1]  # per-sequence start offset in flat buffer
        lengths = row_offsets[1:] - starts  # per-sequence token count
        # Per-sequence shift from flat index to absolute position. cache_lengths
        # (uint32) is the decode start offset; cast to int64 before subtracting.
        # Its batch dim equals ``input_row_offsets_len - 1`` at runtime, but the
        # symbolic dims differ, so rebind to the row-offsets span shape.
        cache_lengths = (
            cache_lengths.to(cpu).cast(DType.int64).rebind(starts.shape)
        )
        delta = cache_lengths - starts
        per_token_delta = ops.repeat_interleave(
            delta, lengths, axis=0, out_dim=total_seq_len
        )
        stop = ops.shape_to_tensor([total_seq_len])
        global_idx = ops.range(
            ops.constant(0, DType.int64, device=cpu),
            stop[0],
            ops.constant(1, DType.int64, device=cpu),
            out_dim=total_seq_len,
            dtype=DType.int64,
            device=cpu,
        )
        positions = (global_idx + per_token_delta).cast(DType.float32)
        floor_scale = ops.constant(self.floor_scale, DType.float32, device=cpu)
        one = ops.constant(1.0, DType.float32, device=cpu)
        floored = ops.floor((positions + one) / floor_scale)
        scales = ops.log(floored + one) * self.attn_scale + 1.0
        return scales.to(device)

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        *,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        total_seq_len = x.shape[0]
        qkv = self.qkv_proj(x)

        if self.use_rope:
            if self.use_qk_norm:
                x_q, x_k, x_v = ops.split(
                    qkv, [self.q_dim, self.kv_dim, self.kv_dim], axis=-1
                )
                x_q = l2_norm(
                    x_q.reshape((-1, self.n_heads, self.head_dim)),
                    self.rms_norm_eps,
                ).reshape((-1, self.q_dim))
                x_k = l2_norm(
                    x_k.reshape((-1, self.n_kv_heads, self.head_dim)),
                    self.rms_norm_eps,
                ).reshape((-1, self.kv_dim))
                qkv = ops.concat((x_q, x_k, x_v), axis=-1)

            freqs_cis = ops.cast(freqs_cis, qkv.dtype).to(qkv.device)
            xq = rope_split_store_ragged(
                kv_params=self.kv_params,
                qkv=qkv,
                input_row_offsets=input_row_offsets,
                freqs_cis=freqs_cis,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=self.n_heads,
                interleaved=self.rope.interleaved,
            )
            xq = xq.reshape((-1, self.n_heads, self.head_dim))
            mask_variant = MHAMaskVariant.CHUNKED_CAUSAL_MASK
            local_window_size = self.attention_chunk_size
        else:
            x_q, x_k, x_v = ops.split(
                qkv, [self.q_dim, self.kv_dim, self.kv_dim], axis=-1
            )
            xq = x_q.reshape((-1, self.n_heads, self.head_dim))
            x_k = x_k.reshape((-1, self.n_kv_heads, self.head_dim))
            x_v = x_v.reshape((-1, self.n_kv_heads, self.head_dim))
            store_k_cache_ragged(
                kv_collection, x_k, input_row_offsets, layer_idx
            )
            store_v_cache_ragged(
                kv_collection, x_v, input_row_offsets, layer_idx
            )
            if self.attn_temperature_tuning:
                scales = self._attention_temperature(
                    total_seq_len,
                    input_row_offsets,
                    kv_collection.cache_lengths,
                    xq.device,
                )
                xq = xq * ops.reshape(scales, (-1, 1, 1)).cast(xq.dtype)
            mask_variant = MHAMaskVariant.CAUSAL_MASK
            local_window_size = -1

        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=mask_variant,
            scale=self.scale,
            local_window_size=local_window_size,
        )
        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)
