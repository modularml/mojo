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
"""An opaque KV Cache optimized vanilla attention mechanism, with Mask Variants provided inside the Kernel."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops

from ..kernels import (
    flash_attention_ragged,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from ..kv_cache import (
    KVCacheParams,
    MHAKVCacheParams,
    PagedCacheValues,
)
from ..layer import Module
from ..linear import Linear
from ..stacked_linear import StackedLinear
from .mask_config import MHAMaskVariant


@dataclass
class RaggedAttention(Module):
    """Layer that computes the self attention score for ragged inputs."""

    def __init__(
        self,
        *,
        mask_variant: MHAMaskVariant,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        devices: list[DeviceRef] | None = None,
        dtype: DType = DType.float32,
        linear_cls: Callable[..., Linear] = Linear,
        stacked_qkv: bool = False,
        scale: float | None = None,
        has_bias: bool = False,
        clip_qkv: float | None = None,
    ) -> None:
        """Initializes the attention layer.

        Args:
            rope: The rope layer to borrow the freqs_cis value from.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: Number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache Params, including the number of kv heads, the head dim, and data type.
            dtype: DType of the
            devices: Device to place the weights and run the computation. If
                multiple are provided, the first device is used.
            linear_cls: Linear class to use for the outputs dense layer.
            stacked_qkv: Whether the weights are stacked together.
            scale: Value used to scale the results of the attention output.
            has_bias: Whether to use an attention bias.
            clip_qkv: If provided, the QKV weights are clamped between
                `[-clip_qkv, clip_qkv]`
        """
        if has_bias:
            raise ValueError(
                "RaggedAttention does not yet support `has_bias=True`."
            )

        super().__init__()
        self.mask_variant = mask_variant
        self.n_heads = num_attention_heads

        self.kv_params = kv_params
        self.has_bias = has_bias
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        self.devices = devices or [DeviceRef.CPU()]

        kv_weight_dim = (
            hidden_size // num_attention_heads
        ) * num_key_value_heads

        self.stacked_qkv = stacked_qkv

        self.qkv_proj = StackedLinear(
            in_dim=hidden_size,
            out_dims=[hidden_size, kv_weight_dim, kv_weight_dim],
            names=["q_proj", "k_proj", "v_proj"],
            dtype=dtype,
            device=self.devices[0],
            stacked=stacked_qkv,
            has_bias=has_bias,
            linear_cls=linear_cls,
            clip_weight=clip_qkv,
        )

        self.o_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=hidden_size,
            dtype=dtype,
            device=self.devices[0],
        )

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        **kwargs,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        # QKV matmul.
        qkv = self.qkv_proj(x)

        # Split into Q, K, V and store K/V to cache.
        head_dim = self.kv_params.head_dim
        assert isinstance(self.kv_params, MHAKVCacheParams)
        n_kv_heads = self.kv_params.n_kv_heads
        q_dim = self.n_heads * head_dim
        kv_dim = n_kv_heads * head_dim
        x_q, x_k, x_v = ops.split(qkv, [q_dim, kv_dim, kv_dim], axis=-1)
        x_k = x_k.reshape((-1, n_kv_heads, head_dim))
        x_v = x_v.reshape((-1, n_kv_heads, head_dim))
        store_k_cache_ragged(
            kv_collection, x_k, kwargs["input_row_offsets"], layer_idx
        )
        store_v_cache_ragged(
            kv_collection, x_v, kwargs["input_row_offsets"], layer_idx
        )
        xq = x_q.reshape((-1, self.n_heads, head_dim))

        # Calculate Flash Attention.
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=self.mask_variant,
            scale=self.scale,
        )

        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])

        return self.o_proj(attn_out)
