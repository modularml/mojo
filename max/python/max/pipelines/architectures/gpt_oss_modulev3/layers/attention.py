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

"""GptOss Attention Layer."""

from __future__ import annotations

import math

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.nn.common_layers.functional_kernels import (
    flash_attention_ragged,
    rope_split_store_ragged,
)
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.rotary_embedding import (
    YarnRotaryEmbedding,
)
from max.experimental.tensor import Tensor
from max.nn.attention import MHAMaskVariant
from max.nn.kv_cache import KVCacheParams


class GptOssAttention(Module[..., Tensor]):
    """Implementation of the distributed attention layer for the GptOss text model.

    Depending on the layer type, the attention layer can be either a full attention
    layer or a sliding window attention layer. This layer generates the attention mask
    based on the layer type.

    This layer also supports sink attention, which is a technique to improve the
    attention mechanism by adding an extra logit column that acts as an attention
    sink.
    """

    def __init__(
        self,
        *,
        rope: YarnRotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        mask_variant: MHAMaskVariant,
        scale: float | None = None,
        has_bias: bool = False,
        local_window_size: int = 1024,
    ) -> None:
        """Initializes the attention layer.

        Args:
            rope: Rotary embedding used for all attention layers (full + sliding window).
            num_attention_heads: The number of attention heads.
            num_key_value_heads: The number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache Params, including the number of kv heads, the
                head dim, and data type.
            layer_idx: Index of this layer within its KV group (not the
                global decoder index).
            linear_cls: Linear class to use for the outputs dense layer.
            scale: Value used to scale the results of the attention output.
            has_bias: Whether to use an attention bias. Defaults to False.
            qk_norm_eps: Value to use for numerical stability. Defaults to 1e-6.
        """

        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.kv_params = kv_params
        self.has_bias = has_bias
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        self.local_window_size = local_window_size
        self.mask_variant = mask_variant

        # Initialize sinks parameter for each attention head
        self.sinks = Tensor.zeros([num_attention_heads])

        self.q_weight_dim = self.kv_params.head_dim * num_attention_heads
        self.kv_weight_dim = self.kv_params.head_dim * num_key_value_heads

        self.q_proj = Linear(
            in_dim=hidden_size,
            out_dim=self.q_weight_dim,
            bias=self.has_bias,
        )
        self.k_proj = Linear(
            in_dim=hidden_size,
            out_dim=self.kv_weight_dim,
            bias=self.has_bias,
        )
        self.v_proj = Linear(
            in_dim=hidden_size,
            out_dim=self.kv_weight_dim,
            bias=self.has_bias,
        )

        self.o_proj = Linear(
            in_dim=self.q_weight_dim,
            out_dim=hidden_size,
            bias=self.has_bias,
        )

    @property
    def wqkv(self) -> Tensor:
        """The concatenation of q, k, and v weight vectors."""
        wq: Tensor = self.q_proj.weight
        wk: Tensor = self.k_proj.weight
        wv: Tensor = self.v_proj.weight
        return F.concat([wq, wk, wv], axis=0)

    @property
    def wqkv_bias(self) -> Tensor | None:
        """The concatenation of q, k, and v bias weight vectors."""
        if not self.has_bias:
            return None

        if (
            self.q_proj.bias is None
            or self.k_proj.bias is None
            or self.v_proj.bias is None
        ):
            raise ValueError(
                "Projection bias is None, but has_bias=True was specified."
            )

        return F.concat(
            [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias], axis=0
        )

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        **kwargs,
    ) -> Tensor:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())

        # QKV matmul: graph-level weight concat, then a single matmul.
        wqkv = self.wqkv
        qkv = x @ wqkv.T
        if self.wqkv_bias is not None:
            qkv = qkv + self.wqkv_bias

        # Fused rope + split + KV store.
        rope = self.rope
        freqs_cis = F.cast(rope.freqs_cis, qkv.dtype).to(qkv.device)
        xq = rope_split_store_ragged(
            kv_params=self.kv_params,
            qkv=qkv,
            input_row_offsets=kwargs["input_row_offsets"],
            freqs_cis=freqs_cis,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
            interleaved=rope.interleaved,
        )
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        # Calculate Flash Attention with sinks.
        # The sinks parameter modifies the attention computation by adding an extra
        # logit column that acts as an attention sink.
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=self.mask_variant,
            scale=self.scale,
            local_window_size=self.local_window_size,
            sink_weights=self.sinks,
        )
        attn_out = F.reshape(attn_out, shape=[total_seq_len, -1])
        ret = self.o_proj(attn_out)
        return ret
