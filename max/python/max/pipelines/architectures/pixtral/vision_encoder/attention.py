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


import math

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, TensorValueLike, ops
from max.nn.layer import Module
from max.nn.linear import Linear

from .attention_utils import rotate_half


class Attention(Module):
    n_heads: int
    dim: int
    head_dim: int  # hidden_size // self.n_heads

    k_proj: Linear
    v_proj: Linear
    q_proj: Linear
    o_proj: Linear

    def __init__(
        self,
        n_heads: int,
        dim: int,
        head_dim: int,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = head_dim
        self.q_proj = Linear(
            dim, dim, dtype=dtype, device=device, has_bias=False
        )
        self.k_proj = Linear(
            dim, dim, dtype=dtype, device=device, has_bias=False
        )
        self.v_proj = Linear(
            dim, dim, dtype=dtype, device=device, has_bias=False
        )
        self.o_proj = Linear(
            dim, dim, dtype=dtype, device=device, has_bias=False
        )

    def apply_rotary_embedding(
        self,
        xq: TensorValue,
        xk: TensorValue,
        cos: TensorValue,
        sin: TensorValue,
        unsqueeze_dim: int = 0,
    ) -> tuple[TensorValue, TensorValue]:
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            xq (`TensorValueLike`): The query tensor.
            xk (`TensorValueLike`): The key tensor.
            cos (`TensorValueLike`): The cosine part of the rotary embedding.
            sin (`TensorValueLike`): The sine part of the rotary embedding.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos and
                sin so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos and sin have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos and sin broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(TensorValueLike)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = ops.unsqueeze(cos, unsqueeze_dim)
        sin = ops.unsqueeze(sin, unsqueeze_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)

        q_embed = (xq * cos) + (rotate_half(xq) * sin)
        k_embed = (xk * cos) + (rotate_half(xk) * sin)
        return q_embed, k_embed

    def attention(
        self,
        xq: TensorValue,
        xk: TensorValue,
        xv: TensorValue,
        attn_mask: TensorValueLike,
    ) -> TensorValue:
        xv = xv.transpose(1, 2)

        scale = math.sqrt(1.0 / self.head_dim)
        # xk shape = batch_size=1, n_heads=16, head_dim=64, image_seq_len=160
        scores = xq @ ops.transpose(xk, 2, 3)
        # Note, the graph compiler currently requires the order of operands
        # to be `scores * scale` in order to pattern match the fused attention
        # operator.
        # attn_mask and pixel_values are model inputs. scores is self-attention
        # scores between all patches in pixel_values.
        # attn_mask shape = ("n_images", 1, "num_patches_in_image", "num_patches_in_image")
        # scores shape = ("n_images", "n_heads", "num_patches_in_image", "num_patches_in_image")
        attn_mask = TensorValue(attn_mask)

        attn_mask = ops.rebind(
            attn_mask, (scores.shape[0], 1, scores.shape[2], scores.shape[3])
        )
        # If there's an accuracy issues, check
        # the dtypes here first.
        attn_mask = attn_mask.cast(scores.dtype)
        # This avoids the symbolic dimension mismatch issue
        scores = ops.softmax(scores * scale + attn_mask)

        return scores @ xv

    def __call__(
        self,
        x: TensorValue,
        attention_mask: TensorValueLike,
        position_embeddings: tuple[TensorValue, TensorValue],
    ) -> TensorValue:
        """Computes attention on x.

        Args:
            x: Activations with shape (batch, seq_len, dim).
            attention_mask: a mask to ensure different blocks of patches (images)
            can only attend to patches within their respective block (image).
            position_embeddings:

        Returns the result of multi-headed self attention on the input.
        """

        batch_size, n_patches = x.shape[0], x.shape[1]
        # matmul weights
        xq = self.q_proj(x)
        xk = self.k_proj(x)
        xv = self.v_proj(x)

        xq = ops.reshape(
            xq, [batch_size, n_patches, self.n_heads, self.head_dim]
        )
        xk = ops.reshape(
            xk, [batch_size, n_patches, self.n_heads, self.head_dim]
        )
        xv = ops.reshape(
            xv, [batch_size, n_patches, self.n_heads, self.head_dim]
        )

        cos, sin = position_embeddings
        xq, xk = self.apply_rotary_embedding(xq, xk, cos, sin, unsqueeze_dim=0)

        output = (
            self.attention(xq, xk, xv, attention_mask)
            .transpose(1, 2)
            .reshape([batch_size, n_patches, -1])
        )
        return self.o_proj(output)
