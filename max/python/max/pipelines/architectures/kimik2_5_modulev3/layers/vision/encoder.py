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

"""Encoder layers for Kimi K2.5 vision tower."""

from __future__ import annotations

from max.experimental.nn import Module, ModuleList
from max.experimental.nn.norm.layer_norm import LayerNorm
from max.experimental.tensor import Tensor

from .attention import Attention
from .mlp import MLP2
from .rotary_embedding import Rope2DPosEmbRepeated


class EncoderBlock(Module[[Tensor, Tensor, Tensor, Tensor], Tensor]):
    """Vision encoder layer with QKV-packed self-attention and MLP.

    Args:
        num_heads: Number of attention heads.
        hidden_dim: Hidden dimension of the encoder.
        mlp_dim: Inner dimension of the feed-forward MLP.
        has_bias: Whether linear projections include bias terms.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        has_bias: bool = False,
    ) -> None:
        super().__init__()
        self.norm0 = LayerNorm(dim=hidden_dim)
        self.norm1 = LayerNorm(dim=hidden_dim)
        self.attn = Attention(
            num_heads=num_heads, hidden_dim=hidden_dim, has_bias=has_bias
        )
        self.mlp = MLP2(
            dim=(hidden_dim, mlp_dim, hidden_dim), has_bias=has_bias
        )

    def forward(
        self,
        x: Tensor,
        input_row_offsets: Tensor,
        max_seq_len: Tensor,
        rope_freqs_cis: Tensor,
    ) -> Tensor:
        """Full encoder forward pass.

        Args:
            x: Packed input tensor of shape (n_patches, hidden_dim).
            input_row_offsets: Cumulative sequence lengths of shape
                (batch_size + 1,), dtype uint32.
            max_seq_len: Maximum sequence length, shape (1,), dtype uint32.
            rope_freqs_cis: Precomputed [cos, sin] pairs of shape
                (n_patches, head_dim // 2, 2).

        Returns:
            Output tensor of shape (n_patches, hidden_dim).
        """
        residual = x
        x = self.norm0(x)
        x = self.attn(x, input_row_offsets, max_seq_len, rope_freqs_cis)
        x = residual + x

        residual = x
        x = self.norm1(x)
        x = self.mlp(x)
        x = residual + x

        return x


class Encoder(Module[[Tensor, Tensor, Tensor, Tensor], Tensor]):
    """Full vision encoder.

    Wraps an initial :obj:`Rope2DPosEmbRepeated`, multiple :obj:`EncoderBlock`
    blocks, and a final :obj:`LayerNorm`.

    Args:
        num_heads: Number of attention heads.
        hidden_dim: Hidden dimension of the encoder.
        mlp_dim: Inner dimension of the feed-forward MLP found in each
            underlying :obj:`EncoderBlock`.
        num_layers: Number of encoder layers.
        rope_max_height: Maximum grid height for RoPE frequencies.
        rope_max_width: Maximum grid width for RoPE frequencies.
        rope_theta: Base for the RoPE inverse-frequency exponent.
        has_bias: Whether linear projections include bias terms.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        num_layers: int,
        rope_max_height: int,
        rope_max_width: int,
        rope_theta: float,
        has_bias: bool = False,
    ) -> None:
        super().__init__()
        self.rope_2d = Rope2DPosEmbRepeated(
            dim=hidden_dim // num_heads,
            max_height=rope_max_height,
            max_width=rope_max_width,
            theta_base=rope_theta,
        )
        self.blocks = ModuleList(
            [
                EncoderBlock(
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    mlp_dim=mlp_dim,
                    has_bias=has_bias,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNorm(dim=hidden_dim)

    def forward(
        self,
        x: Tensor,
        input_row_offsets: Tensor,
        max_seq_len: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        """Full encoder forward pass.

        Args:
            x: Packed input tensor of shape (n_patches, hidden_dim).
            input_row_offsets: Cumulative sequence lengths of shape
                (batch_size + 1,), dtype uint32.
            max_seq_len: Maximum sequence length, shape (1,), dtype uint32.
            position_ids: 1-D int tensor of flat grid indices
                (row * max_width + col) for RoPE lookup, dtype int64.

        Returns:
            Output tensor of shape (n_patches, hidden_dim).
        """
        rope_freqs_cis = self.rope_2d(position_ids)
        for block in self.blocks:
            x = block(x, input_row_offsets, max_seq_len, rope_freqs_cis)
        x = self.norm(x)
        return x
