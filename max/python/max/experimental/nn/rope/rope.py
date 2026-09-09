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
"""The rope embedding used within the model."""

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.tensor import Tensor, defaults, defaults_like
from max.graph import Dim, DimLike

from ..module import Module, module_dataclass


def theta(dim: int, base: float) -> Tensor:
    """Returns inverse-exponential frequencies for rotary positional embeddings.

    See 'Roformer: Enhanced Transformer with Rotary Embedding'
    (arxiv.org/pdf/2104.09864).

    Args:
        dim: The embedding dimension. By convention each component
            of the complex valued embedding is considered its own dim
            in the embedding, so output has shape ``dim // 2``.
        base: Scaling factor for the frequency.

    Returns:
        The 1D frequency tensor with shape ``(dim // 2,)``.
    """
    dtype, _ = defaults()
    # Use float64 for higher range in the exponential
    iota = Tensor.arange(dim, step=2, dtype=DType.float64)
    frequencies = base ** (-iota / dim)
    return frequencies.cast(dtype)


def embed(
    frequencies: Tensor,
    max_sequence_length: int,
) -> Tensor:
    """Computes the frequency tensor for complex exponentials in cis representation.

    Uses ``cos(s) + i * sin(s)`` for the given sequence length.

    Args:
        frequencies: Frequencies to embed in the cyclic space
        max_sequence_length: The number of positional embeddings to compute

    Returns:
        The embedded frequency tensor with shape
        ``(max_sequence_length, n / 2, 2)``.
    """
    with defaults_like(frequencies):
        t = Tensor.arange(max_sequence_length, dtype=DType.float64)
        # [max_seq_len*2, n // 2]
        freqs = F.outer(t, frequencies).cast(frequencies.dtype)
        # [max_seq_len*2, n // 2, 2]
        return F.stack([F.cos(freqs), F.sin(freqs)], axis=-1)


def positional_embedding(
    dim: int, base: float, max_sequence_length: int
) -> Tensor:
    """Computes rotary positional embeddings up to a specified sequence length.

    See 'Roformer: Enhanced Transformer with Rotary Embedding'
    (arxiv.org/pdf/2104.09864).

    Args:
        dim: The embedding dimension. By convention each component
            of the complex valued embedding is considered its own dim.
        base: Scaling factor for the frequency
        max_sequence_length: The number of positional embeddings to compute.

    Returns:
        RoPE positional embeddings of shape ``(max_sequence_length, dim / 2, 2)``.
    """
    return embed(theta(dim, base), max_sequence_length)


@module_dataclass
class RotaryEmbedding(Module[..., Tensor]):
    """Applies Rotary Positional Embeddings (RoPE) to input tensors.

    RoPE encodes positional information using complex-valued rotations applied
    to query and key vectors in attention. This encoding is relative, allowing
    models to better generalize to sequences longer than those seen during training.

    See "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    (https://arxiv.org/abs/2104.09864)

    .. code-block:: python

        from max.experimental import random
        from max.experimental.nn.rope import RotaryEmbedding
        from max.experimental.tensor import Tensor

        # RotaryEmbedding wraps a precomputed RoPE rotation table of shape
        # (max_sequence_length, head_dim // 2, 2).
        rope = RotaryEmbedding(weight=Tensor.zeros([2048, 64, 2]))

        # Apply to query or key tensors in attention.
        # Shape: (batch, seq_len, num_heads, head_dim)
        random.set_seed(0)
        query = random.normal([4, 128, 12, 128])
        query_with_rope = rope(query, start_pos=0)

        print(query_with_rope.shape)  # [4, 128, 12, 128]

    .. invisible-code-block: python

        assert query_with_rope.shape == [4, 128, 12, 128]
    """

    weight: Tensor
    #: Pre-computed embeddings of shape [max_sequence_length, n // 2, 2]

    @property
    def dim(self) -> int:
        """Returns the embedding dimension."""
        return int(self.weight.shape[1]) * 2

    @property
    def max_sequence_length(self) -> int:
        """Returns the maximum sequence length."""
        return int(self.weight.shape[0])

    def __rich_repr__(self):
        yield "dim", self.dim
        yield "max_sequence_length", self.max_sequence_length

    @F.functional
    def forward(self, x: Tensor, start_pos: DimLike = 0) -> Tensor:
        """Applies rotary positional embeddings (RoPE) to `x`.

        seq_len is inferred from the shape of `x`.

        Args:
            x: Activation tensor with shape (batch, seq_len, n_kv_heads, head_dim).
                x is interpreted as a complex number valued tensor where the
                `head_dim` dimension is alternating pairs of (real, imaginary)
                parts.
            start_pos: starting position of input tensor, defaults to 0 if None

        Returns:
            Input activation tensor with rotary positional embeddings applied and
            the same shape as `x`.
        """
        seq_len = x.shape[1]
        start_pos = Dim(start_pos)

        x_complex = F.as_interleaved_complex(x)
        freqs_cis = self.weight[start_pos : start_pos + seq_len, None, ...]
        return F.complex_mul(x_complex, freqs_cis).reshape(x.shape)


class TransposedRotaryEmbedding(RotaryEmbedding):
    """Applies RoPE using a transposed head-dimension layout."""

    @F.functional
    def forward(self, x: Tensor, start_pos: DimLike = 0) -> Tensor:
        """Applies rotary positional embeddings (RoPE) to `x`.

        The representation of `x` is transposed within the final dimension
        compared to traditional `RotaryEmbedding`.

        seq_len is inferred from the shape of `x`.

        Args:
            x: Activation tensor with shape (batch, seq_len, n_kv_heads, head_dim).
                x is interpreted as a complex number valued tensor where the
                first half of `head_dim` are the real parts and the last half
                are the imaginary parts.
            start_pos: starting position of input tensor, defaults to 0 if None

        Returns:
            Input activation tensor with rotary positional embeddings applied and
            the same shape as `x`.
        """
        seq_len = x.shape[1]
        *rest, head_dim = x.shape
        start_pos = Dim(start_pos)

        x_complex = x.reshape((*rest, 2, head_dim // 2)).T
        freqs_cis = self.weight[start_pos : start_pos + seq_len, None, ...]
        return F.complex_mul(x_complex, freqs_cis).T.reshape(x.shape)
