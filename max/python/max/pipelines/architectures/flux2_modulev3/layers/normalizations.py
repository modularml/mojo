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
"""FLUX.2 ModuleV3 normalization layers."""

from __future__ import annotations

from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.linear import Linear
from max.experimental.nn.norm import LayerNorm
from max.experimental.tensor import Tensor


class AdaLayerNormContinuous(Module[..., Tensor]):
    """Adaptive LayerNorm with continuous timestep conditioning.

    Mirrors the legacy
    :class:`max.pipelines.architectures.flux2.layers.normalizations.AdaLayerNormContinuous`.

    The conditioning embedding is passed through ``silu`` and a
    ``Linear(cond_dim -> 2 * embed_dim)`` projection.  The result is
    chunked into ``(scale, shift)`` which modulate the LayerNorm-ed
    input as ``x_norm * (1 + scale) + shift`` (broadcast over the
    sequence dimension).
    """

    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = True,
        *,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        self.linear = Linear(
            conditioning_embedding_dim, embedding_dim * 2, bias=bias
        )
        self.norm = LayerNorm(
            embedding_dim,
            eps=eps,
            elementwise_affine=elementwise_affine,
            use_bias=bias,
        )

    def forward(
        self,
        x: Tensor,
        conditioning_embedding: Tensor,
    ) -> Tensor:
        """Apply adaptive layer normalization.

        Args:
            x: Input tensor of shape ``[B, S, D]``.
            conditioning_embedding: Conditioning embedding (the FLUX.2
                fused time+guidance vector) of shape ``[B, D_cond]``.

        Returns:
            Tensor of shape ``[B, S, D]``.
        """
        conditioning_embedding = conditioning_embedding.cast(x.dtype)
        emb = self.linear(F.silu(conditioning_embedding))
        scale, shift = F.chunk(emb, chunks=2, axis=1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
