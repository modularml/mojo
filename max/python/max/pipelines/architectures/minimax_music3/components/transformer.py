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
"""The flow-matching diffusion transformer that denoises Flow-VAE audio latents.

36 blocks of bidirectional self-attention over the latent timeline, conditioned on
frame-aligned hidden states from the autoregressive stage. Flow-matching time runs
from 0 (noise) to 1 (data).

Four details of this architecture are easy to get subtly wrong, and each is called
out at the code below: conditioning enters by channel concatenation rather than
cross-attention, the timestep enters as a prepended token rather than as a
modulation, only the first 32 of each head's 64 dimensions rotate, and the
feed-forward gate is the *second* half of the ``ff_in`` output.

Runs in bfloat16 because float32 weights would be 9.8 GiB of a 22.5 GiB card. That
costs real accuracy -- the velocity output lands 2.1e-2 from the float32 reference,
growing about 10x across the 36 blocks -- so this component is gated against a
bfloat16 reference, with the float32 one alongside to show the dtype's own cost.
"""

from __future__ import annotations

import math

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import LayerNorm, Linear, Module, ModuleList
from max.experimental.nn.common_layers.functional_kernels import (
    flash_attention_gpu,
)
from max.experimental.tensor import Tensor
from max.graph import TensorType
from max.nn.attention import MHAMaskVariant

from ..model_config import TransformerConfig


class FourierEmbedding(Module[[Tensor], Tensor]):
    """Random Fourier features over the flow-matching time in ``[0, 1]``.

    The projection is a trained checkpoint weight, not a fixed schedule, so it
    cannot be reconstructed from the usual sinusoidal formula.
    """

    weight: Tensor

    def __init__(self, embedding_dim: int) -> None:
        self.weight = Tensor.zeros([embedding_dim // 2, 1])

    def forward(self, timestep: Tensor) -> Tensor:
        angles = (
            2.0
            * math.pi
            * F.unsqueeze(timestep, -1)
            @ F.transpose(self.weight, 0, 1).cast(timestep.dtype)
        )
        return F.concat([F.cos(angles), F.sin(angles)], axis=-1)


class TimestepEmbedding(Module[[Tensor], Tensor]):
    """The two-layer SiLU projection of ``diffusers``' ``TimestepEmbedding``."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        self.linear_1 = Linear(in_dim, out_dim, bias=True)
        self.linear_2 = Linear(out_dim, out_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_2(F.silu(self.linear_1(x)))


def partial_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotates only the leading ``rotary_dim`` of each head, leaving the rest.

    Rotate-half (GPT-NeoX) rather than pair-interleaved, so the fused RoPE kernel
    used elsewhere in MAX does not apply: it rotates ``x[i]`` against
    ``x[i + rotary_dim / 2]``, not against its neighbor.

    Args:
        x: ``(batch, seq, heads, head_dim)``.
        cos: ``(seq, rotary_dim)``.
        sin: ``(seq, rotary_dim)``.
    """
    rotary_dim = int(cos.shape[-1])
    cos = F.unsqueeze(cos, 1).cast(x.dtype)
    sin = F.unsqueeze(sin, 1).cast(x.dtype)
    rotated, passthrough = x[..., :rotary_dim], x[..., rotary_dim:]
    half = rotary_dim // 2
    swapped = F.concat([-rotated[..., half:], rotated[..., :half]], axis=-1)
    return F.concat([rotated * cos + swapped * sin, passthrough], axis=-1)


class Attention(Module[..., Tensor]):
    """Bidirectional self-attention with partial rotary embeddings."""

    def __init__(self, dim: int, heads: int, head_dim: int) -> None:
        self.heads = heads
        self.head_dim = head_dim
        inner_dim = heads * head_dim
        self.to_q = Linear(dim, inner_dim, bias=False)
        self.to_k = Linear(dim, inner_dim, bias=False)
        self.to_v = Linear(dim, inner_dim, bias=False)
        # A ModuleList in the reference, whose second entry is a no-op dropout;
        # the name has to survive so the checkpoint key matches.
        self.to_out = ModuleList([Linear(inner_dim, dim, bias=False)])

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        batch, seq = x.shape[0], x.shape[1]
        shape = [batch, seq, self.heads, self.head_dim]
        query = partial_rotary(F.reshape(self.to_q(x), shape), cos, sin)
        key = partial_rotary(F.reshape(self.to_k(x), shape), cos, sin)
        value = F.reshape(self.to_v(x), shape)

        # No mask: every position attends to every other, including the
        # prepended timestep token.
        attended = flash_attention_gpu(
            query,
            key,
            value,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        return self.to_out[0](
            F.reshape(attended, [batch, seq, self.heads * self.head_dim])
        )


class TransformerBlock(Module[..., Tensor]):
    """Pre-norm attention followed by a pre-norm gated feed-forward."""

    def __init__(
        self, dim: int, heads: int, head_dim: int, ff_inner_dim: int
    ) -> None:
        # `keep_dtype=False` normalizes in float32 and casts back, which is both
        # what torch does on bfloat16 input and what this port was gated
        # against; the experimental default would reduce in bfloat16.
        self.norm1 = LayerNorm(dim, keep_dtype=False)
        self.attn = Attention(dim, heads, head_dim)
        self.norm2 = LayerNorm(dim, keep_dtype=False)
        self.ff_in = Linear(dim, ff_inner_dim * 2, bias=True)
        self.ff_out = Linear(ff_inner_dim, dim, bias=True)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        gated, gate = F.chunk(self.ff_in(self.norm2(x)), 2, axis=-1)
        # SiLU applies to the *second* half. Swapping them is a plausible-looking
        # graph that produces plausible-sounding audio, so it is worth stating.
        return x + self.ff_out(gated * F.silu(gate))


class Transformer(Module[..., Tensor]):
    """Predicts the flow-matching velocity of a noisy latent sequence."""

    def __init__(self, config: TransformerConfig) -> None:
        self.config = config
        inner_dim = config.num_attention_heads * config.attention_head_dim
        # Latent, an equally-sized zero block, and the conditioning, stacked on
        # the channel axis -- there is no cross-attention in this model.
        concat_channels = 2 * config.in_channels + config.condition_dim

        self.time_proj = FourierEmbedding(config.fourier_embedding_dim)
        self.time_embed = TimestepEmbedding(
            config.fourier_embedding_dim, inner_dim
        )
        # 1-tap convolutions without bias, which channels-last makes plain
        # matmuls; both are used residually.
        self.preprocess_conv = Linear(
            concat_channels, concat_channels, bias=False
        )
        self.proj_in = Linear(concat_channels, inner_dim, bias=False)
        self.transformer_blocks = ModuleList(
            [
                TransformerBlock(
                    inner_dim,
                    config.num_attention_heads,
                    config.attention_head_dim,
                    config.ff_inner_dim,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.proj_out = Linear(inner_dim, config.in_channels, bias=False)
        self.postprocess_conv = Linear(
            config.in_channels, config.in_channels, bias=False
        )

    @property
    def dtype(self) -> DType:
        """The dtype the transformer computes in, taken from its parameters."""
        return self.proj_in.weight.dtype

    def input_types(self, latent_length: int) -> tuple[TensorType, ...]:
        """Graph inputs for one velocity evaluation over ``latent_length``.

        Single-batch, one branch per call. The serving path evaluates both
        classifier-free-guidance branches in one call instead -- see
        ``GuidedTransformer`` in
        :mod:`~max.pipelines.architectures.minimax_music3.diffusion`.
        """
        config = self.config
        return (
            TensorType(
                self.dtype,
                [1, config.in_channels, latent_length],
                device=self.device,
            ),
            TensorType(self.dtype, [1], device=self.device),
            TensorType(
                self.dtype,
                [1, latent_length, config.condition_dim],
                device=self.device,
            ),
        )

    def rotary(self, seq_len: int) -> tuple[Tensor, Tensor]:
        """Build the rotary tables for a sequence length.

        Both lengths are known when the graph is built, so the tables fold to
        constants rather than costing anything per execution.
        """
        rotary_dim = self.config.rotary_dim
        exponents = (
            F.arange(0, rotary_dim, 2, dtype=DType.float32, device=self.device)
            / rotary_dim
        )
        inv_freq = 1.0 / (self.config.rotary_theta**exponents)
        angles = F.outer(
            F.arange(0, seq_len, dtype=DType.float32, device=self.device),
            inv_freq,
        )
        # Duplicated rather than interleaved, to match rotate-half.
        angles = F.concat([angles, angles], axis=-1)
        # Kept float32 here and narrowed where they are applied, matching the
        # reference's order of operations.
        return F.cos(angles), F.sin(angles)

    def forward(
        self,
        latents: Tensor,
        timestep: Tensor,
        condition: Tensor,
    ) -> Tensor:
        """Predicts the velocity field at one flow-matching time.

        Args:
            latents: Noisy latents, ``(batch, in_channels, length)``.
            timestep: Flow-matching time in ``[0, 1]``, ``(batch,)``.
            condition: Frame-aligned conditioning, ``(batch, length, condition_dim)``.
                Zeros for the unconditional branch of classifier-free guidance.

        Returns:
            The predicted velocity, shaped like ``latents``.
        """
        latents = latents.cast(self.dtype)
        # Everything below is channels-last, so the reference's four transposes
        # collapse into this one.
        x = F.transpose(latents, 1, 2)
        zeros = F.constant(0, self.dtype, self.device).broadcast_to(x.shape)
        x = F.concat([x, zeros, condition.cast(self.dtype)], axis=-1)
        x = self.preprocess_conv(x) + x
        x = self.proj_in(x)

        temb = self.time_embed(self.time_proj(timestep.cast(self.dtype)))
        # The timestep rides along as an extra token rather than modulating the
        # blocks, and is dropped again before the output projection.
        x = F.concat([F.unsqueeze(temb, 1), x], axis=1)

        cos, sin = self.rotary(int(x.shape[1]))
        for block in self.transformer_blocks:
            x = block(x, cos, sin)

        velocity = self.proj_out(x[:, 1:, :])
        velocity = self.postprocess_conv(velocity) + velocity
        return F.transpose(velocity, 1, 2)
