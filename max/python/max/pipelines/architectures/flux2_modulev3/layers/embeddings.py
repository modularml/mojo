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
"""FLUX.2 ModuleV3 timestep, guidance, and rotary position embeddings."""

from __future__ import annotations

import math

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.linear import Linear
from max.experimental.tensor import Tensor


def get_timestep_embedding(
    timesteps: Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> Tensor:
    """Return sinusoidal timestep embeddings (DDPM-style).

    Args:
        timesteps: 1-D tensor of shape ``[B]`` of timestep values.
        embedding_dim: Channel count of the produced embedding.
        flip_sin_to_cos: If True, swap the sin/cos halves of the output
            so the cos half comes first.
        downscale_freq_shift: Shift applied when downscaling the
            frequency grid.
        scale: Multiplier on the inner product before sin/cos.
        max_period: Base used to spread the frequencies; matches the
            ``max_period`` argument to DDPM-style embeddings.

    Returns:
        Tensor of shape ``[B, embedding_dim]`` in float32.
    """
    half_dim = embedding_dim // 2

    exponent = -math.log(max_period) * F.range(
        0, half_dim, dtype=DType.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = F.exp(exponent)

    timesteps_expanded = timesteps.unsqueeze(-1).cast(DType.float32)
    emb_expanded = emb.unsqueeze(0)

    emb = scale * timesteps_expanded * emb_expanded
    emb = F.concat([F.sin(emb), F.cos(emb)], axis=-1)

    if flip_sin_to_cos:
        emb = F.concat([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)

    if embedding_dim % 2 == 1:
        emb = F.pad(emb, [0, 0, 0, 1])

    return emb


def apply_rotary_emb(
    x: Tensor,
    freqs_cis: tuple[Tensor, Tensor],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
    sequence_dim: int = 2,
) -> Tensor:
    """Apply real-valued rotary embeddings to ``x``.

    Mirrors the legacy
    :func:`max.pipelines.architectures.flux2.layers.embeddings.apply_rotary_emb`
    expressed against :class:`~max.experimental.tensor.Tensor`.

    Args:
        x: Input tensor of shape ``[B, H, S, D]`` (``sequence_dim=2``) or
            ``[B, S, H, D]`` (``sequence_dim=1``).
        freqs_cis: Tuple ``(cos, sin)`` of frequency tensors, shape
            ``[S, D]``.
        use_real: Must be True; only real-valued RoPE is supported.
        use_real_unbind_dim: Reshape strategy for splitting the head
            dim into pairs. ``-1`` (FLUX/CogVideoX) reshapes to
            ``[..., D/2, 2]``; ``-2`` (Stable Audio etc.) reshapes to
            ``[..., 2, D/2]``.
        sequence_dim: Index of the sequence dimension in ``x``.

    Returns:
        Tensor of the same shape and dtype as ``x``.

    Raises:
        NotImplementedError: If ``use_real=False``.
        ValueError: If ``sequence_dim`` or ``use_real_unbind_dim`` is
            unsupported.
    """
    if not use_real:
        raise NotImplementedError("Only use_real=True is supported")

    cos, sin = freqs_cis
    if sequence_dim == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif sequence_dim == 1:
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
    else:
        raise ValueError(f"`sequence_dim={sequence_dim}` but should be 1 or 2.")

    input_dtype = x.dtype
    if use_real_unbind_dim == -1:
        x_shape = list(x.shape)
        x_reshaped = x.reshape(x_shape[:-1] + [x_shape[-1] // 2, 2])
        x_real = x_reshaped[..., 0]
        x_imag = x_reshaped[..., 1]
        x_rotated_real = -x_imag
        x_rotated_imag = x_real
        x_rotated_stacked = F.stack([x_rotated_real, x_rotated_imag], axis=-1)
        x_rotated = x_rotated_stacked.reshape(x_shape)
    elif use_real_unbind_dim == -2:
        x_shape = list(x.shape)
        x_reshaped = x.reshape(x_shape[:-1] + [2, x_shape[-1] // 2])
        x_real = x_reshaped[..., 0, :]
        x_imag = x_reshaped[..., 1, :]
        x_rotated = F.concat([-x_imag, x_real], axis=-1)
    else:
        raise ValueError(
            f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2."
        )

    x_float = x.cast(DType.float32)
    x_rotated_float = x_rotated.cast(DType.float32)
    cos_float = cos.cast(DType.float32)
    sin_float = sin.cast(DType.float32)

    out = x_float * cos_float + x_rotated_float * sin_float
    out = out.cast(input_dtype)
    return out


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Tensor,
    theta: float = 10000.0,
    use_real: bool = True,
    repeat_interleave_real: bool = True,
) -> tuple[Tensor, Tensor]:
    """Precompute real-valued rotary cos/sin for a single axis.

    Args:
        dim: Embedding dimension for this axis. Must be even.
        pos: Position indices of shape ``[S]``.
        theta: Base frequency.
        use_real: Must be True; only real-valued RoPE is supported.
        repeat_interleave_real: Must be True; only the
            stack+reshape repeat path is supported.

    Returns:
        Tuple ``(cos, sin)`` each shape ``[S, dim]``.

    Raises:
        AssertionError: If ``dim`` is odd.
        NotImplementedError: If ``use_real=False`` or
            ``repeat_interleave_real=False``.
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"

    if not use_real:
        raise NotImplementedError("Only use_real=True is supported")
    if not repeat_interleave_real:
        raise NotImplementedError(
            "Only repeat_interleave_real=True is supported"
        )

    freq_exponent = (
        F.range(0, dim, 2, dtype=DType.float32, device=pos.device) / dim
    )
    freq = 1.0 / (theta**freq_exponent)

    freqs = F.outer(pos, freq)
    cos_emb = F.cos(freqs)
    sin_emb = F.sin(freqs)

    # repeat_interleave(2, dim=1) via stack+reshape:
    # PyTorch: [a, b, c] -> [a, a, b, b, c, c].
    cos_stacked = F.stack([cos_emb, cos_emb], axis=2)
    sin_stacked = F.stack([sin_emb, sin_emb], axis=2)
    freqs_cos = cos_stacked.reshape(
        [cos_emb.shape[0], cos_stacked.shape[1] * cos_stacked.shape[2]]
    )
    freqs_sin = sin_stacked.reshape(
        [sin_emb.shape[0], sin_stacked.shape[1] * sin_stacked.shape[2]]
    )

    return freqs_cos, freqs_sin


class Timesteps(Module[[Tensor], Tensor]):
    """Sinusoidal timestep embedding.

    Stateless: holds only the embedding hyperparameters; no parameters
    to load.  ``forward`` delegates to :func:`get_timestep_embedding`.
    """

    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: float = 1.0,
    ) -> None:
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: Tensor) -> Tensor:
        """Return sinusoidal embeddings of ``timesteps``."""
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class TimestepEmbedding(Module[[Tensor], Tensor]):
    """Two-Linear MLP applied after the sinusoidal embedding.

    Activation is hardcoded to ``silu`` to match FLUX.2; the
    ``act_fn`` / ``post_act_fn`` / ``cond_proj_dim`` extension points
    from the legacy implementation are intentionally rejected here
    because FLUX.2 never uses them.
    """

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int | None = None,
        post_act_fn: str | None = None,
        cond_proj_dim: int | None = None,
        sample_proj_bias: bool = True,
    ) -> None:
        if act_fn != "silu":
            raise NotImplementedError(
                f"Only act_fn='silu' is supported; got {act_fn!r}."
            )
        if post_act_fn is not None:
            raise NotImplementedError(
                f"post_act_fn must be None; got {post_act_fn!r}."
            )
        if cond_proj_dim is not None:
            raise NotImplementedError(
                f"cond_proj_dim must be None; got {cond_proj_dim!r}."
            )

        self.linear_1 = Linear(
            in_channels, time_embed_dim, bias=sample_proj_bias
        )
        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = Linear(
            time_embed_dim, time_embed_dim_out, bias=sample_proj_bias
        )

    def forward(self, sample: Tensor) -> Tensor:
        """Apply ``linear_1 -> silu -> linear_2``."""
        sample = self.linear_1(sample)
        sample = F.silu(sample)
        sample = self.linear_2(sample)
        return sample


class Flux2TimestepGuidanceEmbeddings(Module[..., Tensor]):
    """Combined timestep + guidance embeddings used by FLUX.2.

    Mirrors the legacy
    :class:`max.pipelines.architectures.flux2.flux2.Flux2TimestepGuidanceEmbeddings`.
    Shares a single ``Timesteps`` projector across both inputs and
    sums the two MLP outputs into a single conditioning vector that
    drives every modulation in the transformer body.

    When ``guidance_embeds=False`` (Klein-style checkpoints), no
    guidance MLP is constructed; ``forward`` ignores the ``guidance``
    argument in that case.
    """

    def __init__(
        self,
        *,
        in_channels: int = 256,
        embedding_dim: int = 6144,
        bias: bool = False,
        guidance_embeds: bool = True,
    ) -> None:
        self.time_proj = Timesteps(
            num_channels=in_channels,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=embedding_dim,
            sample_proj_bias=bias,
        )
        self.guidance_embedder: TimestepEmbedding | None
        if guidance_embeds:
            self.guidance_embedder = TimestepEmbedding(
                in_channels=in_channels,
                time_embed_dim=embedding_dim,
                sample_proj_bias=bias,
            )
        else:
            self.guidance_embedder = None

    def forward(self, timestep: Tensor, guidance: Tensor) -> Tensor:
        """Return ``timestep_emb + guidance_emb`` (or just timestep)."""
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            timesteps_proj.cast(timestep.dtype)
        )
        if self.guidance_embedder is not None:
            guidance_proj = self.time_proj(guidance)
            guidance_emb = self.guidance_embedder(
                guidance_proj.cast(guidance.dtype)
            )
            return timesteps_emb + guidance_emb
        return timesteps_emb


class Flux2PosEmbed(Module[[Tensor], tuple[Tensor, Tensor]]):
    """Multi-axis FLUX.2 rotary position embedding generator.

    Given a per-token ID tensor of shape ``[S, len(axes_dim)]``,
    produces concatenated ``(cos, sin)`` of shape ``[S, sum(axes_dim)]``
    suitable for the FLUX.2 attention kernels.

    Stateless: holds only ``theta`` and ``axes_dim``; no parameters.
    """

    def __init__(self, theta: int, axes_dim: tuple[int, ...]) -> None:
        self.theta = theta
        self.axes_dim = tuple(axes_dim)

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        """Compute the multi-axis RoPE for the given position IDs.

        Args:
            ids: Position IDs of shape ``[S, len(axes_dim)]``. May be
                int64 (the FLUX.2 default); cast to float32 internally.

        Returns:
            Tuple ``(cos, sin)`` each shape ``[S, sum(axes_dim)]``.
        """
        pos = ids.cast(DType.float32) if ids.dtype != DType.float32 else ids

        cos_out: list[Tensor] = []
        sin_out: list[Tensor] = []
        for i, axis_dim in enumerate(self.axes_dim):
            cos, sin = get_1d_rotary_pos_embed(
                axis_dim,
                pos[..., i],
                theta=self.theta,
                use_real=True,
                repeat_interleave_real=True,
            )
            cos_out.append(cos)
            sin_out.append(sin)

        freqs_cos = F.concat(cos_out, axis=-1)
        freqs_sin = F.concat(sin_out, axis=-1)
        return freqs_cos, freqs_sin
