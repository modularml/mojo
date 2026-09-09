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

"""Embeddings for QwenImage transformer: timestep projection and 3D RoPE."""

import math

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.layer import Module
from max.nn.linear import Linear


def get_timestep_embedding(
    timesteps: TensorValue,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> TensorValue:
    """Create sinusoidal timestep embeddings."""
    half_dim = embedding_dim // 2

    exponent = -math.log(max_period) * ops.range(
        0, half_dim, dtype=DType.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = ops.exp(exponent)
    timesteps_f32 = ops.cast(timesteps, DType.float32)
    emb = ops.outer(timesteps_f32, emb) * scale
    emb = ops.concat([ops.sin(emb), ops.cos(emb)], axis=-1)

    if flip_sin_to_cos:
        emb = ops.concat([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)

    if embedding_dim % 2 == 1:
        emb = ops.pad(emb, [0, 0, 0, 1])

    return emb


def apply_rotary_emb(
    x: TensorValue,
    freqs_cis: tuple[TensorValue, TensorValue],
    sequence_dim: int = 1,
) -> TensorValue:
    """Apply rotary embeddings to input tensor (complex-multiply path).

    Matches diffusers' ``use_real=False`` path:
        view x as complex pairs, multiply by ``cos + i·sin``, flatten back.

    Because MAX graph has no complex dtype we expand the multiplication
    manually:  ``(x_re + i·x_im)(cos + i·sin)``
             = ``(x_re·cos - x_im·sin) + i·(x_re·sin + x_im·cos)``

    Args:
        x: Input tensor [B, S, H, D] (sequence_dim=1).
        freqs_cis: ``(cos, sin)`` each of shape ``[S, D//2]``.
        sequence_dim: Dimension index for sequence length (1 or 2).
    """
    cos, sin = freqs_cis  # [S, D//2]

    # Broadcast freqs to match x layout
    if sequence_dim == 2:
        # x: [B, H, S, D]  →  cos/sin: [1, 1, S, D//2]
        cos = ops.unsqueeze(ops.unsqueeze(cos, 0), 0)
        sin = ops.unsqueeze(ops.unsqueeze(sin, 0), 0)
    elif sequence_dim == 1:
        # x: [B, S, H, D]  →  cos/sin: [1, S, 1, D//2]
        cos = ops.unsqueeze(ops.unsqueeze(cos, 0), 2)
        sin = ops.unsqueeze(ops.unsqueeze(sin, 0), 2)
    else:
        raise ValueError(f"`sequence_dim={sequence_dim}` but should be 1 or 2.")

    input_dtype = x.dtype
    x_shape = list(x.shape)

    # Split last dim into (D//2, 2) pairs — real and imaginary parts
    x_pairs = ops.reshape(x, x_shape[:-1] + [x_shape[-1] // 2, 2])
    x_re = x_pairs[..., 0]  # [B, S, H, D//2]
    x_im = x_pairs[..., 1]  # [B, S, H, D//2]

    # Complex multiply in float32
    x_re = ops.cast(x_re, DType.float32)
    x_im = ops.cast(x_im, DType.float32)
    cos = ops.cast(cos, DType.float32)
    sin = ops.cast(sin, DType.float32)

    out_re = x_re * cos - x_im * sin
    out_im = x_re * sin + x_im * cos

    # Interleave back: [B, S, H, D//2, 2] → [B, S, H, D]
    out = ops.stack([out_re, out_im], axis=-1)
    out = ops.reshape(out, x_shape)
    return ops.cast(out, input_dtype)


def get_1d_rotary_pos_embed(
    dim: int,
    pos: TensorValue,
    theta: float = 10000.0,
) -> tuple[TensorValue, TensorValue]:
    """Precompute rotary position embeddings for one axis.

    Returns ``(cos, sin)`` each of shape ``[S, dim // 2]``, matching the
    complex-multiply convention used by diffusers.
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")
    freq_exponent = (
        ops.range(
            0,
            dim,
            2,
            dtype=DType.float32,
            device=pos.device,
        )
        / dim
    )
    freq = 1.0 / (theta**freq_exponent)
    freqs = ops.outer(pos, freq)  # [S, dim // 2]
    return ops.cos(freqs), ops.sin(freqs)


class Timesteps(Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: float = 1.0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def __call__(self, timesteps: TensorValue) -> TensorValue:
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


class TimestepEmbedding(Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        sample_proj_bias: bool = True,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.GPU(),  # noqa: B008
    ) -> None:
        super().__init__()
        self.linear_1 = Linear(
            in_dim=in_channels,
            out_dim=time_embed_dim,
            dtype=dtype,
            device=device,
            has_bias=sample_proj_bias,
        )
        self.linear_2 = Linear(
            in_dim=time_embed_dim,
            out_dim=time_embed_dim,
            dtype=dtype,
            device=device,
            has_bias=sample_proj_bias,
        )

    def __call__(self, sample: TensorValue) -> TensorValue:
        sample = self.linear_1(sample)
        sample = ops.silu(sample)
        sample = self.linear_2(sample)
        return sample


class QwenImageTimestepProjEmbeddings(Module):
    """Timestep-only projection embeddings (no guidance embedding).

    Unlike Flux2 which combines timestep + guidance, QwenImage only uses timestep
    since guidance_embeds=False.
    """

    def __init__(
        self,
        in_channels: int = 256,
        embedding_dim: int = 3072,
        bias: bool = False,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.GPU(),  # noqa: B008
    ):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=in_channels,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=embedding_dim,
            sample_proj_bias=bias,
            dtype=dtype,
            device=device,
        )

    def __call__(self, timestep: TensorValue) -> TensorValue:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            ops.cast(timesteps_proj, timestep.dtype)
        )
        return timesteps_emb


class QwenImagePosEmbed(Module):
    """3D Rotary Position Embeddings for QwenImage.

    Uses axes_dims_rope = (16, 56, 56) for (T, H, W) dimensions,
    compared to Flux2's 4D (32, 32, 32, 32).
    """

    theta: int
    axes_dim: tuple[int, ...]

    def __init__(self, theta: int, axes_dim: tuple[int, ...]):
        super().__init__()
        self.theta = theta
        self.axes_dim = tuple(axes_dim)

    def __call__(self, ids: TensorValue) -> tuple[TensorValue, TensorValue]:
        """Compute rotary position embeddings from position IDs.

        Args:
            ids: Position IDs of shape [S, len(axes_dim)] (3D: T, H, W).

        Returns:
            Tuple of (cos, sin) tensors of shape [S, sum(axes_dim)//2].
        """
        cos_out = []
        sin_out = []

        pos = ops.cast(ids, DType.float32)

        for i in range(len(self.axes_dim)):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[..., i],
                theta=self.theta,
            )
            cos_out.append(cos)
            sin_out.append(sin)

        freqs_cos = ops.concat(cos_out, axis=-1)
        freqs_sin = ops.concat(sin_out, axis=-1)

        return freqs_cos, freqs_sin
