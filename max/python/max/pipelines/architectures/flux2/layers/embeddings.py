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
from max.graph import DeviceRef, TensorValue, ops
from max.nn.activation import activation_function_from_name
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
    """This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings."""
    half_dim = embedding_dim // 2

    # Create frequency bands
    exponent = -math.log(max_period) * ops.range(
        0, half_dim, dtype=DType.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = ops.exp(exponent)

    # Expand timesteps: [B] -> [B, 1]
    # Expand emb: [half_dim] -> [1, half_dim]
    timesteps_expanded = ops.cast(ops.unsqueeze(timesteps, -1), DType.float32)
    # emb_expanded = F.reshape(emb, [1] * len(timesteps.shape) + [half_dim])
    emb_expanded = ops.unsqueeze(emb, 0)

    # Multiply: [B, 1] * [1, half_dim] -> [B, half_dim]
    emb = scale * timesteps_expanded * emb_expanded
    emb = ops.concat([ops.sin(emb), ops.cos(emb)], axis=-1)

    # Concatenate sin and cos
    if flip_sin_to_cos:
        emb = ops.concat([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)

    if embedding_dim % 2 == 1:
        emb = ops.pad(emb, [0, 0, 0, 1])

    return emb


def apply_rotary_emb(
    x: TensorValue,
    freqs_cis: tuple[TensorValue, TensorValue],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
    sequence_dim: int = 2,
) -> TensorValue:
    """Apply rotary embeddings to input tensor.

    Args:
        x: Input tensor of shape [B, H, S, D] (if sequence_dim=2) or
            [B, S, H, D] (if sequence_dim=1).
        freqs_cis: Tuple of (cos, sin) frequency tensors of shape [S, D].
        use_real: If True, use real-valued RoPE. If False, raises NotImplementedError.
        use_real_unbind_dim: Dimension to unbind for real-valued RoPE.
            -1: Used for Flux, CogVideoX, Hunyuan-Dit (reshape to [..., D//2, 2])
            -2: Used for Stable Audio, OmniGen, CogView4, Cosmos (reshape to [..., 2, D//2])
        sequence_dim: Dimension index for sequence length (1 or 2).

    Returns:
        Tensor with rotary embeddings applied.

    Raises:
        NotImplementedError: If use_real is False.
    """
    if not use_real:
        raise NotImplementedError("Only use_real=True is supported")

    cos, sin = freqs_cis  # [S, D]
    # Expand cos/sin to match x shape based on sequence_dim
    if sequence_dim == 2:
        cos = ops.unsqueeze(ops.unsqueeze(cos, 0), 0)
        sin = ops.unsqueeze(ops.unsqueeze(sin, 0), 0)
    elif sequence_dim == 1:
        cos = ops.unsqueeze(ops.unsqueeze(cos, 0), 2)
        sin = ops.unsqueeze(ops.unsqueeze(sin, 0), 2)
    else:
        raise ValueError(f"`sequence_dim={sequence_dim}` but should be 1 or 2.")

    input_dtype = x.dtype
    if use_real_unbind_dim == -1:
        # Used for Flux, CogVideoX, Hunyuan-Dit
        # Reshape x: [..., D] -> [..., D//2, 2]
        x_shape = list(x.shape)
        x_reshaped = ops.reshape(
            x,
            x_shape[:-1] + [x_shape[-1] // 2, 2],
        )
        # Split into real and imaginary parts: [..., D//2, 2] -> 2 x [..., D//2]
        x_real = x_reshaped[..., 0]
        x_imag = x_reshaped[..., 1]

        # Create rotated version: stack([-x_imag, x_real], dim=-1) then flatten
        # This creates [..., D//2, 2]
        x_rotated_real = -x_imag
        x_rotated_imag = x_real
        x_rotated_stacked = ops.stack([x_rotated_real, x_rotated_imag], axis=-1)

        # Flatten back to [..., D]
        x_rotated = ops.reshape(x_rotated_stacked, x_shape)
    elif use_real_unbind_dim == -2:
        # Used for Stable Audio, OmniGen, CogView4, Cosmos
        # Reshape x: [..., D] -> [..., 2, D//2]
        x_shape = list(x.shape)
        x_reshaped = ops.reshape(
            x,
            x_shape[:-1] + [2, x_shape[-1] // 2],
        )
        # Split into real and imaginary parts: [..., 2, D//2] -> 2 x [..., D//2]
        x_real = x_reshaped[..., 0, :]
        x_imag = x_reshaped[..., 1, :]

        # Create rotated version: cat([-x_imag, x_real], dim=-1)
        x_rotated = ops.concat([-x_imag, x_real], axis=-1)
    else:
        raise ValueError(
            f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2."
        )

    # Apply rotation: x * cos + x_rotated * sin
    # Cast to float32 for computation, then back to input dtype
    x_float = ops.cast(x, DType.float32)
    x_rotated_float = ops.cast(x_rotated, DType.float32)
    cos_float = ops.cast(cos, DType.float32)
    sin_float = ops.cast(sin, DType.float32)

    out = x_float * cos_float + x_rotated_float * sin_float
    out = ops.cast(out, input_dtype)

    return out


def get_1d_rotary_pos_embed(
    dim: int,
    pos: TensorValue,
    theta: float = 10000.0,
    use_real: bool = True,
    repeat_interleave_real: bool = True,
) -> tuple[TensorValue, TensorValue]:
    """Precompute real-valued rotary position embeddings for one axis.

    Args:
        dim: Dimension of the embedding (must be even).
        pos: Position indices tensor of shape [S].
        theta: Base frequency for the sinusoidal encoding.
        use_real: If True, use real-valued RoPE.
        repeat_interleave_real: If True, repeat cos/sin along the embedding
            dimension. If False, raises NotImplementedError.

    Returns:
        Tuple of (cos, sin) tensors for rotary position embedding.
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"

    if not use_real:
        raise NotImplementedError("Only use_real=True is supported")

    if not repeat_interleave_real:
        raise NotImplementedError(
            "Only repeat_interleave_real=True is supported"
        )

    # Create frequency bands: [dim/2]
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

    # Compute outer product: [S, dim/2]
    freqs = ops.outer(pos, freq)

    # Compute cos and sin: [S, dim/2]
    cos_emb = ops.cos(freqs)
    sin_emb = ops.sin(freqs)

    # Repeat interleave: [S, dim/2] -> [S, dim]
    # repeat_interleave not supported on GPU, use stack + reshape instead
    # PyTorch: repeat_interleave(2, dim=1) makes [a, b, c] -> [a, a, b, b, c, c]
    # MAX equivalent: stack([x, x], axis=2) + reshape
    cos_stacked = ops.stack([cos_emb, cos_emb], axis=2)
    sin_stacked = ops.stack([sin_emb, sin_emb], axis=2)
    freqs_cos = ops.reshape(
        cos_stacked,
        [cos_emb.shape[0], cos_stacked.shape[1] * cos_stacked.shape[2]],
    )
    freqs_sin = ops.reshape(
        sin_stacked,
        [sin_emb.shape[0], sin_stacked.shape[1] * sin_stacked.shape[2]],
    )

    return freqs_cos, freqs_sin


class Timesteps(Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: float = 1.0,
    ):
        """Initialize Timesteps module.

        Args:
            num_channels: Number of embedding channels.
            flip_sin_to_cos: Whether to flip sin and cos in the embedding.
            downscale_freq_shift: Shift for frequency downscaling.
            scale: Scale factor for the embeddings.
        """
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def __call__(self, timesteps: TensorValue) -> TensorValue:
        """Convert timesteps to embeddings.

        Args:
            timesteps: Timestep tensor of shape [B].

        Returns:
            Sinusoidal embeddings of shape [B, num_channels].
        """
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
        act_fn: str = "silu",
        out_dim: int | None = None,
        post_act_fn: str | None = None,
        cond_proj_dim: int | None = None,
        sample_proj_bias: bool = True,
        *,
        dtype: DType = DType.bfloat16,
        device: DeviceRef = DeviceRef.GPU(),  # noqa: B008
    ) -> None:
        """Initialize TimestepEmbedding MLP.

        Args:
            in_channels: Number of input channels.
            time_embed_dim: Hidden dimension for timestep embeddings.
            act_fn: Activation function name (default: "silu").
            out_dim: Output dimension (defaults to time_embed_dim).
            post_act_fn: Optional post-activation function name.
            cond_proj_dim: Optional conditioning projection dimension.
            sample_proj_bias: Whether to use bias in linear layers.

            dtype: Weight dtype.
            device: Weight device.
        """
        super().__init__()
        self.linear_1 = Linear(
            in_dim=in_channels,
            out_dim=time_embed_dim,
            dtype=dtype,
            device=device,
            has_bias=sample_proj_bias,
        )
        if cond_proj_dim is not None:
            self.cond_proj: Linear | None = Linear(
                in_dim=cond_proj_dim,
                out_dim=in_channels,
                dtype=dtype,
                device=device,
                has_bias=False,
            )
        else:
            self.cond_proj = None

        self.act = activation_function_from_name(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = Linear(
            in_dim=time_embed_dim,
            out_dim=time_embed_dim_out,
            dtype=dtype,
            device=device,
            has_bias=sample_proj_bias,
        )
        self.post_act = (
            None
            if post_act_fn is None
            else activation_function_from_name(post_act_fn)
        )

    def __call__(self, sample: TensorValue) -> TensorValue:
        """Process timestep embeddings through MLP.

        Args:
            sample: Input tensor of shape [B, in_channels].

        Returns:
            Output tensor of shape [B, time_embed_dim].
        """
        if self.cond_proj is not None:
            sample = sample + self.cond_proj(sample)
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample
