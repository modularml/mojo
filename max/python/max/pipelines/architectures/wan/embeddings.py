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

from __future__ import annotations

import math
from collections.abc import Callable

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.activation import activation_function_from_name
from max.nn.kernels import rope_ragged_with_position_ids
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.quant_config import QuantConfig


def get_timestep_embedding(
    timesteps: TensorValue,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> TensorValue:
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * ops.range(
        0, half_dim, dtype=DType.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = ops.exp(exponent)
    timesteps_expanded = ops.cast(ops.unsqueeze(timesteps, 1), DType.float32)
    emb_expanded = ops.unsqueeze(emb, 0)
    emb = scale * timesteps_expanded * emb_expanded
    emb = ops.concat([ops.sin(emb), ops.cos(emb)], axis=-1)
    if flip_sin_to_cos:
        emb = ops.concat([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
    if embedding_dim % 2 == 1:
        emb = ops.pad(emb, (0, 0, 0, 1))
    return emb


def apply_rotary_emb_fused(
    q: TensorValue,
    k: TensorValue,
    rotary_emb: tuple[TensorValue, TensorValue],
) -> tuple[TensorValue, TensorValue]:
    """Pair-interleaved RoPE on Q/K via the fused ragged kernel.

    Builds the interleaved freqs_cis layout once from Wan's repeat-
    interleaved cos/sin and dispatches both Q and K rotations through
    ``rope_ragged_with_position_ids``.
    """
    cos, sin = rotary_emb
    seq_len = cos.shape[0]
    head_dim = cos.shape[1]

    # Repeat-interleaved [c0,c0,c1,c1,...] -> interleaved-pairs [c0,s0,c1,s1,...].
    cos_pairs = ops.reshape(cos, [seq_len, head_dim // 2, 2])[..., 0]
    sin_pairs = ops.reshape(sin, [seq_len, head_dim // 2, 2])[..., 0]
    freqs_cis = ops.reshape(
        ops.stack([cos_pairs, sin_pairs], axis=-1),
        [seq_len, head_dim],
    )

    batch_size = q.shape[0]
    q_seq_len = q.shape[1]
    num_heads = q.shape[2]
    head_dim_qk = q.shape[3]

    position_ids = ops.broadcast_to(
        ops.unsqueeze(
            ops.range(0, q_seq_len, 1, dtype=DType.uint32, device=q.device),
            0,
        ),
        [batch_size, q_seq_len],
    )
    position_ids = ops.reshape(position_ids, [batch_size * q_seq_len])

    q_ragged = ops.reshape(q, [batch_size * q_seq_len, num_heads, head_dim_qk])
    k_ragged = ops.reshape(k, [batch_size * q_seq_len, num_heads, head_dim_qk])

    q_rotated = rope_ragged_with_position_ids(
        q_ragged, freqs_cis, position_ids, interleaved=True
    )
    k_rotated = rope_ragged_with_position_ids(
        k_ragged, freqs_cis, position_ids, interleaved=True
    )

    return (
        ops.reshape(q_rotated, [batch_size, q_seq_len, num_heads, head_dim_qk]),
        ops.reshape(k_rotated, [batch_size, q_seq_len, num_heads, head_dim_qk]),
    )


class Timesteps(Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: float = 1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def __call__(self, timesteps: TensorValue) -> TensorValue:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


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
        device: DeviceRef = DeviceRef.CPU(),  # noqa: B008
        quant_config: QuantConfig | None = None,
    ):
        super().__init__()
        # Under FP8 the weight is float8_e4m3fn with per-tensor scales; the
        # bias stays bfloat16 via ``quant_config.bias_dtype``.
        weight_dtype = (
            DType.float8_e4m3fn if quant_config is not None else dtype
        )
        self.linear_1 = Linear(
            in_dim=in_channels,
            out_dim=time_embed_dim,
            dtype=weight_dtype,
            device=device,
            has_bias=sample_proj_bias,
            quant_config=quant_config,
        )
        self.cond_proj: Linear | None
        if cond_proj_dim is not None:
            self.cond_proj = Linear(
                in_dim=cond_proj_dim,
                out_dim=in_channels,
                dtype=weight_dtype,
                device=device,
                has_bias=False,
                quant_config=quant_config,
            )
        else:
            self.cond_proj = None
        self.act = activation_function_from_name(act_fn)
        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = Linear(
            in_dim=time_embed_dim,
            out_dim=time_embed_dim_out,
            dtype=weight_dtype,
            device=device,
            has_bias=sample_proj_bias,
            quant_config=quant_config,
        )
        self.post_act: Callable[[TensorValue], TensorValue] | None
        if post_act_fn is not None:
            self.post_act = activation_function_from_name(post_act_fn)
        else:
            self.post_act = None

    def __call__(self, sample: TensorValue) -> TensorValue:
        if self.cond_proj is not None:
            sample = sample + self.cond_proj(sample)
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample
