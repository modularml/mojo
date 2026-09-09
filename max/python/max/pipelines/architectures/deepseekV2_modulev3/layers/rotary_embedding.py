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
"""Deepseek YaRN rotary embedding for the ModuleV3 API."""

from __future__ import annotations

import math

from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers.rotary_embedding import RotaryEmbedding
from max.experimental.tensor import Tensor
from max.graph import Dim
from max.nn.rotary_embedding import DeepseekYarnRopeScalingParams


def _yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    """Computes the mscale factor for YaRN interpolation."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekYarnRotaryEmbedding(RotaryEmbedding):
    """Deepseek YaRN Rotary Position Embedding for the ModuleV3 API.

    Unlike the generic :class:`YarnRotaryEmbedding`, this variant applies an
    ``mscale / mscale_all_dim`` ratio to the frequency tensor (as used by
    DeepSeek-V2).

    Args:
        dim: Rope dimension (``qk_rope_head_dim`` for DeepSeek-V2).
        n_heads: Number of attention heads.
        theta: Base for computing RoPE frequencies.
        max_seq_len: Maximum sequence length.
        device: Device the embedding tensors live on.
        head_dim: Per-head dimension; defaults to ``dim // n_heads``.
        interleaved: Whether to interpret pairs as interleaved complex.
        scaling_params: DeepSeek YaRN scaling parameters.
    """

    scaling_params: DeepseekYarnRopeScalingParams | None = None

    def __init__(
        self,
        dim: int,
        n_heads: int,
        theta: float,
        max_seq_len: int,
        device: Device,
        head_dim: int | None = None,
        interleaved: bool = True,
        scaling_params: DeepseekYarnRopeScalingParams | None = None,
    ) -> None:
        if scaling_params is not None:
            self.scaling_params = scaling_params

        super().__init__(
            dim,
            n_heads,
            theta,
            max_seq_len,
            device,
            head_dim,
            interleaved,
        )

    def freqs_cis_base(self) -> Tensor:
        if self._freqs_cis is None:
            if self.scaling_params is None:
                raise ValueError("scaling_params must be provided")

            _mscale = float(
                _yarn_get_mscale(
                    self.scaling_params.scaling_factor,
                    self.scaling_params.mscale,
                )
                / _yarn_get_mscale(
                    self.scaling_params.scaling_factor,
                    self.scaling_params.mscale_all_dim,
                )
            )

            inv_freqs = self._compute_yarn_freqs()

            t = F.arange(
                0,
                self.max_seq_len,
                1,
                out_dim=self.max_seq_len,
                device=CPU(),
                dtype=DType.float32,
            )
            freqs = F.outer(t, inv_freqs)
            cos = F.cos(freqs) * _mscale
            sin = F.sin(freqs) * _mscale
            self._freqs_cis = F.stack([cos, sin], axis=-1)

        assert isinstance(self._freqs_cis, Tensor)
        return self._freqs_cis

    def compute_scale(self, user_scale: float | None = None) -> float:
        """Returns the attention scale factor with YaRN mscale adjustment."""
        assert self.scaling_params
        scale = super().compute_scale(user_scale)
        mscale = _yarn_get_mscale(
            self.scaling_params.scaling_factor, self.scaling_params.mscale
        )
        return scale * mscale * mscale

    def _compute_yarn_freqs(self) -> Tensor:
        if self.scaling_params is None:
            raise ValueError("scaling_params must be provided")

        dim_2 = Dim(self.dim // 2)
        range_output = F.arange(
            start=0,
            stop=self.dim,
            step=2,
            out_dim=dim_2,
            device=CPU(),
            dtype=DType.float32,
        )

        freq_base = self.theta ** (range_output / float(self.dim))
        freq_extra = F.cast(1.0 / freq_base, DType.float32)
        freq_inter = F.cast(
            1.0 / (self.scaling_params.scaling_factor * freq_base),
            DType.float32,
        )

        low, high = self._yarn_find_correction_range(
            self.scaling_params.beta_fast,
            self.scaling_params.beta_slow,
            self.dim,
            int(self.theta),
            self.scaling_params.original_max_position_embeddings,
        )

        inv_freq_mask = 1.0 - self._yarn_linear_ramp_mask(
            low, high, dim_2
        ).cast(DType.float32)

        return freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

    def _yarn_find_correction_range(
        self,
        beta_fast: float,
        beta_slow: float,
        dim: int,
        base: float,
        original_max_position: int,
    ) -> tuple[Tensor, Tensor]:
        low_rot = F.constant(beta_fast, dtype=DType.float32, device=CPU())
        high_rot = F.constant(beta_slow, dtype=DType.float32, device=CPU())

        low = F.floor(
            self._yarn_find_correction_dim(
                low_rot, dim, base, original_max_position
            )
        )
        high = (
            F.trunc(
                self._yarn_find_correction_dim(
                    high_rot, dim, base, original_max_position
                )
            )
            + 1
        )

        return F.max(low, 0), F.min(high, dim - 1)

    def _yarn_find_correction_dim(
        self,
        num_rotations: Tensor,
        dim: int,
        base: float,
        max_position_embeddings: int,
    ) -> Tensor:
        max_pos = F.constant(
            float(max_position_embeddings),
            dtype=DType.float32,
            device=CPU(),
        )
        base_tensor = F.constant(float(base), dtype=DType.float32, device=CPU())
        dim_tensor = F.constant(float(dim), dtype=DType.float32, device=CPU())

        return (dim_tensor * F.log(max_pos / (num_rotations * 2 * math.pi))) / (
            2 * F.log(base_tensor)
        )

    def _yarn_linear_ramp_mask(
        self, min_val: Tensor, max_val: Tensor, dim: Dim
    ) -> Tensor:
        diff = max_val - min_val
        diff = F.where(
            diff == 0,
            F.constant(0.001, dtype=DType.float32, device=CPU()),
            diff,
        )

        linear_func = (
            F.arange(
                0, dim, 1, out_dim=dim, device=CPU(), dtype=DType.int64
            ).cast(DType.float32)
            - min_val
        ) / diff

        return F.clip(linear_func, 0, 1)
