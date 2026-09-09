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
"""Native FP8 linear for the Ideogram 4 DiT on MI355.

The ``ideogram-4-fp8`` checkpoint ships ``float8_e4m3fn`` Linear weights with a
per-output-channel ``weight_scale``. The default loader dequantizes them to
bf16; this module instead keeps the weight packed and runs the projection as a
native FP8 GEMM: the bf16 activation is dynamically quantized per token
(colwise) and multiplied by the rowwise-scaled FP8 weight via
:func:`~max.nn.kernels.dynamic_scaled_matmul`, which dispatches to the tuned
AMD FP8 matmul (plus a scale epilogue) on gfx950 and returns bf16.

The weight and the per-token activation share float32 rowwise/colwise scales,
mirroring the proven dynamic path in :mod:`max.nn.quant_ops`. Only bias-free
projections are supported (every FP8 Linear in the Ideogram 4 DiT is
bias-free).
"""

from __future__ import annotations

import functools
import operator

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.graph import DimLike
from max.nn.kernels import (
    dynamic_scaled_matmul,
    quantize_dynamic_scaled_float8,
)
from max.nn.quant_config import (
    InputScaleSpec,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)

# Scales are kept in float32 to preserve the checkpoint's per-channel scale
# precision (the per-token activation scales are produced in the same dtype).
_SCALE_DTYPE = DType.float32

_INPUT_SCALE_SPEC = InputScaleSpec(
    granularity=ScaleGranularity.COLWISE,
    origin=ScaleOrigin.DYNAMIC,
    dtype=_SCALE_DTYPE,
)
_WEIGHT_SCALE_SPEC = WeightScaleSpec(
    granularity=ScaleGranularity.ROWWISE,
    dtype=_SCALE_DTYPE,
)

# ``dynamic_scaled_matmul`` / ``quantize_dynamic_scaled_float8`` operate on
# graph values; wrap them so they accept experimental ``Tensor`` arguments.
_quantize = F.functional(quantize_dynamic_scaled_float8)
_scaled_matmul = F.functional(dynamic_scaled_matmul)


def fp8_quantize_2d(x_2d: Tensor) -> tuple[Tensor, Tensor]:
    """Dynamically quantize a rank-2 ``[M, K]`` bf16 activation to FP8.

    Returns ``(x_fp8, x_scales)`` with per-token (colwise) float32 scales.
    """
    return _quantize(
        x_2d,
        _INPUT_SCALE_SPEC,
        _WEIGHT_SCALE_SPEC,
        scales_type=_SCALE_DTYPE,
        out_type=DType.float8_e4m3fn,
    )


def fp8_matmul_2d(
    x_fp8: Tensor,
    x_scales: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
) -> Tensor:
    """Rowwise/colwise FP8 GEMM of a pre-quantized ``[M, K]`` activation.

    ``weight`` is ``[N, K]`` FP8 and ``weight_scale`` is ``[N, 1]`` float32;
    returns the ``[M, N]`` bf16 product. Dispatches to the tuned AMD FP8
    matmul (plus scale epilogue) on gfx950.
    """
    return _scaled_matmul(
        x_fp8,
        weight,
        x_scales,
        weight_scale,
        _INPUT_SCALE_SPEC,
        _WEIGHT_SCALE_SPEC,
        out_type=DType.bfloat16,
    )


class Ideogram4FP8Linear(Module[[Tensor], Tensor]):
    """A bias-free linear projection executed as a native FP8 GEMM.

    By convention the weight is stored transposed, i.e. ``weight.shape ==
    [out_dim, in_dim]``, matching :class:`~max.experimental.nn.Linear` and the
    checkpoint layout.

    Args:
        in_dim: The contraction (input) dimension.
        out_dim: The output dimension.
    """

    weight: Tensor
    """The ``float8_e4m3fn`` weight of shape ``[out_dim, in_dim]``."""

    weight_scale: Tensor
    """The per-output-channel float32 scale of shape ``[out_dim, 1]``."""

    def __init__(self, in_dim: DimLike, out_dim: DimLike) -> None:
        # Zero placeholders only fix the parameter shape/dtype; the real
        # values are bound from the checkpoint at ``compile(weights=...)``.
        self.weight = Tensor.zeros([out_dim, in_dim], dtype=DType.float8_e4m3fn)
        self.weight_scale = Tensor.zeros([out_dim, 1], dtype=_SCALE_DTYPE)

    def forward(self, x: Tensor) -> Tensor:
        """Applies the FP8 linear projection to ``x``.

        The quantize/matmul kernels require rank-2 operands, so any leading
        dimensions are flattened to a single token axis and restored on the
        output.
        """
        lead = list(x.shape[:-1])
        k = x.shape[-1]
        m = functools.reduce(operator.mul, lead)
        x_2d = F.reshape(x, [m, k])
        x_fp8, x_scales = fp8_quantize_2d(x_2d)
        y_2d = fp8_matmul_2d(x_fp8, x_scales, self.weight, self.weight_scale)
        return F.reshape(y_2d, [*lead, self.weight.shape[0]])
