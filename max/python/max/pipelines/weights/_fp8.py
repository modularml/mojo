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

"""Host-side FP8 (E4M3FN) weight decoding for pipeline weight adapters.

Some checkpoints store a quantized Linear weight as a ``float8_e4m3fn`` tensor
plus a per-output-channel ``weight_scale``. The engine's host cast model can't
lower ``float8_e4m3fn -> float32``, so these helpers reinterpret the underlying
buffer as ``uint8`` and decode it through a precomputed table, dequantizing on
the host once at load.
"""

from __future__ import annotations

import functools

import numpy as np
import numpy.typing as npt
from max.dtype import DType
from max.graph.weights import WeightData


@functools.lru_cache(maxsize=1)
def e4m3fn_lut() -> npt.NDArray[np.float32]:
    """256-entry table decoding every float8_e4m3fn byte to float32.

    OCP E4M3FN: 1 sign / 4 exponent (bias 7) / 3 mantissa bits, finite-only
    (the sole NaN encoding is S.1111.111; max finite magnitude is 448).
    Verified bit-exact against torch.float8_e4m3fn.
    """
    table = np.zeros(256, dtype=np.float32)
    for byte in range(256):
        sign = -1.0 if (byte & 0x80) else 1.0
        exp = (byte >> 3) & 0x0F
        mant = byte & 0x07
        if exp == 0:
            value = (mant / 8.0) * 2.0 ** (1 - 7)
        elif exp == 15 and mant == 7:
            value = np.nan
        else:
            value = (1.0 + mant / 8.0) * 2.0 ** (exp - 7)
        table[byte] = sign * value
    return table


def fp8_e4m3fn_to_float32(weight: WeightData) -> npt.NDArray[np.float32]:
    """Decode a float8_e4m3fn weight to float32 via raw-byte table lookup."""
    raw = np.from_dlpack(weight.to_buffer().view(DType.uint8))
    return e4m3fn_lut()[raw]


def dequantize_rowwise_fp8(
    weight: WeightData,
    scale: WeightData,
    name: str,
    *,
    out_dtype: DType = DType.float32,
) -> WeightData:
    """Dequantize a per-output-channel (rowwise) FP8 weight.

    ``weight`` is float8_e4m3fn ``[N, K]`` paired with a per-row float ``scale``
    (``[N]`` or ``[N, 1]``). Returns ``weight * scale[:, None]`` as ``out_dtype``
    (float32 by default; pass bfloat16 for a bf16 compute path).
    """
    w_f32 = fp8_e4m3fn_to_float32(weight)
    if w_f32.ndim != 2:
        raise ValueError(
            f"FP8 weight '{name}' expected 2-D, got shape {w_f32.shape}"
        )
    scale_f32 = (
        np.from_dlpack(scale.astype(DType.float32).data)
        .astype(np.float32)
        .reshape(-1)
    )
    if scale_f32.shape[0] != w_f32.shape[0]:
        raise ValueError(
            f"FP8 scale for '{name}' has length {scale_f32.shape[0]} but "
            f"weight has {w_f32.shape[0]} output channels"
        )
    deq = np.ascontiguousarray(w_f32 * scale_f32[:, None])
    wd = WeightData.from_numpy(deq, name)
    return wd if out_dtype == DType.float32 else wd.astype(out_dtype)
