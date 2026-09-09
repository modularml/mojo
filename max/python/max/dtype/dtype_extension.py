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

"""Extension for max.dtype to support additional attributes."""

from collections import namedtuple

from numpy import finfo as np_finfo

from .dtype import DType

_FInfoData = namedtuple("_FInfoData", ["bits", "eps", "max", "min", "tiny"])

# Hardcoded finfo values for float dtypes that numpy doesn't support.
# Values are derived from IEEE 754 / OCP MX specifications.
_HARDCODED_FINFO: dict[DType, _FInfoData] = {
    DType.bfloat16: _FInfoData(
        bits=16,
        eps=2**-7,  # 0.0078125
        max=(2 - 2**-7) * 2**127,  # 3.3895313892515355e+38
        min=-(2 - 2**-7) * 2**127,
        tiny=2**-126,  # 1.1754943508222875e-38
    ),
    DType.float8_e4m3fn: _FInfoData(
        bits=8,
        eps=2**-3,  # 0.125
        max=448.0,
        min=-448.0,
        tiny=2**-6,  # 0.015625
    ),
    DType.float8_e4m3fnuz: _FInfoData(
        bits=8,
        eps=2**-3,  # 0.125
        max=240.0,
        min=-240.0,
        tiny=2**-7,  # 0.0078125
    ),
    DType.float8_e5m2: _FInfoData(
        bits=8,
        eps=2**-2,  # 0.25
        max=57344.0,
        min=-57344.0,
        tiny=2**-14,  # 6.103515625e-05
    ),
    DType.float8_e5m2fnuz: _FInfoData(
        bits=8,
        eps=2**-2,  # 0.25
        max=57344.0,
        min=-57344.0,
        tiny=2**-15,  # 3.0517578125e-05
    ),
    DType.float8_e8m0fnu: _FInfoData(
        bits=8,
        eps=1.0,
        max=float(2**127),
        min=float(2**-127),  # No sign bit; min is smallest positive value.
        tiny=float(2**-127),
    ),
    DType.float4_e2m1fn: _FInfoData(
        bits=4,
        eps=0.5,
        max=6.0,
        min=-6.0,
        tiny=1.0,
    ),
    DType.float6_e2m3fn: _FInfoData(
        bits=6,
        eps=2**-3,  # 0.125
        max=7.5,
        min=-7.5,
        tiny=1.0,
    ),
    DType.float6_e3m2fn: _FInfoData(
        bits=6,
        eps=2**-2,  # 0.25
        max=28.0,
        min=-28.0,
        tiny=2**-2,  # 0.25
    ),
}


class finfo:
    """Numerical properties of a floating point :class:`~max.dtype.DType`.

    This is modeled after ``torch.finfo``, providing ``bits``, ``eps``,
    ``max``, ``min``, ``tiny``, ``smallest_normal``, and ``dtype``
    attributes for every MAX float dtype, including bfloat16, float8, and
    float4 types that numpy cannot represent.

    Args:
        dtype: A floating-point ``DType`` to query.

    Raises:
        TypeError: If *dtype* is not a floating-point type.
    """

    bits: int
    eps: float
    max: float
    min: float
    tiny: float
    dtype: DType

    def __init__(self, dtype: DType):
        if not dtype.is_float():
            raise TypeError(
                f"finfo only supports floating-point types, got {dtype.name}"
            )

        hardcoded = _HARDCODED_FINFO.get(dtype)
        if hardcoded is not None:
            self.bits = hardcoded.bits
            self.eps = hardcoded.eps
            self.max = hardcoded.max
            self.min = hardcoded.min
            self.tiny = hardcoded.tiny
        else:
            info = np_finfo(dtype.to_numpy())
            self.bits = info.bits
            self.eps = float(info.eps)
            self.max = float(info.max)
            self.min = float(info.min)
            self.tiny = float(info.tiny)
        self.dtype = dtype

    @property
    def smallest_normal(self) -> float:
        """Alias for ``tiny`` (``torch.finfo`` compatibility)."""
        return self.tiny

    def __repr__(self) -> str:
        return (
            f"finfo(dtype={self.dtype.name}, bits={self.bits},"
            f" eps={self.eps}, max={self.max}, min={self.min},"
            f" tiny={self.tiny})"
        )


DType.finfo = finfo  # type: ignore[attr-defined]
