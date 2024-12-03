# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
"""Implements the math package."""

# In Python, these are in the math module, so we also expose them here.
from utils.numerics import inf, isfinite, isinf, isnan, nan, nextafter, ulp

from .constants import e, pi, tau

# These are not part of Python's `math` module, but we define them here.
from .math import (
    Ceilable,
    CeilDivable,
    CeilDivableRaising,
    Floorable,
    Truncable,
    acos,
    acosh,
    align_down,
    align_up,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    cbrt,
    ceil,
    ceildiv,
    clamp,
    copysign,
    cos,
    cosh,
    erf,
    erfc,
    exp,
    exp2,
    expm1,
    factorial,
    floor,
    fma,
    frexp,
    gamma,
    gcd,
    hypot,
    iota,
    isclose,
    isqrt,
    j0,
    j1,
    lcm,
    ldexp,
    lgamma,
    log,
    log1p,
    log2,
    log10,
    logb,
    modf,
    recip,
    remainder,
    scalb,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    trunc,
    y0,
    y1,
)
