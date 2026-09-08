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
"""Math functions and constants: trig, exponential, logarithmic, and special functions.

The `math` package provides mathematical functions and constants for numerical
computation. It includes standard mathematical operations from trigonometry,
exponential and logarithmic functions, special functions, and numerical
utilities. This package implements both precise mathematical operations and fast
approximations for performance-critical code, along with support for rounding,
clamping, and IEEE 754 floating-point behavior.

Use this package for scientific computing, numerical algorithms, graphics and
game development, statistical calculations, or any application requiring
mathematical operations beyond basic arithmetic. The `fast` module provides
optimized approximations when absolute precision can be traded for performance.
"""

# In Python, these are in the `math` package, so we also expose them here.
from std.utils.numerics import inf, isfinite, isinf, isnan, nan, nextafter

from .constants import e, pi, tau

# These are not part of Python's `math` package, but we define them here.
from .math import (
    Absable,
    Ceilable,
    CeilDivable,
    CeilDivableRaising,
    DivModable,
    Floorable,
    Powable,
    Roundable,
    Truncable,
    abs,
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
    divmod,
    erf,
    erfc,
    exp,
    exp2,
    expm1,
    comb,
    factorial,
    floor,
    fma,
    frexp,
    gamma,
    gcd,
    hypot,
    iota,
    isclose,
    rsqrt,
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
    max,
    min,
    modf,
    perm,
    pow,
    recip,
    remainder,
    round,
    scalb,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    trunc,
    ulp,
    y0,
    y1,
)
