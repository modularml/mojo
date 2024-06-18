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

from .math import (
    Ceilable,
    CeilDivable,
    CeilDivableRaising,
    Floorable,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    cbrt,
    ceil,
    # comb,  # TODO: implement this
    copysign,
    cos,
    cosh,
    # degrees,  # TODO:  implement this
    # dist,  # TODO:  implement this
    # e,  # TODO:  implement this
    erf,
    erfc,
    exp,
    exp2,
    expm1,
    # fabs,  # TODO:  implement this
    factorial,
    floor,
    # fmod,  # TODO:  implement this
    frexp,
    # fsum,  # TODO:  implement this
    gamma,
    gcd,
    hypot,
    isclose,
    # isqrt,  # TODO:  implement this
    lcm,
    ldexp,
    lgamma,
    log,
    log10,
    log1p,
    log2,
    # modf,  # TODO:  implement this
    # perm,  # TODO:  implement this
    # pi,  # TODO:  implement this
    # pow,  # TODO:  implement this. Note that it's different from the builtin.
    # prod,  # TODO:  implement this
    # radians,  # TODO:  implement this
    remainder,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    # tau,  # TODO:  implement this
    trunc,
)

# These are not part of Python's `math` module, but we define them here.
from .math import (
    align_down,
    align_up,
    ceildiv,
    fma,
    iota,
    j0,
    j1,
    logb,
    rsqrt,
    scalb,
    y0,
    y1,
)


# In Python, these are in the math module, so we also expose them here.
from utils.numerics import inf, isfinite, isinf, isnan, nan, nextafter, ulp
