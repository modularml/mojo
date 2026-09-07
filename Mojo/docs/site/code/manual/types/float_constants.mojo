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

from std.math import copysign
from std.utils.numerics import isfinite, isinf, isnan
from std.testing import *


def main() raises:
    var inf = FloatLiteral.infinity
    assert_false(isfinite(inf))  # `False`
    assert_true(isinf(inf))  # `True`

    var neginf = FloatLiteral.negative_infinity
    assert_false(isfinite(neginf))  # `False`
    assert_true(isinf(neginf))  # `True`

    var nan = FloatLiteral.nan
    assert_true(isnan(nan))  # `True`

    var negzero = FloatLiteral.negative_zero
    assert_almost_equal(negzero, 0.0)  # `True`

    assert_equal(negzero, 0.0)  # `True`
    assert_true(copysign(1.0, negzero) < 0)  # `True`
