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
# RUN: %mojo %s

from math import modf

from test_utils import libm_call
from testing import assert_almost_equal, assert_equal


def test_modf():
    var i32 = modf(Int32(123))
    assert_equal(i32[0], 123)
    assert_equal(i32[1], 0)

    var f32 = modf(Float32(123.5))
    assert_almost_equal(f32[0], 123)
    assert_almost_equal(f32[1], 0.5)

    var f64 = modf(Float64(123.5))
    assert_almost_equal(f64[0], 123)
    assert_almost_equal(f64[1], 0.5)

    f64 = modf(Float64(0))
    assert_almost_equal(f64[0], 0)
    assert_almost_equal(f64[1], 0)

    f64 = modf(Float64(0.5))
    assert_almost_equal(f64[0], 0)
    assert_almost_equal(f64[1], 0.5)

    f64 = modf(Float64(-0.5))
    assert_almost_equal(f64[0], -0)
    assert_almost_equal(f64[1], -0.5)

    f64 = modf(Float64(-1.5))
    assert_almost_equal(f64[0], -1)
    assert_almost_equal(f64[1], -0.5)


def main():
    test_modf()
