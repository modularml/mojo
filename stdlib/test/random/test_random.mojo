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

from random import randn_float64, random_float64, random_si64, random_ui64, seed

from testing import assert_equal, assert_true


def test_random():
    for _ in range(100):
        var random_float = random_float64(0, 1)
        assert_true(
            random_float >= 0,
            "Value " + str(random_float) + " is not above or equal to 0",
        )
        assert_true(
            random_float <= 1,
            "Value " + str(random_float) + " is not below or equal to 1",
        )

        var random_signed = random_si64(-255, 255)
        assert_true(
            random_signed >= -255,
            "Signed value "
            + str(random_signed)
            + " is not above or equal to -255",
        )
        assert_true(
            random_signed <= 255,
            "Signed value "
            + str(random_signed)
            + " is not below or equal to 255",
        )

        var random_unsigned = random_ui64(0, 255)
        assert_true(
            random_unsigned >= 0,
            "Unsigned value "
            + str(random_unsigned)
            + " is not above or equal to 0",
        )
        assert_true(
            random_unsigned <= 255,
            "Unsigned value "
            + str(random_unsigned)
            + " is not below or equal to 255",
        )

    var random_normal = randn_float64(0, 1)
    # it's quite hard to verify that the values returned are forming a normal distribution


def test_seed():
    seed(5)
    var some_float = random_float64(0, 1)
    var some_signed_integer = random_si64(-255, 255)
    var some_unsigned_integer = random_ui64(0, 255)

    seed(5)
    assert_equal(some_float, random_float64(0, 1))
    assert_equal(some_signed_integer, random_si64(-255, 255))
    assert_equal(some_unsigned_integer, random_ui64(0, 255))


def main():
    test_random()
    test_seed()
