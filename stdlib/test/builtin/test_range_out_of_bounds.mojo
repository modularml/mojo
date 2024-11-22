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
# REQUIRES: has_not
# RUN: not --crash mojo -D ASSERT=all %s 2>&1

from testing import assert_equal


def test_range_getitem_uint_out_of_bounds():
    assert_equal(range(UInt(0), UInt.MAX)[0], UInt(0), "range(0, UInt.MAX)[0]")
    assert_equal(
        range(UInt(0), UInt.MAX)[UInt.MAX - 1],
        UInt.MAX - 1,
        "range(0, UInt.MAX)[UInt.MAX - 1]",
    )
    assert_equal(
        range(UInt(10), UInt(0), UInt(1)).__len__(),
        0,
        "invalid start/end does not generate a zero length range",
    )

    # validate basic range guarantees
    var rng_obj = range(UInt(0), UInt(10))
    assert_equal(rng_obj.__len__(), 10, "incorrect range length")

    # ensure out-of-bounds access calls abort through the assert handler
    _ = rng_obj[10]


def main():
    test_range_getitem_uint_out_of_bounds()
