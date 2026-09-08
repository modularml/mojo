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

from layout import TileTensor, row_major
from nn._ragged_utils import get_batch_from_row_offsets
from std.testing import assert_equal


def test_get_batch_from_row_offsets() raises:
    comptime batch_size = 9
    var storage = Array[UInt32, batch_size + 1](
        fill_with=lambda (i: Int) -> UInt32: UInt32(i * 100)
    )
    var prefix_sums = TileTensor(storage, row_major[batch_size + 1]())

    assert_equal(
        get_batch_from_row_offsets(prefix_sums, 100),
        1,
    )
    assert_equal(
        get_batch_from_row_offsets(prefix_sums, 0),
        0,
    )
    assert_equal(
        get_batch_from_row_offsets(prefix_sums, 899),
        8,
    )
    assert_equal(
        get_batch_from_row_offsets(prefix_sums, 555),
        5,
    )


def main() raises:
    test_get_batch_from_row_offsets()
