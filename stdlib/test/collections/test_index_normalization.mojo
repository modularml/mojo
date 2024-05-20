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

from collections._index_normalization import (
    get_out_of_bounds_error_message,
    normalize_index,
)
from testing import assert_equal


def test_out_of_bounds_message():
    assert_equal(
        get_out_of_bounds_error_message["List"](5, 2),
        (
            "The List has a length of 2. Thus the index provided should be"
            " between -2 (inclusive) and 2 (exclusive) but the index value 5"
            " was used. Aborting now to avoid an out-of-bounds access."
        ),
    )

    assert_equal(
        get_out_of_bounds_error_message["List"](0, 0),
        (
            "The List has a length of 0. Thus it's not possible to access its"
            " values with an index but the index value 0 was used. Aborting now"
            " to avoid an out-of-bounds access."
        ),
    )
    assert_equal(
        get_out_of_bounds_error_message["InlineArray"](8, 0),
        (
            "The InlineArray has a length of 0. Thus it's not possible to"
            " access its values with an index but the index value 8 was used."
            " Aborting now to avoid an out-of-bounds access."
        ),
    )


def test_normalize_index():
    container = List[Int](1, 1, 1, 1)
    assert_equal(normalize_index[""](-4, container), 0)
    assert_equal(normalize_index[""](-3, container), 1)
    assert_equal(normalize_index[""](-2, container), 2)
    assert_equal(normalize_index[""](-1, container), 3)
    assert_equal(normalize_index[""](0, container), 0)
    assert_equal(normalize_index[""](1, container), 1)
    assert_equal(normalize_index[""](2, container), 2)
    assert_equal(normalize_index[""](3, container), 3)


def main():
    test_out_of_bounds_message()
    test_normalize_index()
