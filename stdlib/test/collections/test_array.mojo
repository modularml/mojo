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
# RUN: %mojo --debug-level full %s

from testing import assert_equal

from utils.inlined_string import _ArrayMem


def main():
    test_array_mem()


def test_array_mem():
    var array = _ArrayMem[Int, 4](1)

    assert_equal(array.SIZE, 4)
    assert_equal(len(array), 4)

    # ==================================
    # Test pointer operations
    # ==================================

    var ptr = array.unsafe_ptr()
    assert_equal(ptr[0], 1)
    assert_equal(ptr[1], 1)
    assert_equal(ptr[2], 1)
    assert_equal(ptr[3], 1)
