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
# RUN: %mojo -debug-level full %s

from testing import assert_equal


fn test_range_len() raises:
    assert_equal(range(10).__len__(), 10)
    assert_equal(range(0, 10).__len__(), 10)
    assert_equal(range(5, 10).__len__(), 5)
    assert_equal(range(10, 0, -1).__len__(), 10)
    assert_equal(range(0, 10, 2).__len__(), 5)
    assert_equal(range(38, -13, -23).__len__(), 3)


fn test_range_getitem() raises:
    assert_equal(range(10)[5], 5)
    assert_equal(range(0, 10)[3], 3)
    assert_equal(range(5, 10)[3], 8)
    assert_equal(range(10, 0, -1)[2], 8)
    assert_equal(range(0, 10, 2)[4], 8)


def main():
    test_range_len()
    test_range_getitem()
