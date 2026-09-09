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

from std.sys import size_of

from std.testing import assert_equal
from std.testing import TestSuite


def test_size_of_dtypes() raises:
    assert_equal(size_of[DType.int8](), 1)
    assert_equal(size_of[DType.int16](), 2)
    assert_equal(size_of[DType.int32](), 4)
    assert_equal(size_of[DType.int64](), 8)
    assert_equal(size_of[DType.float32](), 4)
    assert_equal(size_of[DType.float64](), 8)
    assert_equal(size_of[DType.bool](), 1)
    assert_equal(size_of[DType.int](), 8)
    assert_equal(size_of[DType.uint](), 8)
    assert_equal(size_of[DType.float8_e4m3fn](), 1)
    assert_equal(size_of[DType.float8_e5m2fnuz](), 1)
    assert_equal(size_of[DType.float8_e4m3fnuz](), 1)
    assert_equal(size_of[DType.bfloat16](), 2)
    assert_equal(size_of[DType.float16](), 2)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
