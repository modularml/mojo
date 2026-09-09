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

from std.testing import *
from std.testing import TestSuite

from std.compile import compile_info


def test_compile_llvm() raises:
    def my_add_function[
        dtype: DType, size: Int
    ](x: SIMD[dtype, size], y: SIMD[dtype, size]) -> SIMD[dtype, size]:
        return x + y

    comptime func = my_add_function[.float32, 4]
    assert_true("fadd" in compile_info[func, emission_kind="llvm"]())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
