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

from std.sys import simd_bit_width
from std.sys.info import CompilationTarget

from std.testing import assert_equal, assert_false, assert_true
from std.testing import TestSuite


def test_arch_query() raises:
    assert_true(CompilationTarget.is_arm())

    assert_false(CompilationTarget.is_x86())

    assert_true(CompilationTarget.has_neon())

    assert_equal(simd_bit_width(), 128)

    assert_false(CompilationTarget.has_avx512f())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
