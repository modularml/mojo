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

from std.sys.compile import DebugLevel, OptimizationLevel

from std.testing import assert_equal
from std.testing import TestSuite


def test_compile_options() raises:
    assert_equal(Int(OptimizationLevel), 3)
    assert_equal(String(DebugLevel), "none")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
