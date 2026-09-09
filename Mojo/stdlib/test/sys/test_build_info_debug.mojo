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

from std.sys._build import is_debug_build, is_release_build

from std.testing import assert_false, assert_true
from std.testing import TestSuite


def test_is_debug() raises:
    assert_true(is_debug_build())
    assert_false(is_release_build())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
