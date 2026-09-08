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

from std.testing import TestSuite, assert_equal

from std.utils._select import _select_register_value


def test_select_register_value() raises:
    assert_equal(_select_register_value(True, 42, 100), 42)
    assert_equal(_select_register_value(False, 42, 100), 100)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
