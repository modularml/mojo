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

from std.os.path import is_absolute

from std.testing import TestSuite, assert_false, assert_true


def test_is_absolute() raises:
    assert_true(is_absolute("/"))
    assert_true(is_absolute("/foo"))
    assert_true(is_absolute("/foo/bar"))

    assert_false(is_absolute(""))
    assert_false(is_absolute("foo/bar"))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
