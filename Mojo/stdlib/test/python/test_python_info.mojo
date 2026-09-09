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

from std.python import Python
from std.python._cpython import PythonVersion
from std.testing import assert_equal, TestSuite


def _test_python_version(mut python: Python) raises:
    var version = "3.10.8 (main, Nov 24 2022, 08:08:27) [Clang 14.0.6 ]"
    var python_version = PythonVersion(version)
    assert_equal(python_version.major, 3)
    assert_equal(python_version.minor, 10)
    assert_equal(python_version.patch, 8)


def test_with_python_version() raises:
    var python = Python()
    _test_python_version(python)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
