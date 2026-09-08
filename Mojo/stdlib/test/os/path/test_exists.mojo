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

from std.os.path import exists, lexists
from std.pathlib import Path, cwd

from std.reflection import source_location
from std.testing import TestSuite, assert_false, assert_true


def test_exists() raises:
    assert_true(exists(source_location().file_name()))
    assert_true(lexists(source_location().file_name()))

    assert_false(exists("this/file/does/not/exist"))
    assert_false(lexists("this/file/does/not/exist"))

    assert_true(exists(cwd()))
    assert_true(lexists(cwd()))

    assert_true(exists(Path()))
    assert_true(lexists(Path()))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
