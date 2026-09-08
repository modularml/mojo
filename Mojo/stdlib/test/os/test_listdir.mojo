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

from std.os import listdir
from std.pathlib import Path

from std.testing import TestSuite, assert_true


def test_listdir() raises:
    var ls = listdir(Path())
    assert_true(len(ls) > 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
