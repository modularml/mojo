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
# RUN: echo -n | %mojo %s

import std.sys
from std.io.io import _fdopen

from std.testing import testing, TestSuite


def test_read_until_delimiter_raises_eof() raises:
    var stdin = _fdopen["r"](std.sys.stdin)
    with testing.assert_raises(contains="EOF"):
        _ = stdin.read_until_delimiter("\n")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
