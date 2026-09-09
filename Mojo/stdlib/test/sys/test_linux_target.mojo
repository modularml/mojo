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
#
# This file is only run on linux targets.
#
# ===----------------------------------------------------------------------=== #

from std.sys import CompilationTarget

from std.testing import assert_false, assert_true
from std.testing import TestSuite


def test_os_query() raises:
    assert_false(CompilationTarget.is_macos())
    assert_true(CompilationTarget.is_linux())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
