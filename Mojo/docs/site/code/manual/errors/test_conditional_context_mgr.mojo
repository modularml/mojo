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
import std.time

from conditional_context_mgr import ConditionalTimer, flaky_identity
from std.testing import assert_raises, TestSuite


def test_conditional_timer_no_error() raises:
    with ConditionalTimer():
        print("Beginning no-error execution")
        std.time.sleep(0.1)
        var i = 1
        _ = flaky_identity(i)
        print("Ending no-error execution")


def test_conditional_timer_suppressed_error() raises:
    var i = 2
    with assert_raises(contains="just a warning"):
        _ = flaky_identity(i)

    with ConditionalTimer():
        print("Beginning no-error execution")
        std.time.sleep(0.1)
        _ = flaky_identity(i)
        print("Ending no-error execution")


def test_conditional_timer_propagated_error() raises:
    with assert_raises(contains="really bad"):
        with ConditionalTimer():
            print("Beginning propagated error execution")
            std.time.sleep(0.1)
            var i = 4
            _ = flaky_identity(i)
            # We should not reach this line


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
