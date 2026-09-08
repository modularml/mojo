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

from context_mgr import Timer
from std.testing import assert_raises, TestSuite


def test_timer_no_error() raises:
    with Timer():
        print("Beginning no-error execution")
        std.time.sleep(0.1)
        print("Ending no-error execution")


def test_timer_error() raises:
    with assert_raises(contains="simulated error"):
        with Timer():
            print("Beginning error execution")
            std.time.sleep(0.1)
            raise "simulated error"
            # We should not reach this line


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
