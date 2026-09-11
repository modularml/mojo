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

# These tests currently abort the test process instead of failing, which
# would hide every other result in the file they live in. They are kept
# separate from test_runner.mojo until the shrinker no longer aborts.

from std.testing import assert_equal, assert_raises, TestSuite
from std.testing.prop import PropTest, PropTestConfig, Rng, Strategy
from std.testing.prop._shrinking import Shrinker
from std.testing.prop.strategy.string_strategy import *


def test_shrink_string_to_single_char() raises:
    """Shrinking a string must be able to drop characters, which changes how
    many stream words the strategy consumes."""

    def prop(var s: String) raises:
        if s.byte_length() > 0:
            raise Error("nonempty")

    var strat = String.ascii_strategy(only_printable=True)
    # A continue word before each of three characters, then a stop word.
    var shrinker = Shrinker[type_of(strat), prop](
        strat^,
        [UInt64.MAX, 0, UInt64.MAX, 0, UInt64.MAX, 0, 0],
        Error("nonempty"),
    )

    assert_equal(shrinker.shrink().value, " ")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
