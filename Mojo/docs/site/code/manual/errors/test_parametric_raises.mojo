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
from std.testing import assert_equal, assert_raises, TestSuite

from parametric_raises import run_action, fetch_data, parse_config, get_value


def test_run_action_network_error() raises:
    with assert_raises(contains="NetworkError"):
        _ = run_action(fetch_data)


def test_run_action_parse_error() raises:
    with assert_raises(contains="ParseError"):
        _ = run_action(parse_config)


def test_run_action_no_error() raises:
    assert_equal(run_action(get_value), 99)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
