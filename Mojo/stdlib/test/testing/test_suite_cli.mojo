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

from std.testing import assert_equal, assert_false, TestSuite
from std.testing.suite import TestSuiteReport


def nonconforming_name() raises:
    raise Error("should not be run")


def test_failing() raises:
    raise Error("should be raised")


def test_passing_1() raises:
    pass


def test_passing_2() raises:
    pass


def test_skipped() raises:
    raise Error("should be skipped")


def main() raises:
    var suite = TestSuite.discover_tests[__functions_in_module()]()
    var report: TestSuiteReport
    try:
        suite.skip[test_skipped]()
        report = suite.generate_report()
    except e:
        suite^.abandon()
        raise e^

    # NOTE: The suite should not fail, since we skip the failing test from the
    # CLI. The flags are passed directly to the bazel target.
    suite^.run()

    assert_equal(report.failures, 0)
    assert_equal(report.skipped, 3)
    assert_equal(report.passed, 1)
    assert_equal(len(report.reports), 4)

    assert_equal(report.reports[0].name, "test_failing")
    assert_false(report.reports[1].error)

    assert_equal(report.reports[1].name, "test_passing_1")
    assert_false(report.reports[1].error)

    assert_equal(report.reports[2].name, "test_passing_2")
    assert_false(report.reports[2].error)

    assert_equal(report.reports[3].name, "test_skipped")
    assert_false(report.reports[3].error)
