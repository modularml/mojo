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

from std.testing import (
    assert_raises,
    assert_equal,
    assert_false,
    assert_true,
    TestSuite,
)
from std.testing.suite import TestResult, TestReport, TestSuiteReport


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
    comptime funcs = __functions_in_module()
    var suite = TestSuite.discover_tests[funcs]()

    suite.skip[test_skipped]()
    suite.skip[nonconforming_name]()
    with assert_raises(
        contains=(
            "trying to skip a test that is not registered in the suite:"
            " nonconforming_name"
        )
    ):
        suite^.run()

    suite = TestSuite.discover_tests[funcs]()
    var report: TestSuiteReport
    try:
        suite.skip[test_skipped]()
        report = suite.generate_report()
    except e:
        suite^.abandon()
        raise e^

    # Make sure running the suite fails, since we have a failing test.
    with assert_raises():
        suite^.run()

    assert_equal(report.failures, 1)
    assert_equal(report.skipped, 1)
    assert_equal(report.passed, 2)
    assert_equal(len(report.reports), 4)

    assert_equal(report.reports[0].name, "test_failing")
    assert_equal(String(report.reports[0].error), "should be raised")

    assert_equal(report.reports[1].name, "test_passing_1")
    assert_false(report.reports[1].error)

    assert_equal(report.reports[2].name, "test_passing_2")
    assert_false(report.reports[2].error)

    assert_equal(report.reports[3].name, "test_skipped")
    assert_false(report.reports[3].error)

    # Separately test skipping all tests; suppress the report to avoid spam.
    var skip_all_suite = TestSuite.discover_tests[funcs]()
    skip_all_suite^.run(quiet=True, skip_all=True)

    # The `__functions_in_module()` reflection returns a Tuple, which we can't
    # slice into, so we manually build a list of functions to test that
    # discovery fails if a test function has a nonconforming signature.
    def test_nonconforming_signature(x: Int) raises:
        raise Error("should not be run")

    comptime failing_funcs = Tuple(
        test_nonconforming_signature, funcs[1], funcs[2], funcs[3], funcs[4]
    )
    with assert_raises(
        contains="'test_nonconforming_signature' has nonconforming signature"
    ):
        var suite = TestSuite.discover_tests[failing_funcs]()
        suite^.abandon()

    var result_repr = String()
    TestResult.PASS.write_repr_to(result_repr)
    assert_equal(result_repr, "TestResult.PASS")

    result_repr = String()
    TestResult.FAIL.write_repr_to(result_repr)
    assert_equal(result_repr, "TestResult.FAIL")

    result_repr = String()
    TestResult.SKIP.write_repr_to(result_repr)
    assert_equal(result_repr, "TestResult.SKIP")

    var test_report = TestReport.passed(name="my_test", duration_ns=1234)
    var report_repr = String()
    test_report.write_repr_to(report_repr)
    assert_true(report_repr.startswith("TestReport("))
    assert_true("name=" in report_repr)
    assert_true("result=TestResult.PASS" in report_repr)
    assert_true("duration_ns=1234" in report_repr)

    var suite_report_repr = String()
    report.write_repr_to(suite_report_repr)
    assert_true(suite_report_repr.startswith("TestSuiteReport("))
    assert_true("passed=2" in suite_report_repr)
    assert_true("failed=1" in suite_report_repr)
    assert_true("skipped=1" in suite_report_repr)
