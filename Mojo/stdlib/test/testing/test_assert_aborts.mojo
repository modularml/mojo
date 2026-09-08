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

from std.os import abort
from std.testing import assert_raises, _assert_aborts, TestSuite
from std.time import sleep


def test_passes_when_closure_aborts() raises:
    def aborts():
        abort("assert_aborts self-test: aborted as expected")

    _assert_aborts(
        aborts, contains="assert_aborts self-test: aborted as expected"
    )


def test_passes_when_capturing_closure_aborts() raises:
    var name = "mojo"

    def aborts() {imm}:
        abort(name)

    _assert_aborts(aborts, contains=name)


def test_passes_when_closure_aborts_and_no_contains_given() raises:
    def aborts():
        abort("assert_aborts self-test: message is irrelevant here")

    _assert_aborts(aborts)


def test_raises_when_closure_does_not_abort() raises:
    def does_not_abort():
        pass

    with assert_raises(contains="expected the process to abort"):
        _assert_aborts(does_not_abort, contains="this message is never checked")


def test_raises_when_message_does_not_match() raises:
    def aborts_with_a_different_message():
        abort("assert_aborts self-test: the real failure message")

    with assert_raises(contains="did not contain the expected message"):
        _assert_aborts(
            aborts_with_a_different_message,
            contains="this text will never appear",
        )


def test_raises_when_closure_does_not_abort_in_time() raises:
    def hangs():
        sleep(3600.0)

    with assert_raises(contains="was still running after"):
        _assert_aborts(hangs, timeout=0.5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
