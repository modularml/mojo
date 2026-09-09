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

from std.collections import List
from std.os.path import exists
from std.os import Process, setenv
from std.os.process import Pipe

from std.testing import (
    assert_false,
    assert_raises,
    assert_true,
    assert_equal,
)


def test_pipe() raises:
    var p = Pipe()
    var s = Array[UInt8, 5](fill=0)
    p.write_bytes("hello".as_bytes())
    assert_equal(p.read_bytes(Span(s)), 5)
    assert_true(Span(s) == "hello".as_bytes())
    p.set_output_only()
    with assert_raises():
        _ = p.read_bytes(Span(s))


def test_process_run() raises:
    print("== test_process_run")
    # CHECK-LABEL: == test_process_run
    # CHECK-NEXT: == TEST_ECHO
    var command = "echo"
    _ = Process.run(command, ["== TEST_ECHO"])


def test_process_wait() raises:
    print("== test_process_wait")
    var command = "echo"
    var p = Process.run(command, ["== TEST_WAIT"])
    assert_true(p.wait().has_exited())
    assert_equal(p.wait().exit_code.value(), 0)
    assert_false(p.kill())


def test_process_kill() raises:
    print("== test_process_kill")
    var command = "sleep"
    var p = Process.run(command, ["60"])
    assert_true(p.kill())
    assert_true(p.wait().has_exited())


def test_process_run_missing() raises:
    print("== test_process_run_missing")
    # CHECK-LABEL: == test_process_run_missing
    # CHECK-NEXT: Failed to execute ThIsFiLeCoUlDNoTPoSsIbLlYExIsT.NoTAnExTeNsIoN, EINT error code: 2
    var missing_executable_file = (
        "ThIsFiLeCoUlDNoTPoSsIbLlYExIsT.NoTAnExTeNsIoN"
    )

    # verify that the test file does not exist before starting the test
    assert_false(
        exists(missing_executable_file),
        "Unexpected file '" + missing_executable_file + "' it should not exist",
    )

    try:
        _ = Process.run(missing_executable_file, List[String]())
    except e:
        print(e)

    with assert_raises():
        _ = Process.run(missing_executable_file, List[String]())


def test_process_inherits_env() raises:
    # Set a unique env var and verify the child process inherits it.
    # `printenv VAR` exits 0 if the variable is set, 1 if unset.
    _ = setenv("_MOJO_TEST_ENV_INHERIT", "inherited_ok")

    var p = Process.run("printenv", ["_MOJO_TEST_ENV_INHERIT"])
    var status = p.wait()
    assert_equal(status.exit_code.value(), 0)


def main() raises:
    test_process_run()
    test_process_run_missing()
    test_process_wait()
    test_process_kill()
    test_pipe()
    test_process_inherits_env()
