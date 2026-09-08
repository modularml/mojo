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
"""Asserts that code aborts the process, without losing the rest of the test suite.

A hard abort (e.g. a failed `debug_assert`) kills the whole process, so it
can't be caught with `try`/`except`. This runs the code under test in a
re-exec'd child and checks that the child crashed with the expected message.
"""

from std.os import abort, remove
from std.os.env import getenv, setenv, unsetenv
from std.os.path import exists
from std.os.process import Process, ProcessStatus
from std.reflection import call_location, SourceLocation
from std.sys import argv
from std.sys._io import stderr, stdout
from std.sys._libc import dup2
from std.sys.compile import SanitizeAddress
from std.tempfile import NamedTemporaryFile
from std.time import perf_counter, sleep

# Which call site (file:line:col) the re-exec'd child should run.
comptime _LOCATION_ENV = "__MOJO_TEST_EXPECT_ABORT_LOCATION_TARGET"
# Path the child's stdout/stderr get redirected to before running.
comptime _OUTPUT_ENV = "__MOJO_TEST_EXPECT_ABORT_OUTPUT"
# How long to wait for the child to abort before killing it.
comptime _DEFAULT_TIMEOUT = 30.0
# How often to check on the child while waiting for it.
comptime _POLL_INTERVAL = 0.01


@always_inline
def _assert_aborts(
    f: Some[def() raises],
    *,
    contains: Optional[String] = None,
    timeout: Float64 = _DEFAULT_TIMEOUT,
) raises:
    """Asserts that calling `f` aborts the process.

    Runs `f` in a re-exec'd child process so the abort doesn't take down the
    rest of the suite. Passes if `f` aborts; if `contains` is given, also
    requires that substring to appear in the child's captured stdout/stderr.

    Example:

    ```mojo
    from std.os import abort
    from std.testing import assert_aborts, TestSuite

    def test_abort() raises:
        def trigger() raises {} -> None:
            abort("oh no!")

        assert_aborts(trigger, contains="oh no!")

    def main() raises:
        TestSuite.discover_tests[__functions_in_module()]().run()
    ```

    Args:
        f: The closure to run. It is expected to abort the process.
        contains: A substring expected to appear in the aborting process's
            combined stdout and stderr. If omitted, only the abort itself
            is checked.
        timeout: How many seconds to wait for the child to abort before
            killing it and failing.

    Raises:
        If `f` returns without aborting, the child isn't killed by a
        signal, the child doesn't abort within `timeout` seconds, or (when
        `contains` is given) the captured output doesn't contain it.
    """
    # TODO: Get this to run under ASAN
    # Every _assert_aborts() call re-execs the whole test binary, and
    # under ASAN that re-exec is slow enough that a handful of calls can push
    # a test past its CI timeout.
    comptime if SanitizeAddress:
        return

    _assert_aborts_impl(
        f, contains=contains, timeout=timeout, location=call_location()
    )


@no_inline
def _assert_aborts_impl(
    f: Some[def() raises],
    *,
    contains: Optional[String],
    timeout: Float64,
    location: SourceLocation,
) raises:
    var target = getenv(_LOCATION_ENV)

    # We're the re-exec'd child for this exact call site, run it.
    if target == String(location):
        _run(f)
        return

    # No location set: this is the original, top-level call.
    if not target:
        _spawn_and_check(location, contains, timeout)
        return

    # A different (sibling) call site, so we should skip.
    # The child spawned for that call site will check it.
    return


def _run(f: Some[def() raises]) raises:
    """Redirects stdout/stderr to the parent-provided file, then calls `f`."""
    with open(getenv(_OUTPUT_ENV), "w") as out_file:
        var fd = Int32(out_file._get_raw_fd())
        # redirect stdout/stderr to the output file
        _ = dup2(fd, Int32(stdout.value))
        _ = dup2(fd, Int32(stderr.value))
    f()


def _spawn_and_check(
    location: SourceLocation, contains: Optional[String], timeout: Float64
) raises:
    """Re-execs this binary scoped to `location`, then checks how it died."""
    var self_argv = argv()
    if self_argv[0].endswith(".mojo"):
        raise Error(
            t"assert_aborts needs to re-run this test as a separate"
            t" process, but there's no compiled binary to re-run. It"
            t" looks like this was started with `mojo {self_argv[0]}`"
            t" directly. Build it first (`mojo build {self_argv[0]}`)"
            t" and run the result, or run it as its normal `mojo_test`"
            t" bazel target."
        )

    var tmp = NamedTemporaryFile(mode="w", delete=False)
    var out_path = tmp.name
    tmp.close()

    var rest = List[String]()
    for arg in self_argv[1:]:
        rest.append(String(arg))

    # `setenv` mutates our own environment, which `Process.run` inherits
    # into the child.
    _ = setenv(_LOCATION_ENV, String(location))
    _ = setenv(_OUTPUT_ENV, out_path)

    def unsetenvs():
        _ = unsetenv(_LOCATION_ENV)
        _ = unsetenv(_OUTPUT_ENV)

    var status: ProcessStatus
    var timed_out = False
    try:
        var child = Process.run(String(self_argv[0]), rest)
        var deadline = perf_counter() + timeout
        status = child.poll()
        while not status.has_exited():
            if perf_counter() >= deadline:
                _ = child.kill()
                status = child.wait()
                timed_out = True
                break
            sleep(_POLL_INTERVAL)
            status = child.poll()
    except e:
        # process.run/wait failed!
        # Restore the environment before propagating.
        unsetenvs()
        raise e^

    unsetenvs()

    var captured: String
    try:
        captured = open(out_path, "r").read()
        remove(out_path)
    except e:
        abort(
            t"The test framework failed to open/remove temp file '{out_path}'"
            t" used for assert_aborts.\nError: {e}"
        )

    if timed_out:
        raise Error(
            t"assert_aborts: expected the process to abort, but it was still"
            t" running after {timeout}s and was killed. The code under test"
            t" may be hanging rather than aborting.\nCaptured output:"
            t"\n{captured}"
        )
    if not status.term_signal:
        var note = contains.map(
            lambda (s: String) -> String: String(
                t' (expected substring: "{s}")'
            )
        ).or_else("")
        raise Error(
            t"assert_aborts: expected the process to abort, but it exited"
            t" normally{note}.\nCaptured output:\n{captured}"
        )
    if contains:
        var message = contains.value()
        if message not in captured:
            raise Error(
                t"assert_aborts: the process aborted, but the captured output"
                t" did not contain the expected message.\nExpected substring:"
                t" '{message}'\nActual output:\n{captured}"
            )
