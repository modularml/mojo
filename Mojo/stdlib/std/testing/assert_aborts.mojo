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

from std.ffi import CStringSlice
from std.os import abort
from std.os.env import getenv
from std.os.process import Pipe, Process, ProcessStatus
from std.reflection import call_location, SourceLocation
from std.sys import argv
from std.sys._io import stderr, stdout
from std.sys._libc import _get_environ, close, dup2
from std.sys.compile import SanitizeAddress
from std.time import perf_counter, sleep

# Which call site (file:line:col) the re-exec'd child should run.
comptime _LOCATION_ENV = "__MOJO_TEST_EXPECT_ABORT_LOCATION_TARGET"
# File descriptor number (as a string) the child should dup2 onto stdout/stderr.
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
    """Redirects stdout/stderr to the parent-provided fd, then calls `f`."""
    var fd = Int32(Int(getenv(_OUTPUT_ENV)))
    _ = dup2(fd, Int32(stdout.value))
    _ = dup2(fd, Int32(stderr.value))
    _ = close(fd)
    f()


def _build_child_env(location: SourceLocation, write_fd: Int) -> List[String]:
    """Copies the current environment, adds the assert_aborts control vars."""
    var env = List[String]()
    var envp = _get_environ()
    if not envp:
        env.append(String(t"{_LOCATION_ENV}={location}"))
        env.append(String(t"{_OUTPUT_ENV}={write_fd}"))
        return env^

    var ptr = envp.value()
    var i = 0
    while True:
        var entry = ptr[unsafe_offset=i]
        if not entry:
            break
        var s = String(unsafe_from_utf8=entry.value().as_bytes())
        if not s.startswith(_LOCATION_ENV + "=") and not s.startswith(
            _OUTPUT_ENV + "="
        ):
            env.append(s)
        i += 1

    env.append(String(t"{_LOCATION_ENV}={location}"))
    env.append(String(t"{_OUTPUT_ENV}={write_fd}"))
    return env^


def _read_all(mut p: Pipe) raises -> String:
    """Reads all bytes from the read end of a pipe."""
    var result = List[UInt8]()
    var buf = Array[UInt8, 4096](fill=0)
    while True:
        var n = p.read_bytes(Span(buf))
        if n == 0:
            break
        for i in range(n):
            result.append(buf[i])
    result.append(0)
    return String(unsafe_from_utf8=result^)


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

    # The write end must NOT be close-on-exec so the child inherits it.
    var output_pipe = Pipe(in_close_on_exec=True, out_close_on_exec=False)
    var write_fd = rebind[Int](output_pipe.fd_out.value())

    var child_env = _build_child_env(location, write_fd)

    var rest = List[String]()
    for arg in self_argv[1:]:
        rest.append(String(arg))

    var child = Process.run(String(self_argv[0]), rest, env=child_env)
    # Close the parent's copy of the write end so we get EOF when
    # the child exits (or aborts).
    output_pipe.set_input_only()

    var status: ProcessStatus
    var timed_out = False
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

    var captured = _read_all(output_pipe)

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
