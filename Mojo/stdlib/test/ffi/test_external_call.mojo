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

import std.os
from std.ffi import CStringSlice, c_char, c_int, external_call
from std.memory import alloc
from std.os import remove
from std.pathlib import Path
from std.sys._libc import FcntlCommands, FcntlFDFlags, fcntl
from std.tempfile import gettempdir
from std.testing import TestSuite, assert_equal, assert_false, assert_true


struct RegisterPassablePointer(RegisterPassable):
    var pointer: OptionalPointer[NoneType, UntrackedOrigin[mut=True]]


def test_external_call_handles_rp_return_types() raises:
    var path = "/does/not/exist/here/file.file"
    var mode = "r"
    var result = external_call["fopen", RegisterPassablePointer](
        path.as_c_string_slice(), mode.as_c_string_slice()
    )
    assert_false(result.pointer)


def test_snprintf_mixed_variadic_args() raises:
    """Formats through a real C variadic callee.

    Every argument after the format string travels the variadic path, which
    AAPCS on ARM64 macOS passes on the stack rather than in registers. Without
    a variadic callee declaration these land in the wrong slots and the
    conversions read unrelated memory.
    """
    comptime SIZE = 64
    var allocation = alloc[c_char]({count = SIZE}).into_managed()
    var buf = allocation.unsafe_ptr()
    var fmt = "[%d|%s|%d|%.2f|%c]".as_c_string_slice()
    var word = "mid".as_c_string_slice()

    var written = external_call["snprintf", c_int, num_fixed_args=3](
        buf,
        Int(SIZE),
        fmt.ptr(),
        c_int(42),
        word.ptr(),
        c_int(-7),
        Float64(2.5),
        c_int(ord("z")),
    )

    var formatted = String(
        StringSlice(unsafe_from_utf8=CStringSlice(unsafe_from_ptr=buf))
    )

    assert_equal(formatted, "[42|mid|-7|2.50|z]")
    assert_equal(Int(written), 18)


def test_open_honors_mode_argument() raises:
    """Checks that `open()`'s variadic mode argument reaches the callee.

    `_open_file` passes mode `0o666` as a variadic argument to `open(2)`. When
    that argument is misplaced the file is created with whatever bits happen to
    be in the register the kernel reads, which used to require an `fchmod()`
    fixup afterwards.
    """
    var path = Path(gettempdir().value()) / "test_external_call_variadic.txt"
    try:
        remove(path)
    except:
        pass

    with open(path, "w") as f:
        f.write("variadic")

    var mode = std.os.stat(path).st_mode
    remove(path)

    # The requested 0o666 is masked by umask, so only assert the bits umask
    # cannot clear for the owner, plus the absence of execute bits.
    assert_equal(mode & 0o600, 0o600)
    assert_equal(mode & 0o111, 0)


def test_fcntl_forwards_variadic_argument() raises:
    """Round-trips a flag through `fcntl()`'s variadic third argument.

    The wrapper takes the flag through an argument pack, which reaches the
    callee as a reference unless the pack is loaded, so this asserts on the
    flag `fcntl()` actually applied rather than on its return value alone.
    """
    var path = Path(gettempdir().value()) / "test_external_call_fcntl.txt"
    try:
        remove(path)
    except:
        pass

    with open(path, "w") as f:
        var fd = c_int(f.handle)
        assert_true(
            fcntl(fd, FcntlCommands.F_SETFD, FcntlFDFlags.FD_CLOEXEC) != -1
        )
        var flags = fcntl(fd, FcntlCommands.F_GETFD, 0)
        assert_equal(flags & FcntlFDFlags.FD_CLOEXEC, FcntlFDFlags.FD_CLOEXEC)

    remove(path)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
