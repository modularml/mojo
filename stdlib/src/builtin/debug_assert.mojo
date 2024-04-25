# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
"""Implements a debug assert.

These are Mojo built-ins, so you don't need to import them.
"""


from os import abort
from sys._build import is_kernels_debug_build
from sys import triple_is_nvidia_cuda, is_defined
from builtin._location import __call_location, _SourceLocation


@always_inline
fn debug_assert[stringable: Stringable](cond: Bool, msg: stringable):
    """Asserts that the condition is true.

    The `debug_assert` is similar to `assert` in C++. It is a no-op in release
    builds unless MOJO_ENABLE_ASSERTIONS is defined.

    Right now, users of the mojo-sdk must explicitly specify `-D MOJO_ENABLE_ASSERTIONS`
    to enable assertions.  It is not sufficient to compile programs with `-debug-level full`
    for enabling assertions in the library.

    Parameters:
        stringable: The type of the message.

    Args:
        cond: The bool value to assert.
        msg: The message to display on failure.
    """

    # Print an error and fail.
    alias err = is_kernels_debug_build() or is_defined[
        "MOJO_ENABLE_ASSERTIONS"
    ]()

    # Print a warning, but do not fail (useful for testing assert behavior).
    alias warn = is_defined["ASSERT_WARNING"]()

    @parameter
    if err or warn:
        if not cond:
            _debug_assert_msg[err](msg, __call_location())


@no_inline
fn _debug_assert_msg[
    err: Bool, stringable: Stringable
](msg: stringable, loc: _SourceLocation):
    """Aborts with (or prints) the given message and location.

    Note that it's important that this function doesn't get inlined; otherwise,
    an indirect recursion of @always_inline functions is possible (e.g. because
    abort's implementation could use debug_assert)
    """

    @parameter
    if triple_is_nvidia_cuda():
        # On GPUs, assert shouldn't allocate.

        @parameter
        if err:
            abort()
        else:
            print("Assert Warning")
        return

    @parameter
    if err:
        abort(loc.prefix("Assert Error: " + str(msg)))
    else:
        print(loc.prefix("Assert Warning:"), str(msg))
