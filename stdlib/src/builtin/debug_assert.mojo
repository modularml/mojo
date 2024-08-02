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
from sys import is_defined, triple_is_nvidia_cuda
from sys._build import is_kernels_debug_build

from builtin._location import __call_location, _SourceLocation

# Print an error and fail.
alias _ERROR_ON_ASSERT = is_kernels_debug_build() or is_defined[
    "MOJO_ENABLE_ASSERTIONS"
]()

# Print a warning, but do not fail (useful for testing assert behavior).
alias _WARN_ON_ASSERT = is_defined["ASSERT_WARNING"]()


@always_inline
fn debug_assert[
    func: fn () capturing -> Bool, formattable: Formattable
](message: formattable):
    """Asserts that the condition is true.

    The `debug_assert` is similar to `assert` in C++. It is a no-op in release
    builds unless MOJO_ENABLE_ASSERTIONS is defined.

    Right now, users of the mojo-sdk must explicitly specify `-D MOJO_ENABLE_ASSERTIONS`
    to enable assertions.  It is not sufficient to compile programs with `-debug-level full`
    for enabling assertions in the library.

    Parameters:
        func: The function to invoke to check if the assertion holds. Can be used
            if the function is side-effecting, in which case a debug_assert taking
            a Bool will evaluate the expression producing the Bool even in release mode.
        formattable: The type of the message.

    Args:
        message: The message to convert to `String` before displaying it on failure.
    """

    @parameter
    if _ERROR_ON_ASSERT or _WARN_ON_ASSERT:
        debug_assert(func(), message)


@always_inline
fn debug_assert[formattable: Formattable](cond: Bool, message: formattable):
    """Asserts that the condition is true.

    The `debug_assert` is similar to `assert` in C++. It is a no-op in release
    builds unless MOJO_ENABLE_ASSERTIONS is defined.

    Right now, users of the mojo-sdk must explicitly specify `-D MOJO_ENABLE_ASSERTIONS`
    to enable assertions.  It is not sufficient to compile programs with `-debug-level full`
    for enabling assertions in the library.

    Parameters:
        formattable: The type of the message.

    Args:
        cond: The bool value to assert.
        message: The message to convert to `String` before displaying it on failure.
    """

    @parameter
    if _ERROR_ON_ASSERT or _WARN_ON_ASSERT:
        if cond:
            return
        _debug_assert_msg[is_warning=_WARN_ON_ASSERT](
            message, __call_location()
        )


@no_inline
fn _debug_assert_msg[
    formattable: Formattable, //, *, is_warning: Bool = False
](msg: formattable, loc: _SourceLocation):
    """Aborts with (or prints) the given message and location.

    This function is intentionally marked as no_inline to reduce binary size.

    Note that it's important that this function doesn't get inlined; otherwise,
    an indirect recursion of @always_inline functions is possible (e.g. because
    abort's implementation could use debug_assert)
    """

    @parameter
    if triple_is_nvidia_cuda():
        # On GPUs, assert shouldn't allocate.

        @parameter
        if is_warning:
            print("Assert Warning")
        else:
            abort()
    else:

        @parameter
        if is_warning:
            print(loc.prefix("Assert Warning:"), msg)
        else:
            abort(loc.prefix("Assert Error: " + String.format_sequence(msg)))
