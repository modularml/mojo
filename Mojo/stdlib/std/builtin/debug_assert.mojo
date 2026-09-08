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
"""Implements run-time assertions.

These are Mojo built-ins, so you don't need to import them.
"""

from std._plugin import CurrentPlugin
from std.format._utils import _FixedWriteBuffer
from std.io.io import _printf
from std.os import abort
from std.sys import (
    is_amd_gpu,
    is_apple_gpu,
    is_gpu,
    is_nvidia_gpu,
)
from std.sys._amdgpu import (
    printf_append_args,
    printf_append_string_n,
    printf_begin,
)
from std.sys._build import is_debug_build
from std.sys.intrinsics import assume
from std.sys.defines import get_defined_string
from std.collections.string.string_span import (
    _get_kgen_string,
    get_static_string,
)
from std.reflection import call_location, SourceLocation

comptime ASSERT_MODE = get_defined_string["ASSERT", "safe"]()
"""The compile-time assertion mode from the ASSERT environment variable."""

comptime _NO_MESSAGE = "assertion failed"
"""Must remain a string literal: the no-message path passes `byte_length() + 1`
and relies on the trailing nul in static memory."""


@always_inline("nodebug")
def _string_free_comptime_assert[
    cond: Bool, msg: StaticString, *extra: StaticString
]():
    """Compile-time assertion that avoids `String` to prevent circular deps.

    This exists because `debug_assert` cannot use `comptime assert` with
    `String`-based messages — `String` transitively depends on `debug_assert`,
    which would create infinite recursion in the parameter domain.
    """

    __mlir_op.`kgen.param.assert`[
        cond=cond.__mlir_bool__(),
        message=_get_kgen_string[msg, *extra](),
    ]()


@no_inline
def _assert_enabled[assert_mode: StaticString, cpu_only: Bool]() -> Bool:
    _string_free_comptime_assert[
        ASSERT_MODE == "none"
        or ASSERT_MODE == "warn"
        or ASSERT_MODE == "safe"
        or ASSERT_MODE == "all",
        "-D ASSERT=",
        ASSERT_MODE,
        " but must be one of: none, warn, safe, all",
    ]()
    _string_free_comptime_assert[
        assert_mode == "none" or assert_mode == "safe",
        "assert_mode=",
        assert_mode,
        " but must be one of: none, safe",
    ]()

    # FIXME: Enable assertions on Apple GPU after MOCO-2405 is fixed
    comptime if ASSERT_MODE == "none" or (
        is_gpu() and cpu_only
    ) or is_apple_gpu():
        return False
    elif ASSERT_MODE == "all" or ASSERT_MODE == "warn" or is_debug_build():
        return True
    else:
        return ASSERT_MODE == assert_mode


@no_inline
def _debug_assert_fail_format[
    *Ts: Writable
](location: SourceLocation, *messages: *Ts):
    var message = _FixedWriteBuffer()

    comptime for i in range(messages.__len__()):
        messages[i].write_to(message)

    var cstr = message.nul_terminate()
    var bytes_with_nul = cstr.as_bytes_with_nul()
    _debug_assert_msg(
        bytes_with_nul.unsafe_ptr(), len(bytes_with_nul), location
    )


@always_inline
def _debug_assert_fail[*Ts: Writable](*messages: *Ts, location: SourceLocation):
    """Reports a failed assertion, formatting the message if there is one."""

    comptime if messages.__len__() == 0:
        _debug_assert_msg(
            _NO_MESSAGE.ptr(), _NO_MESSAGE.byte_length() + 1, location
        )
    else:
        _debug_assert_fail_format(location, *messages)


@always_inline
def debug_assert[
    Cond: def() -> Bool,
    assert_mode: StaticString = "none",
    *Ts: Writable,
    cpu_only: Bool = False,
](cond: Cond, *messages: *Ts, location: Optional[SourceLocation] = None):
    """Asserts that the condition is true at run time.

    If the condition is false, the assertion displays the given message and
    causes the program to exit.

    You can pass in multiple arguments to generate a formatted
    message. No string allocation occurs unless the assertion is triggered.

    ```mojo
    var x = 0
    debug_assert(x > 0, "expected x to be more than 0 but got: ", x)
    ```

    Normal assertions are off by default—they only run when the program is
    compiled with all assertions enabled. You can set the `assert_mode` to
    `safe` to create an assertion that's on by default:

    ```mojo
    var x = 0
    debug_assert[assert_mode="safe"](
        x > 0, "expected x to be more than 0 but got: ", x
    )
    ```

    Use the `ASSERT` variable to turn assertions on or off when building or
    running a Mojo program:

    ```sh
    mojo -D ASSERT=all main.mojo
    ```

    The `ASSERT` variable takes the following values:

    - all: Turn on all assertions.
    - safe: Turn on "safe" assertions only. This is the default.
    - none: Turn off all assertions, for performance at the cost of safety.
    - warn: Turn on all assertions, but print any errors instead of exiting.

    To ensure that you have no run-time penalty from your assertions even when
    they're disabled, make sure there are no side effects in your message and
    condition expressions. For example:

    ```mojo
    var person = "name: john, age: 50"
    var name = "john"
    debug_assert(String("name: ", name) in person, "unexpected name")
    ```

    This will have a run-time penalty due to allocating a `String` in the
    condition expression, even when assertions are disabled. To avoid this, pass
    the condition as a closure so it runs only when the assertion is turned on:

    ```mojo
    def main():
        var person = "name: john, age: 50"
        var name = "john"

        def check_name() {name, person} -> Bool:
            return String("name: ", name) in person

        debug_assert(check_name, "unexpected name")
    ```

    If you need to allocate, and so don't want the assert to ever run on GPU,
    you can set it to CPU only:

    ```mojo
    def main():
        var person = "name: john, age: 50"
        var name = "john"

        def check_name() {name, person} -> Bool:
            return String("name: ", name) in person

        debug_assert[cpu_only=True](check_name, "unexpected name")
    ```

    For compile-time assertions, see [`comptime
    assert`](/docs/manual/metaprogramming/constraints/#compile-time-assertions)

    Parameters:
        Cond: The type of the closure to invoke to check if the assertion holds.
        assert_mode: Determines when the assert is turned on.
            - default ("none"): Turned on when compiled with `-D ASSERT=all`.
            - "safe": Turned on by default.
        Ts: The element types for the message arguments.
        cpu_only: If true, only run the assert on CPU.

    Args:
        cond: The closure to invoke to check if the assertion holds.
        messages: A set of [`Writable`](/docs/std/format/Writable/)
            arguments to convert to a `String` message.
        location: Source location to report on assertion failure.
    """

    comptime if _assert_enabled[assert_mode, cpu_only]():
        if cond():
            return

        # `debug_assert` is used a lot, so its body must generate short, concise
        # code. Avoid standard library functions that contain error/abort code
        # inside; those cause code bloat at every call site.
        #
        # `call_location()` must be resolved here rather than in the callee, so
        # that the reported location stays at this assert's own caller.
        _debug_assert_fail(
            *messages, location=location.or_else(call_location())
        )


@always_inline
def debug_assert[
    assert_mode: StaticString = "none",
    *Ts: Writable,
    cpu_only: Bool = False,
    _use_compiler_assume: Bool = False,
](cond: Bool, *messages: *Ts, location: Optional[SourceLocation] = None):
    """Asserts that the condition is true at run time.

    If the condition is false, the assertion displays the given message and
    causes the program to exit.

    You can pass in multiple arguments to generate a formatted
    message. No string allocation occurs unless the assertion is triggered.

    ```mojo
    var x = 0
    debug_assert(x > 0, "expected x to be more than 0 but got: ", x)
    ```

    Normal assertions are off by default—they only run when the program is
    compiled with all assertions enabled. You can set the `assert_mode` to
    `safe` to create an assertion that's on by default:

    ```mojo
    var x = 0
    debug_assert[assert_mode="safe"](
        x > 0, "expected x to be more than 0 but got: ", x
    )
    ```

    Use the `ASSERT` variable to turn assertions on or off when building or
    running a Mojo program:

    ```sh
    mojo -D ASSERT=all main.mojo
    ```

    The `ASSERT` variable takes the following values:

    - all: Turn on all assertions.
    - safe: Turn on "safe" assertions only. This is the default.
    - none: Turn off all assertions, for performance at the cost of safety.
    - warn: Turn on all assertions, but print any errors instead of exiting.

    To ensure that you have no run-time penalty from your assertions even when
    they're disabled, make sure there are no side effects in your message and
    condition expressions. For example:

    ```mojo
    var person = "name: john, age: 50"
    var name = "john"
    debug_assert(String("name: ", name) in person, "unexpected name")
    ```

    This will have a run-time penalty due to allocating a `String` in the
    condition expression, even when assertions are disabled. To avoid this, pass
    the condition as a closure so it runs only when the assertion is turned on:

    ```mojo
    def main():
        var person = "name: john, age: 50"
        var name = "john"

        def check_name() {name, person} -> Bool:
            return String("name: ", name) in person

        debug_assert(check_name, "unexpected name")
    ```

    If you need to allocate, and so don't want the assert to ever run on GPU,
    you can set it to CPU only:

    ```mojo
    def main():
        var person = "name: john, age: 50"
        var name = "john"

        def check_name() {name, person} -> Bool:
            return String("name: ", name) in person

        debug_assert[cpu_only=True](check_name, "unexpected name")
    ```

    For compile-time assertions, see [`comptime
    assert`](/docs/manual/metaprogramming/constraints/#compile-time-assertions)

    Parameters:
        assert_mode: Determines when the assert is turned on.
            - default ("none"): Turned on when compiled with `-D ASSERT=all`.
            - "safe": Turned on by default.
        Ts: The element types for the message arguments.
        cpu_only: If true, only run the assert on CPU.
        _use_compiler_assume: If true, assume the condition is true for repeated checks,
            to help the compiler optimize (default False).


    Args:
        cond: The bool value to assert.
        messages: A set of [`Writable`](/docs/std/format/Writable/)
            arguments to convert to a `String` message.
        location: Source location to report on assertion failure.
    """

    comptime if _assert_enabled[assert_mode, cpu_only]():
        if cond:
            return

        # See the sibling overload: avoid stdlib functions carrying error/abort
        # code, and resolve `call_location()` here rather than in the callee.
        _debug_assert_fail(
            *messages, location=location.or_else(call_location())
        )

    elif _use_compiler_assume:
        assume(cond)


# The two no-message overloads below omit `location` on purpose: they are
# optimized for fast compile time, because they are used frequently. An
# `Optional[SourceLocation]` and an `or_else` call generate extra IR and thus
# slow compilation down.
@always_inline
def debug_assert[
    Cond: def() -> Bool,
    assert_mode: StaticString = "none",
    cpu_only: Bool = False,
](cond: Cond):
    """Asserts that the condition is true at run time, with no message.

    On failure this reports `assertion failed` at the call site. Use the
    overload taking `messages` to report a formatted message instead.

    ```mojo
    def main():
        var x = 1

        def check() {x} -> Bool:
            return x > 0

        debug_assert(check)
    ```

    Parameters:
        Cond: The type of the closure to invoke to check if the assertion holds.
        assert_mode: Determines when the assert is turned on.
        cpu_only: If true, only run the assert on CPU.

    Args:
        cond: The closure to invoke to check if the assertion holds.
    """

    comptime if _assert_enabled[assert_mode, cpu_only]():
        if cond():
            return

        _debug_assert_fail(location=call_location())


@always_inline
def debug_assert[
    assert_mode: StaticString = "none",
    cpu_only: Bool = False,
    _use_compiler_assume: Bool = False,
](cond: Bool):
    """Asserts that the condition is true at run time, with no message.

    On failure this reports `assertion failed` at the call site. Use the
    overload taking `messages` to report a formatted message instead.

    ```mojo
    var x = 1
    debug_assert(x > 0)
    ```

    Parameters:
        assert_mode: Determines when the assert is turned on.
        cpu_only: If true, only run the assert on CPU.
        _use_compiler_assume: If true, assume the condition is true for repeated
            checks, to help the compiler optimize (default False).

    Args:
        cond: The bool value to assert.
    """

    comptime if _assert_enabled[assert_mode, cpu_only]():
        if cond:
            return

        _debug_assert_fail(location=call_location())
    elif _use_compiler_assume:
        assume(cond)


@always_inline
def debug_assert[
    assert_mode: StaticString = "none",
    cpu_only: Bool = False,
    _use_compiler_assume: Bool = False,
](cond: Bool, message: StringLiteral):
    """Asserts that the condition is true at run time.

    If the condition is false, the assertion displays the given message and
    causes the program to exit.

    You can pass in multiple arguments to generate a formatted
    message. No string allocation occurs unless the assertion is triggered.

    ```mojo
    var x = 0
    debug_assert(x > 0, "expected x to be more than 0 but got: ", x)
    ```

    Normal assertions are off by default—they only run when the program is
    compiled with all assertions enabled. You can set the `assert_mode` to
    `safe` to create an assertion that's on by default:

    ```mojo
    var x = 0
    debug_assert[assert_mode="safe"](
        x > 0, "expected x to be more than 0 but got: ", x
    )
    ```

    Use the `ASSERT` variable to turn assertions on or off when building or
    running a Mojo program:

    ```sh
    mojo -D ASSERT=all main.mojo
    ```

    The `ASSERT` variable takes the following values:

    - all: Turn on all assertions.
    - safe: Turn on "safe" assertions only. This is the default.
    - none: Turn off all assertions, for performance at the cost of safety.
    - warn: Turn on all assertions, but print any errors instead of exiting.

    To ensure that you have no run-time penalty from your assertions even when
    they're disabled, make sure there are no side effects in your message and
    condition expressions. For example:

    ```mojo
    var person = "name: john, age: 50"
    var name = "john"
    debug_assert(String("name: ", name) in person, "unexpected name")
    ```

    This will have a run-time penalty due to allocating a `String` in the
    condition expression, even when assertions are disabled. To avoid this, pass
    the condition as a closure so it runs only when the assertion is turned on:

    ```mojo
    def main():
        var person = "name: john, age: 50"
        var name = "john"

        def check_name() {name, person} -> Bool:
            return String("name: ", name) in person

        debug_assert(check_name, "unexpected name")
    ```

    If you need to allocate, and so don't want the assert to ever run on GPU,
    you can set it to CPU only:

    ```mojo
    def main():
        var person = "name: john, age: 50"
        var name = "john"

        def check_name() {name, person} -> Bool:
            return String("name: ", name) in person

        debug_assert[cpu_only=True](check_name, "unexpected name")
    ```

    For compile-time assertions, see [`comptime
    assert`](/docs/manual/metaprogramming/constraints/#compile-time-assertions)

    Parameters:
        assert_mode: Determines when the assert is turned on.
            - default ("none"): Turned on when compiled with `-D ASSERT=all`.
            - "safe": Turned on by default.
        cpu_only: If true, only run the assert on CPU.
        _use_compiler_assume: If true, assume the condition is true for repeated checks,
            to help the compiler optimize (default False).

    Args:
        cond: The bool value to assert.
        message: A static string message.
    """

    comptime if _assert_enabled[assert_mode, cpu_only]():
        if cond:
            return

        _debug_assert_msg(
            message.ptr(),
            # include StringLiteral null terminator for printf statements
            message.byte_length() + 1,
            call_location(),
        )
    elif _use_compiler_assume:
        assume(cond)


@no_inline
def _debug_assert_msg(
    message: ImmPointer[Byte, _], length: Int, loc: SourceLocation
):
    """Aborts with (or prints) the given message and location.

    This function is intentionally marked as no_inline to reduce binary size.

    Note that it's important that this function doesn't get inlined; otherwise,
    an indirect recursion of @always_inline functions is possible (e.g. because
    abort's implementation could use debug_assert)
    """

    # The `else` branch is required (not just for clarity): its `_printf`
    # paths recurse via `Optional.value()` → `debug_assert`. `comptime if
    # X: return` does not elide unconditional post-return code, so the
    # fallback must live in `else:` to be comptime-elided when a plugin
    # owns emission.
    comptime if CurrentPlugin._handles_debug_assert:
        CurrentPlugin.debug_assert_emit_fn(message, length, loc)
        comptime if ASSERT_MODE != "warn":
            abort()
        return
    else:
        if __is_run_in_comptime_interpreter:
            print("At: ", loc, ": Assert Error: ", message, sep="")

            comptime if ASSERT_MODE != "warn":
                abort()
            return

        comptime fmt = (
            "At: %s:%llu:%llu: block: [%llu,%llu,%llu] thread: [%llu,%llu,%llu]"
            " Assert Error: %s\n"
        )

        comptime if is_nvidia_gpu():
            from std._gpu.primitives.id import block_idx, thread_idx

            _printf[fmt](
                loc.file_name().as_c_string_slice(),
                loc.line(),
                loc.column(),
                UInt(block_idx.x),
                UInt(block_idx.y),
                UInt(block_idx.z),
                UInt(thread_idx.x),
                UInt(thread_idx.y),
                UInt(thread_idx.z),
                message,
            )
        # TODO(MSTDL-1783): fix `_printf` not working on AMDGPU with %s args
        elif is_amd_gpu():
            from std._gpu.primitives.id import block_idx, thread_idx

            var fd = printf_begin()
            # Each appended string must carry its own nul terminator so the AMD
            # fprintf service can delimit it; `as_bytes()` omits the nul, which
            # corrupts output when a string's length is a multiple of 8.
            # `get_static_string` guarantees a trailing nul in static memory
            # just past the returned range.
            var fmt_str = get_static_string[fmt]()
            _ = printf_append_string_n(
                fd,
                Span(
                    unsafe_ptr=fmt_str.as_bytes().unsafe_ptr(),
                    length=fmt_str.byte_length() + 1,
                ),
                False,
            )
            # Runtime %s types must be passed as separate append_string calls.
            # `file_name()` is a string literal, so its trailing nul lives in
            # static memory just past the range (same guarantee the NVIDIA `%s`
            # path relies on).
            var file_name = loc.file_name()
            _ = printf_append_string_n(
                fd,
                Span(
                    unsafe_ptr=file_name.as_bytes().unsafe_ptr(),
                    length=file_name.byte_length() + 1,
                ),
                False,
            )
            # Can only pass 7 args at a time
            _ = printf_append_args(
                fd,
                7,
                UInt64(loc.line()),
                UInt64(loc.column()),
                UInt64(block_idx.x),
                UInt64(block_idx.y),
                UInt64(block_idx.z),
                UInt64(thread_idx.x),
                UInt64(thread_idx.y),
                0,
            )
            # Pass last arg
            _ = printf_append_args(
                fd, 1, UInt64(thread_idx.z), 0, 0, 0, 0, 0, 0, 0
            )
            # Append message and finalize
            _ = printf_append_string_n(
                fd, Span(unsafe_ptr=message, length=length), True
            )
        else:
            _printf["At: %s:%llu:%llu: Assert Error: %s\n"](
                loc.file_name().as_c_string_slice(),
                loc.line(),
                loc.column(),
                message,
            )

        comptime if ASSERT_MODE != "warn":
            abort()
