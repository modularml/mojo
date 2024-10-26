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
from sys._build import is_debug_build
from sys.param_env import env_get_string
from sys.ffi import external_call, c_uint, c_size_t, c_char
from sys.info import sizeof
from memory import UnsafePointer
from utils.string_slice import _ConcatStr


from builtin._location import __call_location, _SourceLocation

alias defined_mode = env_get_string["ASSERT", "safe"]()


@no_inline
fn _assert_enabled[assert_mode: StringLiteral, cpu_only: Bool]() -> Bool:
    constrained[
        defined_mode in ["none", "warn", "safe", "all"],
        "-D ASSERT="
        + defined_mode
        + " but must be one of: none, warn, safe, all",
    ]()
    constrained[
        assert_mode in ["none", "safe"],
        "assert_mode=" + assert_mode + " but must be one of: none, safe",
    ]()

    @parameter
    if defined_mode == "none" or (triple_is_nvidia_cuda() and cpu_only):
        return False
    elif defined_mode == "all" or defined_mode == "warn" or is_debug_build():
        return True
    else:
        return defined_mode == assert_mode


@always_inline
fn debug_assert[
    assert_mode: StringLiteral = "none", cpu_only: Bool = False
](cond: Bool, message: StringLiteral):
    """Asserts that the condition is true.

    Parameters:
        assert_mode: Determines when the assert is turned on.
        cpu_only: If true, only run the assert on CPU.

    Args:
        cond: The function to invoke to check if the assertion holds.
        message: A StringLiteral to print on failure.


    Pass in a condition and single StringLiteral to stop execution and
    show an error on failure:

    ```mojo
    x = 0
    debug_assert(x > 0, "x is not more than 0")
    ```

    By default it's a no-op, you can change the assertion level for example:

    ```sh
    mojo -D ASSERT=all main.mojo
    ```

    Assertion modes:

    - none: turn off all assertions for performance at the cost of safety.
    - warn: print any errors instead of aborting.
    - safe: standard safety checks that are on even in release builds.
    - all: turn on all assertions.

    You can set the `assert_mode` to `safe` so the assertion runs even in
    release builds:

    ```mojo
    debug_assert[assert_mode="safe"](
        x > 0, "expected x to be more than 0 but got: ", x
    )
    ```

    To ensure that you have no runtime penality from your assertion in release
    builds, make sure there are no side effects in your message and condition.

    On GPU this will also show the the block and thread idx's on failure.
    .
    """

    @parameter
    if _assert_enabled[assert_mode, cpu_only]():
        if cond:
            return
        _debug_assert_msg_literal(message, __call_location())


@always_inline
fn debug_assert[
    cond: fn () capturing [_] -> Bool,
    assert_mode: StringLiteral = "none",
    cpu_only: Bool = False,
    *Ts: Writable,
](*messages: *Ts):
    """Asserts that the condition is true.

    Parameters:
        cond: The function to invoke to check if the assertion holds.
        assert_mode: Determines when the assert is turned on.
        cpu_only: If true, only run the assert on CPU.
        Ts: The element types conforming to `Writable` for the message.

    Args:
        messages: Arguments to convert to a `String` message.


    You can pass in multiple args that are `Writable` to generate a formatted
    message, by default this will be a no-op:

    ```mojo
    x = 0
    debug_assert(x > 0, "expected x to be more than 0 but got: ", x)
    ```

    You can change the assertion level for example:

    ```sh
    mojo -D ASSERT=all main.mojo
    ```

    Assertion modes:

    - none: turn off all assertions for performance at the cost of safety.
    - warn: print any errors instead of aborting.
    - safe: standard safety checks that are on even in release builds.
    - all: turn on all assertions.

    You can set the `assert_mode` to `safe` so the assertion runs even in
    release builds:

    ```mojo
    debug_assert[assert_mode="safe"](
        x > 0, "expected x to be more than 0 but got: ", x
    )
    ```

    To ensure that you have no runtime penality from your assertion in release
    builds, make sure there are no side effects in your message and condition.
    Take this example:

    ```mojo
    person = "name: john, age: 50"
    name = "john"
    debug_assert(str("name: ") + name == person, "unexpected name")
    ```

    This will have a runtime penality due to allocating a `String` in the
    condition even in release builds, you must put the condition inside a
    closure so it only runs when the assertion is turned on:

    ```mojo
    fn check_name() capturing -> Bool:
        return str("name: ") + name == person

    debug_assert[check_name]("unexpected name")
    ```

    If you need to allocate, and so don't want the assert to ever run on GPU,
    you can set it to CPU only:

    ```mojo
    debug_assert[check_name, cpu_only=True]("unexpected name")
    ```
    .
    """

    @parameter
    if _assert_enabled[assert_mode, cpu_only]():
        if cond():
            return
        _debug_assert_msg(messages, __call_location())


@always_inline
fn debug_assert[
    assert_mode: StringLiteral = "none",
    cpu_only: Bool = False,
    *Ts: Writable,
](cond: Bool, *messages: *Ts):
    """Asserts that the condition is true.

    Parameters:
        assert_mode: Determines when the assert is turned on.
        cpu_only: If true, only run the assert on CPU.
        Ts: The element types conforming to `Writable` for the message.

    Args:
        cond: The bool value to assert.
        messages: Arguments to convert to a `String` message.

    You can pass in multiple args that are `Writable` to generate a formatted
    message, by default this will be a no-op:

    ```mojo
    x = 0
    debug_assert(x > 0, "expected x to be more than 0 but got: ", x)
    ```

    You can change the assertion level for example:

    ```sh
    mojo -D ASSERT=all main.mojo
    ```

    Assertion modes:

    - none: turn off all assertions for performance at the cost of safety.
    - warn: print any errors instead of aborting.
    - safe: standard safety checks that are on even in release builds.
    - all: turn on all assertions.

    You can set the `assert_mode` to `safe` so the assertion runs even in
    release builds:

    ```mojo
    debug_assert[assert_mode="safe"](
        x > 0, "expected x to be more than 0 but got: ", x
    )
    ```

    To ensure that you have no runtime penality from your assertion in release
    builds, make sure there are no side effects in your message and condition.
    Take this example:

    ```mojo
    person = "name: john, age: 50"
    name = "john"
    debug_assert(str("name: ") + name == person, "unexpected name")
    ```

    This will have a runtime penality due to allocating a `String` in the
    condition even in release builds, you must put the condition inside a
    closure so it only runs when the assertion is turned on:

    ```mojo
    fn check_name() capturing -> Bool:
        return str("name: ") + name == person

    debug_assert[check_name]("unexpected name")
    ```

    If you need to allocate, and so don't want the assert to ever run on GPU,
    you can set it to CPU only:

    ```mojo
    debug_assert[check_name, cpu_only=True]("unexpected name")
    ```
    .
    """

    @parameter
    if _assert_enabled[assert_mode, cpu_only]():
        if cond:
            return
        _debug_assert_msg(messages, __call_location())


@no_inline
fn _debug_assert_msg(
    messages: VariadicPack[_, Writable, *_], loc: _SourceLocation
):
    """Aborts with (or prints) the given message and location.

    This function is intentionally marked as no_inline to reduce binary size.

    Note that it's important that this function doesn't get inlined; otherwise,
    an indirect recursion of @always_inline functions is possible (e.g. because
    abort's implementation could use debug_assert)
    """

    @parameter
    if triple_is_nvidia_cuda():
        external_call["__assertfail", NoneType](
            "debug_assert message must be a single StringLiteral on GPU"
            .unsafe_cstr_ptr(),
            loc.file_name.unsafe_cstr_ptr(),
            c_uint(loc.line),
            # TODO(MSTDL-962) pass through the funciton name here
            "kernel".unsafe_cstr_ptr(),
            c_size_t(sizeof[Int8]()),
        )

    else:
        message = _ConcatStr(capacity=5)

        @parameter
        if defined_mode == "warn":
            message.append("Assert Warning: ", String.write(messages))
            print(loc.prefix(message^))
        else:
            message.append("Assert Error: ", String.write(messages))
            abort(loc.prefix(message^))


@no_inline
fn _debug_assert_msg_literal(message: StringLiteral, loc: _SourceLocation):
    """Aborts with (or prints) the given message and location.

    This function is intentionally marked as no_inline to reduce binary size.

    Note that it's important that this function doesn't get inlined; otherwise,
    an indirect recursion of @always_inline functions is possible (e.g. because
    abort's implementation could use debug_assert)
    """

    @parameter
    if triple_is_nvidia_cuda():
        external_call["__assertfail", NoneType](
            message.unsafe_cstr_ptr(),
            loc.file_name.unsafe_cstr_ptr(),
            c_uint(loc.line),
            # TODO(MSTDL-962) pass through the funciton name here
            "kernel".unsafe_cstr_ptr(),
            c_size_t(sizeof[Int8]()),
        )
    else:

        @parameter
        if defined_mode == "warn":
            print(loc.prefix(_ConcatStr("Assert Warning: ", message)))
        else:
            abort(loc.prefix(_ConcatStr("Assert Error: ", message)))
