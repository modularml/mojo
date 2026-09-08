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
"""Tests for raising coroutine support in AsyncRT.

These tests explore incrementally more complex scenarios for using
async def ... raises with the async runtime.
"""

from std.memory import alloc
from std.runtime._asyncrt import create_task, create_raising_task
from std.testing import assert_equal, assert_true


# ===-----------------------------------------------------------------------===#
# Raising async functions used by the tests
# ===-----------------------------------------------------------------------===#


async def add_async(a: Int, b: Int) raises -> Int:
    """A raising async function that succeeds."""
    return a + b


async def failing_async() raises -> Int:
    """A raising async function that always raises."""
    raise Error("intentional error from async task")


async def conditional_raise(should_fail: Bool) raises -> Int:
    """A raising async function that conditionally raises."""
    if should_fail:
        raise Error("conditional failure")
    return 42


# ===-----------------------------------------------------------------------===#
# Phase 1: Use non-raising wrappers to catch errors from raising
# async functions. This tests that try/except works within async functions
# when awaiting RaisingCoroutines.
# ===-----------------------------------------------------------------------===#


def test_raising_async_success_via_wrapper() raises:
    """Test that a raising async function can succeed through a wrapper."""

    async def wrapper() -> Int:
        try:
            return await add_async(10, 20)
        except:
            return -1

    var task = create_task(wrapper())
    assert_equal(task.wait(), 30)


def test_raising_async_failure_via_wrapper() raises:
    """Test that a raising async function's error is caught in the wrapper."""

    async def wrapper() -> Int:
        try:
            return await failing_async()
        except:
            return -1

    var task = create_task(wrapper())
    assert_equal(task.wait(), -1)


# TODO:  MOCO-3408: Using String(e) inside an async try/except triggers an MLIR
# "operand #0 does not dominate this use" error at string.mojo:49 when
# compiled with debug info. Repro: `mojo -g <this file>`
# The call in main() is commented out to avoid the crash.
def test_raising_async_error_message_via_wrapper() raises:
    """Test that the error message is preserved when caught in a wrapper."""

    async def wrapper() -> String:
        try:
            _ = await failing_async()
            return String("no error")
        except e:
            return String(e)

    var task = create_task(wrapper())
    assert_equal(task.wait(), "intentional error from async task")


def test_raising_async_conditional_via_wrapper() raises:
    """Test conditional raising with both success and failure paths."""

    async def success_wrapper() -> Int:
        try:
            return await conditional_raise(should_fail=False)
        except:
            return -1

    async def failure_wrapper() -> Int:
        try:
            return await conditional_raise(should_fail=True)
        except:
            return -1

    var t_ok = create_task(success_wrapper())
    var t_fail = create_task(failure_wrapper())
    assert_equal(t_ok.wait(), 42)
    assert_equal(t_fail.wait(), -1)


# ===-----------------------------------------------------------------------===#
# Phase 2: Direct raising task support via create_raising_task.
# RaisingTask.wait() takes `deinit self`, so we transfer with `^`.
# ===-----------------------------------------------------------------------===#


def test_raising_task_success() raises:
    """Test that a RaisingTask works when the coroutine succeeds."""
    var task = create_raising_task(add_async(10, 20))
    assert_equal(task^.wait(), 30)


def test_raising_task_error() raises:
    """Test that a RaisingTask propagates errors correctly."""
    var task = create_raising_task(failing_async())
    var caught = False
    try:
        _ = task^.wait()
    except e:
        caught = True
        assert_equal(String(e), "intentional error from async task")
    assert_true(caught, "expected error was not raised")


def test_raising_task_conditional() raises:
    """Test RaisingTask with conditional success/failure."""
    var t_ok = create_raising_task(conditional_raise(should_fail=False))
    assert_equal(t_ok^.wait(), 42)

    var t_fail = create_raising_task(conditional_raise(should_fail=True))
    var caught = False
    try:
        _ = t_fail^.wait()
    except e:
        caught = True
        assert_equal(String(e), "conditional failure")
    assert_true(caught, "expected error was not raised")


def test_raising_task_multiple_success() raises:
    """Test multiple successful raising tasks."""
    var t1 = create_raising_task(add_async(1, 2))
    var r1 = t1^.wait()
    var t2 = create_raising_task(add_async(3, 4))
    var r2 = t2^.wait()
    assert_equal(r1, 3)
    assert_equal(r2, 7)


# ===-----------------------------------------------------------------------===#
# Phase 3: Await RaisingTask inside async functions via __await__.
# This uses non-raising wrappers so we can collect the results with
# regular Tasks.
# ===-----------------------------------------------------------------------===#


def test_raising_task_await_success() raises:
    """Test that RaisingTask can be awaited in an async function."""

    async def wrapper() -> Int:
        var task = create_raising_task(add_async(10, 20))
        try:
            return await task^
        except:
            return -1

    var task = create_task(wrapper())
    assert_equal(task.wait(), 30)


def test_raising_task_await_error() raises:
    """Test that RaisingTask await propagates errors to the caller."""

    async def wrapper() -> Int:
        var task = create_raising_task(failing_async())
        try:
            return await task^
        except:
            return -1

    var task = create_task(wrapper())
    assert_equal(task.wait(), -1)


def test_raising_task_await_conditional() raises:
    """Test RaisingTask await with conditional success/failure paths."""

    async def success_wrapper() -> Int:
        var task = create_raising_task(conditional_raise(should_fail=False))
        try:
            return await task^
        except:
            return -1

    async def failure_wrapper() -> Int:
        var task = create_raising_task(conditional_raise(should_fail=True))
        try:
            return await task^
        except:
            return -1

    var t_ok = create_task(success_wrapper())
    var t_fail = create_task(failure_wrapper())
    assert_equal(t_ok.wait(), 42)
    assert_equal(t_fail.wait(), -1)


def test_raising_task_await_chained() raises:
    """Test awaiting a RaisingTask from within a raising async function."""

    async def outer() raises -> Int:
        var inner = create_raising_task(add_async(5, 10))
        return await inner^

    var task = create_raising_task(outer())
    assert_equal(task^.wait(), 15)


def main() raises:
    # Phase 1: wrapper pattern
    test_raising_async_success_via_wrapper()
    test_raising_async_failure_via_wrapper()
    # TODO: MOCO-3408
    # test_raising_async_error_message_via_wrapper()  # see TODO above
    test_raising_async_conditional_via_wrapper()

    # Phase 2: direct raising task support
    test_raising_task_success()
    test_raising_task_error()
    test_raising_task_conditional()
    test_raising_task_multiple_success()

    # Phase 3: await RaisingTask inside async functions
    test_raising_task_await_success()
    test_raising_task_await_error()
    test_raising_task_await_conditional()
    test_raising_task_await_chained()

    print("All raising asyncrt tests passed!")
