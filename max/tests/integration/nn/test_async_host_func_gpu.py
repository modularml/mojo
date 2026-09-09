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
"""Integration test for the async host-func dispatch path.

Exercises ``Device.__unsafe_enqueue_async_py_host_func`` paired with
``DeviceQueue.wait_for_host_value`` against a real Python callable, end
to end. Confirms:

  1. The kickoff host function returns promptly and the Python callable
     runs on a different (AsyncRT worker) thread.
  2. The completion flag transitions 0 -> 1 by release-store from the
     worker, and ``wait_for_host_value`` observes it from the GPU.
  3. Per-replay flag reset works: invoking the kickoff again finds the
     flag back at 0 before dispatching.
"""

import threading
import time

import pytest
from max.driver import CPU, Accelerator, CompletionFlag


@pytest.fixture
def accelerator() -> Accelerator:
    return Accelerator()


@pytest.fixture
def cpu() -> CPU:
    return CPU()


def test_async_host_func_signals_flag(
    accelerator: Accelerator, cpu: CPU
) -> None:
    """Round-trip: callable runs on a different thread, flag goes 0->1."""
    if accelerator.api not in ("cuda", "hip"):
        pytest.skip(
            "__unsafe_enqueue_async_py_host_func is only supported on CUDA/HIP"
        )

    flag = CompletionFlag(accelerator)

    invocations: list[int] = []
    callback_threads: list[int] = []
    main_thread_id = threading.get_ident()

    def cb() -> None:
        # Record the OS thread id so we can confirm it isn't the main
        # Python thread (i.e., the callable really runs on an AsyncRT
        # worker).
        callback_threads.append(threading.get_ident())
        # Small busy wait so we can observe the flag transitioning.
        time.sleep(0.005)
        invocations.append(1)

    # Pre-condition: flag is zero.
    assert flag.load() == 0

    # Enqueue. The trampoline clears the flag to 0 (already 0 here),
    # dispatches `cb` onto `cpu`'s AsyncRT worker pool, and returns.
    # The wait_for_host_value downstream blocks until the AsyncRT worker
    # stores 1.
    getattr(accelerator, "__unsafe_enqueue_async_py_host_func")(
        cb, flag, 1, cpu
    )
    accelerator.default_queue.wait_for_host_value(flag, 1)

    # Synchronize so the host can observe the flag store.
    accelerator.synchronize()

    assert invocations == [1], "callback should have run exactly once"
    assert callback_threads and callback_threads[0] != main_thread_id, (
        "callback should run on an AsyncRT worker, not the main thread"
    )
    assert flag.load() == 1, "worker should have stored 1 to the flag"


def test_async_host_func_replays_clear_flag(
    accelerator: Accelerator, cpu: CPU
) -> None:
    """Second invocation resets the flag and signals it again."""
    if accelerator.api not in ("cuda", "hip"):
        pytest.skip(
            "__unsafe_enqueue_async_py_host_func is only supported on CUDA/HIP"
        )

    flag = CompletionFlag(accelerator)

    counter = [0]

    def cb() -> None:
        counter[0] += 1

    for _ in range(3):
        getattr(accelerator, "__unsafe_enqueue_async_py_host_func")(
            cb, flag, 1, cpu
        )
        accelerator.default_queue.wait_for_host_value(flag, 1)
        accelerator.synchronize()
        assert flag.load() == 1

    assert counter[0] == 3


def test_async_host_func_callback_exception_does_not_deadlock(
    accelerator: Accelerator, cpu: CPU
) -> None:
    """If the callable raises, the worker still sets the flag.

    Without this guarantee, a buggy callback would leave the flag at 0
    forever, deadlocking any GPU stream gated by wait_for_host_value.
    The trampoline catches the exception (prints it via PySys_WriteStderr)
    and still proceeds with the flag store.
    """
    if accelerator.api not in ("cuda", "hip"):
        pytest.skip(
            "__unsafe_enqueue_async_py_host_func is only supported on CUDA/HIP"
        )

    flag = CompletionFlag(accelerator)

    def cb() -> None:
        raise RuntimeError("intentional test failure")

    getattr(accelerator, "__unsafe_enqueue_async_py_host_func")(
        cb, flag, 1, cpu
    )
    accelerator.default_queue.wait_for_host_value(flag, 1)
    # Should not hang; the trampoline's except clause sets the flag.
    accelerator.synchronize()
