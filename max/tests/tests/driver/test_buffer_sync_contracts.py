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
"""Conformance tests pinning driver-buffer synchronization contracts (DRIV-311).

MAX relies on two host-read APIs synchronizing pending device work before they
return, but nothing asserted it: ``__dlpack__``/``to_numpy`` on a non-pinned
device buffer (``Buffer.cpp`` ``dlPack``) and ``item()`` (``Buffer.cpp``, whose
reads bypass the stream chain so an explicit sync is inserted). Both funnel
through ``DeviceBuffer::synchronize`` -> ``DeviceContext::synchronize``.

Each test gates the device's default stream with a host-signalled
``CompletionFlag``: ``wait_for_host_value`` stalls the stream until the host
signals, so any work enqueued afterwards on that stream cannot run until
release. A read that synchronizes therefore cannot complete while the gate is
held -- the deterministic signal that distinguishes "synchronized" from "raced
ahead", with no reliance on copy timing. Every gate is released in a ``finally``
and every worker thread is joined with a timeout so a contract regression fails
the assertion instead of hanging CI.

These are the conformance tests both the AsyncRT DeviceContext and the neo
Driver must keep passing before any DRIV-211 semantic change flips behavior.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np
import pytest
from max.driver import (
    Accelerator,
    Buffer,
    CompletionFlag,
    DevicePinnedBuffer,
    accelerator_count,
)
from max.dtype import DType

# Held while the gate blocks a synchronizing read: short enough to keep the
# suite quick, long enough that a non-blocking (regressed) read finishes first.
_GATE_HOLD_SECONDS = 2.0
# Generous upper bound for the released read to finish; only reached on failure.
_JOIN_TIMEOUT_SECONDS = 60.0


@pytest.fixture
def gpu() -> Accelerator:
    if accelerator_count() == 0:
        pytest.skip("requires a GPU")
    device = Accelerator()
    if device.api not in ("cuda", "hip"):
        pytest.skip("stream host-value gating requires CUDA/HIP")
    return device


def _run_gated_read(
    gpu: Accelerator,
    *,
    enqueue_gated_write: Callable[[], None],
    read_under_test: Callable[[], object],
) -> tuple[bool, object]:
    """Runs ``read_under_test`` against a write stalled behind a stream gate.

    ``enqueue_gated_write`` lands on the same gated default stream, so a
    synchronizing read cannot finish until the gate releases. Returns
    ``(completed_while_gated, value)``: ``completed_while_gated`` is ``False``
    for a read that synchronized; ``value`` is the released read's result.
    """
    flag = CompletionFlag(gpu)
    gate_stream = gpu.default_queue
    done = threading.Event()
    box: dict[str, object] = {}

    def worker() -> None:
        try:
            box["value"] = read_under_test()
        except BaseException as exc:
            box["error"] = exc
        finally:
            done.set()

    worker_thread = threading.Thread(target=worker, daemon=True)
    gate_stream.wait_for_host_value(flag, 1)
    completed_while_gated = True
    try:
        enqueue_gated_write()
        worker_thread.start()
        completed_while_gated = done.wait(timeout=_GATE_HOLD_SECONDS)
    finally:
        flag.signal(1)
        if worker_thread.ident is not None:
            worker_thread.join(timeout=_JOIN_TIMEOUT_SECONDS)

    assert worker_thread.ident is not None
    assert not worker_thread.is_alive(), (
        "read thread did not finish within"
        f" {_JOIN_TIMEOUT_SECONDS}s after the gate released"
    )
    if "error" in box:
        error = box["error"]
        assert isinstance(error, BaseException)
        raise AssertionError("read raised on the worker thread") from error
    return completed_while_gated, box["value"]


def test_dlpack_export_synchronizes(gpu: Accelerator) -> None:
    """``to_numpy``/``__dlpack__`` on a non-pinned device buffer synchronizes.

    DRIV-311 contract 1: a fill enqueued behind the gate must be observed, so
    the read blocks until release instead of returning the pre-write zeros.
    """
    filled = np.arange(1, 5, dtype=np.int32)
    buf = Buffer.from_numpy(np.zeros(4, dtype=np.int32)).to(gpu)
    src = Buffer.from_numpy(filled)

    completed_while_gated, value = _run_gated_read(
        gpu,
        enqueue_gated_write=lambda: buf.inplace_copy_from(src),
        read_under_test=buf.to_numpy,
    )

    assert not completed_while_gated, (
        "to_numpy()/__dlpack__ returned before the gated write completed;"
        " the DLPack export did not synchronize"
    )
    assert isinstance(value, np.ndarray)
    np.testing.assert_array_equal(value, filled)


def test_item_synchronizes(gpu: Accelerator) -> None:
    """``item()`` observes previously enqueued work.

    DRIV-311 contract 2: ``item()`` reads host-mapped memory directly, bypassing
    the stream chain, so it must insert an explicit sync to see the gated fill.
    """
    pinned = DevicePinnedBuffer.zeros(shape=[1], dtype=DType.int32, device=gpu)
    src = Buffer.from_numpy(np.array([7], dtype=np.int32)).to(gpu)

    completed_while_gated, value = _run_gated_read(
        gpu,
        enqueue_gated_write=lambda: pinned.inplace_copy_from(src),
        read_under_test=pinned.item,
    )

    assert not completed_while_gated, (
        "item() returned before the gated write completed; it did not"
        " synchronize the pending device work"
    )
    assert value == 7
