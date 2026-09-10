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
"""Contract tests for the host hazard record layer, on a real accelerator.

Precision is only observable on a device whose queue can be stalled from the
host. Each test gates the default queue with a host-signalled
``CompletionFlag`` and places the producing write *ahead* of that gate, so the
producer finishes on its own: a read that returns while the gate is held was
scoped to that producer, and one that blocks drained the device instead.

Each test carries a positive control -- a full ``device.synchronize()`` on a
second thread, required to stay blocked -- so a gate that stopped engaging
fails the test rather than making it vacuous. What no test here can show is a
wait that resolves against nothing, since the producer completes either way;
that is answered by the staging case of
``test_staging_dlpack_does_not_synchronize`` in the device-pinned suite.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import numpy as np
import pytest
from max._core.driver import (
    HostHazardCompletion,
    _release_buffers_to_borrowed,
)
from max.driver import (
    Accelerator,
    Buffer,
    CompletionFlag,
    DevicePinnedBuffer,
    Usage,
    accelerator_count,
)
from max.dtype import DType

# Held while the gate blocks the queue: short enough to keep the suite quick,
# long enough that a read which does not block finishes well inside it.
_GATE_HOLD_SECONDS = 2.0
# Generous upper bound for the released read to finish; only reached on failure.
_JOIN_TIMEOUT_SECONDS = 60.0

_SENTINEL = np.arange(1, 5, dtype=np.int32)


@pytest.fixture
def gpu() -> Accelerator:
    if accelerator_count() == 0:
        pytest.skip("requires a GPU")
    device = Accelerator()
    if device.api not in ("cuda", "hip"):
        pytest.skip("stream host-value gating requires CUDA/HIP")
    return device


def _read_ahead_of_a_gate(
    gpu: Accelerator,
    host_buf: Buffer,
    *,
    enqueue_write: Callable[[], None],
) -> tuple[bool, object]:
    """Reads ``host_buf`` on a worker thread while the default queue is gated.

    A copy between two otherwise unused buffers is queued behind the gate, so
    what a conservative read absorbs is real work on unrelated buffers rather
    than the gate itself.

    Returns ``(completed_while_gated, value)``.
    """
    flag = CompletionFlag(gpu)
    queue = gpu.default_queue
    unrelated_dst = Buffer(DType.int32, (4,), device=gpu)
    unrelated_src = Buffer(DType.int32, (4,), device=gpu)
    done = threading.Event()
    control_done = threading.Event()
    box: dict[str, object] = {}

    def worker() -> None:
        try:
            box["value"] = host_buf.to_numpy()
        except BaseException as exc:
            box["error"] = exc
        finally:
            done.set()

    def control() -> None:
        try:
            gpu.synchronize()
        finally:
            control_done.set()

    worker_thread = threading.Thread(target=worker, daemon=True)
    control_thread = threading.Thread(target=control, daemon=True)
    completed_while_gated = True
    control_completed_while_gated = True
    try:
        enqueue_write()
        queue.wait_for_host_value(flag, 1)
        unrelated_dst.inplace_copy_from(unrelated_src)
        worker_thread.start()
        control_thread.start()
        completed_while_gated = done.wait(timeout=_GATE_HOLD_SECONDS)
        # A hold of its own, since a read that returned early spent little of
        # the first one.
        control_completed_while_gated = control_done.wait(
            timeout=_GATE_HOLD_SECONDS
        )
    finally:
        flag.signal(1)
        for thread in (worker_thread, control_thread):
            if thread.ident is not None:
                thread.join(timeout=_JOIN_TIMEOUT_SECONDS)

    assert worker_thread.ident is not None
    assert not worker_thread.is_alive() and not control_thread.is_alive(), (
        "a thread did not finish within"
        f" {_JOIN_TIMEOUT_SECONDS}s after the gate released"
    )
    assert not control_completed_while_gated, (
        "a full device sync returned while the gate was held; the gate never"
        " engaged, so nothing else this test observed means anything"
    )
    if "error" in box:
        error = box["error"]
        assert isinstance(error, BaseException)
        raise AssertionError("read raised on the worker thread") from error
    return completed_while_gated, box["value"]


def test_read_not_blocked_by_unrelated_gated_work(gpu: Accelerator) -> None:
    """A host read waits for the copy that filled the buffer, and no further.

    This cannot distinguish a read that waits for nothing, since the producer
    completes either way; ``test_staging_dlpack_does_not_synchronize`` in the
    device-pinned suite rules that out.
    """
    staging = Buffer(
        dtype=DType.int32, shape=[4], device=gpu, usage=Usage.STAGING
    )
    src = Buffer.from_numpy(_SENTINEL).to(gpu)
    assert staging._host_hazard_tracked

    completed_while_gated, value = _read_ahead_of_a_gate(
        gpu, staging, enqueue_write=lambda: staging.inplace_copy_from(src)
    )

    assert completed_while_gated, (
        "the staging read absorbed unrelated work queued behind the gate;"
        " it waited on the device rather than on its own producer"
    )
    assert isinstance(value, np.ndarray)
    np.testing.assert_array_equal(value, _SENTINEL)


def test_a_borrowed_handle_inherits_its_parents_history(
    gpu: Accelerator,
) -> None:
    """A borrowed handle carries a copy of the parent's record, so a copy
    issued through it is stamped and the read stays scoped to that copy.
    """
    staging = Buffer(
        dtype=DType.int32, shape=[4], device=gpu, usage=Usage.STAGING
    )
    src = Buffer.from_numpy(_SENTINEL).to(gpu)
    (staging,) = _release_buffers_to_borrowed([staging])
    assert staging._host_hazard_tracked

    completed_while_gated, value = _read_ahead_of_a_gate(
        gpu,
        staging,
        enqueue_write=lambda: staging._inplace_copy_from(src),
    )

    assert completed_while_gated, (
        "the read drained the device; a borrowed handle with a record should"
        " wait on its own copy"
    )
    assert isinstance(value, np.ndarray)
    np.testing.assert_array_equal(value, _SENTINEL)


def test_stamps_skip_untracked_buffers(gpu: Accelerator) -> None:
    """Nothing is written to an untracked buffer's slot, in either direction."""
    completion = HostHazardCompletion.record_on(gpu.default_queue)
    pinned = DevicePinnedBuffer(dtype=DType.int32, shape=[4], device=gpu)

    pinned._stamp_write(completion)
    pinned._stamp_read(completion)

    assert not pinned._host_hazard_tracked


@pytest.fixture
def accelerator() -> Accelerator:
    """Any accelerator: this case needs a tracked buffer, not a stalled queue."""
    if accelerator_count() == 0:
        pytest.skip("requires an accelerator")
    return Accelerator()


# Large enough that a D2H cannot land between its enqueue and the read that
# follows, small enough to allocate three of them on any accelerator.
_CHURN_ELEMENTS = 4 * 1024 * 1024
_CHURN_ROUNDS = 512


def _staging_read_behind_queued_work(
    accelerator: Accelerator, usage: Usage
) -> np.ndarray:
    """Snapshots a staging buffer whose D2H is queued behind a device chain.

    Needs no queue gate: the chain cannot have run, so a read that does not
    wait sees the host pre-fill rather than the copy.
    """
    staging = Buffer(
        dtype=DType.int32,
        shape=[_CHURN_ELEMENTS],
        device=accelerator,
        usage=usage,
    )
    staging.to_numpy()[:] = 0

    src = Buffer.from_numpy(np.full(_CHURN_ELEMENTS, 7, dtype=np.int32)).to(
        accelerator
    )
    scratch = Buffer(DType.int32, (_CHURN_ELEMENTS,), device=accelerator)
    accelerator.synchronize()

    for _ in range(_CHURN_ROUNDS):
        scratch.inplace_copy_from(src)
    staging.inplace_copy_from(src)

    # Zero-copy on staging, so snapshot before the DMA can land.
    return staging.to_numpy().copy()


def test_a_tracked_staging_read_waits_for_its_copy(
    accelerator: Accelerator,
) -> None:
    """One usage flag apart, so only the wait can explain the difference."""
    tracked = _staging_read_behind_queued_work(accelerator, Usage.STAGING)
    untracked = _staging_read_behind_queued_work(
        accelerator, Usage.STAGING | Usage.UNTRACKED
    )
    accelerator.synchronize()

    assert (tracked == 7).all(), (
        "a tracked staging read returned before the copy that filled it"
    )
    if not (untracked == 0).any():
        pytest.skip(
            "the untracked read already saw the copy, so the assertion above"
            " proves nothing on this hardware"
        )


def _write_behind_a_gated_read(gpu: Accelerator, host_buf: Buffer) -> bool:
    """Writes ``host_buf`` from a worker while a device read of it is gated.

    The H2D out of ``host_buf`` is queued *behind* the gate, so it cannot have
    run. A write that returns while the gate is held did not wait for it. A
    gate that never engaged lets the read finish and the write return, so it
    fails the caller's assertion rather than passing vacuously.
    """
    flag = CompletionFlag(gpu)
    queue = gpu.default_queue
    dev_dst = Buffer(DType.int32, (4,), device=gpu)
    done = threading.Event()
    box: dict[str, object] = {}

    def worker() -> None:
        try:
            host_buf[0] = 99
        except BaseException as exc:
            box["error"] = exc
        finally:
            done.set()

    worker_thread = threading.Thread(target=worker, daemon=True)
    completed_while_gated = True
    try:
        queue.wait_for_host_value(flag, 1)
        dev_dst.inplace_copy_from(host_buf)
        worker_thread.start()
        completed_while_gated = done.wait(timeout=_GATE_HOLD_SECONDS)
    finally:
        flag.signal(1)
        if worker_thread.ident is not None:
            worker_thread.join(timeout=_JOIN_TIMEOUT_SECONDS)

    assert not worker_thread.is_alive(), (
        f"the write did not finish within {_JOIN_TIMEOUT_SECONDS}s after the"
        " gate released"
    )
    if "error" in box:
        error = box["error"]
        assert isinstance(error, BaseException)
        raise AssertionError("write raised on the worker thread") from error
    return completed_while_gated


def test_a_host_write_waits_for_a_pending_device_read(gpu: Accelerator) -> None:
    """The write-after-read direction: a reader must be drained first."""
    staging = Buffer(
        dtype=DType.int32, shape=[4], device=gpu, usage=Usage.STAGING
    )
    assert staging._host_hazard_tracked

    assert not _write_behind_a_gated_read(gpu, staging), (
        "the host write returned while the device was still reading the"
        " buffer; the reader set was not drained"
    )


def test_an_untracked_host_write_takes_no_wait(gpu: Accelerator) -> None:
    """Control: the gate stalls the queue, not the host."""
    pinned = DevicePinnedBuffer(dtype=DType.int32, shape=[4], device=gpu)
    assert not pinned._host_hazard_tracked

    assert _write_behind_a_gated_read(gpu, pinned), (
        "an untracked write blocked, so the test above cannot attribute its"
        " block to the reader set"
    )
