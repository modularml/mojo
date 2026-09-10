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
"""Hazard record tests that need two accelerators to mean anything.

Several of this layer's decisions are about *which* queue a token belongs on,
and on one device the right answer and the wrong one are the same object.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest
from max._core.driver import HostHazardCompletion
from max.driver import (
    Accelerator,
    Buffer,
    CompletionFlag,
    Usage,
    accelerator_count,
)
from max.driver._hazard import stamp_device_write
from max.dtype import DType

_READY_TIMEOUT_SECONDS = 10.0
# Held while the gate blocks one device's queue.
_GATE_HOLD_SECONDS = 2.0
# Generous upper bounds, only reached on failure.
_ENQUEUE_TIMEOUT_SECONDS = 10.0
_JOIN_TIMEOUT_SECONDS = 60.0

_SENTINEL = np.arange(1, 5, dtype=np.int32)


@pytest.fixture
def gpus() -> tuple[Accelerator, Accelerator]:
    if accelerator_count() < 2:
        pytest.skip("requires two GPUs")
    first = Accelerator(id=0)
    if first.api not in ("cuda", "hip"):
        pytest.skip("stream host-value gating requires CUDA/HIP")
    return first, Accelerator(id=1)


def test_stamp_device_write_takes_one_token_per_device(
    gpus: tuple[Accelerator, Accelerator],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Buffers are grouped by their own queue: a single token spanning both
    devices would under-wait whichever one did not record it.
    """
    gpu0, gpu1 = gpus
    first = Buffer(DType.int32, (4,), device=gpu0, usage=Usage.STAGING)
    second = Buffer(DType.int32, (4,), device=gpu0, usage=Usage.STAGING)
    third = Buffer(DType.int32, (4,), device=gpu1, usage=Usage.STAGING)

    queues: list[int] = []
    original = HostHazardCompletion.record_on

    def recording(queue):  # noqa: ANN001, ANN202
        queues.append(queue._device_context_ptr())
        return original(queue)

    monkeypatch.setattr(HostHazardCompletion, "record_on", recording)

    stamp_device_write([first, second, third])

    assert sorted(queues) == sorted(
        {
            first.stream._device_context_ptr(),
            third.stream._device_context_ptr(),
        }
    ), "one token per device, and only per device"


def _wait_until_ready(completion: HostHazardCompletion) -> bool:
    deadline = time.monotonic() + _READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if completion.is_ready():
            return True
        time.sleep(0.01)
    return False


def test_a_gated_device_does_not_hold_up_the_other_devices_token(
    gpus: tuple[Accelerator, Accelerator],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The grouping resolves against two real queues, not two labels: stalling
    one device separates the tokens in time.
    """
    gpu0, gpu1 = gpus
    open_buf = Buffer(DType.int32, (4,), device=gpu0, usage=Usage.STAGING)
    gated_buf = Buffer(DType.int32, (4,), device=gpu1, usage=Usage.STAGING)
    gated_queue = gpu1.default_queue
    assert (
        gated_buf.stream._device_context_ptr()
        == gated_queue._device_context_ptr()
    ), "the buffer must be held by the queue this test gates"

    flag = CompletionFlag(gpu1)
    gated_queue.wait_for_host_value(flag, 1)
    try:
        by_queue: dict[int, HostHazardCompletion] = {}
        original = HostHazardCompletion.record_on

        def capturing(queue):  # noqa: ANN001, ANN202
            completion = original(queue)
            by_queue[queue._device_context_ptr()] = completion
            return completion

        monkeypatch.setattr(HostHazardCompletion, "record_on", capturing)
        stamp_device_write([open_buf, gated_buf])
        open_key = open_buf.stream._device_context_ptr()
        gated_key = gated_buf.stream._device_context_ptr()
        assert set(by_queue) == {open_key, gated_key}, (
            "the two devices did not each get a token"
        )
        open_token, gated_token = by_queue[open_key], by_queue[gated_key]

        assert _wait_until_ready(open_token), (
            "the ungated device's token never resolved"
        )
        assert not gated_token.is_ready(), (
            "a token recorded behind the gate reported ready; it is not on the"
            " queue it claims"
        )
    finally:
        flag.signal(1)

    gated_token.wait()


def test_a_copy_into_pinned_host_memory_is_waited_on_the_source_queue(
    gpus: tuple[Accelerator, Accelerator],
) -> None:
    """The host-destination branch's mistake is an under-wait, so measure it.

    A token taken on the destination's own queue would be ready the moment it
    was recorded, and a staging export synchronizes nothing underneath to
    catch that.

    Pinned destination memory is what keeps the copy asynchronous -- into
    pageable memory it would block its calling thread and stall the gate --
    and it is enqueued off the main thread so a blocking copy fails the
    assertion rather than holding the gate shut.
    """
    gpu0, gpu1 = gpus
    src = Buffer.from_numpy(_SENTINEL).to(gpu0)
    dst = Buffer(dtype=DType.int32, shape=[4], device=gpu1, usage=Usage.STAGING)
    gpu0.synchronize()

    flag = CompletionFlag(gpu0)
    enqueued = threading.Event()
    read_done = threading.Event()
    box: dict[str, object] = {}

    def writer() -> None:
        try:
            dst.inplace_copy_from(src)
        except BaseException as exc:
            box["write_error"] = exc
        finally:
            enqueued.set()

    def reader() -> None:
        try:
            box["value"] = dst.to_numpy()
        except BaseException as exc:
            box["read_error"] = exc
        finally:
            read_done.set()

    writer_thread = threading.Thread(target=writer, daemon=True)
    reader_thread = threading.Thread(target=reader, daemon=True)
    enqueued_while_gated = False
    completed_while_gated = True
    try:
        gpu0.default_queue.wait_for_host_value(flag, 1)
        writer_thread.start()
        enqueued_while_gated = enqueued.wait(timeout=_ENQUEUE_TIMEOUT_SECONDS)
        if enqueued_while_gated:
            reader_thread.start()
            completed_while_gated = read_done.wait(timeout=_GATE_HOLD_SECONDS)
    finally:
        flag.signal(1)
        for thread in (writer_thread, reader_thread):
            if thread.ident is not None:
                thread.join(timeout=_JOIN_TIMEOUT_SECONDS)

    assert not writer_thread.is_alive() and not reader_thread.is_alive(), (
        "a worker did not finish within"
        f" {_JOIN_TIMEOUT_SECONDS}s after the gate released"
    )
    for key in ("write_error", "read_error"):
        if key in box:
            error = box[key]
            assert isinstance(error, BaseException)
            raise AssertionError(f"{key} on a worker thread") from error
    assert enqueued_while_gated, (
        "the copy did not return while its queue was gated, so nothing about"
        " the read that follows it can be measured"
    )
    assert not completed_while_gated, (
        "the host read returned while the copy filling it was still stalled"
        " on the source device's queue"
    )
    assert isinstance(box["value"], np.ndarray)
    np.testing.assert_array_equal(box["value"], _SENTINEL)
