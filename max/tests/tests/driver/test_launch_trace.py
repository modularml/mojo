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
"""``max.driver`` launch trace: ``begin``/``take``/``launch_trace``.

The launch trace records the operations (kernel launches, memcpys, memsets)
enqueued across all streams, letting tests assert which device work a code path
issues and on which stream -- the mechanism-level complement to the DRIV-311
black-box sync-contract tests. See ``test_launch_trace_graph_gpu.py`` for
capturing a compiled graph's internal kernel launches without a stream handle.
"""

from __future__ import annotations

import numpy as np
import pytest
from max import driver
from max.driver import Accelerator, Buffer, LaunchTraceEntry, accelerator_count


@pytest.fixture
def gpu() -> Accelerator:
    if accelerator_count() == 0:
        pytest.skip("requires a GPU")
    device = Accelerator()
    if device.api not in ("cuda", "hip"):
        pytest.skip("launch tracing records entries only on CUDA/HIP")
    return device


def test_launch_trace_records_h2d_copy(gpu: Accelerator) -> None:
    """An enqueued host-to-device copy appears as an HtoD memcpy entry."""
    dst = Buffer.from_numpy(np.zeros(4, dtype=np.int32)).to(gpu)
    src = Buffer.from_numpy(np.arange(4, dtype=np.int32))

    with driver.launch_trace() as entries:
        dst.inplace_copy_from(src)

    h2d = [
        e
        for e in entries
        if e.kind == LaunchTraceEntry.OperationKind.MEMCPY
        and e.memcpy_kind == LaunchTraceEntry.MemcpyKind.HTOD
    ]
    assert h2d, f"no HtoD memcpy entry in trace: {entries}"
    assert any(e.memcpy_byte_size == 16 for e in h2d)
    assert all(e.semantic_hash != 0 for e in h2d)
    # A single-stream trace still tags every entry with a stream index.
    assert all(e.stream_index >= 0 for e in entries)


def test_launch_trace_context_manager_stops_on_exception(
    gpu: Accelerator,
) -> None:
    """``launch_trace`` stops recording even when the block raises."""
    dst = Buffer.from_numpy(np.zeros(4, dtype=np.int32)).to(gpu)
    src = Buffer.from_numpy(np.arange(4, dtype=np.int32))

    with pytest.raises(RuntimeError):
        with driver.launch_trace() as entries:
            dst.inplace_copy_from(src)
            raise RuntimeError("boom")

    # The block's work was still captured before it raised.
    assert entries, "context manager dropped entries when the block raised"
    # Recording stopped despite the exception, so nothing leaks into a
    # subsequent take.
    dst.inplace_copy_from(src)
    assert driver.take_launch_trace() == []


def test_take_launch_trace_stops_and_clears(gpu: Accelerator) -> None:
    """``take_launch_trace`` returns each entry once and stops recording."""
    dst = Buffer.from_numpy(np.zeros(4, dtype=np.int32)).to(gpu)
    src = Buffer.from_numpy(np.arange(4, dtype=np.int32))

    driver.begin_launch_trace()
    dst.inplace_copy_from(src)
    assert driver.take_launch_trace()

    # Tracing stopped: further work is not recorded and nothing is retained.
    dst.inplace_copy_from(src)
    assert driver.take_launch_trace() == []


def test_take_launch_trace_without_begin_is_empty(gpu: Accelerator) -> None:
    """Without ``begin_launch_trace`` nothing is recorded."""
    dst = Buffer.from_numpy(np.zeros(4, dtype=np.int32)).to(gpu)
    dst.inplace_copy_from(Buffer.from_numpy(np.arange(4, dtype=np.int32)))
    assert driver.take_launch_trace() == []
