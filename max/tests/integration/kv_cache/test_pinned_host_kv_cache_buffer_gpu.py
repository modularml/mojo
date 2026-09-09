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

"""Tests for the fast-path pinned host buffer used as the host KV cache.

The host KV cache buffer is a 2-D, page-locked ``DevicePinnedBuffer`` of shape
``[num_blocks, bytes_per_block]`` allocated via
``_unsafe_alloc_fast_pinned_buffer``; ``BlockOffloadEngine`` indexes row ``bid``
as block ``bid``. These tests cover that block-keyed access pattern plus the
lower-level fast-path mechanics.
"""

import time
from collections.abc import Iterator

import numpy as np
import pytest
from max.driver import (
    Accelerator,
    DevicePinnedBuffer,
    _unsafe_alloc_fast_pinned_buffer,
    _unsafe_free_fast_pinned_buffer,
    accelerator_count,
)
from max.dtype import DType

pytestmark = pytest.mark.skipif(
    accelerator_count() == 0, reason="No GPU available"
)

BYTES_PER_BLOCK = 1024

# Fast pinned buffers are freed explicitly (not GC'd), so track every buffer a
# test allocates and release them after the test. These tests only touch the
# region from the host (to_numpy on slices), with no in-flight GPU copies, so
# no stream synchronization is needed before freeing.
_allocated: list[DevicePinnedBuffer] = []


@pytest.fixture(autouse=True)
def _free_fast_pinned_buffers() -> Iterator[None]:
    yield
    for buf in _allocated:
        _unsafe_free_fast_pinned_buffer(buf)
    _allocated.clear()


def _make_buffer(
    total_num_blocks: int,
    bytes_per_block: int = BYTES_PER_BLOCK,
) -> DevicePinnedBuffer:
    """A 2-D fast pinned host buffer, as BlockOffloadEngine allocates."""
    buf = _unsafe_alloc_fast_pinned_buffer(
        DType.uint8,
        [total_num_blocks, bytes_per_block],
        Accelerator(),
    )
    _allocated.append(buf)
    return buf


def _page_np(buf: DevicePinnedBuffer, bid: int) -> np.ndarray:
    """1-D NumPy view of block ``bid`` (matches the connector's access)."""
    return buf[bid, :].to_numpy()


def test_shape_and_pinned() -> None:
    buf = _make_buffer(3)
    assert tuple(buf.shape) == (3, BYTES_PER_BLOCK)
    assert buf.pinned


def test_page_oob_too_large() -> None:
    buf = _make_buffer(10)
    with pytest.raises(IndexError, match="out of bounds"):
        _ = buf[10, :]


def test_page_oob_too_negative() -> None:
    buf = _make_buffer(10)
    with pytest.raises(IndexError, match="out of bounds"):
        _ = buf[-11, :]


def test_page_view_returns_correct_size() -> None:
    buf = _make_buffer(4)
    view = buf[0, :]
    assert view.num_elements == BYTES_PER_BLOCK


def test_numpy_page_view_shape() -> None:
    buf = _make_buffer(4)
    arr = _page_np(buf, 0)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (BYTES_PER_BLOCK,)
    assert arr.dtype == np.uint8


def test_numpy_page_view_is_writable() -> None:
    buf = _make_buffer(4)
    _page_np(buf, 0)[:] = 42
    assert np.all(_page_np(buf, 0) == 42)


def test_pages_are_independent() -> None:
    buf = _make_buffer(4)
    _page_np(buf, 0)[:] = 1
    _page_np(buf, 1)[:] = 2
    assert np.all(_page_np(buf, 0) == 1)
    assert np.all(_page_np(buf, 1) == 2)


def test_pages_are_contiguous() -> None:
    """Adjacent pages must be at consecutive byte addresses (single region)."""
    buf = _make_buffer(4)
    addr0 = _page_np(buf, 0).ctypes.data
    addr1 = _page_np(buf, 1).ctypes.data
    addr2 = _page_np(buf, 2).ctypes.data
    assert addr1 - addr0 == BYTES_PER_BLOCK
    assert addr2 - addr1 == BYTES_PER_BLOCK


# --- fast-path direct tests ---------------------------------------------- #


def _make_fast(
    total_bytes: int,
    threads: int = 16,
    chunk_bytes: int = 512 * 1024 * 1024,
) -> DevicePinnedBuffer:
    buf = _unsafe_alloc_fast_pinned_buffer(
        DType.uint8,
        [total_bytes],
        Accelerator(),
        threads=threads,
        chunk_bytes=chunk_bytes,
    )
    _allocated.append(buf)
    return buf


def _bytes_np(fb: DevicePinnedBuffer, off: int, nbytes: int) -> np.ndarray:
    """Zero-copy NumPy view over a byte range of a 1-D buffer (slice+to_numpy)."""
    return fb[off : off + nbytes].to_numpy()


def test_fast_pinned_buffer_small_alloc_is_contiguous() -> None:
    nbytes = 4 * 1024 * 1024
    t0 = time.perf_counter()
    fb = _make_fast(nbytes)
    elapsed = time.perf_counter() - t0
    print(
        f"\nDevicePinnedBuffer(fast=True, {nbytes / 1024 / 1024:.0f} MiB) alloc: "
        f"{elapsed * 1e3:.1f} ms"
    )
    assert fb.num_elements == nbytes
    # The region is one gap-free run: every sub-view's address is base + offset.
    base = fb._data_ptr()
    page = 1024 * 1024
    for off in range(0, nbytes, page):
        assert _bytes_np(fb, off, page).ctypes.data == base + off


def test_fast_pinned_buffer_numpy_view_roundtrip() -> None:
    fb = _make_fast(1024 * 1024)
    a = _bytes_np(fb, 0, 4096)
    b = _bytes_np(fb, 4096, 4096)
    a[:] = 11
    b[:] = 22
    assert np.all(_bytes_np(fb, 0, 4096) == 11)
    assert np.all(_bytes_np(fb, 4096, 4096) == 22)
    # Adjacent views are contiguous.
    assert b.ctypes.data - a.ctypes.data == 4096


def test_fast_pinned_buffer_view_is_pinned_buffer() -> None:
    fb = _make_fast(1024 * 1024)
    view = fb[0:4096]
    assert view.num_elements == 4096
    assert view.pinned


def test_fast_pinned_buffer_custom_threads_and_chunk_bytes() -> None:
    # threads and chunk_bytes are tunables; smaller-than-default values must
    # still allocate a correct, contiguous, registered region. 8 MiB over
    # 1 MiB chunks exercises the multi-chunk register path with 2 workers.
    nbytes = 8 * 1024 * 1024
    fb = _make_fast(nbytes, threads=2, chunk_bytes=1024 * 1024)
    assert fb.num_elements == nbytes
    a = _bytes_np(fb, 0, 4096)
    a[:] = 7
    assert np.all(_bytes_np(fb, 0, 4096) == 7)
    # Region remains a single gap-free run regardless of chunking.
    base = fb._data_ptr()
    assert _bytes_np(fb, nbytes - 4096, 4096).ctypes.data == base + (
        nbytes - 4096
    )
