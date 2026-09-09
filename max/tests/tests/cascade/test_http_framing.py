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
"""Tests for the HTTP transport's length-prefixed frame reassembly.

Exercises :py:func:`_read_framed` against a real
:py:class:`aiohttp.StreamReader` rather than a stub, because the batching
behaviour it relies on -- ``readany`` returning every buffered byte in one
call -- is aiohttp's, and a stub would keep passing if a version bump changed
it.
"""

from __future__ import annotations

import asyncio
import struct
from collections.abc import Sequence

import pytest
from aiohttp.base_protocol import BaseProtocol
from aiohttp.streams import StreamReader
from max.experimental.cascade.http_runtime.client import _read_framed


def _frame(payload: bytes) -> bytes:
    """Length-prefix ``payload`` the way the server writes it."""
    return struct.pack(">I", len(payload)) + payload


def _reader() -> StreamReader:
    """Build a stream that tests feed directly, with no socket behind it."""
    loop = asyncio.get_running_loop()
    protocol = BaseProtocol(loop)
    # StreamReader refuses to wait on a protocol it considers disconnected, and
    # a bare Transport is enough: the only calls that reach it are aiohttp's
    # flow-control hooks, which swallow the base class's NotImplementedError.
    protocol.transport = asyncio.Transport()
    return StreamReader(protocol, limit=2**16, loop=loop)


async def _settle() -> None:
    """Let every other ready task run to its next suspension point."""
    for _ in range(5):
        await asyncio.sleep(0)


async def _read_all(chunks: Sequence[bytes]) -> list[list[bytes]]:
    """Return the batches ``_read_framed`` yields for a chunked byte stream.

    Each chunk is fed a full event-loop settle after the previous one, so the
    reader really does have to wait on a chunk that ends mid-frame instead of
    finding the rest already buffered.
    """
    reader = _reader()

    async def _feed() -> None:
        for chunk in chunks:
            await _settle()
            reader.feed_data(chunk)
        await _settle()
        reader.feed_eof()

    feeder = asyncio.ensure_future(_feed())
    batches = [batch async for batch in _read_framed(reader)]
    await feeder
    return batches


@pytest.mark.asyncio
async def test_single_frame() -> None:
    assert await _read_all([_frame(b"one")]) == [[b"one"]]


@pytest.mark.asyncio
async def test_zero_length_frame() -> None:
    """An empty payload is a real frame, not an end-of-stream marker."""
    assert await _read_all([_frame(b"")]) == [[b""]]


@pytest.mark.asyncio
async def test_clean_eof_yields_nothing() -> None:
    assert await _read_all([]) == []


@pytest.mark.asyncio
async def test_frames_buffered_together_arrive_as_one_batch() -> None:
    """The whole backlog drains in a single wake.

    This is the property the transport's batching rests on: while the consumer
    is busy, frames pile up, and the next drain hands over all of them at once.
    """
    payloads = [b"one", b"two", b"three"]
    batches = await _read_all([b"".join(_frame(p) for p in payloads)])
    assert batches == [payloads]


@pytest.mark.asyncio
async def test_frame_split_mid_header() -> None:
    frame = _frame(b"one")
    assert await _read_all([frame[:2], frame[2:]]) == [[b"one"]]


@pytest.mark.asyncio
async def test_frame_split_mid_payload() -> None:
    frame = _frame(b"hello")
    assert await _read_all([frame[:6], frame[6:]]) == [[b"hello"]]


@pytest.mark.asyncio
async def test_trailing_partial_frame_waits_for_the_rest() -> None:
    """A complete frame is delivered without waiting on the one behind it."""
    first, second = _frame(b"one"), _frame(b"two")
    batches = await _read_all([first + second[:3], second[3:]])
    assert batches == [[b"one"], [b"two"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_size", [1, 3, 7, 16, 64])
async def test_frames_survive_arbitrary_chunk_boundaries(
    chunk_size: int,
) -> None:
    """Frame order and content are independent of where reads happen to split."""
    payloads = [b"", b"a", b"bc" * 5, b"d" * 40, b"e", b"f" * 17]
    stream = b"".join(_frame(p) for p in payloads)
    chunks = [
        stream[i : i + chunk_size] for i in range(0, len(stream), chunk_size)
    ]

    batches = await _read_all(chunks)

    assert [f for batch in batches for f in batch] == payloads
    assert all(batches), "an empty batch would be a spurious wake"


@pytest.mark.asyncio
async def test_truncated_payload_reports_the_expected_frame_size() -> None:
    reader = _reader()
    reader.feed_data(struct.pack(">I", 10) + b"abc")
    reader.feed_eof()

    with pytest.raises(asyncio.IncompleteReadError) as excinfo:
        async for _ in _read_framed(reader):
            pass

    assert excinfo.value.partial == struct.pack(">I", 10) + b"abc"
    assert excinfo.value.expected == 14


@pytest.mark.asyncio
async def test_truncated_header_reports_the_header_size() -> None:
    reader = _reader()
    reader.feed_data(b"\x00\x00")
    reader.feed_eof()

    with pytest.raises(asyncio.IncompleteReadError) as excinfo:
        async for _ in _read_framed(reader):
            pass

    assert excinfo.value.partial == b"\x00\x00"
    assert excinfo.value.expected == 4
