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

"""Regression tests for DI prefill-request context deserialization.

The DI decode node serializes ``PrefillRequest`` with the architecture's
concrete context subclass (e.g. ``KimiK2_5TextAndVisionContext``), but msgspec
decodes each field at its *declared* type. If the prefill dispatcher's decoder
is bound to the base ``TextContext``, the VLM subclass — and its vision fields
— are silently dropped on the wire, crashing the prefill worker's
vision-encode drive (``as_vision_context_batches``) on the first request.
These tests round-trip a request through the real dispatcher sockets and
assert the concrete context type survives.
"""

from __future__ import annotations

import queue
import time

import numpy as np
import pytest
import zmq
from max._core import nixl
from max.pipelines.architectures.kimik2_5.context import (
    KimiK2_5TextAndVisionContext,
)
from max.pipelines.context import TextContext, TokenBuffer
from max.pipelines.context.context import TextAndVisionContext
from max.pipelines.modeling.types import RequestID
from max.serve.scheduler.base import PrefillRequest
from max.serve.scheduler.di_dispatchers import (
    DecodeDispatcherClient,
    PrefillDispatcherServer,
)
from max.serve.worker_interface._zmq_queue import generate_zmq_ipc_path

_TIMEOUT = 10.0

# The dispatcher's wire type unions in ``KVTransferEngineMetadata``, whose
# ``memory_type`` field is annotated ``nixl.MemoryType``. Outside linux-x86_64
# the NIXL bindings are an empty stub module, so msgspec cannot resolve that
# annotation and the dispatcher cannot be constructed at all. DI is unreachable
# on those platforms anyway.
pytestmark = pytest.mark.skipif(
    not hasattr(nixl, "MemoryType"),
    reason="DI dispatchers need the NIXL bindings (linux-x86_64 only)",
)


def _round_trip(
    context: TextContext, context_type: type[TextContext] | None = None
) -> PrefillRequest[TextContext]:
    """Sends a ``PrefillRequest`` through real dispatcher sockets and returns
    the decoded request as seen by the prefill scheduler."""
    server_addr = generate_zmq_ipc_path()
    if context_type is None:
        server = PrefillDispatcherServer(bind_addr=server_addr)
    else:
        server = PrefillDispatcherServer(
            bind_addr=server_addr, context_type=context_type
        )
    client = DecodeDispatcherClient(bind_addr=generate_zmq_ipc_path())
    request = PrefillRequest(
        id=context.request_id,
        context=context,
        transfer_engine_name="test_engine",
        dst_block_ids=[0, 1],
        dst_replica_idx=0,
    )
    deadline = time.monotonic() + _TIMEOUT
    while True:
        try:
            client.send_request_nowait(request, server_addr)
            break
        except zmq.Again:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.001)
    while True:
        try:
            decoded, _identity = server.recv_request_nowait()
            break
        except queue.Empty:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.001)
    assert isinstance(decoded, PrefillRequest)
    return decoded


def test_prefill_request_preserves_vlm_context_subclass() -> None:
    """A VLM context subclass must survive the decode->prefill wire intact.

    Mirrors Kimi K2.x / Gemma4 under DI: the tokenizer always builds the
    ``TextAndVisionContext`` subclass, even for text-only requests.
    """
    context = KimiK2_5TextAndVisionContext(
        request_id=RequestID(),
        max_length=128,
        tokens=TokenBuffer(np.arange(8, dtype=np.int64)),
        vision_token_ids=[163605],
        grid_thws=np.array([[1, 4, 4]], dtype=np.int64),
        image_token_indices=np.array([2, 3], dtype=np.int32),
        max_h=4,
        max_w=4,
    )
    decoded = _round_trip(context, context_type=KimiK2_5TextAndVisionContext)

    # The exact narrowing the prefill worker's vision drive performs.
    assert isinstance(decoded.context, TextAndVisionContext)
    assert isinstance(decoded.context, KimiK2_5TextAndVisionContext)
    np.testing.assert_array_equal(decoded.context.grid_thws, [[1, 4, 4]])
    np.testing.assert_array_equal(decoded.context.image_token_indices, [2, 3])
    assert decoded.context.vision_token_ids == [163605]
    np.testing.assert_array_equal(decoded.context.tokens.all, np.arange(8))


def test_prefill_request_plain_text_context_unchanged() -> None:
    """Text-only DI models (plain ``TextContext``) keep working unchanged."""
    context = TextContext(
        request_id=RequestID(),
        max_length=128,
        tokens=TokenBuffer(np.arange(8, dtype=np.int64)),
    )
    decoded = _round_trip(context)

    assert type(decoded.context) is TextContext
    assert decoded.id == context.request_id
    np.testing.assert_array_equal(decoded.context.tokens.all, np.arange(8))
