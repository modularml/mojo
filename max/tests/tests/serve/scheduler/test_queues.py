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

import queue
import sys
import time
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import pytest
import zmq
from max.pipelines.context import (
    ImageMetadata,
    TextAndVisionContext,
    TextContext,
    TokenBuffer,
)
from max.pipelines.modeling.types import (
    RequestID,
    SharedMemoryArray,
    msgpack_numpy_decoder,
    msgpack_numpy_encoder,
)
from max.serve.worker_interface._zmq_queue import (
    ZmqConfig,
    ZmqPullSocket,
    ZmqPushSocket,
    _validate_zmq_address,
    generate_zmq_ipc_path,
)
from pytest_mock import MockerFixture


def dataclass_equal(left: Any, right: Any) -> bool:
    """Deep equality for dataclasses, handling numpy arrays and nested dataclasses.

    - Requires both `left` and `right` to be dataclass instances of the same type.
    - For each field:
        * If both values are dataclasses, compare them recursively.
        * If both values are numpy arrays, use np.array_equal.
        * Otherwise, use regular ==.
    """

    def _eq(lv: Any, rv: Any) -> bool:
        # Identity fast-path
        if lv is rv:
            return True

        # Nested dataclasses: recurse
        if is_dataclass(lv) and is_dataclass(rv):
            if type(lv) is not type(rv):
                return False
            for f in fields(lv):
                if not _eq(getattr(lv, f.name), getattr(rv, f.name)):
                    return False
            return True

        # NumPy array handling
        if isinstance(lv, np.ndarray) or isinstance(rv, np.ndarray):
            if not (isinstance(lv, np.ndarray) and isinstance(rv, np.ndarray)):
                return False  # one is array, the other is not
            return np.array_equal(lv, rv)

        # Fallback: normal equality
        return lv == rv

    if not (is_dataclass(left) and is_dataclass(right)):
        raise TypeError("dataclass_equal expects two dataclass instances")

    if type(left) is not type(right):
        return False

    for f in fields(left):
        if not _eq(getattr(left, f.name), getattr(right, f.name)):
            return False

    return True


def test_serialization_and_deserialization_through_queue_with_msgpack() -> None:
    test_address = generate_zmq_ipc_path()
    push_socket = ZmqPushSocket[tuple[RequestID, TextContext]](
        endpoint=test_address, payload_type=tuple[RequestID, TextContext]
    )

    pull_socket = ZmqPullSocket[tuple[RequestID, TextContext]](
        endpoint=test_address,
        payload_type=tuple[RequestID, TextContext],
    )

    context = (
        RequestID(),
        TextContext(
            request_id=RequestID(),
            max_length=15,
            tokens=TokenBuffer(np.ones(5, dtype=np.int64)),
        ),
    )

    push_socket.put(context)
    time.sleep(1)
    received_context = pull_socket.get_nowait()

    assert context[0] == received_context[0]
    assert dataclass_equal(context[1], received_context[1])


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="Shared memory via /dev/shm is not supported on macOS",
)
def test_vision_context_shared_memory_fallback(mocker: MockerFixture) -> None:
    """Test that vision context serialization falls back gracefully when shared memory is exhausted."""

    # Create realistic vision context with InternVL-sized image
    shape = (10, 32, 32, 3, 14, 14)
    img = np.random.rand(*shape).astype(np.float32)

    context = TextAndVisionContext(
        request_id=RequestID("test-request"),
        max_length=50,
        tokens=TokenBuffer(np.array([0, 1, 22, 22, 4], dtype=np.int64)),
        images=[ImageMetadata(start_idx=2, end_idx=4, pixel_values=img)],
        vision_token_ids=[22],
    )

    # Test the encoder directly
    encoder = msgpack_numpy_encoder(use_shared_memory=True)

    # Test 1: Fallback case - when shared memory allocation fails
    mocker.patch(
        "max.pipelines.modeling.types.utils.shared_memory._ndarray_to_shared_memory",
        return_value=None,
    )

    # Encode with fallback
    encoded_data = encoder(("test_req_id", context))
    # Decode to verify
    decoded = msgpack_numpy_decoder(tuple[str, TextAndVisionContext])(
        encoded_data
    )
    req_id, decoded_context = decoded

    assert req_id == "test_req_id"
    # In fallback case, images should be numpy arrays after round-trip
    assert isinstance(decoded_context.images[0].pixel_values, np.ndarray)
    assert np.allclose(decoded_context.images[0].pixel_values, img)

    # Verify original context wasn't modified
    assert isinstance(context.images[0].pixel_values, np.ndarray)

    # Test 2: Success case - when shared memory allocation succeeds
    mock_shm = SharedMemoryArray(
        name="test_shm_123", shape=shape, dtype="float32"
    )
    mocker.patch(
        "max.pipelines.modeling.types.utils.shared_memory._ndarray_to_shared_memory",
        return_value=mock_shm,
    )

    # Create a new context for second test
    context2 = TextAndVisionContext(
        request_id=RequestID("test-request-2"),
        max_length=50,
        tokens=TokenBuffer(np.array([0, 1, 22, 22, 4], dtype=np.int64)),
        images=[ImageMetadata(start_idx=2, end_idx=4, pixel_values=img)],
        vision_token_ids=[22],
    )

    # Encode with shared memory
    encoded_data2 = encoder(("test_req_id_2", context2))

    # Verify original context wasn't modified
    assert isinstance(context2.images[0].pixel_values, np.ndarray)

    # The encoded data should contain shared memory references
    # We can verify this by checking the encoded bytes contain the __shm__ marker
    # We add an assert message so that pytest does not print the huge encoded_data2
    # to stdout on failure.
    assert b"__shm__" in encoded_data2, (
        "Encoded data must contain __shm__ marker"
    )


def test_zmq_push_pull_queue_basic_functionality() -> None:
    """Test basic put and get_nowait functionality."""
    push_queue, pull_queue = ZmqConfig[int](int).pair()

    push_queue.put(42)
    time.sleep(1)
    result = pull_queue.get_nowait()
    assert result == 42


def test_zmq_push_pull_queue_with_complex_data() -> None:
    """Test queue with complex data structures using pickle serialization."""

    context = TextContext(
        request_id=RequestID(),
        max_length=15,
        tokens=TokenBuffer(np.array([1, 1, 1, 1, 1], dtype=np.int64)),
    )
    test_data = ("test_id", context)

    push_queue, pull_queue = ZmqConfig[tuple[str, TextContext]](
        tuple[str, TextContext]
    ).pair()

    push_queue.put(test_data)
    time.sleep(1)
    result = pull_queue.get_nowait()

    assert result[0] == test_data[0]
    assert dataclass_equal(result[1], test_data[1])


def test_zmq_push_pull_queue_with_custom_serialization() -> None:
    """Test queue with custom msgpack serialization."""
    context = TextContext(
        request_id=RequestID(),
        max_length=10,
        tokens=TokenBuffer(np.array([1, 2, 3, 4, 5], dtype=np.int64)),
    )
    test_data = (context.request_id, context)

    push_queue, pull_queue = ZmqConfig[tuple[RequestID, TextContext]](
        tuple[RequestID, TextContext]
    ).pair()

    try:
        push_queue.put(test_data)
        time.sleep(1)
        result = pull_queue.get_nowait()

        assert result[0] == test_data[0]
        assert dataclass_equal(result[1], test_data[1])
    finally:
        push_queue.close()
        pull_queue.close()


def test_zmq_push_pull_queue_empty_queue_raises_exception() -> None:
    """Test that get_nowait raises queue.Empty when queue is empty."""
    pull_queue = ZmqConfig[str](str).pull()

    with pytest.raises(queue.Empty):
        pull_queue.get_nowait()


def test_zmq_push_pull_queue_multiple_items() -> None:
    """Test queue with multiple items maintains order (FIFO)."""
    test_items = ["first", "second", "third", "fourth"]

    push_queue, pull_queue = ZmqConfig[str](str).pair()

    # Put all items
    for item in test_items:
        push_queue.put(item)
        time.sleep(1)

    # Get all items and verify order
    results = []
    for _ in test_items:
        results.append(pull_queue.get_nowait())

    assert results == test_items


def test_zmq_push_pull_queue_closed_state() -> None:
    """Test that operations fail when queue is closed."""
    push_queue, pull_queue = ZmqConfig[str](str).pair()
    push_queue.close()
    pull_queue.close()

    with pytest.raises(RuntimeError, match="Socket is closed"):
        push_queue.put_nowait("sample_str")

    with pytest.raises(RuntimeError, match="Socket is closed"):
        pull_queue.get_nowait()


def test_zmq_push_pull_queue_endpoint_validation() -> None:
    """Test that invalid endpoints raise ValueError."""
    with pytest.raises(
        ValueError,
        match=r"ZMQ address must start with tcp://, ipc://, or inproc://. Found: invalid://endpoint",
    ):
        ZmqPushSocket(endpoint="invalid://endpoint", payload_type=int)

    with pytest.raises(
        ValueError,
        match=r"ZMQ address must start with tcp://, ipc://, or inproc://. Found: ",
    ):
        ZmqPullSocket(endpoint="", payload_type=int)

    # OK
    ZmqPullSocket(
        endpoint="ipc://" + "a" * zmq.IPC_PATH_MAX_LEN, payload_type=int
    )
    # Not OK because path is empty
    with pytest.raises(
        ValueError,
        match=r"ZMQ IPC requires a path after the protocol. Found: ipc://",
    ):
        ZmqPullSocket(endpoint="ipc://", payload_type=int)

    # Not OK because too long
    with pytest.raises(ValueError, match="ZMQ IPC path is too long: ipc://"):
        ZmqPullSocket(
            endpoint="ipc://" + ("a" * (zmq.IPC_PATH_MAX_LEN + 1)),
            payload_type=int,
        )


def test_zmq_tcp_endpoint_validation() -> None:
    """Test TCP address validation including IPv4, IPv6, and edge cases."""
    # Valid IPv4 / hostname / wildcard
    _validate_zmq_address("tcp://127.0.0.1:5555")
    _validate_zmq_address("tcp://localhost:5555")
    _validate_zmq_address("tcp://*:5555")
    _validate_zmq_address("tcp://0.0.0.0:5555")
    _validate_zmq_address("tcp://hostname:1")
    _validate_zmq_address("tcp://hostname:65535")

    # Valid bracketed IPv6
    _validate_zmq_address("tcp://[::1]:5555")
    _validate_zmq_address("tcp://[::]:5555")
    _validate_zmq_address("tcp://[2600:1f18:b39:ec10:1cfb::9]:5555")

    # Missing port
    with pytest.raises(ValueError, match="tcp://host:port"):
        _validate_zmq_address("tcp://localhost")

    # IPv6 without port
    with pytest.raises(ValueError, match="tcp://host:port"):
        _validate_zmq_address("tcp://[::1]")

    # Non-numeric port
    with pytest.raises(ValueError, match="ZMQ tcp port must be a number"):
        _validate_zmq_address("tcp://localhost:abc")

    # Port 0 (out of range)
    with pytest.raises(
        ValueError, match="ZMQ tcp port must be between 1 and 65535"
    ):
        _validate_zmq_address("tcp://localhost:0")

    # Port > 65535
    with pytest.raises(ValueError):
        _validate_zmq_address("tcp://localhost:65536")


def test_zmq_push_pull_queue_with_vision_context() -> None:
    """Test queue with complex vision context data."""
    # Create vision context with image data
    shape = (2, 3, 224, 224)
    img = np.random.rand(*shape).astype(np.float32)

    context = TextAndVisionContext(
        request_id=RequestID("test-vision-request"),
        max_length=50,
        tokens=TokenBuffer(np.array([0, 1, 22, 22, 4], dtype=np.int64)),
        images=[ImageMetadata(start_idx=2, end_idx=4, pixel_values=img)],
        vision_token_ids=[22],
    )

    test_data = ("vision_test", context)

    push_queue, pull_queue = ZmqConfig[tuple[str, TextAndVisionContext]](
        tuple[str, TextAndVisionContext]
    ).pair()

    push_queue.put(test_data)
    time.sleep(1)
    result = pull_queue.get_nowait()

    assert result[0] == test_data[0]
    assert result[1].request_id == test_data[1].request_id
    assert np.array_equal(result[1].tokens.array, test_data[1].tokens.array)
    assert np.allclose(
        result[1].images[0].pixel_values, test_data[1].images[0].pixel_values
    )
