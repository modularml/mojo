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

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType
from max.nn import Linear


def test_execution_trace_capture_replay() -> None:
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    accelerator = Accelerator()
    session = InferenceSession(devices=[accelerator])

    with Graph(
        "execution_trace_capture",
        input_types=[TensorType(DType.float32, [4], device=DeviceRef.GPU(0))],
    ) as graph:
        graph.output(graph.inputs[0].tensor + 1)

    model = session.load(graph)
    graph_key = 1
    input_tensor = Buffer.from_numpy(np.arange(4, dtype=np.float32)).to(
        model.input_devices[0]
    )

    (baseline,) = model.execute(input_tensor)
    np.testing.assert_allclose(
        baseline.to_numpy(), np.arange(4, dtype=np.float32) + 1
    )

    (captured_output,) = model.capture(graph_key, input_tensor)
    model.replay(graph_key, input_tensor)

    # (captured_output,) = model.execute(input_tensor)
    np.testing.assert_allclose(
        captured_output.to_numpy(), np.arange(4, dtype=np.float32) + 1
    )

    # Replay with original input values and verify output.
    model.replay(graph_key, input_tensor)
    np.testing.assert_allclose(
        captured_output.to_numpy(), np.arange(4, dtype=np.float32) + 1
    )

    # Update input in-place and replay to verify the graph uses updated values.
    updated_values = Buffer.from_numpy(np.arange(4, dtype=np.float32) + 3).to(
        model.input_devices[0]
    )
    input_tensor.inplace_copy_from(updated_values)

    model.replay(graph_key, input_tensor)
    np.testing.assert_allclose(
        captured_output.to_numpy(), np.arange(4, dtype=np.float32) + 4
    )


def test_same_shapes_different_input_buffers() -> None:
    """Test that caller-provided graph keys isolate input-buffer signatures."""
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    accelerator = Accelerator()
    session = InferenceSession(devices=[accelerator])

    # Create a simple graph that adds 10 to the input
    with Graph(
        "same_shapes_test",
        input_types=[TensorType(DType.float32, [4], device=DeviceRef.GPU(0))],
    ) as graph:
        graph.output(graph.inputs[0].tensor + 10)

    model = session.load(graph)
    graph_key1 = 11
    graph_key2 = 12

    # Create two input buffers with the same shape but different values and underlying memory
    input_tensor1 = Buffer.from_numpy(
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    ).to(model.input_devices[0])
    input_tensor2 = Buffer.from_numpy(
        np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    ).to(model.input_devices[0])

    # Capture with distinct caller keys so each buffer signature is isolated.
    (captured_output1,) = model.capture(graph_key1, input_tensor1)
    (captured_output2,) = model.capture(graph_key2, input_tensor2)

    # Replay both graphs to populate output buffers
    model.replay(graph_key1, input_tensor1)
    model.replay(graph_key2, input_tensor2)

    # Verify each replay wrote to its own output buffer
    np.testing.assert_allclose(
        captured_output1.to_numpy(),
        np.array([11.0, 12.0, 13.0, 14.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        captured_output2.to_numpy(),
        np.array([15.0, 16.0, 17.0, 18.0], dtype=np.float32),
    )

    # Replay input1 again and verify only captured_output1 is updated
    model.replay(graph_key1, input_tensor1)
    np.testing.assert_allclose(
        captured_output1.to_numpy(),
        np.array([11.0, 12.0, 13.0, 14.0], dtype=np.float32),
    )
    # captured_output2 should remain unchanged
    np.testing.assert_allclose(
        captured_output2.to_numpy(),
        np.array([15.0, 16.0, 17.0, 18.0], dtype=np.float32),
    )

    # Update input1 in-place and replay to verify the graph reads from the
    # correct buffer.
    updated_values1 = Buffer.from_numpy(
        np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    ).to(model.input_devices[0])
    input_tensor1.inplace_copy_from(updated_values1)

    model.replay(graph_key1, input_tensor1)
    np.testing.assert_allclose(
        captured_output1.to_numpy(),
        np.array([20.0, 30.0, 40.0, 50.0], dtype=np.float32),
    )
    # captured_output2 should still be unchanged
    np.testing.assert_allclose(
        captured_output2.to_numpy(),
        np.array([15.0, 16.0, 17.0, 18.0], dtype=np.float32),
    )


def test_graph_capture_rejects_unsafe_key_reuse() -> None:
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    accelerator = Accelerator()
    session = InferenceSession(devices=[accelerator])

    with Graph(
        "unsafe_graph_key_reuse_test",
        input_types=[TensorType(DType.float32, [4], device=DeviceRef.GPU(0))],
    ) as graph:
        graph.output(graph.inputs[0].tensor + 10)

    model = session.load(graph)
    graph_key = 21
    input_tensor1 = Buffer.from_numpy(
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    ).to(model.input_devices[0])
    input_tensor2 = Buffer.from_numpy(
        np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    ).to(model.input_devices[0])

    model.capture(graph_key, input_tensor1)

    with pytest.raises(RuntimeError, match="Unsafe graph key reuse"):
        model.replay(graph_key, input_tensor2)

    with pytest.raises(RuntimeError, match="Graph keys already captured"):
        model.capture(graph_key, input_tensor2)


def test_replay_with_external_allocations() -> None:
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    accelerator = Accelerator()
    session = InferenceSession(devices=[accelerator])

    # Use Linear layer which internally uses matmul with transpose_b=True.
    # Shape (M=65, N=6144, K=4096) with bfloat16 triggers SM100 dispatch
    # for native Mojo kernels (not vendor BLAS), which is required for
    # CUDA stream capture and launch trace verification.
    # Dimensions mapping: M=sequence_length, K=in_features, N=out_features
    sequence_length = 65  # M=65-81 range has tuning config for this shape
    in_features = 4096
    out_features = 6144  # (N=6144, K=4096) has tuning config in SM100 dispatch

    max_linear = Linear(
        in_dim=in_features,
        out_dim=out_features,
        dtype=DType.bfloat16,
        has_bias=False,
        device=DeviceRef.GPU(),
    )

    # Initialize weights with random bfloat16 values using torch
    weight_tensor = torch.randn(out_features, in_features, dtype=torch.bfloat16)
    max_linear.load_state_dict({"weight": weight_tensor})

    with Graph(
        "buffer_reuse_test",
        input_types=[
            TensorType(
                DType.bfloat16,
                [sequence_length, in_features],
                device=DeviceRef.GPU(),
            )
        ],
    ) as graph:
        graph.output(max_linear(graph.inputs[0].tensor))

    model = session.load(graph, weights_registry=max_linear.state_dict())
    graph_key = 1

    # Create input buffer using torch for bfloat16 support
    input_tensor = torch.randn(
        sequence_length, in_features, dtype=torch.bfloat16, device="cuda"
    )
    input_buf = Buffer.from_dlpack(input_tensor)

    results = model.capture(graph_key, input_buf)
    model.replay(graph_key, input_buf)

    external_buffers = []
    for _ in range(10):
        external_buffers.append(
            Buffer(DType.float32, [256, 256], device=model.input_devices[0])
        )
    accelerator.synchronize()

    external_buffers.clear()
    accelerator.synchronize()

    for _ in range(10):
        external_buffers.append(
            Buffer(DType.float32, [256, 256], device=model.input_devices[0])
        )
    accelerator.synchronize()

    del model
    accelerator.synchronize()

    del external_buffers
    accelerator.synchronize()

    del results
    accelerator.synchronize()

    del input_tensor
    del input_buf
    accelerator.synchronize()


def test_debug_verify_replay() -> None:
    """Test that debug_verify_replay validates captured graphs match eager execution."""
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    accelerator = Accelerator()
    accelerator.synchronize()

    session = InferenceSession(devices=[accelerator])

    # Create a simple graph that adds 2 to the input
    with Graph(
        "debug_verify_test",
        input_types=[TensorType(DType.float32, [4], device=DeviceRef.GPU(0))],
    ) as graph:
        graph.output(graph.inputs[0].tensor + 2)

    model = session.load(graph)
    graph_key = 1
    input_tensor = Buffer.from_numpy(np.arange(4, dtype=np.float32)).to(
        model.input_devices[0]
    )

    # Capture the graph
    model.capture(graph_key, input_tensor)

    verify_input = Buffer.from_numpy(np.arange(4, dtype=np.float32) + 10).to(
        model.input_devices[0]
    )

    # Verify that the captured graph matches eager execution even when the
    # verification run uses a different input buffer.
    model.debug_verify_replay(graph_key, verify_input)

    # Verify the captured output is still correct after verification
    model.replay(graph_key, input_tensor)
    (output,) = model.execute(input_tensor)
    np.testing.assert_allclose(
        output.to_numpy(), np.arange(4, dtype=np.float32) + 2
    )


def test_release_captured_graph_allows_recapture_and_blocks_replay() -> None:
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    accelerator = Accelerator()
    session = InferenceSession(devices=[accelerator])

    with Graph(
        "release_capture_test",
        input_types=[TensorType(DType.float32, [4], device=DeviceRef.GPU(0))],
    ) as graph:
        graph.output(graph.inputs[0].tensor + 5)

    model = session.load(graph)
    graph_key = 7
    input_tensor = Buffer.from_numpy(np.arange(4, dtype=np.float32)).to(
        model.input_devices[0]
    )

    (captured_output,) = model.capture(graph_key, input_tensor)
    model.replay(graph_key, input_tensor)
    np.testing.assert_allclose(
        captured_output.to_numpy(), np.arange(4, dtype=np.float32) + 5
    )

    del captured_output
    model.release_captured_graph(graph_key)

    with pytest.raises(RuntimeError, match="No captured graph"):
        model.replay(graph_key, input_tensor)

    (recaptured_output,) = model.capture(graph_key, input_tensor)
    model.replay(graph_key, input_tensor)
    np.testing.assert_allclose(
        recaptured_output.to_numpy(), np.arange(4, dtype=np.float32) + 5
    )


def test_release_captured_graph_unknown_key_is_noop() -> None:
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    accelerator = Accelerator()
    session = InferenceSession(devices=[accelerator])

    with Graph(
        "release_capture_unknown_key_test",
        input_types=[TensorType(DType.float32, [4], device=DeviceRef.GPU(0))],
    ) as graph:
        graph.output(graph.inputs[0].tensor + 1)

    model = session.load(graph)

    # Releasing on a model with no captures yet is a no-op.
    model.release_captured_graph(99)

    input_tensor = Buffer.from_numpy(np.arange(4, dtype=np.float32)).to(
        model.input_devices[0]
    )
    model.capture(1, input_tensor)

    # Releasing an unknown key after some captures exist is also a no-op.
    model.release_captured_graph(99)

    # The original capture is unaffected.
    model.replay(1, input_tensor)


def test_capture_releases_input_dlpack_producer() -> None:
    """Regression test for GEX-3732.

    `ModelHandle::capture`, `replay`, and `debug_verify_replay` route
    every input buffer through `collectInputBuffers`, which takes a
    refcounted `DeviceBufferRef` from `Buffer::getInputStorage()`. Prior to
    GEX-3732 that path instead allocated a `DeviceMemory` view per input
    and never freed it, so each call leaked a handle to the underlying
    `DeviceBuffer`. That handle transitively pinned the DLPack capsule's
    destructor — and through it, the Python `DLManagedTensor` producer —
    alive indefinitely.

    This test wraps a `torch.Tensor` as the DLPack producer, routes it
    through all three engine paths, then asserts that dropping the
    user-visible references actually releases the producer. With the leak
    in place the `weakref` would still resolve.
    """
    if accelerator_count() == 0:
        pytest.skip("GPU not available")

    accelerator = Accelerator()
    session = InferenceSession(devices=[accelerator])

    with Graph(
        "input_buffer_release_test",
        input_types=[TensorType(DType.float32, [4], device=DeviceRef.GPU(0))],
    ) as graph:
        graph.output(graph.inputs[0].tensor + 1)

    model = session.load(graph)

    producer = torch.zeros(4, dtype=torch.float32, device="cuda:0")
    producer_ref = weakref.ref(producer)

    input_buffer = Buffer.from_dlpack(producer)

    # Exercise all three graph capture functions.
    (output,) = model.capture(0, input_buffer)
    model.replay(0, input_buffer)
    model.debug_verify_replay(0, input_buffer)

    # Drop every user-visible reference. The buffer refs each handler
    # collected are released when they go out of scope, so the ref chain back
    # to `producer` unwinds: DeviceBufferRef drops -> ~DeviceBuffer fires ->
    # DLPack destructor lambda runs -> DLManagedTensor deleter releases the
    # producer.
    del output
    model.release_captured_graph(0)
    del input_buffer
    del producer
    gc.collect()
    accelerator.synchronize()

    assert producer_ref() is None, (
        "Input DLPack producer leaked through "
        "model.capture/replay/debug_verify_replay."
    )
