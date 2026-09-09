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
"""Test the max.graph Python bindings."""

from __future__ import annotations

import numpy as np
import pytest
from max.driver import Buffer, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DevicePlacementPolicy, DeviceRef, Graph, TensorType, ops

device_ref = DeviceRef.GPU() if accelerator_count() > 0 else DeviceRef.CPU()


@pytest.mark.parametrize(
    "input,updates,indices,axis,expected",
    [
        (
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [[1.1, 2.2], [3.3, 4.4]],
            [[0, 1], [3, 2]],
            0,
            [[1.1, 2.0], [3.0, 2.2], [5.0, 4.4], [3.3, 8.0]],
        ),
        (
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [[1.1, 2.2], [3.3, 4.4]],
            [[1, 0], [0, 1]],
            1,
            [[2.2, 1.1], [3.3, 4.4], [5.0, 6.0], [7.0, 8.0]],
        ),
    ],
)
def test_scatter(
    session: InferenceSession,
    input: list[list[float]],
    updates: list[list[float]],
    indices: list[list[int]],
    axis: int,
    expected: list[list[float]],
) -> None:
    input_np = np.array(input, dtype=np.float32)
    input_type = TensorType(DType.float32, input_np.shape, device_ref)
    with Graph("scatter", input_types=[input_type]) as graph:
        input_val = ops.transfer_to(graph.inputs[0].tensor, DeviceRef.CPU())
        updates_val = ops.constant(
            updates, DType.float32, device=DeviceRef.CPU()
        )
        indices_val = ops.constant(indices, DType.int32, device=DeviceRef.CPU())
        out = ops.scatter(input_val, updates_val, indices_val, axis)
        graph.output(out)

    model = session.load(graph)
    input_tensor = Buffer.from_numpy(input_np).to(model.input_devices[0])

    result = model.execute(input_tensor)[0]
    assert isinstance(result, Buffer)

    np.testing.assert_equal(
        result.to_numpy(), np.array(expected, dtype=np.float32)
    )


@pytest.mark.parametrize(
    "input_data,updates_data,indices_data,expected",
    [
        # 1D scatter_nd
        (
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 20.0],
            [[1], [3]],
            [1.0, 10.0, 3.0, 20.0, 5.0],
        ),
        # 1D scatter_nd with negative indices
        (
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [10.0, 20.0],
            [[-4], [-2]],
            [1.0, 10.0, 3.0, 20.0, 5.0],
        ),
        # 2D scatter_nd with 1D indices
        (
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[10.0, 11.0, 12.0], [13.0, 14.0, 15.0]],
            [[0], [2]],
            [[10.0, 11.0, 12.0], [4.0, 5.0, 6.0], [13.0, 14.0, 15.0]],
        ),
        # 2D scatter_nd with 2D indices
        (
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [10.0, 20.0, 30.0],
            [[0, 1], [1, 2], [2, 0]],
            [[1.0, 10.0, 3.0], [4.0, 5.0, 20.0], [30.0, 8.0, 9.0]],
        ),
        # Empty updates
        (
            [1.0, 2.0, 3.0, 4.0],
            [],
            np.empty((0, 1), dtype=np.int32),
            [1.0, 2.0, 3.0, 4.0],
        ),
    ],
)
def test_scatter_nd(
    session: InferenceSession,
    input_data: list[float] | list[list[float]],
    updates_data: list[float] | list[list[float]],
    indices_data: list[list[int]] | np.ndarray,
    expected: list[float] | list[list[float]],
) -> None:
    """Test scatter_nd operation with various input configurations."""
    input_array = np.array(input_data, dtype=np.float32)
    input_type = TensorType(DType.float32, input_array.shape, device_ref)

    with Graph("scatter_nd", input_types=[input_type]) as graph:
        input_val = graph.inputs[0].tensor
        updates = ops.constant(updates_data, DType.float32, device=device_ref)
        indices = ops.constant(indices_data, DType.int32, device=device_ref)
        out = ops.scatter_nd(input_val, updates, indices)
        graph.output(out)

    model = session.load(graph)
    input_tensor = Buffer.from_numpy(input_array).to(model.input_devices[0])

    result = model.execute(input_tensor)[0]
    assert isinstance(result, Buffer)

    np.testing.assert_equal(
        result.to_numpy(), np.array(expected, dtype=np.float32)
    )


def _reduce_updates(
    np_reduce: np.ufunc, n_idx: int, cols: int, n_targets: int
) -> np.ndarray:
    """Builds update values that keep the reduction exact in float32 in any
    application order: all ones for add; ones with four twos per target row
    for mul (each target scaled by exactly 2**4); small modular integers for
    max/min (order-independent)."""
    if np_reduce is np.add:
        return np.ones((n_idx, cols), dtype=np.float32)
    if np_reduce is np.multiply:
        updates = np.ones((n_idx, cols), dtype=np.float32)
        updates[: 4 * n_targets] = 2.0
        return updates
    values = (np.arange(n_idx, dtype=np.float32) * 37) % 251
    return np.broadcast_to(values[:, None], (n_idx, cols)).copy()


_SCATTER_ND_REDUCE_OPS = [
    (ops.scatter_nd_add, np.add),
    (ops.scatter_nd_mul, np.multiply),
    (ops.scatter_nd_max, np.maximum),
    (ops.scatter_nd_min, np.minimum),
]


def test_scatter_nd_reduce_duplicate_indices(
    session: InferenceSession,
) -> None:
    """Duplicate index vectors must reduce, not race: 8192 update rows
    collide on 4 target rows, so on GPU thousands of threads reduce into the
    same elements concurrently. Update values keep each reduction exact in
    float32 in any application order, so each result must match the serial
    numpy reference exactly. All four reduce ops share one graph so the
    test compiles once.
    """
    rows, cols, n_idx, n_targets = 64, 16, 8192, 4

    input_array = (np.arange(rows * cols, dtype=np.float32) % 7).reshape(
        rows, cols
    )
    indices_data = ((np.arange(n_idx, dtype=np.int32) % n_targets) * 5 + 3)[
        :, None
    ]
    all_updates = [
        _reduce_updates(np_reduce, n_idx, cols, n_targets)
        for _, np_reduce in _SCATTER_ND_REDUCE_OPS
    ]

    input_type = TensorType(DType.float32, input_array.shape, device_ref)
    with Graph("scatter_nd_reduce_dup", input_types=[input_type]) as graph:
        input_val = graph.inputs[0].tensor
        indices = ops.constant(indices_data, DType.int32, device=device_ref)
        graph.output(
            *(
                op(
                    input_val,
                    ops.constant(
                        updates_data, DType.float32, device=device_ref
                    ),
                    indices,
                )
                for (op, _), updates_data in zip(
                    _SCATTER_ND_REDUCE_OPS, all_updates, strict=False
                )
            )
        )

    model = session.load(graph)
    input_tensor = Buffer.from_numpy(input_array).to(model.input_devices[0])

    results = model.execute(input_tensor)
    for (op, np_reduce), updates_data, result in zip(
        _SCATTER_ND_REDUCE_OPS, all_updates, results, strict=False
    ):
        assert isinstance(result, Buffer)
        expected = input_array.copy()
        np_reduce.at(expected, indices_data.ravel(), updates_data)
        np.testing.assert_equal(
            result.to_numpy(), expected, err_msg=op.__name__
        )


_SCATTER_REDUCE_OPS = [
    (ops.scatter_add, np.add),
    (ops.scatter_mul, np.multiply),
    (ops.scatter_max, np.maximum),
    (ops.scatter_min, np.minimum),
]


def test_scatter_reduce_parallel_duplicate_indices(
    session: InferenceSession,
) -> None:
    """The scatter-elements reduce ops run on CPU and split their updates
    across worker threads once the update count exceeds the elementwise
    grain size (32768); duplicate indices must still reduce atomically.
    100k updates collide on 8 target rows; values keep each reduction exact
    in float32 in any application order, so each result must match the
    serial numpy reference exactly. All four reduce ops share one graph so
    the test compiles once.
    """
    rows, n_idx, n_targets = 64, 100_000, 8

    input_array = (np.arange(rows, dtype=np.float32) % 7)[:, None]
    indices_data = ((np.arange(n_idx, dtype=np.int32) % n_targets) * 7 + 1)[
        :, None
    ]
    all_updates = [
        _reduce_updates(np_reduce, n_idx, 1, n_targets)
        for _, np_reduce in _SCATTER_REDUCE_OPS
    ]

    input_type = TensorType(DType.float32, input_array.shape, DeviceRef.CPU())
    with Graph("scatter_reduce_dup", input_types=[input_type]) as graph:
        input_val = graph.inputs[0].tensor
        indices = ops.constant(
            indices_data, DType.int32, device=DeviceRef.CPU()
        )
        graph.output(
            *(
                op(
                    input_val,
                    ops.constant(
                        updates_data, DType.float32, device=DeviceRef.CPU()
                    ),
                    indices,
                    axis=0,
                )
                for (op, _), updates_data in zip(
                    _SCATTER_REDUCE_OPS, all_updates, strict=False
                )
            )
        )

    model = session.load(graph)
    input_tensor = Buffer.from_numpy(input_array).to(model.input_devices[0])

    results = model.execute(input_tensor)
    for (op, np_reduce), updates_data, result in zip(
        _SCATTER_REDUCE_OPS, all_updates, results, strict=False
    ):
        assert isinstance(result, Buffer)
        expected = input_array.copy()
        np_reduce.at(expected[:, 0], indices_data.ravel(), updates_data[:, 0])
        np.testing.assert_equal(
            result.to_numpy(), expected, err_msg=op.__name__
        )


@pytest.mark.skipif(
    accelerator_count() == 0, reason="requires a GPU to test device check"
)
def test_scatter_raises_on_gpu() -> None:
    """ops.scatter raises ValueError at graph construction time on GPU input."""
    with pytest.raises(ValueError, match=r"ops\.scatter"):
        with Graph(
            "scatter_gpu",
            input_types=[
                TensorType(DType.float32, [4, 2], device=DeviceRef.GPU())
            ],
            strict_device_placement=DevicePlacementPolicy.Error,
        ):
            input_val = Graph.current.inputs[0].tensor
            updates = ops.constant(
                [[1.1, 2.2], [3.3, 4.4]], DType.float32, device=DeviceRef.GPU()
            )
            indices = ops.constant(
                [[0, 1], [3, 2]], DType.int32, device=DeviceRef.GPU()
            )
            ops.scatter(input_val, updates, indices, axis=0)
