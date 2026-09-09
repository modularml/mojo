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
"""Test the max.graph Python bindings for reduce_scatter_rms_norm."""

from collections.abc import Callable

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn import Signals

H = 128


def _graph_inputs(
    shapes: list[list[int]],
    devices: list[DeviceRef],
    signals: Signals,
    residual_shapes: list[list[int]] | None = None,
) -> list[TensorType]:
    """Per-device tensor input types: activations, gammas, then residuals.

    The residual block appears only when a case needs residuals to differ from
    the activations; the caller appends the signal buffers after these.
    """
    types = [
        TensorType(dtype=DType.bfloat16, shape=shape, device=dev)
        for shape, dev in zip(shapes, devices, strict=True)
    ] + [
        TensorType(dtype=DType.bfloat16, shape=[H], device=dev)
        for dev in devices
    ]
    if residual_shapes is not None:
        types += [
            TensorType(dtype=DType.bfloat16, shape=shape, device=dev)
            for shape, dev in zip(residual_shapes, devices, strict=True)
        ]
    return types


def _build(
    shapes: list[list[int]],
    devices: list[DeviceRef],
    group_size: int | None = None,
    residual_shapes: list[list[int]] | None = None,
    residual_transform: Callable[[list[TensorValue]], list[TensorValue]]
    | None = None,
) -> tuple[list[TensorValue], list[TensorValue]]:
    num_gpus = len(devices)
    signals = Signals(devices)
    with Graph(
        "reduce_scatter_rms_norm",
        input_types=_graph_inputs(shapes, devices, signals, residual_shapes)
        + list(signals.input_types()),
    ) as graph:
        inputs = [v.tensor for v in graph.inputs[:num_gpus]]
        # Shapes only: the residual need only match its input's
        # shape/dtype/device, so reuse them. Numerics live in the exec test.
        residuals = inputs
        num_tensor_blocks = 2
        if residual_shapes is not None:
            residuals = [
                v.tensor for v in graph.inputs[2 * num_gpus : 3 * num_gpus]
            ]
            num_tensor_blocks = 3
        if residual_transform is not None:
            residuals = residual_transform(residuals)
        normed, residual = ops.reduce_scatter_rms_norm(
            inputs=inputs,
            signal_buffers=[
                v.buffer for v in graph.inputs[num_tensor_blocks * num_gpus :]
            ],
            gammas=[v.tensor for v in graph.inputs[num_gpus : 2 * num_gpus]],
            epsilon=1e-6,
            residuals=residuals,
            group_size=group_size,
        )
        graph.output(*normed, *residual)
        return normed, residual


def _build_without_residual(
    shapes: list[list[int]], devices: list[DeviceRef]
) -> tuple[Graph, list[TensorValue], list[TensorValue]]:
    """Build the op with `residuals` omitted, returning the graph to inspect."""
    num_gpus = len(devices)
    signals = Signals(devices)
    with Graph(
        "reduce_scatter_rms_norm_no_residual",
        input_types=_graph_inputs(shapes, devices, signals)
        + list(signals.input_types()),
    ) as graph:
        normed, residual = ops.reduce_scatter_rms_norm(
            inputs=[v.tensor for v in graph.inputs[:num_gpus]],
            signal_buffers=[v.buffer for v in graph.inputs[2 * num_gpus :]],
            gammas=[v.tensor for v in graph.inputs[num_gpus : 2 * num_gpus]],
            epsilon=1e-6,
        )
        graph.output(*normed, *residual)
        return graph, normed, residual


def test_reduce_scatter_rms_norm_without_residual_shapes() -> None:
    """Omitting the residual is legal and does not change the shard binning."""
    num_gpus = 4
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    _, normed, residual = _build_without_residual([[9, H]] * num_gpus, devices)

    expected_rows = [3, 2, 2, 2]
    for dev_idx, device in enumerate(devices):
        for out in (normed[dev_idx], residual[dev_idx]):
            assert out.device == device
            assert out.shape == [expected_rows[dev_idx], H]


def test_reduce_scatter_rms_norm_has_residual_attr_tracks_the_argument() -> (
    None
):
    """`has_residual` is what the kernel gates on, so gate the attribute.

    The operand slots are filled either way (the op's variadic groups must
    match in size), so the operands cannot distinguish the two forms -- only
    this attribute can, which is exactly why it is asserted rather than the
    presence of a residual operand.

    It is asserted as present-when-true rather than `= false` when off: it is
    a `DefaultValuedAttr` defaulting to false, so MLIR elides it at that value
    and the handler reads the default.
    """
    num_gpus = 2
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]

    graph, _, _ = _build_without_residual([[8, H]] * num_gpus, devices)
    assert "has_residual" not in str(graph)

    signals = Signals(devices)
    with Graph(
        "reduce_scatter_rms_norm_with_residual",
        input_types=_graph_inputs([[8, H]] * num_gpus, devices, signals)
        + list(signals.input_types()),
    ) as with_res:
        inputs = [v.tensor for v in with_res.inputs[:num_gpus]]
        normed, residual = ops.reduce_scatter_rms_norm(
            inputs=inputs,
            signal_buffers=[v.buffer for v in with_res.inputs[2 * num_gpus :]],
            gammas=[v.tensor for v in with_res.inputs[num_gpus : 2 * num_gpus]],
            epsilon=1e-6,
            residuals=inputs,
        )
        with_res.output(*normed, *residual)
    assert "has_residual = true" in str(with_res)


def test_reduce_scatter_rms_norm_full_world_shapes() -> None:
    """Default group_size is the whole world; rows split across all devices."""
    num_gpus = 4
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    normed, residual = _build([[9, H]] * num_gpus, devices)

    # 9 rows over 4 devices -> 3,2,2,2 (remainder to the low ranks).
    expected_rows = [3, 2, 2, 2]
    for dev_idx, device in enumerate(devices):
        for out in (normed[dev_idx], residual[dev_idx]):
            assert out.device == device
            assert out.shape == [expected_rows[dev_idx], H]


def test_reduce_scatter_rms_norm_grouped_ragged() -> None:
    """Grouped ragged binning keys off the GROUP, not the device count.

    Rows are hard-coded rather than recomputed from the implementation's own
    formula: 8 devices as 2 groups of 4, with different per-group row counts, so
    a global-rank or world-sized divisor gives visibly wrong shapes.
    """
    num_gpus = 8
    group_size = 4
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    shapes = [[5, H]] * group_size + [[3, H]] * group_size
    # group 0 (5 rows over 4) -> 2,1,1,1 ; group 1 (3 rows over 4) -> 1,1,1,0
    expected_rows = [2, 1, 1, 1, 1, 1, 1, 0]

    normed, residual = _build(shapes, devices, group_size=group_size)
    for dev_idx, device in enumerate(devices):
        for out in (normed[dev_idx], residual[dev_idx]):
            assert out.device == device
            assert out.shape == [expected_rows[dev_idx], H]


def test_reduce_scatter_rms_norm_group_size_must_divide_inputs() -> None:
    """A group that does not tile the device list is rejected."""
    num_gpus = 6
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    with pytest.raises(
        ValueError,
        match=r"group_size to evenly divide the number of input tensors",
    ):
        _build([[8, H]] * num_gpus, devices, group_size=4)


def test_reduce_scatter_rms_norm_group_size_one_rejected() -> None:
    """group_size=1 is a no-op reduction; the kernel asserts ngpus >= 2."""
    num_gpus = 4
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    with pytest.raises(ValueError, match=r"group_size to be at least 2"):
        _build([[8, H]] * num_gpus, devices, group_size=1)


def test_reduce_scatter_rms_norm_shape_mismatch_within_group() -> None:
    """Shapes must agree WITHIN a group even though groups may differ."""
    num_gpus = 4
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    shapes = [[8, H], [7, H], [8, H], [8, H]]
    with pytest.raises(
        ValueError,
        match=r"same shape across all input tensors in each group",
    ):
        _build(shapes, devices, group_size=2)


def test_reduce_scatter_rms_norm_differing_shapes_across_groups_ok() -> None:
    """DP replicas are independent collectives, so their shapes may differ.

    The pre-grouping builder compared every input against ``inputs[0]``, which
    rejects this outright.
    """
    num_gpus = 4
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    shapes = [[8, H], [8, H], [4, H], [4, H]]
    normed, residual = _build(shapes, devices, group_size=2)

    expected_rows = [4, 4, 2, 2]
    for dev_idx in range(num_gpus):
        for out in (normed[dev_idx], residual[dev_idx]):
            assert out.shape == [expected_rows[dev_idx], H]


def test_reduce_scatter_rms_norm_rank_mismatch_rejected() -> None:
    """Rank must match across the whole world, not just within a group."""
    num_gpus = 4
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    shapes = [[8, H], [8, H], [4, H, 1], [4, H, 1]]
    with pytest.raises(ValueError, match=r"same rank across all input tensors"):
        _build(shapes, devices, group_size=2)


def test_reduce_scatter_rms_norm_residual_count_mismatch() -> None:
    """One residual per device; a short list is not a partial fold."""
    num_gpus = 4
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    with pytest.raises(ValueError, match=r"number of residuals"):
        _build(
            [[8, H]] * num_gpus,
            devices,
            residual_transform=lambda residuals: residuals[:-1],
        )


def test_reduce_scatter_rms_norm_shard_shaped_residual_rejected() -> None:
    """The footgun the kernel-side extent guard exists for.

    The residual is indexed by GLOBAL row, so a shard-shaped one runs past its
    own storage on every rank whose shard does not start at row 0.
    """
    num_gpus = 4
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    with pytest.raises(ValueError, match=r"residual to match its"):
        _build(
            [[8, H]] * num_gpus,
            devices,
            residual_shapes=[[8 // num_gpus, H]] * num_gpus,
        )


def test_reduce_scatter_rms_norm_residual_device_mismatch() -> None:
    """Each rank adds its OWN residual, so a cross-device pairing is wrong.

    Shapes are uniform, so reversing changes only the device and the earlier
    shape/dtype check cannot mask the one under test.
    """
    num_gpus = 4
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    with pytest.raises(ValueError, match=r"residual to live on its"):
        _build(
            [[8, H]] * num_gpus,
            devices,
            residual_transform=lambda residuals: residuals[::-1],
        )
