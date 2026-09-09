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
"""Test the max.graph Python bindings for allgather_rms_norm.

The composite op has no MLIR verifier, so this builder is the only place a bad
``group_size`` or an intra-group shape mismatch is rejected with a diagnostic
rather than a Mojo comptime assert.
"""

import pytest
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.nn import Signals

H = 128


def _graph_inputs(
    shapes: list[list[int]], devices: list[DeviceRef]
) -> list[TensorType]:
    """Per-device shards, then per-device gammas (length = that shard's cols)."""
    return [
        TensorType(dtype=DType.bfloat16, shape=shape, device=dev)
        for shape, dev in zip(shapes, devices, strict=True)
    ] + [
        TensorType(dtype=DType.bfloat16, shape=[shape[-1]], device=dev)
        for shape, dev in zip(shapes, devices, strict=True)
    ]


def _build(
    shapes: list[list[int]],
    devices: list[DeviceRef],
    group_size: int | None = None,
) -> tuple[list[TensorValue], list[TensorValue]]:
    num_gpus = len(devices)
    signals = Signals(devices)
    with Graph(
        "allgather_rms_norm",
        input_types=_graph_inputs(shapes, devices)
        + list(signals.input_types()),
    ) as graph:
        normed, residual = ops.allgather_rms_norm(
            inputs=[v.tensor for v in graph.inputs[:num_gpus]],
            signal_buffers=[v.buffer for v in graph.inputs[2 * num_gpus :]],
            gammas=[v.tensor for v in graph.inputs[num_gpus : 2 * num_gpus]],
            epsilon=1e-6,
            group_size=group_size,
        )
        graph.output(*normed, *residual)
        return normed, residual


def test_allgather_rms_norm_full_world_shapes() -> None:
    """Default group_size is the whole world; every device holds all the rows."""
    num_gpus = 4
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    normed, residual = _build([[3, H]] * num_gpus, devices)

    for dev_idx, device in enumerate(devices):
        for out in (normed[dev_idx], residual[dev_idx]):
            assert out.device == device
            assert out.shape == [12, H]


def test_allgather_rms_norm_full_world_unequal_shards() -> None:
    """Axis 0 may differ per shard; the gathered dim is their sum."""
    num_gpus = 4
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    normed, residual = _build([[3, H], [5, H], [1, H], [2, H]], devices)

    for dev_idx in range(num_gpus):
        for out in (normed[dev_idx], residual[dev_idx]):
            assert out.shape == [11, H]


def test_allgather_rms_norm_grouped_unequal_shards() -> None:
    """Unequal heights within a group sum per group."""
    num_gpus = 4
    group_size = 2
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    # group 0: 3 + 5 -> 8 ; group 1: 6 + 1 -> 7.
    normed, residual = _build(
        [[3, H], [5, H], [6, H], [1, H]], devices, group_size=group_size
    )

    expected_rows = [8, 8, 7, 7]
    for dev_idx in range(num_gpus):
        for out in (normed[dev_idx], residual[dev_idx]):
            assert out.shape == [expected_rows[dev_idx], H]


@pytest.mark.parametrize("group_size", [None, 2])
def test_allgather_rms_norm_accepts_reduce_scatter_shards(
    group_size: int | None,
) -> None:
    """The production composition: reduce-scatter residual -> fused all-gather.

    This is what M3 feeds the op, and it is the shape regime a static-shaped
    test cannot reach. ``reducescatter.sum`` ragged-bins axis 0 as
    ``(S + (g-1-lr)) // g``, so over a SYMBOLIC row dim each group-local rank
    gets a structurally distinct ``Dim`` expression that never compares equal to
    its peers'. A builder that requires whole-shape equality within a group
    therefore rejects every real graph while passing every uniform-shaped test.

    Both topologies are covered because the full-world one is the path that
    already ships: this must stay a build-time no-op there.
    """
    num_gpus = 4
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    signals = Signals(devices)
    rows = "total_seq_len"

    with Graph(
        "reduce_scatter_then_allgather_rms_norm",
        input_types=[
            TensorType(dtype=DType.bfloat16, shape=[rows, H], device=dev)
            for dev in devices
        ]
        + [
            TensorType(dtype=DType.bfloat16, shape=[H], device=dev)
            for dev in devices
        ]
        + list(signals.input_types()),
    ) as graph:
        acts = [v.tensor for v in graph.inputs[:num_gpus]]
        gammas = [v.tensor for v in graph.inputs[num_gpus : 2 * num_gpus]]
        sigs = [v.buffer for v in graph.inputs[2 * num_gpus :]]

        shards = ops.reducescatter.sum(acts, sigs, group_size=group_size)
        # Sanity: the shards really are structurally distinct, or this test
        # would pass for the wrong reason.
        width = group_size or num_gpus
        assert shards[0].shape[0] != shards[1].shape[0], (
            "reduce-scatter shards are not ragged; this test no longer covers "
            "the regime it was written for"
        )

        normed, residual = ops.allgather_rms_norm(
            inputs=shards,
            signal_buffers=sigs,
            gammas=gammas,
            epsilon=1e-6,
            group_size=group_size,
        )
        graph.output(*normed, *residual)

    # Gathering a group's own shards reassembles that group's rows exactly.
    for dev_idx in range(num_gpus):
        group_start = (dev_idx // width) * width
        expected = shards[group_start].shape[0]
        for other in range(group_start + 1, group_start + width):
            expected = expected + shards[other].shape[0]
        for out in (normed[dev_idx], residual[dev_idx]):
            assert out.shape == [expected, H]


def test_allgather_rms_norm_grouped_shapes() -> None:
    """The gathered dim sums the GROUP's shards, not the world's.

    Shapes are hard-coded rather than recomputed from the implementation's own
    formula: 8 devices as 2 groups of 4 with different per-group heights, so a
    world-wide sum gives a visibly wrong shape.
    """
    num_gpus = 8
    group_size = 4
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    # group 0: 4 x 2 rows -> 8 gathered; group 1: 4 x 3 rows -> 12 gathered.
    shapes = [[2, H]] * group_size + [[3, H]] * group_size
    expected_rows = [8] * group_size + [12] * group_size

    normed, residual = _build(shapes, devices, group_size=group_size)
    for dev_idx, device in enumerate(devices):
        for out in (normed[dev_idx], residual[dev_idx]):
            assert out.device == device
            assert out.shape == [expected_rows[dev_idx], H]


def test_allgather_rms_norm_group_size_must_divide_inputs() -> None:
    """A group that does not tile the device list is rejected."""
    num_gpus = 6
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    with pytest.raises(
        ValueError,
        match=r"group_size to evenly divide the number of input tensors",
    ):
        _build([[8, H]] * num_gpus, devices, group_size=4)


def test_allgather_rms_norm_group_size_one_rejected() -> None:
    """group_size=1 is a no-op gather; the kernel asserts ngpus >= 2."""
    num_gpus = 4
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    with pytest.raises(ValueError, match=r"group_size to be at least 2"):
        _build([[8, H]] * num_gpus, devices, group_size=1)


def test_allgather_rms_norm_non_gathered_dim_mismatch_within_group() -> None:
    """Non-gathered dims must agree WITHIN a group (axis 0 stays exempt)."""
    num_gpus = 4
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    shapes = [[8, H], [8, H + 1], [8, H], [8, H]]
    with pytest.raises(
        ValueError,
        match=r"same shape in all dimensions except axis 0",
    ):
        _build(shapes, devices, group_size=2)


def test_allgather_rms_norm_differing_shapes_across_groups_ok() -> None:
    """DP replicas are independent collectives, so their shapes may differ.

    The pre-grouping builder compared every input against ``inputs[0]``
    world-wide, which rejects this outright.
    """
    num_gpus = 4
    group_size = 2
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    shapes = [[8, H], [8, H], [4, H], [4, H]]

    normed, residual = _build(shapes, devices, group_size=group_size)
    expected_rows = [16, 16, 8, 8]
    for dev_idx in range(num_gpus):
        for out in (normed[dev_idx], residual[dev_idx]):
            assert out.shape == [expected_rows[dev_idx], H]


def test_allgather_rms_norm_rank_mismatch_rejected() -> None:
    """Rank must match across the whole world, not just within a group."""
    num_gpus = 4
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    shapes = [[8, H], [8, H], [4, H, 1], [4, H, 1]]
    with pytest.raises(ValueError, match=r"same rank across all input tensors"):
        _build(shapes, devices, group_size=2)


def test_allgather_rms_norm_rank_three_rejected() -> None:
    """The fused kernel indexes rows/cols directly, so 2D only."""
    num_gpus = 2
    devices = [DeviceRef.GPU(id=i) for i in range(num_gpus)]
    with pytest.raises(ValueError, match=r"2D-only"):
        _build([[4, H, 1]] * num_gpus, devices)
