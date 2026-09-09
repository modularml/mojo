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
"""Tests for MoE, TensorParallelMoE, and ExpertParallelMoE (ModuleV3)."""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest
from max.driver import Device
from max.experimental import functional as F
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.nn.common_layers.moe import (
    MoE,
    TensorParallelMoE,
)
from max.experimental.sharding import (
    DeviceMesh,
    Partial,
    PlacementMapping,
    Replicated,
)
from max.experimental.tensor import Tensor

# Small dimensions for fast graph-trace tests.
_HIDDEN_DIM = 256
_NUM_EXPERTS = 8
_NUM_EXPERTS_PER_TOKEN = 2
_MOE_DIM = 128
_SEQ_LEN = 4


def _make_fake_gpu(id: int = 0) -> MagicMock:
    label = "gpu"
    fake = MagicMock(spec=Device)
    fake.id = id
    fake.label = label
    fake.__eq__ = MagicMock(  # type: ignore[method-assign]
        side_effect=lambda other: (
            getattr(other, "id", None) == id
            and getattr(other, "label", None) == label
        )
    )
    fake.__hash__ = MagicMock(return_value=hash((id, label)))  # type: ignore[method-assign]
    return fake


@pytest.fixture(autouse=True)
def mock_accelerator() -> Iterator[MagicMock]:
    with patch("max.graph.type.Accelerator") as mock:
        mock.side_effect = _make_fake_gpu
        yield mock


def test_layer(mock_accelerator: MagicMock) -> None:
    """Traces a single-device MoE in a lazy context and verifies output shape and device."""
    devices = [mock_accelerator()]

    with F.lazy():
        layer = MoE(
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
            moe_dim=_MOE_DIM,
        ).to(devices[0])

        gate_up_proj = layer.gate_up_proj
        assert len(gate_up_proj) == 1
        assert list(gate_up_proj[0].shape) == [
            _NUM_EXPERTS,
            2 * _MOE_DIM,
            _HIDDEN_DIM,
        ]
        assert gate_up_proj[0].device == devices[0]

        down_proj = layer.down_proj
        assert len(down_proj) == 1
        assert list(down_proj[0].shape) == [_NUM_EXPERTS, _HIDDEN_DIM, _MOE_DIM]
        assert down_proj[0].device == devices[0]

        x = Tensor.zeros([_SEQ_LEN, _HIDDEN_DIM], device=devices[0])
        out = layer(x)

    assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
    assert out.device == devices[0]


def test_tensor_parallel_layer(mock_accelerator: MagicMock) -> None:
    """Traces a TensorParallelMoE in a lazy context and verifies output mapping."""
    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        num_devices = len(devices)
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))
        replicated_mapping = PlacementMapping(mesh, (Replicated(),))

        layer = TensorParallelMoE(
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            num_experts_per_token=_NUM_EXPERTS_PER_TOKEN,
            moe_dim=_MOE_DIM,
        ).to(mesh)

        # Each device holds a slice of every expert: 2*moe_dim is split along
        # the output axis by num_devices. The weights are a per-device bundle
        # of single-device tensors, one per mesh device.
        gate_up_proj = layer.gate_up_proj
        assert len(gate_up_proj) == num_devices
        for i, shard in enumerate(gate_up_proj):
            assert list(shard.shape) == [
                _NUM_EXPERTS,
                2 * _MOE_DIM // num_devices,
                _HIDDEN_DIM,
            ]
            assert shard.device == devices[i]

        # down_proj weight is [hidden_dim, moe_dim]; shard_and_stack splits the
        # last axis (moe_dim, the contraction dim) by num_devices.
        down_proj = layer.down_proj
        assert len(down_proj) == num_devices
        for i, shard in enumerate(down_proj):
            assert list(shard.shape) == [
                _NUM_EXPERTS,
                _HIDDEN_DIM,
                _MOE_DIM // num_devices,
            ]
            assert shard.device == devices[i]

        x = Tensor.zeros([_SEQ_LEN, _HIDDEN_DIM], device=replicated_mapping)
        out = layer(x)

    assert list(out.shape) == [_SEQ_LEN, _HIDDEN_DIM]
    assert out.mapping.mesh == mesh
    # Output is a partial sum that must be all-reduced across TP ranks.
    assert out.mapping.to_placements() == (Partial(),)
