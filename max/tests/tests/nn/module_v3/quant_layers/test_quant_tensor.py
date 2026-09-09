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

"""CPU-only checks for the quantized tensor wrappers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.sharding import DeviceMesh, Sharded
from max.experimental.tensor import Tensor
from max.pipelines.architectures.deepseekV3_modulev3.layers.quant_tensor import (
    FP8BlockTensor,
    NVFP4Activation,
    NVFP4Tensor,
    QTensor,
    all_fp8_block,
    all_nvfp4,
)


def test_init_stores_tensors_and_block_size() -> None:
    """``__init__`` keeps the passed tensors and block size verbatim."""
    with F.lazy():
        data = Tensor.zeros((4, 8), dtype=DType.float8_e4m3fn)
        weight_scale_inv = Tensor.zeros((1, 1), dtype=DType.float32)
        qt = FP8BlockTensor(
            data=data, weight_scale_inv=weight_scale_inv, block_size=(128, 128)
        )

        assert qt.data is data
        assert qt.weight_scale_inv is weight_scale_inv
        assert qt.block_size == (128, 128)


def test_init_default_block_size() -> None:
    """The default block size is ``(128, 128)``."""
    with F.lazy():
        qt = FP8BlockTensor(
            data=Tensor.zeros((4, 8), dtype=DType.float8_e4m3fn),
            weight_scale_inv=Tensor.zeros((1, 1), dtype=DType.float32),
        )
        assert qt.block_size == (128, 128)


def test_zeros_dtypes_are_kernel_fixed() -> None:
    """``zeros()`` pins the FP8 data and float32 scale dtypes."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((256, 512))
        assert qt.data.dtype == DType.float8_e4m3fn
        assert qt.weight_scale_inv.dtype == DType.float32


def test_zeros_data_shape_matches_request() -> None:
    """``zeros()`` data buffer has exactly the requested shape."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((256, 512))
        assert list(qt.data.shape) == [256, 512]


def test_zeros_scale_shape_divisible() -> None:
    """Scale shape is ``shape / block_size`` when dims divide evenly."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((256, 512), block_size=(128, 128))
        # 256 / 128 == 2, 512 / 128 == 4
        assert list(qt.weight_scale_inv.shape) == [2, 4]
        assert qt.block_size == (128, 128)


def test_zeros_scale_shape_uses_ceildiv() -> None:
    """Non-divisible dims round up: ``ceil(rows / m)``, ``ceil(cols / k)``."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((130, 100), block_size=(128, 128))
        # ceil(130 / 128) == 2, ceil(100 / 128) == 1
        assert list(qt.weight_scale_inv.shape) == [2, 1]


def test_zeros_custom_block_size() -> None:
    """A non-default block size flows into both block_size and scale shape."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((256, 512), block_size=(64, 256))
        assert qt.block_size == (64, 256)
        # 256 / 64 == 4, 512 / 256 == 2
        assert list(qt.weight_scale_inv.shape) == [4, 2]


def test_zeros_single_block() -> None:
    """A shape smaller than one block still yields a 1x1 scale grid."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((16, 32), block_size=(128, 128))
        assert list(qt.data.shape) == [16, 32]
        assert list(qt.weight_scale_inv.shape) == [1, 1]


def test_is_module_subclass() -> None:
    """QTensor / FP8BlockTensor are Modules for parameter discovery."""
    assert issubclass(QTensor, Module)
    assert issubclass(FP8BlockTensor, QTensor)


def test_local_parameters_discovered() -> None:
    """``data`` and ``weight_scale_inv`` are discoverable as local parameters."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((256, 512))
        local = dict(qt.local_parameters)

        assert set(local) == {"data", "weight_scale_inv"}
        assert local["data"] is qt.data
        assert local["weight_scale_inv"] is qt.weight_scale_inv


def test_block_size_is_not_a_parameter() -> None:
    """``block_size`` is metadata, not a discovered tensor parameter."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((256, 512))
        names = {name for name, _ in qt.parameters}
        assert names == {"data", "weight_scale_inv"}


def test_parameters_discovered_through_parent_module() -> None:
    """Nested in a Module, the inner tensors get qualified ``weight.*`` names."""

    class _Wrapper(Module[[], None]):
        def __init__(self) -> None:
            super().__init__()
            self.weight = FP8BlockTensor.zeros((256, 512))

        def forward(self) -> None:  # pragma: no cover - never called
            raise NotImplementedError

    with F.lazy():
        wrapper = _Wrapper()
        names = {name for name, _ in wrapper.parameters}
        assert names == {"weight.data", "weight.weight_scale_inv"}


def test_forward_raises() -> None:
    """QTensors are data wrappers, not callable layers."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((256, 512))
        with pytest.raises(
            NotImplementedError, match="QTensor is not a callable layer"
        ):
            qt()


def test_forward_method_raises_directly() -> None:
    """Calling ``forward()`` directly raises the same guard."""
    with F.lazy():
        qt = FP8BlockTensor.zeros((256, 512))
        with pytest.raises(
            NotImplementedError, match="QTensor is not a callable layer"
        ):
            qt.forward()


# --------------------------------------------------------------------------- #
# FP8BlockTensor.shard() -- Slice 3
# --------------------------------------------------------------------------- #


def _fake_gpu(id: int = 0) -> Device:
    """Minimal mock GPU device matching the conftest helper."""

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


def test_fp8_shard_row_axis_data_shape() -> None:
    """Sharding on axis 0 (row axis) halves the data row extent per device.

    Stacked weight ``[E, 2*moe_dim, hidden]`` sharded on axis 1 (the moe_dim
    row axis, which is axis 1 of the 3-D stacked tensor).  Here we test a
    simpler 2-D weight, axis 0.
    """
    with F.lazy(), patch("max.graph.type.Accelerator") as mock:
        mock.side_effect = _fake_gpu
        devices = [mock(0), mock(1)]
        num_devices = len(devices)
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))

        # 2-D FP8 weight [256, 256] with (128, 128) blocks.
        qt = FP8BlockTensor.zeros((256, 256), block_size=(128, 128))
        sharded = qt.shard(axis=0, mesh=mesh)

        # data: [256, 256] → each device [128, 256]
        assert sharded.data.placements == (Sharded(axis=0),)
        for shard in sharded.data.local_shards:
            assert list(shard.shape) == [256 // num_devices, 256]

        # weight_scale_inv: [2, 2] → each device [1, 2]
        assert sharded.weight_scale_inv.placements == (Sharded(axis=0),)
        for shard in sharded.weight_scale_inv.local_shards:
            assert list(shard.shape) == [2 // num_devices, 2]

        # block_size preserved
        assert sharded.block_size == (128, 128)


def test_fp8_shard_col_axis_data_shape() -> None:
    """Sharding on axis -1 (col axis) halves the data col extent per device.

    This mirrors the TP ``down_proj`` shard: ``[E, hidden, moe_dim]`` sharded
    on axis -1.  We test a 2-D proxy ``[256, 256]``.
    """
    with F.lazy(), patch("max.graph.type.Accelerator") as mock:
        mock.side_effect = _fake_gpu
        devices = [mock(0), mock(1)]
        num_devices = len(devices)
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))

        qt = FP8BlockTensor.zeros((256, 256), block_size=(128, 128))
        sharded = qt.shard(axis=-1, mesh=mesh)

        # data last dim: 256 → 128 per device
        for shard in sharded.data.local_shards:
            assert list(shard.shape) == [256, 256 // num_devices]

        # weight_scale_inv last dim: 2 → 1 per device
        for shard in sharded.weight_scale_inv.local_shards:
            assert list(shard.shape) == [2, 2 // num_devices]


def test_fp8_shard_3d_down_proj_shape() -> None:
    """Sharding a 3-D ``[E, H, moe_dim]`` weight on axis -1 (TP down_proj).

    Uses 2 devices and (128, 128) blocks.  ``hidden=256``, ``moe_dim=512``.
    After shard on axis=-1 each device should hold ``[E, H, moe_dim//2]``.
    """
    with F.lazy(), patch("max.graph.type.Accelerator") as mock:
        mock.side_effect = _fake_gpu
        devices = [mock(0), mock(1)]
        num_devices = len(devices)
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))

        num_experts = 4
        hidden = 256
        moe_dim = 512

        # FP8BlockTensor.zeros only accepts 2-D shapes; build directly.
        data = Tensor.zeros(
            (num_experts, hidden, moe_dim), dtype=DType.float8_e4m3fn
        )
        # weight_scale_inv shape: [E, ceil(H/128), ceil(moe_dim/128)] = [4, 2, 4]
        weight_scale_inv = Tensor.zeros(
            (num_experts, hidden // 128, moe_dim // 128), dtype=DType.float32
        )
        qt = FP8BlockTensor(
            data=data, weight_scale_inv=weight_scale_inv, block_size=(128, 128)
        )
        sharded = qt.shard(axis=-1, mesh=mesh)

        for shard in sharded.local_shards:
            assert list(shard.data.shape) == [
                num_experts,
                hidden,
                moe_dim // num_devices,
            ]
            assert list(shard.weight_scale_inv.shape) == [
                num_experts,
                hidden // 128,
                moe_dim // 128 // num_devices,
            ]


# --------------------------------------------------------------------------- #
# NVFP4Tensor
# --------------------------------------------------------------------------- #


def test_nvfp4_zeros_dtypes_are_kernel_fixed() -> None:
    """``zeros()`` pins the packed-uint8 data and the two scale dtypes."""
    with F.lazy():
        qt = NVFP4Tensor.zeros((256, 512))
        assert qt.data.dtype == DType.uint8
        assert qt.weight_scale.dtype == DType.float8_e4m3fn
        assert qt.weight_scale_2.dtype == DType.float32
        assert qt.input_scale.dtype == DType.float32


def test_nvfp4_zeros_packs_two_values_per_byte() -> None:
    """``zeros()`` takes the logical shape; ``data`` halves the K extent."""
    with F.lazy():
        qt = NVFP4Tensor.zeros((256, 512))
        assert list(qt.data.shape) == [256, 256]


def test_nvfp4_zeros_scale_grid_uses_logical_columns() -> None:
    """The block-scale grid is derived from the *unpacked* column count."""
    with F.lazy():
        qt = NVFP4Tensor.zeros((256, 512))
        # 256 / 1 == 256 rows, 512 / 16 == 32 blocks along K.
        assert list(qt.weight_scale.shape) == [256, 32]
        assert qt.block_size == (1, 16)


def test_nvfp4_zeros_scale_shape_uses_ceildiv() -> None:
    """A partial trailing block still gets a scale."""
    with F.lazy():
        qt = NVFP4Tensor.zeros((8, 24))
        # ceil(24 / 16) == 2
        assert list(qt.weight_scale.shape) == [8, 2]


def test_nvfp4_zeros_global_scales_are_host_scalars() -> None:
    """``weight_scale_2`` / ``input_scale`` are rank-0 and stay on the host."""
    with F.lazy():
        qt = NVFP4Tensor.zeros((256, 512))
        assert list(qt.weight_scale_2.shape) == []
        assert list(qt.input_scale.shape) == []
        assert qt.weight_scale_2.device.is_host
        assert qt.input_scale.device.is_host


def test_nvfp4_init_stores_tensors_and_block_size() -> None:
    """``__init__`` keeps the passed tensors and block size verbatim."""
    with F.lazy():
        data = Tensor.zeros((4, 8), dtype=DType.uint8)
        weight_scale = Tensor.zeros((4, 1), dtype=DType.float8_e4m3fn)
        weight_scale_2 = Tensor.zeros((), dtype=DType.float32)
        input_scale = Tensor.zeros((), dtype=DType.float32)
        qt = NVFP4Tensor(
            data=data,
            weight_scale=weight_scale,
            weight_scale_2=weight_scale_2,
            input_scale=input_scale,
            block_size=(1, 16),
        )

        assert qt.data is data
        assert qt.weight_scale is weight_scale
        assert qt.weight_scale_2 is weight_scale_2
        assert qt.input_scale is input_scale
        assert qt.block_size == (1, 16)


def test_nvfp4_local_parameters_discovered() -> None:
    """All four leaves are discoverable as local parameters."""
    with F.lazy():
        qt = NVFP4Tensor.zeros((256, 512))
        local = dict(qt.local_parameters)

        assert set(local) == {
            "data",
            "weight_scale",
            "weight_scale_2",
            "input_scale",
        }
        assert local["data"] is qt.data


def test_nvfp4_parameters_discovered_through_parent_module() -> None:
    """Nested in a Module, the leaves get qualified ``weight.*`` names.

    These are exactly the names the NVFP4 weight adapter emits.
    """

    class _Wrapper(Module[[], None]):
        def __init__(self) -> None:
            super().__init__()
            self.weight = NVFP4Tensor.zeros((256, 512))

        def forward(self) -> None:  # pragma: no cover - never called
            raise NotImplementedError

    with F.lazy():
        wrapper = _Wrapper()
        names = {name for name, _ in wrapper.parameters}
        assert names == {
            "weight.data",
            "weight.weight_scale",
            "weight.weight_scale_2",
            "weight.input_scale",
        }


def test_nvfp4_is_qtensor_subclass() -> None:
    assert issubclass(NVFP4Tensor, QTensor)


def test_nvfp4_forward_raises() -> None:
    """NVFP4Tensors are data wrappers, not callable layers."""
    with F.lazy():
        qt = NVFP4Tensor.zeros((256, 512))
        with pytest.raises(
            NotImplementedError, match="QTensor is not a callable layer"
        ):
            qt()


def test_nvfp4_shard_row_axis() -> None:
    """Sharding on axis 0 co-shards data and the block-scale grid."""
    with F.lazy(), patch("max.graph.type.Accelerator") as mock:
        mock.side_effect = _fake_gpu
        devices = [mock(0), mock(1)]
        num_devices = len(devices)
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))

        qt = NVFP4Tensor.zeros((256, 512))
        sharded = qt.shard(axis=0, mesh=mesh)

        assert sharded.data.placements == (Sharded(axis=0),)
        assert sharded.weight_scale.placements == (Sharded(axis=0),)
        for shard in sharded.local_shards:
            assert list(shard.data.shape) == [256 // num_devices, 256]
            assert list(shard.weight_scale.shape) == [256 // num_devices, 32]
        assert sharded.block_size == (1, 16)


def test_nvfp4_shard_keeps_global_scales_whole() -> None:
    """The per-tensor scales are not sharded -- every shard sees them all."""
    with F.lazy(), patch("max.graph.type.Accelerator") as mock:
        mock.side_effect = _fake_gpu
        mesh = DeviceMesh((mock(0), mock(1)), (2,), (TP,))

        qt = NVFP4Tensor.zeros((256, 512))
        sharded = qt.shard(axis=0, mesh=mesh)

        assert sharded.weight_scale_2 is qt.weight_scale_2
        assert sharded.input_scale is qt.input_scale
        for shard in sharded.local_shards:
            assert shard.weight_scale_2 is qt.weight_scale_2
            assert shard.input_scale is qt.input_scale


def test_nvfp4_shard_3d_down_proj_shape() -> None:
    """A stacked ``[E, N, K/2]`` down-proj weight shards on the packed K axis."""
    with F.lazy(), patch("max.graph.type.Accelerator") as mock:
        mock.side_effect = _fake_gpu
        devices = [mock(0), mock(1)]
        num_devices = len(devices)
        mesh = DeviceMesh(tuple(devices), (num_devices,), (TP,))

        num_experts = 4
        hidden = 256
        moe_dim = 512

        qt = NVFP4Tensor(
            data=Tensor.zeros(
                (num_experts, hidden, moe_dim // 2), dtype=DType.uint8
            ),
            weight_scale=Tensor.zeros(
                (num_experts, hidden, moe_dim // 16),
                dtype=DType.float8_e4m3fn,
            ),
            weight_scale_2=Tensor.zeros((num_experts,), dtype=DType.float32),
            input_scale=Tensor.zeros((num_experts,), dtype=DType.float32),
        )
        sharded = qt.shard(axis=-1, mesh=mesh)

        for shard in sharded.local_shards:
            assert list(shard.data.shape) == [
                num_experts,
                hidden,
                moe_dim // 2 // num_devices,
            ]
            assert list(shard.weight_scale.shape) == [
                num_experts,
                hidden,
                moe_dim // 16 // num_devices,
            ]


def test_nvfp4_to_device_moves_only_the_block_leaves() -> None:
    """``to()`` transfers data/weight_scale; the global scales stay on host."""
    with F.lazy(), patch("max.graph.type.Accelerator") as mock:
        mock.side_effect = _fake_gpu
        device = mock(0)

        qt = NVFP4Tensor.zeros((256, 512))
        moved = qt.to(device)

        assert moved.data.device == device
        assert moved.weight_scale.device == device
        assert moved.weight_scale_2 is qt.weight_scale_2
        assert moved.input_scale is qt.input_scale
        assert moved.block_size == qt.block_size


def test_qtensor_base_rejects_sharding_helpers() -> None:
    """The base class refuses the placement helpers subclasses must implement."""
    with F.lazy():
        qt = QTensor()
        with pytest.raises(NotImplementedError, match="local_shards"):
            _ = qt.local_shards
        with pytest.raises(NotImplementedError, match="to\\(\\)"):
            qt.to(CPU())


# --------------------------------------------------------------------------- #
# Homogeneity type guards
# --------------------------------------------------------------------------- #


def test_type_guards_narrow_homogeneous_bundles() -> None:
    """``all_nvfp4`` / ``all_fp8_block`` only accept a uniform bundle."""
    with F.lazy():
        nvfp4 = NVFP4Tensor.zeros((256, 512))
        fp8 = FP8BlockTensor.zeros((256, 512))
        plain = Tensor.zeros((256, 512), dtype=DType.bfloat16)

        assert all_nvfp4([nvfp4, nvfp4])
        assert not all_nvfp4([nvfp4, fp8])
        assert not all_nvfp4([nvfp4, plain])
        assert all_fp8_block([fp8, fp8])
        assert not all_fp8_block([fp8, nvfp4])


# --------------------------------------------------------------------------- #
# NVFP4Activation
# --------------------------------------------------------------------------- #


def test_nvfp4_activation_leaves() -> None:
    """The dispatched-token wrapper exposes its four leaves as parameters."""
    with F.lazy():
        act = NVFP4Activation(
            data=Tensor.zeros((8, 128), dtype=DType.uint8),
            scales=Tensor.zeros((1, 1, 32, 4, 4), dtype=DType.float8_e4m3fn),
            input_scale=Tensor.zeros((), dtype=DType.float32),
            scales_offset=Tensor.zeros((4,), dtype=DType.uint32),
        )

        assert {name for name, _ in act.parameters} == {
            "data",
            "scales",
            "input_scale",
            "scales_offset",
        }
        assert list(act.data.shape) == [8, 128]
