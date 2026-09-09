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
"""Tests for LatentAttentionWithRope (ModuleV3, single-GPU)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from unittest.mock import MagicMock, patch

import pytest
from max.driver import Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.mesh_axis import DP, TP
from max.experimental.nn.common_layers.multi_latent_attention import (
    LatentAttentionWithRope,
    tensor_parallel_latent_attention_with_rope,
)
from max.experimental.sharding import (
    DeviceMesh,
    PlacementMapping,
    Replicated,
    Sharded,
)
from max.experimental.tensor import Tensor
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Shape,
    SymbolicDim,
    TensorValue,
)
from max.nn.kv_cache import KVCacheParams, MLAKVCacheParams

# Small model dimensions for fast graph-trace tests.
_N_HEADS = 8
_HIDDEN_SIZE = 256
_Q_LORA_RANK = 32
_KV_LORA_RANK = 64
_QK_NOPE_HEAD_DIM = 32
_QK_ROPE_HEAD_DIM = 16
_V_HEAD_DIM = 32
_NUM_LAYERS = 2
_PAGE_SIZE = 128
_CACHE_HEAD_DIM = _KV_LORA_RANK + _QK_ROPE_HEAD_DIM  # 80


def _make_kv_params(devices: Sequence[Device]) -> KVCacheParams:
    return MLAKVCacheParams(
        dtype=DType.float32,
        head_dim=_CACHE_HEAD_DIM,
        num_layers=_NUM_LAYERS,
        devices=[DeviceRef.from_device(device) for device in devices],
        num_q_heads=_N_HEADS,
        page_size=_PAGE_SIZE,
    )


def _make_layer(
    kv_params: KVCacheParams, *, q_lora_rank: int | None
) -> LatentAttentionWithRope:
    return LatentAttentionWithRope(
        num_attention_heads=_N_HEADS,
        num_key_value_heads=1,
        hidden_size=_HIDDEN_SIZE,
        kv_params=kv_params,
        layer_idx=0,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=_KV_LORA_RANK,
        qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        v_head_dim=_V_HEAD_DIM,
        graph_mode="prefill",
    )


def _build_kv_collection(
    kv_params: KVCacheParams,
    batch_size: int | str,
    n_pages: int,
    devices: Sequence[Device],
) -> PagedCacheValues:
    kv_inputs = kv_params.get_symbolic_inputs()

    sym_values = {
        "total_num_pages": n_pages,
        "batch_size": batch_size,
        "max_num_pages": n_pages,
        "steps_remaining": 2,
    }

    def resolve_shape(shape: Shape) -> list[int | str]:
        result: list[int | str] = []
        for dim in shape:
            if isinstance(dim, SymbolicDim):
                key = next(k for k in sym_values if dim.name.endswith(k))
                result.append(sym_values[key])
            else:
                result.append(int(dim))
        return result

    graph_values: list[BufferValue | TensorValue] = []
    for per_device_types in kv_inputs.inputs:
        for field_type in per_device_types.flatten():
            t = Tensor.zeros(
                resolve_shape(field_type.shape),
                dtype=field_type.dtype,
                device=field_type.device.to_device(),
            )
            if isinstance(field_type, BufferType):
                graph_values.append(BufferValue(t))
            else:
                graph_values.append(TensorValue(t))

    kv_concrete = kv_inputs.unflatten(iter(graph_values))
    mapping = PlacementMapping(
        DeviceMesh(tuple(devices), (len(devices),), ("axis",)), (Replicated(),)
    )
    return PagedCacheValues.from_upstream(kv_concrete.inputs, mapping)


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


@pytest.mark.parametrize("q_lora_rank", [None, _Q_LORA_RANK])
def test_layer(mock_accelerator: MagicMock, q_lora_rank: int | None) -> None:
    """Traces the layer in a lazy context and verifies output shape and device."""
    batch_size = 1
    total_seq_len = 4
    n_pages = 4

    devices = [mock_accelerator()]
    kv_params = _make_kv_params(devices)

    with F.lazy():
        layer = _make_layer(kv_params, q_lora_rank=q_lora_rank).to(devices[0])

        x = Tensor.zeros([total_seq_len, _HIDDEN_SIZE], device=devices[0])
        freqs_cis = Tensor.zeros(
            [total_seq_len, _QK_ROPE_HEAD_DIM], device=devices[0]
        )
        input_row_offsets = Tensor.zeros([batch_size + 1], dtype=DType.uint32)
        kv_collection = _build_kv_collection(
            kv_params, batch_size, n_pages, devices
        )

        out = layer(x, kv_collection, freqs_cis, input_row_offsets)

    assert list(out.shape) == [total_seq_len, _HIDDEN_SIZE]
    assert out.device == devices[0]


@pytest.mark.parametrize("q_lora_rank", [None, _Q_LORA_RANK])
def test_tensor_parallel_layer(
    mock_accelerator: MagicMock, q_lora_rank: int | None
) -> None:
    batch_size = 1
    total_seq_len = 4
    n_pages = 4

    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        kv_params = _make_kv_params(devices)

        mesh = DeviceMesh(tuple(devices), (len(devices),), (TP,))
        replicated_mapping = PlacementMapping(mesh, (Replicated(),))

        layer = tensor_parallel_latent_attention_with_rope(
            _make_layer(kv_params, q_lora_rank=q_lora_rank)
        ).to(mesh)

        x = Tensor.zeros(
            [total_seq_len, _HIDDEN_SIZE], device=replicated_mapping
        )
        freqs_cis = Tensor.zeros(
            [total_seq_len, _QK_ROPE_HEAD_DIM], device=replicated_mapping
        )
        input_row_offsets = Tensor.zeros(
            [batch_size + 1], dtype=DType.uint32, device=replicated_mapping
        )
        kv_collection = _build_kv_collection(
            kv_params, batch_size, n_pages, devices
        )

        out = layer(x, kv_collection, freqs_cis, input_row_offsets)

    assert list(out.shape) == [total_seq_len, _HIDDEN_SIZE]
    assert out.mapping.mesh == mesh


@pytest.mark.parametrize("q_lora_rank", [None, _Q_LORA_RANK])
def test_data_parallel_layer(
    mock_accelerator: MagicMock, q_lora_rank: int | None
) -> None:
    # TODO(MXF-295): Use dynamic input sizes, once data parallelism support
    # is fixed.
    batch_size = 2
    total_seq_len = 6
    n_pages = 4

    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        kv_params = _make_kv_params(devices)

        mesh = DeviceMesh(tuple(devices), (len(devices),), (DP,))
        replicated_mapping = PlacementMapping(mesh, (Replicated(),))
        data_parallel_mapping = PlacementMapping(mesh, (Sharded(0),))

        layer = _make_layer(kv_params, q_lora_rank=q_lora_rank).to(mesh)

        x = Tensor.zeros(
            [total_seq_len, _HIDDEN_SIZE], device=data_parallel_mapping
        )
        freqs_cis = Tensor.zeros(
            [total_seq_len, _QK_ROPE_HEAD_DIM], device=replicated_mapping
        )
        input_row_offsets = Tensor.zeros(
            [batch_size + 1], dtype=DType.uint32, device=data_parallel_mapping
        )
        kv_collection = _build_kv_collection(
            kv_params, batch_size, n_pages, devices
        )

        out = layer(x, kv_collection, freqs_cis, input_row_offsets)

    assert list(out.shape) == [total_seq_len, _HIDDEN_SIZE]
    assert out.mapping.mesh == mesh


@pytest.mark.parametrize("q_lora_rank", [None, _Q_LORA_RANK])
def test_data_parallel_layer_symbolic(
    mock_accelerator: MagicMock, q_lora_rank: int | None
) -> None:
    """Traces the layer in a lazy context and verifies output shape and device."""
    max_seq_len = 6
    n_pages = 4

    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        kv_params = _make_kv_params(devices)

        mesh = DeviceMesh(tuple(devices), (len(devices),), (DP,))
        replicated_mapping = PlacementMapping(mesh, (Replicated(),))
        data_parallel_mapping = PlacementMapping(mesh, (Sharded(0),))

        layer = _make_layer(kv_params, q_lora_rank=q_lora_rank)

        x = Tensor.zeros(
            ["dynamic_size", _HIDDEN_SIZE], device=data_parallel_mapping
        )
        freqs_cis = Tensor.zeros(
            [max_seq_len, _QK_ROPE_HEAD_DIM], device=replicated_mapping
        )
        input_row_offsets = Tensor.zeros(
            ["dynamic_batch"], dtype=DType.uint32, device=data_parallel_mapping
        )
        kv_collection = _build_kv_collection(
            kv_params, "dynamic_batch", n_pages, devices
        )

        out = layer(x, kv_collection, freqs_cis, input_row_offsets)

    assert out.shape == x.shape
    assert out.mapping.mesh == mesh
