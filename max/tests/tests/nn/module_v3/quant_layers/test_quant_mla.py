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

"""Lazy-trace tests for QuantizedLatentAttentionWithRope."""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import MagicMock

import pytest
from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.sharding import (
    DeviceMesh,
    Partial,
    PlacementMapping,
    Replicated,
    Sharded,
)
from max.experimental.tensor import Tensor, default_dtype
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Shape,
    SymbolicDim,
    TensorValue,
)
from max.nn.kv_cache import KVCacheParams, MLAKVCacheParams
from max.nn.quant_config import QuantConfig
from max.pipelines.architectures.deepseekV3_modulev3.layers.quant_mla import (
    QuantizedLatentAttentionWithRope,
    tensor_parallel_latent_attention_with_rope,
)
from max.pipelines.architectures.deepseekV3_modulev3.layers.quant_tensor import (
    FP8BlockTensor,
    NVFP4Tensor,
)

# Block-aligned dimensions (multiples of 128 on the quantized axes) so the FP8
# weight-scale grid divides evenly and the kv_b_proj scale reshapes are exact.
_N_HEADS = 2
_HIDDEN_SIZE = 256
_Q_LORA_RANK = 128
_KV_LORA_RANK = 128
_QK_NOPE_HEAD_DIM = 128
_QK_ROPE_HEAD_DIM = 64
_V_HEAD_DIM = 128
_QK_HEAD_DIM = _QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM  # 192
_CACHE_HEAD_DIM = _KV_LORA_RANK + _QK_ROPE_HEAD_DIM  # 192
_NUM_LAYERS = 2
_PAGE_SIZE = 128


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
    kv_params: KVCacheParams,
    *,
    q_lora_rank: int | None,
    quant_config: QuantConfig | None,
) -> QuantizedLatentAttentionWithRope:
    return QuantizedLatentAttentionWithRope(
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
        quant_config=quant_config,
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


def _expected_parameters(
    *, q_lora_rank: int | None, quantized: bool
) -> set[str]:
    def proj(name: str) -> set[str]:
        if quantized:
            return {f"{name}.data", f"{name}.weight_scale_inv"}
        return {name}

    names: set[str] = set()
    if q_lora_rank is not None:
        names |= proj("q_a_proj")
        names.add("q_a_layernorm.weight")
        names |= proj("q_b_proj")
    else:
        names |= proj("q_proj")
    # The kv_a layernorm gamma is a plain (unquantized) tensor.
    names.add("kv_a_proj_layernorm")
    names |= proj("kv_a_proj_with_mqa")
    names |= proj("kv_b_proj")
    # o_proj is a QuantizedLinear (bias=False) -> nested under "o_proj".
    names |= {f"o_proj.{p}" for p in proj("weight")}
    return names


# --------------------------------------------------------------------------- #
# Construction / quantized flag
# --------------------------------------------------------------------------- #


def test_mla_bf16_not_quantized(mock_accelerator: MagicMock) -> None:
    device = mock_accelerator()
    kv_params = _make_kv_params([device])
    with F.lazy():
        layer = _make_layer(kv_params, q_lora_rank=None, quant_config=None).to(
            device
        )
        assert layer.quantized is False
        assert layer.weight_block_size is None


def test_mla_fp8_quantized(
    mock_accelerator: MagicMock, fp8_quant_config: QuantConfig
) -> None:
    device = mock_accelerator()
    kv_params = _make_kv_params([device])
    with F.lazy():
        layer = _make_layer(
            kv_params, q_lora_rank=None, quant_config=fp8_quant_config
        ).to(device)
        assert layer.quantized is True
        assert (
            layer.weight_block_size == fp8_quant_config.weight_scale.block_size
        )


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("q_lora_rank", [None, _Q_LORA_RANK])
def test_mla_bf16_parameters(
    mock_accelerator: MagicMock, q_lora_rank: int | None
) -> None:
    device = mock_accelerator()
    kv_params = _make_kv_params([device])
    with F.lazy():
        layer = _make_layer(
            kv_params, q_lora_rank=q_lora_rank, quant_config=None
        ).to(device)
        names = {name for name, _ in layer.parameters}
        assert names == _expected_parameters(
            q_lora_rank=q_lora_rank, quantized=False
        )


@pytest.mark.parametrize("q_lora_rank", [None, _Q_LORA_RANK])
def test_mla_fp8_parameters(
    mock_accelerator: MagicMock,
    fp8_quant_config: QuantConfig,
    q_lora_rank: int | None,
) -> None:
    device = mock_accelerator()
    kv_params = _make_kv_params([device])
    with F.lazy():
        layer = _make_layer(
            kv_params,
            q_lora_rank=q_lora_rank,
            quant_config=fp8_quant_config,
        ).to(device)
        names = {name for name, _ in layer.parameters}
        assert names == _expected_parameters(
            q_lora_rank=q_lora_rank, quantized=True
        )


# --------------------------------------------------------------------------- #
# Weight types / shapes
# --------------------------------------------------------------------------- #


def test_mla_bf16_weight_types(mock_accelerator: MagicMock) -> None:
    device = mock_accelerator()
    kv_params = _make_kv_params([device])
    with F.lazy():
        layer = _make_layer(kv_params, q_lora_rank=None, quant_config=None).to(
            device
        )
        assert isinstance(layer.q_proj, Tensor)
        assert list(layer.q_proj.shape) == [
            _N_HEADS * _QK_HEAD_DIM,
            _HIDDEN_SIZE,
        ]
        assert isinstance(layer.kv_b_proj, Tensor)


def test_mla_fp8_weight_types(
    mock_accelerator: MagicMock, fp8_quant_config: QuantConfig
) -> None:
    device = mock_accelerator()
    kv_params = _make_kv_params([device])
    with F.lazy():
        layer = _make_layer(
            kv_params, q_lora_rank=None, quant_config=fp8_quant_config
        ).to(device)
        assert isinstance(layer.q_proj, FP8BlockTensor)
        assert layer.q_proj.data.dtype == DType.float8_e4m3fn
        assert layer.q_proj.weight_scale_inv.dtype == DType.float32
        assert list(layer.q_proj.data.shape) == [
            _N_HEADS * _QK_HEAD_DIM,
            _HIDDEN_SIZE,
        ]
        assert isinstance(layer.kv_b_proj, FP8BlockTensor)


@pytest.mark.parametrize("q_lora_rank", [None, _Q_LORA_RANK])
def test_mla_wqkv_bf16(
    mock_accelerator: MagicMock, q_lora_rank: int | None
) -> None:
    """The fused q||kv_a weight concatenates on the output axis."""
    device = mock_accelerator()
    kv_params = _make_kv_params([device])
    with F.lazy():
        layer = _make_layer(
            kv_params, q_lora_rank=q_lora_rank, quant_config=None
        ).to(device)
        q_rows = (
            _Q_LORA_RANK if q_lora_rank is not None else _N_HEADS * _QK_HEAD_DIM
        )
        wqkv = layer.wqkv
        assert isinstance(wqkv, Tensor)
        assert list(wqkv.shape) == [q_rows + _CACHE_HEAD_DIM, _HIDDEN_SIZE]


def test_mla_wqkv_fp8(
    mock_accelerator: MagicMock, fp8_quant_config: QuantConfig
) -> None:
    device = mock_accelerator()
    kv_params = _make_kv_params([device])
    with F.lazy():
        layer = _make_layer(
            kv_params, q_lora_rank=None, quant_config=fp8_quant_config
        ).to(device)
        wqkv = layer.wqkv
        assert isinstance(wqkv, FP8BlockTensor)
        assert list(wqkv.data.shape) == [
            _N_HEADS * _QK_HEAD_DIM + _CACHE_HEAD_DIM,
            _HIDDEN_SIZE,
        ]


# --------------------------------------------------------------------------- #
# Forward
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("q_lora_rank", [None, _Q_LORA_RANK])
def test_mla_bf16_forward(
    mock_accelerator: MagicMock, q_lora_rank: int | None
) -> None:
    """bf16 prefill forward maps ``[S, hidden]`` -> ``[S, hidden]``."""
    batch_size = 1
    total_seq_len = 4
    n_pages = 4

    device = mock_accelerator()
    kv_params = _make_kv_params([device])
    with F.lazy(), default_dtype(DType.bfloat16):
        layer = _make_layer(
            kv_params, q_lora_rank=q_lora_rank, quant_config=None
        ).to(device)

        x = Tensor.zeros(
            [total_seq_len, _HIDDEN_SIZE], device=device, dtype=DType.bfloat16
        )
        freqs_cis = Tensor.zeros(
            [total_seq_len, _QK_ROPE_HEAD_DIM],
            device=device,
            dtype=DType.bfloat16,
        )
        input_row_offsets = Tensor.zeros([batch_size + 1], dtype=DType.uint32)
        layer_idx = F.constant(0, DType.uint32, device=CPU())
        kv_collection = _build_kv_collection(
            kv_params, batch_size, n_pages, [device]
        )
        layer_idx = F.constant(0, DType.uint32, device=CPU())

        out = layer(x, kv_collection, freqs_cis, layer_idx, input_row_offsets)

    assert list(out.shape) == [total_seq_len, _HIDDEN_SIZE]
    assert out.dtype == x.dtype
    assert out.device == device


@pytest.mark.parametrize("q_lora_rank", [None, _Q_LORA_RANK])
def test_mla_fp8_forward(
    mock_accelerator: MagicMock,
    fp8_quant_config: QuantConfig,
    q_lora_rank: int | None,
) -> None:
    """FP8 prefill forward maps ``[S, hidden]`` -> ``[S, hidden]`` (bf16 out)."""
    batch_size = 1
    total_seq_len = 4
    n_pages = 4

    device = mock_accelerator()
    kv_params = _make_kv_params([device])
    with F.lazy():
        layer = _make_layer(
            kv_params,
            q_lora_rank=q_lora_rank,
            quant_config=fp8_quant_config,
        ).to(device)

        x = Tensor.zeros(
            [total_seq_len, _HIDDEN_SIZE], dtype=DType.bfloat16, device=device
        )
        freqs_cis = Tensor.zeros(
            [total_seq_len, _QK_ROPE_HEAD_DIM], device=device
        )
        input_row_offsets = Tensor.zeros([batch_size + 1], dtype=DType.uint32)
        layer_idx = F.constant(0, DType.uint32, device=CPU())
        kv_collection = _build_kv_collection(
            kv_params, batch_size, n_pages, [device]
        )

        out = layer(x, kv_collection, freqs_cis, layer_idx, input_row_offsets)

    assert list(out.shape) == [total_seq_len, _HIDDEN_SIZE]
    assert out.dtype == DType.bfloat16


# --------------------------------------------------------------------------- #
# Tensor parallelism
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("q_lora_rank", [None, _Q_LORA_RANK])
def test_mla_fp8_tensor_parallel(
    mock_accelerator: MagicMock,
    fp8_quant_config: QuantConfig,
    q_lora_rank: int | None,
) -> None:
    """FP8 MLA tensor parallelism co-shards each weight's data and scales.

    The rowwise projections (``q_b_proj`` / ``kv_b_proj`` / ``q_proj``) shard
    the packed data and block-scale grid on axis 0; ``o_proj`` is columnwise
    (axis 1). The block-scaled FP8 kernels propagate those placements so the
    head-parallel attention output reduces to a partial sum at ``o_proj``.
    """
    batch_size = 1
    total_seq_len = 4
    n_pages = 4

    with F.lazy():
        devices = [mock_accelerator(0), mock_accelerator(1)]
        kv_params = _make_kv_params(devices)
        mesh = DeviceMesh(tuple(devices), (len(devices),), (TP,))
        replicated_mapping = PlacementMapping(mesh, (Replicated(),))

        layer = tensor_parallel_latent_attention_with_rope(
            _make_layer(
                kv_params,
                q_lora_rank=q_lora_rank,
                quant_config=fp8_quant_config,
            )
        ).to(mesh)

        # Rowwise weights co-shard data and scales on axis 0.
        assert isinstance(layer.kv_b_proj, FP8BlockTensor)
        assert layer.kv_b_proj.data.mapping.to_placements() == (Sharded(0),)
        assert layer.kv_b_proj.weight_scale_inv.mapping.to_placements() == (
            Sharded(0),
        )
        # o_proj is columnwise: data and scales shard the contraction (axis 1).
        assert isinstance(layer.o_proj.weight, FP8BlockTensor)
        assert layer.o_proj.weight.data.mapping.to_placements() == (Sharded(1),)
        assert layer.o_proj.weight.weight_scale_inv.mapping.to_placements() == (
            Sharded(1),
        )

        x = Tensor.zeros(
            [total_seq_len, _HIDDEN_SIZE],
            dtype=DType.bfloat16,
            device=replicated_mapping,
        )
        freqs_cis = Tensor.zeros(
            [total_seq_len, _QK_ROPE_HEAD_DIM], device=replicated_mapping
        )
        input_row_offsets = Tensor.zeros(
            [batch_size + 1], dtype=DType.uint32, device=replicated_mapping
        )
        layer_idx = F.constant(0, DType.uint32, device=CPU())
        kv_collection = _build_kv_collection(
            kv_params, batch_size, n_pages, devices
        )

        out = layer(x, kv_collection, freqs_cis, layer_idx, input_row_offsets)

    assert list(out.shape) == [total_seq_len, _HIDDEN_SIZE]
    assert out.dtype == DType.bfloat16
    assert out.mapping.mesh == mesh
    # o_proj is row-parallel, so the attention output is a partial sum; the
    # all-reduce that resolves it lives in the transformer block.
    assert out.mapping.to_placements() == (Partial(),)


# --------------------------------------------------------------------------- #
# NVFP4
# --------------------------------------------------------------------------- #

_NVFP4_LEAVES = ("data", "weight_scale", "weight_scale_2", "input_scale")


def _expected_nvfp4_parameters(*, q_lora_rank: int | None) -> set[str]:
    """NVFP4 checkpoints quantize only ``o_proj``; q/kv stay bf16."""
    names = _expected_parameters(q_lora_rank=q_lora_rank, quantized=False)
    names.discard("o_proj.weight")
    return names | {f"o_proj.weight.{leaf}" for leaf in _NVFP4_LEAVES}


def test_mla_nvfp4_leaves_latent_projections_bf16(
    mock_accelerator: MagicMock, nvfp4_quant_config: QuantConfig
) -> None:
    """``quantized`` tracks the FP8 path only: NVFP4 keeps q/kv in bf16."""
    device = mock_accelerator()
    kv_params = _make_kv_params([device])
    with F.lazy():
        layer = _make_layer(
            kv_params, q_lora_rank=None, quant_config=nvfp4_quant_config
        ).to(device)

        assert layer.quantized is False
        assert layer.weight_block_size is None
        assert isinstance(layer.q_proj, Tensor)
        assert isinstance(layer.kv_a_proj_with_mqa, Tensor)
        assert isinstance(layer.kv_b_proj, Tensor)
        # Only the output projection runs on the FP4 kernels.
        assert isinstance(layer.o_proj.weight, NVFP4Tensor)
        assert list(layer.o_proj.weight.data.shape) == [
            _HIDDEN_SIZE,
            _N_HEADS * _V_HEAD_DIM // 2,
        ]


@pytest.mark.parametrize("q_lora_rank", [None, _Q_LORA_RANK])
def test_mla_nvfp4_parameters(
    mock_accelerator: MagicMock,
    nvfp4_quant_config: QuantConfig,
    q_lora_rank: int | None,
) -> None:
    device = mock_accelerator()
    kv_params = _make_kv_params([device])
    with F.lazy():
        layer = _make_layer(
            kv_params,
            q_lora_rank=q_lora_rank,
            quant_config=nvfp4_quant_config,
        ).to(device)
        names = {name for name, _ in layer.parameters}
        assert names == _expected_nvfp4_parameters(q_lora_rank=q_lora_rank)


def test_mla_nvfp4_wqkv_stays_bf16(
    mock_accelerator: MagicMock, nvfp4_quant_config: QuantConfig
) -> None:
    """The fused q||kv_a weight is a plain concat, not a quantized bundle."""
    device = mock_accelerator()
    kv_params = _make_kv_params([device])
    with F.lazy():
        layer = _make_layer(
            kv_params, q_lora_rank=None, quant_config=nvfp4_quant_config
        ).to(device)
        wqkv = layer.wqkv
        assert isinstance(wqkv, Tensor)
        assert list(wqkv.shape) == [
            _N_HEADS * _QK_HEAD_DIM + _CACHE_HEAD_DIM,
            _HIDDEN_SIZE,
        ]


def test_mla_nvfp4_forward(
    mock_accelerator: MagicMock,
    nvfp4_quant_config: QuantConfig,
    sm100_arch: None,
) -> None:
    """NVFP4 prefill forward maps ``[S, hidden]`` -> ``[S, hidden]`` (bf16)."""
    batch_size = 1
    total_seq_len = 4
    n_pages = 4

    device = mock_accelerator()
    kv_params = _make_kv_params([device])
    with F.lazy(), default_dtype(DType.bfloat16):
        layer = _make_layer(
            kv_params, q_lora_rank=None, quant_config=nvfp4_quant_config
        ).to(device)

        x = Tensor.zeros(
            [total_seq_len, _HIDDEN_SIZE], dtype=DType.bfloat16, device=device
        )
        freqs_cis = Tensor.zeros(
            [total_seq_len, _QK_ROPE_HEAD_DIM],
            dtype=DType.bfloat16,
            device=device,
        )
        input_row_offsets = Tensor.zeros([batch_size + 1], dtype=DType.uint32)
        layer_idx = F.constant(0, DType.uint32, device=CPU())
        kv_collection = _build_kv_collection(
            kv_params, batch_size, n_pages, [device]
        )

        out = layer(x, kv_collection, freqs_cis, layer_idx, input_row_offsets)

    assert list(out.shape) == [total_seq_len, _HIDDEN_SIZE]
    assert out.dtype == DType.bfloat16


def test_mla_nvfp4_tensor_parallel(
    mock_accelerator: MagicMock,
    nvfp4_quant_config: QuantConfig,
    sm100_arch: None,
) -> None:
    """NVFP4 ``o_proj`` is row-parallel: block leaves shard, globals replicate."""
    batch_size = 1
    total_seq_len = 4
    n_pages = 4

    with F.lazy(), default_dtype(DType.bfloat16):
        devices = [mock_accelerator(0), mock_accelerator(1)]
        kv_params = _make_kv_params(devices)
        mesh = DeviceMesh(tuple(devices), (len(devices),), (TP,))
        replicated_mapping = PlacementMapping(mesh, (Replicated(),))

        layer = tensor_parallel_latent_attention_with_rope(
            _make_layer(
                kv_params,
                q_lora_rank=None,
                quant_config=nvfp4_quant_config,
            )
        ).to(mesh)

        weight = layer.o_proj.weight
        assert isinstance(weight, NVFP4Tensor)
        # o_proj is columnwise: the contraction axis (1) is sharded.
        assert weight.data.mapping.to_placements() == (Sharded(1),)
        assert weight.weight_scale.mapping.to_placements() == (Sharded(1),)
        # A per-tensor scale cannot be split, so both globals are replicated.
        assert weight.weight_scale_2.mapping.to_placements() == (Replicated(),)
        assert weight.input_scale.mapping.to_placements() == (Replicated(),)

        x = Tensor.zeros(
            [total_seq_len, _HIDDEN_SIZE],
            dtype=DType.bfloat16,
            device=replicated_mapping,
        )
        freqs_cis = Tensor.zeros(
            [total_seq_len, _QK_ROPE_HEAD_DIM],
            dtype=DType.bfloat16,
            device=replicated_mapping,
        )
        input_row_offsets = Tensor.zeros(
            [batch_size + 1], dtype=DType.uint32, device=replicated_mapping
        )
        layer_idx = F.constant(0, DType.uint32, device=CPU())
        kv_collection = _build_kv_collection(
            kv_params, batch_size, n_pages, devices
        )

        out = layer(x, kv_collection, freqs_cis, layer_idx, input_row_offsets)

    assert list(out.shape) == [total_seq_len, _HIDDEN_SIZE]
    assert out.dtype == DType.bfloat16
    assert out.mapping.mesh == mesh
    assert out.mapping.to_placements() == (Partial(),)
