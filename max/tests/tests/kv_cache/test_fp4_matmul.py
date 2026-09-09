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

"""Tests for FP4 matmul kernels in max.nn.kernels."""

from __future__ import annotations

from typing import Any

import pytest
from max.dtype import DType
from max.graph import BufferType, DeviceRef, Graph, TensorType, TensorValue
from max.nn.kernels import (
    _fused_qkv_ragged_matmul_scaled_float4 as fused_qkv_ragged_matmul_scaled_float4,
)
from max.nn.kernels import (
    block_scales_interleave,
    dynamic_block_scaled_matmul,
    grouped_matmul_block_scaled,
    quantize_dynamic_block_scaled,
)
from max.nn.kv_cache import (
    KVCacheParams,
    MHAKVCacheParams,
    PagedCacheValues,
)


class FusedQKVRaggedMatmulScaledFloat4:
    """Wrapper for testing fused_qkv_ragged_matmul_scaled_float4."""

    def __init__(
        self,
        kv_params: KVCacheParams,
        kv_collection: PagedCacheValues,
        n_heads: int,
    ) -> None:
        self.kv_params = kv_params
        self.kv_collection = kv_collection
        self.n_heads = n_heads

    def __call__(
        self,
        input: TensorValue,
        input_row_offsets: TensorValue,
        wqkv: TensorValue,
        layer_idx: TensorValue,
        input_scale: TensorValue,
        kv_scales: TensorValue,
        weight_scale: TensorValue,
        weight_scale_2: TensorValue,
    ) -> TensorValue:
        x, x_scales = quantize_dynamic_block_scaled(
            input,
            tensor_sf=1.0 / input_scale,
            scales_type=DType.float8_e4m3fn,
            out_type=DType.uint8,  # fp4-e2m1fnX2
        )

        weight_scale = weight_scale.to(x.device)
        weight_scale = block_scales_interleave(
            weight_scale,
        )

        return fused_qkv_ragged_matmul_scaled_float4(
            self.kv_params,
            x,
            input_row_offsets,
            wqkv,
            self.kv_collection,
            layer_idx,
            self.n_heads,
            x_scales,
            weight_scale,
            input_scale * weight_scale_2,
            kv_scales,
        )


def test_fused_qkv_ragged_matmul_scaled_float4_valid() -> None:
    """Tests fused_qkv_ragged_matmul_scaled_float4 with all tensors on same device."""
    device = DeviceRef.CPU()

    # Create KV cache parameters
    kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=64,
        num_layers=1,
        page_size=128,
        devices=[device],
    )

    with Graph(
        "fused_qkv_ragged_matmul_scaled_float4",
        input_types=[
            # input
            TensorType(DType.bfloat16, shape=(10, 512), device=device),
            # input_row_offsets
            TensorType(DType.uint32, shape=(3,), device=device),
            # wqkv
            TensorType(DType.uint8, shape=(512, 1536), device=device),
            # layer_idx
            TensorType(DType.uint32, shape=(), device=device),
            # input_scale
            TensorType(DType.float32, shape=(), device=device),
            # kv_scales
            TensorType(DType.bfloat16, shape=(1, 1), device=device),
            # weight_scale
            TensorType(
                DType.float8_e4m3fn, shape=(1536, 512 // 16), device=device
            ),
            # weight_scale_2
            TensorType(DType.float32, shape=(), device=device),
            # KV cache collection inputs
            # blocks: [num_pages, 2, n_kv_heads, page_size, head_dim]
            BufferType(
                DType.bfloat16, shape=(16, 2, 8, 128, 64), device=device
            ),
            # cache_lengths: [batch_size]
            TensorType(DType.uint32, shape=(2,), device=device),
            # lookup_table: [batch_size, max_pages]
            TensorType(DType.uint32, shape=(2, 8), device=device),
            # max_prompt_length: scalar
            TensorType(DType.uint32, shape=(1,), device=device),
            # max_cache_length: scalar
            TensorType(DType.uint32, shape=(1,), device=device),
        ],
    ) as graph:
        (
            input_tensor,
            input_row_offsets,
            wqkv,
            layer_idx,
            input_scale,
            kv_scales,
            weight_scale,
            weight_scale_2,
            blocks,
            cache_lengths,
            lookup_table,
            max_prompt_length,
            max_cache_length,
        ) = graph.inputs

        kv_collection = PagedCacheValues(
            blocks.buffer,
            cache_lengths.tensor,
            lookup_table.tensor,
            max_prompt_length.tensor,
            max_cache_length.tensor,
        )

        tester = FusedQKVRaggedMatmulScaledFloat4(kv_params, kv_collection, 32)

        # Now call the kernel - should not raise any errors when all devices match
        output = tester(
            input_tensor.tensor,
            input_row_offsets.tensor,
            wqkv.tensor,
            layer_idx.tensor,
            input_scale.tensor,
            kv_scales.tensor,
            weight_scale.tensor,
            weight_scale_2.tensor,
        )
        assert output.shape == [10, 32 * 64]  # [seq_len, n_heads * head_dim]
        assert output.dtype == DType.bfloat16


def test_dynamic_block_scaled_1d1d_matmul_fp4() -> None:
    """Tests dynamic_block_scaled_1d1d_matmul_fp4 with valid inputs."""
    device = DeviceRef.CPU()
    with Graph(
        "dynamic_block_scaled_matmul",
        input_types=[
            # a
            TensorType(DType.uint8, shape=(127, 129), device=device),
            # b
            TensorType(DType.uint8, shape=(129, 129), device=device),
            # a_scales
            TensorType(
                DType.float8_e4m3fn, shape=(1, 5, 32, 4, 4), device=device
            ),
            # b_scales
            TensorType(
                DType.float8_e4m3fn, shape=(2, 5, 32, 4, 4), device=device
            ),
        ],
    ) as graph:
        a, b, a_scales, b_scales = (inp.tensor for inp in graph.inputs)

        output = dynamic_block_scaled_matmul(
            a,
            b,
            a_scales,
            b_scales,
            1.0,
        )
        assert output.shape == [127, 129]
        assert output.dtype == DType.bfloat16


@pytest.mark.parametrize(
    "sf_vector_size,scales_type,expected_scales_shape",
    [
        (16, DType.float8_e4m3fn, [2, 3, 32, 4, 4]),
        (32, DType.float8_e8m0fnu, [2, 2, 32, 4, 4]),
    ],
    ids=["nvfp4", "mxfp4"],
)
def test_quantize_dynamic_block_scaled_fp4(
    sf_vector_size: int,
    scales_type: DType,
    expected_scales_shape: list[int],
) -> None:
    """Tests quantize_dynamic_block_scaled with valid FP4 inputs."""
    from max.nn.kernels import _is_sm10x_gpu

    device = DeviceRef.CPU()
    with Graph(
        "quantize_dynamic_block_scaled",
        input_types=[
            TensorType(DType.bfloat16, shape=(129, 192), device=device),
        ],
    ) as graph:
        (input,) = (inp.tensor for inp in graph.inputs)

        quantized_output, scales = quantize_dynamic_block_scaled(
            input,
            1.0,
            sf_vector_size=sf_vector_size,
            scales_type=scales_type,
        )
        assert quantized_output.shape == [129, 96]
        assert quantized_output.dtype == DType.uint8
        assert scales.dtype == scales_type
        if _is_sm10x_gpu():
            assert scales.shape == expected_scales_shape
        else:
            assert scales.shape == [129, 192 // sf_vector_size]


def test_quantize_mxfp4() -> None:
    """Tests quantize_dynamic_block_scaled with MXFP4 params."""
    from max.nn.kernels import _is_sm10x_gpu

    device = DeviceRef.CPU()
    with Graph(
        "quantize_mxfp4",
        input_types=[
            TensorType(DType.bfloat16, shape=(128, 128), device=device),
        ],
    ) as graph:
        (input,) = (inp.tensor for inp in graph.inputs)

        quantized_output, scales = quantize_dynamic_block_scaled(
            input,
            1.0,
            sf_vector_size=32,
            scales_type=DType.float8_e8m0fnu,
        )
        assert quantized_output.shape == [128, 64]
        assert quantized_output.dtype == DType.uint8
        assert scales.dtype == DType.float8_e8m0fnu
        if _is_sm10x_gpu():
            assert scales.shape == [1, 1, 32, 4, 4]
        else:
            assert scales.shape == [128, 4]


@pytest.mark.parametrize(
    "sf_vector_size,scales_dtype,input_shape,expected_output_shape",
    [
        (16, DType.float8_e4m3fn, (129, 136), [2, 34, 32, 4, 4]),
        (32, DType.float8_e8m0fnu, (129, 68), [2, 17, 32, 4, 4]),
    ],
    ids=["nvfp4", "mxfp4"],
)
def test_block_scales_interleave(
    sf_vector_size: int,
    scales_dtype: DType,
    input_shape: tuple[int, int],
    expected_output_shape: list[int],
) -> None:
    """Tests block_scales_interleave with valid inputs."""
    device = DeviceRef.CPU()
    with Graph(
        "block_scales_interleave",
        input_types=[
            TensorType(scales_dtype, shape=input_shape, device=device),
        ],
    ) as graph:
        (scales,) = (inp.tensor for inp in graph.inputs)

        scales_interleaved = block_scales_interleave(
            scales,
            sf_vector_size=sf_vector_size,
        )
        assert scales_interleaved.shape == expected_output_shape
        assert scales_interleaved.dtype == scales_dtype


# Block-scaled scale factor layout constants.
# NOTE: tcgen05 scale factors are stored in a 5D layout:
# (M // 32 // 4, K // VEC_SIZE // 4, 32, 4, 4).
# Shape of scale factor MN-group.
_SF_ATOM_M = (32, 4)
# Number of scale factors per K-group.
_SF_ATOM_K = 4
_SF_MN_GROUP_SIZE = _SF_ATOM_M[0] * _SF_ATOM_M[1]  # 128


def _get_fp4_input_types(
    device: DeviceRef,
    num_experts: int = 3,
    total_tokens: int = 99,
    N: int = 256,
    K: int = 512,
    hidden_dtype: DType = DType.uint8,
    scales_dtype: DType = DType.float8_e4m3fn,
    a_scales_dtype: DType | None = None,
    sf_vector_size: int = 16,
    a_scales_shape: tuple[int, ...] | None = None,
) -> list[TensorType | BufferType]:
    """Returns input types for grouped_matmul_block_scaled tests."""
    if a_scales_dtype is None:
        a_scales_dtype = scales_dtype
    sf_k_group_size = _SF_ATOM_K * sf_vector_size
    num_scale_rows = (total_tokens + _SF_MN_GROUP_SIZE - 1) // _SF_MN_GROUP_SIZE
    K_groups = (K + sf_k_group_size - 1) // sf_k_group_size
    N_groups = (N + _SF_MN_GROUP_SIZE - 1) // _SF_MN_GROUP_SIZE

    if a_scales_shape is None:
        a_scales_shape = (
            num_scale_rows,
            K_groups,
            _SF_ATOM_M[0],
            _SF_ATOM_M[1],
            _SF_ATOM_K,
        )

    # Only a uint8 activation row is nibble-packed. W4A8 hands the kernel E4M3
    # activations, one byte per element, against still-packed weights.
    hidden_k = K // 2 if hidden_dtype == DType.uint8 else K

    return [
        TensorType(hidden_dtype, shape=(total_tokens, hidden_k), device=device),
        TensorType(DType.uint8, shape=(num_experts, N, K // 2), device=device),
        TensorType(a_scales_dtype, shape=a_scales_shape, device=device),
        TensorType(
            scales_dtype,
            shape=(
                num_experts,
                N_groups,
                K_groups,
                _SF_ATOM_M[0],
                _SF_ATOM_M[1],
                _SF_ATOM_K,
            ),
            device=device,
        ),
        TensorType(
            DType.uint32, shape=(1,), device=device
        ),  # expert_start_indices
        TensorType(
            DType.uint32, shape=(num_experts,), device=device
        ),  # a_scale_offsets
        TensorType(
            DType.int32, shape=(num_experts,), device=device
        ),  # expert_ids
        TensorType(
            DType.float32, shape=(num_experts,), device=device
        ),  # expert_scales
        TensorType(
            DType.uint32, shape=(2,), device=device
        ),  # expert_usage_stats_host
    ]


def _call_fp4_matmul(
    input_types: list[TensorType | BufferType],
) -> TensorValue:
    """Builds a graph calling grouped_matmul_block_scaled."""
    with Graph("test_fp4", input_types=input_types) as graph:
        inputs = graph.inputs
        return grouped_matmul_block_scaled(
            inputs[0].tensor,
            inputs[1].tensor,
            inputs[2].tensor,
            inputs[3].tensor,
            inputs[4].tensor,
            inputs[5].tensor,
            inputs[6].tensor,
            inputs[7].tensor,
            inputs[8].tensor,
        )


@pytest.mark.parametrize(
    "sf_vector_size,scales_dtype",
    [
        (16, DType.float8_e4m3fn),
        (32, DType.float8_e8m0fnu),
    ],
    ids=["nvfp4", "mxfp4"],
)
def test_grouped_matmul_block_scaled_valid(
    sf_vector_size: int, scales_dtype: DType
) -> None:
    """Tests grouped_matmul_block_scaled with valid inputs."""
    input_types = _get_fp4_input_types(
        DeviceRef.CPU(),
        sf_vector_size=sf_vector_size,
        scales_dtype=scales_dtype,
    )
    output = _call_fp4_matmul(input_types)
    assert output.shape == [99, 256]
    assert output.dtype == DType.bfloat16


@pytest.mark.parametrize(
    "kwargs,error_type,error_match",
    [
        (
            {"hidden_dtype": DType.bfloat16},
            ValueError,
            "expected hidden_states and weight to have the same dtype",
        ),
        (
            {"a_scales_dtype": DType.float32},
            ValueError,
            "expected a_scales and b_scales to have the same dtype",
        ),
        (
            {"scales_dtype": DType.float32},
            TypeError,
            "a_scales dtype must be float8_e4m3fn \\(NVFP4\\) or float8_e8m0fnu \\(MXFP4/MXFP8\\)",
        ),
        (
            {"a_scales_shape": (1, 8)},
            ValueError,
            "expected a_scales to have rank 5, was 2",
        ),
    ],
)
def test_grouped_matmul_block_scaled_invalid(
    kwargs: dict[str, Any], error_type: type[Exception], error_match: str
) -> None:
    """Tests grouped_matmul_block_scaled rejects invalid inputs."""
    input_types = _get_fp4_input_types(DeviceRef.CPU(), **kwargs)
    with pytest.raises(error_type, match=error_match):
        _call_fp4_matmul(input_types)


def test_grouped_matmul_block_scaled_w4a8_valid() -> None:
    """Tests grouped_matmul_block_scaled accepts the mixed W4A8 operand pair.

    E4M3 activations against nibble-packed E2M1 weights is the one pair whose
    operands differ, so it must clear the same-dtype check, and its weight rows
    must be compared against the activations in elements rather than in bytes.
    """
    input_types = _get_fp4_input_types(
        DeviceRef.CPU(),
        hidden_dtype=DType.float8_e4m3fn,
        scales_dtype=DType.float8_e8m0fnu,
        sf_vector_size=32,
    )
    output = _call_fp4_matmul(input_types)
    assert output.shape == [99, 256]
    assert output.dtype == DType.bfloat16


def test_grouped_matmul_block_scaled_w4a8_rejects_nvfp4_scales() -> None:
    """Tests the W4A8 dtype relaxation is gated on the scale dtype too.

    The kernel implements the mixed operand pair only on E8M0 group-32 scales.
    Admitting it under NVFP4 scales here would defer the error to a Mojo
    comptime assert partway through graph compilation.
    """
    input_types = _get_fp4_input_types(
        DeviceRef.CPU(),
        hidden_dtype=DType.float8_e4m3fn,
        scales_dtype=DType.float8_e4m3fn,
        sf_vector_size=16,
    )
    with pytest.raises(
        ValueError,
        match="expected hidden_states and weight to have the same dtype",
    ):
        _call_fp4_matmul(input_types)


def test_grouped_matmul_block_scaled_w4a8_rejects_k_mismatch() -> None:
    """Tests W4A8 weights are still held to the activations' K extent.

    The reported expectation is in packed bytes, since that is the shape the
    caller has to supply.
    """
    input_types = _get_fp4_input_types(
        DeviceRef.CPU(),
        hidden_dtype=DType.float8_e4m3fn,
        scales_dtype=DType.float8_e8m0fnu,
        sf_vector_size=32,
    )
    # Halve the weights' packed row: 128 bytes covers 256 elements, not 512.
    input_types[1] = TensorType(
        DType.uint8, shape=(3, 256, 128), device=DeviceRef.CPU()
    )

    with pytest.raises(
        ValueError,
        match=r"expected weight is of shape \[num_experts, \*, 256\]",
    ):
        _call_fp4_matmul(input_types)


def test_grouped_matmul_block_scaled_rejects_misplaced_scales() -> None:
    """Tests grouped_matmul_block_scaled rejects scales on another device."""
    input_types = _get_fp4_input_types(DeviceRef.CPU())
    input_types[2] = TensorType(
        DType.float8_e4m3fn,
        shape=(1, 8, _SF_ATOM_M[0], _SF_ATOM_M[1], _SF_ATOM_K),
        device=DeviceRef.GPU(),
    )

    with pytest.raises(
        ValueError,
        match="expected hidden_states and a_scales to have the same device",
    ):
        _call_fp4_matmul(input_types)
