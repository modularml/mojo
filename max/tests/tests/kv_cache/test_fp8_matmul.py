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

"""Tests for FP8 matmul kernels in max.nn.kernels."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from max.dtype import DType
from max.graph import BufferType, DeviceRef, Graph, TensorType, TensorValue
from max.nn import (
    InputScaleSpec,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.nn.kernels import (
    _fused_qkv_ragged_matmul_scaled_float8 as fused_qkv_ragged_matmul_scaled_float8,
)
from max.nn.kernels import (
    batched_dynamic_scaled_fp8_matmul,
    dynamic_scaled_matmul,
    grouped_dynamic_scaled_fp8_matmul,
    matmul_k_cache_ragged_scaled_float8,
)
from max.nn.kv_cache import (
    KVCacheParams,
    MHAKVCacheParams,
    PagedCacheValues,
)


class DynamicScaledMatmul:
    return_type: DType
    input_scale_spec: InputScaleSpec
    weight_scale_spec: WeightScaleSpec
    """Return type of the `dynamic_scaled_matmul` custom op."""

    def __init__(
        self,
        return_type: DType,
        input_scale_spec: InputScaleSpec,
        weight_scale_spec: WeightScaleSpec,
    ) -> None:
        self.return_type = return_type
        self.input_scale_spec = input_scale_spec
        self.weight_scale_spec = weight_scale_spec

    def __call__(
        self,
        a: TensorValue,
        b: TensorValue,
        a_scales: TensorValue,
        b_scales: TensorValue,
    ) -> TensorValue:
        return dynamic_scaled_matmul(
            a,
            b,
            a_scales,
            b_scales,
            input_scale_spec=self.input_scale_spec,
            weight_scale_spec=self.weight_scale_spec,
            out_type=self.return_type,
        )


def test_dynamic_scaled_matmul_rowwise() -> None:
    """Tests dynamic_scaled_matmul with valid inputs."""
    device = DeviceRef.CPU()
    with Graph(
        "dynamic_scaled_matmul",
        input_types=[
            # a
            TensorType(DType.float8_e4m3fn, shape=(2, 4), device=device),
            # b
            TensorType(DType.float8_e4m3fn, shape=(3, 4), device=device),
            # a_scales
            TensorType(DType.bfloat16, shape=(1, 2), device=device),
            # b_scales
            TensorType(DType.bfloat16, shape=(3, 1), device=device),
        ],
    ) as graph:
        a, b, a_scales, b_scales = (inp.tensor for inp in graph.inputs)

        # Test with row-wise weight scales.
        output_rowwise = dynamic_scaled_matmul(
            a,
            b,
            a_scales,
            b_scales,
            input_scale_spec=InputScaleSpec(
                granularity=ScaleGranularity.COLWISE,
                origin=ScaleOrigin.DYNAMIC,
                dtype=DType.bfloat16,
            ),
            weight_scale_spec=WeightScaleSpec(
                granularity=ScaleGranularity.ROWWISE,
                dtype=DType.bfloat16,
            ),
        )
        assert output_rowwise.shape == [2, 3]
        assert output_rowwise.dtype == DType.bfloat16


@pytest.mark.parametrize(
    "a_dtype, b_dtype, a_scales_dtype, b_scales_dtype, err_msg_part",
    [
        # a.dtype != b.dtype
        (
            DType.float16,
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.bfloat16,
            "a and b dtypes",
        ),
        # a.dtype != b.dtype
        (
            DType.float8_e4m3fn,
            DType.float16,
            DType.bfloat16,
            DType.bfloat16,
            "a and b dtypes",
        ),
        # a_scales.dtype != b_scales.dtype
        (
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.float16,
            DType.bfloat16,
            "scales dtypes",
        ),
        # a_scales.dtype != b_scales.dtype
        (
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            DType.float16,
            "scales dtypes",
        ),
    ],
)
def test_dynamic_scaled_matmul_dtype_mismatch(
    a_dtype: DType,
    b_dtype: DType,
    a_scales_dtype: DType,
    b_scales_dtype: DType,
    err_msg_part: str,
) -> None:
    """Tests dtype mismatches."""
    device = DeviceRef.CPU()
    with pytest.raises(TypeError, match=err_msg_part):
        Graph(
            "dynamic_scaled_matmul",
            forward=DynamicScaledMatmul(
                return_type=DType.bfloat16,
                input_scale_spec=InputScaleSpec(
                    granularity=ScaleGranularity.COLWISE,
                    origin=ScaleOrigin.DYNAMIC,
                    dtype=a_scales_dtype,
                ),
                weight_scale_spec=WeightScaleSpec(
                    granularity=ScaleGranularity.ROWWISE,
                    dtype=b_scales_dtype,
                ),
            ),
            input_types=[
                # a
                TensorType(a_dtype, shape=(2, 4), device=device),
                # b
                TensorType(b_dtype, shape=(3, 4), device=device),
                # a_scales
                TensorType(a_scales_dtype, shape=(1, 2), device=device),
                # b_scales
                TensorType(b_scales_dtype, shape=(3, 1), device=device),
            ],
        )


class FusedQKVRaggedMatmulScaledFloat8:
    """Wrapper for testing fused_qkv_ragged_matmul_scaled_float8."""

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
        weight_scale: TensorValue,
        bias: TensorValue | None = None,
    ) -> TensorValue:
        return fused_qkv_ragged_matmul_scaled_float8(
            self.kv_params,
            input,
            input_row_offsets,
            wqkv,
            self.kv_collection,
            layer_idx,
            self.n_heads,
            input_scale,
            weight_scale,
            bias,
        )


def test_fused_qkv_ragged_matmul_scaled_float8_valid() -> None:
    """Tests fused_qkv_ragged_matmul_scaled_float8 with all tensors on same device."""
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
        "fused_qkv_ragged_matmul_scaled_float8",
        input_types=[
            # input
            TensorType(DType.float8_e4m3fn, shape=(10, 512), device=device),
            # input_row_offsets
            TensorType(DType.uint32, shape=(3,), device=device),
            # wqkv
            TensorType(DType.float8_e4m3fn, shape=(512, 1536), device=device),
            # layer_idx
            TensorType(DType.uint32, shape=(), device=device),
            # input_scale
            TensorType(DType.bfloat16, shape=(1, 1), device=device),
            # weight_scale
            TensorType(DType.bfloat16, shape=(1, 1), device=device),
            # bias
            TensorType(DType.bfloat16, shape=(1536,), device=device),
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
            weight_scale,
            bias,
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

        # Now call the kernel - should not raise any errors when all devices match
        output = fused_qkv_ragged_matmul_scaled_float8(
            kv_params,
            input_tensor.tensor,
            input_row_offsets.tensor,
            wqkv.tensor,
            kv_collection,
            layer_idx.tensor,
            32,  # n_heads
            input_scale.tensor,
            weight_scale.tensor,
            bias.tensor,
        )
        assert output.shape == [10, 32 * 64]  # [seq_len, n_heads * head_dim]
        assert output.dtype == DType.bfloat16


@pytest.mark.parametrize(
    "input_dev, wqkv_dev, row_off_dev, in_scale_dev, w_scale_dev, err_msg_part",
    [
        # Individual device mismatches
        ("cpu", "gpu", "cpu", "cpu", "cpu", r"wqkv=gpu:0\n"),
        ("cpu", "cpu", "gpu", "cpu", "cpu", r"input_row_offsets=gpu:0\n"),
        ("cpu", "cpu", "cpu", "gpu", "cpu", r"input_scale=gpu:0\n"),
        ("cpu", "cpu", "cpu", "cpu", "gpu", r"weight_scale=gpu:0"),
        # Multiple device mismatches
        (
            "cpu",
            "gpu",
            "gpu",
            "cpu",
            "cpu",
            r"wqkv=gpu:0\n.*input_row_offsets=gpu:0\n",
        ),
        (
            "cpu",
            "cpu",
            "cpu",
            "gpu",
            "gpu",
            r"input_scale=gpu:0\n.*weight_scale=gpu:0",
        ),
        # All on wrong device
        (
            "cpu",
            "gpu",
            "gpu",
            "gpu",
            "gpu",
            r"wqkv=gpu:0\n.*input_row_offsets=gpu:0\n.*input_scale=gpu:0\n.*weight_scale=gpu:0",
        ),
        # Reverse case - input on GPU, others on CPU
        (
            "gpu",
            "cpu",
            "cpu",
            "cpu",
            "cpu",
            r"wqkv=cpu:0\n.*input_row_offsets=cpu:0\n.*input_scale=cpu:0\n.*weight_scale=cpu:0",
        ),
    ],
)
def test_fused_qkv_ragged_matmul_scaled_float8_device_mismatch(
    input_dev: str,
    wqkv_dev: str,
    row_off_dev: str,
    in_scale_dev: str,
    w_scale_dev: str,
    err_msg_part: str,
) -> None:
    """Tests that device mismatches raise appropriate errors."""

    # Map device strings to DeviceRef
    def get_device(dev_str: str) -> DeviceRef:
        return DeviceRef.GPU(0) if dev_str == "gpu" else DeviceRef.CPU()

    # Create KV cache parameters (can use real object for device tests)
    kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=64,
        num_layers=1,
        page_size=128,
        devices=[get_device(wqkv_dev)],
    )

    kv_collection = Mock(spec=PagedCacheValues)

    with pytest.raises(ValueError, match=err_msg_part):
        Graph(
            "fused_qkv_ragged_matmul_scaled_float8",
            forward=FusedQKVRaggedMatmulScaledFloat8(
                kv_params, kv_collection, n_heads=32
            ),
            input_types=[
                # input
                TensorType(
                    DType.float8_e4m3fn,
                    shape=(10, 512),
                    device=get_device(input_dev),
                ),
                # input_row_offsets
                TensorType(
                    DType.uint32, shape=(3,), device=get_device(row_off_dev)
                ),
                # wqkv
                TensorType(
                    DType.float8_e4m3fn,
                    shape=(512, 1536),
                    device=get_device(wqkv_dev),
                ),
                # layer_idx - must always be on CPU
                TensorType(DType.uint32, shape=(), device=DeviceRef.CPU()),
                # input_scale
                TensorType(
                    DType.bfloat16,
                    shape=(1, 1),
                    device=get_device(in_scale_dev),
                ),
                # weight_scale
                TensorType(
                    DType.bfloat16, shape=(1, 1), device=get_device(w_scale_dev)
                ),
            ],
        )


def test_fused_qkv_ragged_matmul_scaled_float8_layer_idx_device() -> None:
    """Tests that layer_idx must be on CPU device."""
    device = DeviceRef.GPU(0)

    # Create KV cache parameters
    kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=64,
        num_layers=1,
        page_size=128,
        devices=[device],
    )

    kv_collection = Mock(spec=PagedCacheValues)

    with pytest.raises(
        ValueError,
        match="expected layer_idx to be on CPU device, but got gpu:0",
    ):
        Graph(
            "fused_qkv_ragged_matmul_scaled_float8",
            forward=FusedQKVRaggedMatmulScaledFloat8(
                kv_params, kv_collection, n_heads=32
            ),
            input_types=[
                # input
                TensorType(DType.float8_e4m3fn, shape=(10, 512), device=device),
                # input_row_offsets
                TensorType(DType.uint32, shape=(3,), device=device),
                # wqkv
                TensorType(
                    DType.float8_e4m3fn, shape=(512, 1536), device=device
                ),
                # layer_idx - incorrectly on GPU
                TensorType(DType.uint32, shape=(), device=device),
                # input_scale
                TensorType(DType.bfloat16, shape=(1, 1), device=device),
                # weight_scale
                TensorType(DType.bfloat16, shape=(1, 1), device=device),
            ],
        )


def test_matmul_k_cache_ragged_scaled_float8_valid() -> None:
    """Tests matmul_k_cache_ragged_scaled_float8 with all tensors on same device."""
    device = DeviceRef.CPU()
    scale_granularity = (1, 128, 128)

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
        "matmul_k_cache_ragged_scaled_float8",
        input_types=[
            # hidden_states
            TensorType(DType.float8_e4m3fn, shape=(10, 512), device=device),
            # input_row_offsets
            TensorType(DType.uint32, shape=(3,), device=device),
            # weight
            TensorType(DType.float8_e4m3fn, shape=(8, 512), device=device),
            # input_scale
            TensorType(DType.float32, shape=(10, 4), device=device),
            # weight_scale
            TensorType(DType.float32, shape=(4, 1), device=device),
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
            # layer_idx
            TensorType(DType.uint32, shape=(), device=device),
        ],
    ) as graph:
        (
            hidden_states,
            input_row_offsets,
            weight,
            input_scale,
            weight_scale,
            blocks,
            cache_lengths,
            lookup_table,
            max_prompt_length,
            max_cache_length,
            layer_idx,
        ) = graph.inputs
        kv_collection = PagedCacheValues(
            blocks.buffer,
            cache_lengths.tensor,
            lookup_table.tensor,
            max_prompt_length.tensor,
            max_cache_length.tensor,
        )
        matmul_k_cache_ragged_scaled_float8(
            kv_params,
            hidden_states.tensor,
            input_row_offsets.tensor,
            weight.tensor,
            input_scale.tensor,
            weight_scale.tensor,
            kv_collection,
            scale_granularity,
            layer_idx.tensor,
        )


def test_matmul_k_cache_ragged_scaled_float8_invalid() -> None:
    """Tests matmul_k_cache_ragged_scaled_float8 with invalid inputs."""
    device = DeviceRef.CPU()
    scale_granularity = (1, 128, 128)

    # Create KV cache parameters
    kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=64,
        num_layers=1,
        page_size=128,
        devices=[device],
    )

    def get_valid_input_types() -> list[TensorType | BufferType]:
        return [
            # hidden_states
            TensorType(DType.float8_e4m3fn, shape=(10, 512), device=device),
            # input_row_offsets
            TensorType(DType.uint32, shape=(3,), device=device),
            # weight
            TensorType(DType.float8_e4m3fn, shape=(8, 512), device=device),
            # input_scale
            TensorType(DType.float32, shape=(10, 4), device=device),
            # weight_scale
            TensorType(DType.float32, shape=(4, 1), device=device),
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
            # layer_idx
            TensorType(DType.uint32, shape=(), device=device),
        ]

    def try_create_graph(input_types: list[TensorType | BufferType]) -> None:
        with Graph(
            "matmul_k_cache_ragged_scaled_float8",
            input_types=input_types,
        ) as graph:
            (
                hidden_states,
                input_row_offsets,
                weight,
                input_scale,
                weight_scale,
                blocks,
                cache_lengths,
                lookup_table,
                max_prompt_length,
                max_cache_length,
                layer_idx,
            ) = graph.inputs
            kv_collection = PagedCacheValues(
                blocks.buffer,
                cache_lengths.tensor,
                lookup_table.tensor,
                max_prompt_length.tensor,
                max_cache_length.tensor,
            )
            matmul_k_cache_ragged_scaled_float8(
                kv_params,
                hidden_states.tensor,
                input_row_offsets.tensor,
                weight.tensor,
                input_scale.tensor,
                weight_scale.tensor,
                kv_collection,
                scale_granularity,
                layer_idx.tensor,
            )

    # Test 1: hidden_states and weight dtype mismatch
    invalid_types = get_valid_input_types()
    invalid_types[0].dtype = DType.bfloat16  # hidden_states
    with pytest.raises(
        ValueError,
        match="expected hidden_states and weight to have the same dtype",
    ):
        try_create_graph(invalid_types)

    # Test 2: weight and hidden_states dtype mismatch (reverse case)
    invalid_types = get_valid_input_types()
    invalid_types[2].dtype = DType.bfloat16  # weight
    with pytest.raises(
        ValueError,
        match="expected hidden_states and weight to have the same dtype",
    ):
        try_create_graph(invalid_types)

    # Test 3: input_row_offsets with invalid dtype (should be uint32)
    invalid_types = get_valid_input_types()
    invalid_types[1].dtype = DType.int32  # input_row_offsets
    with pytest.raises(
        ValueError, match="expected input_row_offsets to have dtype uint32"
    ):
        try_create_graph(invalid_types)

    # Test 4: layer_idx with invalid dtype (should be uint32)
    invalid_types = get_valid_input_types()
    invalid_types[10].dtype = DType.int32  # layer_idx
    with pytest.raises(
        ValueError, match="expected layer_idx to have dtype uint32"
    ):
        try_create_graph(invalid_types)

    # Test 5: hidden_states with invalid rank (should be 2)
    invalid_types = get_valid_input_types()
    invalid_types[0] = TensorType(
        # rank 3
        DType.float8_e4m3fn,
        shape=(10, 512, 1),
        device=device,
    )
    with pytest.raises(
        ValueError, match="expected hidden_states to have rank 2"
    ):
        try_create_graph(invalid_types)


def test_grouped_dynamic_scaled_fp8_matmul_valid() -> None:
    """Tests grouped_dynamic_scaled_fp8_matmul with all tensors on same device."""
    device = DeviceRef.CPU()

    with Graph(
        "grouped_dynamic_scaled_fp8_matmul",
        input_types=[
            # hidden_states
            TensorType(DType.float8_e4m3fn, shape=(99, 512), device=device),
            # weight
            TensorType(DType.float8_e4m3fn, shape=(3, 256, 512), device=device),
            # a_scales
            TensorType(DType.float32, shape=(4, 99), device=device),
            # b_scales
            TensorType(DType.float32, shape=(3, 2, 4), device=device),
            # expert_start_indices
            TensorType(DType.uint32, shape=(1,), device=device),
            # expert_ids
            TensorType(DType.int32, shape=(3,), device=device),
            # expert_usage_stats_host
            TensorType(DType.uint32, shape=(2,), device=device),
        ],
    ) as graph:
        (
            hidden_states,
            weight,
            a_scales,
            b_scales,
            expert_start_indices,
            expert_ids,
            expert_usage_stats_host,
        ) = graph.inputs

        output = grouped_dynamic_scaled_fp8_matmul(
            hidden_states.tensor,
            weight.tensor,
            a_scales.tensor,
            b_scales.tensor,
            expert_start_indices.tensor,
            expert_ids.tensor,
            expert_usage_stats_host.tensor,
            input_scale_spec=InputScaleSpec(
                granularity=ScaleGranularity.BLOCK,
                origin=ScaleOrigin.DYNAMIC,
                dtype=DType.float32,
                block_size=(1, 128),
            ),
            weight_scale_spec=WeightScaleSpec(
                granularity=ScaleGranularity.BLOCK,
                dtype=DType.float32,
                block_size=(128, 128),
            ),
            out_type=DType.bfloat16,
        )
        assert output.shape == [99, 256]
        assert output.dtype == DType.bfloat16


def test_grouped_dynamic_scaled_fp8_matmul_invalid() -> None:
    """Tests grouped_dynamic_scaled_fp8_matmul with invalid inputs."""
    device = DeviceRef.CPU()

    def get_valid_input_types() -> list[TensorType | BufferType]:
        return [
            # hidden_states
            TensorType(DType.float8_e4m3fn, shape=(99, 512), device=device),
            # weight
            TensorType(DType.float8_e4m3fn, shape=(3, 256, 512), device=device),
            # a_scales
            TensorType(DType.float32, shape=(4, 99), device=device),
            # b_scales
            TensorType(DType.float32, shape=(3, 2, 4), device=device),
            # expert_start_indices
            TensorType(DType.uint32, shape=(1,), device=device),
            # expert_ids
            TensorType(DType.int32, shape=(3,), device=device),
            # expert_usage_stats_host
            TensorType(DType.uint32, shape=(2,), device=device),
        ]

    def try_create_graph(input_types: list[TensorType | BufferType]) -> None:
        with Graph(
            "grouped_dynamic_scaled_fp8_matmul",
            input_types=input_types,
        ) as graph:
            (
                hidden_states,
                weight,
                a_scales,
                b_scales,
                expert_start_indices,
                expert_ids,
                expert_usage_stats_host,
            ) = graph.inputs

        grouped_dynamic_scaled_fp8_matmul(
            hidden_states.tensor,
            weight.tensor,
            a_scales.tensor,
            b_scales.tensor,
            expert_start_indices.tensor,
            expert_ids.tensor,
            expert_usage_stats_host.tensor,
            input_scale_spec=InputScaleSpec(
                granularity=ScaleGranularity.BLOCK,
                origin=ScaleOrigin.DYNAMIC,
                dtype=DType.float32,
                block_size=(1, 128),
            ),
            weight_scale_spec=WeightScaleSpec(
                granularity=ScaleGranularity.BLOCK,
                dtype=DType.float32,
                block_size=(128, 128),
            ),
            out_type=DType.bfloat16,
        )

    # Test 1: hidden_states and weight dtype mismatch
    invalid_types = get_valid_input_types()
    invalid_types[0].dtype = DType.bfloat16  # hidden_states
    with pytest.raises(
        TypeError,
        match="hidden_states and weight dtypes must be float8_e4m3fn",
    ):
        try_create_graph(invalid_types)

    # Test 2: weight and hidden_states dtype mismatch (reverse case)
    invalid_types = get_valid_input_types()
    invalid_types[1].dtype = DType.bfloat16  # weight
    with pytest.raises(
        TypeError,
        match="hidden_states and weight dtypes must be float8_e4m3fn",
    ):
        try_create_graph(invalid_types)

    # Test 3: input_scale and weight_scale dtype mismatch
    invalid_types = get_valid_input_types()
    invalid_types[2].dtype = DType.float16  # input_scale
    with pytest.raises(
        TypeError,
        match="a_scales and b_scales dtypes must be float32",
    ):
        try_create_graph(invalid_types)

    # Test 4: hidden_states with invalid rank (should be 2)
    invalid_types = get_valid_input_types()
    invalid_types[0] = TensorType(
        # rank 3
        DType.float8_e4m3fn,
        shape=(10, 512, 1),
        device=device,
    )
    with pytest.raises(ValueError, match="expected hidden_states of rank 2"):
        try_create_graph(invalid_types)

    # Test 5: weight with invalid rank (should be 3)
    invalid_types = get_valid_input_types()
    invalid_types[1] = TensorType(
        # rank 2
        DType.float8_e4m3fn,
        shape=(3, 256),
        device=device,
    )
    with pytest.raises(ValueError, match="expected weight of rank 3 but got"):
        try_create_graph(invalid_types)

    # Test 6: hidden state and weight shape mismatch
    invalid_types = get_valid_input_types()
    invalid_types[0] = TensorType(
        # rank 2
        DType.float8_e4m3fn,
        shape=(3, 512),
        device=device,
    )
    invalid_types[1] = TensorType(
        # rank 3
        DType.float8_e4m3fn,
        shape=(99, 512, 1024),
        device=device,
    )
    with pytest.raises(ValueError, match="expected weight is of shape "):
        try_create_graph(invalid_types)

    # Test 7: weight shape and expert_ids mismatch
    invalid_types = get_valid_input_types()
    invalid_types[1] = TensorType(
        # rank 3
        DType.float8_e4m3fn,
        shape=(16, 512, 1024),
        device=device,
    )
    invalid_types[5] = TensorType(
        # rank 1
        DType.int32,
        shape=(15,),
        device=device,
    )
    with pytest.raises(ValueError, match="expected weight is of shape "):
        try_create_graph(invalid_types)


def test_batched_dynamic_scaled_fp8_matmul_valid() -> None:
    """Tests batched_dynamic_scaled_fp8_matmul with all tensors on same device."""
    device = DeviceRef.CPU()

    with Graph(
        "batched_dynamic_scaled_fp8_matmul",
        input_types=[
            # hidden_states
            TensorType(DType.float8_e4m3fn, shape=(3, 99, 512), device=device),
            # weight
            TensorType(DType.float8_e4m3fn, shape=(3, 256, 512), device=device),
            # a_scales
            TensorType(DType.float32, shape=(3, 4, 99), device=device),
            # b_scales
            TensorType(DType.float32, shape=(3, 2, 4), device=device),
        ],
    ) as graph:
        (
            hidden_states,
            weight,
            a_scales,
            b_scales,
        ) = graph.inputs

        output = batched_dynamic_scaled_fp8_matmul(
            hidden_states.tensor,
            weight.tensor,
            a_scales.tensor,
            b_scales.tensor,
            input_scale_spec=InputScaleSpec(
                granularity=ScaleGranularity.BLOCK,
                origin=ScaleOrigin.DYNAMIC,
                dtype=DType.float32,
                block_size=(1, 128),
            ),
            weight_scale_spec=WeightScaleSpec(
                granularity=ScaleGranularity.BLOCK,
                dtype=DType.float32,
                block_size=(128, 128),
            ),
            out_type=DType.bfloat16,
        )
        assert output.shape == [3, 99, 256]
        assert output.dtype == DType.bfloat16


def test_batched_dynamic_scaled_fp8_matmul_invalid() -> None:
    """Tests batched_dynamic_scaled_fp8_matmul with invalid inputs."""
    device = DeviceRef.CPU()

    def get_valid_input_types() -> list[TensorType | BufferType]:
        return [
            # hidden_states
            TensorType(DType.float8_e4m3fn, shape=(3, 99, 512), device=device),
            # weight
            TensorType(DType.float8_e4m3fn, shape=(3, 256, 512), device=device),
            # a_scales
            TensorType(DType.float32, shape=(3, 4, 99), device=device),
            # b_scales
            TensorType(DType.float32, shape=(3, 2, 4), device=device),
        ]

    def try_create_graph(input_types: list[TensorType | BufferType]) -> None:
        with Graph(
            "batched_dynamic_scaled_fp8_matmul",
            input_types=input_types,
        ) as graph:
            (
                hidden_states,
                weight,
                a_scales,
                b_scales,
            ) = graph.inputs

        batched_dynamic_scaled_fp8_matmul(
            hidden_states.tensor,
            weight.tensor,
            a_scales.tensor,
            b_scales.tensor,
            input_scale_spec=InputScaleSpec(
                granularity=ScaleGranularity.BLOCK,
                origin=ScaleOrigin.DYNAMIC,
                dtype=DType.float32,
                block_size=(1, 128),
            ),
            weight_scale_spec=WeightScaleSpec(
                granularity=ScaleGranularity.BLOCK,
                dtype=DType.float32,
                block_size=(128, 128),
            ),
            out_type=DType.bfloat16,
        )

    # Test 1: hidden_states and weight dtype mismatch
    invalid_types = get_valid_input_types()
    invalid_types[0].dtype = DType.bfloat16  # hidden_states
    with pytest.raises(
        TypeError,
        match="a and b dtypes must match,",
    ):
        try_create_graph(invalid_types)

    # Test 2: weight and hidden_states dtype mismatch (reverse case)
    invalid_types = get_valid_input_types()
    invalid_types[1].dtype = DType.bfloat16  # weight
    with pytest.raises(
        TypeError,
        match="a and b dtypes must match,",
    ):
        try_create_graph(invalid_types)

    # Test 3: input_scale and weight_scale dtype mismatch
    invalid_types = get_valid_input_types()
    invalid_types[2].dtype = DType.float16  # input_scale
    with pytest.raises(
        TypeError,
        match="a_scales and b_scales dtypes must be float32,",
    ):
        try_create_graph(invalid_types)

    # Test 4: hidden_states with invalid rank (should be 3)
    invalid_types = get_valid_input_types()
    invalid_types[0] = TensorType(
        # rank 2
        DType.float8_e4m3fn,
        shape=(10, 512),
        device=device,
    )
    with pytest.raises(ValueError, match="A and B must be rank 3 tensors"):
        try_create_graph(invalid_types)

    # Test 5: weight with invalid rank (should be 3)
    invalid_types = get_valid_input_types()
    invalid_types[1] = TensorType(
        # rank 2
        DType.float8_e4m3fn,
        shape=(3, 256),
        device=device,
    )
    with pytest.raises(ValueError, match="A and B must be rank 3 tensors"):
        try_create_graph(invalid_types)

    # Test 6: a_scales with invalid rank (should be 3)
    invalid_types = get_valid_input_types()
    invalid_types[2] = TensorType(
        # rank 2
        DType.float32,
        shape=(10, 512),
        device=device,
    )
    with pytest.raises(
        ValueError, match="A_scales and B_scales must be rank 3 tensors"
    ):
        try_create_graph(invalid_types)

    # Test 7: b_scales with invalid rank (should be 3)
    invalid_types = get_valid_input_types()
    invalid_types[3] = TensorType(
        # rank 2
        DType.float32,
        shape=(3, 256),
        device=device,
    )
    with pytest.raises(
        ValueError, match="A_scales and B_scales must be rank 3 tensors"
    ):
        try_create_graph(invalid_types)
