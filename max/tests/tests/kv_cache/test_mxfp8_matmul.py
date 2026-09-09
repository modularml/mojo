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

"""Build-time tests for the MXFP8 fused QKV matmul kernel in max.nn.kernels."""

from __future__ import annotations

import pytest
from max.dtype import DType
from max.graph import BufferType, DeviceRef, Graph, TensorType, TensorValue
from max.nn.kernels import (
    _fused_qkv_ragged_matmul_scaled_mxfp8 as fused_qkv_ragged_matmul_scaled_mxfp8,
)
from max.nn.kernels import (
    block_scales_interleave,
    quantize_dynamic_block_scaled,
)
from max.nn.kv_cache import (
    KVCacheParams,
    MHAKVCacheParams,
    PagedCacheValues,
)

_HIDDEN = 512
_HEAD_DIM = 64
_N_HEADS = 32
_N_KV_HEADS = 8
_QKV_DIM = (_N_HEADS + 2 * _N_KV_HEADS) * _HEAD_DIM


class FusedQKVRaggedMatmulScaledMXFP8:
    """Wrapper mirroring how quantized_fused_qkv_matmul drives the MXFP8 op."""

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
        weight_scale: TensorValue,
    ) -> TensorValue:
        x, x_scales = quantize_dynamic_block_scaled(
            input,
            sf_vector_size=32,
            scales_type=DType.float8_e8m0fnu,
            out_type=DType.float8_e4m3fn,
        )
        weight_scale = block_scales_interleave(weight_scale, sf_vector_size=32)

        return fused_qkv_ragged_matmul_scaled_mxfp8(
            self.kv_params,
            x,
            input_row_offsets,
            wqkv,
            self.kv_collection,
            layer_idx,
            self.n_heads,
            x_scales,
            weight_scale,
        )


def _build_graph(device: DeviceRef) -> TensorValue:
    kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=_N_KV_HEADS,
        head_dim=_HEAD_DIM,
        num_layers=1,
        page_size=128,
        devices=[device],
    )

    with Graph(
        "fused_qkv_ragged_matmul_scaled_mxfp8",
        input_types=[
            TensorType(DType.bfloat16, shape=(10, _HIDDEN), device=device),
            TensorType(DType.uint32, shape=(3,), device=device),
            TensorType(
                DType.float8_e4m3fn,
                shape=(_QKV_DIM, _HIDDEN),
                device=device,
            ),
            TensorType(DType.uint32, shape=(), device=DeviceRef.CPU()),
            TensorType(
                DType.float8_e8m0fnu,
                shape=(_QKV_DIM, _HIDDEN // 32),
                device=device,
            ),
            # blocks: [num_pages, 2, n_kv_heads, page_size, head_dim]
            BufferType(
                DType.bfloat16,
                shape=(16, 2, _N_KV_HEADS, 128, _HEAD_DIM),
                device=device,
            ),
            TensorType(DType.uint32, shape=(2,), device=device),
            TensorType(DType.uint32, shape=(2, 8), device=device),
            TensorType(DType.uint32, shape=(1,), device=device),
            TensorType(DType.uint32, shape=(1,), device=device),
        ],
    ) as graph:
        (
            input_tensor,
            input_row_offsets,
            wqkv,
            layer_idx,
            weight_scale,
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

        tester = FusedQKVRaggedMatmulScaledMXFP8(
            kv_params, kv_collection, _N_HEADS
        )
        output = tester(
            input_tensor.tensor,
            input_row_offsets.tensor,
            wqkv.tensor,
            layer_idx.tensor,
            weight_scale.tensor,
        )
        graph.output(output)
        return output


def test_fused_qkv_ragged_matmul_scaled_mxfp8_valid() -> None:
    """Builds the MXFP8 fused QKV graph with all tensors on the same device."""
    # The Q projection is the only graph output. K and V are written in place.
    output = _build_graph(DeviceRef.CPU())
    assert output.shape == [10, _N_HEADS * _HEAD_DIM]
    assert output.dtype == DType.bfloat16


def test_fused_qkv_ragged_matmul_scaled_mxfp8_device_mismatch() -> None:
    """The kernel rejects operands that do not share the input's device."""
    device = DeviceRef.CPU()
    kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=_N_KV_HEADS,
        head_dim=_HEAD_DIM,
        num_layers=1,
        page_size=128,
        devices=[device],
    )
    with Graph(
        "fused_qkv_ragged_matmul_scaled_mxfp8_bad_device",
        input_types=[
            TensorType(DType.float8_e4m3fn, shape=(10, _HIDDEN), device=device),
            TensorType(DType.uint32, shape=(3,), device=device),
            TensorType(
                DType.float8_e4m3fn,
                shape=(_QKV_DIM, _HIDDEN),
                device=device,
            ),
            TensorType(DType.uint32, shape=(), device=DeviceRef.CPU()),
            TensorType(
                DType.float8_e8m0fnu,
                shape=(1, 1, 32, 4, 4),
                device=device,
            ),
            # weight_scale on a different (GPU) device than input.
            TensorType(
                DType.float8_e8m0fnu,
                shape=(1, 1, 32, 4, 4),
                device=DeviceRef.GPU(),
            ),
            BufferType(
                DType.bfloat16,
                shape=(16, 2, _N_KV_HEADS, 128, _HEAD_DIM),
                device=device,
            ),
            TensorType(DType.uint32, shape=(2,), device=device),
            TensorType(DType.uint32, shape=(2, 8), device=device),
            TensorType(DType.uint32, shape=(1,), device=device),
            TensorType(DType.uint32, shape=(1,), device=device),
        ],
    ) as graph:
        (
            x,
            input_row_offsets,
            wqkv,
            layer_idx,
            input_scale,
            weight_scale,
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
        with pytest.raises(ValueError, match="same device"):
            fused_qkv_ragged_matmul_scaled_mxfp8(
                kv_params,
                x.tensor,
                input_row_offsets.tensor,
                wqkv.tensor,
                kv_collection,
                layer_idx.tensor,
                _N_HEADS,
                input_scale.tensor,
                weight_scale.tensor,
            )
