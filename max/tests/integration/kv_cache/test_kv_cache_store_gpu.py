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

from __future__ import annotations

import numpy as np
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import (
    store_k_cache_padded,
    store_k_cache_ragged,
    store_k_scale_cache_ragged,
)
from max.nn.kv_cache import (
    KVCacheQuantizationConfig,
    MHAKVCacheParams,
    PagedCacheValues,
)
from test_common.simple_kv_cache import paged_kv_cache_inputs

TOTAL_NUM_PAGES = 16


def _kv_params() -> MHAKVCacheParams:
    return MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=8,
        head_dim=64,
        num_layers=1,
        page_size=32,
        devices=[DeviceRef.GPU()],
    )


def test_kv_cache_store_ragged_executes() -> None:
    device = Accelerator()
    kv_params = _kv_params()

    prompt_lens = [33, 66, 1]
    batch_size = len(prompt_lens)
    total_seq_len = sum(prompt_lens)

    x_cache_type = TensorType(
        DType.float32,
        [total_seq_len, kv_params.n_kv_heads, kv_params.head_dim],
        device=DeviceRef.GPU(),
    )
    offsets_type = TensorType(
        DType.uint32,
        [batch_size + 1],
        device=DeviceRef.GPU(),
    )
    kv_symbolic_inputs = kv_params.get_symbolic_inputs().inputs[0]
    blocks_type = kv_symbolic_inputs.kv_blocks
    cache_lengths_type = kv_symbolic_inputs.cache_lengths
    lookup_table_type = kv_symbolic_inputs.lookup_table
    max_prompt_length_type = kv_symbolic_inputs.max_prompt_length
    max_cache_length_type = kv_symbolic_inputs.max_cache_length

    with Graph(
        "kv_cache_store_ragged",
        input_types=[
            x_cache_type,
            offsets_type,
            blocks_type,
            cache_lengths_type,
            lookup_table_type,
            max_prompt_length_type,
            max_cache_length_type,
        ],
    ) as graph:
        (
            x_cache_in,
            input_row_offsets_in,
            blocks_in,
            cache_lengths_in,
            lookup_table_in,
            max_prompt_length_in,
            max_cache_length_in,
        ) = graph.inputs
        kv_collection = PagedCacheValues(
            blocks_in.buffer,
            cache_lengths_in.tensor,
            lookup_table_in.tensor,
            max_prompt_length_in.tensor,
            max_cache_length_in.tensor,
        )
        layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())
        store_k_cache_ragged(
            kv_collection,
            x_cache_in.tensor,
            input_row_offsets_in.tensor,
            layer_idx,
        )
        graph.output(x_cache_in.tensor)

    session = InferenceSession(devices=[device])
    model = session.load(graph)
    runtime_inputs = paged_kv_cache_inputs(
        kv_params, prompt_lens, total_num_pages=TOTAL_NUM_PAGES
    )
    assert not runtime_inputs.kv_blocks.to_numpy().any()

    offsets = np.array(
        [0, prompt_lens[0], prompt_lens[0] + prompt_lens[1], total_seq_len],
        dtype=np.uint32,
    )
    rng = np.random.default_rng(0)
    x_cache_np = rng.standard_normal(
        x_cache_type.shape.static_dims, dtype=np.float32
    )
    x_cache_data = Buffer.from_numpy(x_cache_np).to(device)
    offsets_data = Buffer.from_numpy(offsets).to(device)
    model(
        x_cache_data,
        offsets_data,
        runtime_inputs.kv_blocks,
        runtime_inputs.cache_lengths,
        runtime_inputs.lookup_table,
        runtime_inputs.max_prompt_length,
        runtime_inputs.max_cache_length,
    )

    assert runtime_inputs.kv_blocks.to_numpy().any()


def test_kv_cache_store_padded_executes() -> None:
    device = Accelerator()
    kv_params = _kv_params()

    valid_lengths = [33, 66, 1]
    batch_size = len(valid_lengths)
    padded_seq_len = max(valid_lengths)

    x_cache_type = TensorType(
        DType.float32,
        [batch_size, padded_seq_len, kv_params.n_kv_heads, kv_params.head_dim],
        device=DeviceRef.GPU(),
    )
    valid_lengths_type = TensorType(
        DType.uint32,
        [batch_size],
        device=DeviceRef.GPU(),
    )
    kv_symbolic_inputs = kv_params.get_symbolic_inputs().inputs[0]
    blocks_type = kv_symbolic_inputs.kv_blocks
    cache_lengths_type = kv_symbolic_inputs.cache_lengths
    lookup_table_type = kv_symbolic_inputs.lookup_table
    max_prompt_length_type = kv_symbolic_inputs.max_prompt_length
    max_cache_length_type = kv_symbolic_inputs.max_cache_length

    with Graph(
        "kv_cache_store_padded",
        input_types=[
            x_cache_type,
            valid_lengths_type,
            blocks_type,
            cache_lengths_type,
            lookup_table_type,
            max_prompt_length_type,
            max_cache_length_type,
        ],
    ) as graph:
        (
            x_cache_in,
            valid_lengths_in,
            blocks_in,
            cache_lengths_in,
            lookup_table_in,
            max_prompt_length_in,
            max_cache_length_in,
        ) = graph.inputs
        kv_collection = PagedCacheValues(
            blocks_in.buffer,
            cache_lengths_in.tensor,
            lookup_table_in.tensor,
            max_prompt_length_in.tensor,
            max_cache_length_in.tensor,
        )
        layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())
        store_k_cache_padded(
            kv_collection,
            x_cache_in.tensor,
            valid_lengths_in.tensor,
            layer_idx,
        )
        graph.output(x_cache_in.tensor)

    session = InferenceSession(devices=[device])
    model = session.load(graph)
    runtime_inputs = paged_kv_cache_inputs(
        kv_params, valid_lengths, total_num_pages=TOTAL_NUM_PAGES
    )
    assert not runtime_inputs.kv_blocks.to_numpy().any()

    lengths = np.array(valid_lengths, dtype=np.uint32)
    rng = np.random.default_rng(1)
    x_cache_np = rng.standard_normal(
        x_cache_type.shape.static_dims, dtype=np.float32
    )
    x_cache_data = Buffer.from_numpy(x_cache_np).to(device)
    lengths_data = Buffer.from_numpy(lengths).to(device)
    model(
        x_cache_data,
        lengths_data,
        runtime_inputs.kv_blocks,
        runtime_inputs.cache_lengths,
        runtime_inputs.lookup_table,
        runtime_inputs.max_prompt_length,
        runtime_inputs.max_cache_length,
    )

    assert runtime_inputs.kv_blocks.to_numpy().any()


def _kv_params_fp8() -> MHAKVCacheParams:
    """Page layout for an FP8 quantized cache, which also carries kv_scales."""
    return MHAKVCacheParams(
        dtype=DType.float8_e4m3fn,
        n_kv_heads=1,
        head_dim=128,
        num_layers=1,
        page_size=128,
        devices=[DeviceRef.GPU()],
        kvcache_quant_config=KVCacheQuantizationConfig(
            scale_dtype=DType.float32,
            quantization_granularity=128,
        ),
    )


def test_store_k_scale_cache_executes() -> None:
    """Test that store_k_scale_cache kernel executes and writes to kv_scales buffer."""
    device = Accelerator()
    kv_params = _kv_params_fp8()

    prompt_lens = [33, 66, 1]
    batch_size = len(prompt_lens)
    total_seq_len = sum(prompt_lens)

    assert kv_params.kvcache_quant_config is not None
    quantization_granularity = (
        kv_params.kvcache_quant_config.quantization_granularity
    )
    head_dim_granularity = kv_params.head_dim // quantization_granularity

    x_k_scale_type = TensorType(
        DType.float32,
        [total_seq_len, kv_params.n_kv_heads, head_dim_granularity],
        device=DeviceRef.GPU(),
    )
    offsets_type = TensorType(
        DType.uint32,
        [batch_size + 1],
        device=DeviceRef.GPU(),
    )

    kv_symbolic_inputs = kv_params.get_symbolic_inputs()

    with Graph(
        "store_k_scale_cache",
        input_types=[
            x_k_scale_type,
            offsets_type,
            *kv_symbolic_inputs.flatten(),
        ],
    ) as graph:
        x_k_scale_in = graph.inputs[0].tensor
        input_row_offsets_in = graph.inputs[1].tensor
        blocks_in = graph.inputs[2].buffer
        cache_lengths_in = graph.inputs[3].tensor
        lookup_table_in = graph.inputs[4].tensor
        max_prompt_length_in = graph.inputs[5].tensor
        max_cache_length_in = graph.inputs[6].tensor
        kv_scales_in = graph.inputs[7].buffer if len(graph.inputs) > 7 else None

        kv_collection = PagedCacheValues(
            kv_blocks=blocks_in,
            cache_lengths=cache_lengths_in,
            lookup_table=lookup_table_in,
            max_prompt_length=max_prompt_length_in,
            max_cache_length=max_cache_length_in,
            kv_scales=kv_scales_in,
        )

        layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())
        store_k_scale_cache_ragged(
            kv_collection,
            x_k_scale_in,
            input_row_offsets_in,
            layer_idx,
            quantization_granularity,
        )
        graph.output(x_k_scale_in)

    session = InferenceSession(devices=[device])
    model = session.load(graph)

    runtime_inputs = paged_kv_cache_inputs(
        kv_params, prompt_lens, total_num_pages=8
    )
    assert runtime_inputs.kv_scales is not None
    assert not runtime_inputs.kv_scales.to_numpy().any()

    offsets = np.array(
        [0, prompt_lens[0], prompt_lens[0] + prompt_lens[1], total_seq_len],
        dtype=np.uint32,
    )
    rng = np.random.default_rng(42)
    x_k_scale_np = rng.standard_normal(
        x_k_scale_type.shape.static_dims, dtype=np.float32
    )
    x_k_scale_data = Buffer.from_numpy(x_k_scale_np).to(device)
    offsets_data = Buffer.from_numpy(offsets).to(device)

    model(
        x_k_scale_data,
        offsets_data,
        *runtime_inputs.flatten(),
    )

    assert runtime_inputs.kv_scales.to_numpy().any()
