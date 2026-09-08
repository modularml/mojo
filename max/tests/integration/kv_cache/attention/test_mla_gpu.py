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


import numpy as np
import pytest
import torch
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kernels import flare_mla_decompress_k_cache, flare_mla_prefill_plan
from max.nn.kv_cache import MHAKVCacheParams, MLAKVCacheParams
from test_common.simple_kv_cache import paged_kv_cache_inputs
from torch.utils.dlpack import from_dlpack


def test_mla_prefill_plan() -> None:
    """Tests the mla_prefill_plan custom op."""
    # Set up hyperparameters for the test.
    device0 = Accelerator(0)
    session = InferenceSession(devices=[device0])

    page_size = 128
    kv_params = MLAKVCacheParams(
        dtype=DType.bfloat16,
        head_dim=128,
        num_layers=1,
        page_size=page_size,
        num_q_heads=8,
        devices=[DeviceRef.GPU()],
    )
    prompt_lens = [160, 200]
    batch_size = len(prompt_lens)
    buffer_tok_size = 256

    # Set MLIR types for the graph.
    input_row_offsets_type = TensorType(
        DType.uint32, shape=["input_row_offsets_len"], device=DeviceRef.GPU()
    )

    def construct() -> Graph:
        with Graph(
            "call_mla_prefill_plan",
            input_types=[
                input_row_offsets_type,
                *kv_params.flattened_kv_inputs(),
            ],
        ) as g:
            input_row_offsets = g.inputs[0].tensor
            layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())

            kv_collection = (
                kv_params.get_symbolic_inputs()
                .unflatten(iter(g.inputs[1:]))
                .inputs[0]
            )

            results = flare_mla_prefill_plan(
                kv_params,
                input_row_offsets,
                kv_collection,
                layer_idx,
                buffer_tok_size,
            )

            g.output(results[0].tensor, results[1].tensor, results[2].tensor)
        return g

    graph = construct()

    # Compile and init the model.
    model = session.load(graph)

    # Compute input row offsets for ragged tensors.
    input_row_offsets = Buffer(DType.uint32, [batch_size + 1])
    running_sum = 0
    for i in range(batch_size):
        input_row_offsets[i] = running_sum
        running_sum += prompt_lens[i]
    input_row_offsets[batch_size] = running_sum

    kv_inputs = paged_kv_cache_inputs(kv_params, prompt_lens, total_num_pages=8)

    results = model.execute(input_row_offsets.to(device0), *kv_inputs.flatten())

    # Hardcoded reference for:
    # page_size = 128, buffer_tok_size = 256, prompt_lens = [160, 200]
    # aligned lengths = [256, 256]
    buffer_row_offsets_ref = np.zeros((16, batch_size + 1), dtype=np.uint32)
    buffer_row_offsets_ref[0] = np.array([0, 256, 256], dtype=np.uint32)
    buffer_row_offsets_ref[1] = np.array([0, 0, 256], dtype=np.uint32)

    cache_offsets_ref = np.zeros((16, batch_size), dtype=np.uint32)
    cache_offsets_ref[0] = np.array([0, 0], dtype=np.uint32)
    cache_offsets_ref[1] = np.array([256, 0], dtype=np.uint32)
    cache_offsets_ref[2:16] = np.array([256, 256], dtype=np.uint32)

    buffer_lengths_ref = -1 * np.ones((16,), dtype=np.int32)
    buffer_lengths_ref[0:2] = 256

    assert np.all(
        from_dlpack(results[0]).cpu().numpy() == buffer_row_offsets_ref
    )
    max_cache_offsets = from_dlpack(results[1]).cpu().numpy()
    # the last row is not used.
    assert np.all(max_cache_offsets[:, :-1] == cache_offsets_ref)
    assert np.all(from_dlpack(results[2]).cpu().numpy() == buffer_lengths_ref)


def test_mla_decompress_k_cache() -> None:
    """Tests the mla_decompress_k_cache custom op."""
    # Set up hyperparameters for the test.
    device0 = Accelerator(0)
    session = InferenceSession(devices=[device0])

    page_size = 128
    kv_params = MLAKVCacheParams(
        dtype=DType.float32,
        head_dim=576,
        num_layers=1,
        page_size=page_size,
        num_q_heads=128,
        devices=[DeviceRef.GPU()],
    )
    prompt_lens = [10, 30]
    batch_size = len(prompt_lens)

    # Set MLIR types for the graph.
    input_row_offsets_type = TensorType(
        DType.uint32, ["input_row_offsets_len"], device=DeviceRef.GPU()
    )
    weight_type = TensorType(
        DType.float32,
        [4096, 512],
        device=DeviceRef.GPU(),
    )

    def construct() -> Graph:
        with Graph(
            "call_mla_decompress_k_cache",
            input_types=[
                input_row_offsets_type,
                weight_type,
                *kv_params.flattened_kv_inputs(),
            ],
        ) as g:
            input_row_offsets = g.inputs[0].tensor
            weight = g.inputs[1].tensor
            layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())

            kv_collection = (
                kv_params.get_symbolic_inputs()
                .unflatten(iter(g.inputs[2:]))
                .inputs[0]
            )

            # Allocate a page-aligned buffer to hold decompressed KV cache.
            buffer_tok_size = 256

            (buffer_row_offsets, cache_offsets, buffer_lengths) = (
                flare_mla_prefill_plan(
                    kv_params,
                    input_row_offsets,
                    kv_collection,
                    layer_idx,
                    buffer_tok_size,
                )
            )

            buffer_lengths_host = buffer_lengths.to(DeviceRef.CPU())

            result = flare_mla_decompress_k_cache(
                kv_params,
                buffer_row_offsets[0, :],  # Process first chunk only
                cache_offsets[0, :],
                buffer_lengths_host[0],
                weight,
                kv_collection,
                layer_idx,
                buffer_tok_size,
            )

            g.output(result)
        return g

    graph = construct()

    # Compile and init the model.
    model = session.load(graph)

    # Compute input row offsets for ragged tensors.
    input_row_offsets = Buffer(DType.uint32, [batch_size + 1])
    running_sum = 0
    for i in range(batch_size):
        input_row_offsets[i] = running_sum
        running_sum += prompt_lens[i]
    input_row_offsets[batch_size] = running_sum

    kv_runtime_inputs = paged_kv_cache_inputs(
        kv_params, prompt_lens, total_num_pages=8
    )

    new_blocks = torch.randn(
        size=kv_runtime_inputs.kv_blocks.shape, dtype=torch.float32
    )

    kv_runtime_inputs.kv_blocks = Buffer.from_numpy(new_blocks.numpy()).to(
        device0
    )

    assert kv_runtime_inputs.attention_dispatch_metadata is not None

    weight = (
        torch.randn(size=weight_type.shape.static_dims, dtype=torch.float32)
        / 128.0
    )

    results = model.execute(
        input_row_offsets.to(device0),
        Buffer.from_numpy(weight.numpy()).to(device0),
        *kv_runtime_inputs.flatten(),
    )

    # With page-aligned spans and 256-token chunks, chunk 0 covers request 0 and 1.
    # blocks shape: [block_num, kv_dim, layers, page_size, num_heads, head_dim]
    ref_chunk0_cache = torch.concatenate(
        (
            new_blocks[0, 0, 0, :page_size, 0, :512],
            new_blocks[1, 0, 0, :page_size, 0, :512],
        ),
        dim=0,
    )
    ref_output = ref_chunk0_cache @ weight.T

    graph_output = from_dlpack(results[0]).cpu()

    torch.testing.assert_close(
        ref_output,
        graph_output,
        rtol=1e-3,
        atol=1e-3,
    )


def test_mla_decompress_k_cache_only_k() -> None:
    """Tests the mla_decompress_k_cache custom op."""
    # Set up hyperparameters for the test.
    device0 = Accelerator(0)
    session = InferenceSession(devices=[device0])

    page_size = 128
    kv_params = MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=1,
        head_dim=576,
        num_layers=1,
        page_size=page_size,
        devices=[DeviceRef.GPU()],
    )

    # Set MLIR types for the graph.
    input_row_offsets_type = TensorType(
        DType.uint32, ["input_row_offsets_len"], device=DeviceRef.GPU()
    )
    weight_type = TensorType(
        DType.float32,
        [4096, 512],
        device=DeviceRef.GPU(),
    )

    def construct() -> Graph:
        with Graph(
            "call_mla_decompress_k_cache",
            input_types=[
                input_row_offsets_type,
                weight_type,
                *kv_params.flattened_kv_inputs(),
            ],
        ) as g:
            input_row_offsets = g.inputs[0].tensor
            weight = g.inputs[1].tensor
            layer_idx = ops.constant(0, DType.uint32, device=DeviceRef.CPU())

            kv_collection = (
                kv_params.get_symbolic_inputs()
                .unflatten(iter(g.inputs[2:]))
                .inputs[0]
            )

            # Allocate a buffer to hold KV cache for 60 decompressed tokens
            buffer_tok_size = 60

            (buffer_row_offsets, cache_offsets, buffer_lengths) = (
                flare_mla_prefill_plan(
                    kv_params,
                    input_row_offsets,
                    kv_collection,
                    layer_idx,
                    buffer_tok_size,
                )
            )

            buffer_lengths_host = buffer_lengths.to(DeviceRef.CPU())

            result = flare_mla_decompress_k_cache(
                kv_params,
                buffer_row_offsets[0, :],  # Process first chunk only
                cache_offsets[0, :],
                buffer_lengths_host[0],
                weight,
                kv_collection,
                layer_idx,
                buffer_tok_size,
            )

            g.output(result)
        return g

    with pytest.raises(Exception):
        graph = construct()
        _ = session.load(graph)
