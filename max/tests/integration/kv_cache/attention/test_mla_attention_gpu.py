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
"""Test pipelines MLA attention layer."""

import numpy as np
from max.driver import CPU, Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import flare_mla_prefill_ragged
from max.nn.kv_cache import MHAKVCacheParams, PagedCacheValues
from test_common.simple_kv_cache import paged_kv_cache_inputs


def test_kv_cache_paged_mla_prefill(gpu_session: InferenceSession) -> None:
    device = Accelerator()
    session = gpu_session
    num_q_heads = 32
    q_head_dim = 192
    k_head_dim = 128
    num_layers = 1
    kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=1,
        head_dim=576,
        num_layers=num_layers,
        page_size=128,
        devices=[DeviceRef.GPU()],
    )
    prompt_lens = [10, 30]
    batch_size = len(prompt_lens)
    total_seq_len = sum(prompt_lens)
    input_type = TensorType(
        DType.bfloat16,
        ["total_seq_len", num_q_heads, q_head_dim],
        DeviceRef.GPU(),
    )
    k_buffer_type = TensorType(
        DType.bfloat16,
        ["total_seq_len", num_q_heads, k_head_dim],
        DeviceRef.GPU(),
    )
    v_buffer_type = TensorType(
        DType.bfloat16,
        ["total_seq_len", num_q_heads, k_head_dim],
        DeviceRef.GPU(),
    )
    input_row_offsets_type = TensorType(
        DType.uint32, ["input_row_offsets_len"], DeviceRef.GPU()
    )

    def construct() -> Graph:
        with Graph(
            "call_mla_prefill",
            input_types=[
                input_type,
                input_row_offsets_type,
                k_buffer_type,
                v_buffer_type,
                *kv_params.flattened_kv_inputs(),
            ],
        ) as g:
            (
                input,
                input_row_offsets,
                k_buffer,
                v_buffer,
                blocks,
                cache_lengths,
                lookup_table,
                max_prompt_length,
                max_cache_length,
                _attention_dispatch_metadata,
            ) = g.inputs

            layer_idx = ops.constant(0, DType.uint32, DeviceRef.CPU())

            kv_collection = PagedCacheValues(
                blocks.buffer,
                cache_lengths.tensor,
                lookup_table.tensor,
                max_prompt_length.tensor,
                max_cache_length.tensor,
            )
            result = flare_mla_prefill_ragged(
                kv_params,
                input.tensor,
                k_buffer.tensor,
                v_buffer.tensor,
                input_row_offsets.tensor,
                input_row_offsets.tensor,  # actually buffer_row_offsets
                cache_lengths.tensor,
                kv_collection,
                layer_idx,
                MHAMaskVariant.CAUSAL_MASK,
                1,  # scale
            )
            g.output(result.cast(DType.float32))
        return g

    g = construct()

    input_row_offsets = Buffer(
        DType.uint32,
        [batch_size + 1],
    )
    running_sum = 0
    for i in range(batch_size):
        input_row_offsets[i] = running_sum
        running_sum += prompt_lens[i]
    input_row_offsets[batch_size] = running_sum
    input_row_offsets = input_row_offsets.to(device)

    kv_runtime_inputs = paged_kv_cache_inputs(
        kv_params, prompt_lens, total_num_pages=8
    )
    model = session.load(g)

    input_tensor = Buffer.zeros(
        (total_seq_len, num_q_heads, q_head_dim), dtype=DType.bfloat16
    )
    k_buffer_tensor = Buffer.zeros(
        (total_seq_len, num_q_heads, k_head_dim), dtype=DType.bfloat16
    )
    v_buffer_tensor = Buffer.zeros(
        (total_seq_len, num_q_heads, k_head_dim), dtype=DType.bfloat16
    )

    result = model.execute(
        input_tensor.to(device),
        input_row_offsets.to(device),
        k_buffer_tensor.to(device),
        v_buffer_tensor.to(device),
        *kv_runtime_inputs.flatten(),
    )[0]
    assert isinstance(result, Buffer)

    host = CPU(0)
    assert np.all(np.isfinite(result.to(host).to_numpy()))
