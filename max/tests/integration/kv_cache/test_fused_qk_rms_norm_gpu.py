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

from dataclasses import dataclass

import numpy as np
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Dim, Graph, TensorType, TensorValue, ops
from max.graph.buffer_utils import cast_tensor_to
from max.nn.kernels import fused_qk_ragged_rms_norm, rms_norm_key_cache
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParams,
    MHAKVCacheParams,
    PagedCacheValues,
)
from test_common.simple_kv_cache import paged_kv_cache_inputs


@dataclass(frozen=True)
class FusedQKRMSNormModel:
    kv_params: KVCacheParams
    layer_idx: int
    total_seq_len: int
    epsilon: float
    weight_offset: float

    def __call__(
        self,
        q: TensorValue,
        q_gamma: TensorValue,
        k_gamma: TensorValue,
        input_row_offsets: TensorValue,
        *graph_inputs: TensorValue,
    ) -> tuple[TensorValue, TensorValue]:
        kv_collection = PagedCacheValues(
            kv_blocks=graph_inputs[0].buffer,
            cache_lengths=graph_inputs[1].tensor,
            lookup_table=graph_inputs[2].tensor,
            max_prompt_length=graph_inputs[3].tensor,
            max_cache_length=graph_inputs[4].tensor,
        )
        layer_idx = ops.constant(
            self.layer_idx, DType.uint32, device=DeviceRef.CPU()
        )
        q_norm = fused_qk_ragged_rms_norm(
            self.kv_params,
            q,
            input_row_offsets,
            kv_collection,
            q_gamma,
            k_gamma,
            self.epsilon,
            layer_idx,
            self.weight_offset,
        )

        q_ref = ops.rms_norm(
            ops.reshape(q, shape=[-1, self.kv_params.head_dim]),
            weight=q_gamma,
            epsilon=self.epsilon,
            weight_offset=self.weight_offset,
        )
        q_ref = ops.reshape(q_ref, shape=q.shape)

        return ops.cast(q_norm, DType.float32), ops.cast(q_ref, DType.float32)


@dataclass(frozen=True)
class UnfusedKeyRMSNormModel:
    kv_params: KVCacheParams
    layer_idx: int
    total_seq_len: int
    epsilon: float
    weight_offset: float

    def __call__(
        self,
        k_gamma: TensorValue,
        input_row_offsets: TensorValue,
        *graph_inputs: TensorValue,
    ) -> None:
        rms_norm_key_cache(
            self.kv_params,
            PagedCacheValues(
                kv_blocks=graph_inputs[0].buffer,
                cache_lengths=graph_inputs[1].tensor,
                lookup_table=graph_inputs[2].tensor,
                max_prompt_length=graph_inputs[3].tensor,
                max_cache_length=graph_inputs[4].tensor,
            ),
            gamma=k_gamma,
            epsilon=self.epsilon,
            layer_idx=ops.constant(
                self.layer_idx, DType.uint32, device=DeviceRef.CPU()
            ),
            total_seq_len=Dim(self.total_seq_len),
            input_row_offsets=input_row_offsets,
            weight_offset=self.weight_offset,
            multiply_before_cast=True,
            per_head_norm=True,
        )


def test_fused_qk_rms_norm_matches_unfused_gpu() -> None:
    dtype = DType.bfloat16
    device = Accelerator()
    device_ref = DeviceRef.GPU()
    session = InferenceSession(devices=[device])

    seq_len = 4
    n_q_heads = 4
    layer_idx = 0
    epsilon = 1e-5
    weight_offset = 1.0
    kv_params = MHAKVCacheParams(
        dtype=dtype,
        n_kv_heads=2,
        head_dim=64,
        num_layers=1,
        page_size=128,
        devices=[device_ref],
    )

    input_types = [
        TensorType(
            dtype,
            [seq_len, n_q_heads, kv_params.head_dim],
            device=device_ref,
        ),
        TensorType(dtype, [kv_params.head_dim], device=device_ref),
        TensorType(dtype, [kv_params.head_dim], device=device_ref),
        TensorType(DType.uint32, [2], device=device_ref),
        *kv_params.flattened_kv_inputs(),
    ]
    fused_graph = Graph(
        "fused_qk_rms_norm",
        forward=FusedQKRMSNormModel(
            kv_params, layer_idx, seq_len, epsilon, weight_offset
        ),
        input_types=input_types,
    )
    fused_model = session.load(fused_graph)

    unfused_graph = Graph(
        "unfused_k_rms_norm",
        forward=UnfusedKeyRMSNormModel(
            kv_params, layer_idx, seq_len, epsilon, weight_offset
        ),
        input_types=[
            TensorType(dtype, [kv_params.head_dim], device=device_ref),
            TensorType(DType.uint32, [2], device=device_ref),
            *kv_params.flattened_kv_inputs(),
        ],
    )
    unfused_model = session.load(unfused_graph)

    graph_inputs = paged_kv_cache_inputs(
        kv_params, [seq_len], total_num_pages=4
    )

    rng = np.random.default_rng(0)
    q_np = rng.standard_normal(
        (seq_len, n_q_heads, kv_params.head_dim), dtype=np.float32
    )
    q_gamma_np = rng.standard_normal(kv_params.head_dim, dtype=np.float32)
    k_gamma_np = rng.standard_normal(kv_params.head_dim, dtype=np.float32)
    kv_blocks_np = rng.standard_normal(
        graph_inputs.kv_blocks.shape, dtype=np.float32
    )

    fused_inputs = KVCacheInputsPerDevice(
        kv_blocks=cast_tensor_to(
            Buffer.from_numpy(kv_blocks_np.copy()), dtype
        ).to(device),
        cache_lengths=graph_inputs.cache_lengths,
        lookup_table=graph_inputs.lookup_table,
        max_prompt_length=graph_inputs.max_prompt_length,
        max_cache_length=graph_inputs.max_cache_length,
        kv_scales=graph_inputs.kv_scales,
        attention_dispatch_metadata=graph_inputs.attention_dispatch_metadata,
    )
    unfused_inputs = KVCacheInputsPerDevice(
        kv_blocks=cast_tensor_to(
            Buffer.from_numpy(kv_blocks_np.copy()), dtype
        ).to(device),
        cache_lengths=graph_inputs.cache_lengths,
        lookup_table=graph_inputs.lookup_table,
        max_prompt_length=graph_inputs.max_prompt_length,
        max_cache_length=graph_inputs.max_cache_length,
        kv_scales=graph_inputs.kv_scales,
        attention_dispatch_metadata=graph_inputs.attention_dispatch_metadata,
    )

    input_row_offsets = Buffer.from_numpy(np.array([0, seq_len], np.uint32)).to(
        device
    )
    q = cast_tensor_to(Buffer.from_numpy(q_np), dtype).to(device)
    q_gamma = cast_tensor_to(Buffer.from_numpy(q_gamma_np), dtype).to(device)
    k_gamma = cast_tensor_to(Buffer.from_numpy(k_gamma_np), dtype).to(device)

    q_fused, q_ref = fused_model(
        q,
        q_gamma,
        k_gamma,
        input_row_offsets,
        *fused_inputs.flatten(),
    )
    unfused_model(k_gamma, input_row_offsets, *unfused_inputs.flatten())

    np.testing.assert_allclose(
        q_fused.to_numpy(), q_ref.to_numpy(), rtol=1e-2, atol=1e-2
    )
    np.testing.assert_allclose(
        cast_tensor_to(fused_inputs.kv_blocks, DType.float32).to_numpy(),
        cast_tensor_to(unfused_inputs.kv_blocks, DType.float32).to_numpy(),
        rtol=1e-2,
        atol=1e-2,
    )
