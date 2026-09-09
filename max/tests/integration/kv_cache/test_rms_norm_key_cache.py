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
import pytest
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Dim, Graph, TensorType, TensorValue, ops
from max.nn.kernels import rms_norm_key_cache
from max.nn.kv_cache import (
    KVCacheInputsPerDevice,
    KVCacheParams,
    MHAKVCacheParams,
    PagedCacheValues,
)
from test_common.simple_kv_cache import paged_kv_cache_inputs


@dataclass(frozen=True)
class RMSNormKeyCacheModel:
    """Model containing a single matmul KV ragged op."""

    kv_params: KVCacheParams
    """Hyperparameters describing this instance of the KV cache."""

    layer_idx: int
    """Layer index of the KV cache collection."""

    total_seq_len: int
    """Total sequence length: sum(input_row_offsets)."""

    rms_norm_cols: int | None = None
    """Number of columns in the RMSNorm operation."""

    per_head_norm: bool = True
    """Whether to normalize separately for each head."""

    def __call__(
        self,
        gamma: TensorValue,
        input_row_offsets: TensorValue,
        *graph_inputs: TensorValue,
    ) -> None:
        """Stages a graph consisting of a matmul KV cache ragged custom op.

        This contains both the matmul KV cache ragged custom op and a "fetch"
        op to get a KVCacheCollection.
        """
        rms_norm_key_cache(
            self.kv_params,
            PagedCacheValues(
                kv_blocks=graph_inputs[0].buffer,
                cache_lengths=graph_inputs[1].tensor,
                lookup_table=graph_inputs[2].tensor,
                max_prompt_length=graph_inputs[3].tensor,
                max_cache_length=graph_inputs[4].tensor,
            ),
            gamma=gamma,
            epsilon=1e-5,
            layer_idx=ops.constant(
                self.layer_idx, DType.uint32, device=DeviceRef.CPU()
            ),
            total_seq_len=Dim(self.total_seq_len),
            input_row_offsets=input_row_offsets,
            rms_norm_cols=self.rms_norm_cols,
            weight_offset=0.0,
            per_head_norm=self.per_head_norm,
        )


@pytest.mark.parametrize(
    "dtype",
    [DType.float32],
)
def test_rms_norm_key_cache(session: InferenceSession, dtype: DType) -> None:
    seq_lens = [10, 4]
    kv_params = MHAKVCacheParams(
        dtype=dtype,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        page_size=128,
        devices=[DeviceRef.CPU()],
    )
    # Stage the fetch op + custom matmul KV cache ragged op graph.
    gamma_type = TensorType(
        dtype, shape=[kv_params.head_dim], device=DeviceRef.CPU()
    )
    input_row_offsets_type = TensorType(
        DType.uint32, ["batch_size_plus_1"], device=DeviceRef.CPU()
    )
    graph = Graph(
        "matmul_kv_cache_ragged",
        forward=RMSNormKeyCacheModel(
            kv_params, layer_idx=0, total_seq_len=sum(seq_lens)
        ),
        input_types=[
            gamma_type,
            input_row_offsets_type,
            *kv_params.flattened_kv_inputs(),
        ],
    )

    # Compile and init the model.
    model = session.load(graph)

    graph_inputs = paged_kv_cache_inputs(kv_params, seq_lens, total_num_pages=8)

    # First set KV blocks to all ones so that RMSNorm changes them.
    kv_blocks = graph_inputs.kv_blocks
    all_ones = np.ones(kv_blocks.shape, dtype=kv_blocks.dtype.to_numpy())

    # Create new KVCacheInputs with updated first element
    graph_inputs = KVCacheInputsPerDevice(
        kv_blocks=Buffer.from_numpy(all_ones.copy()),
        cache_lengths=graph_inputs.cache_lengths,
        lookup_table=graph_inputs.lookup_table,
        max_prompt_length=graph_inputs.max_prompt_length,
        max_cache_length=graph_inputs.max_cache_length,
        kv_scales=graph_inputs.kv_scales,
        attention_dispatch_metadata=graph_inputs.attention_dispatch_metadata,
    )

    gamma = np.random.randn(kv_params.head_dim).astype(dtype.to_numpy())
    input_row_offsets = np.array([0, *np.cumsum(seq_lens)], dtype=np.uint32)
    model(gamma, input_row_offsets, *graph_inputs.flatten())

    # Check that the RMSNorm wrote output to the KV cache.
    assert (graph_inputs.kv_blocks.to_numpy() != all_ones).any()


@pytest.mark.parametrize(
    "dtype",
    [DType.float32],
)
def test_partial_rms_norm_key_cache(
    session: InferenceSession, dtype: DType
) -> None:
    seq_lens = [
        10,
    ]
    gamma_size = 512
    kv_params = MHAKVCacheParams(
        dtype=dtype,
        n_kv_heads=1,
        head_dim=576,
        num_layers=1,
        page_size=128,
        devices=[DeviceRef.CPU()],
    )
    # Stage the fetch op + custom matmul KV cache ragged op graph.
    gamma_type = TensorType(dtype, shape=[gamma_size], device=DeviceRef.CPU())
    input_row_offsets_type = TensorType(
        DType.uint32, ["batch_size_plus_1"], device=DeviceRef.CPU()
    )
    graph = Graph(
        "matmul_kv_cache_ragged",
        forward=RMSNormKeyCacheModel(
            kv_params,
            layer_idx=0,
            total_seq_len=sum(seq_lens),
            rms_norm_cols=gamma_size,
        ),
        input_types=[
            gamma_type,
            input_row_offsets_type,
            *kv_params.flattened_kv_inputs(),
        ],
    )

    # Compile and init the model.
    model = session.load(graph)

    graph_inputs = paged_kv_cache_inputs(kv_params, seq_lens, total_num_pages=8)

    # First set KV blocks to all ones so that RMSNorm changes them.
    kv_blocks = graph_inputs.kv_blocks
    all_ones = np.ones(kv_blocks.shape, dtype=kv_blocks.dtype.to_numpy())

    # Create new KVCacheInputs with updated first element
    graph_inputs = KVCacheInputsPerDevice(
        kv_blocks=Buffer.from_numpy(all_ones.copy()),
        cache_lengths=graph_inputs.cache_lengths,
        lookup_table=graph_inputs.lookup_table,
        max_prompt_length=graph_inputs.max_prompt_length,
        max_cache_length=graph_inputs.max_cache_length,
        kv_scales=graph_inputs.kv_scales,
        attention_dispatch_metadata=graph_inputs.attention_dispatch_metadata,
    )

    gamma = np.random.randn(gamma_size).astype(dtype.to_numpy())
    input_row_offsets = np.array([0, *np.cumsum(seq_lens)], dtype=np.uint32)
    model(gamma, input_row_offsets, *graph_inputs.flatten())

    # shape: [batch_size,kv_dim,num_layers,max_seq_len,n_kv_heads,head_dim]
    kv_block = graph_inputs.kv_blocks.to_numpy()

    # Check that the first 512 elements of each head is normalized
    for seq_idx in range(seq_lens[0]):
        assert np.isclose(
            kv_block[0, 0, 0, seq_idx, 0, :gamma_size], gamma, rtol=1e-05
        ).all()

    # Check that the last 64 elements of each head is unchanged
    for seq_idx in range(seq_lens[0]):
        assert np.isclose(
            kv_block[0, 0, 0, seq_idx, 0, gamma_size:],
            np.ones((kv_params.head_dim - gamma_size), dtype=np.float32),
        ).all()


@pytest.mark.parametrize(
    "dtype",
    [DType.float32],
)
def test_rms_norm_new_key_cache(
    session: InferenceSession, dtype: DType
) -> None:
    seq_lens = [
        10,
    ]
    gamma_size = 128
    kv_params = MHAKVCacheParams(
        dtype=dtype,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        page_size=128,
        devices=[DeviceRef.CPU()],
    )
    # Stage the fetch op + custom matmul KV cache ragged op graph.
    gamma_type = TensorType(dtype, shape=[gamma_size], device=DeviceRef.CPU())
    input_row_offsets_type = TensorType(
        DType.uint32, ["batch_size_plus_1"], device=DeviceRef.CPU()
    )
    graph = Graph(
        "matmul_kv_cache_ragged",
        forward=RMSNormKeyCacheModel(
            kv_params,
            layer_idx=0,
            total_seq_len=sum(seq_lens),
            rms_norm_cols=gamma_size,
        ),
        input_types=[
            gamma_type,
            input_row_offsets_type,
            *kv_params.flattened_kv_inputs(),
        ],
    )

    # Compile and init the model.
    model = session.load(graph)

    # Unlike the previous tests, each request already holds ``seq_lens[i]``
    # tokens in the cache, so the kernel must only touch the new tokens.
    graph_inputs = paged_kv_cache_inputs(
        kv_params, seq_lens, cache_lengths=seq_lens, total_num_pages=8
    )

    # First set KV blocks to all ones so that RMSNorm changes them.
    kv_blocks = graph_inputs.kv_blocks
    all_ones = np.ones(kv_blocks.shape, dtype=kv_blocks.dtype.to_numpy())

    # Create new KVCacheInputs with updated first element
    graph_inputs = KVCacheInputsPerDevice(
        kv_blocks=Buffer.from_numpy(all_ones.copy()),
        cache_lengths=graph_inputs.cache_lengths,
        lookup_table=graph_inputs.lookup_table,
        max_prompt_length=graph_inputs.max_prompt_length,
        max_cache_length=graph_inputs.max_cache_length,
        kv_scales=graph_inputs.kv_scales,
        attention_dispatch_metadata=graph_inputs.attention_dispatch_metadata,
    )

    gamma = np.random.randn(gamma_size).astype(dtype.to_numpy())
    input_row_offsets = np.array([0, *np.cumsum(seq_lens)], dtype=np.uint32)
    model(gamma, input_row_offsets, *graph_inputs.flatten())

    # shape: [batch_size,kv_dim,num_layers,max_seq_len,n_kv_heads,head_dim]
    kv_block = graph_inputs.kv_blocks.to_numpy()

    # check that for the first 10 tokens all heads are unchanged
    for seq_idx in range(10):
        for head_idx in range(kv_params.n_kv_heads):
            assert np.isclose(
                kv_block[0, 0, 0, seq_idx, head_idx, :gamma_size],
                np.ones((gamma_size), dtype=np.float32),
            ).all()

    # check that for the next 10 tokens, all heads are normalized
    for seq_idx in range(10, 20):
        for head_idx in range(kv_params.n_kv_heads):
            assert np.isclose(
                kv_block[0, 0, 0, seq_idx, head_idx, :gamma_size],
                gamma,
                rtol=1e-05,
            ).all()


@pytest.mark.parametrize(
    "kv_dtype, gamma_dtype",
    [
        (DType.float32, DType.float16),
        (DType.bfloat16, DType.float32),
    ],
)
def test_rms_norm_key_cache_dtype_mismatch(
    session: InferenceSession, kv_dtype: DType, gamma_dtype: DType
) -> None:
    """Tests that a TypeError is raised when gamma dtype mismatches kv dtype."""
    seq_lens = [10]
    kv_params = MHAKVCacheParams(
        dtype=kv_dtype,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        page_size=128,
        devices=[DeviceRef.CPU()],
    )

    # Stage the fetch op + custom matmul KV cache ragged op graph.
    gamma_type = TensorType(
        gamma_dtype, shape=[kv_params.head_dim], device=DeviceRef.CPU()
    )
    input_row_offsets_type = TensorType(
        DType.uint32, ["batch_size_plus_1"], device=DeviceRef.CPU()
    )
    expected_msg = (
        f"expected gamma dtype {gamma_dtype} to match KV dtype {kv_dtype}"
    )
    with pytest.raises(TypeError, match=expected_msg):
        Graph(
            "matmul_kv_cache_ragged",
            forward=RMSNormKeyCacheModel(
                kv_params, layer_idx=0, total_seq_len=sum(seq_lens)
            ),
            input_types=[
                gamma_type,
                input_row_offsets_type,
                *kv_params.flattened_kv_inputs(),
            ],
        )


def test_rms_norm_key_cache_per_token_norm(session: InferenceSession) -> None:
    """Test RMS normalization applied per token (across all heads) rather than per head."""
    seq_lens = [5, 3]
    n_kv_heads = 4
    head_dim = 64

    kv_params = MHAKVCacheParams(
        dtype=DType.float32,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        num_layers=1,
        page_size=128,
        devices=[DeviceRef.CPU()],
    )
    # For per token normalization, gamma has shape [n_kv_heads * head_dim]
    # This means normalization is applied across all heads for each token
    total_features = n_kv_heads * head_dim
    gamma_type = TensorType(
        DType.float32, shape=[total_features], device=DeviceRef.CPU()
    )
    input_row_offsets_type = TensorType(
        DType.uint32, ["batch_size_plus_1"], device=DeviceRef.CPU()
    )

    # Stage the graph with per_head_norm=False for per token normalization
    graph = Graph(
        "rms_norm_key_cache_per_token",
        forward=RMSNormKeyCacheModel(
            kv_params,
            layer_idx=0,
            total_seq_len=sum(seq_lens),
            per_head_norm=False,  # This enables per token normalization
        ),
        input_types=[
            gamma_type,
            input_row_offsets_type,
            *kv_params.flattened_kv_inputs(),
        ],
    )

    # Compile and init the model
    model = session.load(graph)

    graph_inputs = paged_kv_cache_inputs(kv_params, seq_lens, total_num_pages=8)

    # First set KV blocks to all ones so that RMSNorm changes them.
    kv_blocks = graph_inputs.kv_blocks
    all_ones = np.ones(kv_blocks.shape, dtype=kv_blocks.dtype.to_numpy())

    # Create new KVCacheInputs with updated first element
    graph_inputs = KVCacheInputsPerDevice(
        kv_blocks=Buffer.from_numpy(all_ones.copy()),
        cache_lengths=graph_inputs.cache_lengths,
        lookup_table=graph_inputs.lookup_table,
        max_prompt_length=graph_inputs.max_prompt_length,
        max_cache_length=graph_inputs.max_cache_length,
        kv_scales=graph_inputs.kv_scales,
        attention_dispatch_metadata=graph_inputs.attention_dispatch_metadata,
    )

    # Create gamma weights for per token normalization
    gamma = np.random.randn(total_features).astype(np.float32)
    input_row_offsets = np.array([0, *np.cumsum(seq_lens)], dtype=np.uint32)

    # Run the model
    model(gamma, input_row_offsets, *graph_inputs.flatten())

    # Verify that normalization was applied per token (across all heads)
    kv_block = graph_inputs.kv_blocks.to_numpy()

    # For per token norm, verify that normalization was applied consistently
    # across all heads for each token by checking that the output has expected properties
    total_seq_len = sum(seq_lens)
    for token_idx in range(total_seq_len):
        batch_idx = 0 if token_idx < seq_lens[0] else 1
        local_token_idx = (
            token_idx if token_idx < seq_lens[0] else token_idx - seq_lens[0]
        )

        # since we set all ones, the output should be the same as the gamma
        token_values = kv_block[
            batch_idx, 0, 0, local_token_idx, :, :
        ].flatten()

        assert np.isclose(token_values, gamma, rtol=1e-05).all()
