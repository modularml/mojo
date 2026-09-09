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

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParams,
    KVCacheQuantizationConfig,
    KVConnectorType,
    MHAKVCacheParams,
    MLAKVCacheParams,
    compute_num_device_blocks,
    estimated_memory_size,
)
from max.pipelines.kv_cache.config import KVConnectorConfig

INF = 999999999
GIB = 1024 * 1024 * 1024


def create_params(
    dp: int = 1,
    tp: int = 1,
    page_size: int = 128,
    dtype: DType = DType.float32,
    quantization_config: KVCacheQuantizationConfig = KVCacheQuantizationConfig(),  # noqa: B008
) -> KVCacheParams:
    return MHAKVCacheParams(
        dtype=dtype,
        n_kv_heads=8,
        head_dim=128,
        num_layers=1,
        page_size=page_size,
        data_parallel_degree=dp,
        devices=[DeviceRef.GPU(i) for i in range(tp * dp)],
        kvcache_quant_config=quantization_config,
    )


def test_basic() -> None:
    params = create_params()
    assert (
        compute_num_device_blocks(
            params=params,
            available_cache_memory=GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == 1024
    )
    assert (
        estimated_memory_size(
            params=params,
            available_cache_memory=GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == GIB
    )


def test_unaligned() -> None:
    params = create_params()
    assert (
        compute_num_device_blocks(
            params=params,
            available_cache_memory=GIB + 7,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == 1024
    )
    assert (
        estimated_memory_size(
            params=params,
            available_cache_memory=GIB + 7,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == GIB
    )


def test_big_mem() -> None:
    params = create_params()
    assert (
        compute_num_device_blocks(
            params=params,
            available_cache_memory=17 * GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == 17 * 1024
    )
    assert (
        estimated_memory_size(
            params=params,
            available_cache_memory=17 * GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == 17 * GIB
    )


def test_small_batch_and_seq_len() -> None:
    params = create_params()
    assert (
        compute_num_device_blocks(
            params=params,
            available_cache_memory=GIB,
            max_batch_size=4,
            max_seq_len=1000,
        )
        == 32
    )


def test_tp2() -> None:
    params = create_params(tp=2)
    assert (
        compute_num_device_blocks(
            params=params,
            available_cache_memory=GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == 1024
    )
    assert (
        estimated_memory_size(
            params=params,
            available_cache_memory=GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == GIB
    )


def test_limited_mem() -> None:
    params = create_params()
    with pytest.raises(
        RuntimeError,
        match="Insufficient cache memory to allocate even a single page",
    ):
        compute_num_device_blocks(
            params=params,
            available_cache_memory=1,
            max_batch_size=INF,
            max_seq_len=INF,
        )


def test_max_seq_len_exceeds_capacity() -> None:
    params = create_params()
    # 1 GiB fits 1024 pages of 128 tokens each; one extra token needs a 1025th.
    oversized_seq_len = 1024 * 128 + 1

    # By default the oversized config only warns (memory estimation probes
    # such configs during binary search).
    assert (
        compute_num_device_blocks(
            params=params,
            available_cache_memory=GIB,
            max_batch_size=1,
            max_seq_len=oversized_seq_len,
        )
        == 1024
    )

    with pytest.raises(
        RuntimeError,
        match="one request at the max sequence length",
    ):
        compute_num_device_blocks(
            params=params,
            available_cache_memory=GIB,
            max_batch_size=1,
            max_seq_len=oversized_seq_len,
            require_max_seq_len_fits=True,
        )

    # A config that exactly fits does not raise.
    assert (
        compute_num_device_blocks(
            params=params,
            available_cache_memory=GIB,
            max_batch_size=1,
            max_seq_len=1024 * 128,
            require_max_seq_len_fits=True,
        )
        == 1024
    )


def test_dp2() -> None:
    params = create_params(dp=2)
    assert (
        compute_num_device_blocks(
            params=params,
            available_cache_memory=GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == 512
    )
    assert (
        estimated_memory_size(
            params=params,
            available_cache_memory=GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == GIB
    )


def test_weird_page_size() -> None:
    params = create_params(page_size=777)
    assert (
        compute_num_device_blocks(
            params=params,
            available_cache_memory=GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == 168
    )
    assert (
        estimated_memory_size(
            params=params,
            available_cache_memory=GIB,
            max_batch_size=INF,
            max_seq_len=INF,
        )
        == 1069350912
    )


def test_bytes_per_block() -> None:
    dtype = DType.float32
    n_kv_heads = 1
    head_dim = 24
    num_layers = 17
    page_size = 128
    data_parallel_degree = 1

    params = MHAKVCacheParams(
        dtype=dtype,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        page_size=page_size,
        data_parallel_degree=data_parallel_degree,
        devices=[DeviceRef.GPU()],
    )

    assert params.bytes_per_block == 417792


def test_quantized_kv_cache() -> None:
    fp8_params = create_params(dp=2, dtype=DType.float8_e4m3fn)
    fp32_params = create_params(dp=2, dtype=DType.float32)

    fp8_estimated_memory = estimated_memory_size(
        params=fp8_params,
        available_cache_memory=GIB,
        max_batch_size=INF,
        max_seq_len=INF,
    )
    fp32_estimated_memory = estimated_memory_size(
        params=fp32_params,
        available_cache_memory=GIB,
        max_batch_size=INF,
        max_seq_len=INF,
    )
    assert fp32_estimated_memory == GIB
    # The estimated memory is the same in both cases since more blocks can be allocated when the KVCache is compressed.
    assert fp8_estimated_memory < fp32_estimated_memory

    # The number of bytes / block should be ~4x smaller for FP8 KVCache (compared to FP32).
    assert fp8_params.bytes_per_block < fp32_params.bytes_per_block

    fp8_num_device_blocks = compute_num_device_blocks(
        params=fp8_params,
        available_cache_memory=GIB,
        max_batch_size=INF,
        max_seq_len=INF,
    )
    fp32_num_device_blocks = compute_num_device_blocks(
        params=fp32_params,
        available_cache_memory=GIB,
        max_batch_size=INF,
        max_seq_len=INF,
    )
    # Nearly ~4x more blocks can be allocated when using FP8 KVCache (compared to FP32).
    assert fp8_num_device_blocks > fp32_num_device_blocks


def _create_mla_params(tp: int, is_mla: bool = True) -> KVCacheParams:
    """Create KVCacheParams for MLA with a tiered connector and host swap space."""
    shared_kwargs = dict(
        dtype=DType.float32,
        head_dim=128,
        num_layers=1,
        page_size=128,
        data_parallel_degree=1,
        devices=[DeviceRef.GPU(i) for i in range(tp)],
        enable_prefix_caching=True,
        kv_connector_config=KVConnectorConfig(
            type=KVConnectorType.tiered,
            host_offload_max_gb=1,
        ),
    )
    if is_mla:
        return MLAKVCacheParams(num_q_heads=32, **shared_kwargs)  # type: ignore[arg-type]
    return MHAKVCacheParams(n_kv_heads=8, **shared_kwargs)  # type: ignore[arg-type]


def test_host_blocks_mla_tp_scaling() -> None:
    """With TP MLA, host block count should be independent of TP degree.

    ``replicates_kv_across_tp`` is what the host row width keys off (via
    ``KVCacheMemory.replicated``), so it must be set exactly when the per-device
    ``bytes_per_block`` grew by the TP degree. See
    ``test_rust_tier_host_row.py`` for the row width itself.
    """
    params_tp1 = _create_mla_params(tp=1)
    params_tp8 = _create_mla_params(tp=8)

    # TP=1 MLA doesn't replicate (only 1 device), so no adjustment.
    assert not params_tp1.replicates_kv_across_tp
    # TP=8 MLA replicates KV on every device.
    assert params_tp8.replicates_kv_across_tp

    # bytes_per_block grows with TP (each device holds full KV); the host stores
    # one copy, which is what keeps the host block count TP-independent.
    assert params_tp8.bytes_per_block == params_tp1.bytes_per_block * 8


def test_host_blocks_non_mla_tp_no_scaling() -> None:
    """Without MLA, host blocks should NOT scale by TP degree."""
    params_tp1 = _create_mla_params(tp=1, is_mla=False)
    params_tp8 = _create_mla_params(tp=8, is_mla=False)

    assert not params_tp1.replicates_kv_across_tp
    assert not params_tp8.replicates_kv_across_tp

    # Non-MLA shards KV across TP, so bytes_per_block is already constant
    # regardless of TP and the host stores every shard.
    assert params_tp1.bytes_per_block == params_tp8.bytes_per_block
