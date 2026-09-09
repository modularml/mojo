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
"""Regression test for MLA graph-capture cache-length bucketing coverage.

Graph capture probes a set of cache lengths and captures one graph per distinct
resolved ``num_partitions``. At replay a runtime cache length is bucketed *up*
to the nearest probed length and the graph captured at that length is replayed.
This relies on ``num_partitions`` being monotonic non-decreasing in cache
length, so the bucketed (longer) probe always yields ``num_partitions >=`` the
runtime value -- a captured graph with enough partitions. These tests verify
that invariant against the real ``mo.mla.compute_dispatch_args.scalar`` op.
"""

from collections.abc import Sequence

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MLAKVCacheParams
from max.nn.kv_cache.utils import MLAAttnKey

NUM_HEADS = 128
BATCH_SIZES = [1, 2, 4, 8, 16, 31, 32, 33, 63, 64, 65, 96, 128]
MAX_CACHE_LENGTH = 16384


def _resolve_np(
    params: MLAKVCacheParams,
    batch_size: int,
    cache_length: int,
) -> int:
    """Returns num_partitions from the real Mojo dispatch kernel."""
    key = params.resolve_attn_key(batch_size, 1, cache_length)
    assert isinstance(key, MLAAttnKey)
    return key.num_partitions


def _bucket_cache_length(
    cache_length: int, probe_lengths_sorted: Sequence[int]
) -> int:
    """Rounds a cache length up to the smallest probed length >= it."""
    candidates = [p for p in probe_lengths_sorted if p >= cache_length]
    return min(candidates) if candidates else max(probe_lengths_sorted)


def _make_mla_params(dtype: DType) -> MLAKVCacheParams:
    """Builds single-device MLA params backed by the real custom op."""
    return MLAKVCacheParams(
        dtype=dtype,
        head_dim=576,
        num_layers=1,
        devices=[DeviceRef.GPU()],
        num_q_heads=NUM_HEADS,
    )


@pytest.fixture(scope="module")
def mla_params() -> MLAKVCacheParams:
    """MLA params with a BF16 KV cache."""
    return _make_mla_params(DType.bfloat16)


@pytest.fixture(scope="module")
def mla_params_fp8() -> MLAKVCacheParams:
    """MLA params with an FP8 KV cache (``is_fp8_kv_dtype`` True)."""
    return _make_mla_params(DType.float8_e4m3fn)


def _assert_bucketing_covers(params: MLAKVCacheParams) -> None:
    """For every cache length, the bucketed probe covers its num_partitions."""
    probe_lengths = sorted(
        set(params.graph_capture_probe_cache_lengths(MAX_CACHE_LENGTH))
    )
    for batch_size in BATCH_SIZES:
        probe_np = {
            length: _resolve_np(params, batch_size, length)
            for length in probe_lengths
        }
        for cache_length in range(1, MAX_CACHE_LENGTH + 1):
            runtime_np = _resolve_np(params, batch_size, cache_length)
            bucketed = _bucket_cache_length(cache_length, probe_lengths)
            assert probe_np[bucketed] >= runtime_np, (
                f"batch_size={batch_size}, cache_length={cache_length}: "
                f"bucketed probe {bucketed} has num_partitions "
                f"{probe_np[bucketed]} < runtime num_partitions {runtime_np}."
            )


def test_mla_bucketing_coverage(
    mla_params: MLAKVCacheParams,
) -> None:
    """Bucketing a cache length up always reaches a graph with enough np."""
    _assert_bucketing_covers(mla_params)


def test_mla_bucketing_coverage_fp8(
    mla_params_fp8: MLAKVCacheParams,
) -> None:
    """Coverage holds when the KV cache is FP8 (different np thresholds)."""
    _assert_bucketing_covers(mla_params_fp8)


def test_fp8_produces_fewer_partitions(
    mla_params: MLAKVCacheParams,
    mla_params_fp8: MLAKVCacheParams,
) -> None:
    """FP8 never produces more partitions than BF16 (``np_fp8 <= np_bf16``).

    FP8 uses higher ``min_pages_per_split`` thresholds and a doubled
    ``_np1_cache_threshold``, both of which can only reduce (never increase)
    the partition count relative to BF16.  This test sweeps a broad range of
    ``(batch_size, cache_length)`` combinations to verify the invariant.
    """
    for batch_size in [1, 2, 4, 8, 16, 32, 64, 96, 128]:
        for cl in [
            1,
            64,
            128,
            256,
            300,
            384,
            450,
            511,
            512,
            1024,
            2048,
            2176,
            2177,
            3000,
            4096,
            8192,
            16384,
        ]:
            np_bf16 = _resolve_np(mla_params, batch_size, cl)
            np_fp8 = _resolve_np(mla_params_fp8, batch_size, cl)
            assert np_fp8 <= np_bf16, (
                f"FP8 produced more partitions than BF16: "
                f"bs={batch_size}, cl={cl}, np_fp8={np_fp8}, "
                f"np_bf16={np_bf16}"
            )
