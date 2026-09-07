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

"""Constructor for hybrid SWA + full KV trees used."""

from __future__ import annotations

from unittest.mock import Mock

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.architectures.gpt_oss.hybrid_kv_params_util import (
    hybrid_swa_full_kv_params,
)
from max.pipelines.lib import KVCacheConfig


def _pipeline_config() -> Mock:
    model = Mock()
    model.data_parallel_degree = 1
    pipeline_config = Mock()
    pipeline_config.model = model
    pipeline_config.speculative = None
    return pipeline_config


def test_hybrid_swa_full_kv_params_gpt_oss_pattern() -> None:
    layer_types = ["sliding_attention", "full_attention"] * 12
    params = hybrid_swa_full_kv_params(
        layer_types=layer_types,
        sliding_window=128,
        pipeline_config=_pipeline_config(),
        devices=[DeviceRef.CPU()],
        kv_cache_config=KVCacheConfig(),
        cache_dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=64,
    )
    sliding, full = params.children.values()
    assert list(params.children) == ["sliding_attention", "full_attention"]
    assert isinstance(sliding, MHAKVCacheParams)
    assert isinstance(full, MHAKVCacheParams)
    assert sliding.num_layers == 12
    assert full.num_layers == 12
    assert sliding.window_size == 128
    assert full.window_size is None
    assert sliding.group_id.is_sliding_window()
    assert full.group_id.is_full()


def test_hybrid_swa_full_kv_params_olmo3_pattern() -> None:
    layer_types = [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ] * 8
    params = hybrid_swa_full_kv_params(
        layer_types=layer_types,
        sliding_window=4096,
        pipeline_config=_pipeline_config(),
        devices=[DeviceRef.CPU()],
        kv_cache_config=KVCacheConfig(),
        cache_dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=128,
    )
    sliding, full = params.children.values()
    assert isinstance(sliding, MHAKVCacheParams)
    assert isinstance(full, MHAKVCacheParams)
    assert sliding.num_layers == 24
    assert full.num_layers == 8
    assert sliding.window_size == 4096
    assert full.group_id.is_full()


def test_hybrid_swa_full_kv_params_gemma3_pattern() -> None:
    # Pattern 6: five sliding, then one full.
    layer_types = (["sliding_attention"] * 5 + ["full_attention"]) * 6
    params = hybrid_swa_full_kv_params(
        layer_types=layer_types,
        sliding_window=1024,
        pipeline_config=_pipeline_config(),
        devices=[DeviceRef.CPU()],
        kv_cache_config=KVCacheConfig(),
        cache_dtype=DType.bfloat16,
        n_kv_heads=4,
        head_dim=256,
    )
    sliding, full = params.children.values()
    assert isinstance(sliding, MHAKVCacheParams)
    assert isinstance(full, MHAKVCacheParams)
    assert sliding.num_layers == 30
    assert full.num_layers == 6
    assert sliding.window_size == 1024
    assert full.window_size is None


def test_hybrid_swa_full_kv_params_step3p5_unequal_heads() -> None:
    layer_types = ["sliding_attention", "full_attention"] * 8
    params = hybrid_swa_full_kv_params(
        layer_types=layer_types,
        sliding_window=512,
        pipeline_config=_pipeline_config(),
        devices=[DeviceRef.CPU()],
        kv_cache_config=KVCacheConfig(),
        cache_dtype=DType.bfloat16,
        n_kv_heads=8,
        head_dim=128,
        sliding_n_kv_heads=16,
        full_n_kv_heads=8,
    )
    sliding, full = params.children.values()
    assert isinstance(sliding, MHAKVCacheParams)
    assert isinstance(full, MHAKVCacheParams)
    assert sliding.num_layers == 8
    assert full.num_layers == 8
    assert sliding.n_kv_heads == 16
    assert full.n_kv_heads == 8
    assert sliding.window_size == 512
    assert full.window_size is None
