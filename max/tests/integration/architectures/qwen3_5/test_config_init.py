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

"""Config init tests for Qwen3.5.

The MHA kernel selects ``tile_size == head_dim`` and the KV cache page size
must be at least the tile size, so Qwen3.5 (``head_dim=256``) needs a larger
effective page size than the framework default of 128. That bump must land on
the constructed :class:`KVCacheParams` without mutating the shared
:class:`KVCacheConfig` (which other consumers read).
"""

from types import SimpleNamespace
from unittest.mock import Mock

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams
from max.pipelines.architectures.qwen3_5.model_config import Qwen3_5Config
from max.pipelines.lib import KVCacheConfig


def _text_config(head_dim: int) -> SimpleNamespace:
    return SimpleNamespace(
        head_dim=head_dim,
        num_key_value_heads=8,
        num_hidden_layers=8,
        layer_types=[
            "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
            for i in range(8)
        ],
    )


def _pipeline_config() -> Mock:
    pipeline_config = Mock()
    pipeline_config.model.data_parallel_degree = 1
    return pipeline_config


def _construct_kv_params(
    head_dim: int, kv_cache_config: KVCacheConfig
) -> KVCacheParams:
    return Qwen3_5Config.construct_kv_params(
        huggingface_config=_text_config(head_dim),
        pipeline_config=_pipeline_config(),
        devices=[DeviceRef.CPU()],
        kv_cache_config=kv_cache_config,
        cache_dtype=DType.bfloat16,
    )


def test_page_size_bumped_to_head_dim_without_config_mutation() -> None:
    kv_cache_config = KVCacheConfig()
    assert kv_cache_config.kv_cache_page_size == 128

    kv_params = _construct_kv_params(256, kv_cache_config)

    assert kv_params.page_size == 256
    # The shared config must not observe the bump.
    assert kv_cache_config.kv_cache_page_size == 128


def test_page_size_respects_larger_user_value() -> None:
    kv_cache_config = KVCacheConfig(kv_cache_page_size=512)
    kv_params = _construct_kv_params(256, kv_cache_config)
    assert kv_params.page_size == 512


def test_page_size_unchanged_for_small_head_dim() -> None:
    kv_cache_config = KVCacheConfig()
    kv_params = _construct_kv_params(128, kv_cache_config)
    assert kv_params.page_size == 128
