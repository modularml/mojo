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

"""Constructor for hybrid sliding + full KV trees"""

from __future__ import annotations

from collections.abc import Sequence

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MultiKVCacheParams
from max.pipelines.lib import KVCacheConfig, PipelineConfig


def hybrid_swa_full_kv_params(
    *,
    layer_types: Sequence[str],
    sliding_window: int,
    pipeline_config: PipelineConfig,
    devices: list[DeviceRef],
    kv_cache_config: KVCacheConfig,
    cache_dtype: DType,
    n_kv_heads: int,
    head_dim: int,
    allow_kv_head_replication: bool = False,
    sliding_n_kv_heads: int | None = None,
    sliding_head_dim: int | None = None,
    full_n_kv_heads: int | None = None,
    full_head_dim: int | None = None,
) -> MultiKVCacheParams:
    sliding_layers = 0
    full_layers = 0
    for attention_type in layer_types:
        if attention_type == "sliding_attention":
            sliding_layers += 1
        elif attention_type == "full_attention":
            full_layers += 1
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

    num_spec_tokens = (
        (pipeline_config.speculative.num_speculative_tokens or 0)
        if pipeline_config.speculative
        else 0
    )
    speculative_method = (
        pipeline_config.speculative.speculative_method
        if pipeline_config.speculative
        else None
    )
    sliding_kv_params = kv_cache_config.to_params(
        dtype=cache_dtype,
        n_kv_heads=(
            n_kv_heads if sliding_n_kv_heads is None else sliding_n_kv_heads
        ),
        head_dim=head_dim if sliding_head_dim is None else sliding_head_dim,
        num_layers=sliding_layers,
        devices=devices,
        data_parallel_degree=pipeline_config.model.data_parallel_degree,
        speculative_method=speculative_method,
        num_draft_tokens=num_spec_tokens,
        allow_kv_head_replication=allow_kv_head_replication,
        window_size=sliding_window,
    )
    full_kv_params = kv_cache_config.to_params(
        dtype=cache_dtype,
        n_kv_heads=n_kv_heads if full_n_kv_heads is None else full_n_kv_heads,
        head_dim=head_dim if full_head_dim is None else full_head_dim,
        num_layers=full_layers,
        devices=devices,
        data_parallel_degree=pipeline_config.model.data_parallel_degree,
        speculative_method=speculative_method,
        num_draft_tokens=num_spec_tokens,
        allow_kv_head_replication=allow_kv_head_replication,
        window_size=None,
    )
    return MultiKVCacheParams.from_params(
        {
            "sliding_attention": sliding_kv_params,
            "full_attention": full_kv_params,
        }
    )
