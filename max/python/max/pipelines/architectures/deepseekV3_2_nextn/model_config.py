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
"""Config for DeepseekV3.2 NextN (Next-N token prediction) models."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache.cache_params import (
    KVCacheParamInterface,
    KVCacheParams,
    KVCacheQuantizationConfig,
    MultiKVCacheParams,
)
from max.pipelines.architectures.deepseekV3_2.model_config import (
    DeepseekV3_2Config,
)
from max.pipelines.lib import KVCacheConfig, PipelineConfig
from transformers import AutoConfig


@dataclass(kw_only=True)
class DeepseekV3_2NextNConfig(DeepseekV3_2Config):
    """Configuration for DeepseekV3.2 NextN model.

    The NextN (Next-N token prediction) model is a single-layer decoder that
    takes both input embeddings and hidden states from a base model as input,
    concatenates them, and processes through a single DeepSeek-V3.2 sparse
    decoder layer to predict the next token. Unlike the DeepSeek-V3 NextN draft
    (plain MLA), this draft layer runs the lightning indexer and therefore
    needs both the MLA and indexer KV caches, sized to a single layer.
    """

    def __post_init__(self) -> None:
        # Call parent validation first.
        super().__post_init__()

        # NextN supports DP attention (dp == num_devices) or TP attention
        # (dp == 1), matching the base DeepseekV3.2 config validation.
        num_devices = len(self.devices)
        if self.data_parallel_degree not in (1, num_devices):
            raise ValueError(
                "DeepseekV3.2 NextN requires data_parallel_degree "
                f"({self.data_parallel_degree}) to be 1 (TP attention) or equal "
                f"to the number of devices ({num_devices})."
            )

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        """Get the {mla, indexer} KV cache params for the NextN draft model.

        The NextN model has only a single decoder layer, so both caches store
        a single layer's worth of KV pairs.
        """
        data_parallel_degree = pipeline_config.model.data_parallel_degree

        kvcache_quant_config = None
        if cache_dtype in (
            DType.float8_e4m3fn,
            DType.float8_e4m3fnuz,
        ):
            kvcache_quant_config = KVCacheQuantizationConfig(
                scale_dtype=DType.int8, quantization_granularity=32
            )

        mla_kv_params = kv_cache_config.to_params(
            dtype=cache_dtype,
            n_kv_heads=1,
            head_dim=huggingface_config.kv_lora_rank
            + huggingface_config.qk_rope_head_dim,
            num_layers=1,  # MTP only has a single decoder layer.
            devices=devices,
            data_parallel_degree=data_parallel_degree,
            is_mla=True,
            num_q_heads=huggingface_config.num_attention_heads,
            kvcache_quant_config=kvcache_quant_config,
        )
        assert isinstance(mla_kv_params, KVCacheParams)

        # Mirror DeepseekV3_2Config: the indexer K cache is always float8_e4m3fn
        # with per-token float32 scales.
        indexer_kvcache_quant_config = KVCacheQuantizationConfig(
            scale_dtype=DType.float32, quantization_granularity=32
        )
        indexer_kv_params = kv_cache_config.to_params(
            dtype=DType.float8_e4m3fn,
            n_kv_heads=1,
            head_dim=huggingface_config.index_head_dim,
            num_layers=1,
            devices=devices,
            data_parallel_degree=data_parallel_degree,
            is_mla=True,
            num_q_heads=huggingface_config.num_attention_heads,
            kvcache_quant_config=indexer_kvcache_quant_config,
        )
        assert isinstance(indexer_kv_params, KVCacheParams)

        return MultiKVCacheParams.from_params(
            {"mla": mla_kv_params, "indexer": indexer_kv_params}
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        """NextN only has a single decoder layer."""
        return 1
