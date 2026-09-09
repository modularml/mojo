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

"""Configuration classes for MAX pipelines."""

from max.nn.kv_cache.cache_params import KVConnectorType
from max.pipelines.diffusion.config import (
    DenoisingCacheConfig,
    DenoisingCacheSettings,
)
from max.pipelines.kv_cache.config import (
    KVCacheConfig,
    KVConnectorConfig,
)
from max.pipelines.lib.pipeline_runtime_config import (
    DEFAULT_MAX_BATCH_INPUT_TOKENS,
)
from max.pipelines.lora import LoRAConfig
from max.pipelines.modeling.config_enums import (
    PipelineRole,
    RepoType,
    RopeType,
    SupportedEncoding,
    is_float4_encoding,
    parse_supported_encoding_from_file_name,
    supported_encoding_dtype,
    supported_encoding_quantization,
    supported_encoding_supported_devices,
    supported_encoding_supported_on,
)
from max.pipelines.speculative.config import (
    SpeculativeConfig,
    SpeculativeMethod,
)

from ..pipeline_args import PipelineArgs
from .config import (
    PipelineConfig,
    PrometheusMetricsMode,
)
from .model_config import (
    MAXModelConfig,
    MAXModelConfigBase,
    _format_config_entries,
)
from .profiling_config import ProfilingConfig

__all__ = [
    "DEFAULT_MAX_BATCH_INPUT_TOKENS",
    "DenoisingCacheConfig",
    "DenoisingCacheSettings",
    "KVCacheConfig",
    "KVConnectorConfig",
    "KVConnectorType",
    "LoRAConfig",
    "MAXModelConfig",
    "MAXModelConfigBase",
    "PipelineArgs",
    "PipelineConfig",
    "PipelineRole",
    "ProfilingConfig",
    "PrometheusMetricsMode",
    "RepoType",
    "RopeType",
    "SpeculativeConfig",
    "SpeculativeMethod",
    "SupportedEncoding",
    "_format_config_entries",
    "is_float4_encoding",
    "parse_supported_encoding_from_file_name",
    "supported_encoding_dtype",
    "supported_encoding_quantization",
    "supported_encoding_supported_devices",
    "supported_encoding_supported_on",
]
