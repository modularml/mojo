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

"""Types to interface with ML pipelines such as text/token/pixel generation."""

from max.pipelines.weights.hf_utils import download_weight_files

from .architectures import register_all_models
from .diffusion.pipeline import PixelGenerationPipeline
from .lib.config import (
    KVCacheConfig,
    LoRAConfig,
    MAXModelConfig,
    PipelineArgs,
    PipelineConfig,
    PipelineRole,
    ProfilingConfig,
    PrometheusMetricsMode,
    RepoType,
    RopeType,
    SpeculativeConfig,
    SupportedEncoding,
    is_float4_encoding,
    parse_supported_encoding_from_file_name,
    supported_encoding_dtype,
    supported_encoding_quantization,
    supported_encoding_supported_devices,
    supported_encoding_supported_on,
)
from .lib.embeddings_pipeline import EmbeddingsPipeline, EmbeddingsPipelineType
from .lib.interfaces import (
    GenerateMixin,
    ModelInputs,
    ModelOutputs,
    PipelineModel,
)
from .lib.memory_estimation import MemoryEstimator
from .lib.pipeline_runtime_config import PipelineRuntimeConfig
from .lib.pipeline_variants.text_generation import (
    TextGenerationPipeline,
    TextGenerationPipelineInterface,
)
from .lib.registry import PIPELINE_REGISTRY, SupportedArchitecture
from .lib.tokenizer import (
    IdentityPipelineTokenizer,
    TextAndVisionTokenizer,
    TextTokenizer,
)
from .lib.utils import upper_bounded_default
from .lora import ADAPTER_CONFIG_FILE
from .sampling.sampling_config import SamplingConfig

# Hydrate the registry.
register_all_models()

__all__ = [
    "ADAPTER_CONFIG_FILE",
    "PIPELINE_REGISTRY",
    "EmbeddingsPipeline",
    "EmbeddingsPipelineType",
    "GenerateMixin",
    "IdentityPipelineTokenizer",
    "KVCacheConfig",
    "LoRAConfig",
    "MAXModelConfig",
    "MemoryEstimator",
    "ModelInputs",
    "ModelOutputs",
    "PipelineArgs",
    "PipelineConfig",
    "PipelineModel",
    "PipelineRole",
    "PipelineRuntimeConfig",
    "PixelGenerationPipeline",
    "ProfilingConfig",
    "PrometheusMetricsMode",
    "RepoType",
    "RopeType",
    "SamplingConfig",
    "SpeculativeConfig",
    "SupportedArchitecture",
    "SupportedEncoding",
    "TextAndVisionTokenizer",
    "TextGenerationPipeline",
    "TextGenerationPipelineInterface",
    "TextTokenizer",
    "download_weight_files",
    "is_float4_encoding",
    "parse_supported_encoding_from_file_name",
    "supported_encoding_dtype",
    "supported_encoding_quantization",
    "supported_encoding_supported_devices",
    "supported_encoding_supported_on",
    "upper_bounded_default",
]
