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

from max.graph.weights import WeightsFormat
from max.pipelines.architectures.llama3_modulev3 import weight_adapters
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    SupportedArchitecture,
    TextTokenizer,
)
from max.pipelines.modeling.types import PipelineTask

from .batch_processor import Qwen3EmbeddingModuleV3BatchProcessor
from .model import Qwen3EmbeddingModel
from .model_config import Qwen3EmbeddingConfig

qwen3_embedding_modulev3_arch = SupportedArchitecture(
    name="Qwen3ForCausalLM_ModuleV3",
    task=PipelineTask.EMBEDDINGS_GENERATION,
    example_repo_ids=[
        "Qwen/Qwen3-Embedding-0.6B",
        "Qwen/Qwen3-Embedding-4B",
        "Qwen/Qwen3-Embedding-8B",
    ],
    default_encoding=Qwen3EmbeddingConfig.DEFAULT_ENCODING,
    supported_encodings=Qwen3EmbeddingConfig.SUPPORTED_ENCODINGS,
    pipeline_model=Qwen3EmbeddingModel,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
        WeightsFormat.gguf: weight_adapters.convert_gguf_state_dict,
    },
    config=Qwen3EmbeddingConfig,
    batching=Qwen3EmbeddingModuleV3BatchProcessor,
    multi_gpu_supported=False,
)
