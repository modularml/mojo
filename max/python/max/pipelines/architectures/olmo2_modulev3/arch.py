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
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import (
    SupportedArchitecture,
    TextTokenizer,
)
from max.pipelines.modeling.types import PipelineTask

from ..llama3_modulev3.batch_processor import Llama3ModuleV3BatchProcessor
from . import weight_adapters
from .model import Olmo2Model
from .model_config import Olmo2Config

olmo2_modulev3_arch = SupportedArchitecture(
    name="Olmo2ForCausalLM_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "allenai/OLMo-2-0425-1B-Instruct",
        "allenai/OLMo-2-1124-7B",
        "allenai/OLMo-2-1124-13B-Instruct",
        "allenai/OLMo-2-0325-32B-Instruct",
        "allenai/OLMo-2-1124-7B-GGUF",
    ],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=Olmo2Config.DEFAULT_ENCODING,
    supported_encodings=Olmo2Config.SUPPORTED_ENCODINGS,
    pipeline_model=Olmo2Model,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    multi_gpu_supported=False,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
        WeightsFormat.gguf: weight_adapters.convert_gguf_state_dict,
    },
    config=Olmo2Config,
    batching=Llama3ModuleV3BatchProcessor,
    memory_planner=PagedMemoryPlanner,
)
