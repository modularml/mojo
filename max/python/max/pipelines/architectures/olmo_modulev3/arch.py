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
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from ..llama3_modulev3 import weight_adapters
from .model import OlmoModel
from .model_config import OlmoConfig

olmo_modulev3_arch = SupportedArchitecture(
    name="OlmoForCausalLM_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["allenai/OLMo-1B-hf", "allenai/OLMo-1B-0724-hf"],
    default_weights_format=WeightsFormat.gguf,
    default_encoding=OlmoConfig.DEFAULT_ENCODING,
    supported_encodings=OlmoConfig.SUPPORTED_ENCODINGS,
    pipeline_model=OlmoModel,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
        WeightsFormat.gguf: weight_adapters.convert_gguf_state_dict,
    },
    config=OlmoConfig,
    memory_planner=PagedMemoryPlanner,
)
