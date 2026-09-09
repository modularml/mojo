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

from ..llama3 import weight_adapters
from ..llama3.batch_processor import Llama3BatchProcessor
from ..llama3.model_config import Llama3Config
from .model import Phi3Model

phi3_arch = SupportedArchitecture(
    name="Phi3ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["microsoft/phi-4", "microsoft/Phi-3.5-mini-instruct"],
    default_weights_format=WeightsFormat.gguf,
    default_encoding="bfloat16",
    supported_encodings={
        "float32",
        "bfloat16",
    },
    pipeline_model=Phi3Model,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
        WeightsFormat.gguf: weight_adapters.convert_gguf_state_dict,
    },
    config=Llama3Config,
    batching=Llama3BatchProcessor,
    memory_planner=PagedMemoryPlanner,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
