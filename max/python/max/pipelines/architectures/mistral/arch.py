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

from . import weight_adapters
from .batch_processor import MistralBatchProcessor
from .model import MistralModel
from .model_config import MistralConfig

mistral_arch = SupportedArchitecture(
    name="MistralForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["mistralai/Mistral-Nemo-Instruct-2407"],
    default_encoding=MistralConfig.DEFAULT_ENCODING,
    supported_encodings=MistralConfig.SUPPORTED_ENCODINGS,
    multi_gpu_supported=True,
    pipeline_model=MistralModel,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=MistralConfig,
    batching=MistralBatchProcessor,
    memory_planner=PagedMemoryPlanner,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
