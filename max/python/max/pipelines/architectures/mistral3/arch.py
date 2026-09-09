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
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .model import Mistral3Model
from .model_config import Mistral3Config
from .tokenizer import Mistral3Tokenizer

mistral3_arch = SupportedArchitecture(
    name="Mistral3ForConditionalGeneration",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["mistralai/Mistral-Small-3.1-24B-Instruct-2503"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=Mistral3Config.DEFAULT_ENCODING,
    supported_encodings=Mistral3Config.SUPPORTED_ENCODINGS,
    multi_gpu_supported=True,
    pipeline_model=Mistral3Model,
    tokenizer=Mistral3Tokenizer,
    context_type=TextContext,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=Mistral3Config,
    memory_planner=PagedMemoryPlanner,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
