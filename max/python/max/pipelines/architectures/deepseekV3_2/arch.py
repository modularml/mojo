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
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from ..deepseekV3.batch_processor import DeepseekV3BatchProcessor
from . import weight_adapters
from .memory_planner import DeepseekV3_2MemoryPlanner
from .model import DeepseekV3_2Model
from .model_config import DeepseekV3_2Config

deepseekV3_2_arch = SupportedArchitecture(
    name="DeepseekV32ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "deepseek-ai/DeepSeek-V3.2",
        "deepseek-ai/DeepSeek-V3.2-Exp",
    ],
    default_encoding=DeepseekV3_2Config.DEFAULT_ENCODING,
    supported_encodings=DeepseekV3_2Config.SUPPORTED_ENCODINGS,
    multi_gpu_supported=True,
    pipeline_model=DeepseekV3_2Model,
    batching=DeepseekV3BatchProcessor,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    config=DeepseekV3_2Config,
    memory_planner=DeepseekV3_2MemoryPlanner,
)
