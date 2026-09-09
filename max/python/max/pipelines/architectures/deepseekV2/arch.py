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

from max.experimental.cascade.pipelines.common_textgen import (
    CommonTextGenPipeline,
)
from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import (
    SupportedArchitecture,
    TextTokenizer,
)
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .batch_processor import DeepseekV2BatchProcessor
from .model import DeepseekV2Model
from .model_config import DeepseekV2Config

deepseekV2_arch = SupportedArchitecture(
    name="DeepseekV2ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "deepseek-ai/DeepSeek-V2-Lite-Chat",
    ],
    default_encoding=DeepseekV2Config.DEFAULT_ENCODING,
    supported_encodings=DeepseekV2Config.SUPPORTED_ENCODINGS,
    multi_gpu_supported=True,
    pipeline_model=DeepseekV2Model,
    batching=DeepseekV2BatchProcessor,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    requires_max_batch_context_length=True,
    config=DeepseekV2Config,
    memory_planner=PagedMemoryPlanner,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
    cascade_pipeline_factory=CommonTextGenPipeline,
)
