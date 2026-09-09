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

from .batch_processor import LFM2BatchProcessor
from .model import LFM2Model
from .model_config import LFM2Config
from .weight_adapters import convert_lfm2_safetensor_state_dict

lfm2_arch = SupportedArchitecture(
    name="Lfm2ForCausalLM",
    default_encoding=LFM2Config.DEFAULT_ENCODING,
    task=PipelineTask.TEXT_GENERATION,
    supported_encodings=LFM2Config.SUPPORTED_ENCODINGS,
    example_repo_ids=["LiquidAI/LFM2.5-350M", "LiquidAI/LFM2.5-350M-Base"],
    pipeline_model=LFM2Model,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    required_arguments={
        "allow_safetensors_weights_fp32_bf16_bidirectional_cast": True,
        "trust_remote_code": True,
    },
    multi_gpu_supported=False,
    weight_adapters={
        WeightsFormat.safetensors: convert_lfm2_safetensor_state_dict,
    },
    config=LFM2Config,
    batching=LFM2BatchProcessor,
    memory_planner=PagedMemoryPlanner,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
