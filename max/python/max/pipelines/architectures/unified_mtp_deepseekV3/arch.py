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

from ..deepseekV3.memory_planner import DeepseekV3MemoryPlanner
from ..deepseekV3.model_config import DeepseekV3Config
from .batch_processor import UnifiedMTPDeepseekV3BatchProcessor
from .model import UnifiedMTPDeepseekV3Model
from .weight_adapters import convert_with_mtp_state_dict

unified_mtp_deepseekV3_arch = SupportedArchitecture(
    name="UnifiedMTPDeepseekV3ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "deepseek-ai/DeepSeek-V3",
    ],
    default_encoding=DeepseekV3Config.DEFAULT_ENCODING,
    supported_encodings={"bfloat16", "float8_e4m3fn", "float4_e2m1fnx2"},
    multi_gpu_supported=True,
    pipeline_model=UnifiedMTPDeepseekV3Model,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: convert_with_mtp_state_dict,
    },
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    config=DeepseekV3Config,
    memory_planner=DeepseekV3MemoryPlanner,
    batching=UnifiedMTPDeepseekV3BatchProcessor,
)
