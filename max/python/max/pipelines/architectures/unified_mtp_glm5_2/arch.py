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
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask

from ..deepseekV3_2.memory_planner import DeepseekV3_2MemoryPlanner
from ..glm5_1.model_config import Glm5_1Config
from ..glm5_1.reasoning import (
    GlmReasoningParser,  # noqa: F401  registers "glm45"
)
from ..glm5_1.tokenizer import GlmTokenizer
from ..glm5_1.tool_parser import GlmToolParser  # noqa: F401  registers "glm45"
from .batch_processor import UnifiedMTPGlm5_2BatchProcessor
from .model import UnifiedMTPGlm5_2Model
from .weight_adapters import convert_with_mtp_state_dict

unified_mtp_glm5_2_arch = SupportedArchitecture(
    name="UnifiedMTPGlmMoeDsaForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "zai-org/GLM-5.2-FP8",
    ],
    default_encoding="float8_e4m3fn",
    supported_encodings={
        "float4_e2m1fnx2",
        "float8_e4m3fn",
        "bfloat16",
    },
    multi_gpu_supported=True,
    pipeline_model=UnifiedMTPGlm5_2Model,
    tokenizer=GlmTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: convert_with_mtp_state_dict,
    },
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    config=Glm5_1Config,
    memory_planner=DeepseekV3_2MemoryPlanner,
    batching=UnifiedMTPGlm5_2BatchProcessor,
    tool_parser="glm45",
    reasoning_parser="glm45",
    default_structured_output_backend="xgrammar",
    # Matches the base GlmMoeDsaForCausalLM arch; see the rationale there.
    default_structured_output_any_whitespace=True,
)
