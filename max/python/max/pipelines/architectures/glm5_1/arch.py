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
from max.pipelines.architectures.deepseekV3.batch_processor import (
    DeepseekV3BatchProcessor,
)
from max.pipelines.architectures.deepseekV3_2 import weight_adapters
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask

from .model import Glm5_1Model
from .model_config import Glm5_1Config
from .reasoning import GlmReasoningParser  # noqa: F401  registers "glm45"
from .tokenizer import GlmTokenizer
from .tool_parser import GlmToolParser  # noqa: F401  registers "glm45"

glm5_1_arch = SupportedArchitecture(
    name="GlmMoeDsaForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "zai-org/GLM-5.1",
        "zai-org/GLM-5.1-FP8",
        "zai-org/GLM-5.2",
        "zai-org/GLM-5.2-FP8",
        "zai-org/GLM-5",
    ],
    default_encoding=Glm5_1Config.DEFAULT_ENCODING,
    supported_encodings=Glm5_1Config.SUPPORTED_ENCODINGS,
    multi_gpu_supported=True,
    pipeline_model=Glm5_1Model,
    batching=DeepseekV3BatchProcessor,
    tokenizer=GlmTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    config=Glm5_1Config,
    memory_planner=PagedMemoryPlanner,
    tool_parser="glm45",
    reasoning_parser="glm45",
    default_structured_output_backend="xgrammar",
    # GLM strongly prefers pretty-printed JSON: under the compact grammar its
    # content-bearing continuations are all masked at the first array decision
    # and the schema's shortest terminator wins (measured 8/8 degenerate
    # {"findings":[]} vs 0/8 whitespace-tolerant on GLM-5.2). Whitespace runs
    # stay bounded via STRUCTURED_OUTPUT_MAX_WHITESPACE_RUN.
    default_structured_output_any_whitespace=True,
)
