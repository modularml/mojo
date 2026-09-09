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
from max.pipelines.architectures.qwen3vl_moe.context import (
    Qwen3VLTextAndVisionContext,
)
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask

from .batch_processor import Qwen3_5BatchProcessor
from .memory_planner import Qwen3_5MemoryPlanner
from .model import Qwen3_5Model
from .model_config import Qwen3_5Config
from .reasoning import Qwen3_5ReasoningParser  # noqa: F401  registers "qwen3_5"
from .tokenizer import Qwen3_5Tokenizer
from .tool_parser import Qwen3_5ToolParser  # noqa: F401  registers "qwen3_5"
from .weight_adapters import convert_qwen3_5_state_dict

qwen3_5_arch = SupportedArchitecture(
    name="Qwen3_5ForConditionalGeneration",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["Qwen/Qwen3.5-27B", "Qwen/Qwen3.8-27B"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=Qwen3_5Config.DEFAULT_ENCODING,
    supported_encodings=Qwen3_5Config.SUPPORTED_ENCODINGS,
    pipeline_model=Qwen3_5Model,
    tokenizer=Qwen3_5Tokenizer,
    context_type=Qwen3VLTextAndVisionContext,
    weight_adapters={
        WeightsFormat.safetensors: convert_qwen3_5_state_dict,
    },
    required_arguments={
        "enable_prefix_caching": False,  # TODO: Remove when Deltanet supports prefix caching
    },
    config=Qwen3_5Config,
    batching=Qwen3_5BatchProcessor,
    multi_gpu_supported=True,
    tool_parser="qwen3_5",
    reasoning_parser="qwen3_5",
    memory_planner=Qwen3_5MemoryPlanner,
    # Requires Qwen3_5Model.release_warmup_state (SupportsSSMStateWarmup):
    # each capture-warmup probe claims state pool slots that must be released
    # before the next probe, or warmup exhausts the pool.
    supports_device_graph_capture=True,
)
