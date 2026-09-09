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

from max.dtype import DType
from max.graph.weights import WeightsFormat
from max.pipelines.architectures.qwen3vl_moe.context import (
    Qwen3VLTextAndVisionContext,
)
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask
from transformers import AutoConfig

from ..qwen3_5.memory_planner import Qwen3_5MemoryPlanner
from ..qwen3_5.model_config import Qwen3_5Config
from ..qwen3_5.reasoning import Qwen3_5ReasoningParser  # noqa: F401
from ..qwen3_5.tokenizer import Qwen3_5Tokenizer
from ..qwen3_5.tool_parser import Qwen3_5ToolParser  # noqa: F401
from .batch_processor import UnifiedMTPQwen3_5BatchProcessor
from .model import UnifiedMTPQwen3_5Model
from .weight_adapters import convert_qwen3_5_with_mtp_state_dict


class UnifiedMTPQwen3_5Config(Qwen3_5Config):
    """Qwen3.5's config with the vision-cache facts withdrawn.

    The fused MTP graph is text-only -- ``_create_model_config`` drops
    ``vision_config`` so no encoder is compiled -- while the checkpoint it
    reads still declares a vision tower. Reporting the base architecture's
    per-entry estimate would make memory planning reserve a slice of the KV
    pool for an encoder cache this graph can never fill.
    """

    @classmethod
    def estimate_vision_cache_entry_bytes(
        cls, huggingface_config: AutoConfig
    ) -> int:
        return 0

    @classmethod
    def get_vision_cache_row_spec(
        cls, huggingface_config: AutoConfig
    ) -> tuple[int, DType] | None:
        return None


unified_mtp_qwen3_5_arch = SupportedArchitecture(
    name="UnifiedMTPQwen3_5ForConditionalGeneration",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["RadixArk/Qwen3.8-27B-NVFP4"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=Qwen3_5Config.DEFAULT_ENCODING,
    supported_encodings=Qwen3_5Config.SUPPORTED_ENCODINGS,
    pipeline_model=UnifiedMTPQwen3_5Model,
    tokenizer=Qwen3_5Tokenizer,
    context_type=Qwen3VLTextAndVisionContext,
    weight_adapters={
        WeightsFormat.safetensors: convert_qwen3_5_with_mtp_state_dict,
    },
    required_arguments={
        "enable_prefix_caching": False,
    },
    config=UnifiedMTPQwen3_5Config,
    batching=UnifiedMTPQwen3_5BatchProcessor,
    multi_gpu_supported=True,
    tool_parser="qwen3_5",
    reasoning_parser="qwen3_5",
    # Reused as-is even though this graph's ABI declares a shadow pool set
    # beside the live one: `load_model` allocates neither, so the planner's
    # single-pool reservation is what MAX actually owns here. Sizing both sets
    # belongs to whoever allocates them -- the serving engine.
    memory_planner=Qwen3_5MemoryPlanner,
    # The rollback replays the verify pass's own intermediates, whose row
    # count depends on how many tokens were accepted, so the graph's shapes
    # change from step to step.
    supports_device_graph_capture=False,
)
