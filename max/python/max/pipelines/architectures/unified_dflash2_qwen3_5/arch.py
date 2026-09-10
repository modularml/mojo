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

from ..qwen3_5.model_config import Qwen3_5Config
from ..qwen3_5.reasoning import Qwen3_5ReasoningParser  # noqa: F401
from ..qwen3_5.tokenizer import Qwen3_5Tokenizer
from ..qwen3_5.tool_parser import Qwen3_5ToolParser  # noqa: F401
from .batch_processor import UnifiedDflash2Qwen3_5BatchProcessor
from .memory_planner import UnifiedDflash2Qwen3_5MemoryPlanner
from .model import UnifiedDflash2Qwen3_5Model
from .model_config import UnifiedDflash2Qwen3_5Config, dflash2_draft_width
from .weight_adapters import convert_qwen3_5_with_dflash2_state_dict

unified_dflash2_qwen3_5_arch = SupportedArchitecture(
    name="UnifiedDflash2Qwen3_5ForConditionalGeneration",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["RadixArk/Qwen3.8-27B-NVFP4"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=UnifiedDflash2Qwen3_5Config.DEFAULT_ENCODING,
    # The base class' constant rather than the alias on the fused config: the
    # docs table resolves this statically, and an alias renders an empty
    # encodings column (see generate-models-table.py).
    supported_encodings=Qwen3_5Config.SUPPORTED_ENCODINGS,
    pipeline_model=UnifiedDflash2Qwen3_5Model,
    tokenizer=Qwen3_5Tokenizer,
    context_type=Qwen3VLTextAndVisionContext,
    weight_adapters={
        WeightsFormat.safetensors: convert_qwen3_5_with_dflash2_state_dict,
    },
    required_arguments={
        "enable_prefix_caching": False,
    },
    config=UnifiedDflash2Qwen3_5Config,
    # The checkpoint fixes the width, so an omitted --num-speculative-tokens
    # resolves here rather than staying None while the graph runs at seven.
    checkpoint_draft_width=dflash2_draft_width,
    batching=UnifiedDflash2Qwen3_5BatchProcessor,
    multi_gpu_supported=False,
    tool_parser="qwen3_5",
    reasoning_parser="qwen3_5",
    # The base Qwen3.5 planner, unwrapped onto the target half. It reserves
    # one pool set even though this graph declares a shadow beside the live
    # one: `load_model` allocates neither, so a single-pool reservation is
    # what MAX actually owns here.
    memory_planner=UnifiedDflash2Qwen3_5MemoryPlanner,
    # Inherited from the MTP graph's rollback: the replay's row count depends
    # on how many tokens were accepted, so shapes change from step to step.
    supports_device_graph_capture=False,
    supports_overlap_scheduler=False,
)
