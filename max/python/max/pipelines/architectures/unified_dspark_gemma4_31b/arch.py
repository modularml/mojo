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

from ..gemma4.memory_planner import Gemma4MemoryPlanner
from ..gemma4.tokenizer import Gemma4Tokenizer
from ..speculators_common.draft_config import speculators_dspark_width
from .batch_processor import UnifiedDSparkGemma4_31BBatchProcessor
from .model import UnifiedDSparkGemma4_31BModel
from .model_config import UnifiedDSparkGemma4_31BConfig
from .weight_adapters import convert_safetensor_state_dict

unified_dspark_gemma4_31b_arch = SupportedArchitecture(
    name="UnifiedDSparkGemma4_31BForCausalLM",
    example_repo_ids=[
        "google/gemma-4-31B-it",
        "nvidia/Gemma-4-31B-IT-NVFP4",
    ],
    default_encoding=UnifiedDSparkGemma4_31BConfig.DEFAULT_ENCODING,
    supported_encodings=UnifiedDSparkGemma4_31BConfig.SUPPORTED_ENCODINGS,
    pipeline_model=UnifiedDSparkGemma4_31BModel,
    context_type=TextContext,
    tokenizer=Gemma4Tokenizer,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    weight_adapters={
        WeightsFormat.safetensors: convert_safetensor_state_dict,
    },
    task=PipelineTask.TEXT_GENERATION,
    config=UnifiedDSparkGemma4_31BConfig,
    memory_planner=Gemma4MemoryPlanner,
    # Capture-safe including structured output: the graph binds the shared
    # (pinned_bitmask, wait_payload, device_bitmask_scratch) triple whose
    # in-graph wait + H2D replay against process-lifetime pinned buffers
    # (see StructuredOutputOverlapState), the same wiring kimik2_5's dflash
    # recipes ship with capture on.
    supports_device_graph_capture=True,
    batching=UnifiedDSparkGemma4_31BBatchProcessor,
    tool_parser="gemma4",
    reasoning_parser="gemma4",
    # Backend resolution runs after the speculative arch rewrite, so the
    # base gemma4 arch's declaration never applies here.
    default_structured_output_backend="xgrammar",
    checkpoint_draft_width=speculators_dspark_width,
)

# The generic draft-side registration ("DSparkDraftModel") lives in the
# shared ..dspark_draft package.
