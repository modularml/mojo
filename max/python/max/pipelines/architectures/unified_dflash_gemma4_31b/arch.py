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
from max.pipelines.speculative._dflash import dflash_draft_width

from ..gemma4.memory_planner import Gemma4MemoryPlanner
from ..gemma4.tokenizer import Gemma4Tokenizer
from .batch_processor import UnifiedDflashGemma4_31BBatchProcessor
from .model import UnifiedDflashGemma4_31BModel
from .model_config import UnifiedDflashGemma4_31BConfig
from .weight_adapters import convert_safetensor_state_dict

unified_dflash_gemma4_31b_arch = SupportedArchitecture(
    name="UnifiedDflashGemma4_31BForCausalLM",
    example_repo_ids=[
        "google/gemma-4-31B-it",
        "nvidia/Gemma-4-31B-IT-NVFP4",
    ],
    default_encoding=UnifiedDflashGemma4_31BConfig.DEFAULT_ENCODING,
    supported_encodings=UnifiedDflashGemma4_31BConfig.SUPPORTED_ENCODINGS,
    pipeline_model=UnifiedDflashGemma4_31BModel,
    context_type=TextContext,
    # Gemma4Tokenizer + TextContext is the shipped text-only gemma4 pairing
    # (see gemma4_assistant): it exposes the reasoning delimiter ids the
    # thinking-phase tracking needs, which plain TextTokenizer does not.
    tokenizer=Gemma4Tokenizer,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    weight_adapters={
        WeightsFormat.safetensors: convert_safetensor_state_dict,
    },
    task=PipelineTask.TEXT_GENERATION,
    config=UnifiedDflashGemma4_31BConfig,
    memory_planner=Gemma4MemoryPlanner,
    supports_device_graph_capture=True,
    batching=UnifiedDflashGemma4_31BBatchProcessor,
    tool_parser="gemma4",
    reasoning_parser="gemma4",
    # Backend resolution runs after the registry rewrites the arch name, so
    # the base gemma4 arch's declaration never applies here; pin it or tool
    # grammars silently fall through to the global default.
    default_structured_output_backend="xgrammar",
    checkpoint_draft_width=dflash_draft_width,
)

# The generic draft-side registration ("DFlashDraftModel") lives in the
# shared ..dflash_llama3 package.
