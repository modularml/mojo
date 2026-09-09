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
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask
from max.pipelines.speculative._dflash import dflash_draft_width

from ..kimik2_5 import weight_adapters
from ..kimik2_5.context import KimiK2_5TextAndVisionContext
from ..kimik2_5.memory_planner import KimiK25MemoryPlanner
from ..kimik2_5.tokenizer import KimiK2_5VLTokenizer
from .batch_processor import UnifiedDflashKimiK25BatchProcessor
from .model import UnifiedDflashKimiK25Model
from .model_config import UnifiedDflashKimiK25Config

unified_dflash_kimi_k25_arch = SupportedArchitecture(
    name="UnifiedDflashKimiK25ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "nvidia/Kimi-K2.5-NVFP4",
    ],
    default_encoding=UnifiedDflashKimiK25Config.DEFAULT_ENCODING,
    supported_encodings=UnifiedDflashKimiK25Config.SUPPORTED_ENCODINGS,
    multi_gpu_supported=True,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    pipeline_model=UnifiedDflashKimiK25Model,
    batching=UnifiedDflashKimiK25BatchProcessor,
    tokenizer=KimiK2_5VLTokenizer,
    context_type=KimiK2_5TextAndVisionContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_kimik2_5_safetensor_state_dict,
    },
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    config=UnifiedDflashKimiK25Config,
    tool_parser="kimik2_5",
    reasoning_parser="kimik2_5",
    memory_planner=KimiK25MemoryPlanner,
    supports_device_graph_capture=False,
    checkpoint_draft_width=dflash_draft_width,
)
