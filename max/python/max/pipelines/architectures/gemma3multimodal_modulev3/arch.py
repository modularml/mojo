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
from max.pipelines.context import TextAndVisionContext
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import SupportedArchitecture, TextAndVisionTokenizer
from max.pipelines.modeling.types import InputModality, PipelineTask

from .batch_processor import Gemma3MultiModalModuleV3BatchProcessor
from .model import Gemma3MultiModalModelV3
from .model_config import Gemma3ForConditionalGenerationConfig

example_repo_ids = [
    # it = Instruction tuned (recommended).
    # pt = Pre-trained.
    "google/gemma-3-4b-it",
    "google/gemma-3-4b-pt",
    "google/gemma-3-12b-it",
    "google/gemma-3-12b-pt",
    "google/gemma-3-27b-it",
    "google/gemma-3-27b-pt",
]

gemma3_multimodal_modulev3_arch = SupportedArchitecture(
    name="Gemma3ForConditionalGeneration_ModuleV3",
    example_repo_ids=example_repo_ids,
    default_encoding=Gemma3ForConditionalGenerationConfig.DEFAULT_ENCODING,
    supported_encodings=Gemma3ForConditionalGenerationConfig.SUPPORTED_ENCODINGS,
    pipeline_model=Gemma3MultiModalModelV3,
    task=PipelineTask.TEXT_GENERATION,
    tokenizer=TextAndVisionTokenizer,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    required_arguments={
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
    },
    context_type=TextAndVisionContext,
    config=Gemma3ForConditionalGenerationConfig,
    batching=Gemma3MultiModalModuleV3BatchProcessor,
    memory_planner=PagedMemoryPlanner.with_activation_reservation(
        15 * 1024**3, always_signal_buffers=True
    ),
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
