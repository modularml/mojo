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

from . import weight_adapters
from .batch_processor import KimiK2_5BatchProcessor
from .context import KimiK2_5TextAndVisionContext
from .memory_planner import KimiK25MemoryPlanner
from .model import KimiK2_5Model
from .model_config import KimiK2_5Config
from .tokenizer import KimiK2_5VLTokenizer

# Kimi-K2.5 (ModuleV3). The vision encoder runs single-device; the DeepseekV3
# language tower is multi-GPU (TP attention + EP MoE), so multi_gpu_supported
# stays True. The EAGLE/dflash speculative-decode variants are not migrated
# (they depend on ``eagle_common``, which has no V3 counterpart).
kimik2_5_modulev3_arch = SupportedArchitecture(
    name="KimiK25ForConditionalGeneration_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "nvidia/Kimi-K2.5-NVFP4",
        "nvidia/Kimi-K2.6-NVFP4",
    ],
    default_encoding=KimiK2_5Config.DEFAULT_ENCODING,
    supported_encodings=KimiK2_5Config.SUPPORTED_ENCODINGS,
    multi_gpu_supported=True,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    pipeline_model=KimiK2_5Model,
    batching=KimiK2_5BatchProcessor,
    tokenizer=KimiK2_5VLTokenizer,
    context_type=KimiK2_5TextAndVisionContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_kimik2_5_safetensor_state_dict,
    },
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    config=KimiK2_5Config,
    tool_parser="kimik2_5",
    reasoning_parser="kimik2_5",
    memory_planner=KimiK25MemoryPlanner,
)

kimivl_modulev3_arch = SupportedArchitecture(
    name="KimiVLForConditionalGeneration_ModuleV3",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "moonshotai/Kimi-VL-A3B-Instruct",
    ],
    default_encoding=KimiK2_5Config.DEFAULT_ENCODING,
    supported_encodings=KimiK2_5Config.SUPPORTED_ENCODINGS,
    multi_gpu_supported=True,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    pipeline_model=KimiK2_5Model,
    batching=KimiK2_5BatchProcessor,
    tokenizer=KimiK2_5VLTokenizer,
    context_type=KimiK2_5TextAndVisionContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_kimivl_safetensor_state_dict,
    },
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    config=KimiK2_5Config,
    tool_parser="kimik2_5",
    reasoning_parser="kimik2_5",
    memory_planner=KimiK25MemoryPlanner,
)
