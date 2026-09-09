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
from .model_config import KimiK2_5Config, KimiK2_5TextConfig
from .tokenizer import KimiK2_5VLTokenizer
from .unified_eagle_mha_pipeline_model import Eagle3MHAKimiK25Model
from .unified_eagle_pipeline_model import Eagle3KimiK25Model

kimik2_5_arch = SupportedArchitecture(
    name="KimiK25ForConditionalGeneration",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "nvidia/Kimi-K2.5-NVFP4",
        "nvidia/Kimi-K2.6-NVFP4",
        "amd/Kimi-K2.7-Code-MXFP4",
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

kimivl_arch = SupportedArchitecture(
    name="KimiVLForConditionalGeneration",
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
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)

eagle3_kimik25_arch = SupportedArchitecture(
    name="Eagle3DeepseekV2ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["nvidia/Kimi-K2.5-NVFP4", "nvidia/Kimi-K2.6-NVFP4"],
    default_encoding=KimiK2_5TextConfig.DEFAULT_ENCODING,
    supported_encodings=KimiK2_5TextConfig.SUPPORTED_ENCODINGS,
    multi_gpu_supported=True,
    pipeline_model=Eagle3KimiK25Model,
    tokenizer=KimiK2_5VLTokenizer,
    context_type=KimiK2_5TextAndVisionContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_kimik2_5_safetensor_state_dict,
    },
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    config=KimiK2_5TextConfig,
    tool_parser="kimik2_5",
    reasoning_parser="kimik2_5",
    memory_planner=KimiK25MemoryPlanner,
)

eagle3_mha_kimik25_arch = SupportedArchitecture(
    name="Eagle3MHAKimiK25ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["nvidia/Kimi-K2.5-NVFP4", "nvidia/Kimi-K2.6-NVFP4"],
    default_encoding=KimiK2_5TextConfig.DEFAULT_ENCODING,
    supported_encodings=KimiK2_5TextConfig.SUPPORTED_ENCODINGS,
    multi_gpu_supported=True,
    pipeline_model=Eagle3MHAKimiK25Model,
    tokenizer=KimiK2_5VLTokenizer,
    context_type=KimiK2_5TextAndVisionContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_kimik2_5_safetensor_state_dict,
    },
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    config=KimiK2_5TextConfig,
    tool_parser="kimik2_5",
    reasoning_parser="kimik2_5",
    memory_planner=KimiK25MemoryPlanner,
)
