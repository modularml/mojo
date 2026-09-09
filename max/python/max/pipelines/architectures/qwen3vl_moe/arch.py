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
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from .batch_processor import Qwen3VLMoeBatchProcessor
from .context import Qwen3VLTextAndVisionContext
from .model import Qwen3VLModel
from .model_config import Qwen3VLConfig
from .tokenizer import Qwen3VLTokenizer
from .weight_adapters import convert_qwen3vl_model_state_dict

qwen3vl_moe_arch = SupportedArchitecture(
    name="Qwen3VLMoeForConditionalGeneration",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "Qwen/Qwen3-VL-30B-A3B-Instruct",
    ],
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    default_encoding=Qwen3VLConfig.DEFAULT_ENCODING,
    supported_encodings=Qwen3VLConfig.SUPPORTED_ENCODINGS,
    weight_adapters={
        WeightsFormat.safetensors: convert_qwen3vl_model_state_dict,
    },
    pipeline_model=Qwen3VLModel,
    batching=Qwen3VLMoeBatchProcessor,
    tokenizer=Qwen3VLTokenizer,
    context_type=Qwen3VLTextAndVisionContext,
    required_arguments={
        "enable_chunked_prefill": False,
    },
    config=Qwen3VLConfig,
    memory_planner=PagedMemoryPlanner.with_activation_reservation(
        10 * 1024**3, always_signal_buffers=True
    ),
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)

# Register the same architecture under Qwen's non-MoE name for models like Qwen3-VL-4B-Instruct
# repo https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
qwen3vl_arch = SupportedArchitecture(
    name="Qwen3VLForConditionalGeneration",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-2B-Instruct"],
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    default_encoding=Qwen3VLConfig.DEFAULT_ENCODING,
    supported_encodings=Qwen3VLConfig.SUPPORTED_ENCODINGS,
    weight_adapters={
        WeightsFormat.safetensors: convert_qwen3vl_model_state_dict,
    },
    pipeline_model=Qwen3VLModel,
    batching=Qwen3VLMoeBatchProcessor,
    tokenizer=Qwen3VLTokenizer,
    context_type=Qwen3VLTextAndVisionContext,
    required_arguments={
        "enable_chunked_prefill": False,
    },
    config=Qwen3VLConfig,
    memory_planner=PagedMemoryPlanner.with_activation_reservation(
        10 * 1024**3, always_signal_buffers=True
    ),
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
