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
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import (
    SupportedArchitecture,
    TextTokenizer,
)
from max.pipelines.modeling.types import PipelineTask

from ..llama3 import weight_adapters
from .batch_processor import Qwen3BatchProcessor
from .model import Qwen3Model
from .model_config import Qwen3Config
from .weight_adapters import convert_qwen3_moe_state_dict

qwen3_arch = SupportedArchitecture(
    name="Qwen3ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["Qwen/Qwen3-8B", "Qwen/Qwen3-30B-A3B"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=Qwen3Config.DEFAULT_ENCODING,
    supported_encodings=Qwen3Config.SUPPORTED_ENCODINGS,
    pipeline_model=Qwen3Model,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=Qwen3Config,
    batching=Qwen3BatchProcessor,
    multi_gpu_supported=True,
    memory_planner=PagedMemoryPlanner.with_activation_reservation(
        0, always_signal_buffers=True
    ),
    supports_device_graph_capture=False,
)

# Qwen3MoE architecture - uses the same model and config as Qwen3,
# but with MoE-specific weight adapter to handle expert weight stacking
qwen3_moe_arch = SupportedArchitecture(
    name="Qwen3MoeForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "Qwen/Qwen3-30B-A3B-Instruct",
        "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
    ],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=Qwen3Config.DEFAULT_ENCODING,
    supported_encodings=Qwen3Config.SUPPORTED_ENCODINGS,
    pipeline_model=Qwen3Model,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    weight_adapters={
        WeightsFormat.safetensors: convert_qwen3_moe_state_dict,
    },
    config=Qwen3Config,
    batching=Qwen3BatchProcessor,
    multi_gpu_supported=True,
    memory_planner=PagedMemoryPlanner.with_activation_reservation(
        0, always_signal_buffers=True
    ),
    supports_device_graph_capture=False,
)
