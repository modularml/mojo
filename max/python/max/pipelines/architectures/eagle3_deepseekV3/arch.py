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
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from ..deepseekV3 import weight_adapters as deepseekV3_weight_adapters
from ..deepseekV3.memory_planner import DeepseekV3MemoryPlanner
from ..deepseekV3.model_config import DeepseekV3Config
from .batch_processor import (
    Eagle3DeepseekV3BatchProcessor,
    Eagle3MHADeepseekV3BatchProcessor,
)
from .mha_pipeline import Eagle3MHADeepseekV3Model
from .model import Eagle3DeepseekV3Model

eagle3_deepseekV3_arch = SupportedArchitecture(
    name="Eagle3DeepseekV3ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    # TODO(MXSERV-7): Move ``austinpowers/Kimi-K2.5-NVFP4-DeepseekV3`` to
    # the official Modular HF org so CI doesn't depend on a personal account.
    example_repo_ids=[
        "austinpowers/Kimi-K2.5-NVFP4-DeepseekV3",
    ],
    default_encoding=DeepseekV3Config.DEFAULT_ENCODING,
    supported_encodings={"bfloat16", "float8_e4m3fn", "float4_e2m1fnx2"},
    multi_gpu_supported=True,
    pipeline_model=Eagle3DeepseekV3Model,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: deepseekV3_weight_adapters.convert_safetensor_state_dict,
    },
    batching=Eagle3DeepseekV3BatchProcessor,
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    config=DeepseekV3Config,
    memory_planner=DeepseekV3MemoryPlanner,
)

eagle3_mha_deepseekV3_arch = SupportedArchitecture(
    name="Eagle3MHADeepseekV3ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "austinpowers/Kimi-K2.5-NVFP4-DeepseekV3",
    ],
    default_encoding=DeepseekV3Config.DEFAULT_ENCODING,
    supported_encodings={"bfloat16", "float8_e4m3fn", "float4_e2m1fnx2"},
    multi_gpu_supported=True,
    pipeline_model=Eagle3MHADeepseekV3Model,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: deepseekV3_weight_adapters.convert_safetensor_state_dict,
    },
    batching=Eagle3MHADeepseekV3BatchProcessor,
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    config=DeepseekV3Config,
    memory_planner=DeepseekV3MemoryPlanner,
)
