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
from max.pipelines.lib import (
    SupportedArchitecture,
    TextTokenizer,
)
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .batch_processor import DeepseekV3NextNBatchProcessor
from .memory_planner import DeepseekV3NextNMemoryPlanner
from .model import DeepseekV3NextNModel
from .model_config import DeepseekV3NextNConfig

deepseekV3_nextn_arch = SupportedArchitecture(
    name="DeepseekV3ForCausalLMNextN",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "SGLang/DeepSeek-V3-NextN",
    ],
    default_encoding=DeepseekV3NextNConfig.DEFAULT_ENCODING,
    supported_encodings=DeepseekV3NextNConfig.SUPPORTED_ENCODINGS,
    multi_gpu_supported=True,
    pipeline_model=DeepseekV3NextNModel,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    batching=DeepseekV3NextNBatchProcessor,
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    config=DeepseekV3NextNConfig,
    memory_planner=DeepseekV3NextNMemoryPlanner,
)
