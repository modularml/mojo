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
from .batch_processor import GptOssBatchProcessor
from .memory_planner import GptOssMemoryPlanner
from .model import GptOssModel
from .model_config import GptOssConfig

gpt_oss_arch = SupportedArchitecture(
    name="GptOssForCausalLM",
    example_repo_ids=[
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "unsloth/gpt-oss-20b-BF16",
    ],
    default_encoding=GptOssConfig.DEFAULT_ENCODING,
    supported_encodings=GptOssConfig.SUPPORTED_ENCODINGS,
    pipeline_model=GptOssModel,
    batching=GptOssBatchProcessor,
    task=PipelineTask.TEXT_GENERATION,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=GptOssConfig,
    memory_planner=GptOssMemoryPlanner,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
