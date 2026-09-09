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

from . import weight_adapters
from .batch_processor import MiniMaxM2BatchProcessor
from .memory_planner import MiniMaxM2MemoryPlanner
from .model import MiniMaxM2Model
from .model_config import MiniMaxM2Config
from .tokenizer import MiniMaxM2Tokenizer

minimax_m2_arch = SupportedArchitecture(
    name="MiniMaxM2ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "MiniMaxAI/MiniMax-M2.7",
        "MiniMaxAI/MiniMax-M2.5",
        "lukealonso/MiniMax-M2.7-NVFP4",
        "amd/MiniMax-M2.7-MXFP4",
    ],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=MiniMaxM2Config.DEFAULT_ENCODING,
    supported_encodings=MiniMaxM2Config.SUPPORTED_ENCODINGS,
    pipeline_model=MiniMaxM2Model,
    tokenizer=MiniMaxM2Tokenizer,
    context_type=TextContext,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=MiniMaxM2Config,
    multi_gpu_supported=True,
    batching=MiniMaxM2BatchProcessor,
    tool_parser="minimax_m2",
    reasoning_parser="minimax_m2",
    memory_planner=MiniMaxM2MemoryPlanner,
)
