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
"""Hy3-preview registration shell."""

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    SupportedArchitecture,
    TextTokenizer,
)
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .batch_processor import HyV3BatchProcessor
from .memory_planner import HyV3MemoryPlanner
from .model import HYV3Model
from .model_config import HYV3Config

hy_v3_arch = SupportedArchitecture(
    name="HYV3ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "tencent/Hy3-preview",
    ],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=HYV3Config.DEFAULT_ENCODING,
    supported_encodings=HYV3Config.SUPPORTED_ENCODINGS,
    pipeline_model=HYV3Model,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=HYV3Config,
    multi_gpu_supported=True,
    batching=HyV3BatchProcessor,
    memory_planner=HyV3MemoryPlanner,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
