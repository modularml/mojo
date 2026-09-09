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

from .batch_processor import Step3p5BatchProcessor
from .model import Step3p5Model
from .model_config import Step3p5Config
from .weight_adapters import convert_step3p5_state_dict

step3p5_arch = SupportedArchitecture(
    name="Step3p5ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["stepfun-ai/Step-3.5-Flash"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=Step3p5Config.DEFAULT_ENCODING,
    supported_encodings=Step3p5Config.SUPPORTED_ENCODINGS,
    pipeline_model=Step3p5Model,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    weight_adapters={
        WeightsFormat.safetensors: convert_step3p5_state_dict,
    },
    config=Step3p5Config,
    batching=Step3p5BatchProcessor,
    multi_gpu_supported=True,
    memory_planner=PagedMemoryPlanner.with_activation_reservation(
        0, always_signal_buffers=True
    ),
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
