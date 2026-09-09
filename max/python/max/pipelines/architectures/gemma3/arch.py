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
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .batch_processor import Gemma3BatchProcessor
from .model import Gemma3Model
from .model_config import Gemma3Config

gemma3_arch = SupportedArchitecture(
    name="Gemma3ForCausalLM",
    example_repo_ids=[
        # it = Instruction tuned (recommended).
        # pt = Pre-trained.
        "google/gemma-3-1b-it",
        "google/gemma-3-1b-pt",
        # We have a different architecture for >=4B models. See gemma3multimodal
        # for more information.
    ],
    default_encoding=Gemma3Config.DEFAULT_ENCODING,
    supported_encodings=Gemma3Config.SUPPORTED_ENCODINGS,
    pipeline_model=Gemma3Model,
    batching=Gemma3BatchProcessor,
    task=PipelineTask.TEXT_GENERATION,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=Gemma3Config,
    memory_planner=PagedMemoryPlanner.with_activation_reservation(
        0, always_signal_buffers=True
    ),
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
