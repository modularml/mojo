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
)
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .batch_processor import MambaBatchProcessor
from .model import MambaModel
from .model_config import MambaConfig
from .tokenizer import MambaTokenizer

mamba_arch = SupportedArchitecture(
    name="MambaForCausalLM",
    example_repo_ids=[
        "state-spaces/mamba-130m-hf",
    ],
    default_encoding=MambaConfig.DEFAULT_ENCODING,
    supported_encodings=MambaConfig.SUPPORTED_ENCODINGS,
    pipeline_model=MambaModel,
    batching=MambaBatchProcessor,
    tokenizer=MambaTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    task=PipelineTask.TEXT_GENERATION,
    config=MambaConfig,
    memory_planner=PagedMemoryPlanner,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
