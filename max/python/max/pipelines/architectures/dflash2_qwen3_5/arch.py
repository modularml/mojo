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

from ..llama3 import weight_adapters as llama3_weight_adapters
from ..llama3.batch_processor import Llama3BatchProcessor
from ..llama3.model_config import Llama3Config
from .model import DFlash2Qwen3_5Model

# Registers the DFlash2 draft model's HuggingFace architecture name
# (``DFlash2DraftModel``) with MAX's pipeline registry. The actual
# pipeline_model is a placeholder that raises on execute — DFlash2 drafts
# are only ever run through the fused DFlash2 pipeline. This registration
# exists so that PipelineConfig validation succeeds when the recipe YAML
# path is used, mirroring ``dflash_llama3.arch``.
dflash2_qwen3_5_arch = SupportedArchitecture(
    name="DFlash2DraftModel",
    example_repo_ids=[
        "z-lab/Qwen3.8-27B-DFlash2",
    ],
    default_encoding="bfloat16",
    supported_encodings={
        "bfloat16",
        "float32",
    },
    pipeline_model=DFlash2Qwen3_5Model,
    batching=Llama3BatchProcessor,
    context_type=TextContext,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.safetensors,
    # The draft body itself is single-device (DFlash2Qwen3_5 enforces that),
    # but this placeholder must not fail PipelineConfig validation when the
    # target it drafts for is served across several GPUs.
    multi_gpu_supported=True,
    weight_adapters={
        WeightsFormat.safetensors: llama3_weight_adapters.convert_safetensor_state_dict,
    },
    task=PipelineTask.TEXT_GENERATION,
    config=Llama3Config,
    memory_planner=PagedMemoryPlanner,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
