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
from max.pipelines.context import TextAndVisionContext
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from .batch_processor import Idefics3BatchProcessor
from .model import Idefics3Model
from .model_config import Idefics3Config
from .tokenizer import Idefics3Tokenizer

idefics3_arch = SupportedArchitecture(
    name="Idefics3ForConditionalGeneration",
    task=PipelineTask.TEXT_GENERATION,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    example_repo_ids=["HuggingFaceM4/Idefics3-8B-Llama3"],
    default_encoding=Idefics3Config.DEFAULT_ENCODING,
    supported_encodings=Idefics3Config.SUPPORTED_ENCODINGS,
    pipeline_model=Idefics3Model,
    tokenizer=Idefics3Tokenizer,
    context_type=TextAndVisionContext,
    default_weights_format=WeightsFormat.safetensors,
    required_arguments={
        "enable_chunked_prefill": False,
        "enable_prefix_caching": False,
    },
    config=Idefics3Config,
    batching=Idefics3BatchProcessor,
    memory_planner=PagedMemoryPlanner,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
