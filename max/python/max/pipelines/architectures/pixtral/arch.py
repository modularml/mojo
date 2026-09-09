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
from max.pipelines.context.context_validators import (
    validate_requires_vision_context,
)
from max.pipelines.kv_cache.memory_planner import PagedMemoryPlanner
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from . import weight_adapters
from .batch_processor import PixtralBatchProcessor
from .model import PixtralModel
from .model_config import PixtralConfig
from .tokenizer import PixtralTokenizer

pixtral_arch = SupportedArchitecture(
    name="LlavaForConditionalGeneration",
    task=PipelineTask.TEXT_GENERATION,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    example_repo_ids=["mistral-experimental/pixtral-12b"],
    default_encoding=PixtralConfig.DEFAULT_ENCODING,
    supported_encodings=PixtralConfig.SUPPORTED_ENCODINGS,
    pipeline_model=PixtralModel,
    tokenizer=PixtralTokenizer,
    context_type=TextAndVisionContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    required_arguments={
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
    },
    context_validators=[
        validate_requires_vision_context,
    ],
    config=PixtralConfig,
    batching=PixtralBatchProcessor,
    memory_planner=PagedMemoryPlanner,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
