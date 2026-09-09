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
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from .batch_processor import InternVLBatchProcessor
from .memory_planner import InternVLMemoryPlanner
from .model import InternVLModel
from .model_config import InternVLConfig
from .tokenizer import InternVLTokenizer

internvl_arch = SupportedArchitecture(
    name="InternVLChatModel",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=["OpenGVLab/InternVL3-8B-Instruct"],
    default_encoding=InternVLConfig.DEFAULT_ENCODING,
    supported_encodings=InternVLConfig.SUPPORTED_ENCODINGS,
    pipeline_model=InternVLModel,
    tokenizer=InternVLTokenizer,
    context_type=TextAndVisionContext,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    required_arguments={
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
    },
    config=InternVLConfig,
    batching=InternVLBatchProcessor,
    memory_planner=InternVLMemoryPlanner,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
