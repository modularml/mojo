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
"""Inkling architecture registration."""

from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from .memory_planner import InklingMemoryPlanner
from .model import InklingModel
from .model_config import InklingConfig
from .reasoning import InklingReasoningParser  # noqa: F401  registers "inkling"
from .tokenizer import InklingTokenizer
from .tool_parser import InklingToolParser  # noqa: F401  registers "inkling"
from .weight_adapters import convert_safetensor_state_dict

inkling_arch = SupportedArchitecture(
    name="InklingForConditionalGeneration",
    task=PipelineTask.TEXT_GENERATION,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    example_repo_ids=["thinkingmachines/Inkling"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=InklingConfig.DEFAULT_ENCODING,
    supported_encodings=InklingConfig.SUPPORTED_ENCODINGS,
    pipeline_model=InklingModel,
    tokenizer=InklingTokenizer,
    context_type=TextAndVisionContext,
    config=InklingConfig,
    tool_parser="inkling",
    # The tool-call grammar is an xgrammar structural tag, so llguidance cannot
    # compile it.
    default_structured_output_backend="xgrammar",
    weight_adapters={
        WeightsFormat.safetensors: convert_safetensor_state_dict,
    },
    # Prefix caching not yet supported for SSM/hybrid models.
    required_arguments={"enable_prefix_caching": False},
    multi_gpu_supported=True,
    supports_device_graph_capture=True,
    memory_planner=InklingMemoryPlanner,
    reasoning_parser="inkling",
)
