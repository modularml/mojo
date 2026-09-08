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
"""Unified Inkling MTP architecture registration."""

from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextAndVisionContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from ..inkling.model_config import InklingConfig
from ..inkling.tokenizer import InklingTokenizer
from .memory_planner import UnifiedMTPInklingMemoryPlanner
from .model import UnifiedMTPInklingModel
from .weight_adapters import convert_with_mtp_state_dict

unified_mtp_inkling_arch = SupportedArchitecture(
    name="UnifiedMTPInklingForConditionalGeneration",
    task=PipelineTask.TEXT_GENERATION,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    example_repo_ids=["thinkingmachines/Inkling"],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=InklingConfig.DEFAULT_ENCODING,
    supported_encodings={"bfloat16", "float4_e2m1fnx2"},
    pipeline_model=UnifiedMTPInklingModel,
    tokenizer=InklingTokenizer,
    context_type=TextAndVisionContext,
    config=InklingConfig,
    tool_parser="inkling",
    default_structured_output_backend="xgrammar",
    reasoning_parser="inkling",
    weight_adapters={
        WeightsFormat.safetensors: convert_with_mtp_state_dict,
    },
    required_arguments={"enable_prefix_caching": False},
    multi_gpu_supported=True,
    supports_device_graph_capture=True,
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    memory_planner=UnifiedMTPInklingMemoryPlanner,
)
