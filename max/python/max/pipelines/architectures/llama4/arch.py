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
"""Registers the Llama4 (text-only) architecture."""

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    SupportedArchitecture,
    TextTokenizer,
)
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .model import Llama4Model
from .model_config import Llama4Config

# Text-only Llama 4 Scout. The released Scout checkpoint declares
# ``Llama4ForConditionalGeneration`` (multimodal); the text tower is also
# registered under that name so the language-model weights can be loaded while
# the vision tower is skipped by the weight adapter.
_EXAMPLE_REPO_IDS = [
    "unsloth/Llama-4-Scout-17B-16E-Instruct",
]

llama4_arch = SupportedArchitecture(
    name="Llama4ForCausalLM",
    example_repo_ids=_EXAMPLE_REPO_IDS,
    default_encoding=Llama4Config.DEFAULT_ENCODING,
    supported_encodings=Llama4Config.SUPPORTED_ENCODINGS,
    pipeline_model=Llama4Model,
    task=PipelineTask.TEXT_GENERATION,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    # ``grouped_matmul_ragged`` raises ``hipErrorStreamCaptureUnsupported``
    # during HIP graph-capture warmup on ROCm, so opt out of auto device-graph
    # capture. This lets Llama4 Scout serve on MI355X out-of-the-box without
    # ``--no-device-graph-capture`` until the kernel gains stream-capture
    # support.
    supports_device_graph_capture=False,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=Llama4Config,
)

llama4_conditional_arch = SupportedArchitecture(
    name="Llama4ForConditionalGeneration",
    example_repo_ids=_EXAMPLE_REPO_IDS,
    default_encoding=Llama4Config.DEFAULT_ENCODING,
    supported_encodings=Llama4Config.SUPPORTED_ENCODINGS,
    pipeline_model=Llama4Model,
    task=PipelineTask.TEXT_GENERATION,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    # ``grouped_matmul_ragged`` raises ``hipErrorStreamCaptureUnsupported``
    # during HIP graph-capture warmup on ROCm, so opt out of auto device-graph
    # capture. This lets Llama4 Scout serve on MI355X out-of-the-box without
    # ``--no-device-graph-capture`` until the kernel gains stream-capture
    # support.
    supports_device_graph_capture=False,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=Llama4Config,
)
