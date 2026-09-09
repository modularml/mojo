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

# ===----------------------------------------------------------------------=== #
# GritLM architecture registration for MAX pipeline.
# ===----------------------------------------------------------------------=== #
"""GritLM architecture descriptor."""

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .model import GritLMModel
from .model_config import GritLMConfig

gritlm_arch = SupportedArchitecture(
    # Must match the 'architectures' field in GritLM's config.json
    name="GritLM",
    example_repo_ids=["parasail-ai/GritLM-7B-vllm"],
    default_encoding="bfloat16",
    supported_encodings={"float32", "bfloat16"},
    pipeline_model=GritLMModel,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    # GritLM uses standard Mistral RoPE (non-interleaved safetensors)
    rope_type="normal",
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    task=PipelineTask.TEXT_GENERATION,
    config=GritLMConfig,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)
