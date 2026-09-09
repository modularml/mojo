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

"""MiniCPM architecture registration for the MAX pipeline registry."""

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import (
    SupportedArchitecture,
    TextTokenizer,
)
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .model import MiniCPMModel
from .model_config import MiniCPMConfig

minicpm_arch = SupportedArchitecture(
    name="MiniCPMForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "Goekdeniz-Guelmez/MiniCPM-2B-dpo-bf16-safetensors",
    ],
    default_encoding="bfloat16",
    supported_encodings={"float32", "bfloat16"},
    context_type=TextContext,
    pipeline_model=MiniCPMModel,
    tokenizer=TextTokenizer,
    rope_type="normal",
    # The official openbmb checkpoints ship as pytorch_model.bin, not safetensors.
    # _resolve_weight_path will look for files matching this format first.
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
        WeightsFormat.gguf: weight_adapters.convert_gguf_state_dict,
    },
    multi_gpu_supported=False,
    config=MiniCPMConfig,
)
