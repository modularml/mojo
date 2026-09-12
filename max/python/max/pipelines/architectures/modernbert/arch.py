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
"""Architecture registration for ModernBERT embedding models."""

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .model import ModernBertPipelineModel
from .model_config import ModernBertConfig
from .tokenizer import ModernBertTokenizer

modernbert_arch = SupportedArchitecture(
    name="ModernBertModel",
    task=PipelineTask.EMBEDDINGS_GENERATION,
    example_repo_ids=[
        "answerdotai/ModernBERT-base",
        "answerdotai/ModernBERT-large",
    ],
    default_encoding="bfloat16",
    supported_encodings={
        "float32",
        "bfloat16",
    },
    pipeline_model=ModernBertPipelineModel,
    tokenizer=ModernBertTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    required_arguments={"enable_prefix_caching": False},
    config=ModernBertConfig,
)

modernbert_for_masked_lm_arch = SupportedArchitecture(
    name="ModernBertForMaskedLM",
    task=PipelineTask.EMBEDDINGS_GENERATION,
    example_repo_ids=[
        "answerdotai/ModernBERT-base",
        "answerdotai/ModernBERT-large",
    ],
    default_encoding="bfloat16",
    supported_encodings={
        "float32",
        "bfloat16",
    },
    pipeline_model=ModernBertPipelineModel,
    tokenizer=ModernBertTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    required_arguments={"enable_prefix_caching": False},
    config=ModernBertConfig,
)
