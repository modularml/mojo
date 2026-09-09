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
"""Architecture registration for Bert sentence transformer."""

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .batch_processor import BertBatchProcessor
from .model import BertPipelineModel
from .model_config import BertModelConfig
from .tokenizer import BertTokenizer

bert_arch = SupportedArchitecture(
    name="BertModel",
    task=PipelineTask.EMBEDDINGS_GENERATION,
    example_repo_ids=[
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-MiniLM-L12-v2",
    ],
    default_encoding=BertModelConfig.DEFAULT_ENCODING,
    supported_encodings=BertModelConfig.SUPPORTED_ENCODINGS,
    pipeline_model=BertPipelineModel,
    tokenizer=BertTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    required_arguments={"enable_prefix_caching": False},
    config=BertModelConfig,
    batching=BertBatchProcessor,
)
