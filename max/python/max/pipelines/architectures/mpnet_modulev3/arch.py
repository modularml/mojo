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
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .batch_processor import MPNetModuleV3BatchProcessor
from .model import MPNetPipelineModel
from .model_config import MPNetConfig

mpnet_modulev3_arch = SupportedArchitecture(
    name="MPNetForMaskedLM_ModuleV3",
    task=PipelineTask.EMBEDDINGS_GENERATION,
    example_repo_ids=[
        "sentence-transformers/all-mpnet-base-v2",
    ],
    default_encoding=MPNetConfig.DEFAULT_ENCODING,
    supported_encodings=MPNetConfig.SUPPORTED_ENCODINGS,
    pipeline_model=MPNetPipelineModel,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    required_arguments={"enable_prefix_caching": False},
    config=MPNetConfig,
    batching=MPNetModuleV3BatchProcessor,
)
