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
"""Registers the Laguna architecture (``LagunaForCausalLM``) with MAX.

Laguna is poolside's decoder-only sparse-MoE language model: sigmoid expert
routing with a per-expert score-correction bias, a per-element softplus
attention-output gate, and per-head QK-RMSNorm. This registration is verified
against ``poolside/Laguna-M.1-NVFP4`` (131B, compressed-tensors NVFP4 experts)
on a single B200.
"""

from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .model import LagunaModel
from .model_config import LagunaConfig
from .tokenizer import LagunaTokenizer

laguna_arch = SupportedArchitecture(
    name="LagunaForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "poolside/Laguna-M.1-NVFP4",
    ],
    default_weights_format=WeightsFormat.safetensors,
    default_encoding=LagunaConfig.DEFAULT_ENCODING,
    supported_encodings=LagunaConfig.SUPPORTED_ENCODINGS,
    pipeline_model=LagunaModel,
    tokenizer=LagunaTokenizer,
    context_type=TextContext,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=LagunaConfig,
    # Single-GPU only for now: verified end-to-end on one B200. Multi-GPU
    # expert parallelism needs a dedicated batch processor (mirroring
    # ``minimax_m2``) to inject the EP/DP graph inputs; the EP graph
    # scaffolding in ``model.py`` is in place but not yet validated.
    multi_gpu_supported=False,
    reasoning_parser="laguna",
    tool_parser="laguna",
)
