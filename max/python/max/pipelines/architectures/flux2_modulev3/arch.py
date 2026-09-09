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

from __future__ import annotations

from max.graph.weights import WeightsFormat
from max.pipelines.architectures.flux2 import Flux2ArchConfig
from max.pipelines.architectures.flux2.tokenizer import Flux2Tokenizer
from max.pipelines.context import PixelContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from .flux2_executor import FLUXModule

flux2_modulev3_arch = SupportedArchitecture(
    name="Flux2Pipeline_ModuleV3",
    task=PipelineTask.PIXEL_GENERATION,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    default_encoding=Flux2ArchConfig.DEFAULT_ENCODING,
    supported_encodings={"bfloat16", "float4_e2m1fnx2"},
    example_repo_ids=[
        "black-forest-labs/FLUX.2-dev",
        "black-forest-labs/FLUX.2-dev-NVFP4",
    ],
    pipeline_model=FLUXModule,
    context_type=PixelContext,
    default_weights_format=WeightsFormat.safetensors,
    tokenizer=Flux2Tokenizer,
    config=Flux2ArchConfig,
)
