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
from max.pipelines.context import PixelContext
from max.pipelines.diffusion.config import GENERIC_TAYLORSEER_DEFAULTS
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from ..qwen_image.arch import QwenImageArchConfig
from .pipeline_qwen_image_edit import QwenImageEditPipeline
from .tokenizer import QwenImageEditTokenizer

qwen_image_edit_arch = SupportedArchitecture(
    name="QwenImageEditPipeline",
    task=PipelineTask.PIXEL_GENERATION,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    default_encoding=QwenImageArchConfig.DEFAULT_ENCODING,
    supported_encodings={"bfloat16"},
    example_repo_ids=[
        "Qwen/Qwen-Image-Edit-2511",
    ],
    pipeline_model=QwenImageEditPipeline,  # type: ignore[arg-type]
    context_type=PixelContext,
    default_weights_format=WeightsFormat.safetensors,
    tokenizer=QwenImageEditTokenizer,
    config=QwenImageArchConfig,
    denoising_cache_defaults=GENERIC_TAYLORSEER_DEFAULTS,
)

qwen_image_edit_plus_arch = SupportedArchitecture(
    name="QwenImageEditPlusPipeline",
    task=PipelineTask.PIXEL_GENERATION,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    default_encoding=QwenImageArchConfig.DEFAULT_ENCODING,
    supported_encodings={"bfloat16"},
    example_repo_ids=[
        "Qwen/Qwen-Image-Edit-2511",
    ],
    pipeline_model=QwenImageEditPipeline,  # type: ignore[arg-type]
    context_type=PixelContext,
    default_weights_format=WeightsFormat.safetensors,
    tokenizer=QwenImageEditTokenizer,
    config=QwenImageArchConfig,
    denoising_cache_defaults=GENERIC_TAYLORSEER_DEFAULTS,
)
