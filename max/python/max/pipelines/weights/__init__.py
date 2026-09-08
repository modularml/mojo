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
"""Weight loading utilities for MAX pipelines."""

from .hf_utils import (
    HuggingFaceRepo,
    download_weight_files,
    generate_local_model_path,
    is_diffusion_pipeline,
    try_to_load_from_cache,
    validate_hf_repo_access,
)
from .quant import (
    apply_fused_kernel_flags,
    build_modelopt_nvfp4_config,
    gptq_quant_config,
    parse_quant_config,
    resolve_hf_quant_config,
)
from .weight_loading import AUTO_CAST_ENV_VAR, auto_cast_weights_from_env
from .weight_path_parser import WeightPathParser

__all__ = [
    "AUTO_CAST_ENV_VAR",
    "HuggingFaceRepo",
    "WeightPathParser",
    "apply_fused_kernel_flags",
    "auto_cast_weights_from_env",
    "build_modelopt_nvfp4_config",
    "download_weight_files",
    "generate_local_model_path",
    "gptq_quant_config",
    "is_diffusion_pipeline",
    "parse_quant_config",
    "resolve_hf_quant_config",
    "try_to_load_from_cache",
    "validate_hf_repo_access",
]
