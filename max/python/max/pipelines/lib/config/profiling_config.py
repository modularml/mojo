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
"""MAX profiling configuration."""

from __future__ import annotations

import os
from typing import get_args

from max.config import ConfigFileModel
from max.engine import GPUProfilingMode
from pydantic import ConfigDict, Field, PrivateAttr, field_validator


class ProfilingConfig(ConfigFileModel):
    """Configuration for the GPU (NVTX/Nsight) profiler."""

    model_config = ConfigDict(frozen=True)

    # validate_default so the MODULAR_ENABLE_PROFILING fallback below also
    # applies when the field is not provided at all.
    gpu_profiling: GPUProfilingMode = Field(
        default="off",
        validate_default=True,
        description="Whether to enable GPU profiling of the model.",
    )
    """Whether to enable GPU profiling of the model."""

    _config_file_section_name: str = PrivateAttr(default="profiling_config")
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""

    @field_validator("gpu_profiling", mode="before")
    @classmethod
    def _normalize_gpu_profiling(cls, value: object) -> object:
        """Applies MODULAR_ENABLE_PROFILING when the value is "off"."""
        if value == "off":
            gpu_profiling_env = os.environ.get(
                "MODULAR_ENABLE_PROFILING", "off"
            )
            valid_values = list(get_args(GPUProfilingMode))
            if gpu_profiling_env not in valid_values:
                raise ValueError(
                    "gpu_profiling must be one of: " + ", ".join(valid_values)
                )
            return gpu_profiling_env
        return value
