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
"""DFlash speculative decoding for Llama3 with unified graph compilation."""

from max.pipelines.lib.interfaces.batch_processor import PersistentInputBuffers
from max.pipelines.speculative._dflash import DflashDraftHFConfig

from .arch import unified_dflash_llama3_arch
from .model import (
    UnifiedDflashLlama3Inputs,
    UnifiedDflashLlama3Model,
)
from .model_config import UnifiedDflashLlama3Config

__all__ = [
    "DflashDraftHFConfig",
    "PersistentInputBuffers",
    "UnifiedDflashLlama3Config",
    "UnifiedDflashLlama3Inputs",
    "UnifiedDflashLlama3Model",
    "unified_dflash_llama3_arch",
]
