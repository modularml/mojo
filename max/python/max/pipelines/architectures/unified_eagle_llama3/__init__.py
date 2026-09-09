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
"""EAGLE speculative decoding draft model for Llama 3 with unified graph compilation."""

from max.pipelines.lib.interfaces.batch_processor import PersistentInputBuffers

from .arch import unified_eagle_llama3_arch
from .model import (
    UnifiedEagleLlama3Inputs,
    UnifiedEagleLlama3Model,
)
from .model_config import UnifiedEagleLlama3Config

__all__ = [
    "PersistentInputBuffers",
    "UnifiedEagleLlama3Config",
    "UnifiedEagleLlama3Inputs",
    "UnifiedEagleLlama3Model",
    "unified_eagle_llama3_arch",
]
