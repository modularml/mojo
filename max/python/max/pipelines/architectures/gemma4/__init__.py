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
"""Gemma 4 vision-language architecture for multimodal text generation."""

from .arch import gemma4_arch, gemma4_unified_arch
from .model import Gemma3_MultiModalModel, Gemma3MultiModalModelInputs
from .model_config import (
    Gemma4ForConditionalGenerationConfig,
    Gemma4TextConfig,
    Gemma4VisionConfig,
)
from .reasoning import Gemma4ReasoningParser
from .tool_parser import Gemma4ToolParser

__all__ = [
    "Gemma3MultiModalModelInputs",
    "Gemma3_MultiModalModel",
    "Gemma4ForConditionalGenerationConfig",
    "Gemma4ReasoningParser",
    "Gemma4TextConfig",
    "Gemma4ToolParser",
    "Gemma4VisionConfig",
    "gemma4_arch",
    "gemma4_unified_arch",
]
