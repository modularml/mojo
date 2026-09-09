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
"""Unified DSpark speculative decoding for Gemma4."""

from .arch import gemma4_dspark_draft_arch, unified_dspark_gemma4_12b_arch
from .dspark_gemma4 import (
    DSparkGemma4,
    DSparkGemma4Attention,
    DSparkGemma4DecoderLayer,
    DSparkGemma4DraftConfig,
    DSparkMarkovHead,
)
from .model import UnifiedDSparkGemma4_12BInputs, UnifiedDSparkGemma4_12BModel
from .model_config import UnifiedDSparkGemma4_12BConfig
from .unified_dspark_gemma4_12b import UnifiedDSparkGemma4_12B

__all__ = [
    "DSparkGemma4",
    "DSparkGemma4Attention",
    "DSparkGemma4DecoderLayer",
    "DSparkGemma4DraftConfig",
    "DSparkMarkovHead",
    "UnifiedDSparkGemma4_12B",
    "UnifiedDSparkGemma4_12BConfig",
    "UnifiedDSparkGemma4_12BInputs",
    "UnifiedDSparkGemma4_12BModel",
    "gemma4_dspark_draft_arch",
    "unified_dspark_gemma4_12b_arch",
]
