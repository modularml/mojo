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
"""Qwen3.5 fused with a DFlash2 block drafter for speculative decoding."""

from .arch import unified_dflash2_qwen3_5_arch
from .model import (
    GRAPH_NAME,
    UnifiedDflash2Qwen3_5Inputs,
    UnifiedDflash2Qwen3_5Model,
)
from .model_config import UnifiedDflash2Qwen3_5Config

__all__ = [
    "GRAPH_NAME",
    "UnifiedDflash2Qwen3_5Config",
    "UnifiedDflash2Qwen3_5Inputs",
    "UnifiedDflash2Qwen3_5Model",
    "unified_dflash2_qwen3_5_arch",
]
