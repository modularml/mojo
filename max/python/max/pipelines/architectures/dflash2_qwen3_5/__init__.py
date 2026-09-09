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
"""DFlash2 draft model for Qwen3.5-family targets.

DFlash v1's non-causal block drafter plus a two-tap grouped dynamic
convolution around each sublayer and a top-k candidate-path selector.
"""

from .arch import dflash2_qwen3_5_arch
from .dflash2_qwen3_5 import DFlash2Qwen3_5, DFlash2TransformerBlock
from .layers import DFlash2CandidateSelector, DFlash2GroupedConv
from .model import DFlash2Qwen3_5Model

__all__ = [
    "DFlash2CandidateSelector",
    "DFlash2GroupedConv",
    "DFlash2Qwen3_5",
    "DFlash2Qwen3_5Model",
    "DFlash2TransformerBlock",
    "dflash2_qwen3_5_arch",
]
