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
"""DFlash draft model for Kimi K2.5 (MLA target with MHA/GQA draft).

Same DFlash architecture as :mod:`dflash_llama3` (multi-layer non-causal
block transformer with external-KV materialization) but attached to a
Kimi K2.5 (DeepseekV3 MLA) target. The draft itself uses standard MHA/GQA
attention with Deepseek-yarn RoPE so its KV cache geometry is decoupled
from the target's MLA layout. Supports tensor-parallel and
data-parallel device topologies.
"""

from .dflash_kimi_k25 import DFlashKimiK25, DFlashKimiK25DraftConfig

__all__ = [
    "DFlashKimiK25",
    "DFlashKimiK25DraftConfig",
]
