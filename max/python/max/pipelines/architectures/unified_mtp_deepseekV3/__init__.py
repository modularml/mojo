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
"""DeepSeek-V3 multi-token prediction draft model for speculative decoding with unified graph compilation."""

from .arch import unified_mtp_deepseekV3_speculator
from .model import UnifiedMTPDeepseekV3Inputs, UnifiedMTPDeepseekV3Model

__all__ = [
    "UnifiedMTPDeepseekV3Inputs",
    "UnifiedMTPDeepseekV3Model",
    "unified_mtp_deepseekV3_speculator",
]
