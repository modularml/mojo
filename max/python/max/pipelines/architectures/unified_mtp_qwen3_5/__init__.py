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
"""Qwen3.5 MTP head fused with its target for speculative decoding."""

from .arch import unified_mtp_qwen3_5_arch
from .model import UnifiedMTPQwen3_5Inputs, UnifiedMTPQwen3_5Model

__all__ = [
    "UnifiedMTPQwen3_5Inputs",
    "UnifiedMTPQwen3_5Model",
    "unified_mtp_qwen3_5_arch",
]
