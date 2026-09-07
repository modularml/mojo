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
"""Speculative decoding pipelines and configuration for MAX."""

from .config import (
    MAGIC_DRAFT_TOKEN_ID,
    RejectionSamplingStrategy,
    SpeculativeConfig,
    SpeculativeMethod,
    VerifyWidthRange,
)
from .ragged_token_merger import RaggedTokenMerger, ragged_token_merger

__all__ = [
    "MAGIC_DRAFT_TOKEN_ID",
    "RaggedTokenMerger",
    "RejectionSamplingStrategy",
    "SpeculativeConfig",
    "SpeculativeMethod",
    "VerifyWidthRange",
    "ragged_token_merger",
]
