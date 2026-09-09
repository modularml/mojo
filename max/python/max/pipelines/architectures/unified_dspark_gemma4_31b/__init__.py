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
"""Unified DSpark speculative decoding for Gemma4 31B.

The draft checkpoint is in the vLLM speculators format; the shared parsing
and weight handling live in the ``speculators_common`` package.
"""

from .arch import unified_dspark_gemma4_31b_arch

__all__ = ["unified_dspark_gemma4_31b_arch"]
