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

"""Gemma 4 transformer architecture for text generation (ModuleV3 API)."""

from .arch import gemma4_modulev3_arch, gemma4_unified_modulev3_arch
from .inputs import Gemma4Inputs
from .model import Gemma4Model

__all__ = [
    "Gemma4Inputs",
    "Gemma4Model",
    "gemma4_modulev3_arch",
    "gemma4_unified_modulev3_arch",
]
