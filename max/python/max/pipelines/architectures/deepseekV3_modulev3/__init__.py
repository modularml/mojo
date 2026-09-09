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
"""DeepSeek-V3 mixture-of-experts architecture for text generation."""

from .arch import deepseekV3_modulev3_arch
from .model import DeepseekV3Inputs, DeepseekV3Model
from .model_config import DeepseekV3Config

__all__ = [
    "DeepseekV3Config",
    "DeepseekV3Inputs",
    "DeepseekV3Model",
    "deepseekV3_modulev3_arch",
]
