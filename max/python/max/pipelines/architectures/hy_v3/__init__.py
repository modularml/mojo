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
"""Tencent Hunyuan Hy3-preview (`HYV3ForCausalLM`)."""

from .arch import hy_v3_arch
from .model import HYV3Inputs, HYV3Model
from .model_config import HYV3Config

ARCHITECTURES = [hy_v3_arch]

__all__ = [
    "ARCHITECTURES",
    "HYV3Config",
    "HYV3Inputs",
    "HYV3Model",
    "hy_v3_arch",
]
