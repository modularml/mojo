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
"""DeepseekV3 + Eagle3 speculator pipeline."""

from .arch import (
    eagle3_deepseekV3_speculator,
    eagle3_mha_deepseekV3_speculator,
)
from .eagle3_draft import Eagle3DeepseekV3
from .mha_pipeline import Eagle3MHADeepseekV3Inputs, Eagle3MHADeepseekV3Model
from .model import Eagle3DeepseekV3Inputs, Eagle3DeepseekV3Model
from .unified_eagle import Eagle3DeepseekV3Unified
from .weight_adapters import convert_eagle3_draft_state_dict

__all__ = [
    "Eagle3DeepseekV3",
    "Eagle3DeepseekV3Inputs",
    "Eagle3DeepseekV3Model",
    "Eagle3DeepseekV3Unified",
    "Eagle3MHADeepseekV3Inputs",
    "Eagle3MHADeepseekV3Model",
    "convert_eagle3_draft_state_dict",
    "eagle3_deepseekV3_speculator",
    "eagle3_mha_deepseekV3_speculator",
]
