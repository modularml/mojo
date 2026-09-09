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

"""LoRA adapter management for MAX pipelines."""

from .config import LoRAConfig
from .helpers import get_lora_manager
from .lora import ADAPTER_CONFIG_FILE, LoRAModel
from .lora_types import (
    LORA_REQUEST_ENDPOINT,
    LORA_RESPONSE_ENDPOINT,
    LoRAOperation,
    LoRARequest,
    LoRAResponse,
    LoRAStatus,
    LoRAType,
)
from .modulev3 import (
    NUM_ROUTING_INPUTS,
    LoRAManagerV3,
    LoRATargetModule,
)

__all__ = [
    "ADAPTER_CONFIG_FILE",
    "LORA_REQUEST_ENDPOINT",
    "LORA_RESPONSE_ENDPOINT",
    "NUM_ROUTING_INPUTS",
    "LoRAConfig",
    "LoRAManagerV3",
    "LoRAModel",
    "LoRAOperation",
    "LoRARequest",
    "LoRAResponse",
    "LoRAStatus",
    "LoRATargetModule",
    "LoRAType",
    "get_lora_manager",
]
