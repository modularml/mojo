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

from __future__ import annotations

from max.pipelines.context import TextContext
from max.pipelines.lora import LoRAManagerV3


def can_allocate_lora_request(
    ctx: TextContext, active_loras: set[str], lora_manager: LoRAManagerV3 | None
) -> bool:
    """Checks if the LoRA request can be allocated and serviced.

    This function ensures that activating a new LoRA during CE batch construction
    will not cause eviction of LoRAs needed by the TG batch. It allows evicting
    non-protected LoRAs from previous batches to make room for new requests.

    Args:
        ctx: The text context containing the request and model name.
        active_loras: Set of LoRA names that are currently protected (from TG batch).
            These LoRAs must not be evicted.
        lora_manager: The LoRA manager instance.

    Returns:
        True if the LoRA request can be safely allocated without evicting
        protected LoRAs, False otherwise.
    """
    # This should only be called when lora_manager exists
    assert lora_manager is not None

    # Non-LoRA requests can always be allocated
    if not lora_manager.is_lora(ctx.model_name):
        return True

    # If this LoRA is already in the protected set, no additional slot needed
    if ctx.model_name in active_loras:
        return True

    # If this LoRA is already globally active (even if not protected),
    # calling activate_adapter() will just refresh its LRU position
    # without triggering eviction
    if lora_manager.is_active_lora(ctx.model_name):
        return True

    # We need a new slot for this LoRA.
    # Check if (protected LoRAs + new LoRA) fits within capacity.
    # Non-protected globally active LoRAs can be evicted to make room.
    num_protected = len(active_loras)
    return num_protected + 1 <= lora_manager.max_num_loras


def is_lora(ctx: TextContext, lora_manager: LoRAManagerV3 | None) -> bool:
    """Helper that checks the manager is not None and if the context is a lora"""
    return bool(lora_manager and lora_manager.is_lora(ctx.model_name))


def is_active_lora(
    ctx: TextContext, lora_manager: LoRAManagerV3 | None
) -> bool:
    """Helper that checks the manager is not None and if the LoRA is active"""
    return bool(lora_manager and lora_manager.is_active_lora(ctx.model_name))
