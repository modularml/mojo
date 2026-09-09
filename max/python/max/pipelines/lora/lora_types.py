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

"""Shared types for LoRA queue operations."""

from __future__ import annotations

__all__ = [
    "LORA_REQUEST_ENDPOINT",
    "LORA_RESPONSE_ENDPOINT",
    "LoRAOperation",
    "LoRARequest",
    "LoRAResponse",
    "LoRAStatus",
    "LoRAType",
]

from enum import Enum

import msgspec

LORA_REQUEST_ENDPOINT = "lora_request"
LORA_RESPONSE_ENDPOINT = "lora_response"


class LoRAType(Enum):
    """Enumeration for LoRA Types."""

    A = "lora_A"
    """Represents the LoRA A matrix (high rank tensor to low rank tensor)."""

    B = "lora_B"
    """Represents the LoRA B matrix. (low rank tensor to high rank tensor)"""

    BIAS = "lora.bias"
    """Represents the LoRA bias matrix. (added to matrix B)"""


class LoRAOperation(Enum):
    """Enum for different LoRA operations."""

    LOAD = "load"
    """Loads a LoRA adapter into the serving infrastructure."""
    UNLOAD = "unload"
    """Unloads a previously loaded LoRA adapter."""


class LoRAStatus(Enum):
    """Enum for LoRA operation status."""

    SUCCESS = "success"
    """The LoRA operation completed successfully."""
    LOAD_NAME_EXISTS = "load_name_exists"
    """A LoRA adapter with the requested name is already loaded."""
    UNLOAD_NAME_NONEXISTENT = "unload_name_nonexistent"
    """No LoRA adapter with the requested name is currently loaded."""
    LOAD_ERROR = "load_error"
    """An error occurred while loading the LoRA adapter."""
    UNLOAD_ERROR = "unload_error"
    """An error occurred while unloading the LoRA adapter."""
    LOAD_INVALID_PATH = "load_invalid_path"
    """The path provided for the LoRA adapter is invalid or does not exist."""
    LOAD_INVALID_ADAPTER = "load_invalid_adapter"
    """The LoRA adapter at the specified path is malformed or incompatible."""
    UNSPECIFIED_ERROR = "unspecified_error"
    """An unexpected error occurred during the LoRA operation."""


class LoRARequest(msgspec.Struct, omit_defaults=True):
    """Container for LoRA adapter requests."""

    operation: LoRAOperation
    """The type of LoRA operation to perform (load or unload)."""

    lora_name: str
    """The unique name identifying the LoRA adapter."""

    lora_path: str | None = None
    """The filesystem path to the LoRA adapter weights. Required for load operations."""


class LoRAResponse(msgspec.Struct):
    """Response from LoRA operations."""

    status: LoRAStatus
    """The outcome status of the LoRA operation."""

    message: str | list[str]
    """A human-readable description of the operation result or error details."""
