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

"""Defines pipeline-specific exceptions for input validation errors."""

from __future__ import annotations

__all__ = ["InputError", "PromptTooLongError"]


class InputError(ValueError):
    """Exception raised for input validation errors that should be shown to users.

    This exception is specifically designed to provide user-friendly error messages
    for invalid input data, such as missing images for vision models or incorrect
    parameter combinations.
    """


class PromptTooLongError(InputError):
    """Raised when a prompt exceeds the model's maximum input length.

    Exposes ``num_tokens`` and ``max_length`` as attributes so callers can
    handle the failure programmatically (e.g., truncate and retry) instead
    of parsing the message.

    ``limit_description`` describes what is being limited, since the same
    failure mode means different things in different architectures (an LLM
    context window vs. a diffusion text encoder's max sequence length).
    """

    def __init__(
        self,
        num_tokens: int,
        max_length: int,
        *,
        limit_description: str = "configured maximum context length",
    ) -> None:
        self.num_tokens = num_tokens
        self.max_length = max_length
        super().__init__(
            f"Prompt is too long: {num_tokens} tokens exceeds the "
            f"{limit_description} of {max_length} tokens. "
            "Please shorten your prompt."
        )
