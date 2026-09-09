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

"""Reads the plain-text prompt out of an OpenResponses request.

Generative pipelines that take a single prompt -- images, video, audio -- all
accept the same three input shapes, so they share this reader.
"""

from __future__ import annotations

from max.pipelines.request import OpenResponsesRequest
from max.pipelines.request.open_responses import InputTextContent

__all__ = ["retrieve_input_text"]


def retrieve_input_text(request: OpenResponsesRequest) -> str:
    """Returns the prompt text carried by ``request``.

    Accepts the three input shapes the API allows: a bare string, a message
    list whose first message holds a string, or a message list whose first
    message holds content parts (whose text parts are joined with spaces).

    Args:
        request: The request to read the prompt from.

    Returns:
        The prompt text.

    Raises:
        ValueError: If the request carries no text to use as a prompt.
    """
    if isinstance(request.body.input, str):
        return request.body.input

    if isinstance(request.body.input, list):
        if not request.body.input:
            raise ValueError("Input message list cannot be empty.")

        first_message = request.body.input[0]

        if isinstance(first_message.content, str):
            return first_message.content

        if isinstance(first_message.content, list):
            text_parts = [
                item.text
                for item in first_message.content
                if isinstance(item, InputTextContent)
            ]
            if not text_parts:
                raise ValueError(
                    "No text content found in message. Please include at least one "
                    "InputTextContent item with a text prompt."
                )
            return " ".join(text_parts)

        raise ValueError(
            f"Unexpected message content type: {type(first_message.content).__name__}"
        )

    raise ValueError(
        f"Input must be a string or list of messages, got {type(request.body.input).__name__}"
    )
