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

import json
import uuid
from collections.abc import Mapping
from typing import Any

from max.pipelines.lib.tool_parsing import register
from max.pipelines.modeling.types import (
    ParsedToolCall,
    ParsedToolCallDelta,
    ParsedToolResponse,
)

from .json_utils import parse_json_from_text


# TODO: SERVOPT-1219 Rename LlamaToolParser and move to max.pipelines.architecture.
@register("llama")
class LlamaToolParser:
    """Parses Llama-style tool calls from model responses.

    Llama models output tool calls as JSON objects with "name" and "parameters"
    fields embedded in the response text.
    """

    def __init__(self) -> None:
        self._buffer: str = ""

    def parse_complete(self, response: str) -> ParsedToolResponse:
        """Parses a complete response into tool calls."""
        tool_calls: list[ParsedToolCall] = []

        if json_objects := parse_json_from_text(response):
            for tool_data in json_objects:
                if "name" in tool_data and "parameters" in tool_data:
                    short_uuid = str(uuid.uuid4()).replace("-", "")[:16]
                    tool_call = ParsedToolCall(
                        id=f"call_{short_uuid}",
                        name=tool_data.get("name"),
                        arguments=json.dumps(tool_data.get("parameters")),
                    )
                    tool_calls.append(tool_call)
                else:
                    raise ValueError(
                        "Both name and parameters not present in parsed JSON response."
                    )

        return ParsedToolResponse(content=None, tool_calls=tool_calls)

    def parse_delta(self, delta: str) -> list[ParsedToolCallDelta] | None:
        """Parses incremental deltas for streaming tool calls.

        Note: Streaming tool call parsing for Llama is not yet implemented.
        This method accumulates tokens but does not emit chunks.
        """
        self._buffer += delta
        # TODO(SERVOPT-1180): Implement streaming delta parsing
        return None

    def reset(self) -> None:
        """Resets internal state for a new streaming session."""
        self._buffer = ""

    def set_streaming_tool_schemas(
        self, schemas: Mapping[str, dict[str, Any]]
    ) -> None:
        """No-op: this format does not need schema-driven streaming.

        See ``ToolParser.set_streaming_tool_schemas``.
        """
        return
