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
r"""A simple, model-agnostic tool-call format for cascade end-to-end tests.

Real models each emit tool calls in their own marker grammar, so the MAX
pipelines ship a per-architecture :class:`~max.pipelines.modeling.types.ToolParser`.
Exercising the cascade tool-calling path end to end shouldn't require a GPU
model, so this defines one deliberately simple format the ``echo`` pipeline can
replay: each call is ``<tool_call>NAME\n{json args}</tool_call>``, with the name
on the first line and the arguments as a JSON object after it. Splitting the
name from the raw-JSON arguments (rather than a single ``{"name", "arguments"}``
object) is exactly what makes streaming clean, so all the incremental
buffering, partial-marker holdback, and argument diffing come for free from
:class:`~max.pipelines.lib.tool_parsing.StructuralTagToolParser`.
"""

from __future__ import annotations

import re

from max.pipelines.lib.tool_parsing import (
    StructuralTagToolParser,
    generate_call_id,
    register,
)
from max.pipelines.modeling.types import ParsedToolCall

_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


@register("echo")
class EchoToolParser(StructuralTagToolParser):
    r"""Parses the cascade echo ``<tool_call>name\n{args}</tool_call>`` format.

    Flat layout (no outer section wrapper): repeated ``CALL_BEGIN`` …
    ``CALL_END`` pairs, so multiple tool calls in one response parse and stream
    independently.
    """

    CALL_BEGIN = "<tool_call>"
    CALL_END = "</tool_call>"

    def _parse_complete_section(
        self, tool_section: str
    ) -> list[ParsedToolCall]:
        calls: list[ParsedToolCall] = []
        for body in _CALL_RE.findall(tool_section):
            name, _, args = body.partition("\n")
            name = name.strip()
            args = args.strip()
            if not name:
                continue
            calls.append(
                ParsedToolCall(
                    id=generate_call_id(),
                    name=name,
                    arguments=args or "{}",
                )
            )
        return calls

    def _split_tool_call_body(
        self, body: str, is_complete: bool
    ) -> tuple[str | None, str | None]:
        # The name is the first line; the (possibly still-growing) JSON
        # arguments follow. Until the separating newline lands the name is not
        # yet complete, so signal "not ready" to hold everything back.
        name, sep, args = body.partition("\n")
        if not sep:
            return None, None
        return name.strip(), args
