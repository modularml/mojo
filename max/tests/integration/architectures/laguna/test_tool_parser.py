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

from max.pipelines.architectures.laguna.tool_parser import LagunaToolParser

# Exactly what Laguna's chat template renders for an assistant tool call.
_RENDERED = (
    "<tool_call>get_weather\n"
    "<arg_key>city</arg_key>\n"
    "<arg_value>Paris</arg_value>\n"
    "<arg_key>units</arg_key>\n"
    "<arg_value>celsius</arg_value>\n"
    "</tool_call>\n"
)


def test_parse_complete_single_call() -> None:
    parsed = LagunaToolParser().parse_complete("Sure.\n" + _RENDERED)
    assert parsed.content == "Sure."
    assert len(parsed.tool_calls) == 1
    call = parsed.tool_calls[0]
    assert call.name == "get_weather"
    assert json.loads(call.arguments) == {"city": "Paris", "units": "celsius"}


def test_parse_complete_typed_values() -> None:
    # Non-string args are rendered as JSON by the template; strings are bare.
    rendered = (
        "<tool_call>calc\n"
        "<arg_key>n</arg_key>\n<arg_value>42</arg_value>\n"
        "<arg_key>flag</arg_key>\n<arg_value>true</arg_value>\n"
        "<arg_key>items</arg_key>\n<arg_value>[1, 2]</arg_value>\n"
        "<arg_key>label</arg_key>\n<arg_value>hi there</arg_value>\n"
        "</tool_call>"
    )
    call = LagunaToolParser().parse_complete(rendered).tool_calls[0]
    assert json.loads(call.arguments) == {
        "n": 42,
        "flag": True,
        "items": [1, 2],
        "label": "hi there",
    }


def test_parse_complete_no_tool_call() -> None:
    parsed = LagunaToolParser().parse_complete("Just a plain answer.")
    assert parsed.content == "Just a plain answer."
    assert parsed.tool_calls == []


def test_parse_delta_streams_full_call() -> None:
    parser = LagunaToolParser()
    # Feed the rendered call split mid-tag across chunks.
    chunks = [
        "Sure.",
        "\n<tool_",
        "call>get_we",
        "ather\n<arg_key>city</arg_key>\n",  # spellchecker:disable-line
        "<arg_value>Paris</arg_value>\n</tool_",
        "call>\n",
    ]
    name = None
    args = ""
    content = ""
    for ch in chunks:
        out = parser.parse_delta(ch)
        for d in out or []:
            if d.content:
                content += d.content
            if d.name:
                name = d.name
            if d.arguments:
                args += d.arguments
    assert content.strip() == "Sure."
    assert name == "get_weather"
    assert json.loads(args) == {"city": "Paris"}
