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

"""Drives a compiled Inkling tool grammar over a synthetic vocab.

The vocab is 256 single-byte tokens plus one token per Inkling marker, the
marker ids declared special so each is emittable only where the grammar names
it by token id.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np
from max._core import xgrammar as xgr
from max.pipelines.architectures.inkling.tokenizer import TOOL_CALL_JSON_MARKER
from max.pipelines.architectures.inkling.tool_parser import InklingToolParser

MESSAGE_MODEL = "<|message_model|>"
THINKING = "<|content_thinking|>"
TEXT = "<|content_text|>"
END_MESSAGE = "<|end_message|>"
STOP = "<|content_model_end_sampling|>"

MARKERS = [MESSAGE_MODEL, THINKING, TEXT, TOOL_CALL_JSON_MARKER, END_MESSAGE]


def tool(name: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": name, "parameters": parameters},
    }


def obj(
    properties: dict[str, Any], required: Sequence[str] | None = None
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required is not None:
        schema["required"] = list(required)
    return schema


UNIT_ENUM = obj({"unit": {"type": "string", "enum": ["c", "f"]}}, ["unit"])


def call(name: str, args_json: str) -> str:
    """One tool call as the model generates it, markers included."""
    return opened(name) + args_json + "}" + END_MESSAGE


def opened(name: str) -> str:
    """Everything up to the first argument key of a call to ``name``."""
    return f'{name}{TOOL_CALL_JSON_MARKER}{{"name":{json.dumps(name)},"args":'


class Grammar:
    """A compiled Inkling grammar plus the vocab needed to drive it."""

    def __init__(
        self,
        tools: list[dict[str, Any]],
        tool_choice: Any = "auto",
        response_format_schema: dict[str, Any] | None = None,
    ) -> None:
        vocab: list[bytes] = [bytes([i]) for i in range(256)]
        self.ids: dict[str, int] = {}
        for marker in [*MARKERS, STOP]:
            self.ids[marker] = len(vocab)
            vocab.append(marker.encode("utf-8"))
        self.vocab_size = len(vocab)

        info = xgr.TokenizerInfo(
            vocab,
            vocab_type=xgr.VocabType.RAW,
            stop_token_ids=[self.ids[STOP]],
            special_token_ids=[self.ids[m] for m in MARKERS],
        )
        self.compiled = xgr.GrammarCompiler(info).compile_structural_tag(
            InklingToolParser.generate_tool_call_grammar(
                tools=tools,
                tool_choice=tool_choice,
                response_format_schema=response_format_schema,
            )
        )

    def matcher(self) -> Any:
        return xgr.GrammarMatcher(self.compiled)

    def encode(self, text: str) -> list[int]:
        """Token ids for ``text``, markers as single tokens like the real vocab."""
        ids: list[int] = []
        rest = text
        while rest:
            for marker in [*MARKERS, STOP]:
                if rest.startswith(marker):
                    ids.append(self.ids[marker])
                    rest = rest[len(marker) :]
                    break
            else:
                ids.extend(rest[0].encode("utf-8"))
                rest = rest[1:]
        return ids

    def allowed(self, matcher: Any) -> set[int]:
        size = xgr.get_bitmask_size(self.vocab_size)
        bitmask = np.full((size,), -1, dtype=np.int32)
        matcher.fill_next_token_bitmask(bitmask)
        return {
            t
            for t in range(self.vocab_size)
            if (int(bitmask[t >> 5]) >> (t & 31)) & 1
        }

    def drive(self, matcher: Any, text: str) -> bool:
        """Feeds ``text`` token by token; False at the first rejection."""
        return all(matcher.accept_token(t) for t in self.encode(text))
