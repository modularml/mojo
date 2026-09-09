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
"""Cases for the JSON Schema admission scenario."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SERVE: Literal["serve"] = "serve"
REFUSE: Literal["refuse"] = "refuse"


@dataclass(frozen=True)
class AdmissionCase:
    name: str
    expect: Literal["serve", "refuse"]
    schema: dict[str, object]
    prompt: str
    why: str


def _object(property_schema: dict[str, object]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {"v": property_schema},
        "required": ["v"],
        "additionalProperties": False,
    }


CASES: tuple[AdmissionCase, ...] = (
    AdmissionCase(
        "utf8_enum_values",
        SERVE,
        _object({"enum": ["日本語", "Français", "Ελληνικά", "Ω≈ç√"]}),
        'Pick the language tag "Français".',
        "a multi-byte enum value must survive the lowering to EBNF",
    ),
    AdmissionCase(
        "utf8_const_value",
        SERVE,
        _object({"const": "café — naïve — 🎉"}),
        "Echo the only allowed value.",
        "a const is emitted as a literal, so its bytes reach the EBNF verbatim",
    ),
    AdmissionCase(
        "utf8_property_name",
        SERVE,
        {
            "type": "object",
            "properties": {"寿司": {"type": "string"}},
            "required": ["寿司"],
            "additionalProperties": False,
        },
        'Set 寿司 to "yes".',
        "a property name is emitted as a key literal in the same EBNF",
    ),
    AdmissionCase(
        "property_name_with_spaces",
        SERVE,
        {
            "type": "object",
            "properties": {"Autonomous SheetML showcase": {"type": "string"}},
            "required": ["Autonomous SheetML showcase"],
            "additionalProperties": False,
        },
        'Set the showcase field to "ready".',
        "a space in a key is legal JSON and the quoted form can spell it",
    ),
    AdmissionCase(
        "enum_beside_satisfiable_maxlength",
        SERVE,
        _object({"enum": ["ab", "cd"], "maxLength": 8}),
        'Choose "cd".',
        "both values are 2 characters, so maxLength 8 rules out nothing",
    ),
    AdmissionCase(
        "enum_beside_satisfiable_required",
        SERVE,
        _object({"enum": [{"a": 1, "b": 2}], "required": ["a"]}),
        "Return the only allowed object.",
        "the single enumerated object carries the required key",
    ),
    AdmissionCase(
        "enum_beside_enum_descriptions",
        SERVE,
        _object(
            {"enum": ["red", "blue"], "enumDescriptions": ["warm", "cool"]}
        ),
        'Choose "blue".',
        "enumDescriptions is an annotation and constrains no instance",
    ),
    AdmissionCase(
        "enum_beside_unsatisfiable_minlength",
        REFUSE,
        _object({"enum": ["a"], "minLength": 5}),
        "Return the allowed value.",
        "no enumerated value can be 5 characters, so nothing satisfies both",
    ),
    AdmissionCase(
        "enum_multibyte_beside_unsatisfiable_minlength",
        REFUSE,
        _object({"enum": ["é"], "minLength": 2}),
        "Return the allowed value.",
        "one codepoint, two bytes: minLength 2 still rules the value out",
    ),
    AdmissionCase(
        "unique_items_true",
        REFUSE,
        _object(
            {"type": "array", "items": {"type": "integer"}, "uniqueItems": True}
        ),
        "Return three integers.",
        "a regular grammar cannot compare an element against its predecessors",
    ),
    AdmissionCase(
        "format_and_pattern_together",
        REFUSE,
        _object({"type": "string", "format": "email", "pattern": "^a+$"}),
        "Return the allowed string.",
        "one rule cannot enforce both regex tiers without dropping one",
    ),
    AdmissionCase(
        "ref_beside_same_object_sibling",
        REFUSE,
        {
            "type": "object",
            "$defs": {"x": {"type": "string"}},
            "properties": {"v": {"$ref": "#/$defs/x", "maxLength": 3}},
            "required": ["v"],
            "additionalProperties": False,
        },
        "Return a short string.",
        "the supported drafts disagree on whether the sibling applies",
    ),
)
