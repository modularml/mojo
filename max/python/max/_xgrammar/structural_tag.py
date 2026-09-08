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

# Vendored by Modular from github.com/mlc-ai/xgrammar (v0.2.2).
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 xgrammar Contributors
# SPDX-License-Identifier: Apache-2.0

"""Defines all structural tag formats."""

import json
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

try:
    # Python 3.9+
    from typing import Annotated
except ImportError:
    # Python 3.8
    from typing_extensions import Annotated

from pydantic import BaseModel, Field

# ---------- Basic Formats ----------


class ConstStringFormat(BaseModel):
    """A format that matches a constant string."""

    type: Literal["const_string"] = "const_string"
    """The type of the format."""
    value: str
    """The constant string."""


class JSONSchemaFormat(BaseModel):
    """A format that matches a JSON schema."""

    type: Literal["json_schema"] = "json_schema"
    """The type of the format."""
    json_schema: Union[bool, Dict[str, Any]]
    """The JSON schema."""
    style: Literal[
        "json",
        "qwen_xml",
        "minimax_xml",
        "deepseek_xml",
        "glm_xml",
        "minimax_m3_xml",
    ] = "json"
    """How to parse the content. Valid values:

    - ``"json"`` — standard JSON.
    - ``"qwen_xml"`` — Qwen XML (no MAX architecture selects this style today):
      ``<parameter=key>value</parameter>``.
    - ``"minimax_xml"`` — MiniMax M2.5 / M2.7 XML:
      ``<parameter name="key">value</parameter>``. Matches upstream xgrammar
      style name for MiniMax M2 models.
    - ``"deepseek_xml"`` — DeepSeek XML (no MAX architecture selects this style today):
      ``<{dsml_token}parameter name="key" string="true|false">value</{dsml_token}parameter>``.
    - ``"glm_xml"`` — GLM 5 / 5.1 / 5.2 XML:
      ``<arg_key>key</arg_key><arg_value>value</arg_value>``.
    - ``"minimax_m3_xml"`` — MiniMax M3 tag-prefixed XML:
      ``prefix<key>value prefix</key>``.
    """
    any_order: bool = False
    """Allow object properties to appear in any order."""
    string_value_delimiter_token: Optional[Union[str, int]] = None
    """Optional delimiter token (string or id) wrapping string *values*."""
    string_value_forbidden_tokens: List[Union[str, int]] = []
    """Token strings/ids a string value's body must never contain. Requires
    ``string_value_delimiter_token`` (token-level exclusion applies only to token-delimited
    string values); rejected if set without it."""
    additional_bare_key_terminal: str = ""
    """Raw EBNF terminal for generated additional bare keys; declared and
    pattern-constrained keys are unaffected. Left empty for quoted keys."""
    bare_key_literal_forbidden: str = ""
    """EBNF char-class body a bare literal key must avoid. Empty = quoted keys / no constraint."""
    bare_key_pattern_forbidden: str = ""
    """EBNF char-class body a bare pattern-derived key must avoid. Empty = quoted keys / no constraint."""
    max_whitespace_cnt: Optional[int] = None
    """Optional maximum inter-token whitespace count; None means unbounded."""
    require_object_root: bool = False
    """Require the schema root to produce an object-typed value, coercing an unconstrained root to an empty open object."""
    reject_unsupported: bool = False
    """Reject unenforceable schema keywords rather than fall back to unconstrained decoding."""
    strict_mode: bool = True
    """Disallow properties and items not specified in the schema."""
    xml_tag_prefix: str = ""
    """Prefix emitted before every XML open and close tag."""
    any_whitespace: bool = True
    """Allow any whitespace between JSON elements."""
    indent: Optional[int] = None
    """Number of spaces for indentation. None = single line."""
    separators: Optional[Tuple[str, str]] = None
    """Comma and colon separators. None = defaults."""


class AnyTextFormat(BaseModel):
    """A format that matches any text."""

    type: Literal["any_text"] = "any_text"
    """The type of the format."""

    excludes: List[str] = []
    """List of strings that should not appear in the matched text."""


class TokenFormat(BaseModel):
    """A format that matches a single token by ID or string representation."""

    type: Literal["token"] = "token"
    """The type of the format."""

    token: Union[int, str]
    """The token ID (int) or token string (str)."""


class ExcludeTokenFormat(BaseModel):
    """A format that matches a single token, excluding those in the given set."""

    type: Literal["exclude_token"] = "exclude_token"
    """The type of the format."""

    exclude_tokens: List[Union[int, str]] = []
    """List of token IDs or strings to exclude."""


class AnyTokensFormat(BaseModel):
    """A format that matches zero or more tokens, excluding those in the given set."""

    type: Literal["any_tokens"] = "any_tokens"
    """The type of the format."""

    exclude_tokens: List[Union[int, str]] = []
    """List of token IDs or strings to exclude."""


class GrammarFormat(BaseModel):
    """A format that matches an ebnf grammar."""

    type: Literal["grammar"] = "grammar"
    """The type of the format."""

    grammar: str
    """The ebnf grammar."""


class RegexFormat(BaseModel):
    """A format that matches a regex pattern."""

    type: Literal["regex"] = "regex"
    """The type of the format."""

    pattern: str
    """The regex pattern."""


# ---------- Combinatorial Formats ----------


class SequenceFormat(BaseModel):
    """A format that matches a sequence of formats."""

    type: Literal["sequence"] = "sequence"
    """The type of the format."""
    elements: List["Format"]
    """The elements of the sequence."""


class OrFormat(BaseModel):
    """A format that matches one of the formats."""

    type: Literal["or"] = "or"
    """The type of the format."""
    elements: List["Format"]
    """The elements of the or."""


class TagFormat(BaseModel):
    """A format that matches a tag: ``begin content end``.

    The ``end`` field can be a single string or a list of possible end strings.
    When multiple end strings are provided, any of them will be accepted as a valid
    ending for the tag.

    Examples
    --------

    Single end string:

    .. code-block:: python

        TagFormat(begin="<response>", content=..., end="</response>")

    Multiple end strings:

    .. code-block:: python

        TagFormat(begin="<response>", content=..., end=["</response>", "</answer>"])

    """

    type: Literal["tag"] = "tag"
    """The type of the format."""
    begin: Union[str, TokenFormat]
    """The begin tag. Can be a string or a TokenFormat."""
    content: "Format"
    """The content of the tag. It can be any of the formats."""
    end: Union[str, List[str], TokenFormat]
    """The end tag(s). Can be a string, list of strings, or a TokenFormat."""


class TriggeredTagsFormat(BaseModel):
    """A format that matches triggered tags. It can allow any output until a trigger is
    encountered, then dispatch to the corresponding tag; when the end tag is encountered, the
    grammar will allow any following output, until the next trigger is encountered.

    Each tag should be matched by exactly one trigger. "matching" means the trigger should be a
    prefix of the begin tag.

    Tags must use **string** ``begin`` fields, not ``TokenFormat``.
    For token-level dispatch, use ``TokenTriggeredTagsFormat`` instead.

    Examples
    --------

    .. code-block:: python

        structural_tag = TriggeredTagsFormat(
            triggers=["<function="],
            tags=[
                TagFormat(
                    begin="<function=func1>",
                    content=JSONSchemaFormat(json_schema=...),
                    end="</function>",
                ),
                TagFormat(
                    begin="<function=func2>",
                    content=JSONSchemaFormat(json_schema=...),
                    end="</function>",
                ),
            ],
            at_least_one=False,
            stop_after_first=False,
        )

    The above structural tag can accept the following outputs::

        <function=func1>{"name": "John", "age": 30}</function>
        <function=func2>{"name": "Jane", "age": 25}</function>
        any_text<function=func1>{"name": "John", "age": 30}</function>any_text1<function=func2>{"name": "Jane", "age": 25}</function>any_text2

    """

    type: Literal["triggered_tags"] = "triggered_tags"
    """The type of the format."""
    triggers: List[str]
    """The triggers of the triggered tags."""
    tags: List[TagFormat]
    """The tags of the triggered tags."""
    at_least_one: bool = False
    """Whether at least one of the tags must be generated."""
    stop_after_first: bool = False
    """Whether to stop after the first tag is generated."""
    excludes: List[str] = []
    """List of strings that should not appear in the matched text."""


class TokenTriggeredTagsFormat(BaseModel):
    """A format that dispatches to tags based on token-level triggers.

    Similar to TriggeredTagsFormat but uses token IDs instead of string triggers.
    Tags must use ``TokenFormat`` ``begin`` fields, not strings.
    For string-level dispatch, use ``TriggeredTagsFormat`` instead.
    """

    type: Literal["token_triggered_tags"] = "token_triggered_tags"
    """The type of the format."""
    trigger_tokens: List[Union[int, str]]
    """The trigger token IDs or strings."""
    tags: List[TagFormat]
    """The tags to dispatch to."""
    exclude_tokens: List[Union[int, str]] = []
    """List of token IDs or strings to exclude."""
    at_least_one: bool = False
    """Whether at least one tag must be generated."""
    stop_after_first: bool = False
    """Whether to stop after the first tag is generated."""


class TagsWithSeparatorFormat(BaseModel):
    """A format that matches a tags with separator. It can match zero, one, or more tags, separated
    by the separator, with no other text allowed.

    Examples
    --------

    .. code-block:: python

        structural_tag = TagsWithSeparatorFormat(
            tags=[
                TagFormat(begin="<function=func1>", content=JSONSchemaFormat(json_schema=...), end="</function>"),
                TagFormat(begin="<function=func2>", content=JSONSchemaFormat(json_schema=...), end="</function>"),
            ],
            separator=",",
            at_least_one=False,
            stop_after_first=False,
        )

    The above structural tag can accept an empty string, or the following outputs::

        <function=func1>{"name": "John", "age": 30}</function>
        <function=func1>{"name": "John", "age": 30}</function>,<function=func2>{"name": "Jane", "age": 25}</function>
        <function=func1>{"name": "John", "age": 30}</function>,<function=func2>{"name": "Jane", "age": 25}</function>,<function=func1>{"name": "John", "age": 30}</function>
    """

    type: Literal["tags_with_separator"] = "tags_with_separator"
    """The type of the format."""
    tags: List[TagFormat]
    """The tags of the tags with separator."""
    separator: str
    """The separator of the tags with separator."""
    at_least_one: bool = False
    """Whether at least one of the tags must be matched."""
    stop_after_first: bool = False
    """Whether to stop after the first tag is matched."""


class OptionalFormat(BaseModel):
    """A format that matches the content 0 or 1 time (EBNF optional).

    Semantics: the inner format may appear once or not at all.
    """

    type: Literal["optional"] = "optional"
    """The type of the format."""
    content: "Format"
    """The format that may appear 0 or 1 time."""


class PlusFormat(BaseModel):
    """A format that matches the content 1 or more times (EBNF plus).

    Semantics: the inner format must appear at least once.
    """

    type: Literal["plus"] = "plus"
    """The type of the format."""
    content: "Format"
    """The format that must appear at least once."""


class StarFormat(BaseModel):
    """A format that matches the content 0 or more times (EBNF star).

    Semantics: the inner format may appear any number of times.
    """

    type: Literal["star"] = "star"
    """The type of the format."""
    content: "Format"
    """The format that may appear 0 or more times."""


class RepeatFormat(BaseModel):
    """A format that matches the content between min and max times (inclusive).

    Use max=-1 for unbounded upper limit (e.g. "at least min times").
    """

    type: Literal["repeat"] = "repeat"
    """The type of the format."""
    min: int
    """Minimum number of occurrences (inclusive)."""
    max: int
    """Maximum number of occurrences (inclusive). Use -1 for unbounded."""
    content: "Format"
    """The format that is repeated."""


class DispatchFormat(BaseModel):
    """Matches certain patterns in free-form text.
    When certain strings are generated, the following content must follow the corresponding format.
    Certain strings can be excluded from being generated in the free-form part.

    The user specifies a list of (pattern, formats). The LLM can generate any free-form text, but
    when a pattern is matched in the text, the following output must follow the corresponding format.
    The ``loop`` field controls whether the matching of this structure ends after accepting the first
    pattern and format. If False, it ends. If true, the matching continues, and this format will
    continue to allow free-form text and detect the next pattern to be generated.
    The ``excludes`` field controls the strings that cannot be generated in the free-form text.
    It can also control the end of the format:
    ``SequenceFormat(DispatchFormat(..., excludes=""), ConstStringFormat(""))``
    or
    ``TagFormat(begin=..., content=DispatchFormat(..., excludes=""), end="")``
    ends the matching of the format when LLM generates <end>.
    """

    type: Literal["dispatch"] = "dispatch"
    """The type of the format."""
    rules: List[Tuple[str, "Format"]]
    """List of ``(pattern, content format)`` pairs."""
    loop: bool = True
    """If true, after handling one dispatched format, it will continue to allow free-form text
    and match the next pattern. Otherwise, the matching of this format ends after handling the
    first dispatched format."""
    excludes: List[str] = []
    """List of strings that must not appear in the free-form text."""


class TokenDispatchFormat(BaseModel):
    """Matches certain patterns in free-form text.
    When certain tokens are generated, the following content must follow the corresponding format.
    Certain tokens can be excluded from being generated in the free-form part.

    The user specifies a list of (pattern token, formats). The LLM can generate any free-form text, but
    when a pattern is matched in the text, the following output must follow the corresponding format.
    The ``loop`` field controls whether the matching of this structure ends after accepting the first
    pattern and format. If False, it ends. If true, the matching continues, and this format will
    continue to allow free-form text and detect the next pattern to be generated.
    The ``excludes`` field controls the strings that cannot be generated in the free-form text.
    It can also control the end of the format:
    ``SequenceFormat(DispatchFormat(..., excludes=""), ConstStringFormat(""))``
    or
    ``TagFormat(begin=..., content=DispatchFormat(..., excludes=""), end="")``
    ends the matching of the format when LLM generates <end>.
    """

    type: Literal["token_dispatch"] = "token_dispatch"
    """The type of the format."""
    rules: List[Tuple[Union[int, str], "Format"]]
    """List of ``(pattern token, content format)`` pairs. Pattern is token ID or token string."""
    loop: bool = True
    """If true, after one dispatched format, it will continue to allow free-form text
    and match the next pattern. Otherwise, the matching of this format ends after handling the
    first dispatched format."""
    exclude_tokens: List[Union[int, str]] = []
    """List of tokens that must not appear in the free-form text."""


# ---------- Deprecated Formats ----------


class QwenXMLParameterFormat(BaseModel):
    """Deprecated. Use :class:`JSONSchemaFormat` with ``style="qwen_xml"`` instead.

    This format remains available so existing serialized structural tags with
    ``{"type": "qwen_xml_parameter"}`` can still be loaded.

    Examples
    --------
    Use the replacement format for new structural tags:

    .. code-block:: python

        structural_tag = JSONSchemaFormat(
            json_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name", "age"],
            },
            style="qwen_xml",
        )

    The above structural tag can accept the following outputs::

        <parameter=name>Bob</parameter><parameter=age>100</parameter>
        <parameter=name>"Bob&lt;"</parameter><parameter=age>100</parameter>
    """

    type: Literal["qwen_xml_parameter"] = "qwen_xml_parameter"
    """The type of the deprecated format."""

    json_schema: Union[bool, Dict[str, Any]]
    """The JSON schema for the parameters of the function calling."""


# ---------- Discriminated Union ----------


Format = Annotated[
    Union[
        AnyTextFormat,
        ConstStringFormat,
        JSONSchemaFormat,
        GrammarFormat,
        RegexFormat,
        QwenXMLParameterFormat,
        OrFormat,
        SequenceFormat,
        TagFormat,
        TriggeredTagsFormat,
        TokenTriggeredTagsFormat,
        TagsWithSeparatorFormat,
        OptionalFormat,
        PlusFormat,
        StarFormat,
        TokenFormat,
        ExcludeTokenFormat,
        AnyTokensFormat,
        RepeatFormat,
        DispatchFormat,
        TokenDispatchFormat,
    ],
    Field(discriminator="type"),
]
"""Union of all structural tag formats."""


# Solve forward references
if hasattr(BaseModel, "model_rebuild"):
    OrFormat.model_rebuild()
    SequenceFormat.model_rebuild()
    TagFormat.model_rebuild()
    TriggeredTagsFormat.model_rebuild()
    TokenTriggeredTagsFormat.model_rebuild()
    TagsWithSeparatorFormat.model_rebuild()
    OptionalFormat.model_rebuild()
    PlusFormat.model_rebuild()
    StarFormat.model_rebuild()
    RepeatFormat.model_rebuild()
    DispatchFormat.model_rebuild()
    TokenDispatchFormat.model_rebuild()
elif hasattr(BaseModel, "update_forward_refs"):
    OrFormat.update_forward_refs()
    SequenceFormat.update_forward_refs()
    TagFormat.update_forward_refs()
    TriggeredTagsFormat.update_forward_refs()
    TokenTriggeredTagsFormat.update_forward_refs()
    TagsWithSeparatorFormat.update_forward_refs()
    OptionalFormat.update_forward_refs()
    PlusFormat.update_forward_refs()
    StarFormat.update_forward_refs()
    RepeatFormat.update_forward_refs()
    DispatchFormat.update_forward_refs()
    TokenDispatchFormat.update_forward_refs()
else:
    raise RuntimeError("Unsupported pydantic version")


# ---------- Top Level ----------


class StructuralTagItem(BaseModel):
    """Deprecated. Definition of a structural tag item.

    See :meth:`xgrammar.Grammar.from_structural_tag` for more details.
    """

    begin: str
    """The begin tag."""
    schema_: Union[str, Type[BaseModel], Dict[str, Any]] = Field(alias="schema")
    """The schema."""
    end: str
    """The end tag."""


class StructuralTag(BaseModel):
    """
    Describes a complete structural tag structure. It corresponds to
    ``"response_format": {"type": "structural_tag", "format": {...}}`` in API.
    """

    type: Literal["structural_tag"] = "structural_tag"
    """The type must be "structural_tag"."""
    format: Format
    """The format of the structural tag. Could be any of the structural tag formats."""

    @staticmethod
    def from_legacy_structural_tag(
        tags: List[StructuralTagItem], triggers: List[str]
    ) -> "StructuralTag":
        """Convert a legacy structural tag item to a structural tag."""
        return StructuralTag(
            type="structural_tag",
            format=TriggeredTagsFormat(
                type="triggered_tags",
                triggers=triggers,
                tags=[
                    TagFormat(
                        begin=tag.begin,
                        content=JSONSchemaFormat(
                            json_schema=(
                                json.loads(tag.schema_)
                                if isinstance(tag.schema_, str)
                                else (
                                    tag.schema_.model_json_schema()
                                    if isinstance(tag.schema_, type)
                                    and issubclass(tag.schema_, BaseModel)
                                    else tag.schema_
                                )
                            )
                        ),
                        end=tag.end,
                    )
                    for tag in tags
                ],
            ),
        )

    @staticmethod
    def from_json(json_str: Union[str, Dict[str, Any]]) -> "StructuralTag":
        """Convert a JSON string to a structural tag."""
        if isinstance(json_str, str):
            return StructuralTag.model_validate_json(json_str)
        elif isinstance(json_str, dict):
            return StructuralTag.model_validate(json_str)
        else:
            raise ValueError("Invalid JSON string or dictionary")


__all__ = [
    "ConstStringFormat",
    "JSONSchemaFormat",
    "QwenXMLParameterFormat",
    "AnyTextFormat",
    "GrammarFormat",
    "RegexFormat",
    "TokenFormat",
    "ExcludeTokenFormat",
    "AnyTokensFormat",
    "SequenceFormat",
    "OrFormat",
    "TagFormat",
    "TriggeredTagsFormat",
    "TokenTriggeredTagsFormat",
    "TagsWithSeparatorFormat",
    "OptionalFormat",
    "PlusFormat",
    "StarFormat",
    "RepeatFormat",
    "Format",
    "StructuralTagItem",
    "StructuralTag",
]
