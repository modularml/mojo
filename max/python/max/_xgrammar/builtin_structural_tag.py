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

from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from pydantic import TypeAdapter

from .openai_tool_call_schema import (
    AllowedToolChoiceParam,
    BuiltinToolChoiceParam,
    BuiltinToolParam,
    FunctionDefinition,
    FunctionToolParam,
    NamedToolChoiceParam,
    ToolChoiceOptionParam,
    ToolParam,
)
from .structural_tag import (
    AnyTextFormat,
    AnyTokensFormat,
    ConstStringFormat,
    Format,
    JSONSchemaFormat,
    OptionalFormat,
    OrFormat,
    RegexFormat,
    SequenceFormat,
    StarFormat,
    StructuralTag,
    TagFormat,
    TagsWithSeparatorFormat,
    TokenFormat,
    TokenTriggeredTagsFormat,
    TriggeredTagsFormat,
)

# ---------- API Functions ----------


def get_model_structural_tag(
    model: str,
    tools: Optional[List[Union[ToolParam, dict]]] = None,
    tool_choice: Union[ToolChoiceOptionParam, dict, None] = "auto",
    reasoning: bool = True,
    force_reasoning: bool = False,
) -> StructuralTag:
    r"""Get a structural tag for a model's reasoning and tool-call output format.

    Use this function when a serving engine needs a structural tag that matches
    a model's tool-call syntax. Pass the model format, the available tools, and
    the desired tool choice policy.

    This API is designed to resemble OpenAI Chat Completions API.

    Function tools use the OpenAI Chat Completions shape:
    ``{"type": "function", "function": {...}}``.

    Builtin tools use a compact shape:

    - ``type`` is the provider-level builtin tool type, such as
      ``"web_search_preview"``.
    - ``name`` is the exact tool name that may appear in model output. If it is
      omitted, ``type`` is used as the output name.
    - ``parameters`` is the JSON schema used to constrain the arguments emitted
      by the model.

    Examples
    --------
    Ordinary function tool:

    .. code-block:: python

        structural_tag = get_model_structural_tag(
            "llama",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    },
                }
            ],
        )

    Harmony with a builtin web search tool:

    .. code-block:: python

        structural_tag = get_model_structural_tag(
            "harmony",
            tools=[
                {
                    "type": "web_search_preview",
                    "name": "browser.search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                }
            ],
        )

    Force an ordinary function tool:

    .. code-block:: python

        structural_tag = get_model_structural_tag(
            "llama",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    },
                }
            ],
            tool_choice={
                "type": "function",
                "function": {"name": "get_weather"},
            },
        )

    Force a builtin tool by type and output name:

    .. code-block:: python

        structural_tag = get_model_structural_tag(
            "harmony",
            tools=[...],
            tool_choice={"type": "web_search_preview"},
        )

    Allow only a subset of tools:

    .. code-block:: python

        structural_tag = get_model_structural_tag(
            "harmony",
            tools=[...],
            tool_choice={
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": "auto",
                    "tools": [
                        {"type": "function", "function": {"name": "get_weather"}},
                        {"type": "web_search_preview"},
                    ],
                },
            },
        )

    Parameters
    ----------
    model : str
        The model type of the structural tag template. It should be one of the registered values.
    tools : Optional[List[Union[ToolParam, dict]]]
        Function and builtin tools available to the model. Function tools use
        the Chat Completions shape. Builtin tools use ``type`` plus optional
        ``name`` and ``parameters`` fields. Defaults to ``None``, which is
        treated as an empty list.
    tool_choice : Union[ToolChoiceOptionParam, dict, None]
        Controls whether the model may or must call tools. Defaults to
        ``"auto"``.

        - ``"auto"`` lets the model choose between text output and tool calls.
        - ``None`` is treated the same as ``"auto"``.
        - ``"none"`` disables all tools.
        - ``"required"`` requires at least one available tool.
        - ``{"type": "function", "function": {"name": ...}}`` forces one
          function tool.
        - ``{"type": <builtin_type>}`` forces one builtin tool. Builtin tool
          choices are matched by ``type``.
        - ``{"type": "allowed_tools", "allowed_tools": ...}`` limits the
          available tools before applying its ``mode``. Its ``tools`` list may
          contain both function refs and builtin refs. Builtin refs are matched
          by ``type``.
    reasoning : bool
        Whether to enable the reasoning part. Some models, such as Qwen 3.6
        and DeepSeek V4, support both reasoning and non-reasoning modes. If
        ``False``, use the non-reasoning mode. For models that do not support
        reasoning, this has no effect. For models that only support reasoning,
        ``False`` means reasoning with empty content.

    force_reasoning : bool
        Deprecated. Control whether to keep the reasoning part but leave its content empty.
        Now we will embed the model's specific behavior into the structural tag function, so
        only controlling ``reasoning`` is enough.

    Notes
    -----
    If a tool's ``parameters`` field is omitted or ``None``, its generated
    arguments are unconstrained JSON. If a function tool has ``strict=False``,
    its ``parameters`` schema is also treated as unconstrained.

    Returns
    -------
    StructuralTag
        A structural tag for function calling format.

    Raises
    ------
    ValueError
        If tool lists, tool choices, or required tool availability are invalid.
    """

    func = _structural_tag_registry.get(model)
    if func is None:
        supported = list(_structural_tag_registry.keys())
        raise ValueError(f"Unknown format type: {model}, supported types: {supported}")

    function_tools, builtin_tools, simplified_tool_choice = normalize_tool_choice(
        tools, tool_choice
    )

    return func(function_tools, builtin_tools, simplified_tool_choice, reasoning)


# ---------- Helper Functions And Constants ----------


SimplifiedToolChoice = Literal["auto", "required", "forced"]
BuiltinStructuralTagFn = Callable[..., StructuralTag]
_TOOL_ADAPTER = TypeAdapter(ToolParam)
_TOOL_CHOICE_ADAPTER = TypeAdapter(ToolChoiceOptionParam)

_structural_tag_registry: Dict[str, BuiltinStructuralTagFn] = {}


def normalize_tool_choice(
    tools: Optional[List[Union[ToolParam, dict]]] = None,
    tool_choice: Union[ToolChoiceOptionParam, dict, None] = "auto",
) -> Tuple[List[FunctionToolParam], List[BuiltinToolParam], SimplifiedToolChoice]:
    r"""Normalize tools and tool choice for structural tag builders.

    This helper exposes the model-independent part of
    :func:`get_model_structural_tag`. It is intended for serving engines that
    want to own their model-specific structural tag templates while reusing
    OpenAI-style tool and tool-choice handling.

    The return value is not a new public tool-calling protocol. It is a compact
    prepared form for structural tag builder functions:

    - ordinary function tools are returned as ``FunctionToolParam`` objects;
    - builtin/server tools are returned as ``BuiltinToolParam`` objects;
    - public tool-choice values are simplified to ``"auto"``, ``"required"``,
      or ``"forced"``.

    Parameters
    ----------
    tools : Optional[List[Union[ToolParam, dict]]]
        Function and builtin tools available to the model. Function tools use
        the OpenAI Chat Completions shape,
        ``{"type": "function", "function": {...}}``. Builtin tools use
        ``type`` plus optional ``name`` and ``parameters`` fields. ``None`` is
        treated as an empty list.
    tool_choice : Union[ToolChoiceOptionParam, dict, None]
        Controls whether the model may or must call tools. This accepts the
        same values as :func:`get_model_structural_tag`:

        - ``"auto"`` keeps all available tools and lets the builder allow text
          or tool calls.
        - ``None`` is treated as ``"auto"``.
        - ``"none"`` clears all tools and returns simplified choice
          ``"auto"``. Builders already interpret auto with no tools as
          text-only.
        - ``"required"`` keeps all available tools and requires at least one
          function or builtin tool to remain available.
        - ``{"type": "function", "function": {"name": ...}}`` filters to the
          named function tool and returns simplified choice ``"forced"``.
        - ``{"type": <builtin_type>}`` filters to exactly one builtin tool
          whose ``type`` matches and returns simplified choice ``"forced"``.
        - ``{"type": "allowed_tools", "allowed_tools": ...}`` filters to the
          referenced tools and returns the nested allowed-tools ``mode`` as the
          simplified choice.

    Returns
    -------
    Tuple[List[FunctionToolParam], List[BuiltinToolParam], SimplifiedToolChoice]
        A tuple of ``(function_tools, builtin_tools, simplified_tool_choice)``
        ready to pass to a model-specific structural tag builder.

    Raises
    ------
    ValueError
        If ``tools`` is not a list, a referenced tool is missing, a builtin
        tool choice does not match exactly one builtin tool, ``required`` leaves
        no available tools, or ``forced`` does not resolve to exactly one tool.

    Examples
    --------
    Build tool-choice handling with an external model-specific builder:

    .. code-block:: python

        function_tools, builtin_tools, tool_choice = normalize_tool_choice(
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    },
                }
            ],
            tool_choice={
                "type": "function",
                "function": {"name": "get_weather"},
            },
        )

        structural_tag = build_my_model_structural_tag(
            function_tools,
            builtin_tools,
            tool_choice,
            reasoning=True,
        )
    """

    if tools is None:
        tools = []
    if not isinstance(tools, list):
        raise ValueError("The 'tools' argument must be a list.")

    normalized_tools = [
        (
            tool
            if isinstance(tool, (FunctionToolParam, BuiltinToolParam))
            else _TOOL_ADAPTER.validate_python(tool)
        )
        for tool in tools
    ]

    # Model-specific functions need separate lists because builtin tools may use
    # a different output channel or recipient from ordinary function tools.
    function_tools = [tool for tool in normalized_tools if isinstance(tool, FunctionToolParam)]
    builtin_tools = [tool for tool in normalized_tools if isinstance(tool, BuiltinToolParam)]

    if tool_choice is None:
        normalized_tool_choice: ToolChoiceOptionParam = "auto"
    else:
        normalized_tool_choice = _TOOL_CHOICE_ADAPTER.validate_python(tool_choice)

    simplified_tool_choice: SimplifiedToolChoice
    if isinstance(normalized_tool_choice, AllowedToolChoiceParam):
        function_tools, builtin_tools = _filter_allowed_tools(
            function_tools, builtin_tools, normalized_tool_choice
        )
        simplified_tool_choice = normalized_tool_choice.allowed_tools.mode
    elif isinstance(normalized_tool_choice, NamedToolChoiceParam):
        tool_name = normalized_tool_choice.function.name
        function_tools = [tool for tool in function_tools if tool.function.name == tool_name]
        if not function_tools:
            raise ValueError(f"The tool with name '{tool_name}' is not found in the tools list.")
        builtin_tools = []
        simplified_tool_choice = "forced"
    elif isinstance(normalized_tool_choice, BuiltinToolChoiceParam):
        function_tools = []
        builtin_tools = [tool for tool in builtin_tools if tool.type == normalized_tool_choice.type]
        if len(builtin_tools) != 1:
            raise ValueError(
                "Builtin tool choice must match exactly one builtin tool, "
                f"got {len(builtin_tools)} matches."
            )
        simplified_tool_choice = "forced"
    elif normalized_tool_choice == "none":
        # The internal functions already treat auto with no tools as text-only.
        function_tools = []
        builtin_tools = []
        simplified_tool_choice = "auto"
    else:
        simplified_tool_choice = normalized_tool_choice

    if simplified_tool_choice == "required" and not function_tools and not builtin_tools:
        raise ValueError(
            "The 'tools' list is empty, which is not allowed when " "'tool_choice' is 'required'."
        )
    if simplified_tool_choice == "forced" and len(function_tools) + len(builtin_tools) != 1:
        raise ValueError("Forced tool choice must resolve to exactly one tool.")

    return function_tools, builtin_tools, simplified_tool_choice


def _get_function_parameters(
    function: Union[FunctionDefinition, BuiltinToolParam]
) -> Union[Dict[str, Any], bool]:
    """Return the JSON schema used for constrained tool arguments.

    ``None`` parameters and non-strict function tools are intentionally mapped
    to ``True`` so the generated arguments remain syntactically constrained but
    schema-unconstrained.
    """

    if isinstance(function, FunctionDefinition) and function.strict is False:
        return True
    if function.parameters is None:
        return True
    return function.parameters


def _get_builtin_tool_name(tool: BuiltinToolParam) -> str:
    """Return the model-output name for a builtin tool."""

    return tool.name or tool.type


def _filter_allowed_tools(
    tools: List[FunctionToolParam],
    builtin_tools: List[BuiltinToolParam],
    tool_choice: AllowedToolChoiceParam,
) -> Tuple[List[FunctionToolParam], List[BuiltinToolParam]]:
    """Filter tools according to a public allowed-tools tool choice."""

    allowed_function_names = set()
    allowed_builtin_types = set()
    for allowed_tool in tool_choice.allowed_tools.tools:
        if allowed_tool.type == "function":
            if allowed_tool.function is None:
                raise ValueError("Allowed function tool references must include 'function'.")
            allowed_function_names.add(allowed_tool.function.name)
        else:
            allowed_builtin_types.add(allowed_tool.type)

    missing_function_names = allowed_function_names - {tool.function.name for tool in tools}
    if missing_function_names:
        raise ValueError(
            f"Allowed function tools are not found in the tools list: {missing_function_names}."
        )

    filtered_builtin_tools = [tool for tool in builtin_tools if tool.type in allowed_builtin_types]
    matched_builtin_types = {tool.type for tool in filtered_builtin_tools}
    missing_builtin_refs = allowed_builtin_types - matched_builtin_types
    if missing_builtin_refs:
        raise ValueError(
            f"Allowed builtin tools are not found in the tools list: {missing_builtin_refs}."
        )

    filtered_tools = [tool for tool in tools if tool.function.name in allowed_function_names]
    return filtered_tools, filtered_builtin_tools


def register_model_structural_tag(name: str):
    """Register a model-specific structural tag function under *name*.

    The decorated function is stored in the internal registry so that
    :func:`get_model_structural_tag` can look it up by the ``model``
    argument. Use this to add support for a new model format.

    Parameters
    ----------
    name : str
        The model format key, e.g. ``"llama"``, ``"harmony"``.

    Examples
    --------
    .. code-block:: python

        @register_model_structural_tag("my_model")
        def get_my_model_structural_tag(
            tools=None, builtin_tools=None, tool_choice="auto",
            reasoning=True, **kwargs,
        ):
            ...
    """

    def decorator(func):
        _structural_tag_registry[name] = func
        return func

    return decorator


# ---------- Each Built-in Structural Tag Function ----------


@register_model_structural_tag("llama")
def get_llama_structural_tag(
    tools: Optional[List[FunctionToolParam]] = None,
    builtin_tools: Optional[List[BuiltinToolParam]] = None,
    tool_choice: Literal["auto", "required", "forced"] = "auto",
    reasoning: bool = True,
    **kwargs: Any,
) -> StructuralTag:
    """Get Llama style structural tag format.

    Corresponding model key: ``"llama"``.

    Reference: https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/

    Parameters are normalized by :func:`get_model_structural_tag` before this
    function is called:

    - ``tools``: a list of function tools. Each tool should have a ``function``
      object containing ``name`` and ``parameters`` fields.
    - ``reasoning``: ignored because this format has no reasoning part.

    Supported models:

    - Meta-Llama-3
    - Llama-3.1
    - Llama-3.2

    Returns
    -------
    StructuralTag
        A structural tag for function calling format.
        This format is used by Llama 3 and other models that follow the same style.
    """
    TOOL_NAME_PREFIX = '{"name": "'
    PARAMETERS_FIELD_PREFIX = '", "parameters": '
    TOOL_OBJECT_BEGIN_PREFIX = '{"name": "'
    TOOL_OBJECT_PARAMETERS_PREFIX = '", "parameters": '
    TOOLS_TRIGGER = '{"name": '
    THINK_EXCLUDE_TOKENS = ["<think>", "</think>"]

    tools = tools or []
    builtin_tools = builtin_tools or []
    if tool_choice == "auto":
        tags = []
        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            name = function.name
            tags.append(
                TagFormat(
                    begin=(TOOL_OBJECT_BEGIN_PREFIX + name + TOOL_OBJECT_PARAMETERS_PREFIX),
                    content=JSONSchemaFormat(json_schema=parameters),
                    end="}",
                )
            )

        if len(tags) > 0:
            suffix_tag = TriggeredTagsFormat(
                triggers=[TOOLS_TRIGGER], tags=tags, excludes=THINK_EXCLUDE_TOKENS
            )
        else:
            suffix_tag = AnyTextFormat(excludes=THINK_EXCLUDE_TOKENS)

    elif tool_choice == "forced":
        if not tools:
            raise ValueError("Forced tool choice must resolve to exactly one tool.")
        function = tools[0].function
        suffix_tag = TagFormat(
            begin=(TOOL_NAME_PREFIX + function.name + PARAMETERS_FIELD_PREFIX),
            content=JSONSchemaFormat(json_schema=_get_function_parameters(function)),
            end="}",
        )

    elif tool_choice == "required":
        tags = []
        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            name = function.name
            tags.append(
                TagFormat(
                    begin=(TOOL_OBJECT_BEGIN_PREFIX + name + TOOL_OBJECT_PARAMETERS_PREFIX),
                    content=JSONSchemaFormat(json_schema=parameters),
                    end="}",
                )
            )
        assert len(tags) > 0
        suffix_tag = TagsWithSeparatorFormat(tags=tags, separator="", at_least_one=True)

    return StructuralTag(format=suffix_tag)


@register_model_structural_tag("kimi")
def get_kimi_structural_tag(
    tools: Optional[List[FunctionToolParam]] = None,
    builtin_tools: Optional[List[BuiltinToolParam]] = None,
    tool_choice: Literal["auto", "required", "forced"] = "auto",
    reasoning: bool = True,
    **kwargs: Any,
) -> StructuralTag:
    """Get Kimi-K2 style structural tag format.

    Corresponding model key: ``"kimi"``.

    Reference: https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/docs/tool_call_guidance.md

    Parameters are normalized by :func:`get_model_structural_tag` before this
    function is called:

    - ``tools``: a list of function tools. Each tool should have a ``function``
      object containing ``name`` and ``parameters`` fields.
    - ``reasoning``: whether to enable reasoning mode. If ``False``, remove
      the reasoning part and constrain only the following part.

    Supported models:

    - Kimi-K2
    - Kimi-K2.5

    Returns
    -------
    StructuralTag
        A structural tag template.
        This format is used by Kimi-K2 and other models that follow the same style.
    """
    TOOL_CALL_BEGIN = "<|tool_call_begin|>"
    TOOL_CALL_BEGIN_PREFIX = f"{TOOL_CALL_BEGIN}functions."
    TOOL_CALL_SUFFIX = ":"
    TOOL_CALL_ARGUMENT_BEGIN = "<|tool_call_argument_begin|>"
    TOOL_CALL_END = "<|tool_call_end|>"
    TOOL_CALLS_SECTION_BEGIN = "<|tool_calls_section_begin|>"
    TOOL_CALLS_SECTION_END = "<|tool_calls_section_end|>"
    THINK_TAG_END = "</think>"
    THINK_EXCLUDE_TOKENS = ["<think>", "</think>"]

    tools = tools or []
    builtin_tools = builtin_tools or []
    if tool_choice == "auto":
        tags = []
        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            name = function.name
            tags.append(
                TagFormat(
                    begin=f"{TOOL_CALL_BEGIN_PREFIX}{name}{TOOL_CALL_SUFFIX}",
                    content=SequenceFormat(
                        elements=[
                            RegexFormat(pattern=r"\d+"),
                            ConstStringFormat(value=TOOL_CALL_ARGUMENT_BEGIN),
                            JSONSchemaFormat(
                                json_schema=parameters,
                                reject_unsupported=True,
                                max_whitespace_cnt=1,
                                strict_mode=False,
                            ),
                        ]
                    ),
                    end=TOOL_CALL_END,
                )
            )

        if len(tags) > 0:
            inner_tool_calls = TagsWithSeparatorFormat(tags=tags, separator="", at_least_one=True)
            tool_calls = TagFormat(
                begin=TOOL_CALLS_SECTION_BEGIN, content=inner_tool_calls, end=TOOL_CALLS_SECTION_END
            )
            suffix_tag = TriggeredTagsFormat(
                triggers=[TOOL_CALLS_SECTION_BEGIN],
                tags=[tool_calls],
                excludes=[*THINK_EXCLUDE_TOKENS, TOOL_CALL_BEGIN],
            )
        else:
            suffix_tag = AnyTextFormat(excludes=THINK_EXCLUDE_TOKENS)

    elif tool_choice == "forced":
        if not tools:
            raise ValueError("Forced tool choice must resolve to exactly one tool.")
        function = tools[0].function
        suffix_tag = SequenceFormat(
            elements=[
                ConstStringFormat(value=TOOL_CALLS_SECTION_BEGIN),
                TagFormat(
                    begin=f"{TOOL_CALL_BEGIN_PREFIX}{function.name}{TOOL_CALL_SUFFIX}",
                    content=SequenceFormat(
                        elements=[
                            RegexFormat(pattern=r"\d+"),
                            ConstStringFormat(value=TOOL_CALL_ARGUMENT_BEGIN),
                            JSONSchemaFormat(
                                json_schema=_get_function_parameters(function),
                                reject_unsupported=True,
                                max_whitespace_cnt=1,
                                strict_mode=False,
                            ),
                        ]
                    ),
                    end=TOOL_CALL_END,
                ),
                ConstStringFormat(value=TOOL_CALLS_SECTION_END),
            ]
        )
    elif tool_choice == "required":
        tags = []
        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            name = function.name
            tags.append(
                TagFormat(
                    begin=f"{TOOL_CALL_BEGIN_PREFIX}{name}{TOOL_CALL_SUFFIX}",
                    content=SequenceFormat(
                        elements=[
                            RegexFormat(pattern=r"\d+"),
                            ConstStringFormat(value=TOOL_CALL_ARGUMENT_BEGIN),
                            JSONSchemaFormat(
                                json_schema=parameters,
                                reject_unsupported=True,
                                max_whitespace_cnt=1,
                                strict_mode=False,
                            ),
                        ]
                    ),
                    end=TOOL_CALL_END,
                )
            )
        assert len(tags) > 0
        suffix_tag = SequenceFormat(
            elements=[
                ConstStringFormat(value=TOOL_CALLS_SECTION_BEGIN),
                TagsWithSeparatorFormat(tags=tags, separator="", at_least_one=True),
                ConstStringFormat(value=TOOL_CALLS_SECTION_END),
            ]
        )

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    # Optional reasoning prefix (free text up to ``</think>``) whose AnyText
    # excludes the tool markers, so the suffix trigger arms even when no
    # ``</think>`` precedes the tool section (else the prefix swallows the
    # marker as reasoning and the tool call decodes unconstrained).
    prefix_tag = TagFormat(
        begin="",
        content=AnyTextFormat(
            excludes=[TOOL_CALLS_SECTION_BEGIN, TOOL_CALL_BEGIN]
        ),
        end=THINK_TAG_END,
    )
    return StructuralTag(
        format=SequenceFormat(
            elements=[OptionalFormat(content=prefix_tag), suffix_tag]
        )
    )


@register_model_structural_tag("deepseek_r1")
def get_deepseek_r1_structural_tag(
    tools: Optional[List[FunctionToolParam]] = None,
    builtin_tools: Optional[List[BuiltinToolParam]] = None,
    tool_choice: Literal["auto", "required", "forced"] = "auto",
    reasoning: bool = True,
    **kwargs: Any,
) -> StructuralTag:
    """Get DeepSeek-R1 style structural tag format.

    Corresponding model key: ``"deepseek_r1"``.

    Reference: https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/tokenizer_config.json

    Supported models:

    - DeepSeek-R1
    - DeepSeek-R1-0528
    """
    TOOL_CALLS_BEGIN = "<｜tool▁calls▁begin｜>"
    TOOL_CALLS_END = "<｜tool▁calls▁end｜>"
    TOOL_CALL_BEGIN = "<｜tool▁call▁begin｜>"
    TOOL_CALL_END = "<｜tool▁call▁end｜>"
    TOOL_SEP = "<｜tool▁sep｜>"
    JSON_RENDER_BEGIN = "\n```json\n"
    JSON_RENDER_END = "\n```"
    THINK_TAG_END = "</think>"
    THINK_EXCLUDE_TOKENS = ["<think>", "</think>"]

    tools = tools or []
    builtin_tools = builtin_tools or []
    if tool_choice == "auto":
        tags = []
        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            name = function.name
            tags.append(
                TagFormat(
                    begin=f"{TOOL_CALL_BEGIN}function{TOOL_SEP}{name}{JSON_RENDER_BEGIN}",
                    content=JSONSchemaFormat(json_schema=parameters),
                    end=f"{JSON_RENDER_END}{TOOL_CALL_END}",
                )
            )

        if len(tags) > 0:
            inner_tool_calls = TagsWithSeparatorFormat(tags=tags, separator="\n", at_least_one=True)
            tool_calls = TagFormat(
                begin=TOOL_CALLS_BEGIN, content=inner_tool_calls, end=TOOL_CALLS_END
            )
            suffix_tag = TriggeredTagsFormat(
                triggers=[TOOL_CALLS_BEGIN], tags=[tool_calls], excludes=THINK_EXCLUDE_TOKENS
            )
        else:
            suffix_tag = AnyTextFormat(excludes=THINK_EXCLUDE_TOKENS)

    elif tool_choice == "forced":
        if not tools:
            raise ValueError("Forced tool choice must resolve to exactly one tool.")
        function = tools[0].function
        parameters = _get_function_parameters(function)
        suffix_tag = TagFormat(
            begin=f"{TOOL_CALLS_BEGIN}{TOOL_CALL_BEGIN}function{TOOL_SEP}{function.name}{JSON_RENDER_BEGIN}",
            content=JSONSchemaFormat(json_schema=parameters),
            end=f"{JSON_RENDER_END}{TOOL_CALL_END}{TOOL_CALLS_END}",
        )

    elif tool_choice == "required":
        tags = []
        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            name = function.name
            tags.append(
                TagFormat(
                    begin=f"{TOOL_CALL_BEGIN}function{TOOL_SEP}{name}{JSON_RENDER_BEGIN}",
                    content=JSONSchemaFormat(json_schema=parameters),
                    end=f"{JSON_RENDER_END}{TOOL_CALL_END}",
                )
            )

        assert len(tags) > 0
        inner_tool_calls = TagsWithSeparatorFormat(tags=tags, separator="\n", at_least_one=True)
        suffix_tag = TagFormat(begin=TOOL_CALLS_BEGIN, content=inner_tool_calls, end=TOOL_CALLS_END)

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    prefix_tag = TagFormat(begin="", content=AnyTextFormat(), end=THINK_TAG_END)

    return StructuralTag(format=SequenceFormat(elements=[prefix_tag, suffix_tag]))


@register_model_structural_tag("deepseek_v3_1")
def get_deepseek_v3_1_structural_tag(
    tools: Optional[List[FunctionToolParam]] = None,
    builtin_tools: Optional[List[BuiltinToolParam]] = None,
    tool_choice: Literal["auto", "required", "forced"] = "auto",
    reasoning: bool = True,
    **kwargs: Any,
) -> StructuralTag:
    """Get DeepSeek-V3.1 style structural tag format.

    Corresponding model key: ``"deepseek_v3_1"``.

    Reference: https://huggingface.co/deepseek-ai/DeepSeek-V3.1/blob/main/tokenizer_config.json

    Supported models:

    - DeepSeek-V3.1
    - DeepSeek-V3.2-Exp
    """
    TOOL_CALLS_BEGIN = "<｜tool▁calls▁begin｜>"
    TOOL_CALLS_END = "<｜tool▁calls▁end｜>"
    TOOL_CALL_BEGIN = "<｜tool▁call▁begin｜>"
    TOOL_CALL_END = "<｜tool▁call▁end｜>"
    TOOL_SEP = "<｜tool▁sep｜>"
    THINK_TAG_END = "</think>"
    THINK_EXCLUDE_TOKENS = ["<think>", "</think>"]

    tools = tools or []
    builtin_tools = builtin_tools or []
    if tool_choice == "auto":
        tags = []
        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            name = function.name
            tags.append(
                TagFormat(
                    begin=f"{TOOL_CALL_BEGIN}{name}{TOOL_SEP}",
                    content=JSONSchemaFormat(json_schema=parameters),
                    end=TOOL_CALL_END,
                )
            )

        if len(tags) > 0:
            inner_tool_calls = TagsWithSeparatorFormat(tags=tags, separator="", at_least_one=True)
            tool_calls = TagFormat(
                begin=TOOL_CALLS_BEGIN, content=inner_tool_calls, end=TOOL_CALLS_END
            )
            suffix_tag = TriggeredTagsFormat(
                triggers=[TOOL_CALLS_BEGIN], tags=[tool_calls], excludes=THINK_EXCLUDE_TOKENS
            )
        else:
            suffix_tag = AnyTextFormat(excludes=THINK_EXCLUDE_TOKENS)

    elif tool_choice == "forced":
        if not tools:
            raise ValueError("Forced tool choice must resolve to exactly one tool.")
        function = tools[0].function
        parameters = _get_function_parameters(function)
        suffix_tag = TagFormat(
            begin=f"{TOOL_CALLS_BEGIN}{TOOL_CALL_BEGIN}{function.name}{TOOL_SEP}",
            content=JSONSchemaFormat(json_schema=parameters),
            end=f"{TOOL_CALL_END}{TOOL_CALLS_END}",
        )

    elif tool_choice == "required":
        tags = []
        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            name = function.name
            tags.append(
                TagFormat(
                    begin=f"{TOOL_CALL_BEGIN}{name}{TOOL_SEP}",
                    content=JSONSchemaFormat(json_schema=parameters),
                    end=TOOL_CALL_END,
                )
            )

        assert len(tags) > 0
        inner_tool_calls = TagsWithSeparatorFormat(tags=tags, separator="", at_least_one=True)
        suffix_tag = TagFormat(begin=TOOL_CALLS_BEGIN, content=inner_tool_calls, end=TOOL_CALLS_END)

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    prefix_tag = TagFormat(begin="", content=AnyTextFormat(), end=THINK_TAG_END)

    return StructuralTag(format=SequenceFormat(elements=[prefix_tag, suffix_tag]))


@register_model_structural_tag("qwen_3_5")
@register_model_structural_tag("qwen_3_coder")
def get_qwen_3_5_structural_tag(
    tools: Optional[List[FunctionToolParam]] = None,
    builtin_tools: Optional[List[BuiltinToolParam]] = None,
    tool_choice: Literal["auto", "required", "forced"] = "auto",
    reasoning: bool = True,
    **kwargs: Any,
) -> StructuralTag:
    """Get Qwen XML tool-call structural tag format.

    Corresponding model keys: ``"qwen_3_5"`` and ``"qwen_3_coder"``.

    Reference: https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8/blob/main/chat_template.jinja

    Parameters are normalized by :func:`get_model_structural_tag` before this
    function is called:

    - ``tools``: a list of function tools. Each tool should have a ``function``
      object containing ``name`` and ``parameters`` fields.
    - ``reasoning``: whether to add the ``</think>`` reasoning prefix before
      the tool/text suffix.

    Supported models:

    - Qwen3.5
    - Qwen3.6
    - Qwen3-Coder
    - Qwen3-Coder-Next

    Returns
    -------
    StructuralTag
        A structural tag for Qwen XML function calling format.
    """
    TOOL_CALL_BEGIN_PREFIX = "<tool_call>\n<function="
    TOOL_CALL_BEGIN_SUFFIX = ">\n"
    TOOL_CALL_END = "\n</function>\n</tool_call>"
    TOOL_CALL_TRIGGER = "<tool_call>\n<function="
    THINK_TAG_END = "</think>"
    THINK_SUFFIX = "\n\n"
    THINK_EXCLUDE_TOKENS = ["<think>", "</think>"]
    tools = tools or []
    builtin_tools = builtin_tools or []
    if tool_choice == "auto":
        tags = []
        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            name = function.name
            tags.append(
                TagFormat(
                    begin=f"{TOOL_CALL_BEGIN_PREFIX}{name}{TOOL_CALL_BEGIN_SUFFIX}",
                    content=JSONSchemaFormat(json_schema=parameters, style="qwen_xml"),
                    end=TOOL_CALL_END,
                )
            )

        if len(tags) > 0:
            suffix_tag = TriggeredTagsFormat(
                triggers=[TOOL_CALL_TRIGGER], tags=tags, excludes=THINK_EXCLUDE_TOKENS
            )
        else:
            suffix_tag = AnyTextFormat(excludes=THINK_EXCLUDE_TOKENS)

    elif tool_choice == "forced":
        if not tools:
            raise ValueError("Forced tool choice must resolve to exactly one tool.")
        function = tools[0].function
        suffix_tag = TagFormat(
            begin=f"{TOOL_CALL_BEGIN_PREFIX}{function.name}{TOOL_CALL_BEGIN_SUFFIX}",
            content=JSONSchemaFormat(
                json_schema=_get_function_parameters(function), style="qwen_xml"
            ),
            end=TOOL_CALL_END,
        )

    elif tool_choice == "required":
        tags = []
        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            name = function.name
            tags.append(
                TagFormat(
                    begin=f"{TOOL_CALL_BEGIN_PREFIX}{name}{TOOL_CALL_BEGIN_SUFFIX}",
                    content=JSONSchemaFormat(json_schema=parameters, style="qwen_xml"),
                    end=TOOL_CALL_END,
                )
            )

        assert len(tags) > 0
        suffix_tag = TagsWithSeparatorFormat(tags=tags, separator="\n", at_least_one=True)

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    prefix_tag = SequenceFormat(
        elements=[
            TagFormat(begin="", content=AnyTextFormat(), end=THINK_TAG_END),
            ConstStringFormat(value=THINK_SUFFIX),
        ]
    )
    return StructuralTag(format=SequenceFormat(elements=[prefix_tag, suffix_tag]))


get_qwen_3_coder_structural_tag = get_qwen_3_5_structural_tag
"""Deprecated alias for :func:`get_qwen_3_5_structural_tag`."""


@register_model_structural_tag("qwen_3")
def get_qwen_3_structural_tag(
    tools: Optional[List[FunctionToolParam]] = None,
    builtin_tools: Optional[List[BuiltinToolParam]] = None,
    tool_choice: Literal["auto", "required", "forced"] = "auto",
    reasoning: bool = True,
    **kwargs: Any,
) -> StructuralTag:
    """Get Qwen3 style structural tag format.

    Corresponding model key: ``"qwen_3"``.

    Reference: https://qwen.readthedocs.io/en/latest/framework/function_call.html

    Parameters are normalized by :func:`get_model_structural_tag` before this
    function is called:

    - ``tools``: a list of function tools. Each tool should have a ``function``
      object containing ``name`` and ``parameters`` fields.
    - ``reasoning``: whether to enable reasoning mode. If ``False``, remove
      the reasoning part.

    Supported models:

    - Qwen3
    - Qwen3-Next

    Returns
    -------
    StructuralTag
        A structural tag template.
        This format is used by Qwen3 and other models that follow the same style.
    """
    TOOL_CALL_BEGIN_PREFIX = '<tool_call>\n{"name": "'
    ARGUMENTS_FIELD_PREFIX = '", "arguments": '
    TOOL_CALL_END = "}\n</tool_call>"
    TOOL_CALL_TRIGGER = "<tool_call>"
    THINK_TAG_END = "</think>"
    THINK_SUFFIX = "\n\n"
    THINK_EXCLUDE_TOKENS = ["<think>", "</think>"]

    tools = tools or []
    builtin_tools = builtin_tools or []
    if tool_choice == "auto":
        tags = []
        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            name = function.name
            tags.append(
                TagFormat(
                    begin=(TOOL_CALL_BEGIN_PREFIX + name + ARGUMENTS_FIELD_PREFIX),
                    content=JSONSchemaFormat(json_schema=parameters),
                    end=TOOL_CALL_END,
                )
            )
        if len(tags) > 0:
            suffix_tag = TriggeredTagsFormat(
                triggers=[TOOL_CALL_TRIGGER], tags=tags, excludes=THINK_EXCLUDE_TOKENS
            )
        else:
            suffix_tag = AnyTextFormat(excludes=THINK_EXCLUDE_TOKENS)

    elif tool_choice == "forced":
        if not tools:
            raise ValueError("Forced tool choice must resolve to exactly one tool.")
        function = tools[0].function
        suffix_tag = TagFormat(
            begin=(TOOL_CALL_BEGIN_PREFIX + function.name + ARGUMENTS_FIELD_PREFIX),
            content=JSONSchemaFormat(json_schema=_get_function_parameters(function)),
            end=TOOL_CALL_END,
        )

    elif tool_choice == "required":
        tags = []
        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            name = function.name
            tags.append(
                TagFormat(
                    begin=(TOOL_CALL_BEGIN_PREFIX + name + ARGUMENTS_FIELD_PREFIX),
                    content=JSONSchemaFormat(json_schema=parameters),
                    end=TOOL_CALL_END,
                )
            )

        assert len(tags) > 0
        suffix_tag = TagsWithSeparatorFormat(tags=tags, separator="\n", at_least_one=True)

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    prefix_tag = SequenceFormat(
        elements=[
            TagFormat(begin="", content=AnyTextFormat(), end=THINK_TAG_END),
            ConstStringFormat(value=THINK_SUFFIX),
        ]
    )
    return StructuralTag(format=SequenceFormat(elements=[prefix_tag, suffix_tag]))


@register_model_structural_tag("harmony")
def get_harmony_structural_tag(
    tools: Optional[List[FunctionToolParam]] = None,
    builtin_tools: Optional[List[BuiltinToolParam]] = None,
    tool_choice: Literal["auto", "required", "forced"] = "auto",
    reasoning: bool = True,
    **kwargs: Any,
) -> StructuralTag:
    """Get harmony(gpt-oss) style structural tag format.

    Corresponding model key: ``"harmony"``.

    Reference: https://developers.openai.com/cookbook/articles/openai-harmony
    Reference: https://huggingface.co/openai/gpt-oss-120b/blob/main/chat_template.jinja

    Parameters are normalized by :func:`get_model_structural_tag` before this
    function is called:

    - ``tools``: a list of function tools. Each tool should have a ``function``
      object containing ``name`` and ``parameters`` fields.
    - ``builtin_tools``: a list of builtin tools. Each builtin tool should
      provide ``type``, optional ``name``, and ``parameters`` fields.
    - ``reasoning``: whether to enable the analysis channel.

    Supported models:

    - gpt-oss

    Returns
    -------
    StructuralTag
        A structural tag template.
        This format is in OpenAI Harmony Response Format, which is used by GPT-oss
        and other models that follow the same style.
    """
    CALL_END = "<|call|>"
    FINAL_BEGIN = "<|channel|>final<|message|>"
    FINAL_END = ["<|end|>", "<|return|>"]
    ANALYSIS_BEGIN = "<|channel|>analysis<|message|>"
    TAG_SEPARATOR = "<|start|>assistant"

    def _function_tool_tags(name, parameters):
        """Generate tags for all supported harmony function tool call formats."""
        content = JSONSchemaFormat(json_schema=parameters)
        return [
            TagFormat(
                begin=f"<|channel|>commentary to=functions.{name}<|constrain|>json<|message|>",
                content=content,
                end=CALL_END,
            ),
            TagFormat(
                begin=f" to=functions.{name}<|channel|>commentary <|constrain|>json<|message|>",
                content=content,
                end=CALL_END,
            ),
            TagFormat(
                begin=f" to=functions.{name}<|channel|>commentary json<|message|>",
                content=content,
                end=CALL_END,
            ),
        ]

    def _builtin_tool_tags(name, parameters):
        """Generate tags for supported harmony builtin tool call formats."""
        content = JSONSchemaFormat(json_schema=parameters)
        return [
            TagFormat(
                begin=f"<|channel|>commentary to={name} code<|message|>",
                content=content,
                end=CALL_END,
            ),
            TagFormat(
                begin=f" to={name}<|channel|>commentary code<|message|>",
                content=content,
                end=CALL_END,
            ),
        ]

    tools = tools or []
    builtin_tools = builtin_tools or []
    tags = []

    if tool_choice == "auto":

        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            tags.extend(_function_tool_tags(function.name, parameters))

        for tool in builtin_tools:
            parameters = _get_function_parameters(tool)
            name = _get_builtin_tool_name(tool)
            tags.extend(_builtin_tool_tags(name, parameters))

        final_tag = TagFormat(begin=FINAL_BEGIN, content=AnyTextFormat(), end=FINAL_END)
        tags.append(final_tag)

    elif tool_choice == "forced":
        if builtin_tools:
            tags.extend(
                _builtin_tool_tags(
                    _get_builtin_tool_name(builtin_tools[0]),
                    _get_function_parameters(builtin_tools[0]),
                )
            )
        elif tools:
            function = tools[0].function
            tags.extend(_function_tool_tags(function.name, _get_function_parameters(function)))
        else:
            raise ValueError("Forced tool choice must resolve to exactly one tool.")

    elif tool_choice == "required":
        for tool in builtin_tools:
            parameters = _get_function_parameters(tool)
            name = _get_builtin_tool_name(tool)
            tags.extend(_builtin_tool_tags(name, parameters))
        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            tags.extend(_function_tool_tags(function.name, parameters))
        assert len(tags) > 0

    if reasoning:
        analysis_tag = TagFormat(begin=ANALYSIS_BEGIN, content=AnyTextFormat(), end=FINAL_END)
        tags.append(analysis_tag)

    tags_with_separator = TagsWithSeparatorFormat(tags=tags, separator=TAG_SEPARATOR)
    return StructuralTag(format=tags_with_separator)


@register_model_structural_tag("deepseek_v3_2")
def get_deepseek_v3_2_structural_tag(
    tools: Optional[List[FunctionToolParam]] = None,
    builtin_tools: Optional[List[BuiltinToolParam]] = None,
    tool_choice: Literal["auto", "required", "forced"] = "auto",
    reasoning: bool = True,
    **kwargs: Any,
) -> StructuralTag:
    """Get DeepSeek-V3.2 style structural tag format.

    Corresponding model key: ``"deepseek_v3_2"``.

    Supported models:

    - DeepSeek-V3.2
    """
    INVOKE_BEGIN_PREFIX = '<｜DSML｜invoke name="'
    INVOKE_BEGIN_SUFFIX = '">\n'
    # INVOKE_END keeps a trailing "\n" so the final invoke is followed by a
    # single "\n" before </｜DSML｜function_calls>, matching the official
    # DeepSeek-V3.2 chat template. The separator between consecutive invokes
    # is intentionally empty: the chat template joins tool calls with a single
    # "\n" and that "\n" is already supplied by INVOKE_END.
    INVOKE_END = "</｜DSML｜invoke>\n"
    INVOKE_SEPARATOR = ""
    TOOL_CALLS_PREFIX = "\n\n"
    FUNCTION_CALLS_BEGIN = "<｜DSML｜function_calls>\n"
    FUNCTION_CALLS_END = "</｜DSML｜function_calls>"
    FUNCTION_CALLS_TRIGGER = "<｜DSML｜function_calls>"
    THINK_TAG_END = "</think>"
    THINK_EXCLUDE_TOKENS = ["<think>", "</think>"]
    XML_STYLE = "deepseek_xml"

    tools = tools or []
    builtin_tools = builtin_tools or []
    if tool_choice == "auto":
        tags = []
        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            name = function.name
            tags.append(
                TagFormat(
                    begin=(INVOKE_BEGIN_PREFIX + name + INVOKE_BEGIN_SUFFIX),
                    content=JSONSchemaFormat(json_schema=parameters, style=XML_STYLE),
                    end=INVOKE_END,
                )
            )

        # generate function calling triggered tag
        if len(tags) > 0:
            function_calling_tags = TagsWithSeparatorFormat(
                tags=tags, separator=INVOKE_SEPARATOR, at_least_one=True
            )

            suffix_tag = TriggeredTagsFormat(
                triggers=[FUNCTION_CALLS_TRIGGER],
                tags=[
                    TagFormat(
                        begin=FUNCTION_CALLS_BEGIN,
                        content=function_calling_tags,
                        end=FUNCTION_CALLS_END,
                    )
                ],
                excludes=THINK_EXCLUDE_TOKENS,
            )
        else:
            suffix_tag = AnyTextFormat(excludes=THINK_EXCLUDE_TOKENS)

    elif tool_choice == "forced":
        if not tools:
            raise ValueError("Forced tool choice must resolve to exactly one tool.")
        function = tools[0].function
        suffix_tag = SequenceFormat(
            elements=[
                ConstStringFormat(value=TOOL_CALLS_PREFIX + FUNCTION_CALLS_BEGIN),
                TagFormat(
                    begin=(INVOKE_BEGIN_PREFIX + function.name + INVOKE_BEGIN_SUFFIX),
                    content=JSONSchemaFormat(
                        json_schema=_get_function_parameters(function), style=XML_STYLE
                    ),
                    end=INVOKE_END,
                ),
                ConstStringFormat(value=FUNCTION_CALLS_END),
            ]
        )
    elif tool_choice == "required":
        tags = []
        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            name = function.name
            tags.append(
                TagFormat(
                    begin=(INVOKE_BEGIN_PREFIX + name + INVOKE_BEGIN_SUFFIX),
                    content=JSONSchemaFormat(json_schema=parameters, style=XML_STYLE),
                    end=INVOKE_END,
                )
            )
        assert len(tags) > 0
        suffix_tag = SequenceFormat(
            elements=[
                ConstStringFormat(value=TOOL_CALLS_PREFIX + FUNCTION_CALLS_BEGIN),
                TagsWithSeparatorFormat(tags=tags, separator=INVOKE_SEPARATOR, at_least_one=True),
                ConstStringFormat(value=FUNCTION_CALLS_END),
            ]
        )

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    prefix_tag = TagFormat(begin="", content=AnyTextFormat(), end=THINK_TAG_END)

    sequence_format = SequenceFormat(elements=[prefix_tag, suffix_tag])
    return StructuralTag(format=sequence_format)


@register_model_structural_tag("minimax")
def get_minimax_structural_tag(
    tools: Optional[List[FunctionToolParam]] = None,
    builtin_tools: Optional[List[BuiltinToolParam]] = None,
    tool_choice: Literal["auto", "required", "forced"] = "auto",
    reasoning: bool = True,
    **kwargs: Any,
) -> StructuralTag:
    """Get MiniMax-M2.5 style structural tag format.

    Corresponding model key: ``"minimax"``.

    Supported models:

    - MiniMax-M2.5
    - MiniMax-M2.7

    Returns
    -------
    StructuralTag
        A structural tag for MiniMax function calling format.
    """
    INVOKE_BEGIN_PREFIX = '<invoke name="'
    INVOKE_BEGIN_SUFFIX = '">\n'
    INVOKE_END = "</invoke>\n"
    TOOL_CALL_BEGIN = "<minimax:tool_call>\n"
    TOOL_CALL_END = "</minimax:tool_call>"
    TOOL_CALL_TRIGGER = "<minimax:tool_call>"
    THINK_TAG_END = "</think>"
    THINK_SUFFIX = "\n\n"
    EMPTY_THINK_CONTENT = "\n</think>\n\n"
    THINK_EXCLUDE_TOKENS = ["<think>", "</think>"]
    XML_STYLE = "minimax_xml"
    JSON_CONFIG: dict[str, Any] = {
        "style": XML_STYLE,
        "reject_unsupported": True,
        "max_whitespace_cnt": 1,
        "strict_mode": False,
        "require_object_root": True,
    }

    tools = tools or []
    builtin_tools = builtin_tools or []
    if tool_choice == "auto":
        tags = []
        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            name = function.name
            tags.append(
                TagFormat(
                    begin=(INVOKE_BEGIN_PREFIX + name + INVOKE_BEGIN_SUFFIX),
                    content=JSONSchemaFormat(json_schema=parameters, **JSON_CONFIG),
                    end=INVOKE_END,
                )
            )

        # generate function calling triggered tag
        if len(tags) > 0:
            function_calling_tags = TagsWithSeparatorFormat(
                tags=tags, separator="", at_least_one=True
            )

            suffix_tag = TriggeredTagsFormat(
                triggers=[TOOL_CALL_TRIGGER],
                tags=[
                    TagFormat(
                        begin=TOOL_CALL_BEGIN, content=function_calling_tags, end=TOOL_CALL_END
                    )
                ],
                excludes=THINK_EXCLUDE_TOKENS,
            )
        else:
            suffix_tag = AnyTextFormat(excludes=THINK_EXCLUDE_TOKENS)

    elif tool_choice == "forced":
        if not tools:
            raise ValueError("Forced tool choice must resolve to exactly one tool.")
        function = tools[0].function
        suffix_tag = SequenceFormat(
            elements=[
                ConstStringFormat(value="\n" + TOOL_CALL_BEGIN),
                TagFormat(
                    begin=(INVOKE_BEGIN_PREFIX + function.name + INVOKE_BEGIN_SUFFIX),
                    content=JSONSchemaFormat(
                        json_schema=_get_function_parameters(function), **JSON_CONFIG
                    ),
                    end=INVOKE_END,
                ),
                ConstStringFormat(value=TOOL_CALL_END),
            ]
        )
    elif tool_choice == "required":
        tags = []
        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            name = function.name
            tags.append(
                TagFormat(
                    begin=(INVOKE_BEGIN_PREFIX + name + INVOKE_BEGIN_SUFFIX),
                    content=JSONSchemaFormat(json_schema=parameters, **JSON_CONFIG),
                    end=INVOKE_END,
                )
            )
        assert len(tags) > 0
        suffix_tag = SequenceFormat(
            elements=[
                ConstStringFormat(value="\n" + TOOL_CALL_BEGIN),
                TagsWithSeparatorFormat(tags=tags, separator="", at_least_one=True),
                ConstStringFormat(value=TOOL_CALL_END),
            ]
        )

    if reasoning:
        think_tag = TagFormat(begin="", content=AnyTextFormat(), end=THINK_TAG_END)
    else:
        think_tag = ConstStringFormat(value=EMPTY_THINK_CONTENT)
    return StructuralTag(
        format=SequenceFormat(
            elements=[think_tag, ConstStringFormat(value=THINK_SUFFIX), suffix_tag]
        )
    )


@register_model_structural_tag("glm_4_7")
def get_glm_4_7_structural_tag(
    tools: Optional[List[FunctionToolParam]] = None,
    builtin_tools: Optional[List[BuiltinToolParam]] = None,
    tool_choice: Literal["auto", "required", "forced"] = "auto",
    reasoning: bool = True,
    **kwargs: Any,
) -> StructuralTag:
    """Get GLM-4.7/GLM-5 style structural tag format.

    The GLM tool calling format uses XML-like tags:
    ``<tool_call>function_name``
    ``<arg_key>key</arg_key><arg_value>value</arg_value>``
    ``</tool_call>``

    Corresponding model key: ``"glm_4_7"``.

    Parameters are normalized by :func:`get_model_structural_tag` before this
    function is called:

    - ``tools``: a list of function tools. Each tool should have a ``function``
      object containing ``name`` and ``parameters`` fields.
    - ``reasoning``: whether to enable reasoning mode. If ``False``, use the
      non-reasoning mode.

    Supported models:

    - GLM-5
    - GLM-4.7

    Returns
    -------
    StructuralTag
        A structural tag for GLM function calling format.
    """
    TOOL_CALL_BEGIN_PREFIX = "<tool_call>"
    TOOL_CALL_END = "</tool_call>"
    TOOL_CALL_TRIGGER = "<tool_call>"
    THINK_TAG_END = "</think>"
    THINK_EXCLUDE_TOKENS = ["<think>", "</think>"]
    XML_STYLE = "glm_xml"
    # Shared JSONSchemaFormat config for every tool's argument schema: GLM emits
    # bare (glm_xml) values and compiles fail-closed (an object root, plus
    # unenforceable JSON Schema keywords rejected rather than silently dropped).
    JSON_CONFIG: dict[str, Any] = {
        "style": XML_STYLE,
        "strict_mode": False,
        "require_object_root": True,
        "reject_unsupported": True,
        "max_whitespace_cnt": 1,
    }

    tools = tools or []
    builtin_tools = builtin_tools or []
    if tool_choice == "auto":
        tags = []
        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            name = function.name
            tags.append(
                TagFormat(
                    begin=f"{TOOL_CALL_BEGIN_PREFIX}{name}",
                    content=JSONSchemaFormat(json_schema=parameters, **JSON_CONFIG),
                    end=TOOL_CALL_END,
                )
            )

        if len(tags) > 0:
            suffix_tag = TriggeredTagsFormat(
                triggers=[TOOL_CALL_TRIGGER], tags=tags, excludes=THINK_EXCLUDE_TOKENS
            )
        else:
            suffix_tag = AnyTextFormat(excludes=THINK_EXCLUDE_TOKENS)

    elif tool_choice == "forced":
        if not tools:
            raise ValueError("Forced tool choice must resolve to exactly one tool.")
        function = tools[0].function
        suffix_tag = TagFormat(
            begin=f"{TOOL_CALL_BEGIN_PREFIX}{function.name}",
            content=JSONSchemaFormat(
                json_schema=_get_function_parameters(function), **JSON_CONFIG
            ),
            end=TOOL_CALL_END,
        )
    elif tool_choice == "required":
        tags = []
        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            name = function.name
            tags.append(
                TagFormat(
                    begin=f"{TOOL_CALL_BEGIN_PREFIX}{name}",
                    content=JSONSchemaFormat(json_schema=parameters, **JSON_CONFIG),
                    end=TOOL_CALL_END,
                )
            )
        assert len(tags) > 0
        suffix_tag = TagsWithSeparatorFormat(tags=tags, separator="", at_least_one=True)

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    prefix_tag = TagFormat(begin="", content=AnyTextFormat(), end=THINK_TAG_END)

    return StructuralTag(format=SequenceFormat(elements=[prefix_tag, suffix_tag]))


@register_model_structural_tag("gemma_4")
def get_gemma_4_structural_tag(
    tools: Optional[List[FunctionToolParam]] = None,
    builtin_tools: Optional[List[BuiltinToolParam]] = None,
    tool_choice: Literal["auto", "required", "forced"] = "auto",
    reasoning: bool = True,
    **kwargs: Any,
) -> StructuralTag:
    """Get Gemma 4 style structural tag format.

    Gemma 4 uses channel markers for reasoning and tool calls instead of
    ``<think>``/``</think>``:

    - Thinking: ``<|channel>thought\\n...thinking...<channel|>``
    - Tool calls: ``<|tool_call>call:func_name{...}<tool_call|>``
    - Turn end: ``<turn|>``

    Corresponding model key: ``"gemma_4"``.

    Reference: https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4

    Parameters are normalized by :func:`get_model_structural_tag` before this
    function is called:

    - ``tools``: a list of function tools. Each tool should have a
      ``function`` object containing ``name`` and ``parameters`` fields.
    - ``reasoning``: whether to enable reasoning mode. If ``False``, the
      reasoning channel is omitted.
    - ``tool_choice``: ``"auto"`` or ``"required"``. ``"required"`` forces at
      least one tool call.

    Supported models:

    - Gemma-4
    - gemma-4-12b-it
    - gemma-4-26b-a4b-it
    - gemma-4-31b-it
    - gemma-4-e2b-it

    Returns
    -------
    StructuralTag
        A structural tag for Gemma 4 function calling format.
    """
    # The tool-call section markers and the string-value delimiter are emitted as
    # single tokens by the Gemma tokenizer. They are referenced by token ID
    # (TokenFormat begin/end, token-triggered dispatch, and the string
    # delimiter/exclude config) so the grammar keeps matching the model's natural
    # tokens even after they are marked special (masked out of byte-level string
    # content). The "call:" literal that follows the begin marker moves into the
    # tag content so the marker stays a standalone token reference.
    TOOL_CALL_BEGIN_MARKER = "<|tool_call>"
    TOOL_CALL_CONTENT_PREFIX = "call:"
    TOOL_CALL_END_MARKER = "<tool_call|>"
    # The reasoning-channel markers are single Gemma tokens too, so -- like the
    # tool-call markers -- they are referenced by token ID (TokenFormat begin/end
    # and id-resolved exclude_tokens) rather than byte literals. That keeps them
    # emittable at their structural positions after they are marked special
    # (masked out of byte-level content). The "thought\n" text that follows the
    # begin marker moves into the tag content so the marker stays a standalone
    # token reference.
    THINK_CHANNEL_BEGIN_MARKER = "<|channel>"
    THINK_CHANNEL_CONTENT_PREFIX = "thought\n"
    THINK_CHANNEL_END_MARKER = "<channel|>"
    EXCLUDE_TOKENS = [THINK_CHANNEL_BEGIN_MARKER, THINK_CHANNEL_END_MARKER]
    JSON_CONFIG: dict[str, Any] = {
        "style": "json",
        "string_value_delimiter_token": '<|"|>',
        "string_value_forbidden_tokens": ["<|tool_call>", "<tool_call|>"],
        "additional_bare_key_terminal": r"[a-zA-Z_][-a-zA-Z0-9_.]*",
        "bare_key_literal_forbidden": r":{},\x00-\x20\x7f",
        "bare_key_pattern_forbidden": r" \t\n\r\f:{},\"\\\x00-\x1f",
        "max_whitespace_cnt": 1,
        "strict_mode": False,
        "require_object_root": True,
        "reject_unsupported": True,
    }

    def _tool_call_tag(name: str, parameters: dict[str, Any]) -> TagFormat:
        return TagFormat(
            begin=TokenFormat(token=TOOL_CALL_BEGIN_MARKER),
            content=SequenceFormat(
                elements=[
                    ConstStringFormat(value=TOOL_CALL_CONTENT_PREFIX + name),
                    JSONSchemaFormat(json_schema=parameters, **JSON_CONFIG),
                ]
            ),
            end=TokenFormat(token=TOOL_CALL_END_MARKER),
        )

    tools = tools or []
    builtin_tools = builtin_tools or []
    if tool_choice == "auto":
        tags = [
            _tool_call_tag(
                tool.function.name, _get_function_parameters(tool.function)
            )
            for tool in tools
        ]

        if len(tags) > 0:
            suffix_tag = TokenTriggeredTagsFormat(
                trigger_tokens=[TOOL_CALL_BEGIN_MARKER],
                tags=tags,
                exclude_tokens=EXCLUDE_TOKENS,
            )
        else:
            suffix_tag = AnyTextFormat(excludes=EXCLUDE_TOKENS)

    elif tool_choice == "forced":
        if not tools:
            raise ValueError(
                "Forced tool choice must resolve to exactly one tool."
            )
        function = tools[0].function
        suffix_tag = _tool_call_tag(
            function.name, _get_function_parameters(function)
        )

    elif tool_choice == "required":
        tags = [
            _tool_call_tag(
                tool.function.name, _get_function_parameters(tool.function)
            )
            for tool in tools
        ]
        assert len(tags) > 0
        suffix_tag = TagsWithSeparatorFormat(
            tags=tags, separator="", at_least_one=True
        )

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    prefix_tag = TagFormat(
        begin=TokenFormat(token=THINK_CHANNEL_BEGIN_MARKER),
        content=SequenceFormat(
            elements=[
                ConstStringFormat(value=THINK_CHANNEL_CONTENT_PREFIX),
                AnyTextFormat(),
            ]
        ),
        end=TokenFormat(token=THINK_CHANNEL_END_MARKER),
    )
    return StructuralTag(
        format=SequenceFormat(elements=[prefix_tag, suffix_tag])
    )


_INKLING_MESSAGE_MODEL_MARKER = "<|message_model|>"
_INKLING_TOOL_CALL_JSON_MARKER = "<|content_invoke_tool_json|>"
_INKLING_END_MESSAGE_MARKER = "<|end_message|>"
_INKLING_THINKING_MARKER = "<|content_thinking|>"
_INKLING_TEXT_MARKER = "<|content_text|>"


def _inkling_message(content_marker: str, content: Format) -> TagFormat:
    """One Inkling message: a content marker, a body, then ``<|end_message|>``.

    The opening ``<|message_model|>`` is excluded, since a turn's first message
    inherits the one the generation prompt ends with.
    """
    return TagFormat(
        begin=TokenFormat(token=content_marker),
        content=content,
        end=TokenFormat(token=_INKLING_END_MESSAGE_MARKER),
    )


def get_inkling_response_format_branch(json_branch: Format) -> Format:
    """Frames a ``response_format`` JSON answer as a well-formed Inkling turn.

    The JSON becomes the body of a ``<|content_text|>`` message, optionally
    preceded by a thinking message so the model can still reason first.
    """
    return SequenceFormat(
        elements=[
            OptionalFormat(
                content=SequenceFormat(
                    elements=[
                        _inkling_message(_INKLING_THINKING_MARKER, AnyTextFormat()),
                        TokenFormat(token=_INKLING_MESSAGE_MODEL_MARKER),
                    ]
                )
            ),
            _inkling_message(_INKLING_TEXT_MARKER, json_branch),
        ]
    )


_INKLING_MAX_SCHEMA_DEPTH = 32

# Keywords whose values are literal instance data, not subschemas: recursing
# would reorder an object the model has to emit verbatim.
_INKLING_INSTANCE_KEYS = frozenset({"const", "default", "enum", "examples"})


def _inkling_sorted_properties(
    node: Any, depth: int = 0
) -> Any:
    """Re-inserts every ``properties`` mapping in sorted key order.

    The model emits argument keys alphabetically while the JSON schema
    converter enforces declaration order; sorting makes the two agree.
    """
    if depth >= _INKLING_MAX_SCHEMA_DEPTH:
        return node
    if isinstance(node, list):
        return [_inkling_sorted_properties(item, depth + 1) for item in node]
    if not isinstance(node, dict):
        return node

    result: Dict[str, Any] = {}
    for key, value in node.items():
        if key in _INKLING_INSTANCE_KEYS:
            result[key] = value
        elif key == "properties" and isinstance(value, dict):
            result[key] = {
                name: _inkling_sorted_properties(value[name], depth + 1)
                for name in sorted(value)
            }
        else:
            result[key] = _inkling_sorted_properties(value, depth + 1)
    return result


@register_model_structural_tag("inkling")
def get_inkling_structural_tag(
    tools: Optional[List[FunctionToolParam]] = None,
    builtin_tools: Optional[List[BuiltinToolParam]] = None,
    tool_choice: Literal["auto", "required", "forced"] = "auto",
    reasoning: bool = True,
    **kwargs: Any,
) -> StructuralTag:
    """Get Inkling style structural tag format.

    Inkling splits an assistant turn into messages, each opened by
    ``<|message_model|>`` and closed by ``<|end_message|>``:

    - Thinking: ``<|message_model|><|content_thinking|>...<|end_message|>``
    - Text: ``<|message_model|><|content_text|>...<|end_message|>``
    - Tool call: ``<|message_model|>NAME<|content_invoke_tool_json|>``
      ``{"name":"NAME","args":{...}}<|end_message|>``

    Corresponding model key: ``"inkling"``.

    The generation prompt ends at a bare ``<|message_model|>``, so the first
    generated call omits it and every later one emits it. The turn ends with
    ``<|content_model_end_sampling|>``, the model's EOS.

    The function name appears twice per call, bare ahead of the marker and
    again inside the payload. Under ``auto`` only the payload name is
    constrained, and that is the one the parser reports, so tool selection
    still holds.

    Under ``required`` and a named tool the grammar also admits the model's
    canonical preamble -- a thinking message, then optionally a text message --
    since ``<|end_message|>`` closes every message type and so leaves no
    reasoning end token to suspend enforcement on. ``reasoning`` is therefore
    inert: the preamble is admitted either way.

    Builtin tools have no Inkling wire format and are ignored.

    Returns
    -------
    StructuralTag
        A structural tag for Inkling function calling format.
    """
    # Every marker is a special token, emittable only where the grammar names
    # it by token id, so boundaries use TokenFormat rather than literals.
    CALL_NAME_PREFIX = '{"name":"'
    ARGS_FIELD_PREFIX = '","args":'
    JSON_CONFIG: dict[str, Any] = {
        "style": "json",
        # Bounded rather than forbidden: forbidding whitespace also drops the
        # repetition an object with more than one undeclared key needs.
        "max_whitespace_cnt": 1,
        # additionalProperties defaults to true in JSON Schema, so strict mode
        # would mask keys the schema meant to allow.
        "strict_mode": False,
        # require_object_root and reject_unsupported stay off: both turn a
        # schema the converter cannot fully express into a rejected request.
    }

    def _call_tag(name: str, parameters: Union[Dict[str, Any], bool]) -> TagFormat:
        return TagFormat(
            begin=TokenFormat(token=_INKLING_TOOL_CALL_JSON_MARKER),
            content=SequenceFormat(
                elements=[
                    ConstStringFormat(value=CALL_NAME_PREFIX + name + ARGS_FIELD_PREFIX),
                    JSONSchemaFormat(
                        json_schema=_inkling_sorted_properties(parameters),
                        **JSON_CONFIG,
                    ),
                    ConstStringFormat(value="}"),
                ]
            ),
            end=TokenFormat(token=_INKLING_END_MESSAGE_MARKER),
        )

    def _call_unit(name: str, parameters: Union[Dict[str, Any], bool]) -> SequenceFormat:
        # Pins the bare name ahead of the marker to the tool's own name, so it
        # cannot disagree with the payload name.
        return SequenceFormat(
            elements=[
                ConstStringFormat(value=name),
                _call_tag(name, parameters),
            ]
        )

    tools = tools or []
    if tool_choice == "auto":
        tags = [
            _call_tag(tool.function.name, _get_function_parameters(tool.function))
            for tool in tools
        ]
        if not tags:
            return StructuralTag(format=AnyTokensFormat())
        # Nothing is excluded from the region between calls: enforcement never
        # disarms, so that region has to pass the markers framing the
        # surrounding messages.
        return StructuralTag(
            format=TokenTriggeredTagsFormat(
                trigger_tokens=[_INKLING_TOOL_CALL_JSON_MARKER],
                tags=tags,
                exclude_tokens=[],
            )
        )

    if not tools:
        raise ValueError(
            f"Tool choice {tool_choice!r} requires at least one function tool."
        )

    units = [
        _call_unit(tool.function.name, _get_function_parameters(tool.function))
        for tool in tools
    ]
    unit = OrFormat(elements=units)

    # A thinking message, then optionally a text message, or a lone text
    # message. Two is the cap: a repeatable preamble would let the model talk
    # until the token budget ran out without ever calling.
    text_message = _inkling_message(_INKLING_TEXT_MARKER, AnyTextFormat())
    preamble = OrFormat(
        elements=[
            SequenceFormat(
                elements=[
                    _inkling_message(_INKLING_THINKING_MARKER, AnyTextFormat()),
                    OptionalFormat(
                        content=SequenceFormat(
                            elements=[
                                TokenFormat(token=_INKLING_MESSAGE_MODEL_MARKER),
                                text_message,
                            ]
                        )
                    ),
                ]
            ),
            text_message,
        ]
    )
    start = OrFormat(
        elements=[
            unit,
            SequenceFormat(
                elements=[
                    preamble,
                    TokenFormat(token=_INKLING_MESSAGE_MODEL_MARKER),
                    unit,
                ]
            ),
        ]
    )

    if tool_choice == "forced":
        return StructuralTag(format=start)

    return StructuralTag(
        format=SequenceFormat(
            elements=[
                start,
                StarFormat(
                    content=SequenceFormat(
                        elements=[TokenFormat(token=_INKLING_MESSAGE_MODEL_MARKER), unit]
                    )
                ),
            ]
        )
    )


@register_model_structural_tag("deepseek_v4")
def get_deepseek_v4_structural_tag(
    tools: Optional[List[FunctionToolParam]] = None,
    builtin_tools: Optional[List[BuiltinToolParam]] = None,
    tool_choice: Literal["auto", "required", "forced"] = "auto",
    reasoning: bool = True,
    **kwargs: Any,
) -> StructuralTag:
    """Get DeepSeek-V4 style structural tag format.

    Corresponding model key: ``"deepseek_v4"``.

    Supported models:

    - DeepSeek-V4
    """
    INVOKE_BEGIN_PREFIX = '<｜DSML｜invoke name="'
    INVOKE_BEGIN_SUFFIX = '">\n'
    # See get_deepseek_v3_2_structural_tag for the rationale on INVOKE_END +
    # INVOKE_SEPARATOR splitting the single "\n" join that the chat template
    # uses between consecutive <｜DSML｜invoke> blocks.
    INVOKE_END = "</｜DSML｜invoke>\n"
    INVOKE_SEPARATOR = ""
    TOOL_CALLS_PREFIX = "\n\n"
    FUNCTION_CALLS_BEGIN = "<｜DSML｜tool_calls>\n"
    FUNCTION_CALLS_END = "</｜DSML｜tool_calls>"
    FUNCTION_CALLS_TRIGGER = "<｜DSML｜tool_calls>"
    THINK_TAG_END = "</think>"
    THINK_EXCLUDE_TOKENS = ["<think>", "</think>"]
    XML_STYLE = "deepseek_xml"

    tools = tools or []
    builtin_tools = builtin_tools or []
    if tool_choice == "auto":
        tags = []
        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            name = function.name
            tags.append(
                TagFormat(
                    begin=(INVOKE_BEGIN_PREFIX + name + INVOKE_BEGIN_SUFFIX),
                    content=JSONSchemaFormat(json_schema=parameters, style=XML_STYLE),
                    end=INVOKE_END,
                )
            )

        # generate function calling triggered tag
        if len(tags) > 0:
            function_calling_tags = TagsWithSeparatorFormat(
                tags=tags, separator=INVOKE_SEPARATOR, at_least_one=True
            )

            suffix_tag = TriggeredTagsFormat(
                triggers=[FUNCTION_CALLS_TRIGGER],
                tags=[
                    TagFormat(
                        begin=FUNCTION_CALLS_BEGIN,
                        content=function_calling_tags,
                        end=FUNCTION_CALLS_END,
                    )
                ],
                excludes=THINK_EXCLUDE_TOKENS,
            )
        else:
            suffix_tag = AnyTextFormat(excludes=THINK_EXCLUDE_TOKENS)

    elif tool_choice == "forced":
        if not tools:
            raise ValueError("Forced tool choice must resolve to exactly one tool.")
        function = tools[0].function
        suffix_tag = SequenceFormat(
            elements=[
                ConstStringFormat(value=TOOL_CALLS_PREFIX + FUNCTION_CALLS_BEGIN),
                TagFormat(
                    begin=(INVOKE_BEGIN_PREFIX + function.name + INVOKE_BEGIN_SUFFIX),
                    content=JSONSchemaFormat(
                        json_schema=_get_function_parameters(function), style=XML_STYLE
                    ),
                    end=INVOKE_END,
                ),
                ConstStringFormat(value=FUNCTION_CALLS_END),
            ]
        )
    elif tool_choice == "required":
        tags = []
        for tool in tools:
            function = tool.function
            parameters = _get_function_parameters(function)
            name = function.name
            tags.append(
                TagFormat(
                    begin=(INVOKE_BEGIN_PREFIX + name + INVOKE_BEGIN_SUFFIX),
                    content=JSONSchemaFormat(json_schema=parameters, style=XML_STYLE),
                    end=INVOKE_END,
                )
            )
        assert len(tags) > 0
        suffix_tag = SequenceFormat(
            elements=[
                ConstStringFormat(value=TOOL_CALLS_PREFIX + FUNCTION_CALLS_BEGIN),
                TagsWithSeparatorFormat(tags=tags, separator=INVOKE_SEPARATOR, at_least_one=True),
                ConstStringFormat(value=FUNCTION_CALLS_END),
            ]
        )

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    prefix_tag = TagFormat(begin="", content=AnyTextFormat(), end=THINK_TAG_END)

    sequence_format = SequenceFormat(elements=[prefix_tag, suffix_tag])
    return StructuralTag(format=sequence_format)


@register_model_structural_tag("minimax_m3")
def get_minimax_m3_structural_tag(
    tools: Optional[List[FunctionToolParam]] = None,
    builtin_tools: Optional[List[BuiltinToolParam]] = None,
    tool_choice: Literal["auto", "required", "forced"] = "auto",
    reasoning: bool = True,
    **kwargs: Any,
) -> StructuralTag:
    """Get MiniMax-M3 style structural tag format.

    Corresponding model key: ``"minimax_m3"``.

    The MiniMax-M3 tool calling format uses tag-prefixed XML:

    ``]<]minimax[>[<tool_call>``
    ``]<]minimax[>[<invoke name="get_weather">]<]minimax[>[<location>Beijing]<]minimax[>[</location>]<]minimax[>[</invoke>``
    ``]<]minimax[>[</tool_call>``

    Supported models:

    - MiniMax-M3
    """
    SECTION_SEPARATOR = "\n"
    TAG_PREFIX = "]<]minimax[>["
    INVOKE_BEGIN_PREFIX = TAG_PREFIX + '<invoke name="'
    INVOKE_BEGIN_SUFFIX = '">'
    INVOKE_END = TAG_PREFIX + "</invoke>"
    INVOKE_SEPARATOR = "\n"
    TOOL_CALL_BEGIN = "<tool_call>"
    TOOL_CALL_END = "</tool_call>"
    THINK_BEGIN = "<mm:think>"
    THINK_END = "</mm:think>"

    tools = tools or []
    builtin_tools = builtin_tools or []

    def _invoke_tag(name: str, parameters: Any) -> TagFormat:
        return TagFormat(
            begin=INVOKE_BEGIN_PREFIX + name + INVOKE_BEGIN_SUFFIX,
            content=JSONSchemaFormat(
                json_schema=parameters,
                style="minimax_m3_xml",
                xml_tag_prefix=TAG_PREFIX,
                reject_unsupported=True,
                max_whitespace_cnt=1,
                strict_mode=False,
                require_object_root=True,
            ),
            end=INVOKE_END,
        )

    def _forced_or_required_section(invokes: Format) -> SequenceFormat:
        return SequenceFormat(
            elements=[
                ConstStringFormat(value=TAG_PREFIX),
                TagFormat(
                    begin=TokenFormat(token=TOOL_CALL_BEGIN),
                    content=SequenceFormat(
                        elements=[
                            ConstStringFormat(value=SECTION_SEPARATOR),
                            invokes,
                            ConstStringFormat(value=SECTION_SEPARATOR + TAG_PREFIX),
                        ]
                    ),
                    end=TokenFormat(token=TOOL_CALL_END),
                ),
            ]
        )

    if tool_choice == "auto":
        tags = [
            _invoke_tag(t.function.name, _get_function_parameters(t.function))
            for t in tools
        ]
        if tags:
            invokes = TagsWithSeparatorFormat(
                tags=tags, separator=INVOKE_SEPARATOR, at_least_one=True
            )
            suffix_tag = TriggeredTagsFormat(
                triggers=[TAG_PREFIX],
                tags=[
                    TagFormat(
                        begin=TAG_PREFIX,
                        content=SequenceFormat(
                            elements=[
                                TokenFormat(token=TOOL_CALL_BEGIN),
                                ConstStringFormat(value=SECTION_SEPARATOR),
                                invokes,
                                ConstStringFormat(value=SECTION_SEPARATOR + TAG_PREFIX),
                            ]
                        ),
                        end=TokenFormat(token=TOOL_CALL_END),
                    )
                ],
                excludes=[THINK_BEGIN, THINK_END],
            )
        else:
            suffix_tag = AnyTextFormat(excludes=[THINK_BEGIN, THINK_END])

    elif tool_choice == "forced":
        if not tools:
            raise ValueError("Forced tool choice must resolve to exactly one tool.")
        function = tools[0].function
        suffix_tag = _forced_or_required_section(
            _invoke_tag(function.name, _get_function_parameters(function))
        )

    elif tool_choice == "required":
        tags = [
            _invoke_tag(t.function.name, _get_function_parameters(t.function))
            for t in tools
        ]
        assert len(tags) > 0
        suffix_tag = _forced_or_required_section(
            TagsWithSeparatorFormat(
                tags=tags, separator=INVOKE_SEPARATOR, at_least_one=True
            )
        )

    if not reasoning:
        return StructuralTag(format=suffix_tag)

    prefix_tag = TagFormat(
        begin=THINK_BEGIN,
        content=AnyTextFormat(excludes=[TAG_PREFIX + TOOL_CALL_BEGIN]),
        end=THINK_END,
    )
    return StructuralTag(
        format=SequenceFormat(
            elements=[OptionalFormat(content=prefix_tag), suffix_tag]
        )
    )


# Backward-compatible alias
get_builtin_structural_tag = get_model_structural_tag
"""Alias for :func:`get_model_structural_tag`. Deprecated."""
