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

"""OpenAI Chat Completions API tool call schema definitions.

Pydantic models aligned with the official openai-python SDK
(generated from OpenAPI spec by Stainless).

This module models the Chat Completions tool-call shape. The Responses API uses
different flat tool and tool_choice shapes; see the class docstrings below.
"""

# Adapted from openai-python, licensed under Apache License 2.0.
# Original project: https://github.com/openai/openai-python
# Modified for XGrammar.

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel

# ============================================================
# tools: list[FunctionToolParam]
# ============================================================


class FunctionDefinition(BaseModel):
    """A JSON-Schema-based function definition.

    Corresponds to ``openai.types.shared_params.FunctionDefinition``.

    In Chat Completions this object is nested under
    ``tools[].function``. In the Responses API, the function tool fields are
    flat on the tool object itself and correspond to
    ``openai.types.responses.FunctionToolParam``.
    """

    name: str
    """The name of the function to be called.

    Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum
    length of 1024.
    """
    description: Optional[str] = None
    """A description of what the function does, used by the model to choose when and
    how to call the function.
    """
    parameters: Optional[Dict[str, Any]] = None
    """The parameters the functions accepts, described as a JSON Schema object.

    If omitted or set to ``None``, the generated function arguments will be unconstrained.
    """
    strict: Optional[bool] = None
    """Whether to enable strict schema adherence when generating the function call.

    If set to true, the model will follow the exact schema defined in the
    ``parameters`` field.
    """


class FunctionToolParam(BaseModel):
    """A function tool that can be used to generate a response.

    Corresponds to ``openai.types.chat.ChatCompletionFunctionToolParam``.

    Chat Completions shape::

        {"type": "function", "function": {"name": "...", "parameters": {...}}}

    Responses API shape is flat and corresponds to
    ``openai.types.responses.FunctionToolParam``::

        {"type": "function", "name": "...", "parameters": {...}}
    """

    type: Literal["function"] = "function"
    """The type of the tool. Currently, only ``function`` is supported."""
    function: FunctionDefinition
    """The function definition."""


class BuiltinToolParam(BaseModel):
    """A builtin tool whose output should be constrained.

    This mirrors hosted/server tool declarations used by APIs such as OpenAI
    Responses or Anthropic Messages. ``type`` is the provider-facing builtin
    tool type. ``name`` is the tool name that appears in the model output; when
    omitted, callers should use ``type`` as the output name. ``parameters`` is
    the JSON schema required by XGrammar and serving engines for constrained
    decoding.
    """

    type: str
    """The provider-facing builtin tool type."""
    name: Optional[str] = None
    """The output tool name for model. XGrammar-specific field.

    Used to constrain the tool name. Use this when the model emits a tool name that
    differs from the provider ``type``. For example, an OpenAI-style
    ``web_search_preview`` builtin may be emitted as ``browser.search`` by a
    Harmony-style model. Defaults to ``type`` when omitted.
    """
    parameters: Optional[Dict[str, Any]] = None
    """Argument schema for the builtin tool. XGrammar-specific field.

    Hosted/server tool APIs often do not require users to provide this schema,
    but XGrammar and serving engines need it to constrain the arguments emitted
    by the model.
    """


ToolParam = Union[FunctionToolParam, BuiltinToolParam]
"""A function or builtin tool accepted by builtin structural tag APIs."""


# ============================================================
# tool_choice: ToolChoiceOptionParam
# ============================================================


class NamedToolChoiceFunction(BaseModel):
    """The nested function reference used by Chat Completions named tool choice.

    Corresponds to
    ``openai.types.chat.chat_completion_named_tool_choice_param.Function``.

    Responses API named function choice is flat and corresponds to
    ``openai.types.responses.ToolChoiceFunctionParam``; it uses
    ``{"type": "function", "name": "..."}`` without this nested object.
    """

    name: str
    """The name of the function to call."""


class NamedToolChoiceParam(BaseModel):
    """Specifies a tool the model should use.

    Use to force the model to call a specific function.

    Corresponds to ``openai.types.chat.ChatCompletionNamedToolChoiceParam``.

    Chat Completions shape::

        {"type": "function", "function": {"name": "..."}}

    Responses API shape is flat and corresponds to
    ``openai.types.responses.ToolChoiceFunctionParam``::

        {"type": "function", "name": "..."}
    """

    type: Literal["function"] = "function"
    """For function calling, the type is always ``function``."""
    function: NamedToolChoiceFunction
    """The function to call."""


class BuiltinToolChoiceParam(BaseModel):
    """Specifies a builtin tool the model should use.

    ``type`` matches the builtin tool type in ``tools``. Matching is based on
    ``type``; ``name`` is accepted for API compatibility but is not used for
    matching.
    """

    type: str
    """The builtin tool type."""
    name: Optional[str] = None
    """Optional model-output tool name. Builtin choices are matched by ``type``."""


class AllowedToolRef(BaseModel):
    """A reference to a function or builtin tool allowed in this turn.

    Corresponds to one item in
    ``openai.types.chat.ChatCompletionAllowedToolsParam.tools``.

    Chat Completions tool refs are nested, for example
    ``{"type": "function", "function": {"name": "get_weather"}}``.
    Responses API refs are flat dictionaries in
    ``openai.types.responses.ToolChoiceAllowedParam.tools``, for example
    ``{"type": "function", "name": "get_weather"}``.
    """

    type: str
    """The allowed tool type."""
    function: Optional[NamedToolChoiceFunction] = None
    """The function reference when ``type`` is ``"function"``."""
    name: Optional[str] = None
    """Optional model-output builtin tool name. Builtin refs are matched by ``type``."""


class AllowedToolsParam(BaseModel):
    """Constrains the tools available to the model to a pre-defined set.

    Corresponds to ``openai.types.chat.ChatCompletionAllowedToolsParam``.
    In Chat Completions this object is nested under
    ``ChatCompletionAllowedToolChoiceParam.allowed_tools``. In the Responses API,
    the equivalent fields are directly on
    ``openai.types.responses.ToolChoiceAllowedParam``.
    """

    mode: Literal["auto", "required"]
    """Constrains the tools available to the model to a pre-defined set.

    ``auto`` allows the model to pick from among the allowed tools and generate a
    message.

    ``required`` requires the model to call one or more of the allowed tools.
    """
    tools: List[AllowedToolRef]
    """A list of tool definitions that the model should be allowed to call.

    For the Chat Completions API, the list of tool definitions might look like:

    .. code-block:: json

        [
          { "type": "function", "function": { "name": "get_weather" } },
          { "type": "function", "function": { "name": "get_time" } }
        ]
    """


class AllowedToolChoiceParam(BaseModel):
    """Constrains the tools available to the model to a pre-defined set.

    Corresponds to ``openai.types.chat.ChatCompletionAllowedToolChoiceParam``.

    Chat Completions shape::

        {"type": "allowed_tools", "allowed_tools": {"mode": "...", "tools": [...]}}

    Responses API shape is flat and corresponds to
    ``openai.types.responses.ToolChoiceAllowedParam``::

        {"type": "allowed_tools", "mode": "...", "tools": [...]}
    """

    type: Literal["allowed_tools"] = "allowed_tools"
    """Allowed tool configuration type. Always ``allowed_tools``."""
    allowed_tools: AllowedToolsParam
    """Constrains the tools available to the model to a pre-defined set."""


ToolChoiceOptionParam = Union[
    Literal["none", "auto", "required"],
    NamedToolChoiceParam,
    AllowedToolChoiceParam,
    BuiltinToolChoiceParam,
]
"""Controls which (if any) tool is called by the model.

``none`` means the model will not call any tool and instead generates a message.

``auto`` means the model can pick between generating a message or calling one or
more tools.

``required`` means the model must call one or more tools.

Specifying a particular tool via
``{"type": "function", "function": {"name": "my_function"}}`` forces the model
to call that tool.

Corresponds to openai.types.chat.ChatCompletionToolChoiceOptionParam.
"""
