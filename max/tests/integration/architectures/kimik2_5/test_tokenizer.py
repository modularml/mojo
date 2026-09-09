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

"""Tests for the Kimi K2.5 tokenizer, specifically tool handling."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from max.pipelines.architectures.kimik2_5.tokenizer import (
    KimiK2_5VLTokenizer,
    _sanitize_kimi_schema_node,
    _sanitize_kimi_tool_schemas,
)
from max.pipelines.context import (
    SamplingParams,
    TextGenerationResponseFormat,
)
from max.pipelines.modeling.types import (
    RequestID,
    TextGenerationRequest,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)


class TestApplyChatTemplateWithTools:
    """Tests for apply_chat_template with tool definitions."""

    @pytest.fixture
    def mock_delegate(self) -> MagicMock:
        """Create a mock HuggingFace tokenizer delegate."""
        delegate = MagicMock()
        delegate.apply_chat_template = MagicMock(
            return_value="templated_output"
        )
        return delegate

    @pytest.fixture
    def tokenizer_with_mock(
        self, mock_delegate: MagicMock
    ) -> KimiK2_5VLTokenizer:
        """Create a KimiK2_5VLTokenizer with a mocked delegate."""
        with patch.object(
            KimiK2_5VLTokenizer, "__init__", lambda self, *args, **kwargs: None
        ):
            tokenizer = KimiK2_5VLTokenizer.__new__(KimiK2_5VLTokenizer)
            tokenizer.delegate = mock_delegate
            return tokenizer

    def test_apply_chat_template_passes_tools_to_delegate(
        self, tokenizer_with_mock: KimiK2_5VLTokenizer, mock_delegate: MagicMock
    ) -> None:
        """Test that tools are passed to the delegate's apply_chat_template."""
        messages = [
            TextGenerationRequestMessage(
                role="user", content="Search for Python"
            )
        ]
        tools = [
            TextGenerationRequestTool(
                type="function",
                function={
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            )
        ]

        result = tokenizer_with_mock.apply_chat_template(messages, tools)

        # Verify the delegate was called with tools
        mock_delegate.apply_chat_template.assert_called_once()
        call_kwargs = mock_delegate.apply_chat_template.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == tools
        assert result == "templated_output"

    def test_apply_chat_template_without_tools(
        self, tokenizer_with_mock: KimiK2_5VLTokenizer, mock_delegate: MagicMock
    ) -> None:
        """Test apply_chat_template works when tools is None."""
        messages = [TextGenerationRequestMessage(role="user", content="Hello")]

        result = tokenizer_with_mock.apply_chat_template(messages, tools=None)

        mock_delegate.apply_chat_template.assert_called_once()
        call_kwargs = mock_delegate.apply_chat_template.call_args.kwargs
        assert call_kwargs["tools"] is None
        assert result == "templated_output"

    def test_apply_chat_template_with_empty_tools(
        self, tokenizer_with_mock: KimiK2_5VLTokenizer, mock_delegate: MagicMock
    ) -> None:
        """Test apply_chat_template with empty tools list."""
        messages = [TextGenerationRequestMessage(role="user", content="Hello")]
        tools: list[TextGenerationRequestTool] = []

        result = tokenizer_with_mock.apply_chat_template(messages, tools)

        mock_delegate.apply_chat_template.assert_called_once()
        call_kwargs = mock_delegate.apply_chat_template.call_args.kwargs
        assert call_kwargs["tools"] == []
        assert result == "templated_output"

    def test_apply_chat_template_with_multiple_tools(
        self, tokenizer_with_mock: KimiK2_5VLTokenizer, mock_delegate: MagicMock
    ) -> None:
        """Test apply_chat_template with multiple tool definitions."""
        messages = [
            TextGenerationRequestMessage(
                role="user", content="Get weather and search"
            )
        ]
        tools = [
            TextGenerationRequestTool(
                type="function",
                function={
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            ),
            TextGenerationRequestTool(
                type="function",
                function={
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            ),
        ]

        result = tokenizer_with_mock.apply_chat_template(messages, tools)

        mock_delegate.apply_chat_template.assert_called_once()
        call_kwargs = mock_delegate.apply_chat_template.call_args.kwargs
        assert call_kwargs["tools"] == tools
        assert len(call_kwargs["tools"]) == 2
        assert result == "templated_output"

    def test_apply_chat_template_sets_add_generation_prompt(
        self, tokenizer_with_mock: KimiK2_5VLTokenizer, mock_delegate: MagicMock
    ) -> None:
        """Test that add_generation_prompt=True is always set."""
        messages = [TextGenerationRequestMessage(role="user", content="Hello")]

        tokenizer_with_mock.apply_chat_template(messages)

        call_kwargs = mock_delegate.apply_chat_template.call_args.kwargs
        assert call_kwargs["add_generation_prompt"] is True

    def test_apply_chat_template_sets_tokenize_false(
        self, tokenizer_with_mock: KimiK2_5VLTokenizer, mock_delegate: MagicMock
    ) -> None:
        """Test that tokenize=False is set (returns string, not token IDs)."""
        messages = [TextGenerationRequestMessage(role="user", content="Hello")]

        tokenizer_with_mock.apply_chat_template(messages)

        call_kwargs = mock_delegate.apply_chat_template.call_args.kwargs
        assert call_kwargs["tokenize"] is False

    def test_apply_chat_template_forwards_chat_template_options(
        self, tokenizer_with_mock: KimiK2_5VLTokenizer, mock_delegate: MagicMock
    ) -> None:
        """Test that chat_template_options are forwarded to the delegate.

        This is a regression test for MXSERV-79: chat_template_kwargs from
        the request must be forwarded to the Jinja template. Without this
        fix, options like ``{"thinking": false}`` were silently dropped.
        """
        messages = [TextGenerationRequestMessage(role="user", content="Hello")]

        tokenizer_with_mock.apply_chat_template(
            messages, tools=None, thinking=False
        )

        call_kwargs = mock_delegate.apply_chat_template.call_args.kwargs
        # The "thinking" option should be forwarded to the delegate
        assert "thinking" in call_kwargs
        assert call_kwargs["thinking"] is False
        # add_generation_prompt should still be set
        assert call_kwargs["add_generation_prompt"] is True

    def test_apply_chat_template_options_do_not_override_add_generation_prompt(
        self, tokenizer_with_mock: KimiK2_5VLTokenizer, mock_delegate: MagicMock
    ) -> None:
        """Test that caller options can override add_generation_prompt if needed."""
        messages = [TextGenerationRequestMessage(role="user", content="Hello")]

        # Caller explicitly sets add_generation_prompt=False via kwargs
        tokenizer_with_mock.apply_chat_template(
            messages, tools=None, add_generation_prompt=False, thinking=True
        )

        call_kwargs = mock_delegate.apply_chat_template.call_args.kwargs
        # Caller's setting should override the default
        assert call_kwargs["add_generation_prompt"] is False
        assert call_kwargs["thinking"] is True

    def test_apply_chat_template_with_no_extra_options(
        self, tokenizer_with_mock: KimiK2_5VLTokenizer, mock_delegate: MagicMock
    ) -> None:
        """Test apply_chat_template with no extra chat_template_options."""
        messages = [TextGenerationRequestMessage(role="user", content="Hello")]

        tokenizer_with_mock.apply_chat_template(messages, tools=None)

        call_kwargs = mock_delegate.apply_chat_template.call_args.kwargs
        # Only default options should be set
        assert call_kwargs["add_generation_prompt"] is True
        assert call_kwargs["tokenize"] is False
        # "thinking" should not be in kwargs if not provided
        assert "thinking" not in call_kwargs


class TestSanitizeKimiSchemaNode:
    """Tests for ``_sanitize_kimi_schema_node``.

    Kimi K2.5's bundled HF tokenizer (``tool_declaration_ts.py``) only
    recognizes ``$ref``, ``anyOf``, ``enum``, ``type``, and empty ``{}``
    in JSON Schema. The sanitizer rewrites the two constructs Kimi
    rejects:

      * ``oneOf`` → ``anyOf``
      * ``{"const": X}`` → ``{"enum": [X]}``

    Without these rewrites Kimi's parser raises, the exception is
    swallowed inside ``tokenization_kimi.py``, and the prompt is
    rendered without the tool declaration — tool calling silently
    fails for that request.
    """

    def test_one_of_rewritten_to_any_of(self) -> None:
        schema = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
        assert _sanitize_kimi_schema_node(schema) == {
            "anyOf": [{"type": "string"}, {"type": "integer"}]
        }

    def test_bare_const_rewritten_to_enum(self) -> None:
        assert _sanitize_kimi_schema_node({"const": "end"}) == {"enum": ["end"]}

    def test_const_with_explicit_enum_keeps_enum(self) -> None:
        # If the user provided an explicit ``enum`` alongside ``const``
        # the enum wins (it's at least as restrictive) and the const is
        # dropped. JSON Schema treats them as equivalent for a singleton.
        assert _sanitize_kimi_schema_node({"const": "a", "enum": ["a"]}) == {
            "enum": ["a"]
        }

    def test_one_of_and_any_of_at_same_level_merged(self) -> None:
        # When both combinators appear at the same level the branches
        # are concatenated into a single ``anyOf`` so neither set is
        # lost. The merge is a strict relaxation of ``oneOf``'s
        # exclusive-OR semantics, which is fine for tool-call grammars
        # that don't enforce branch exclusivity at the model side.
        schema = {
            "anyOf": [{"type": "string"}],
            "oneOf": [{"type": "integer"}],
        }
        result = _sanitize_kimi_schema_node(schema)
        assert result.keys() == {"anyOf"}
        assert {b["type"] for b in result["anyOf"]} == {"string", "integer"}

    def test_nested_one_of_inside_properties(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "value": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
            },
            "required": ["value"],
        }
        assert _sanitize_kimi_schema_node(schema) == {
            "type": "object",
            "properties": {
                "value": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            },
            "required": ["value"],
        }

    def test_constructs_unsupported_by_kimi_tool_schema(self) -> None:
        # The exact schema that triggered the production
        # "Failed to convert tools to TypeScript style" error.
        schema = {
            "description": (
                "Where to insert (default: end). Only for outline items."
            ),
            "oneOf": [
                {"const": "end"},
                {
                    "properties": {"after": {"type": "string"}},
                    "required": ["after"],
                    "type": "object",
                },
                {
                    "properties": {"index": {"minimum": 0, "type": "integer"}},
                    "required": ["index"],
                    "type": "object",
                },
            ],
        }
        result = _sanitize_kimi_schema_node(schema)
        assert result["description"] == (
            "Where to insert (default: end). Only for outline items."
        )
        assert "oneOf" not in result
        # ``const`` branch was rewritten to ``enum``.
        assert result["anyOf"][0] == {"enum": ["end"]}
        # Object branches passed through unchanged.
        assert result["anyOf"][1]["properties"]["after"] == {"type": "string"}
        assert result["anyOf"][2]["properties"]["index"]["minimum"] == 0

    def test_passthrough_when_no_rewrites_needed(self) -> None:
        # Schemas using only Kimi-supported constructs should round-trip
        # structurally identical (a new dict, but equal contents).
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
                "color": {"enum": ["red", "green", "blue"]},
            },
            "required": ["name"],
            "additionalProperties": False,
        }
        assert _sanitize_kimi_schema_node(schema) == schema

    def test_primitives_pass_through_unchanged(self) -> None:
        # The recursion bottoms out on non-dict/non-list values.
        assert _sanitize_kimi_schema_node("string") == "string"
        assert _sanitize_kimi_schema_node(42) == 42
        assert _sanitize_kimi_schema_node(0.5) == 0.5
        assert _sanitize_kimi_schema_node(True) is True
        assert _sanitize_kimi_schema_node(None) is None

    def test_lists_recursed_element_wise(self) -> None:
        # A list value (e.g. ``anyOf`` branches or ``enum`` values) is
        # walked element-by-element. Schema-shaped items get rewritten;
        # literal values pass through.
        assert _sanitize_kimi_schema_node(
            [{"const": "a"}, {"const": "b"}, "literal"]
        ) == [{"enum": ["a"]}, {"enum": ["b"]}, "literal"]

    def test_empty_dict_returns_empty_dict(self) -> None:
        assert _sanitize_kimi_schema_node({}) == {}

    def test_deeply_nested_one_of(self) -> None:
        # ``oneOf`` nested inside ``items`` nested inside ``properties``
        # should still get rewritten at every depth.
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {"const": "x"},
                            {"type": "string"},
                        ],
                    },
                },
            },
        }
        result = _sanitize_kimi_schema_node(schema)
        items = result["properties"]["tags"]["items"]
        assert "oneOf" not in items
        assert items["anyOf"] == [{"enum": ["x"]}, {"type": "string"}]


class TestSanitizeKimiToolSchemas:
    """Tests for the top-level ``_sanitize_kimi_tool_schemas`` wrapper.

    Wraps :class:`TestSanitizeKimiSchemaNode` with the OpenAI tool
    envelope: ``{"type": "function", "function": {"name": ..., "parameters": ...}}``.
    Only ``function.parameters`` should be touched; everything else is
    copied verbatim.
    """

    def test_none_passes_through(self) -> None:
        assert _sanitize_kimi_tool_schemas(None) is None

    def test_empty_list_passes_through(self) -> None:
        assert _sanitize_kimi_tool_schemas([]) == []

    def test_tool_envelope_preserved(self) -> None:
        # ``type``, ``function.name``, ``function.description`` all
        # round-trip; only ``parameters`` is sanitized.
        tool = TextGenerationRequestTool(
            type="function",
            function={
                "name": "lookup",
                "description": "Look up a record",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"oneOf": [{"const": "id"}, {"type": "string"}]},
                    },
                    "required": ["key"],
                },
            },
        )
        result = _sanitize_kimi_tool_schemas([tool])
        assert result is not None
        assert len(result) == 1
        sanitized = result[0]
        assert sanitized["type"] == "function"
        assert sanitized["function"]["name"] == "lookup"
        assert sanitized["function"]["description"] == "Look up a record"
        params = sanitized["function"]["parameters"]
        assert "oneOf" not in params["properties"]["key"]
        assert params["properties"]["key"]["anyOf"] == [
            {"enum": ["id"]},
            {"type": "string"},
        ]

    def test_tool_with_empty_parameters(self) -> None:
        # A tool that defines no real parameters (e.g. ``get_time()``)
        # should round-trip the empty schema cleanly.
        tool = TextGenerationRequestTool(
            type="function",
            function={
                "name": "now",
                "description": "Returns the time",
                "parameters": {},
            },
        )
        result = _sanitize_kimi_tool_schemas([tool])
        assert result is not None
        assert result[0]["function"]["parameters"] == {}


class TestNewContextGrammarState:
    """``new_context`` must carry grammar enforcement state from the request.

    Regression guard: ``KimiK2_5VLTokenizer.new_context`` overrides
    ``new_context`` and previously omitted ``grammar_state``, so the context
    fell back to a default ``GrammarEnforcementState`` with
    ``grammar_enforced=False``. The context's grammar state must instead reflect the
    request's ``response_format`` (matching ``TextTokenizer``).
    """

    @pytest.fixture
    def tokenizer(self, monkeypatch: pytest.MonkeyPatch) -> KimiK2_5VLTokenizer:
        """A KimiK2_5VLTokenizer with ``new_context``'s collaborators mocked.

        Drives the text-only path (``prompt`` set, no images), so vision
        processing is skipped and only the grammar-state wiring is exercised.
        """
        with patch.object(
            KimiK2_5VLTokenizer, "__init__", lambda self, *args, **kwargs: None
        ):
            tok = KimiK2_5VLTokenizer.__new__(KimiK2_5VLTokenizer)
        tok.delegate = MagicMock()
        tok.max_length = 4096
        # A media-pad id absent from the encoded prompt -> no-image path.
        tok.media_pad_token_id = 999_999
        tok.vision_token_ids = [999_999]
        tok.enable_prefix_caching = False
        tok.enable_vision_caching = False
        tok.vision_processor = MagicMock()
        tok.vision_processor.cfg.merge_kernel_size = 2
        # ``monkeypatch.setattr`` avoids mypy's method-assign error.
        monkeypatch.setattr(
            tok, "encode", AsyncMock(return_value=[1, 2, 3, 4, 5])
        )
        monkeypatch.setattr(
            tok, "create_eos_tracker", AsyncMock(return_value=MagicMock())
        )
        return tok

    @staticmethod
    def _request(
        response_format: TextGenerationResponseFormat | None,
    ) -> TextGenerationRequest:
        return TextGenerationRequest(
            request_id=RequestID("test_grammar_state"),
            model_name="kimi-k2.5",
            prompt=[1, 2, 3, 4, 5],
            sampling_params=SamplingParams(max_new_tokens=8),
            response_format=response_format,
        )

    @pytest.mark.asyncio
    async def test_json_schema_response_format_is_enforced(
        self, tokenizer: KimiK2_5VLTokenizer
    ) -> None:
        """A json_schema response_format yields an enforced grammar state."""
        response_format = TextGenerationResponseFormat(
            type="json_schema",
            json_schema={"type": "object"},
            grammar_enforced=True,
            has_json_schema=True,
            requires_structured_output_flag=True,
        )

        context = await tokenizer.new_context(self._request(response_format))

        assert context.json_schema is not None
        # The key regression assertion: state carried from response_format,
        # not the default grammar_enforced=False.
        assert context.grammar_enforced is True
        assert context.grammar_state.has_json_schema is True
        assert context.grammar_state.requires_structured_output_flag is True

    @pytest.mark.asyncio
    async def test_forced_tool_grammar_is_enforced_from_start(
        self, tokenizer: KimiK2_5VLTokenizer
    ) -> None:
        """A forced tool grammar (tool_choice=required) enforces from token 0."""
        response_format = TextGenerationResponseFormat(
            type="grammar",
            grammar='start: "x"',
            grammar_enforced=True,
            tools_forced=True,
        )

        context = await tokenizer.new_context(self._request(response_format))

        assert context.grammar is not None
        assert context.grammar_enforced is True
        assert context.grammar_state.tools_forced is True

    @pytest.mark.asyncio
    async def test_no_response_format_is_unenforced(
        self, tokenizer: KimiK2_5VLTokenizer
    ) -> None:
        """Without a response_format the state defaults to unenforced."""
        context = await tokenizer.new_context(self._request(None))

        assert context.json_schema is None
        assert context.grammar is None
        assert context.grammar_enforced is False
        assert context.grammar_state.has_json_schema is False
