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

"""OpenAI-compatible request/response schemas for MAX Serve.

Response models are the official Pydantic types from the ``openai`` SDK,
with small subclasses to expose MAX-only response extensions (currently the
``reasoning`` text emitted by reasoning models). Subclassing rather than
relying on pydantic's ``extra='allow'`` keeps the surface explicit so a
typo in field handling code is a static error rather than a silent extra.

Request models are derived from the SDK's ``TypedDict`` "params" types via
``_model_from_typeddict``, then subclassed to add MAX-only sampling /
routing extensions and to give a few fields stricter pydantic shapes
(messages, tools, response_format, tool_choice). They use ``extra='forbid'``
to match OpenAI's behavior on unknown request fields - misspelled or
unsupported fields surface as 4xx errors instead of being silently dropped.
"""

# ruff: noqa: F401 disable unused-import, we re-export on purpose

from __future__ import annotations

from typing import (
    Any,
    Literal,
    get_type_hints,
)

# isort: off
from openai.types import (
    CompletionUsage,
    CreateEmbeddingResponse,
    Embedding,
    Model,
)
from openai.types.chat import (
    ChatCompletion as _OpenAIChatCompletion,
    ChatCompletionChunk as _OpenAIChatCompletionChunk,
    ChatCompletionMessage as _OpenAIChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall as ChatCompletionMessageToolCall,
    ChatCompletionTokenLogprob,
)
from openai.types.chat.chat_completion import (
    Choice as _OpenAIChatCompletionChoice,
    ChoiceLogprobs as ChatCompletionLogprobs,
)
from openai.types.chat.chat_completion_chunk import (
    Choice as _OpenAIChatCompletionStreamChoice,
    ChoiceDelta as _OpenAIChoiceDelta,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    Function as ChatCompletionMessageToolCallFunction,
)
from openai.types.chat.chat_completion_token_logprob import TopLogprob
from openai.types.chat.completion_create_params import (
    CompletionCreateParamsBase as _OpenAIChatCompletionParams,
)
from openai.types.completion import Completion as CreateCompletionResponse
from openai.types.completion_choice import (
    CompletionChoice as CompletionResponseChoice,
    Logprobs as CompletionLogprobs,
)
from openai.types.completion_create_params import (
    CompletionCreateParamsBase as _OpenAITextCompletionParams,
)
from openai.types.completion_usage import (
    CompletionTokensDetails,
    PromptTokensDetails,
)
from openai.types.audio.speech_create_params import (
    SpeechCreateParams as _OpenAISpeechParams,
)
from openai.types.embedding_create_params import (
    EmbeddingCreateParams as _OpenAIEmbeddingParams,
)
from openai.types.shared_params.response_format_json_object import (
    ResponseFormatJSONObject,
)
from openai.types.shared_params.response_format_text import (
    ResponseFormatText,
)

# isort: on
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    model_validator,
)
from typing_extensions import NotRequired, TypedDict

# ---------------------------------------------------------------------------
# Response models.
#
# We subclass the OpenAI SDK pydantic types only to declare the MAX-specific
# ``reasoning`` field (emitted by reasoning models). OpenAI's official chat
# completion shapes don't have this field today (it lives on the newer
# Responses API); other inference servers expose an analogous
# ``reasoning_content``. Clients that don't know about ``reasoning`` will
# either accept it as an OpenAI ``extra='allow'`` field or drop it, so we
# remain a strict superset of the OpenAI wire format.
# ---------------------------------------------------------------------------


class ChatCompletionResponseMessage(_OpenAIChatCompletionMessage):
    """OpenAI assistant message extended with MAX reasoning text.

    Reasoning-capable models emit their chain-of-thought under one of two
    fields, selected by the ``emit_reasoning_content`` runtime flag:
    ``reasoning`` (the OpenAI Responses API naming, the default) or
    ``reasoning_content`` (the alias used by vLLM, SGLang, and the DeepSeek
    API). Exactly one is populated per response; the other stays ``None``.
    """

    reasoning: str | None = None
    reasoning_content: str | None = None


class ChatCompletionStreamResponseDelta(_OpenAIChoiceDelta):
    """OpenAI stream delta extended with MAX reasoning text.

    Mirrors :class:`ChatCompletionResponseMessage`: each delta carries the
    reasoning fragment under ``reasoning`` or ``reasoning_content`` (selected
    by the ``emit_reasoning_content`` runtime flag), never both.
    """

    reasoning: str | None = None
    reasoning_content: str | None = None


class ChatCompletionResponseChoice(_OpenAIChatCompletionChoice):
    """Non-streaming chat completion choice using the MAX-extended message."""

    message: ChatCompletionResponseMessage


class ChatCompletionStreamResponseChoice(_OpenAIChatCompletionStreamChoice):
    """Streaming chat completion choice using the MAX-extended delta."""

    delta: ChatCompletionStreamResponseDelta


class CreateChatCompletionResponse(_OpenAIChatCompletion):
    """Chat completion response using MAX-extended choices."""

    # ``list`` is invariant in pydantic field overrides; mypy needs the ignore
    # but pydantic accepts the narrowed element type at runtime.
    choices: list[ChatCompletionResponseChoice]  # type: ignore[assignment]


class CreateChatCompletionStreamResponse(_OpenAIChatCompletionChunk):
    """Streaming chat completion response using MAX-extended choices."""

    choices: list[ChatCompletionStreamResponseChoice]  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Other response shapes.
# ---------------------------------------------------------------------------


class MaxModel(Model):
    """OpenAI model card extended with MAX-specific fields."""

    max_model_len: int | None = None


class ListModelsResponse(BaseModel):
    object: Literal["list"]
    data: list[MaxModel]


class Error(BaseModel):
    code: str
    message: str
    param: str
    type: str


class ErrorResponse(BaseModel):
    error: Error


# ---------------------------------------------------------------------------
# Request schemas.
#
# The OpenAI SDK ships request shapes as ``TypedDict`` "params" types that
# can't be used directly as FastAPI request bodies. We project them into
# pydantic models here so we get one source of truth for OpenAI's request
# fields - new SDK fields automatically flow through when we bump the
# pinned ``openai`` version.
#
# All request models are configured with ``extra='forbid'`` so misspelled
# or unsupported parameters surface as a 4xx validation error, matching
# ``api.openai.com``'s behavior. MAX-only sampling/routing fields are
# layered on via subclasses.
# ---------------------------------------------------------------------------

_FORBID_EXTRA = ConfigDict(
    extra="forbid",
    # OpenAI's TypedDict params reference further TypedDicts (e.g. for
    # message content parts and tool definitions); pydantic accepts them
    # as ``arbitrary_types_allowed``.
    arbitrary_types_allowed=True,
)


# TypedDicts for tool-call objects inside chat messages. These match
# OpenAI's spec: ``function.name`` and ``function.arguments`` are both
# Required[str].
class _ToolCallFunction(TypedDict):
    name: str
    arguments: str


class _ToolCallParam(TypedDict):
    function: _ToolCallFunction
    id: NotRequired[str]
    type: NotRequired[str]


# Multi-modal content part; image_url/video_url are dicts to accept non-string
# vendor hints (e.g. max_long_side_pixel int, video fps float).
class _ContentPart(TypedDict):
    type: str
    text: NotRequired[str]
    image_url: NotRequired[dict[str, Any]]
    video_url: NotRequired[dict[str, Any]]


# MAX chat message schema. Vendor extensions like ``reasoning_content``
# are first-class fields so pydantic type-checks them at request
# validation time.
class ChatCompletionMessageParam(TypedDict):
    # ``root`` is a vendor role; parsed for all models but gated at the route to
    # those that declare it (``extra_chat_roles``), others get a 400.
    role: Literal[
        "developer", "system", "user", "assistant", "tool", "function", "root"
    ]
    content: NotRequired[str | list[_ContentPart] | None]
    name: NotRequired[str]
    tool_call_id: NotRequired[str]
    tool_calls: NotRequired[list[_ToolCallParam]]
    function_call: NotRequired[_ToolCallFunction]
    refusal: NotRequired[str | None]
    audio: NotRequired[dict[str, str] | None]

    # MAX vendor extensions.
    reasoning_content: NotRequired[str | None]


ChatCompletionMessageParam.__pydantic_config__ = ConfigDict(extra="allow")  # type: ignore[attr-defined]


# Response-format ``json_schema`` arm, forked from OpenAI's to widen
# ``schema``: a boolean is a valid JSON Schema (``true`` = any value,
# ``false`` = none) and the route de-sugars it to the equivalent object
# schema. Declaring the boolean here (not just coercing in a validator)
# keeps the generated OpenAPI schema accurate about what is accepted. The
# ``text``/``json_object`` arms are reused from the SDK unchanged.
class JSONSchema(TypedDict, total=False):
    name: str
    description: str
    schema: dict[str, object] | bool
    strict: bool | None


class ResponseFormatJSONSchema(TypedDict, total=False):
    type: Literal["json_schema"]
    json_schema: JSONSchema


ResponseFormat = (
    ResponseFormatText | ResponseFormatJSONObject | ResponseFormatJSONSchema
)


def _model_from_typeddict(name: str, td: type) -> type[BaseModel]:
    """Builds a pydantic ``BaseModel`` mirroring an OpenAI ``TypedDict``.

    All fields default to ``None`` because OpenAI marks only a few fields
    (e.g. ``model``, ``messages``, ``input``) as ``Required[...]``; we
    re-declare the truly required ones in the subclass below with no
    default.

    Field annotations are widened to ``Optional[T]`` so that clients which
    explicitly serialize unset fields as ``null`` (e.g. ``"tool_choice":
    null``) are accepted as equivalent to omission. OpenAI expresses this
    via ``NotRequired`` on the TypedDict, but the underlying type aliases
    (``ChatCompletionToolChoiceOptionParam`` and friends) are not
    ``Optional`` themselves.
    """
    fields: dict[str, Any] = {}
    # ``get_type_hints`` (without ``include_extras=True``) already strips
    # Required/NotRequired qualifiers, which is what we want here since the
    # top-level pydantic field is declared with a ``None`` default
    # regardless.
    for field_name, annotation in get_type_hints(td).items():
        fields[field_name] = (annotation | None, None)
    return create_model(name, __config__=_FORBID_EXTRA, **fields)


class ReasoningConfig(BaseModel):
    """OpenRouter's ``reasoning`` object (OpenAI only has ``reasoning_effort``).

    Only ``enabled`` is used (mapped to ``enable_thinking`` in the route);
    the rest are accepted but ignored for now.
    """

    model_config = _FORBID_EXTRA

    enabled: bool | None = None
    effort: str | None = None
    max_tokens: int | None = None
    exclude: bool | None = None


class _MaxRequestExtensions(BaseModel):
    """MAX-specific request fields shared by chat and text completions.

    These are NOT part of the OpenAI spec; OpenAI clients won't send them
    and OpenAI servers won't accept them. Other open-source inference
    servers (vLLM, sglang, ...) ship similar extensions.
    """

    model_config = _FORBID_EXTRA

    # Sampling parameters beyond the OpenAI spec.
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    thinking_temperature: float | None = None

    # MiniMax M3 only: ``False`` folds reasoning into ``content`` wrapped in
    # ``<think>...</think>``; ``True`` (default) keeps it in the ``reasoning`` field.
    reasoning_split: bool = True

    # Generation control.
    min_tokens: int | None = None
    stop_token_ids: list[int] | None = None
    ignore_eos: bool = False

    # Routing / cache hints used by disaggregated serving.
    target_endpoint: str | None = None
    dkv_cache_hint: dict[str, Any] | None = None
    # Per-request prefix-cache isolation for multi-tenant deployments.
    cache_salt: str | None = Field(
        default=None,
        max_length=512,
        description=(
            "Per-request salt that isolates this prompt's prefix-cache "
            "entries from other requests. Combined with "
            "kv_cache_hash_seed via XOR. Works under any "
            "kv_cache_hash_algo: a cryptographic guarantee under "
            "sha256/sha256_64, best-effort under ahash64."
        ),
    )

    # OpenRouter reasoning object; mapped to enable_thinking in the route.
    reasoning: ReasoningConfig | None = None


# ---- Auto-generated request bases from OpenAI's TypedDict params ----------
#
# These pull in every OpenAI request field automatically; bumping the
# ``openai`` SDK version is enough to pick up new ones. Required fields
# from the underlying TypedDicts (``model``, ``messages``, ``input``) are
# re-declared on the subclasses below so they have no default.

_ChatCompletionParamsBase = _model_from_typeddict(
    "_ChatCompletionParamsBase", _OpenAIChatCompletionParams
)
_TextCompletionParamsBase = _model_from_typeddict(
    "_TextCompletionParamsBase", _OpenAITextCompletionParams
)
_EmbeddingParamsBase = _model_from_typeddict(
    "_EmbeddingParamsBase", _OpenAIEmbeddingParams
)
_SpeechParamsBase = _model_from_typeddict(
    "_SpeechParamsBase", _OpenAISpeechParams
)


class CreateChatCompletionRequest(
    _MaxRequestExtensions,
    _ChatCompletionParamsBase,  # type: ignore[misc,valid-type]
):
    """OpenAI chat completion request, extended with MAX fields.

    Inherits every OpenAI request field from
    ``CompletionCreateParamsBase``; adds MAX-only sampling/routing fields
    via ``_MaxRequestExtensions``.
    """

    # Required fields - re-declare so they have no default. Each message
    # is validated against :class:`ChatCompletionMessageParam`, our
    # explicit cross-section of the OpenAI message shapes plus MAX
    # vendor extensions (``reasoning_content``). Pydantic emits plain
    # dicts so the route reads fields via dict access.
    model: str
    messages: list[ChatCompletionMessageParam] = Field(min_length=1)

    max_tokens: int | None = None
    max_completion_tokens: int | None = None

    # Re-typed from the SDK union so the ``json_schema`` arm advertises a
    # boolean ``schema`` (a valid JSON Schema). Pydantic emits a plain dict;
    # the route reads it via dict access.
    response_format: ResponseFormat | None = None

    # ``stream`` lives on the OpenAI streaming/non-streaming subclasses, not
    # on ``CompletionCreateParamsBase`` - declare it explicitly here.
    stream: bool | None = False

    # MAX-only chat-template extension.
    chat_template_kwargs: dict[str, Any] | None = None

    # Pre-tokenized prompt injected by the orchestrator for KV cache-aware
    # routing. When set, the route uses these tokens directly instead of
    # tokenizing ``messages`` (the orchestrator must use the same
    # HuggingFace tokenizer as MAX so the IDs match the model vocabulary).
    # If both are provided, ``prompt_tokens`` takes precedence.
    prompt_tokens: list[int] | None = None

    @model_validator(mode="before")
    @classmethod
    def _translate_thinking_to_standard(cls, data: Any) -> Any:
        # A vendor ``thinking`` control ({"type": enabled|disabled|adaptive})
        # is translated to the standard ``enable_thinking``/``thinking``
        # chat-template booleans. ``adaptive`` leaves them unset: templates
        # default to adaptive when no reasoning flag is given, so the two
        # render identically. Client-set ``chat_template_kwargs`` win.
        if not isinstance(data, dict):
            return data
        thinking = data.get("thinking")
        if thinking is None:
            return data
        if not isinstance(thinking, dict) or set(thinking) - {"type"}:
            raise ValueError("`thinking` must be an object with a `type` field")
        mode = thinking.get("type")
        if mode not in ("enabled", "disabled", "adaptive"):
            raise ValueError(
                "`thinking.type` must be one of 'enabled', 'disabled', "
                f"'adaptive'; got {mode!r}"
            )
        data = dict(data)
        data.pop("thinking")
        if mode != "adaptive":
            enabled = mode == "enabled"
            kwargs = dict(data.get("chat_template_kwargs") or {})
            kwargs.setdefault("enable_thinking", enabled)
            kwargs.setdefault("thinking", enabled)
            data["chat_template_kwargs"] = kwargs
        return data

    @model_validator(mode="after")
    def _reconcile_max_completion_tokens(self) -> CreateChatCompletionRequest:
        # Accept both token-limit fields; ``max_completion_tokens`` wins.
        if (
            self.max_completion_tokens is not None
            and self.max_tokens is not None
            and self.max_tokens != self.max_completion_tokens
        ):
            self.max_tokens = self.max_completion_tokens
        return self

    @property
    def resolved_chat_template_kwargs(self) -> dict[str, Any] | None:
        """The kwargs to render the chat template with.

        ``chat_template_kwargs`` is the only channel that reaches the Jinja
        template, so OpenAI's top-level ``reasoning_effort`` and OpenRouter's
        ``reasoning`` object are folded into it here (OpenRouter sends both).

        The effort is taken from ``chat_template_kwargs`` first, then the
        top-level field, then the ``reasoning`` object; whichever wins also
        decides whether the model thinks at all, unless the client set the
        toggle itself. Templates disagree on the name of that toggle, so both
        ``enable_thinking`` and ``thinking`` are set.

        Returns:
            The chat-template kwargs, or ``None`` when the request carries
            none at all.
        """
        kwargs = dict(self.chat_template_kwargs or {})
        effort = (
            kwargs.get("reasoning_effort")
            or self.reasoning_effort
            or (self.reasoning.effort if self.reasoning is not None else None)
        )
        if self.reasoning is None and effort is None:
            return self.chat_template_kwargs

        if self.reasoning is not None and self.reasoning.enabled is not None:
            enable_thinking = self.reasoning.enabled
        else:
            # ``none`` is OpenAI's "don't reason" effort, and a bare
            # ``reasoning`` object carrying neither field asks for no
            # reasoning either.
            enable_thinking = effort is not None and effort != "none"

        # Client may set either spelling of the thinking toggle, here we merge
        # both.
        for key in ("enable_thinking", "thinking"):
            if key in kwargs:
                enable_thinking = bool(kwargs[key])
                break
        kwargs["enable_thinking"] = enable_thinking
        kwargs["thinking"] = enable_thinking
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        return kwargs


class CreateCompletionRequest(
    _MaxRequestExtensions,
    _TextCompletionParamsBase,  # type: ignore[misc,valid-type]
):
    """OpenAI legacy text completion request, extended with MAX fields."""

    model: str
    prompt: str | list[str] | list[int] | list[list[int]]
    stream: bool | None = False


class CreateEmbeddingRequest(_EmbeddingParamsBase):  # type: ignore[misc,valid-type]
    """OpenAI embedding request."""

    model: str
    input: str | list[str] | list[int] | list[list[int]]


class CreateSpeechRequest(_SpeechParamsBase):  # type: ignore[misc,valid-type]
    """OpenAI speech request, extended for generative audio models.

    ``input`` is the text the audio renders, which for a model that sings is
    its lyrics, and ``instructions`` -- OpenAI's field for describing how the
    audio should sound -- carries the style prompt such a model conditions on.

    ``voice`` is required by OpenAI and has no meaning for a model with no
    voice catalog, so it is optional here and ignored. The MAX extensions
    below are the generation controls an audio model has and a text-to-speech
    model does not; each one left unset keeps the model's own default.
    """

    model: str
    input: str

    voice: str | None = None

    audio_duration: float | None = Field(
        default=None,
        gt=0.0,
        description="Upper bound on the generated audio, in seconds.",
    )
    steps: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Denoising steps, for models whose audio comes from a diffusion "
            "or flow-matching stage."
        ),
    )
    guidance_scale: float | None = Field(
        default=None,
        gt=0.0,
        description="Classifier-free guidance scale.",
    )
    seed: int | None = Field(
        default=None,
        description="Seed for the sampling and noise draws.",
    )


# ---------------------------------------------------------------------------
# MAX-only request/response types not part of the OpenAI spec.
# ---------------------------------------------------------------------------


class LoadLoraRequest(BaseModel):
    """MAX-only request to dynamically load a LoRA adapter."""

    model_config = _FORBID_EXTRA

    lora_name: str
    lora_path: str


class UnloadLoraRequest(BaseModel):
    """MAX-only request to dynamically unload a LoRA adapter."""

    model_config = _FORBID_EXTRA

    lora_name: str
