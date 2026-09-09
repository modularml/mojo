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

"""Defines request, context, and output types for text generation pipelines, including multimodal and VLM support."""

from __future__ import annotations

__all__ = [
    "BatchType",
    "CompletedBatchStats",
    "ImageContentPart",
    "MessageContent",
    "TextContentPart",
    "TextGenerationInputs",
    "TextGenerationRequest",
    "TextGenerationRequestFunction",
    "TextGenerationRequestMessage",
    "TextGenerationRequestTool",
    "VideoContentPart",
]

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypedDict,
)

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

from max.pipelines.context import (
    SamplingParams,
    TextGenerationContextType,
    TextGenerationResponseFormat,
)
from max.pipelines.modeling.types.pipeline import PipelineInputs
from max.pipelines.request import RequestID
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TextGenerationRequestFunction(TypedDict):
    """Represents a function definition for a text generation request."""

    name: str
    """The name of the function to be invoked."""

    description: str | None
    """A human-readable description of the function's purpose."""

    parameters: dict[str, Any]
    """A dictionary describing the function's parameters, typically following a JSON schema."""


class TextGenerationRequestTool(TypedDict):
    """Represents a tool definition for a text generation request."""

    type: str
    """The type of the tool, typically indicating the tool's category or usage."""

    function: TextGenerationRequestFunction
    """The function definition associated with the tool, including its name, description, and parameters."""


class _ContentPart(BaseModel):
    """A single part of a multi-modal message content."""

    type: Literal["text", "image"]


class _MessageContentPart(BaseModel):
    """A typed, immutable part of a message's content."""

    type: str = Field(..., description="Content type identifier")
    model_config = ConfigDict(frozen=True)


class TextContentPart(_MessageContentPart):
    """A plain-text content part of a message."""

    type: Literal["text"] = Field(
        default="text", description="Content type identifier"
    )
    text: str = Field(..., description="Text text content")


class ImageContentPart(_MessageContentPart):
    """An image content part of a message."""

    type: Literal["image"] = Field(
        default="image", description="Content type identifier"
    )

    # Optional vendor sizing hints; ``None`` means unset and models may ignore
    # them. ``detail`` is the OpenAI quality tier; models that honor it map the
    # tier to a resolution.
    detail: str | None = Field(
        default=None,
        description="Detail/quality tier hint for image preprocessing",
    )
    max_long_side_pixel: int | None = Field(
        default=None,
        description="Max long-side length in pixels for image preprocessing",
    )


class VideoContentPart(_MessageContentPart):
    """A video content part of a message."""

    type: Literal["video"] = Field(
        default="video", description="Content type identifier"
    )

    # Optional vendor sampling/sizing hints; ``None`` means unset and models
    # may ignore them.
    fps: float | None = Field(
        default=None,
        description="Frames-per-second to sample the video at",
    )
    max_frames: int | None = Field(
        default=None,
        description="Maximum number of frames to sample from the video",
    )
    detail: str | None = Field(
        default=None,
        description="Detail/quality tier hint for video preprocessing",
    )
    max_long_side_pixel: int | None = Field(
        default=None,
        description="Max long-side length in pixels for video preprocessing",
    )


MessageContent = TextContentPart | ImageContentPart | VideoContentPart

# ``root`` is a vendor role; supporting chat templates order it above ``system``.
_MessageRole = Literal[
    "system", "user", "assistant", "tool", "function", "root"
]


class TextGenerationRequestMessage(BaseModel):
    """A single message in a text generation request conversation."""

    # Follows the openai spec

    role: _MessageRole = Field(
        ..., description="Text role of the message sender"
    )

    content: str | list[MessageContent] = Field(
        default="",
        description=(
            "Message text/multimodal content. Defaults to the empty string for "
            "assistant messages that only carry ``tool_calls``."
        ),
    )

    # Tool call fields from:
    # https://github.com/openai/openai-python/blob/38d75d74a5626472cd7d1be9705ea8aba29a6b22/src/openai/types/chat/chat_completion_message_function_tool_call_param.py
    # The KimiK2.5 chat template uses these fields:
    # https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/chat_template.jinja
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Tool calls emitted by an assistant turn in the conversation "
            "history. Each entry follows the OpenAI shape "
            "``{id, type, function: {name, arguments}}``. Passed through "
            "verbatim to the chat template so multi-turn tool-use prompts "
            "render correctly."
        ),
    )

    tool_call_id: str | None = Field(
        default=None,
        description=(
            "Identifier of the assistant ``tool_calls`` entry that this tool "
            "message is responding to. Required by chat templates that emit a "
            "tool-response header referencing the originating call."
        ),
    )

    reasoning_content: str | None = Field(
        default=None,
        description=(
            "Reasoning/thinking content produced alongside an assistant turn."
        ),
    )

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
    )

    @field_validator("content", mode="before")
    @classmethod
    def validate_content_format(cls, v: Any) -> str | list[MessageContent]:
        """Normalizes message content to a string or list of content parts."""
        if v is None:
            # OpenAI permits ``content=None`` for assistant messages whose
            # payload is carried entirely by ``tool_calls``; collapse to the
            # empty string so the chat template still has something to render.
            return ""
        if isinstance(v, str):
            return v

        if not isinstance(v, list):
            raise ValueError(
                f"Invalid content format: {type(v).__name__}. "
                "Expected str or list of content parts."
            )

        normalized: list[MessageContent] = []
        for item in v:
            if isinstance(
                item, (TextContentPart, ImageContentPart, VideoContentPart)
            ):
                normalized.append(item)
                continue

            if not isinstance(item, dict):
                raise ValueError(
                    f"Invalid content part type: {type(item).__name__}. "
                    "Expected dict or _MessageContentPart instance."
                )

            if "type" not in item:
                raise ValueError(
                    f"Malformed message content part: missing 'type' field. Got: {item}"
                )

            content_type = item["type"]

            if content_type == "text":
                text_value = item.get("text") or item.get("content", "")
                normalized.append(TextContentPart(text=text_value))
            elif content_type == "image":
                normalized.append(ImageContentPart())
            elif content_type == "video":
                normalized.append(VideoContentPart())
            elif content_type == "image_url":
                raise ValueError(
                    "image_url content type not supported in internal format. "
                    "Images must be provided as bytes in TextGenerationRequest.images "
                    "with image placeholders (type='image') in message content."
                )
            elif content_type == "video_url":
                raise ValueError(
                    "video_url content type not supported in internal format. "
                    "Videos must be provided as bytes in TextGenerationRequest.videos "
                    "with video placeholders (type='video') in message content."
                )
            else:
                raise ValueError(
                    f"Unsupported message content type: '{content_type}'"
                )

        return normalized

    def flatten_content(self) -> dict[str, Any]:
        """Flattens message content to a role/content dict for text-only messages.

        Preserves OpenAI-style tool-calling metadata (``tool_calls``,
        ``tool_call_id``, ``reasoning_content``) when set so chat templates
        that consume conversation history with tool use receive a faithful
        representation of each turn.
        """
        if isinstance(self.content, str):
            content_str = self.content
        else:
            parts: list[str] = []
            for content in self.content:
                if isinstance(content, TextContentPart):
                    parts.append(content.text)
                else:
                    raise ValueError("only text content can be flattened.")
            content_str = "\n".join(parts)

        flattened: dict[str, Any] = {
            "role": str(self.role),
            "content": content_str,
        }
        if self.tool_calls is not None:
            flattened["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            flattened["tool_call_id"] = self.tool_call_id
        if self.reasoning_content is not None:
            flattened["reasoning_content"] = self.reasoning_content
        return flattened

    @cached_property
    def number_of_images(self) -> int:
        """Returns the number of ImageContentPart instances in the message content."""
        if isinstance(self.content, str):
            return 0
        return sum(
            1 for item in self.content if isinstance(item, ImageContentPart)
        )

    @cached_property
    def number_of_videos(self) -> int:
        """Returns the number of VideoContentPart instances in the message content."""
        if isinstance(self.content, str):
            return 0
        return sum(
            1 for item in self.content if isinstance(item, VideoContentPart)
        )


@dataclass(frozen=True)
class TextGenerationRequest:
    """An immutable request for text token generation from a pipeline."""

    request_id: RequestID = field()
    """A unique identifier for the request."""

    model_name: str = field()
    """
    The name of the model to be used for generating tokens. This should match
    the available models on the server and determines the behavior and
    capabilities of the response generation.
    """
    prompt: str | Sequence[int] | None = None
    """
    The prompt to be processed by the model. This field supports legacy
    completion APIs and can accept either a string or a sequence of integers
    representing token IDs. If not provided, the model may generate output
    based on the messages field.
    """
    messages: list[TextGenerationRequestMessage] = field(default_factory=list)
    """
    A list of messages for chat-based interactions. This is used in chat
    completion APIs, where each message represents a turn in the conversation.
    If provided, the model will generate responses based on these messages.
    """
    images: list[bytes] = field(default_factory=list)
    """
    A list of image byte arrays that can be included as part of the request.
    This field is optional and may be used for multimodal inputs where images
    are relevant to the prompt or task.
    """
    videos: list[bytes] = field(default_factory=list)
    """
    A list of video byte arrays that can be included as part of the request.
    Each video is decoded into frames during preprocessing.
    """
    decoded_images: list[PILImage] = field(default_factory=list)
    """
    Decoded ``PIL.Image`` objects corresponding 1:1 to :attr:`images`, decoded
    once at request admission (the API server validates images by fully
    decoding them, so the decoded result is carried here to avoid a second
    decode in the tokenizer). API-process-only: this is never serialized across
    the worker boundary, so it must stay populated only for the in-process
    tokenization step. Empty when images were not pre-decoded (offline/test
    callers); tokenizers fall back to decoding :attr:`images` in that case.
    """
    tools: list[TextGenerationRequestTool] | None = None
    """
    A list of tools that can be invoked during the generation process. This
    allows the model to utilize external functionalities or APIs to enhance its
    responses.
    """
    response_format: TextGenerationResponseFormat | None = None
    """
    Specifies the desired format for the model's output. When set, it enables
    structured generation, which adheres to the json_schema provided.
    """
    timestamp_ns: int = 0
    """
    The time (in nanoseconds) when the request was received by the server. This
    can be useful for performance monitoring and logging purposes.
    """
    request_path: str = "/"
    """
    The endpoint path for the request. This is typically used for routing and
    logging requests within the server infrastructure.
    """
    logprobs: int = 0
    """
    The number of top log probabilities to return for each generated token. A value
    of 0 means that log probabilities will not be returned. Useful for analyzing
    model confidence in its predictions.
    """
    echo: bool = False
    """
    If set to ``True``, the response will include the original prompt along with
    the generated output. This can be useful for debugging or when you want to
    see how the input relates to the output.
    """
    chat_template_options: dict[str, Any] | None = None
    """
    Optional dictionary of options to pass when applying the chat template.
    """

    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    """Token sampling configuration parameters for the request."""

    target_endpoint: str | None = None
    """
    Optional target endpoint identifier for routing the request to a specific
    service or model instance. This should be used in disaggregate serving
    scenarios, when you want to dynamically route to a specific instance.
    If not specified, the request will be routed to the default endpoint.
    """

    dkv_cache_hint: dict[str, Any] | None = None
    """Cache hint from the Orchestrator for distributed KV cache.

    The serving layer never reads it: it re-serializes the object onto
    ``TextContext.dkv_cache_hint`` and hands those bytes to the dKV connector,
    which parses them in Rust to route each block to the instance that holds
    it. See ``dkv/docs/cache-hint.md``.
    """
    cache_salt: str | None = None
    """Optional per-request salt that isolates this prompt's prefix-cache
    entries from other requests sharing the same tokens.

    Combined with ``kv_cache_hash_seed`` via XOR to seed the block hash.
    Works under any ``kv_cache_hash_algo``: a cryptographic guarantee
    under ``sha256``/``sha256_64``, best-effort under ``ahash64``. Capped
    at 512 chars at the OpenAI schema layer.
    """

    def __str__(self) -> str:
        return str(self.request_id)

    def images_for_processing(self) -> list[bytes | PILImage]:
        """Return the images for tokenizer preprocessing, decoded once.

        Prefers the pre-decoded :attr:`decoded_images` (decoded and validated
        once at the API server) and falls back to the raw :attr:`images` bytes
        for offline and test callers. Tokenizers consume images through this so
        the decode-once policy lives in one place rather than being repeated at
        every per-model decode site.
        """
        return self.decoded_images or self.images

    def __post_init__(self) -> None:
        """Validates mutual exclusivity, image-messaging constraints, and message-image consistency after object initialization."""
        # Convert dict messages to TextGenerationRequestMessage objects
        if self.messages is not None:
            converted_messages: list[TextGenerationRequestMessage] = []
            for msg in self.messages:
                if isinstance(msg, dict):
                    converted_messages.append(
                        TextGenerationRequestMessage(**msg)
                    )
                elif isinstance(msg, TextGenerationRequestMessage):
                    converted_messages.append(msg)
                else:
                    raise TypeError(f"Invalid message type: {type(msg)}")
            # Use object.__setattr__ for frozen dataclass
            object.__setattr__(self, "messages", converted_messages)

        if self.prompt and self.messages:
            raise ValueError(
                "both prompt and messages cannot be provided to TextGenerationRequest"
            )

        if self.images and isinstance(self.prompt, str):
            raise ValueError(
                "string prompts cannot be provided, when images are provided, use messages"
            )

        if self.videos and isinstance(self.prompt, str):
            raise ValueError(
                "string prompts cannot be provided, when videos are provided, use messages"
            )

        if self.images and self.number_of_images != len(self.images):
            raise ValueError(
                f"number of images provided in TextGenerationRequest do not match messages:\n{self.messages}"
            )

        if self.videos and self.number_of_videos != len(self.videos):
            raise ValueError(
                f"number of videos provided in TextGenerationRequest do not match messages:\n{self.messages}"
            )

    @cached_property
    def number_of_images(self) -> int:
        """Returns the total number of image-type contents across all provided messages.

        Returns:
            Total count of image-type contents found in messages.
        """
        return (
            sum(message.number_of_images for message in self.messages)
            if self.messages
            else 0
        )

    @cached_property
    def number_of_videos(self) -> int:
        """Returns the total number of video-type contents across all provided messages.

        Returns:
            Total count of video-type contents found in messages.
        """
        return (
            sum(message.number_of_videos for message in self.messages)
            if self.messages
            else 0
        )


class BatchType(Enum):
    """Type of batch."""

    CE = "CE"
    """Context encoding batch."""
    TG = "TG"
    """Token generation batch."""


@dataclass
class CompletedBatchStats:
    """Execution stats for a batch whose outputs have been synchronized."""

    batch_type: BatchType
    """Type of the completed batch."""

    batch_size: int
    """Number of requests in the completed batch."""

    num_input_tokens: int
    """Number of input tokens in the completed batch."""

    num_context_tokens: int
    """Number of context tokens in the completed batch."""

    execution_time_s: float
    """Execution time of the completed batch, in seconds."""

    num_output_tokens: int | None = None
    """Output tokens produced by the completed batch, when known (currently
    only reported by speculative decoding). ``None`` otherwise."""

    draft_tokens_generated: int = 0
    """Draft tokens generated by the completed batch (speculative decoding)."""

    draft_tokens_accepted: int = 0
    """Draft tokens accepted by the completed batch (speculative decoding)."""

    avg_acceptance_length: float = 0.0
    """Average acceptance length for the completed batch (speculative
    decoding)."""

    max_acceptance_length: int = 0
    """Maximum possible acceptance length, i.e. the configured number of
    speculative tokens (speculative decoding)."""

    acceptance_rate_per_position: list[float] = field(default_factory=list)
    """Per-position draft acceptance rates for the completed batch
    (speculative decoding)."""

    early_sync_duration_s: float | None = None
    """Wall-clock time spent in the early-sync guard's blocking
    ``sync_and_process_outputs()`` call (see
    ``_should_early_sync_prev_batch``), when it fired for this batch.
    ``None`` when the guard did not fire."""

    @property
    def prompt_throughput(self) -> float:
        """Prompt-side throughput of the completed batch in tokens/second."""
        if self.execution_time_s <= 0.0:
            return 0.0
        return self.num_input_tokens / self.execution_time_s

    @property
    def generation_throughput(self) -> float:
        """Generation throughput of the completed batch in tokens/second.

        Uses the known output-token count when available (speculative
        decoding, TG batches); otherwise counts one token per request.
        """
        if self.execution_time_s <= 0.0:
            return 0.0
        if (
            self.num_output_tokens is not None
            and self.batch_type == BatchType.TG
        ):
            return self.num_output_tokens / self.execution_time_s
        return self.batch_size / self.execution_time_s


@dataclass(eq=True)
class TextGenerationInputs(PipelineInputs, Generic[TextGenerationContextType]):
    """Input parameters for text generation pipeline operations."""

    batches: list[list[TextGenerationContextType]]
    """Variable list of batches, with each batch being a list of contexts.

    There can be multiple batches when using data parallelism, in which each
    batch is mapped to a different device replica.
    """

    input_tokens: int = -1
    """Number of input tokens."""

    batch_type: BatchType = BatchType.TG
    """Type of batch."""

    per_replica_input_tokens: list[int] = field(default_factory=list)
    """Per-replica active-token sums, excluding DP padding dummies. Frozen at
    construction: token windows mutate during scheduling, so later reads of
    ``active_length`` no longer describe this batch."""

    per_replica_context_tokens: list[int] = field(default_factory=list)
    """Per-replica processed-token (context) sums, excluding DP padding
    dummies. Frozen at construction like ``per_replica_input_tokens``."""

    def __post_init__(self) -> None:
        self.input_tokens = sum(
            ctx.tokens.active_length for ctx in self.flat_batch
        )
        self.context_tokens = sum(
            ctx.tokens.processed_length for ctx in self.flat_batch
        )
        self.per_replica_input_tokens = [
            sum(
                ctx.tokens.active_length
                for ctx in batch
                if not getattr(ctx, "_is_padding_ctx", False)
            )
            for batch in self.batches
        ]
        self.per_replica_context_tokens = [
            sum(
                ctx.tokens.processed_length
                for ctx in batch
                if not getattr(ctx, "_is_padding_ctx", False)
            )
            for batch in self.batches
        ]
        self.batch_type = BatchType.TG
        for context in self.flat_batch:
            if context.tokens.generated_length == 0:
                self.batch_type = BatchType.CE
                break

    @property
    def flat_batch(self) -> list[TextGenerationContextType]:
        """Flattened list of contexts across all replicas."""
        return [context for batch in self.batches for context in batch]

    @property
    def batch_size(self) -> int:
        """Number of requests in the batch."""
        return sum(
            1 for context in self.flat_batch if not context._is_padding_ctx
        )

    def __bool__(self) -> bool:
        return len(self.flat_batch) > 0

    def __repr__(self) -> str:
        return (
            "TextGenerationInputs("
            f"batch_size={len(self.flat_batch)}, "
            f"batch_type={self.batch_type.value}"
            ")"
        )

    @property
    def enable_echo(self) -> bool:
        """``True`` if any context in the batch has echo enabled."""
        return any(self.batch_echo)

    @property
    def enable_log_probs(self) -> bool:
        """``True`` if any context in the batch requests log probabilities."""
        return any(self.batch_top_log_probs)

    @cached_property
    def batch_top_log_probs(self) -> list[int]:
        """List of requested top log probabilities per context in the batch."""
        return [ctx.log_probabilities for ctx in self.flat_batch]

    @cached_property
    def batch_echo(self) -> list[bool]:
        """List indicating whether echo is enabled for each context in the batch."""
        return [ctx.log_probabilities_echo for ctx in self.flat_batch]
