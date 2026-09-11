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

import base64
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Literal

from openai.types.chat.completion_create_params import ResponseFormat
from PIL import Image
from pydantic import BaseModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing_extensions import TypedDict

DatasetMode = Literal["local", "huggingface"]


@dataclass
class SharedContext:
    """A single entry in the prefix-cache warmup list.

    Represents the longest observed variant of a unique shared context.
    Sending the longest variant is sufficient because all shorter variants
    built from the same base are token-level prefixes of it.
    """

    text: str
    num_tokens: int


class OpenAIImageURL(TypedDict):
    url: str


class OpenAIImage(TypedDict):
    type: Literal["image_url"]
    image_url: OpenAIImageURL


class TextContentBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageURLDetail(BaseModel):
    url: str


class ImageContentBlock(BaseModel):
    # Images are carried in the separate RequestFuncInput.images field and
    # appended to the payload after serialisation; they appear in content lists
    # only at the wire layer, not in prompts constructed by this codebase.
    # The model is defined here so ChatMessage.content can represent the full
    # OpenAI content-block union and round-trip correctly.
    type: Literal["image_url"] = "image_url"
    image_url: ImageURLDetail


class ChatMessage(BaseModel):
    # role is "user", "assistant", or "system" in practice. "system" is
    # used by chat-judge today.
    # TODO: wire sys_prompt_ratio support with "system" role through the
    # multi-turn path.
    role: str
    # content is always list[TextContentBlock] in prompts produced by this
    # codebase.  The str variant exists to match the OpenAI spec and to support
    # _prepend_run_prefix_to_formatted_prompt, which handles both shapes
    # defensively.
    content: str | list[TextContentBlock | ImageContentBlock]


@dataclass
class SampledRequest:
    prompt_formatted: str | list[ChatMessage]
    prompt_len: int
    output_len: int | None
    encoded_images: list[OpenAIImage]
    ignore_eos: bool
    response_format: ResponseFormat | None = None
    # OpenAI-style tool definitions forwarded as the chat-completions ``tools``
    # field. Typed as ``list[dict]`` (not ``ChatCompletionToolUnionParam``) so
    # datasets can pass through whatever shape the server expects; the driver
    # serialises this as-is.
    tools: list[dict[str, Any]] | None = None


@dataclass
class PixelGenerationImageOptions:
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    guidance_scale: float | None = None
    negative_prompt: str | None = None
    seed: int | None = None
    num_frames: int | None = None


@dataclass
class PixelGenerationSampledRequest(SampledRequest):
    """A sampled request for pixel generation.

    prompt_len, output_len, ignore_eos are not used for pixel generation.
    """

    prompt_formatted: str
    prompt_len: int = 0
    output_len: int | None = None
    encoded_images: list[OpenAIImage] = field(default_factory=list)
    ignore_eos: bool = True
    input_image_paths: list[str] = field(default_factory=list)
    image_options: PixelGenerationImageOptions | None = None


MessageSource = Literal["user", "assistant", "system"]


@dataclass
class SessionMessage:
    source: MessageSource
    content: str
    num_tokens: int
    delay_until_next_message: float | None = None
    # An agent-loop round; the wire role is still "user".
    is_agentic: bool = False
    # Any image tokens are already counted in `num_tokens`, so consumers
    # must not add them again.
    images: list[OpenAIImage] = field(default_factory=list)


@dataclass
class ChatSession:
    id: int | None
    messages: Sequence[SessionMessage]
    prefix_turns: int = 0

    @property
    def num_turns(self) -> int:
        """Model invocations, agent-loop rounds included: the request count."""
        return sum(1 for m in self.messages if m.source == "user")


@dataclass
class RequestSamples:
    requests: Sequence[SampledRequest]
    shared_contexts: list[SharedContext] = field(default_factory=list)


@dataclass
class ChatSamples:
    chat_sessions: Sequence[ChatSession]
    shared_contexts: list[SharedContext] = field(default_factory=list)


Samples = RequestSamples | ChatSamples


def estimate_num_tokens(tokenizer: PreTrainedTokenizerBase, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_chat_message(
    source: MessageSource,
    prompt: str,
    tokenizer: PreTrainedTokenizerBase,
    num_tokens: int | None = None,
    delay_until_next_message: float | None = None,
) -> SessionMessage:
    return SessionMessage(
        source,
        prompt,
        num_tokens or estimate_num_tokens(tokenizer, prompt),
        delay_until_next_message,
    )


def encode_image(img: Image.Image) -> OpenAIImage:
    """
    Convert the given PIL.Image.Image to JPEG and encode in base64.
    Returns an openai API image_url content entry with the encoded string.
    """
    img_buffer = BytesIO()
    # Drop alpha channel and convert to jpeg
    img.convert("RGB").save(img_buffer, format="JPEG")
    # Encode in base64 and convert to str
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    # return openai-api dict
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
    }


def encode_image_from_file_path(file_path: str) -> OpenAIImage:
    """
    Read an image file as raw bytes and encode in base64 without any transformations.
    Preserves the exact original file data and determines MIME type from file extension.

    Args:
        file_path: Path to the image file

    Returns:
        OpenAI API image_url content entry with the encoded string

    Raises:
        ValueError: If the file extension is not supported
        FileNotFoundError: If the file does not exist
    """
    extension_to_mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }

    # Check if extension is supported
    _, ext = os.path.splitext(file_path.lower())
    if ext not in extension_to_mime:
        supported_exts = ", ".join(extension_to_mime.keys())
        raise ValueError(
            f"Unsupported image file extension '{ext}'. "
            f"Supported extensions: {supported_exts}"
        )

    # Base64 encode file bytes
    with open(file_path, "rb") as f:
        image_bytes = f.read()

    img_base64 = base64.b64encode(image_bytes).decode("utf-8")

    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{extension_to_mime[ext]};base64,{img_base64}"
        },
    }
