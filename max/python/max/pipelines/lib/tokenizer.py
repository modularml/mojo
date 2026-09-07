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
# mypy: disable-error-code="import-not-found"
"""Implementations of provided tokenizers."""

from __future__ import annotations

import asyncio
import io
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
import numpy.typing as npt
from max.pipelines.context import (
    EOSTracker,
    GrammarEnforcementState,
    ImageMetadata,
    TextAndVisionContext,
    TextContext,
    TokenBuffer,
)
from max.pipelines.context.exceptions import PromptTooLongError
from max.pipelines.modeling.types import (
    PipelineTokenizer,
    TextGenerationRequest,
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)
from max.support.image import find_contiguous_ranges, hash_image
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
from typing_extensions import ParamSpec

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig


def open_image(image: bytes | Image.Image) -> Image.Image:
    """Decode raw image ``bytes`` into a ``PIL.Image``, or pass one through.

    Vision tokenizers receive each image either as raw ``bytes`` (offline and
    test callers) or as a ``PIL.Image`` that the API server already decoded and
    validated once at admission (see
    :attr:`~max.pipelines.modeling.types.TextGenerationRequest.decoded_images`).
    Routing both through this helper lets a tokenizer reuse the pre-decoded
    image instead of decoding the same bytes a second time.

    Args:
        image: Raw encoded image bytes, or an already-decoded ``PIL.Image``.

    Returns:
        The decoded ``PIL.Image``.
    """
    if isinstance(image, Image.Image):
        return image
    return Image.open(io.BytesIO(image))


async def convert_token_to_id(
    tokenizer: PipelineTokenizer[Any, Any, Any],
    token: str,
) -> int | None:
    """Convert a token string to its token ID, or None if not a single token."""
    # Workaround: PipelineTokenizer does not expose convert_tokens_to_ids(),
    # so we encode the string and verify it maps to exactly one token ID.
    encoded = await tokenizer.encode(token, add_special_tokens=False)
    if len(encoded) != 1:
        return None
    return int(encoded[0])


def resolve_single_special_token(delegate: Any, token: str) -> int:
    """Resolve a single special-token string to its id via an HF delegate.

    Suitable for use in tokenizer ``__init__`` where the architecture
    knows ``token`` is registered as a single special token in the
    underlying vocab (for example, reasoning delimiters like ``<think>``
    on Kimi K2.5 or ``<|channel>`` on Gemma 4).

    Raises:
        ValueError: If ``token`` is missing from the vocab (resolves to
            ``unk_token_id``) or maps to more than one id.
    """
    token_id = delegate.convert_tokens_to_ids(token)
    if isinstance(token_id, list):
        raise ValueError(
            f"Special token {token!r} resolved to multiple ids "
            f"({token_id!r}); expected a single id."
        )
    if token_id == delegate.unk_token_id:
        raise ValueError(
            f"Special token {token!r} not found in tokenizer vocabulary "
            f"(resolved to unk_token_id)."
        )
    return int(token_id)


logger = logging.getLogger("max.pipelines")


def encode_dkv_cache_hint(hint: dict[str, Any] | None) -> bytes | None:
    """Re-serializes a request's ``dkv_cache_hint`` to the bytes dKV parses.

    The hint arrives as a decoded JSON object and the dKV connector parses it
    in Rust, so MAX only has to hand back its wire form. Nothing here reads the
    contents: the version gate, the instance table, and every routing decision
    live in ``dkv-connector`` (see ``dkv/docs/cache-hint.md``), which is what
    lets the Orchestrator adopt a new hint schema without redeploying MAX.

    Returns ``None`` when the field is absent, empty, or not encodable, all of
    which the connector serves as an unhinted load. A hint never fails a
    request: an HTTP caller cannot reach the unencodable case, because the
    field is typed ``dict[str, Any]`` and filled from a parsed JSON body, but
    a caller building a request in Python can, and a cache is not worth a
    failed request.
    """
    if not hint:
        return None
    try:
        return json.dumps(hint, separators=(",", ":")).encode()
    except (TypeError, ValueError):
        logger.warning(
            "Dropping an unencodable dkv_cache_hint; serving as though the "
            "cache missed."
        )
        return None


TokenGeneratorContext = TypeVar("TokenGeneratorContext")

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _handle_decode_overflow(
    encoded: npt.NDArray[np.integer[Any]],
    vocab_size: int,
) -> str:
    """Diagnose and raise a helpful OverflowError for token decoding issues.

    Args:
        encoded: The token array that caused the overflow.
        vocab_size: The tokenizer's vocabulary size.
        original_error: The original OverflowError that was caught.

    """
    issues = []

    if (encoded >= vocab_size).any():
        invalid_mask = encoded >= vocab_size
        invalid_indices = np.where(invalid_mask)[0]
        invalid_values = encoded[invalid_mask]
        issues.append(
            f"Token IDs exceeding vocab size ({vocab_size}) at indices "
            f"{invalid_indices.tolist()}: {invalid_values.tolist()}"
        )

    if (encoded < 0).any():
        negative_mask = encoded < 0
        negative_indices = np.where(negative_mask)[0]
        negative_values = encoded[negative_mask]
        issues.append(
            f"Negative token IDs at indices {negative_indices.tolist()}: "
            f"{negative_values.tolist()}"
        )

    if issues:
        error_msg = (
            f"OverflowError during token decoding. Invalid token IDs detected:\n"
            f"  {'; '.join(issues)}\n"
            f"  Vocab size: {vocab_size}, Array shape: {encoded.shape}, "
            f"dtype: {encoded.dtype}"
        )
    else:
        error_msg = (
            f"OverflowError during token decoding (no obvious invalid values). "
            f"Vocab size: {vocab_size}, Array shape: {encoded.shape}, "
            f"dtype: {encoded.dtype}, Token IDs: {encoded.tolist()}"
        )

    logger.error(error_msg)
    return error_msg


class IdentityPipelineTokenizer(
    PipelineTokenizer[TokenGeneratorContext, str, TextGenerationRequest],
):
    """A pass-through tokenizer that returns prompts unchanged."""

    @property
    def eos_token_ids(self) -> set[int]:
        """Returns the end-of-sequence token IDs (empty for identity)."""
        return set()

    @property
    def expects_content_wrapping(self) -> bool:
        """Returns whether this tokenizer expects content wrapping."""
        return False

    async def encode(
        self, prompt: str, add_special_tokens: bool = False
    ) -> str:
        """Returns the prompt unchanged (identity encoding)."""
        return prompt

    async def decode(
        self,
        encoded: str,
        **kwargs,
    ) -> str:
        """Returns the encoded string unchanged (identity decoding)."""
        if isinstance(encoded, str):
            return encoded
        return ""


def max_tokens_to_generate(
    prompt_size: int,
    max_length: int | None,
    max_new_tokens: int | None = None,
) -> int | None:
    """Returns the max number of new tokens to generate."""
    if max_length is None:
        return max_new_tokens
    _difference_between_max_and_prompt = max(max_length - prompt_size, 0)
    if max_new_tokens is None:
        return _difference_between_max_and_prompt
    return min(max_new_tokens, _difference_between_max_and_prompt)


async def run_with_default_executor(
    fn: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs
) -> _R:
    """Runs a callable in the default thread pool executor.

    Args:
        fn: Callable to run.
        *args: Positional arguments for ``fn``.
        **kwargs: Keyword arguments for ``fn``.

    Returns:
        The result of ``fn(*args, **kwargs)``.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args, **kwargs)


def replace_unpaired_surrogates(prompt: str) -> str:
    """Returns ``prompt`` with each unpaired UTF-16 surrogate replaced by U+FFFD.

    A JSON request may legally carry lone surrogate escapes -- for example an
    emoji whose surrogate pair was split by client-side truncation. The parsed
    value is a valid Python :class:`str` but is not encodable as UTF-8, so
    handing it to a HuggingFace fast (Rust) tokenizer raises an opaque
    ``TypeError`` inside ``encode_batch``. Every surrogate code point
    (U+D800--U+DFFF) is mapped to the Unicode replacement character (U+FFFD);
    in JSON-decoded input these are exactly the unpaired surrogates, since the
    parser reconstitutes a valid pair into a single non-surrogate code point.
    Well-formed text is left unchanged. An ASCII fast path returns immediately;
    otherwise the string is probed once with ``encode("utf-8")`` and the
    per-character replacement runs only when that probe fails.
    """
    if prompt.isascii():
        return prompt
    try:
        prompt.encode("utf-8")
    except UnicodeEncodeError:
        return "".join(
            "\ufffd" if "\ud800" <= char <= "\udfff" else char
            for char in prompt
        )
    return prompt


async def build_eos_tracker_for_request(
    eos_token_ids: set[int],
    request: TextGenerationRequest,
    encode_fn: Callable[[str, bool], Awaitable[npt.NDArray[np.integer[Any]]]],
) -> EOSTracker:
    """Builds an :class:`~max.pipelines.modeling.types.EOSTracker` from request sampling params.

    Args:
        eos_token_ids: The model's EOS token IDs from tokenizer/model config.
        request: Generation request; uses ``request.sampling_params`` for stops.
        encode_fn: Async encode callable ``(text, add_special_tokens) -> token ids``.

    Returns:
        Configured :class:`~max.pipelines.modeling.types.EOSTracker` for this request.
    """
    params = request.sampling_params
    resolved_eos_token_ids = set(eos_token_ids)
    eos_sequences: list[list[int]] = []
    if params.ignore_eos:
        resolved_eos_token_ids = set()
    else:
        if params.stop_token_ids:
            resolved_eos_token_ids.update(params.stop_token_ids)
        if params.stop:
            for stop_string in params.stop:
                tokenized = (await encode_fn(stop_string, False)).tolist()
                if tokenized:
                    eos_sequences.append(tokenized)
    return EOSTracker(
        eos_token_ids=resolved_eos_token_ids,
        eos_sequences=eos_sequences,
        eos_stop_strings=params.stop or [],
    )


class TextTokenizer(
    PipelineTokenizer[
        TextContext, npt.NDArray[np.integer[Any]], TextGenerationRequest
    ]
):
    """Encapsulates creation of :class:`TextContext` and specific token encode/decode logic.

    Args:
        model_path: Path to the model/tokenizer
        revision: Git revision/branch to use
        max_length: Maximum sequence length
        trust_remote_code: Whether to trust remote code from the model
        enable_llama_whitespace_fix: Enable whitespace fix for Llama tokenizers
        pipeline_config: Optional pipeline configuration
        chat_template: Optional custom chat template string to override the one
                        shipped with the Hugging Face model config. This allows
                        customizing the prompt formatting for different use cases.
    """

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        enable_llama_whitespace_fix: bool = False,
        chat_template: str | None = None,
        **unused_kwargs,
    ) -> None:
        self.model_path = model_path

        try:
            self.delegate = AutoTokenizer.from_pretrained(
                model_path,
                revision=revision,
                trust_remote_code=trust_remote_code,
                # If `max_length` is None, the max length will be taken
                # from the HuggingFace tokenizer_config.
                model_max_length=max_length,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load tokenizer from {model_path}. "
                "This can happen if:\n"
                "- The model is not fully supported by the transformers python package\n"
                "- Required configuration files are missing\n"
                "- The model path is incorrect\n"
                "- '--trust-remote-code' is needed but not set\n"
            ) from e

        # Override chat template if provided
        # This will be used by the delegate's apply_chat_template method automatically
        self._custom_template_provided = chat_template is not None
        self._chat_template = chat_template
        if chat_template is not None:
            self.delegate.chat_template = chat_template
            logger.info(
                f"Set custom chat template on tokenizer for {model_path}"
            )

        self.max_length = max_length or self.delegate.model_max_length

        # configure Llama whitespace fix if needed
        self._enable_llama_whitespace_fix = (
            enable_llama_whitespace_fix and self._strips_leading_whitespace
        )
        (
            self._llama_whitespace_fix_dummy_token_id,
            self._llama_whitespace_fix_dummy_token_len,
        ) = self._llama_whitespace_fix_dummy_token

        # cache tokenizer eos token ids
        eos_token_id = self.delegate.eos_token_id
        self._eos_token_ids = (
            {eos_token_id} if eos_token_id is not None else set()
        )

        if pipeline_config:
            target_eos = getattr(
                pipeline_config.model.huggingface_config, "eos_token_id", None
            )
            draft_hf = getattr(
                pipeline_config.draft_model, "huggingface_config", None
            )
            draft_eos = getattr(draft_hf, "eos_token_id", None)
            for eos in (target_eos, draft_eos):
                if isinstance(eos, int):
                    self._eos_token_ids.add(eos)
                elif isinstance(eos, list):
                    self._eos_token_ids.update(eos)

    @property
    def eos_token_ids(self) -> set[int]:
        """The full set of token ids that end generation for this model."""
        return set(self._eos_token_ids)

    @cached_property
    def tokenizer_vocab_size(self) -> int:
        """Vocabulary size of the HuggingFace tokenizer delegate."""
        return len(self.delegate)

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None,
        **chat_template_options: Any,
    ) -> str:
        """Applies the delegate chat template to messages (and optional tools)."""
        chat_template_options = {
            "add_generation_prompt": True,
            **chat_template_options,
        }

        flattened = [message.flatten_content() for message in messages]

        try:
            if self._custom_template_provided:
                # A ``trust_remote_code`` tokenizer may override
                # ``apply_chat_template`` to render a format hardcoded in
                # Python, never reading the ``chat_template`` attribute -- which
                # silently discards an explicit ``--chat-template``. Render the
                # requested template through the base implementation so the
                # override is the format that actually reaches the model.
                templated_message = PreTrainedTokenizerBase.apply_chat_template(
                    self.delegate,
                    flattened,
                    chat_template=self._chat_template,
                    tokenize=False,
                    tools=tools,
                    **chat_template_options,
                )
            else:
                templated_message = self.delegate.apply_chat_template(
                    flattened,
                    tokenize=False,
                    tools=tools,
                    **chat_template_options,
                )
        except Exception as e:
            if self._custom_template_provided:
                # Provide additional context when a custom template is used
                error_msg = (
                    f"Failed to apply custom chat template. This may indicate an issue "
                    f"with your custom prompt template. Please check your template syntax "
                    f"and ensure it properly handles the provided messages and tools.\n\n"
                    f"Template variables available:\n"
                    f"- messages: List of conversation messages with 'role' and 'content' fields\n"
                    f"- tools: List of available tools (if provided)\n"
                    f"- add_generation_prompt: Boolean for adding generation prompt\n\n"
                    f"Original error: {type(e).__name__}: {str(e)}"
                )
                raise ValueError(error_msg) from e
            else:
                # Re-raise the original error for default templates
                raise

        assert isinstance(templated_message, str)
        return templated_message

    @property
    def expects_content_wrapping(self) -> bool:
        """Returns whether this tokenizer expects content wrapping."""
        return False

    async def encode(
        self, prompt: str | Sequence[int], add_special_tokens: bool = True
    ) -> npt.NDArray[np.integer[Any]]:
        """Transforms the provided prompt into a token array."""
        encoded_prompt: npt.NDArray[np.integer[Any]]
        if isinstance(prompt, str):

            def _encode_fn(
                prompt: str, add_special_tokens: bool
            ) -> npt.NDArray[np.integer[Any]]:
                return self.delegate.encode(
                    replace_unpaired_surrogates(prompt),
                    add_special_tokens=add_special_tokens,
                )

            # Note: the underlying tokenizer may not be thread safe in some cases, see https://github.com/huggingface/tokenizers/issues/537
            # Add a standard (non-async) lock in the executor thread if needed.
            encoded_prompt = await run_with_default_executor(
                _encode_fn,
                prompt,
                add_special_tokens,
            )

            encoded_prompt = np.array(encoded_prompt)
        else:
            encoded_prompt = np.array(list(prompt))

        if self.max_length and len(encoded_prompt) > self.max_length:
            raise PromptTooLongError(len(encoded_prompt), self.max_length)

        return encoded_prompt

    async def decode(
        self, encoded: npt.NDArray[np.integer[Any]], **kwargs
    ) -> str:
        """Transforms a provided encoded token array back into readable text."""
        # Callers pass plain ints (log-probability responses) and token lists
        # (CLI streaming) as well as arrays; normalize them all.
        if not isinstance(encoded, np.ndarray):
            encoded = np.asarray(encoded)

        # There is an issue where Llama tokenizer strips leading spaces
        # if a single token is decoded at a time. This is a temporary
        # fix until the issue resolved on the Tokenizers side.
        # More information:
        # https://github.com/huggingface/transformers/issues/31643
        # https://github.com/Lightning-AI/litgpt/pull/1559
        if self._enable_llama_whitespace_fix and encoded.size == 1:
            return self._decode_with_llama_whitespace_fix(encoded, **kwargs)

        try:
            return self.delegate.decode(encoded.tolist(), **kwargs)
        except OverflowError as e:
            error_msg = _handle_decode_overflow(encoded, len(self.delegate))
            raise OverflowError(error_msg) from e

    async def _generate_prompt_and_token_ids(
        self,
        prompt: Sequence[int] | str | None,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None = None,
        **chat_template_options: Any,
    ) -> tuple[str | list[int], npt.NDArray[np.integer[Any]]]:
        if isinstance(prompt, str | list):
            return prompt, await self.encode(prompt, add_special_tokens=True)
        elif isinstance(messages, list):
            prompt = self.apply_chat_template(
                messages, tools, **chat_template_options
            )
            return prompt, await self.encode(prompt, add_special_tokens=False)
        else:
            raise ValueError(
                "either prompt must be provided as a list[int] or str, or messages must be provided as a list[TextGenerationRequestMessage]"
            )

    async def _encode_stop_criteria(self, stop: list[str]) -> list[list[int]]:
        """Encodes ``stop`` to be used as stop criteria during generation."""
        stop_tokenized: list[list[int]] = []
        for stop_crit in stop:
            tokenized: list[int] = (
                await self.encode(stop_crit, False)
            ).tolist()
            stop_tokenized.append(tokenized)
        return stop_tokenized

    async def _get_eos_variables(
        self,
        ignore_eos: bool,
        stop_token_ids: list[int] | None,
        stop: list[str] | None,
    ) -> tuple[set[int], list[list[int]]]:
        eos_token_ids = set(self._eos_token_ids)
        eos_sequences = list()

        if ignore_eos:
            eos_token_ids = set()
        elif stop_token_ids:
            eos_token_ids.update(stop_token_ids)
        elif stop:
            eos_sequences = await self._encode_stop_criteria(stop)

        return eos_token_ids, eos_sequences

    async def create_eos_tracker(
        self, request: TextGenerationRequest
    ) -> EOSTracker:
        """Builds an :class:`EOSTracker` from the request sampling params and tokenizer default EOS token IDs."""
        return await build_eos_tracker_for_request(
            self._eos_token_ids,
            request,
            self.encode,
        )

    async def new_context(self, request: TextGenerationRequest) -> TextContext:
        """Create a new TextContext object, leveraging necessary information from TextGenerationRequest."""
        # Encode Prompt / Messages
        _prompt, token_ids = await self._generate_prompt_and_token_ids(
            prompt=request.prompt,
            messages=request.messages,
            tools=request.tools,
            **(request.chat_template_options or {}),
        )

        json_schema = (
            json.dumps(request.response_format.json_schema)
            if request.response_format
            and request.response_format.json_schema is not None
            else None
        )

        grammar = (
            request.response_format.grammar if request.response_format else None
        )

        grammar_state = GrammarEnforcementState.from_response_format(
            request.response_format
        )

        # Calculate Max Length
        max_new_tokens = None
        if request.sampling_params.max_new_tokens is not None:
            max_new_tokens = request.sampling_params.max_new_tokens

        max_gen_tokens = max_tokens_to_generate(
            len(token_ids), self.max_length, max_new_tokens
        )

        token_buffer = TokenBuffer(
            array=token_ids.astype(np.int64, copy=False),
        )

        context = TextContext(
            request_id=request.request_id,
            eos_tracker=await self.create_eos_tracker(request),
            max_length=len(token_ids) + max_gen_tokens
            if max_gen_tokens is not None
            else self.max_length,
            tokens=token_buffer,
            vocab_size=self.tokenizer_vocab_size,
            log_probabilities=request.logprobs,
            log_probabilities_echo=request.echo,
            json_schema=json_schema,
            grammar=grammar,
            grammar_state=grammar_state,
            sampling_params=request.sampling_params,
            model_name=request.model_name,
            target_endpoint=request.target_endpoint,
            dkv_cache_hint=encode_dkv_cache_hint(request.dkv_cache_hint),
            cache_salt=request.cache_salt,
        )

        return context

    @property
    def _strips_leading_whitespace(self) -> bool:
        """Detect if this tokenizer strips leading whitespace on single-token decode.

        SentencePiece tokenizers encode word boundaries as a ``▁`` prefix
        which gets stripped when a single token is decoded in isolation.
        Instead of checking for specific tokenizer class names (which change
        across ``transformers`` versions), we test the behavior directly.
        """
        enc = self.delegate.encode("A B", add_special_tokens=False)
        enc_a = self.delegate.encode("A", add_special_tokens=False)
        b_tokens = enc[len(enc_a) :]
        if not b_tokens:
            return False
        decoded = self.delegate.decode([b_tokens[0]])
        return not decoded.startswith(" ")

    @property
    def _llama_whitespace_fix_dummy_token(self) -> tuple[int, int]:
        dummy_token_id = 33  # \x1e
        dummy_token_decoded = self.delegate.decode([dummy_token_id])
        return dummy_token_id, len(dummy_token_decoded)

    def _decode_with_llama_whitespace_fix(
        self, encoded: npt.NDArray[np.integer[Any]], **kwargs
    ) -> str:
        if encoded.shape == ():
            # The np.insert below will replace the token instead of prepend it
            # if the array is actually a scalar.  Reshape to a 1-length rank-1
            # array in this case.  See MODELS-467 for symptom.
            encoded = encoded.reshape((1,))

        decoded = self.delegate.decode(
            np.insert(
                encoded, 0, self._llama_whitespace_fix_dummy_token_id
            ).tolist(),
            **kwargs,
        )
        return decoded[self._llama_whitespace_fix_dummy_token_len :]


class TextAndVisionTokenizer(
    PipelineTokenizer[
        TextAndVisionContext,
        npt.NDArray[np.integer[Any]],
        TextGenerationRequest,
    ],
):
    """Encapsulates creation of TextAndVisionContext and specific token encode/decode logic."""

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        **unused_kwargs,
    ) -> None:
        self.model_path = model_path

        self.delegate = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            # If `max_length` is None, the max length will be taken
            # from the HuggingFace tokenizer_config.
            model_max_length=max_length,
        )
        self.max_length = max_length or self.delegate.model_max_length

        # Use the pre-loaded HuggingFace config from pipeline_config
        config = pipeline_config.model.huggingface_config

        self.processor = AutoProcessor.from_pretrained(
            model_path, revision=revision, trust_remote_code=trust_remote_code
        )
        eos_token_id = self.delegate.eos_token_id
        self._eos_token_ids = (
            {eos_token_id} if eos_token_id is not None else set()
        )

        huggingface_config = pipeline_config.model.huggingface_config
        if eos_token_id := getattr(huggingface_config, "eos_token_id", None):
            if isinstance(eos_token_id, int):
                self._eos_token_ids.add(eos_token_id)
            elif isinstance(eos_token_id, list):
                self._eos_token_ids.update(eos_token_id)

        self.enable_prefix_caching = (
            pipeline_config.model.kv_cache.enable_prefix_caching
        )

        # Qwen2.5VL uses image_token_id
        # Pixtral uses image_token_index
        vision_token_ids: list[int] = []
        for vision_token_id_name in [
            "image_token_id",
            "image_token_index",
        ]:
            if vision_token_id := getattr(config, vision_token_id_name, None):
                vision_token_ids.append(vision_token_id)
        if not vision_token_ids:
            raise ValueError("vision_token_id not found in model_config config")
        self.vision_token_ids = vision_token_ids

        # This is pixtral specific hack as it also has a image_break_token_id
        if image_break_token_id := getattr(
            self.processor, "image_break_token_id", None
        ):
            self.vision_token_ids.append(image_break_token_id)

    @property
    def eos_token_ids(self) -> set[int]:
        """The full set of token ids that end generation for this model."""
        return set(self._eos_token_ids)

    @cached_property
    def tokenizer_vocab_size(self) -> int:
        """Vocabulary size of the HuggingFace tokenizer delegate."""
        return len(self.delegate)

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None = None,
        **chat_template_options: Any,
    ) -> str:
        """Applies the processor's chat template to the messages.

        Args:
            messages: List of messages for the chat template.
            tools: Optional tools available for the model to invoke.
            **chat_template_options: Template options to forward to the Jinja
                template. Merged with ``add_generation_prompt=True`` default.

        Returns:
            The templated chat message as a string.
        """
        chat_template_options = {
            "add_generation_prompt": True,
            **chat_template_options,
        }
        templated_message = self.processor.apply_chat_template(
            [msg.model_dump(exclude_none=True) for msg in messages],
            tokenize=False,
            tools=tools,
            **chat_template_options,
        )
        assert isinstance(templated_message, str)
        return templated_message

    @property
    def expects_content_wrapping(self) -> bool:
        """Returns whether this tokenizer expects content wrapping."""
        return True

    async def encode(
        self, prompt: str | Sequence[int], add_special_tokens: bool = True
    ) -> npt.NDArray[np.integer[Any]]:
        """Transforms the provided prompt into a token array."""
        encoded_prompt: npt.NDArray[np.integer[Any]]
        if isinstance(prompt, str):

            def _encode_fn(
                prompt: str, add_special_tokens: bool
            ) -> npt.NDArray[np.integer[Any]]:
                return self.delegate.encode(
                    replace_unpaired_surrogates(prompt),
                    add_special_tokens=add_special_tokens,
                )

            # Note: the underlying tokenizer may not be thread safe in some cases, see https://github.com/huggingface/tokenizers/issues/537
            # Add a standard (non-async) lock in the executor thread if needed.
            encoded_prompt = await run_with_default_executor(
                _encode_fn,
                prompt,
                add_special_tokens,
            )

            max_length = self.max_length or self.delegate.model_max_length
            if max_length and len(encoded_prompt) > max_length:
                raise PromptTooLongError(len(encoded_prompt), max_length)

            encoded_prompt = np.array(encoded_prompt)
        else:
            encoded_prompt = np.array(list(prompt))

        return encoded_prompt

    async def decode(
        self, encoded: npt.NDArray[np.integer[Any]] | int, **kwargs
    ) -> str:
        """Transforms a provided encoded token array back into readable text."""
        # Log-probability responses decode one token id (a plain int) at a
        # time; match the text tokenizer's handling.
        if isinstance(encoded, int):
            encoded = np.array(encoded)
        try:
            return self.delegate.decode(encoded.tolist(), **kwargs)
        except OverflowError as e:
            error_msg = _handle_decode_overflow(encoded, len(self.delegate))
            raise OverflowError(error_msg) from e

    async def create_eos_tracker(
        self, request: TextGenerationRequest
    ) -> EOSTracker:
        """Builds an :class:`EOSTracker` from the request sampling params and tokenizer default EOS token IDs."""
        return await build_eos_tracker_for_request(
            self._eos_token_ids,
            request,
            self.encode,
        )

    async def new_context(
        self, request: TextGenerationRequest
    ) -> TextAndVisionContext:
        """Create a new TextAndVisionContext object, leveraging necessary information from TextGenerationRequest."""
        prompt: str | Sequence[int]
        add_special_tokens = True
        if request.prompt is not None:
            prompt = request.prompt
        elif request.messages:
            prompt = self.apply_chat_template(
                request.messages,
                request.tools,
                **(request.chat_template_options or {}),
            )
            add_special_tokens = False
        else:
            raise ValueError(f"{request} does not provide messages or prompt.")

        # open_image reuses the API server's decode-once result, or decodes
        # raw bytes on the offline/test fallback path.
        images = (
            [
                _convert_image_mode(open_image(image), "RGB")
                for image in request.images_for_processing()
            ]
            if request.images
            else None
        )

        # InternVL returns a python list
        processed_inputs = self.processor(
            text=(
                replace_unpaired_surrogates(prompt)
                if isinstance(prompt, str)
                else prompt
            ),
            images=images,
            add_special_tokens=add_special_tokens,
            return_tensors="np",
        )

        if "input_ids" not in processed_inputs:
            raise ValueError(
                "input_ids not provided in AutoProcessor output, please ensure you are using the correct processor for multi-modal inputs."
            )

        # TODO: This is a hack to support both Pixtral and InternVL.
        if isinstance(processed_inputs["input_ids"][0], int):
            encoded_prompt = np.array(
                processed_inputs["input_ids"], dtype=np.int64
            )
        else:
            encoded_prompt = np.array(
                processed_inputs["input_ids"][0], dtype=np.int64
            )

        # TODO(zheng): We should probably just make max_new_tokens an optional
        # instead of -1.
        max_new_tokens = None
        if request.sampling_params.max_new_tokens is not None:
            max_new_tokens = request.sampling_params.max_new_tokens

        max_gen_tokens = max_tokens_to_generate(
            encoded_prompt.shape[0], self.max_length, max_new_tokens
        )

        extra_model_args = dict()

        if images is not None:
            if "pixel_values" not in processed_inputs:
                raise ValueError(
                    "pixel_values not provided in AutoProcessor output, please ensure you are using the correct processor for multi-modal inputs."
                )
            pixel_values = processed_inputs["pixel_values"][0]
            if isinstance(pixel_values, list):
                pixel_values = tuple(pixel_values)
            elif isinstance(pixel_values, np.ndarray):
                pixel_values = (pixel_values,)
            else:
                raise ValueError(
                    f"pixel_values is not a numpy array but it is {type(pixel_values)}"
                )

            if "aspect_ratio_ids" in processed_inputs:
                extra_model_args["aspect_ratio_ids"] = (
                    processed_inputs.aspect_ratio_ids
                )
            if "aspect_ratio_mask" in processed_inputs:
                extra_model_args["aspect_ratio_mask"] = (
                    processed_inputs.aspect_ratio_mask
                )
        else:
            pixel_values = tuple()

        # Pass through image token indices if present
        if "image_token_indices" in processed_inputs:
            extra_model_args["image_token_indices"] = processed_inputs[
                "image_token_indices"
            ]

        json_schema = (
            json.dumps(request.response_format.json_schema)
            if request.response_format
            and request.response_format.json_schema is not None
            else None
        )

        grammar = (
            request.response_format.grammar if request.response_format else None
        )

        grammar_state = GrammarEnforcementState.from_response_format(
            request.response_format
        )

        if self.max_length and encoded_prompt.shape[0] > self.max_length:
            raise PromptTooLongError(encoded_prompt.shape[0], self.max_length)

        start_and_end_idxs = find_contiguous_ranges(
            encoded_prompt, self.vision_token_ids
        )

        token_buffer = TokenBuffer(
            array=encoded_prompt.astype(np.int64, copy=False),
        )

        context = TextAndVisionContext(
            request_id=request.request_id,
            eos_tracker=await self.create_eos_tracker(request),
            extra_model_args=extra_model_args,
            tokens=token_buffer,
            vocab_size=self.tokenizer_vocab_size,
            max_length=encoded_prompt.shape[0] + max_gen_tokens
            if max_gen_tokens is not None
            else self.max_length,
            log_probabilities=request.logprobs,
            log_probabilities_echo=request.echo,
            json_schema=json_schema,
            grammar=grammar,
            grammar_state=grammar_state,
            sampling_params=request.sampling_params,
            dkv_cache_hint=encode_dkv_cache_hint(request.dkv_cache_hint),
            images=[
                ImageMetadata(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    pixel_values=pixels,
                    image_hash=hash_image(pixels)
                    if self.enable_prefix_caching
                    else None,
                )
                for (start_idx, end_idx), pixels in zip(
                    start_and_end_idxs, pixel_values, strict=True
                )
            ],
            vision_token_ids=self.vision_token_ids,
            cache_salt=request.cache_salt,
        )

        return context

    async def _encode_stop_criteria(self, stop: list[str]) -> list[list[int]]:
        """Encodes `stop` to be used as stop criteria during generation."""
        stop_tokenized: list[list[int]] = []
        for stop_crit in stop:
            tokenized: list[int] = (
                await self.encode(stop_crit, False)
            ).tolist()
            stop_tokenized.append(tokenized)

        return stop_tokenized

    async def _get_eos_variables(
        self,
        ignore_eos: bool,
        stop_token_ids: list[int] | None,
        stop: list[str] | None,
    ) -> tuple[set[int], list[list[int]]]:
        eos_token_ids = set(self._eos_token_ids)
        eos_sequences = list()

        if ignore_eos:
            eos_token_ids = set()
        elif stop_token_ids:
            eos_token_ids.update(stop_token_ids)
        elif stop:
            eos_sequences = await self._encode_stop_criteria(stop)

        return eos_token_ids, eos_sequences


def _rgba_to_rgb(
    image: Image.Image,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Convert an RGBA image to RGB with filled background color."""
    assert image.mode == "RGBA"
    converted = Image.new("RGB", image.size, background_color)
    converted.paste(image, mask=image.split()[3])  # 3 is the alpha channel
    return converted


def _convert_image_mode(image: Image.Image, to_mode: str):  # noqa: ANN202
    if image.mode == to_mode:
        return image
    elif image.mode == "RGBA" and to_mode == "RGB":
        return _rgba_to_rgb(image)
    else:
        return image.convert(to_mode)
