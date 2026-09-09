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
"""Cascade worker wrapping a HuggingFace tokenizer for MAX model inference."""

from __future__ import annotations

import importlib
import logging
from collections.abc import AsyncIterable, AsyncIterator, Callable
from contextlib import asynccontextmanager

import numpy as np
import numpy.typing as npt
from max.experimental.cascade.core import worker_method
from max.experimental.cascade.interfaces.gen_ai import (
    ChatMessage,
    GenAITextChunk,
)
from max.experimental.cascade.workers.tokenizer_worker import (
    TokenizerWorker,
    stream_incremental_text,
)
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)
Int32Array = npt.NDArray[np.int32]


def _import_tokenizer_worker_class(
    tokenizer_impl: str,
) -> Callable[[str], TokenizerWorker]:
    """Import the ``TokenizerWorker`` subclass named by *tokenizer_impl*.

    Specified via ``--tokenizer-impl module.path:ClassName``.

    Raises:
        ValueError: If *tokenizer_impl* isn't in ``module:ClassName`` form.
        ImportError: If the module cannot be imported.
        AttributeError: If the module has no attribute named *class_name*.
    """
    module_path, sep, class_name = tokenizer_impl.partition(":")
    if not sep:
        raise ValueError(
            f"tokenizer_impl {tokenizer_impl!r} must be in "
            "'module.path:ClassName' form."
        )
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    assert isinstance(cls, type) and issubclass(cls, TokenizerWorker), (
        f"{tokenizer_impl} is {cls!r}, expected a TokenizerWorker subclass"
    )
    return cls


def make_tokenizer_worker(
    model_path: str, tokenizer_impl: str | None = None
) -> TokenizerWorker:
    """Build the tokenizer worker for *model_path*.

    *tokenizer_impl* names the :class:`TokenizerWorker` subclass to
    construct, as ``"module.path:ClassName"``. Left unset, builds
    :class:`MAXTokenizer` (HuggingFace ``transformers``).
    """
    if tokenizer_impl is None:
        logger.info("No tokenizer specified. Using the HuggingFace tokenizer")
        return MAXTokenizer(model_path)
    try:
        cls = _import_tokenizer_worker_class(tokenizer_impl)
    except (ImportError, AttributeError):
        logger.warning(
            "Tokenizer %r unavailable in this build; using the HuggingFace "
            "tokenizer for %s.",
            tokenizer_impl,
            model_path,
        )
        return MAXTokenizer(model_path)
    return cls(model_path)


def flatten_message(message: ChatMessage) -> dict[str, str]:
    """Render a message as the plain ``{role, content}`` a chat template wants.

    Only text parts survive: HuggingFace chat templates take multimodal content
    in a per-model layout, so a pipeline that accepts images has to build the
    template input itself rather than going through here.

    Raises:
        ValueError: If the message carries a non-text content part.
    """
    text_parts: list[str] = []
    for part in message.content:
        if not isinstance(part, GenAITextChunk):
            raise ValueError(
                f"This tokenizer accepts text content only, got {part.type!r}"
            )
        text_parts.append(part.text)
    return {"role": message.role, "content": "".join(text_parts)}


class MAXTokenizer(TokenizerWorker):
    """Cascade worker that provides Huggingface tokenization."""

    # Unicode replacement character; the HF tokenizer emits it when a byte-level
    # token sequence ends mid-multibyte-character.
    _REPLACEMENT_CHAR = "\ufffd"

    def __init__(self, model_path: str) -> None:
        super().__init__(deploy_hints=["cpu"])
        self.model_path = model_path
        self._tokenizer: PreTrainedTokenizerBase | None = None

    @asynccontextmanager
    async def open(self) -> AsyncIterator[MAXTokenizer]:
        """Load the HuggingFace tokenizer for the worker's lifetime."""
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        yield self

    @worker_method()
    async def encode(
        self, messages: list[ChatMessage]
    ) -> npt.NDArray[np.int32]:
        """Tokenize a conversation into ``int32`` token ids via the chat template."""
        assert self._tokenizer is not None, "MAXTokenizer must be deployed"
        token_ids = self._tokenizer.apply_chat_template(
            [flatten_message(message) for message in messages],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="np",
        )["input_ids"][0]
        return np.asarray(token_ids, dtype=np.int32)

    @worker_method()
    async def decode(
        self, tokens: Int32Array, skip_special_tokens: bool
    ) -> str:
        """Decode ``token`` ids back into text.

        *skip_special_tokens* drops control tokens (e.g. ``<|eot_id|>``) so
        the text matches what max-serve returns rather than leaking them into
        the response.

        This one-shot decode is for non-streaming callers; streaming responses
        should use :meth:`decode_stream`, which handles multibyte characters
        split across chunks.
        """
        assert isinstance(tokens, np.ndarray)
        assert self._tokenizer is not None, "MAXTokenizer must be deployed"
        return self._tokenizer.decode(
            tokens, skip_special_tokens=skip_special_tokens
        )

    @worker_method()
    async def decode_stream(
        self, token_iter: AsyncIterable[Int32Array], skip_special_tokens: bool
    ) -> AsyncIterator[str]:
        """Detokenize a stream of token-id chunks into a stream of text.

        The orchestrator hands this the model worker's token stream, and it
        yields incremental text. When deployed it consumes ``token_iter``
        directly from the model worker (worker-to-worker), so per-token data
        never round-trips through the orchestrator.
        """
        assert self._tokenizer is not None, "MAXTokenizer must be deployed"
        tokenizer = self._tokenizer
        async for text in stream_incremental_text(
            token_iter,
            lambda ids: tokenizer.decode(
                ids, skip_special_tokens=skip_special_tokens
            ),
        ):
            yield text
