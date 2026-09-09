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
"""Call contract and shared streaming decode for the cascade tokenizer workers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator, Callable

import numpy as np
import numpy.typing as npt
from max.experimental.cascade.core import (
    MaybeAsync,
    Result,
    ResultIter,
    Worker,
)
from max.experimental.cascade.interfaces.gen_ai import ChatMessage

Int32Array = npt.NDArray[np.int32]


class TokenizerWorker(Worker, ABC):
    """Base class and call contract shared by the tokenizer workers."""

    model_path: str

    @abstractmethod
    async def encode(
        self, messages: MaybeAsync[list[ChatMessage]], /
    ) -> Result[Int32Array]:
        """Tokenize chat messages into token ids."""
        ...

    @abstractmethod
    async def decode(
        self,
        tokens: MaybeAsync[Int32Array],
        skip_special_tokens: MaybeAsync[bool],
        /,
    ) -> Result[str]:
        """Decode token ids back into text.

        *skip_special_tokens* drops control tokens (e.g. ``<|eot_id|>``) from
        the output; pass ``True`` to match what max-serve returns.
        """
        ...

    @abstractmethod
    async def decode_stream(
        self,
        token_iter: MaybeAsync[AsyncIterable[Int32Array]],
        skip_special_tokens: MaybeAsync[bool],
        /,
    ) -> ResultIter[str]:
        """Detokenize a stream of token-id chunks into a stream of text."""
        ...


# Unicode replacement character emitted when ending a stream mid-character
_REPLACEMENT_CHAR = "\ufffd"


async def stream_incremental_text(
    token_iter: AsyncIterable[Int32Array],
    decode: Callable[[list[int]], str],
) -> AsyncIterator[str]:
    """Detokenize a stream of token-id chunks into a stream of text.

    Offset-based incremental decoding (the approach vLLM/TGI use): a multibyte
    character split across chunks decodes to the Unicode replacement character,
    so emission is deferred until the following chunk(s) complete it. Shared by
    the tokenizer workers; *decode* maps an id window to text for whichever
    backend is in use.
    """
    all_ids: list[int] = []
    # ``prefix_offset``/``read_offset`` bound the window that is re-decoded each
    # step, so cost stays proportional to the un-emitted tail rather than the
    # whole sequence.
    prefix_offset = 0
    read_offset = 0
    async for chunk in token_iter:
        if chunk.size == 0:
            continue
        new_ids = np.asarray(chunk, dtype=np.int32).reshape(-1).tolist()
        all_ids.extend(new_ids)

        prefix_text = decode(all_ids[prefix_offset:read_offset])
        new_text = decode(all_ids[prefix_offset:])

        if len(new_text) > len(prefix_text) and not new_text.endswith(
            _REPLACEMENT_CHAR
        ):
            prefix_offset = read_offset
            read_offset = len(all_ids)
            yield new_text[len(prefix_text) :]

    # Flush any deferred tail: a chunk ending in the replacement char is held
    # back pending completion and would be dropped if the stream ends first.
    # ``read_offset`` trails ``len(all_ids)`` only while a chunk is deferred,
    # so the common case skips the extra decodes.
    if read_offset < len(all_ids):
        final_text = decode(all_ids[prefix_offset:])
        prefix_text = decode(all_ids[prefix_offset:read_offset])
        if len(final_text) > len(prefix_text):
            yield final_text[len(prefix_text) :]
