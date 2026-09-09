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
"""No-network tests for ``MAXTokenizer.decode_stream``.

Exercises the offset-based incremental detokenizer with a fake tokenizer that
models multibyte-UTF-8 splits (partial byte sequences surface as U+FFFD),
so the streaming behaviour can be verified without downloading a real
tokenizer. Focuses on the end-of-stream flush.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import cast

import numpy as np
import pytest
from max.experimental.cascade.workers.max_tokenizer import MAXTokenizer
from max.experimental.cascade.workers.tokenizer_worker import (
    _REPLACEMENT_CHAR,
)
from transformers import PreTrainedTokenizerBase

# Byte fragments per token id. The 4-byte emoji U+1F600 (b"\xf0\x9f\x98\x80")
# is split across tokens 3 and 4, so decoding token 3 alone yields U+FFFD.
_FRAGMENTS: dict[int, bytes] = {
    1: b"ab",
    2: b"c",
    3: b"\xf0\x9f",
    4: b"\x98\x80",
}


class _FakeTokenizer:
    """Decodes token ids by concatenating byte fragments as UTF-8.

    Incomplete trailing sequences decode to the replacement character, exactly
    as a real byte-level tokenizer surfaces a multibyte character split across
    chunk boundaries.
    """

    def decode(
        self, ids: Sequence[int], skip_special_tokens: bool = False
    ) -> str:
        joined = b"".join(_FRAGMENTS[int(i)] for i in ids)
        return joined.decode("utf-8", errors="replace")


def _tokenizer() -> MAXTokenizer:
    tok = MAXTokenizer("fake/model")
    # Inject the fake decoder; a real tokenizer would need a network download.
    tok._tokenizer = cast(PreTrainedTokenizerBase, _FakeTokenizer())
    return tok


async def _decode_stream(
    tok: MAXTokenizer, chunks: Sequence[Sequence[int]]
) -> list[str]:
    async def token_iter() -> AsyncIterator[np.ndarray]:
        for chunk in chunks:
            yield np.array(chunk, dtype=np.int32)

    # A streaming worker_method returns the async iterator when called directly
    # on the instance (the proxy path returns a ResultIter handle instead).
    stream = cast("AsyncIterator[str]", tok.decode_stream(token_iter(), True))
    return [text async for text in stream]


@pytest.mark.asyncio
async def test_decode_stream_flushes_complete_tail_at_end() -> None:
    """A final chunk carrying complete text plus an incomplete multibyte start
    is not dropped: the flush emits the complete remainder.

    The last chunk ``[2, 3]`` decodes to ``"c" + U+FFFD``; the loop defers it
    because it ends in the replacement char, and without the end-of-stream
    flush the complete ``"c"`` would be lost.
    """
    tok = _tokenizer()
    pieces = await _decode_stream(tok, [[1], [2, 3]])

    joined = "".join(pieces)
    # Matches the one-shot decode of the full sequence (which includes the
    # trailing replacement char for the unfinished emoji).
    assert joined == "abc" + _REPLACEMENT_CHAR
    assert "c" in joined


@pytest.mark.asyncio
async def test_decode_stream_joins_multibyte_split_across_chunks() -> None:
    """A multibyte character split across chunks is emitted once, intact."""
    tok = _tokenizer()
    pieces = await _decode_stream(tok, [[1], [3], [4]])

    joined = "".join(pieces)
    assert joined == "ab\U0001f600"
    assert _REPLACEMENT_CHAR not in joined
    # The emoji is emitted exactly once (not partially, per chunk).
    assert joined.count("\U0001f600") == 1
