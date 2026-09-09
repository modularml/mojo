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
"""Tests for structured output support with TikToken-based tokenizers.

Verifies that the _TikTokenAdapter correctly wraps TikToken tokenizers
(like Kimi K2.7's TikTokenTokenizer) for use with llguidance structured
output / grammar-guided decoding.
"""

import json

import llguidance
import llguidance.numpy
import pytest
from llguidance import LLMatcher, LLTokenizer
from llguidance._tokenizer import TokenizerWrapper
from max.pipelines.lib.pipeline_variants.structured_output_backend import (
    _TikTokenAdapter,
)
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)

KIMI_K27_HF_REPO_ID = "nvidia/Kimi-K2.7-Code-NVFP4"


@pytest.fixture(scope="module")
def kimi_tokenizer() -> PreTrainedTokenizerBase:
    """Load Kimi K2.7 tokenizer (TikTokenTokenizer)."""
    return AutoTokenizer.from_pretrained(
        KIMI_K27_HF_REPO_ID,
        trust_remote_code=True,
    )


def test_tiktoken_adapter_accepts_kimi_tokenizer(
    kimi_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Verify _TikTokenAdapter successfully wraps Kimi's tokenizer."""

    # First, verify Kimi K2.7 uses TikTokenTokenizer, not PreTrainedTokenizerFast.
    assert "TikToken" in type(kimi_tokenizer).__name__
    assert not isinstance(kimi_tokenizer, PreTrainedTokenizerFast)

    adapter = _TikTokenAdapter(kimi_tokenizer)

    assert adapter.eos_token_id == kimi_tokenizer.eos_token_id
    assert adapter.bos_token_id == kimi_tokenizer.bos_token_id
    assert len(adapter.tokens) == len(kimi_tokenizer)
    assert all(isinstance(t, bytes) for t in adapter.tokens)


def test_tiktoken_adapter_encodes_special_tokens(
    kimi_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Verify adapter correctly encodes text containing special tokens."""
    adapter = _TikTokenAdapter(kimi_tokenizer)

    # Encode text with a special token
    text_with_special = "Hello <|im_end|> world"
    encoded = adapter(text_with_special)

    # Verify encoding worked and special token was recognized
    assert isinstance(encoded, list)
    assert len(encoded) > 0

    # Decode and verify round-trip
    decoded = kimi_tokenizer.decode(encoded)
    assert "<|im_end|>" in decoded or "im_end" in decoded


def test_tiktoken_adapter_handles_bytes_input(
    kimi_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Verify adapter correctly handles bytes input."""
    adapter = _TikTokenAdapter(kimi_tokenizer)

    text = "Hello world"
    encoded_from_str = adapter(text)
    encoded_from_bytes = adapter(text.encode("utf-8"))

    assert encoded_from_str == encoded_from_bytes


def test_llguidance_integration_with_kimi(
    kimi_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Verify full llguidance integration works with Kimi's TikToken tokenizer."""
    # Create the adapter chain
    adapter = _TikTokenAdapter(kimi_tokenizer)
    wrapper = TokenizerWrapper(adapter)
    vocab_size = len(kimi_tokenizer)
    ll_tokenizer = LLTokenizer(wrapper, n_vocab=vocab_size)

    # Create a JSON schema grammar
    json_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }

    grammar = LLMatcher.grammar_from_json_schema(json.dumps(json_schema))
    matcher = LLMatcher(ll_tokenizer, grammar)

    # Allocate and fill bitmask
    bitmask = llguidance.numpy.allocate_token_bitmask(1, vocab_size)
    llguidance.numpy.fill_next_token_bitmask(matcher, bitmask, index=0)

    # Verify bitmask has expected shape and contains data
    assert bitmask.shape == (1, (vocab_size + 31) // 32)
    # Bitmask should have some bits set (allowed tokens for JSON start)
    assert bitmask.any()


def test_tiktoken_adapter_recovers_raw_control_bytes(
    kimi_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Regression: control-char tokens must map to their TRUE bytes.

    Byte-level BPE renders a raw newline (0x0A) as the surface char 'Ċ'
    (U+010A); ``convert_ids_to_tokens(i).encode("utf-8")`` would yield
    ``b"\\xc4\\x8a"`` (no control byte), making llguidance mask against the
    wrong bytes and admit raw newlines into JSON strings. The adapter must
    reverse the byte->unicode map so the token's bytes contain the raw 0x0A.
    """
    adapter = _TikTokenAdapter(kimi_tokenizer)

    checked = 0
    for tid in range(min(len(kimi_tokenizer), 10000)):
        surface = kimi_tokenizer.convert_ids_to_tokens(tid)
        # 'Ċ' (U+010A) is the byte->unicode surface form of raw newline 0x0A.
        if surface is None or "Ċ" not in surface:
            continue
        assert b"\n"[0] in adapter.tokens[tid], (
            f"token {tid} ({surface!r}) lost its raw newline: "
            f"{adapter.tokens[tid]!r}"
        )
        checked += 1
        if checked >= 20:
            break
    assert checked > 0, "expected some newline-bearing tokens in the vocab"


def test_grammar_rejects_raw_newline_in_json_string(
    kimi_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """End-to-end: the JSON-schema grammar rejects a raw newline mid-string.

    With the corrected token bytes, llguidance must mask out a newline-bearing
    token inside a string value (strict JSON requires it escaped as ``\\n``).
    """
    adapter = _TikTokenAdapter(kimi_tokenizer)
    wrapper = TokenizerWrapper(adapter)
    vocab_size = len(kimi_tokenizer)
    ll_tokenizer = LLTokenizer(wrapper, n_vocab=vocab_size)

    schema = {
        "type": "object",
        "properties": {"reasoning": {"type": "string"}},
        "required": ["reasoning"],
        "additionalProperties": False,
    }
    grammar = LLMatcher.grammar_from_json_schema(
        json.dumps(schema), overrides={"whitespace_pattern": ""}
    )
    matcher = LLMatcher(ll_tokenizer, grammar)

    # Drive the matcher into the open string value.
    prefix = kimi_tokenizer.encode('{"reasoning":"a', allow_special_tokens=True)
    assert matcher.try_consume_tokens(prefix) == len(prefix)

    # A token whose true bytes are a raw newline must be rejected here.
    newline_tok = next(
        tid
        for tid in range(min(vocab_size, 10000))
        if kimi_tokenizer.convert_ids_to_tokens(tid) == "Ċ"
    )
    assert matcher.try_consume_tokens([newline_tok]) == 0


def test_tiktoken_adapter_rejects_non_tiktoken() -> None:
    """Verify _TikTokenAdapter raises ValueError for non-TikToken tokenizers."""
    # Load a fast tokenizer (not TikToken)
    fast_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    assert isinstance(fast_tokenizer, PreTrainedTokenizerFast)

    with pytest.raises(ValueError, match="Structured output requires"):
        _TikTokenAdapter(fast_tokenizer)


class _FakeTikTokenTokenizer:
    """Minimal TikToken-named tokenizer for exercising _TikTokenAdapter's
    byte_decoder guard and fallback as fast, ungated unit tests.

    The class name must contain "TikToken" to pass the adapter's type gate.
    """

    eos_token_id: int = 0
    bos_token_id: int | None = None
    all_special_ids: list[int] = []

    def __init__(
        self,
        vocab: list[str],
        byte_decoder: dict[str, int] | None = None,
    ) -> None:
        self._vocab = vocab
        if byte_decoder is not None:
            self.byte_decoder = byte_decoder

    def get_vocab(self) -> dict[str, int]:
        return {tok: i for i, tok in enumerate(self._vocab)}

    def convert_ids_to_tokens(self, i: int) -> str:
        return self._vocab[i]


def test_tiktoken_adapter_requires_byte_decoder() -> None:
    """A TikToken tokenizer without a byte_decoder is rejected (fail-fast),
    rather than silently using the wrong surface-form bytes."""
    fake = _FakeTikTokenTokenizer(vocab=["a"])  # no byte_decoder attr
    with pytest.raises(ValueError, match="byte_decoder"):
        _TikTokenAdapter(fake)


def test_tiktoken_adapter_maps_bytes_and_falls_back() -> None:
    """byte_decoder is applied to recover true bytes; a surface char absent
    from the map falls back to UTF-8 instead of raising."""
    # 'A' is mapped to the newline byte to prove the map is applied (not the
    # char's own codepoint); 'B' is absent, exercising the KeyError fallback.
    fake = _FakeTikTokenTokenizer(vocab=["A", "B"], byte_decoder={"A": 0x0A})
    adapter = _TikTokenAdapter(fake)
    assert adapter.tokens[0] == b"\n"  # byte_decoder mapping applied
    assert adapter.tokens[1] == b"B"  # KeyError -> UTF-8 fallback


def test_tiktoken_adapter_recovers_exact_true_bytes(
    kimi_tokenizer: PreTrainedTokenizerBase,
) -> None:
    """Control/whitespace tokens decode to their exact true bytes, and a
    normal token is unchanged (the fix rewrites all token byte representations).
    """
    adapter = _TikTokenAdapter(kimi_tokenizer)
    # Invert the tokenizer's own byte->unicode map: byte value -> surrogate char.
    surrogate = {b: c for c, b in kimi_tokenizer.byte_decoder.items()}
    unk = kimi_tokenizer.unk_token_id

    checked = 0
    for byte_val in (0x0A, 0x20, 0x09, 0x0D):  # newline, space, tab, CR
        tid = kimi_tokenizer.convert_tokens_to_ids(surrogate[byte_val])
        if tid is None or tid == unk:
            continue
        assert adapter.tokens[tid] == bytes([byte_val]), (
            f"byte {byte_val:#x} (id {tid}) -> {adapter.tokens[tid]!r}"
        )
        checked += 1
    assert checked > 0, "expected at least one control/whitespace token"

    # A normal multi-byte word token is recovered unchanged.
    normal = kimi_tokenizer.convert_tokens_to_ids("Reserved")
    if normal is not None and normal != unk:
        assert adapter.tokens[normal] == b"Reserved"
