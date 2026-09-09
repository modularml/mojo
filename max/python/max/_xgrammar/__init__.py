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

"""MAX's hack-to-owned xgrammar Python layer.

Wraps the nanobind grammar-engine bindings in :mod:`max._core.xgrammar` and
restores the upstream xgrammar Python-layer surface MAX's structured-output
backend depends on -- ``TokenizerInfo.from_huggingface``, the ``StructuralTag``
models, ``get_builtin_structural_tag``, and a ``StructuralTag``-aware
``compile_structural_tag``.

``TokenizerInfo`` and ``GrammarCompiler`` are thin typed wrappers (they add
Python-layer behavior); ``CompiledGrammar``, ``GrammarMatcher``, and ``VocabType``
are re-exported from the binding unchanged. The structural-tag modules are
vendored verbatim (pure pydantic, torch-free).
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
from max._core import xgrammar as _core
from max._core.xgrammar import (
    CompiledGrammar as CompiledGrammar,
)
from max._core.xgrammar import (
    GrammarMatcher as GrammarMatcher,
)
from max._core.xgrammar import (
    VocabType as VocabType,
)
from max._core.xgrammar import (
    get_bitmask_size as get_bitmask_size,
)
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from .builtin_structural_tag import (
    get_builtin_structural_tag as _get_builtin_structural_tag,
)
from .structural_tag import StructuralTag, StructuralTagItem

try:
    import sentencepiece
except ImportError:
    sentencepiece = None
try:
    import tiktoken
except ImportError:
    tiktoken = None

_VOCAB_TYPE_BY_VALUE = {
    0: VocabType.RAW,
    1: VocabType.BYTE_FALLBACK,
    2: VocabType.BYTE_LEVEL,
}

_NO_STOP_TOKEN_WARNING = (
    "When constructing TokenizerInfo from a huggingface tokenizer, "
    "stop_token_ids is neither provided by user nor found from the tokenizer. "
    "It will be automatically detected."
)


def _detect_metadata_from_hf(backend_str: str) -> dict[str, Any]:
    metadata = json.loads(
        _core.TokenizerInfo.detect_metadata_from_hf(backend_str)
    )
    return {
        "vocab_type": _VOCAB_TYPE_BY_VALUE[metadata["vocab_type"]],
        "add_prefix_space": metadata["add_prefix_space"],
    }


def _is_tiktoken_tokenizer(tokenizer: PreTrainedTokenizerBase) -> bool:
    if tiktoken is None:
        return False
    has_tiktoken_encoding = hasattr(tokenizer, "tokenizer") and isinstance(
        tokenizer.tokenizer, tiktoken.Encoding
    )
    filename_pattern = (
        hasattr(tokenizer, "vocab_files_names")
        and "vocab_file" in tokenizer.vocab_files_names
        and "tiktoken" in tokenizer.vocab_files_names["vocab_file"]
    )
    return has_tiktoken_encoding or filename_pattern


def _is_byte_level_tokenizer(tokenizer: PreTrainedTokenizerBase) -> bool:
    if tiktoken is None:
        return False
    new_ids = tokenizer.encode(r" ")
    if len(new_ids) < 1:
        return False
    new_tokens = tokenizer.convert_ids_to_tokens(new_ids)
    return new_tokens[0] == "Ġ"


def _is_sentencepiece_tokenizer(tokenizer: PreTrainedTokenizerBase) -> bool:
    if sentencepiece is None:
        return False
    has_sp_model_attr = hasattr(tokenizer, "sp_model") and isinstance(
        tokenizer.sp_model, sentencepiece.SentencePieceProcessor
    )
    has_nested_sp_model_attr = (
        hasattr(tokenizer, "tokenizer")
        and hasattr(tokenizer.tokenizer, "sp_model")
        and isinstance(
            tokenizer.tokenizer.sp_model, sentencepiece.SentencePieceProcessor
        )
    ) or (
        hasattr(tokenizer, "tok")
        and isinstance(tokenizer.tok, sentencepiece.SentencePieceProcessor)
    )
    return has_sp_model_attr or has_nested_sp_model_attr


class TokenizerInfo:
    """Vocabulary + metadata for the grammar engine, over the binding."""

    def __init__(
        self,
        encoded_vocab: Sequence[bytes | str],
        vocab_type: VocabType = VocabType.RAW,
        vocab_size: int | None = None,
        stop_token_ids: list[int] | None = None,
        add_prefix_space: bool = False,
        special_token_ids: list[int] | None = None,
    ) -> None:
        self._impl = _core.TokenizerInfo(
            list(encoded_vocab),
            vocab_type,
            vocab_size,
            stop_token_ids,
            add_prefix_space,
            special_token_ids,
        )

    @property
    def vocab_size(self) -> int:
        return self._impl.vocab_size

    @property
    def vocab_type(self) -> VocabType:
        return self._impl.vocab_type

    @property
    def add_prefix_space(self) -> bool:
        return self._impl.add_prefix_space

    @property
    def decoded_vocab(self) -> list[str]:
        return self._impl.decoded_vocab

    @property
    def stop_token_ids(self) -> list[int]:
        return self._impl.stop_token_ids

    @property
    def special_token_ids(self) -> list[int]:
        return self._impl.special_token_ids

    @staticmethod
    def from_huggingface(
        tokenizer: PreTrainedTokenizerBase,
        *,
        vocab_size: int | None = None,
        stop_token_ids: list[int] | int | None = None,
        special_token_ids: list[int] | None = None,
    ) -> TokenizerInfo:
        """Build a :class:`TokenizerInfo` from a HuggingFace tokenizer.

        Ported from upstream xgrammar ``TokenizerInfo.from_huggingface``,
        re-pointed onto the binding. Pure Python over the tokenizer object --
        no torch.

        ``special_token_ids`` force-marks the given ids special: they are
        excluded from byte-level string-content matching but stay emittable at
        grammar positions that reference them by id (``TokenFormat`` /
        ``ExcludeToken`` / string-value delimiter).
        """
        if isinstance(stop_token_ids, int):
            stop_token_ids = [stop_token_ids]
        if isinstance(stop_token_ids, list) and len(stop_token_ids) == 0:
            raise ValueError("stop_token_ids cannot be empty")

        try:
            vocab_dict = tokenizer.get_vocab()
        except AttributeError as e:
            raise ValueError(
                f"Cannot get the vocabulary of the tokenizer {type(tokenizer)}."
                " The tokenizer should have a get_vocab method."
            ) from e

        # max_id can exceed len(vocab) when ids are sparse; size to fit both.
        max_id = max(vocab_dict.values())
        tokenizer_vocab_size = max(len(vocab_dict), max_id + 1)
        vocab_size = vocab_size or tokenizer_vocab_size

        encoded_vocab: list[bytes | str] = [""] * vocab_size
        for token, idx in vocab_dict.items():
            if idx < vocab_size:
                encoded_vocab[idx] = token

        if isinstance(tokenizer, PreTrainedTokenizerFast):
            backend_str = tokenizer.backend_tokenizer.to_str()
            if stop_token_ids is None:
                eos = getattr(tokenizer, "eos_token_id", None)
                if eos is not None:
                    stop_token_ids = [eos]
                else:
                    warnings.warn(_NO_STOP_TOKEN_WARNING)
            metadata = _detect_metadata_from_hf(backend_str)
            return TokenizerInfo(
                encoded_vocab,
                vocab_type=metadata["vocab_type"],
                vocab_size=vocab_size,
                stop_token_ids=stop_token_ids,
                add_prefix_space=metadata["add_prefix_space"],
                special_token_ids=special_token_ids,
            )

        if _is_tiktoken_tokenizer(tokenizer):
            if stop_token_ids is None:
                eos = getattr(tokenizer, "eos_token_id", None)
                if eos is not None:
                    stop_token_ids = [eos]
                else:
                    warnings.warn(_NO_STOP_TOKEN_WARNING)
            vocab_type = VocabType.RAW
            if _is_byte_level_tokenizer(tokenizer):
                vocab_type = VocabType.BYTE_LEVEL
            return TokenizerInfo(
                encoded_vocab,
                vocab_type,
                vocab_size=vocab_size,
                stop_token_ids=stop_token_ids,
                add_prefix_space=False,
                special_token_ids=special_token_ids,
            )

        if _is_sentencepiece_tokenizer(tokenizer):
            sp_model = (
                getattr(tokenizer, "sp_model", None)
                or getattr(
                    getattr(tokenizer, "tokenizer", None), "sp_model", None
                )
                or getattr(tokenizer, "tok", None)
            )
            assert sp_model is not None
            if stop_token_ids is None:
                eos = getattr(tokenizer, "eos_token_id", None)
                if eos is not None:
                    stop_token_ids = [eos]
                else:
                    sp_eos = sp_model.eos_id()
                    if sp_eos != -1:
                        stop_token_ids = [sp_eos]
                    else:
                        warnings.warn(_NO_STOP_TOKEN_WARNING)
            vocab_type = (
                VocabType.BYTE_FALLBACK
                if "<0x0A>" in vocab_dict
                else VocabType.RAW
            )
            return TokenizerInfo(
                encoded_vocab,
                vocab_type=vocab_type,
                vocab_size=vocab_size,
                stop_token_ids=stop_token_ids,
                add_prefix_space=True,
                special_token_ids=special_token_ids,
            )

        raise ValueError(f"Unsupported tokenizer type: {type(tokenizer)}")


class GrammarCompiler:
    """Compiles JSON schemas / grammars / structural tags, over the binding."""

    def __init__(
        self,
        tokenizer_info: TokenizerInfo,
        max_threads: int = 8,
        cache_enabled: bool = True,
        max_memory_bytes: int = -1,
    ) -> None:
        self._impl = _core.GrammarCompiler(
            tokenizer_info._impl,
            max_threads,
            cache_enabled,
            max_memory_bytes,
        )

    def compile_json_schema(
        self,
        schema: str,
        any_whitespace: bool = True,
        strict_mode: bool = True,
        require_object_root: bool = False,
        reject_unsupported: bool = False,
        separators: tuple[str, str] | None = None,
        max_whitespace_cnt: int | None = None,
    ) -> CompiledGrammar:
        return self._impl.compile_json_schema(
            schema,
            any_whitespace=any_whitespace,
            strict_mode=strict_mode,
            require_object_root=require_object_root,
            reject_unsupported=reject_unsupported,
            separators=separators,
            max_whitespace_cnt=max_whitespace_cnt,
        )

    def compile_grammar(
        self, ebnf_str: str, root_rule_name: str = "root"
    ) -> CompiledGrammar:
        return self._impl.compile_grammar(ebnf_str, root_rule_name)

    def compile_builtin_json_grammar(self) -> CompiledGrammar:
        return self._impl.compile_builtin_json_grammar()

    def compile_structural_tag(
        self, tag: StructuralTag | str
    ) -> CompiledGrammar:
        if isinstance(tag, StructuralTag):
            tag = tag.model_dump_json()
        return self._impl.compile_structural_tag(tag)

    def compile_regex(self, regex: str) -> CompiledGrammar:
        return self._impl.compile_regex(regex)

    def clear_cache(self) -> None:
        self._impl.clear_cache()


def allocate_token_bitmask(
    batch_size: int, vocab_size: int
) -> npt.NDArray[np.int32]:
    """Allocate a packed ``[batch_size, ceil(vocab_size/32)]`` int32 bitmask.

    ``-1`` (all bits set) means unconstrained, matching MAX's and llguidance's
    convention; ``fill_next_token_bitmask`` overwrites filled rows. Torch-free
    (numpy), unlike upstream's torch-tensor allocator.
    """
    return np.full(
        (batch_size, get_bitmask_size(vocab_size)), -1, dtype=np.int32
    )


def get_builtin_structural_tag(
    model: str,
    tools: Sequence[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = "auto",
    reasoning: bool = True,
) -> StructuralTag:
    # Permissive typed boundary: callers pass OpenAI-style tool dicts and a
    # string/dict tool_choice; the vendored helper's params are stricter.
    return _get_builtin_structural_tag(
        model,
        tools=tools,
        tool_choice=tool_choice,
        reasoning=reasoning,
    )


__all__ = [
    "CompiledGrammar",
    "GrammarCompiler",
    "GrammarMatcher",
    "StructuralTag",
    "StructuralTagItem",
    "TokenizerInfo",
    "VocabType",
    "allocate_token_bitmask",
    "get_bitmask_size",
    "get_builtin_structural_tag",
]
