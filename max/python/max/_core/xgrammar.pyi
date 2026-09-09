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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

"""Bindings for the xgrammar grammar engine (CPU subset)."""

import enum
from collections.abc import Sequence
from typing import Annotated

import numpy
from numpy.typing import NDArray

class VocabType(enum.Enum):
    RAW = 0

    BYTE_FALLBACK = 1

    BYTE_LEVEL = 2

class TokenizerInfo:
    def __init__(
        self,
        encoded_vocab: list,
        vocab_type: VocabType = VocabType.RAW,
        vocab_size: int | None = None,
        stop_token_ids: Sequence[int] | None = None,
        add_prefix_space: bool = False,
        special_token_ids: Sequence[int] | None = None,
    ) -> None: ...
    @property
    def vocab_type(self) -> VocabType: ...
    @property
    def vocab_size(self) -> int: ...
    @property
    def add_prefix_space(self) -> bool: ...
    @property
    def decoded_vocab(self) -> list[str]: ...
    @property
    def stop_token_ids(self) -> list[int]: ...
    @property
    def special_token_ids(self) -> list[int]: ...
    def dump_metadata(self) -> str: ...
    @staticmethod
    def from_vocab_and_metadata(
        encoded_vocab: list, metadata: str
    ) -> TokenizerInfo: ...
    @staticmethod
    def detect_metadata_from_hf(backend_str: str) -> str: ...

class CompiledGrammar:
    @property
    def memory_size_bytes(self) -> int: ...

class GrammarCompiler:
    def __init__(
        self,
        tokenizer_info: TokenizerInfo,
        max_threads: int = 8,
        cache_enabled: bool = True,
        max_memory_bytes: int = -1,
    ) -> None: ...
    def compile_json_schema(
        self,
        schema: str,
        any_whitespace: bool = True,
        strict_mode: bool = True,
        require_object_root: bool = False,
        reject_unsupported: bool = False,
        separators: tuple[str, str] | None = None,
        max_whitespace_cnt: int | None = None,
    ) -> CompiledGrammar: ...
    def compile_grammar(
        self, ebnf_str: str, root_rule_name: str = "root"
    ) -> CompiledGrammar: ...
    def compile_builtin_json_grammar(self) -> CompiledGrammar: ...
    def compile_structural_tag(
        self, structural_tag_json: str
    ) -> CompiledGrammar: ...
    def compile_regex(self, regex: str) -> CompiledGrammar: ...
    def clear_cache(self) -> None: ...
    @property
    def cache_size_bytes(self) -> int: ...
    @property
    def cache_limit_bytes(self) -> int: ...

class GrammarMatcher:
    def __init__(
        self,
        compiled_grammar: CompiledGrammar,
        override_stop_tokens: Sequence[int] | None = None,
        terminate_without_stop_token: bool = False,
        max_rollback_tokens: int = -1,
    ) -> None: ...
    def accept_token(
        self, token_id: int, debug_print: bool = False
    ) -> bool: ...
    def accept_string(
        self, input_str: str, debug_print: bool = False
    ) -> bool: ...
    def fill_next_token_bitmask(
        self,
        bitmask: Annotated[NDArray[numpy.int32], dict(order="C")],
        index: int = 0,
        debug_print: bool = False,
    ) -> bool: ...
    def find_jump_forward_string(self) -> str: ...
    def rollback(self, num_tokens: int = 1) -> None: ...
    def is_terminated(self) -> bool: ...
    def is_completed(self) -> bool: ...
    def reset(self) -> None: ...
    def fork(self) -> GrammarMatcher: ...
    @property
    def max_rollback_tokens(self) -> int: ...
    @property
    def stop_token_ids(self) -> list[int]: ...

def get_bitmask_size(vocab_size: int) -> int:
    """Number of int32 words in a packed bitmask for the given vocab size."""
