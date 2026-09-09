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

"""Smoke test for the hack-to-owned xgrammar binding.

The xgrammar grammar engine is vendored as a bazel ``cc_library`` (``@xgrammar``)
and bound with nanobind into ``max._core.xgrammar``. MAX deliberately drops the
upstream torch GPU bitmask-apply path, so neither the binding nor its build
closure should pull in torch. These tests run in the bazel sandbox, whose
runfiles contain only declared deps, so the absence of torch from the closure is
a meaningful assertion.
"""

import importlib.util
import sys

import numpy as np
from max._core import xgrammar as xgr


def test_binding_is_torch_free() -> None:
    assert "torch" not in sys.modules
    assert importlib.util.find_spec("torch") is None


def test_byte_vocab_construction() -> None:
    # tiktoken / byte-level vocabs (e.g. Kimi) feed tokens as raw bytes, not str.
    byte_vocab = [b"{", b"}", b'"', b":", b" ", b"a", b"1", b"<eos>"]
    info = xgr.TokenizerInfo(
        byte_vocab,
        vocab_type=xgr.VocabType.RAW,
        stop_token_ids=[len(byte_vocab) - 1],
    )
    assert info.vocab_size == len(byte_vocab)
    mixed = xgr.TokenizerInfo([b"a", "b"], vocab_type=xgr.VocabType.RAW)
    assert mixed.vocab_size == 2


def test_json_schema_bitmask_roundtrip() -> None:
    vocab = [
        "{",
        "}",
        "[",
        "]",
        '"',
        ":",
        ",",
        " ",
        "a",
        "b",
        "1",
        "true",
        "null",
        "<eos>",
    ]
    tokenizer_info = xgr.TokenizerInfo(
        vocab,
        vocab_type=xgr.VocabType.RAW,
        stop_token_ids=[len(vocab) - 1],
    )
    assert tokenizer_info.vocab_size == len(vocab)

    compiler = xgr.GrammarCompiler(tokenizer_info)
    compiled = compiler.compile_json_schema('{"type": "object"}')
    matcher = xgr.GrammarMatcher(compiled)

    size = xgr.get_bitmask_size(tokenizer_info.vocab_size)
    assert size == 1  # ceil(14 / 32)

    bitmask = np.full((size,), -1, dtype=np.int32)
    needs_apply = matcher.fill_next_token_bitmask(bitmask)

    # At the start of a JSON object only "{" (and optionally whitespace) is
    # legal, so the mask is a real constraint, not all-ones (-1).
    assert needs_apply is True
    assert bitmask.dtype == np.int32
    assert int(bitmask[0]) != -1

    assert matcher.accept_token(0) is True
    matcher.rollback(1)
    matcher.reset()
    assert matcher.is_terminated() is False
