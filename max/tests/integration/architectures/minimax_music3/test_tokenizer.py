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
"""The prompt contract, which is part of the checkpoint rather than of the port.

The reference normalizes the caption and the lyrics before tokenizing, and the
model was trained on the result: a change as small as one space changes the
audio. So these pin the reference's behavior -- including the parts of it that
are surprising -- rather than any behavior that seems more reasonable.

Text only, so no GPU, no weights and no tokenizer download.
"""

from __future__ import annotations

import numpy as np
import pytest
from max.pipelines.architectures.minimax_music3.model_config import (
    LanguageModelConfig,
)
from max.pipelines.architectures.minimax_music3.tokenizer import (
    MiniMaxMusic3Tokenizer,
    assemble_prompt,
    clean_caption,
    normalize_lyrics,
)


def test_clean_caption_strips_markdown() -> None:
    caption = "## Heading\n- bullet\n**bold** and *italic*"
    assert clean_caption(caption) == "Heading\nbullet\nbold and italic"


def test_clean_caption_rewrites_a_tag_as_a_sentence() -> None:
    """``<|key value|>`` is how a caption names an attribute without a special
    token surviving into the prompt."""
    assert clean_caption("<|tempo 120 bpm|>") == "tempo is 120 bpm"
    assert clean_caption("<|instrumental|>") == "instrumental"


def test_clean_caption_collapses_blank_lines() -> None:
    assert clean_caption("a\n\n\nb") == "a\nb"


def test_normalize_lyrics_opens_with_a_start_tag() -> None:
    assert normalize_lyrics("hello").startswith("[start]\n")


def test_normalize_lyrics_lowercases_structure_tags() -> None:
    assert normalize_lyrics("[Verse]\nsing") == "[start]\n[verse]\nsing"


def test_normalize_lyrics_drops_words_sharing_a_line_with_a_tag() -> None:
    """Surprising, and deliberate: the reference keeps only the leading tags of
    a line that starts with one, so the words beside them are lost. The model
    sings the difference, so a port that 'fixed' this would be wrong."""
    assert normalize_lyrics("[verse] sing this") == "[start]\n[verse]"


def test_normalize_lyrics_splits_tags_onto_their_own_lines() -> None:
    assert normalize_lyrics("la la\n[chorus]") == "[start]\nla la\n[chorus]"


def test_assemble_prompt_ends_on_the_audio_token() -> None:
    """The prompt stops exactly where generation begins."""
    prompt = assemble_prompt("a waltz", "[verse]\nla")
    assert prompt.endswith("<|im_end|><|audio_start|>")
    assert prompt.startswith("<|im_start|><|caption_start|>")


def test_assemble_prompt_wraps_both_fields() -> None:
    prompt = assemble_prompt("a waltz", "la")
    assert "<|caption_start|>a waltz<|caption_end|>" in prompt
    assert "<|lyrics_start|>[start]\nla<|lyrics_end|>" in prompt


def _tokenizer() -> MiniMaxMusic3Tokenizer:
    """A tokenizer with only the field the masking reads.

    ``__init__`` would download the checkpoint's text tokenizer, and none of
    that is involved in deciding which positions the unconditional row masks.
    """
    tokenizer = object.__new__(MiniMaxMusic3Tokenizer)
    tokenizer._audio_cfg_token_id = LanguageModelConfig().audio_cfg_token_id
    return tokenizer


def test_unconditional_ids_keep_the_frame() -> None:
    """Guidance replaces the content and nothing else: the two rows are batched
    together and share a position grid, so they must stay the same length."""
    conditional = np.arange(10, dtype=np.int64)
    unconditional = _tokenizer().unconditional_ids(conditional)

    assert unconditional.shape == conditional.shape
    assert unconditional[0] == conditional[0]
    np.testing.assert_array_equal(unconditional[-2:], conditional[-2:])


def test_unconditional_ids_mask_the_content() -> None:
    conditional = np.arange(10, dtype=np.int64)
    cfg = LanguageModelConfig().audio_cfg_token_id
    np.testing.assert_array_equal(
        _tokenizer().unconditional_ids(conditional)[1:-2], [cfg] * 7
    )


def test_unconditional_ids_do_not_alias_the_input() -> None:
    conditional = np.arange(10, dtype=np.int64)
    _tokenizer().unconditional_ids(conditional)
    np.testing.assert_array_equal(conditional, np.arange(10))


def test_assemble_prompt_requires_lyrics() -> None:
    """This model always sings; an empty lyric field makes it hum through a
    prompt the checkpoint never saw."""
    tokenizer = _tokenizer()
    for empty in (None, "", "   "):
        with pytest.raises(ValueError, match="needs lyrics"):
            tokenizer.assemble_prompt("a waltz", empty)
