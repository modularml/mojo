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
"""Prompt assembly for MiniMax Music 3.

The assembled text is part of the checkpoint's contract: the reference cleans
the caption and the lyrics with the substitutions below and wraps them in
special tokens, and a change as small as one space changes the audio. So this
mirrors the reference's normalization rather than improving on it.
"""

from __future__ import annotations

import re

import numpy as np
import numpy.typing as npt
from max.pipelines.lib.audio_tokenizer import AudioGenerationTokenizer

from .denoise import NUM_INFERENCE_STEPS
from .model_config import LanguageModelConfig

_IM_START, _IM_END = "<|im_start|>", "<|im_end|>"
_CAPTION_START, _CAPTION_END = "<|caption_start|>", "<|caption_end|>"
_LYRICS_START, _LYRICS_END = "<|lyrics_start|>", "<|lyrics_end|>"
_AUDIO_START = "<|audio_start|>"

MAX_PROMPT_TOKENS = 5_000

DEFAULT_AUDIO_DURATION = 60.0
"""Seconds to render when the request does not ask for a length."""

_SPECIAL_TAG_RE = re.compile(r"<\|([^|]*)\|>")
_LEADING_TAGS_RE = re.compile(r"^[ \t]*((?:\[[^\]]+\][ \t]*)+)")


def clean_caption(caption: str) -> str:
    """Strips the markdown the checkpoint's input contract accepts.

    Also rewrites a ``<|key value|>`` tag as ``key is value``, which is how the
    reference lets a caption name an attribute without a special token
    surviving into the prompt.

    Args:
        caption: The music description as written by the caller.

    Returns:
        The cleaned caption.
    """

    def _rewrite_special_tag(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        parts = inner.split(None, 1)
        return f"{parts[0]} is {parts[1]}" if len(parts) == 2 else inner

    text = _SPECIAL_TAG_RE.sub(_rewrite_special_tag, caption)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*[*+-]\s+", "", line)
        line = re.sub(r"^\s*\*\s+", "", line)
        while "**" in line:
            unbolded = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
            if unbolded == line:
                break
            line = unbolded
        line = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", line)
        lines.append(line.rstrip())
    text = "\n".join(lines)
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    text = text.replace("\u2022 ", "").replace("    ", "")
    return re.sub(r"\n{2,}", "\n", text)


def normalize_lyrics(lyrics: str) -> str:
    """Puts lyrics in the one-tag-per-line form the checkpoint was trained on.

    A line that starts with structure tags keeps only those tags, so any words
    sharing a line with ``[verse]`` are dropped -- surprising, but what the
    reference does, and the model sings the difference.

    Args:
        lyrics: The lyrics as written by the caller.

    Returns:
        The normalized lyrics, opened with a ``[start]`` tag.
    """
    tagged_lines = []
    for line in lyrics.split("\n"):
        match = _LEADING_TAGS_RE.match(line)
        tagged_lines.append(match.group(1).strip() if match else line)
    text = "\n".join(tagged_lines)
    text = text.replace("] ", "]\n")
    text = text.replace(" [", "\n[")
    text = text.replace(" ^ ", "\n")
    text = re.sub(r"\[([^\]]+)\]", lambda m: f"[{m.group(1).lower()}]", text)
    return f"[start]\n{text}"


def assemble_prompt(description: str, lyrics: str) -> str:
    """Returns the checkpoint's prompt for a description and its lyrics.

    Args:
        description: What the music should sound like.
        lyrics: What it should sing.

    Returns:
        The assembled prompt, ending on the token that opens the audio.
    """
    return (
        f"{_IM_START}{_CAPTION_START}{clean_caption(description)}{_CAPTION_END}"
        f"{_LYRICS_START}{normalize_lyrics(lyrics)}{_LYRICS_END}"
        f"{_IM_END}{_AUDIO_START}"
    )


class MiniMaxMusic3Tokenizer(AudioGenerationTokenizer):
    """Builds the conditional and unconditional prompts for MiniMax Music 3.

    Args:
        args: Forwarded to :class:`AudioGenerationTokenizer`.
        kwargs: Forwarded to :class:`AudioGenerationTokenizer`, with
            ``max_length`` defaulting to the checkpoint's prompt limit and the
            duration and step count to this checkpoint's own.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("max_length", MAX_PROMPT_TOKENS)
        kwargs.setdefault("default_audio_duration", DEFAULT_AUDIO_DURATION)
        kwargs.setdefault("default_num_inference_steps", NUM_INFERENCE_STEPS)
        super().__init__(*args, **kwargs)
        self._audio_cfg_token_id = LanguageModelConfig().audio_cfg_token_id

    def assemble_prompt(self, description: str, lyrics: str | None) -> str:
        """Returns the assembled prompt.

        Args:
            description: The request's prompt.
            lyrics: The lyrics to sing.

        Returns:
            The assembled prompt.

        Raises:
            ValueError: If the request carried no lyrics. This model always
                sings, and an empty lyric field makes it hum through a prompt
                the checkpoint never saw in training.
        """
        if lyrics is None or not lyrics.strip():
            raise ValueError(
                "MiniMax Music 3 generates sung music and needs lyrics. Pass "
                "them as the 'lyrics' audio provider option, with structure "
                "tags such as '[verse]' each on their own line."
            )
        return assemble_prompt(description, lyrics)

    def unconditional_ids(
        self, token_ids: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.int64]:
        """Returns the prompt with its content replaced by the CFG token.

        Everything between the opening ``<|im_start|>`` and the closing
        ``<|im_end|><|audio_start|>`` becomes one repeated token, so the
        unconditional prompt is the same length as the conditional prompt.
        The two prompts are batched together.

        Args:
            token_ids: The conditional prompt's token ids.

        Returns:
            The unconditional prompt's token ids.
        """
        unconditional = token_ids.copy()
        unconditional[1:-2] = self._audio_cfg_token_id
        return unconditional
