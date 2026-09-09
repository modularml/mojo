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
"""Audio generation tokenizer base class."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from max.pipelines.context import AudioContext, TokenBuffer
from max.pipelines.context.exceptions import PromptTooLongError
from max.pipelines.lib.request_text import retrieve_input_text
from max.pipelines.lib.tokenizer import run_with_default_executor
from max.pipelines.modeling.types import PipelineTokenizer
from max.pipelines.request import OpenResponsesRequest
from max.pipelines.request.provider_options import AudioProviderOptions
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig


class AudioGenerationTokenizer(
    PipelineTokenizer[AudioContext, npt.NDArray[np.int64], OpenResponsesRequest]
):
    """Turns an audio generation request into an :class:`AudioContext`.

    Audio checkpoints assemble their prompt from more than the request's text
    -- lyrics, a caption, and special tokens that mark each part -- and that
    assembly is part of the checkpoint's contract, so a subclass owns it via
    :meth:`assemble_prompt`. A subclass that guides also overrides
    :meth:`unconditional_ids` to say what its unconditional prompt is.

    Args:
        model_path: Path to the model repository.
        pipeline_config: The resolved pipeline configuration.
        subfolder: Subfolder of the repository holding the tokenizer.
        revision: Git revision of the repository to load.
        max_length: Maximum length of the assembled prompt, in tokens.
        trust_remote_code: Whether to run tokenizer code from the repository.
        default_audio_duration: Duration in seconds to generate when the
            request does not ask for one. Required of the subclass: how long a
            checkpoint renders by default is the checkpoint's property, and a
            number chosen here would silently become every model's.
        default_num_inference_steps: Denoising steps to take when the request
            does not ask for a count, on the same terms.
    """

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        subfolder: str = "tokenizer",
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        default_audio_duration: float,
        default_num_inference_steps: int,
        **unused_kwargs: Any,
    ) -> None:
        self.model_path = model_path
        self.max_length = max_length
        self._default_audio_duration = default_audio_duration
        self._default_num_inference_steps = default_num_inference_steps

        try:
            self.delegate = AutoTokenizer.from_pretrained(
                model_path,
                subfolder=subfolder,
                revision=revision,
                trust_remote_code=trust_remote_code,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to load the tokenizer in '{subfolder}' of "
                f"{model_path}. This can happen if the subfolder is wrong, if "
                "its files are missing, or if '--trust-remote-code' is needed "
                "but not set."
            ) from e

        # Fast tokenizers are not thread safe, and encode runs in a thread pool.
        self._lock = threading.Lock()

    def assemble_prompt(self, description: str, lyrics: str | None) -> str:
        """Returns the text to tokenize, in the checkpoint's prompt format.

        Args:
            description: The request's prompt, describing the audio to make.
            lyrics: Lyrics to sing, if the request carried any.

        Returns:
            The assembled prompt.
        """
        raise NotImplementedError

    def unconditional_ids(
        self, token_ids: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.int64] | None:
        """Returns the unconditional counterpart of ``token_ids``, or None.

        None means the model does not guide, and leaves
        :attr:`AudioContext.negative_tokens` unset.

        Args:
            token_ids: The conditional prompt's token ids.
        """
        return None

    @property
    def eos_token_ids(self) -> set[int]:
        """Returns the end-of-sequence token ids of the prompt tokenizer.

        These terminate the text prompt, not the generated audio: an audio
        model stops on a token of its own that the executor owns.
        """
        eos = self.delegate.eos_token_id
        return {eos} if eos is not None else set()

    @property
    def expects_content_wrapping(self) -> bool:
        """Returns False: audio requests carry a prompt, not chat messages."""
        return False

    async def encode(
        self, prompt: str, add_special_tokens: bool = True
    ) -> npt.NDArray[np.int64]:
        """Tokenizes an already-assembled prompt.

        Args:
            prompt: The assembled prompt.
            add_special_tokens: Whether the tokenizer adds its own special
                tokens. Assembled prompts usually carry theirs already.

        Returns:
            The prompt's token ids.

        Raises:
            PromptTooLongError: If the prompt is longer than ``max_length``.
        """

        def _encode() -> list[int]:
            with self._lock:
                return self.delegate.encode(
                    prompt, add_special_tokens=add_special_tokens
                )

        token_ids = np.asarray(
            await run_with_default_executor(_encode), dtype=np.int64
        )
        if self.max_length is not None and token_ids.size > self.max_length:
            raise PromptTooLongError(
                int(token_ids.size),
                self.max_length,
                limit_description="model's maximum prompt length",
            )
        return token_ids

    async def decode(
        self, encoded: npt.NDArray[np.int64], **kwargs: Any
    ) -> str:
        """Raises: audio generation returns samples, never text."""
        raise NotImplementedError(
            "Decoding is not implemented for audio generation."
        )

    async def new_context(self, request: OpenResponsesRequest) -> AudioContext:
        """Builds the context for one audio generation request.

        Args:
            request: The incoming request.

        Returns:
            The context to generate from.

        Raises:
            ValueError: If the request carries no prompt.
        """
        description = retrieve_input_text(request)
        if not description.strip():
            raise ValueError(
                "The prompt must be a non-empty description of the audio."
            )

        options = request.body.provider_options.audio or AudioProviderOptions()
        token_ids = await self.encode(
            self.assemble_prompt(description, options.lyrics),
            add_special_tokens=False,
        )
        unconditional = self.unconditional_ids(token_ids)

        return AudioContext(
            request_id=request.request_id,
            model_name=request.body.model,
            tokens=TokenBuffer(array=token_ids),
            negative_tokens=(
                None
                if unconditional is None
                else TokenBuffer(array=unconditional)
            ),
            # What the request leaves unset falls back to this checkpoint's
            # own default, not to a framework-wide one.
            audio_duration=(
                options.audio_duration
                if options.audio_duration is not None
                else self._default_audio_duration
            ),
            num_inference_steps=(
                options.steps
                if options.steps is not None
                else self._default_num_inference_steps
            ),
            guidance_scale=options.guidance_scale,
            seed=request.body.seed,
            audio_format=options.audio_format,
        )
