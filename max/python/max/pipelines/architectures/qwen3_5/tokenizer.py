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
"""Qwen3.5 tokenizer with thinking mode disabled by default."""

from __future__ import annotations

from typing import Any

from max.pipelines.architectures.qwen3vl_moe.tokenizer import Qwen3VLTokenizer
from max.pipelines.lib.config import PipelineConfig
from max.pipelines.lib.tokenizer import resolve_single_special_token
from max.pipelines.modeling.types import (
    TextGenerationRequestMessage,
    TextGenerationRequestTool,
)

_THINK_START_TOKEN = "<think>"
_THINK_END_TOKEN = "</think>"


class Qwen3_5Tokenizer(Qwen3VLTokenizer):
    """Tokenizer for Qwen3.5 multimodal models.

    Inherits full image-processing pipeline from :class:`Qwen3VLTokenizer` and
    enables thinking mode (``enable_thinking=True``) by default. Implements
    :class:`~max.pipelines.modeling.types.ReasoningPipelineTokenizer` by
    resolving the ``<think>``/``</think>`` delimiter token IDs at construction.
    """

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        max_new_tokens: int | None = None,
        trust_remote_code: bool = False,
        **unused_kwargs: Any,
    ) -> None:
        super().__init__(
            model_path,
            pipeline_config,
            revision=revision,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            trust_remote_code=trust_remote_code,
            **unused_kwargs,
        )
        self._reasoning_start_token_id: int = resolve_single_special_token(
            self.delegate, _THINK_START_TOKEN
        )
        self._reasoning_end_token_id: int = resolve_single_special_token(
            self.delegate, _THINK_END_TOKEN
        )

    @property
    def reasoning_start_token_id(self) -> int:
        """Token id of ``<think>`` (opens a Qwen3.5 reasoning span)."""
        return self._reasoning_start_token_id

    @property
    def reasoning_end_token_id(self) -> int:
        """Token id of ``</think>`` (closes a Qwen3.5 reasoning span)."""
        return self._reasoning_end_token_id

    def apply_chat_template(
        self,
        messages: list[TextGenerationRequestMessage],
        tools: list[TextGenerationRequestTool] | None = None,
        **chat_template_options: Any,
    ) -> str:
        """Apply chat template with thinking enabled by default.

        Args:
            messages: List of messages for the chat template.
            tools: Optional tools available for the model to invoke.
            **chat_template_options: Template options to forward to the Jinja
                template. Merged with ``add_generation_prompt=True`` and
                ``enable_thinking=True`` defaults.

        Returns:
            The templated chat message as a string.
        """
        enable_thinking = chat_template_options.get("enable_thinking", True)

        return self.delegate.apply_chat_template(
            [msg.model_dump(exclude_none=True) for msg in messages],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
