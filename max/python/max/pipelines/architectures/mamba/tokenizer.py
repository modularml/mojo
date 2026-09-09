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

"""Mamba-specific tokenizer with default chat template support.

Mamba base models (like state-spaces/mamba-130m-hf) don't include a chat
template in their tokenizer configuration. This tokenizer provides a simple
default template to enable OpenAI-compatible chat completions endpoint usage.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from max.pipelines.context import TextContext
from max.pipelines.lib import TextTokenizer

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig

logger = logging.getLogger("max.pipelines")

# Simple passthrough chat template for base models without instruction tuning.
# This template concatenates message content without adding role prefixes,
# since base models (like state-spaces/mamba-130m-hf) are not instruction-tuned
# and role prefixes cause incoherent output.
# For chat applications, consider using an instruction-tuned Mamba variant or
# providing a custom chat template via --chat-template.
DEFAULT_MAMBA_CHAT_TEMPLATE = (
    "{% for message in messages %}{{ message['content'] }}{% endfor %}"
)


class MambaTokenizer(TextTokenizer):
    """Mamba-specific tokenizer that provides a default chat template.

    This tokenizer extends TextTokenizer to provide a basic chat template for
    Mamba base models that don't include one. This enables usage of the
    OpenAI-compatible /v1/chat/completions endpoint.

    Note:
        For best results with chat applications, consider using an
        instruction-tuned Mamba variant or providing a custom chat template
        via the --chat-template CLI option.
    """

    def __init__(
        self,
        model_path: str,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        enable_llama_whitespace_fix: bool = False,
        pipeline_config: PipelineConfig,
        chat_template: str | None = None,
        context_validators: list[Callable[[TextContext], None]] | None = None,
        **unused_kwargs,
    ) -> None:
        """Initialize the Mamba tokenizer with optional default chat template.

        Args:
            model_path: Path to the model/tokenizer.
            revision: Git revision/branch to use.
            max_length: Maximum sequence length.
            trust_remote_code: Whether to trust remote code from the model.
            enable_llama_whitespace_fix: Enable whitespace fix for Llama tokenizers.
            pipeline_config: Optional pipeline configuration.
            chat_template: Optional custom chat template string. If not provided
                and the tokenizer doesn't have one, a default template is used.
            context_validators: Optional list of context validators.
            **unused_kwargs: Additional unused keyword arguments.
        """
        # Initialize parent class
        super().__init__(
            model_path=model_path,
            revision=revision,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
            enable_llama_whitespace_fix=enable_llama_whitespace_fix,
            pipeline_config=pipeline_config,
            chat_template=chat_template,
            context_validators=context_validators,
            **unused_kwargs,
        )

        # If no chat template was provided and the tokenizer doesn't have one,
        # set the default Mamba chat template
        if chat_template is None and self.delegate.chat_template is None:
            self.delegate.chat_template = DEFAULT_MAMBA_CHAT_TEMPLATE
            logger.info(
                f"Set default Mamba chat template for {model_path} "
                "(base model without built-in template)"
            )
