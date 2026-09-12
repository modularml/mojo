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
"""Tokenizer wrapper for ModernBERT embeddings models.

ModernBERT checkpoints may not define an EOS token in tokenizer config.
For pipeline compatibility, fall back to SEP, then PAD.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from max.pipelines.lib import TextTokenizer

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig


class ModernBertTokenizer(TextTokenizer):
    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        enable_llama_whitespace_fix: bool = False,
        chat_template: str | None = None,
        **unused_kwargs,
    ) -> None:
        super().__init__(
            model_path,
            pipeline_config,
            revision=revision,
            max_length=max_length,
            trust_remote_code=trust_remote_code,
            enable_llama_whitespace_fix=enable_llama_whitespace_fix,
            chat_template=chat_template,
            **unused_kwargs,
        )

    @property
    def eos(self) -> int:
        if self.delegate.eos_token_id is not None:
            return self.delegate.eos_token_id

        if self.delegate.sep_token_id is not None:
            return self.delegate.sep_token_id

        if self.delegate.pad_token_id is not None:
            return self.delegate.pad_token_id

        return 0
