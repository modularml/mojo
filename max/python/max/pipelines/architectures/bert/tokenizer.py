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
"""Custom tokenizer for Bert models.

BERT-based models don't have an explicit EOS token, so this tokenizer
uses the SEP token as the EOS token for compatibility with MAX pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from max.pipelines.lib import TextTokenizer

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig


class BertTokenizer(TextTokenizer):
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
        if not self._eos_token_ids:
            if self.delegate.sep_token_id is not None:
                self._eos_token_ids = {self.delegate.sep_token_id}
            elif self.delegate.pad_token_id is not None:
                self._eos_token_ids = {self.delegate.pad_token_id}
            else:
                self._eos_token_ids = {0}
