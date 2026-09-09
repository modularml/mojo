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

"""Laguna tokenizer with reasoning-delimiter resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from max.pipelines.lib import TextTokenizer
from max.pipelines.lib.tokenizer import resolve_single_special_token

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig

# Reasoning span delimiters. Both are special tokens in the Laguna tokenizer
# vocab; resolving them at init lets us implement the
# ``ReasoningPipelineTokenizer`` protocol that ``reasoning_parser="laguna"``
# requires.
_THINK_START_TOKEN = "<think>"
_THINK_END_TOKEN = "</think>"


class LagunaTokenizer(TextTokenizer):
    """A ``TextTokenizer`` that also exposes Laguna's reasoning-delimiter ids.

    Implements the ``ReasoningPipelineTokenizer`` protocol so the overlap
    pipeline's thinking-mode temperature scaling can read the
    ``<think>``/``</think>`` delimiter ids directly.
    """

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        **kwargs,
    ) -> None:
        super().__init__(model_path, pipeline_config, **kwargs)
        self._reasoning_start_token_id: int = resolve_single_special_token(
            self.delegate, _THINK_START_TOKEN
        )
        self._reasoning_end_token_id: int = resolve_single_special_token(
            self.delegate, _THINK_END_TOKEN
        )

    @property
    def reasoning_start_token_id(self) -> int:
        """Token id of ``<think>`` (opens a Laguna reasoning span)."""
        return self._reasoning_start_token_id

    @property
    def reasoning_end_token_id(self) -> int:
        """Token id of ``</think>`` (closes a Laguna reasoning span)."""
        return self._reasoning_end_token_id
