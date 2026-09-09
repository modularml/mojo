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
"""Inkling tokenizer: the shared text-and-vision one on our own processor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from max.pipelines.lib import TextAndVisionTokenizer
from max.pipelines.lib.tokenizer import resolve_single_special_token
from transformers import AutoTokenizer

from .model_config import InklingVisionConfig
from .processor import InklingProcessor, load_processor_config

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig

_THINKING_START_TOKEN = "<|content_thinking|>"
_END_MESSAGE_TOKEN = "<|end_message|>"

# Opens the JSON payload of a tool call. The tool parser matches this exact
# string, so it has to survive detokenization.
TOOL_CALL_JSON_MARKER = "<|content_invoke_tool_json|>"


class InklingTokenizer(TextAndVisionTokenizer):
    """Tokenizer for Inkling, whose own ``AutoProcessor`` class is not shipped
    with the checkpoint, so :class:`InklingProcessor` takes its place.

    Also exposes Inkling's reasoning-delimiter ids, satisfying the
    ``ReasoningPipelineTokenizer`` protocol that
    ``OverlapTextGenerationPipeline`` requires of any architecture that names a
    reasoning parser.

    ``skipped_special_token_ids`` opts the streaming detokenizer into a per-id
    filter so ``<|content_invoke_tool_json|>`` survives decode for the tool
    parser. Only that marker: a request without ``tools`` runs no tool parser,
    so anything else kept here leaks verbatim into ``message.content``.
    """

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        **unused_kwargs: Any,
    ) -> None:
        self.model_path = model_path
        self.delegate = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            model_max_length=max_length,
        )
        self.max_length = max_length or self.delegate.model_max_length

        huggingface_config = pipeline_config.model.huggingface_config
        eos_token_id = self.delegate.eos_token_id
        self._eos_token_ids = (
            {eos_token_id} if eos_token_id is not None else set()
        )
        if eos_token_id := getattr(huggingface_config, "eos_token_id", None):
            if isinstance(eos_token_id, int):
                self._eos_token_ids.add(eos_token_id)
            elif isinstance(eos_token_id, list):
                self._eos_token_ids.update(eos_token_id)

        self.enable_prefix_caching = (
            pipeline_config.model.kv_cache.enable_prefix_caching
        )
        self.processor = InklingProcessor(
            self.delegate,
            load_processor_config(model_path, revision),
            InklingVisionConfig.from_hf(huggingface_config.vision_config),
        )
        self.vision_token_ids = [self.processor.image_token_id]

        self._reasoning_start_token_id: int = resolve_single_special_token(
            self.delegate, _THINKING_START_TOKEN
        )
        self._reasoning_end_token_id: int = resolve_single_special_token(
            self.delegate, _END_MESSAGE_TOKEN
        )

        tool_call_json_id = resolve_single_special_token(
            self.delegate, TOOL_CALL_JSON_MARKER
        )
        self.skipped_special_token_ids: set[int] = set(
            self.delegate.all_special_ids
        ) - {tool_call_json_id}

    @property
    def reasoning_start_token_id(self) -> int:
        """Token id of ``<|content_thinking|>``."""
        return self._reasoning_start_token_id

    @property
    def reasoning_end_token_id(self) -> int:
        """Token id of ``<|end_message|>``."""
        return self._reasoning_end_token_id

    async def decode(
        self, encoded: npt.NDArray[np.integer[Any]] | int, **kwargs
    ) -> str:
        """Decodes tokens, dropping every special id except the tool-call marker.

        ``skip_special_tokens=True`` would drop the marker too, so filter by id
        here and decode with the flag off.
        """
        # Log-probability responses decode a single token id (a plain int).
        token_ids = np.atleast_1d(np.asarray(encoded))

        if not kwargs.get("skip_special_tokens", True):
            return await super().decode(token_ids, **kwargs)

        filtered_ids = [
            token_id
            for token_id in token_ids.tolist()
            if token_id not in self.skipped_special_token_ids
        ]
        return await super().decode(
            np.array(filtered_ids, dtype=np.int64),
            **{**kwargs, "skip_special_tokens": False},
        )
