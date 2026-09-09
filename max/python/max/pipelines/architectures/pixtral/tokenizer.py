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

"""Pixtral-specific tokenizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from max.pipelines.lib import TextAndVisionTokenizer
from transformers import AutoProcessor, AutoTokenizer

if TYPE_CHECKING:
    from max.pipelines.lib import PipelineConfig


class PixtralTokenizer(TextAndVisionTokenizer):
    """Pixtral tokenizer that forces the slow image processor.

    Transformers v5 auto-promotes the saved slow image processor to its fast
    variant whenever torchvision is available, and the fast variant rejects
    ``return_tensors="np"`` in ``validate_fast_preprocess_arguments``. Loading
    with ``use_fast=False`` keeps the pre-v5 default.

    This mirrors :class:`~max.pipelines.lib.TextAndVisionTokenizer.__init__`
    rather than calling ``super().__init__`` so the processor is loaded exactly
    once (with ``use_fast=False``); keep it in sync with the base class.
    """

    def __init__(
        self,
        model_path: str,
        pipeline_config: PipelineConfig,
        *,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        **unused_kwargs,
    ) -> None:
        self.model_path = model_path

        self.delegate = AutoTokenizer.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            # If `max_length` is None, the max length will be taken
            # from the HuggingFace tokenizer_config.
            model_max_length=max_length,
        )
        self.max_length = max_length or self.delegate.model_max_length

        # Use the pre-loaded HuggingFace config from pipeline_config
        config = pipeline_config.model.huggingface_config

        # Force the slow image processor; see the class docstring for why.
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            revision=revision,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )
        eos_token_id = self.delegate.eos_token_id
        self._eos_token_ids = (
            {eos_token_id} if eos_token_id is not None else set()
        )

        if eos_token_id := getattr(config, "eos_token_id", None):
            if isinstance(eos_token_id, int):
                self._eos_token_ids.add(eos_token_id)
            elif isinstance(eos_token_id, list):
                self._eos_token_ids.update(eos_token_id)

        self.enable_prefix_caching = (
            pipeline_config.model.kv_cache.enable_prefix_caching
        )

        # Pixtral uses image_token_index.
        vision_token_ids: list[int] = []
        for vision_token_id_name in ["image_token_id", "image_token_index"]:
            if vision_token_id := getattr(config, vision_token_id_name, None):
                vision_token_ids.append(vision_token_id)
        if not vision_token_ids:
            raise ValueError("vision_token_id not found in model_config config")
        self.vision_token_ids = vision_token_ids

        # This is pixtral specific hack as it also has a image_break_token_id
        if image_break_token_id := getattr(
            self.processor, "image_break_token_id", None
        ):
            self.vision_token_ids.append(image_break_token_id)
