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
"""Inkling's processor: raw images to hMLP patch blocks, and the prompt
surgery that sizes each image's placeholder run.

Hand-rolled after the vLLM reference (``vllm/transformers_utils/processors/
inkling.py`` at v0.26.0), which diverges from what ``processor_config.json``
appears to say; only ``image_mean``, ``image_std`` and ``image_token`` are read
from that file.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import huggingface_hub
import numpy as np
import numpy.typing as npt
from PIL import Image
from transformers import PreTrainedTokenizerBase

from .model_config import InklingVisionConfig

PROCESSOR_CONFIG_FILE = "processor_config.json"

# Raw pixel value the reference pads with, before normalization.
_PAD_RAW_VALUE = -1.0 / 255.0

# Reference class defaults (not processor_config.json): grow toward a
# 2048-pixel long edge by at most 2x; leave larger images alone.
_RESCALE_FRAC = 2.0
_MAX_UPSCALED_LONG_EDGE = 2048


class InklingImageProcessor:
    """Turns an image into the patch block the vision tower consumes."""

    def __init__(
        self,
        processor_config: Mapping[str, Any],
        vision_config: InklingVisionConfig,
    ) -> None:
        block = processor_config.get("image_processor")
        if not isinstance(block, Mapping):
            raise ValueError(
                f"{PROCESSOR_CONFIG_FILE} has no image_processor block, so the "
                "image normalization constants are unknown"
            )
        self.patch_size = vision_config.patch_size
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.image_mean = np.asarray(block["image_mean"], dtype=np.float32)
        self.image_std = np.asarray(block["image_std"], dtype=np.float32)
        self._pad_value = (
            np.float32(_PAD_RAW_VALUE) - self.image_mean
        ) / self.image_std

    def __call__(self, image: Image.Image) -> npt.NDArray[np.float32]:
        """One image to ``[num_patches, T, patch, patch, 3]``.

        The leading axis is the image's patch count, which is also the length
        of its placeholder run in the prompt.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        scaled_size = _scaled_size(image.width, image.height)
        if scaled_size != image.size:
            image = image.resize(scaled_size, resample=Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.uint8)
        height, width, _ = pixels.shape

        rows = -(-height // self.patch_size)
        # The reference always leaves one trailing column of pure padding.
        columns = width // self.patch_size + 1
        padded = np.broadcast_to(
            self._pad_value,
            (rows * self.patch_size, columns * self.patch_size, 3),
        ).copy()
        padded[:height, :width] = (
            pixels.astype(np.float32) * np.float32(1.0 / 255.0)
            - self.image_mean
        ) / self.image_std

        patches = padded.reshape(
            rows, self.patch_size, columns, self.patch_size, 3
        ).transpose(0, 2, 1, 3, 4)
        return np.repeat(
            patches.reshape(
                rows * columns, 1, self.patch_size, self.patch_size, 3
            ),
            self.temporal_patch_size,
            axis=1,
        )


class InklingProcessor:
    """Stands in for the ``AutoProcessor`` the checkpoint names but does not
    ship, in the shape :class:`~max.pipelines.lib.TextAndVisionTokenizer`
    expects: a chat template, and a call that returns token ids alongside one
    patch block per image."""

    def __init__(
        self,
        delegate: PreTrainedTokenizerBase,
        processor_config: Mapping[str, Any],
        vision_config: InklingVisionConfig,
    ) -> None:
        self.delegate = delegate
        self.image_processor = InklingImageProcessor(
            processor_config, vision_config
        )
        image_token = processor_config["image_token"]
        vocab = delegate.get_vocab()
        if image_token not in vocab:
            raise ValueError(
                f"{PROCESSOR_CONFIG_FILE} names {image_token!r} as the image "
                "token, but the tokenizer vocabulary has no such entry"
            )
        self.image_token_id = int(vocab[image_token])

    def apply_chat_template(self, messages: Any, **options: Any) -> str:
        templated = self.delegate.apply_chat_template(messages, **options)
        assert isinstance(templated, str)
        return templated

    def __call__(
        self,
        *,
        text: str | Sequence[int],
        images: Sequence[Image.Image] | None = None,
        add_special_tokens: bool = True,
        **unused_kwargs: Any,
    ) -> dict[str, Any]:
        del unused_kwargs
        token_ids = (
            self.delegate.encode(text, add_special_tokens=add_special_tokens)
            if isinstance(text, str)
            else list(text)
        )
        blocks = [self.image_processor(image) for image in images or []]
        return {
            # Both keys are batched, as a HuggingFace processor returns them.
            "input_ids": [
                expand_image_placeholders(
                    token_ids,
                    self.image_token_id,
                    [block.shape[0] for block in blocks],
                )
            ],
            "pixel_values": [blocks],
        }


def _scaled_size(width: int, height: int) -> tuple[int, int]:
    """Applies the long-edge rescale policy to a raw image size."""
    long_edge = max(width, height)
    target = min(
        long_edge * _RESCALE_FRAC,
        float(max(_MAX_UPSCALED_LONG_EDGE, long_edge)),
    )
    ratio = target / long_edge
    if ratio == 1.0:
        return width, height
    return (
        max(1, math.floor(width * ratio + 0.5)),
        max(1, math.floor(height * ratio + 0.5)),
    )


def load_processor_config(
    model_path: str, revision: str | None = None
) -> dict[str, Any]:
    """Reads ``processor_config.json`` out of the checkpoint."""
    local = Path(model_path) / PROCESSOR_CONFIG_FILE
    if not local.is_file():
        local = Path(
            huggingface_hub.hf_hub_download(
                repo_id=model_path,
                filename=PROCESSOR_CONFIG_FILE,
                revision=revision,
            )
        )
    return json.loads(local.read_text())


def expand_image_placeholders(
    token_ids: Sequence[int], placeholder_id: int, patch_counts: Sequence[int]
) -> list[int]:
    """Grows each single placeholder token into that image's run of patches.

    The chat template emits one placeholder per image; the run length is only
    known once the image has been through the processor.
    """
    markers = [
        index
        for index, token_id in enumerate(token_ids)
        if token_id == placeholder_id
    ]
    if len(markers) != len(patch_counts):
        raise ValueError(
            f"prompt carries {len(markers)} image placeholder(s) but "
            f"{len(patch_counts)} image input(s) were provided"
        )
    # Downstream recovers each image's run as a maximal contiguous stretch of
    # placeholder tokens, so back-to-back runs would fuse into one image.
    if any(second - first == 1 for first, second in pairwise(markers)):
        raise ValueError(
            "prompt places two image placeholders back to back; separate "
            "consecutive images with at least one text token"
        )
    expanded = list(token_ids)
    # Back to front, so the untouched markers keep their indices.
    for index, count in reversed(list(zip(markers, patch_counts, strict=True))):
        expanded[index : index + 1] = [placeholder_id] * count
    return expanded
