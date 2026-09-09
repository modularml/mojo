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

"""Qwen3-VL-MoE's preprocessed-image cache.

Two properties matter, and they pull against each other: preprocessing one
image at a time must produce exactly what the batched call produced, and a
repeated image must not be preprocessed twice.

Qwen3-VL reassembles the batch inline in ``new_context`` rather than in a
helper, so the parity check here is on the property that reassembly relies on:
that the processor's per-image results, stacked, equal the batched call.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from max.pipelines.architectures.qwen3vl_moe import tokenizer as qwen3vl_module
from max.pipelines.architectures.qwen3vl_moe.tokenizer import (
    Qwen3VLImageProcessor,
    Qwen3VLTokenizer,
    _load_image,
)
from max.pipelines.lib import VisionPreprocessCache
from max.support.image import hash_image
from PIL import Image


def _jpeg(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    """Encoded bytes for a distinct solid-colour image."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="JPEG")
    return buffer.getvalue()


# Distinct sizes as well as colours, so a mixed batch also exercises differing
# patch counts per image rather than a uniform grid.
_IMAGE_A = _jpeg(64, 48, (200, 30, 30))
_IMAGE_B = _jpeg(96, 64, (30, 200, 30))
_IMAGE_C = _jpeg(48, 80, (30, 30, 200))


def _processor() -> Qwen3VLImageProcessor:
    """The processor with the defaults ``Qwen3VLTokenizer`` builds it with."""
    return Qwen3VLImageProcessor(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        min_pixels=65536,
        max_pixels=16777216,
    )


def _tokenizer(
    cache_bytes: int = 64 * 1024**2,
) -> tuple[Qwen3VLTokenizer, Qwen3VLImageProcessor]:
    """A tokenizer with only what ``_preprocess_image`` touches.

    ``__init__`` is bypassed rather than mocked because it loads an HF
    tokenizer and config; the image path depends on none of that.
    """
    with patch.object(
        Qwen3VLTokenizer, "__init__", lambda self, *args, **kwargs: None
    ):
        tokenizer = Qwen3VLTokenizer.__new__(Qwen3VLTokenizer)
    processor = _processor()
    tokenizer.img_processor = processor
    tokenizer.enable_prefix_caching = False
    tokenizer._preprocess_cache = VisionPreprocessCache(cache_bytes)
    return tokenizer, processor


def _keys(processor: Qwen3VLImageProcessor, images: list[bytes]) -> list[int]:
    """The content keys ``new_context`` computes, on the same size tier."""
    return [hash_image(raw, processor.max_pixels) for raw in images]


def _spy_on_per_image_work(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Counts per-image preprocessing without changing its behaviour.

    Patched at the module level rather than on the processor instance, because
    ``processor(...)`` resolves ``__call__`` on the type -- an instance
    attribute would be silently ignored and every count would read as zero.
    """
    spy = MagicMock(side_effect=qwen3vl_module.qwen3vl_image_preprocessing)
    monkeypatch.setattr(qwen3vl_module, "qwen3vl_image_preprocessing", spy)
    return spy


class TestBatchedParity:
    """Per-image preprocessing, restacked, must equal the batched call.

    This is the claim the change rests on: the processor calls
    ``qwen3vl_image_preprocessing`` once per image with no cross-image state
    and only stacks at the end, so splitting the loop is not supposed to move a
    single bit. A tolerance-based check would hide exactly the kind of drift
    that matters here, so these compare bit-for-bit.
    """

    @pytest.mark.parametrize(
        "images",
        [
            [_IMAGE_A],
            [_IMAGE_A, _IMAGE_B],
            [_IMAGE_A, _IMAGE_B, _IMAGE_C],
            [_IMAGE_A, _IMAGE_A],
            [_IMAGE_B, _IMAGE_A, _IMAGE_B],
        ],
        ids=["one", "two", "three", "repeat", "interleaved-repeat"],
    )
    def test_cached_per_image_path_is_bit_identical(
        self, images: list[bytes]
    ) -> None:
        tokenizer, processor = _tokenizer()
        decoded = [_load_image({"image": raw}) for raw in images]
        # The batched call this replaced, on a processor with no cache wired.
        expected_pixels, expected_grid, expected_list = _processor()(
            images=decoded, return_tensors="pt"
        )

        per_image = [
            tokenizer._preprocess_image(key, image)
            for key, image in zip(
                _keys(processor, images), decoded, strict=True
            )
        ]
        # The same two expressions new_context reassembles with.
        pixel_values_list = [pixels for pixels, _ in per_image]
        pixels = np.vstack(pixel_values_list)
        grid = np.array([g for _, g in per_image], dtype=np.int32)

        assert pixels.dtype == expected_pixels.dtype
        assert pixels.shape == expected_pixels.shape
        assert np.array_equal(pixels, expected_pixels)
        assert grid.dtype == expected_grid.dtype
        assert np.array_equal(grid, expected_grid)
        for actual, expected in zip(
            pixel_values_list, expected_list, strict=True
        ):
            assert actual.dtype == expected.dtype
            assert np.array_equal(actual, expected)

    def test_a_cache_hit_returns_the_same_values_as_a_miss(self) -> None:
        """A hit must be indistinguishable from preprocessing again."""
        tokenizer, processor = _tokenizer()
        key = _keys(processor, [_IMAGE_A])[0]
        image = _load_image({"image": _IMAGE_A})

        first_pixels, first_grid = tokenizer._preprocess_image(key, image)
        second_pixels, second_grid = tokenizer._preprocess_image(key, image)

        assert np.array_equal(first_pixels, second_pixels)
        assert np.array_equal(first_grid, second_grid)


class TestPreprocessingIsActuallySkipped:
    """The point of the cache: a repeated image does not preprocess again."""

    def test_a_repeated_image_preprocesses_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tokenizer, processor = _tokenizer()
        spy = _spy_on_per_image_work(monkeypatch)
        key = _keys(processor, [_IMAGE_A])[0]
        image = _load_image({"image": _IMAGE_A})

        for _ in range(4):
            tokenizer._preprocess_image(key, image)

        assert spy.call_count == 1

    def test_distinct_images_each_preprocess(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tokenizer, processor = _tokenizer()
        spy = _spy_on_per_image_work(monkeypatch)
        images = [_IMAGE_A, _IMAGE_B, _IMAGE_C]

        for key, raw in zip(_keys(processor, images), images, strict=True):
            tokenizer._preprocess_image(key, _load_image({"image": raw}))

        assert spy.call_count == 3

    def test_a_disabled_cache_preprocesses_every_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a zero budget nothing is retained and nothing is skipped."""
        tokenizer, processor = _tokenizer(cache_bytes=0)
        spy = _spy_on_per_image_work(monkeypatch)
        key = _keys(processor, [_IMAGE_A])[0]
        image = _load_image({"image": _IMAGE_A})

        tokenizer._preprocess_image(key, image)
        tokenizer._preprocess_image(key, image)

        assert spy.call_count == 2

    def test_a_null_key_is_never_cached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``None`` key means "do not cache", even with a live budget.

        Treating it as a cacheable key would collapse every image onto one
        entry and serve the wrong pixels.
        """
        tokenizer, _ = _tokenizer()
        spy = _spy_on_per_image_work(monkeypatch)
        image = _load_image({"image": _IMAGE_A})

        tokenizer._preprocess_image(None, image)
        tokenizer._preprocess_image(None, image)

        assert spy.call_count == 2
        assert len(tokenizer._preprocess_cache) == 0


class TestCachedPayloadsAreFrozen:
    """One payload is shared by every request that reuses the image."""

    def test_a_cached_payload_cannot_be_mutated_in_place(self) -> None:
        tokenizer, processor = _tokenizer()
        key = _keys(processor, [_IMAGE_A])[0]

        pixels, _ = tokenizer._preprocess_image(
            key, _load_image({"image": _IMAGE_A})
        )

        with pytest.raises(ValueError):
            pixels[0] = 0
