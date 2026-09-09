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

"""Kimi K2.5's preprocessed-image cache.

Two properties matter, and they pull against each other: preprocessing one
image at a time must produce exactly what the batched call produced, and a
repeated image must not be preprocessed twice.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from max.pipelines.architectures.kimik2_5.tokenizer import (
    KimiK2_5VLTokenizer,
)
from max.pipelines.architectures.kimik2_5.vision_processor import (
    KimiK2_5VisionProcessor,
    _to_pil,
)
from max.pipelines.context import SamplingParams
from max.pipelines.lib import VisionPreprocessCache
from max.pipelines.modeling.types import (
    ImageContentPart,
    RequestID,
    TextContentPart,
    TextGenerationRequest,
    TextGenerationRequestMessage,
    VideoContentPart,
)
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


def _request(images: list[bytes]) -> TextGenerationRequest:
    """A request carrying ``images``, with the parts it validates against."""
    # Annotated with the full union because list is invariant, so a list built
    # only of ImageContentPart will not satisfy the declared content type.
    content: list[TextContentPart | ImageContentPart | VideoContentPart] = [
        ImageContentPart() for _ in images
    ]
    content.append(TextContentPart(text="Describe these."))
    return TextGenerationRequest(
        request_id=RequestID("test_preprocess_cache"),
        model_name="kimi-k2.5",
        messages=[TextGenerationRequestMessage(role="user", content=content)],
        sampling_params=SamplingParams(max_new_tokens=1),
        images=images,
    )


def _tokenizer(
    cache_bytes: int = 64 * 1024**2,
) -> tuple[KimiK2_5VLTokenizer, KimiK2_5VisionProcessor]:
    """A tokenizer with only what ``_process_images`` touches.

    ``__init__`` is bypassed rather than mocked because it loads an HF
    tokenizer and config; the image path depends on none of that.
    """
    with patch.object(
        KimiK2_5VLTokenizer, "__init__", lambda self, *args, **kwargs: None
    ):
        tokenizer = KimiK2_5VLTokenizer.__new__(KimiK2_5VLTokenizer)
    processor = KimiK2_5VisionProcessor()
    tokenizer.vision_processor = processor
    tokenizer._preprocess_cache = VisionPreprocessCache(cache_bytes)
    return tokenizer, processor


def _batched_reference(
    processor: KimiK2_5VisionProcessor, images: list[bytes]
) -> dict[str, np.ndarray]:
    """What the single batched ``preprocess`` call produced before caching."""
    return processor.preprocess(
        [{"type": "image", "image": _to_pil(raw)} for raw in images]
    )


def _hashes(
    processor: KimiK2_5VisionProcessor, images: list[bytes]
) -> list[int | None]:
    """The content keys ``new_context`` computes, on the same size tier."""
    return [hash_image(raw, processor.cfg.in_patch_limit) for raw in images]


class TestBatchedParity:
    """Per-image preprocessing must reproduce the batched call exactly.

    This is the claim the whole change rests on: the processor resizes,
    normalizes and patchifies each item independently and only concatenates at
    the end, so splitting the loop is not supposed to move a single bit. A
    tolerance-based check would hide exactly the kind of drift that matters
    here, so these compare bit-for-bit.
    """

    @pytest.mark.parametrize(
        "images",
        [
            [_IMAGE_A],
            [_IMAGE_A, _IMAGE_B],
            [_IMAGE_A, _IMAGE_B, _IMAGE_C],
            # The same image twice: the second is served from the cache, and
            # must still land in the batch identically to a fresh preprocess.
            [_IMAGE_A, _IMAGE_A],
            [_IMAGE_B, _IMAGE_A, _IMAGE_B],
        ],
        ids=["one", "two", "three", "repeat", "interleaved-repeat"],
    )
    def test_cached_per_image_path_is_bit_identical(
        self, images: list[bytes]
    ) -> None:
        tokenizer, processor = _tokenizer()
        expected = _batched_reference(processor, images)

        hashes = _hashes(processor, images)
        actual = tokenizer._process_images(_request(images), hashes)

        for key in ("pixel_values", "grid_thws"):
            assert actual[key].dtype == expected[key].dtype, key
            assert actual[key].shape == expected[key].shape, key
            assert np.array_equal(actual[key], expected[key]), key

    def test_a_cache_hit_returns_the_same_values_as_a_miss(self) -> None:
        """Running the identical request twice must not change the output.

        Guards the hit path specifically: the first call populates, the second
        is served from the cache, and the two must agree bit-for-bit.
        """
        tokenizer, processor = _tokenizer()
        images = [_IMAGE_A, _IMAGE_B]
        hashes = _hashes(processor, images)

        first = tokenizer._process_images(_request(images), hashes)
        second = tokenizer._process_images(_request(images), hashes)

        assert np.array_equal(first["pixel_values"], second["pixel_values"])
        assert np.array_equal(first["grid_thws"], second["grid_thws"])


class TestPreprocessingIsActuallySkipped:
    """The point of the cache: a repeated image does not preprocess again."""

    @staticmethod
    def _spy_on(processor: KimiK2_5VisionProcessor) -> MagicMock:
        """Counts calls into the processor without changing its behaviour."""
        spy = MagicMock(side_effect=processor.preprocess)
        processor.preprocess = spy  # type: ignore[method-assign]
        return spy

    def test_a_repeated_image_preprocesses_once(self) -> None:
        tokenizer, processor = _tokenizer()
        spy = self._spy_on(processor)
        images = [_IMAGE_A, _IMAGE_A, _IMAGE_A, _IMAGE_A]

        tokenizer._process_images(_request(images), _hashes(processor, images))

        assert spy.call_count == 1

    def test_distinct_images_each_preprocess(self) -> None:
        tokenizer, processor = _tokenizer()
        spy = self._spy_on(processor)
        images = [_IMAGE_A, _IMAGE_B, _IMAGE_C]

        tokenizer._process_images(_request(images), _hashes(processor, images))

        assert spy.call_count == 3

    def test_a_second_request_reuses_the_first_request_s_work(self) -> None:
        """The cache outlives one request -- the case a conversation hits."""
        tokenizer, processor = _tokenizer()
        images = [_IMAGE_A, _IMAGE_B]
        hashes = _hashes(processor, images)
        tokenizer._process_images(_request(images), hashes)

        spy = self._spy_on(processor)
        tokenizer._process_images(_request(images), hashes)

        assert spy.call_count == 0

    def test_a_disabled_cache_preprocesses_every_time(self) -> None:
        """With a zero budget nothing is retained and nothing is skipped."""
        tokenizer, processor = _tokenizer(cache_bytes=0)
        spy = self._spy_on(processor)
        images = [_IMAGE_A, _IMAGE_A]

        tokenizer._process_images(_request(images), [None, None])

        assert spy.call_count == 2

    def test_a_null_key_is_never_cached(self) -> None:
        """A ``None`` key means "do not cache", even with a live budget.

        Kimi passes ``None`` when no caching is enabled; treating it as a
        cacheable key would collapse every image onto one entry.
        """
        tokenizer, processor = _tokenizer()
        spy = self._spy_on(processor)
        images = [_IMAGE_A, _IMAGE_A]

        tokenizer._process_images(_request(images), [None, None])

        assert spy.call_count == 2
        assert len(tokenizer._preprocess_cache) == 0


class TestCachedPayloadsAreFrozen:
    """One payload is shared by every request that reuses the image."""

    def test_a_cached_payload_cannot_be_mutated_in_place(self) -> None:
        tokenizer, processor = _tokenizer()
        hashes = _hashes(processor, [_IMAGE_A])
        tokenizer._process_images(_request([_IMAGE_A]), hashes)

        key = hashes[0]
        assert key is not None
        cached = tokenizer._preprocess_cache.get(key)
        assert cached is not None
        pixels, _ = cached
        with pytest.raises(ValueError):
            pixels[0] = 0

    def test_the_returned_batch_is_still_writable(self) -> None:
        """Freezing the cached payload must not freeze what callers get.

        Downstream code splits and copies these arrays, so the reassembled
        batch has to behave exactly as it did before the cache existed.
        """
        tokenizer, processor = _tokenizer()
        hashes = _hashes(processor, [_IMAGE_A])
        outputs = tokenizer._process_images(_request([_IMAGE_A]), hashes)

        outputs["pixel_values"][0] = 0
        outputs["grid_thws"][0] = 1
