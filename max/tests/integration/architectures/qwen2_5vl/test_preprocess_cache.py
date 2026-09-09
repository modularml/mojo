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

"""Qwen2.5-VL's preprocessed-image cache.

Two properties matter, and they pull against each other: preprocessing one
image at a time must produce exactly what the batched call produced, and a
repeated image must not be preprocessed twice.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from max.pipelines.architectures.qwen2_5vl import tokenizer as qwen_tokenizer
from max.pipelines.architectures.qwen2_5vl.nn.qwen_vl_utils import (
    MAX_PIXELS,
    fetch_image,
)
from max.pipelines.architectures.qwen2_5vl.tokenizer import (
    Qwen2_5VLImageProcessor,
    Qwen2_5VLTokenizer,
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
        model_name="qwen2.5-vl",
        messages=[TextGenerationRequestMessage(role="user", content=content)],
        sampling_params=SamplingParams(max_new_tokens=1),
        images=images,
    )


def _processor() -> Qwen2_5VLImageProcessor:
    """The processor with Qwen2.5-VL's real vision-config values."""
    return Qwen2_5VLImageProcessor(
        patch_size=14, temporal_patch_size=2, merge_size=2
    )


def _tokenizer(
    cache_bytes: int = 64 * 1024**2,
) -> tuple[Qwen2_5VLTokenizer, Qwen2_5VLImageProcessor]:
    """A tokenizer with only what ``_process_images`` touches.

    ``__init__`` is bypassed rather than mocked because it loads an HF
    tokenizer and config; the image path depends on none of that.
    """
    with patch.object(
        Qwen2_5VLTokenizer, "__init__", lambda self, *args, **kwargs: None
    ):
        tokenizer = Qwen2_5VLTokenizer.__new__(Qwen2_5VLTokenizer)
    processor = _processor()
    tokenizer.img_processor = processor
    tokenizer.enable_prefix_caching = False
    tokenizer._preprocess_cache = VisionPreprocessCache(cache_bytes)
    return tokenizer, processor


def _hashes(images: list[bytes]) -> list[int | None]:
    """The content keys ``_process_images`` computes, on the same size tier."""
    return [hash_image(raw, MAX_PIXELS) for raw in images]


class TestBatchedParity:
    """Per-image preprocessing must reproduce the batched call exactly.

    This is the claim the change rests on: the processor calls
    ``qwen2_5vl_image_preprocessing`` once per image with no cross-image state
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
        tokenizer, _ = _tokenizer()
        # The batched call this replaced, on a processor with no cache wired.
        expected_dict, expected_list = _processor()(
            images=[fetch_image({"image": raw}) for raw in images],
            return_tensors="np",
        )

        pixels, grid, per_image, _ = tokenizer._process_images(_request(images))

        assert pixels.dtype == expected_dict["concatenated_pixel_values"].dtype
        assert pixels.shape == expected_dict["concatenated_pixel_values"].shape
        assert np.array_equal(
            pixels, expected_dict["concatenated_pixel_values"]
        )
        assert grid.dtype == expected_dict["image_grid_thw"].dtype
        assert np.array_equal(grid, expected_dict["image_grid_thw"])
        assert len(per_image) == len(expected_list)
        for actual, expected in zip(per_image, expected_list, strict=True):
            assert actual.dtype == expected.dtype
            assert np.array_equal(actual, expected)

    def test_a_cache_hit_returns_the_same_values_as_a_miss(self) -> None:
        """Running the identical request twice must not change the output."""
        tokenizer, _ = _tokenizer()
        images = [_IMAGE_A, _IMAGE_B]

        first_pixels, first_grid, _, _ = tokenizer._process_images(
            _request(images)
        )
        second_pixels, second_grid, _, _ = tokenizer._process_images(
            _request(images)
        )

        assert np.array_equal(first_pixels, second_pixels)
        assert np.array_equal(first_grid, second_grid)


def _spy_on_per_image_work(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Counts per-image preprocessing without changing its behaviour.

    Patched at the module level rather than on the processor instance, because
    ``processor(...)`` resolves ``__call__`` on the type -- an instance
    attribute would be silently ignored and every count would read as zero.
    """
    spy = MagicMock(side_effect=qwen_tokenizer.qwen2_5vl_image_preprocessing)
    monkeypatch.setattr(qwen_tokenizer, "qwen2_5vl_image_preprocessing", spy)
    return spy


class TestPreprocessingIsActuallySkipped:
    """The point of the cache: a repeated image does not preprocess again."""

    def test_a_repeated_image_preprocesses_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tokenizer, _ = _tokenizer()
        spy = _spy_on_per_image_work(monkeypatch)

        tokenizer._process_images(_request([_IMAGE_A] * 4))

        assert spy.call_count == 1

    def test_distinct_images_each_preprocess(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tokenizer, _ = _tokenizer()
        spy = _spy_on_per_image_work(monkeypatch)

        tokenizer._process_images(_request([_IMAGE_A, _IMAGE_B, _IMAGE_C]))

        assert spy.call_count == 3

    def test_a_second_request_reuses_the_first_request_s_work(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cache outlives one request -- the case a conversation hits."""
        tokenizer, _ = _tokenizer()
        images = [_IMAGE_A, _IMAGE_B]
        tokenizer._process_images(_request(images))

        spy = _spy_on_per_image_work(monkeypatch)
        tokenizer._process_images(_request(images))

        assert spy.call_count == 0

    def test_a_disabled_cache_preprocesses_every_time(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a zero budget nothing is retained and nothing is skipped."""
        tokenizer, _ = _tokenizer(cache_bytes=0)
        spy = _spy_on_per_image_work(monkeypatch)

        tokenizer._process_images(_request([_IMAGE_A, _IMAGE_A]))

        assert spy.call_count == 2


class TestKeysAndMetadata:
    """The digests the cache keys on are the ones the request carries out."""

    def test_the_returned_digests_match_the_raw_byte_keys(self) -> None:
        """The 4th return value is what ImageMetadata.image_hash is built from.

        It must be the raw-encoded-bytes key, since a separate encoder has to
        reproduce it for cache-aware routing.
        """
        tokenizer, _ = _tokenizer()
        images = [_IMAGE_A, _IMAGE_B]

        *_, image_hashes = tokenizer._process_images(_request(images))

        assert image_hashes == _hashes(images)

    def test_a_disabled_cache_with_no_prefix_caching_needs_no_digest(
        self,
    ) -> None:
        """Nothing needs a key, so the bytes are not hashed at all."""
        tokenizer, _ = _tokenizer(cache_bytes=0)

        *_, image_hashes = tokenizer._process_images(
            _request([_IMAGE_A, _IMAGE_B])
        )

        assert image_hashes == [None, None]


class TestCachedPayloadsAreFrozen:
    """One payload is shared by every request that reuses the image."""

    def test_a_cached_payload_cannot_be_mutated_in_place(self) -> None:
        tokenizer, _ = _tokenizer()
        tokenizer._process_images(_request([_IMAGE_A]))

        key = _hashes([_IMAGE_A])[0]
        assert key is not None
        cached = tokenizer._preprocess_cache.get(key)
        assert cached is not None
        pixels, _ = cached
        with pytest.raises(ValueError):
            pixels[0] = 0

    def test_the_returned_batch_is_still_writable(self) -> None:
        """Freezing the cached payload must not freeze the reassembled batch."""
        tokenizer, _ = _tokenizer()

        pixels, grid, _, _ = tokenizer._process_images(_request([_IMAGE_A]))

        pixels[0] = 0
        grid[0] = 1
