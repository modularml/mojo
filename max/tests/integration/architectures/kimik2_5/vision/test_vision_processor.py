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

"""Tests for Kimi K2.5 vision processor.

Validates the custom KimiK2_5VisionProcessor (PIL+NumPy only) against
the HuggingFace reference implementation from nvidia/Kimi-K2.5-NVFP4.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pytest
import torch
from max.pipelines.architectures.kimik2_5.vision_processor import (
    KimiK2_5Processor,
    KimiK2_5VisionProcessor,
    MediaProcConfig,
    _to_pil,
    navit_patchify,
    navit_resize_image,
    navit_resize_video,
    normalize,
    timestamp_as_str,
)
from max.pipelines.context.exceptions import InputError
from PIL import Image
from transformers import AutoImageProcessor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

KIMI_LOGO_URL = "https://huggingface.co/moonshotai/Kimi-K2.5/resolve/main/figures/kimi-logo.png"
KIMI_VIDEO_URL = "https://huggingface.co/moonshotai/Kimi-K2.5/resolve/main/figures/demo_video.mp4"


def _make_rgb_image(width: int, height: int) -> Image.Image:
    """Creates a deterministic synthetic RGB image."""
    rng = np.random.RandomState(42)
    pixels = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _download_image(url: str) -> Image.Image:
    """Downloads an image from a URL and returns it as an RGB PIL Image."""
    import urllib.request

    with urllib.request.urlopen(url) as resp:
        return Image.open(io.BytesIO(resp.read())).convert("RGB")


def _download_video_frames(url: str, num_frames: int = 4) -> list[Image.Image]:
    """Downloads a video and extracts evenly-spaced frames as PIL Images."""
    import tempfile
    import urllib.request

    import av

    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        urllib.request.urlretrieve(url, tmp.name)
        container = av.open(tmp.name)
        stream = container.streams.video[0]
        total = stream.frames or 0
        all_frames: list[Image.Image] = []
        for frame in container.decode(video=0):
            all_frames.append(frame.to_image().convert("RGB"))
        container.close()

    if not all_frames:
        raise RuntimeError(f"No frames decoded from {url}")

    total = len(all_frames)
    if total <= num_frames:
        return all_frames

    indices = np.linspace(0, total - 1, num_frames).round().astype(int)
    return [all_frames[i] for i in indices]


class TestImageLoading:
    """Tests for image byte decoding."""

    def test_to_pil_rejects_malformed_image_bytes(self) -> None:
        with pytest.raises(
            InputError, match="image bytes could not be decoded"
        ):
            _to_pil(b"not a valid image")


# ---------------------------------------------------------------------------
# navit_resize_image tests
# ---------------------------------------------------------------------------


class TestNavitResizeImage:
    """Tests for the NaViT resize calculation."""

    def test_small_image_no_downscale(self) -> None:
        """An image smaller than all limits should not be scaled down."""
        result = navit_resize_image(
            width=28,
            height=28,
            patch_size=14,
            merge_kernel_size=2,
            in_patch_limit=16384,
            patch_limit_on_one_side=512,
            fixed_output_tokens=None,
        )
        assert result["new_width"] == 28
        assert result["new_height"] == 28
        assert result["pad_width"] == 0
        assert result["pad_height"] == 0
        assert result["num_tokens"] == 1
        assert result["sampled_nframes"] == 1

    def test_padding_alignment(self) -> None:
        """Dimensions not aligned to factor are padded correctly."""
        result = navit_resize_image(
            width=30,
            height=30,
            patch_size=14,
            merge_kernel_size=2,
            in_patch_limit=16384,
            patch_limit_on_one_side=512,
            fixed_output_tokens=None,
        )
        factor = 14 * 2
        total_w = result["new_width"] + result["pad_width"]
        total_h = result["new_height"] + result["pad_height"]
        assert total_w % factor == 0, f"Total width {total_w} not aligned"
        assert total_h % factor == 0, f"Total height {total_h} not aligned"

    def test_large_image_scales_down(self) -> None:
        """An image exceeding patch limits should be scaled down."""
        result = navit_resize_image(
            width=8000,
            height=6000,
            patch_size=14,
            merge_kernel_size=2,
            in_patch_limit=16384,
            patch_limit_on_one_side=512,
            fixed_output_tokens=None,
        )
        max_side = 512 * 14
        assert result["new_width"] <= max_side
        assert result["new_height"] <= max_side

    def test_fixed_output_tokens(self) -> None:
        """When fixed_output_tokens is set, num_tokens uses it directly."""
        result = navit_resize_image(
            width=224,
            height=224,
            patch_size=14,
            merge_kernel_size=2,
            in_patch_limit=16384,
            patch_limit_on_one_side=512,
            fixed_output_tokens=256,
        )
        assert result["num_tokens"] == 256

    def test_default_config_typical_image(self) -> None:
        """Standard 1024x768 image with default config."""
        cfg = MediaProcConfig()
        result = navit_resize_image(
            width=1024,
            height=768,
            patch_size=cfg.patch_size,
            merge_kernel_size=cfg.merge_kernel_size,
            in_patch_limit=cfg.in_patch_limit,
            patch_limit_on_one_side=cfg.patch_limit_on_one_side,
            fixed_output_tokens=cfg.fixed_output_tokens,
        )
        factor = cfg.patch_size * cfg.merge_kernel_size
        total_w = result["new_width"] + result["pad_width"]
        total_h = result["new_height"] + result["pad_height"]
        assert total_w % factor == 0
        assert total_h % factor == 0
        assert result["num_tokens"] == (total_h // factor) * (total_w // factor)

    @pytest.mark.parametrize(
        "width,height",
        [
            # Patch grids that scale narrowly under in_patch_limit=16384
            # but pad back over by one row + one column. Each previously
            # OOM'd the vision encoder under shadow traffic.
            (1204, 2716),
            (1484, 2212),
            (1512, 2184),
        ],
        ids=["1204x2716", "1484x2212", "1512x2184"],
    )
    def test_post_pad_patches_within_cap(self, width: int, height: int) -> None:
        """Post-pad patch count must not exceed ``in_patch_limit``."""
        cfg = MediaProcConfig()
        result = navit_resize_image(
            width=width,
            height=height,
            patch_size=cfg.patch_size,
            merge_kernel_size=cfg.merge_kernel_size,
            in_patch_limit=cfg.in_patch_limit,
            patch_limit_on_one_side=cfg.patch_limit_on_one_side,
            fixed_output_tokens=cfg.fixed_output_tokens,
        )
        merge_sq = cfg.merge_kernel_size * cfg.merge_kernel_size
        post_pad_patches = result["num_tokens"] * merge_sq
        assert post_pad_patches <= cfg.in_patch_limit, (
            f"{width}x{height}: {post_pad_patches} patches "
            f"> in_patch_limit={cfg.in_patch_limit}"
        )


# ---------------------------------------------------------------------------
# navit_resize_video tests
# ---------------------------------------------------------------------------


class TestNavitResizeVideo:
    """Tests for the NaViT video resize calculation."""

    def test_basic_video_resize(self) -> None:
        result = navit_resize_video(
            width=640,
            height=480,
            nframes=10,
            avg_fps=30.0,
            sample_fps=2.0,
            patch_size=14,
            merge_kernel_size=2,
            in_patch_limit_each_frame=4096,
            patch_limit_on_one_side=512,
            in_patch_limit_total=None,
            max_num_frames_each_video=None,
            fixed_output_tokens_each_frame=None,
        )
        factor = 14 * 2
        total_w = result["new_width"] + result["pad_width"]
        total_h = result["new_height"] + result["pad_height"]
        assert total_w % factor == 0
        assert total_h % factor == 0

    def test_max_frames_clipping(self) -> None:
        result = navit_resize_video(
            width=640,
            height=480,
            nframes=1000,
            avg_fps=30.0,
            sample_fps=2.0,
            patch_size=14,
            merge_kernel_size=2,
            in_patch_limit_each_frame=4096,
            patch_limit_on_one_side=512,
            in_patch_limit_total=None,
            max_num_frames_each_video=8,
            fixed_output_tokens_each_frame=None,
        )
        assert result["sampled_nframes"] <= 8


# ---------------------------------------------------------------------------
# navit_patchify tests
# ---------------------------------------------------------------------------


class TestNavitPatchify:
    """Tests for the patchification function."""

    def test_output_shapes(self) -> None:
        ps = 14
        t, h, w, c = 1, 28, 42, 3
        pixels = np.random.rand(t, h, w, c).astype(np.float32)
        result = navit_patchify(pixels, ps)

        n_patches = t * (h // ps) * (w // ps)
        assert result["pixel_values"].shape == (n_patches, c, ps, ps)
        np.testing.assert_array_equal(result["grid_thw"], [t, h // ps, w // ps])

    def test_multi_frame(self) -> None:
        """Video chunk with multiple temporal frames."""
        ps = 14
        t, h, w, c = 4, 28, 28, 3
        pixels = np.random.rand(t, h, w, c).astype(np.float32)
        result = navit_patchify(pixels, ps)

        n_patches = t * (h // ps) * (w // ps)
        assert result["pixel_values"].shape == (n_patches, c, ps, ps)
        assert result["grid_thw"][0] == t

    def test_pixel_content_preserved(self) -> None:
        """Verify that patchified values match the original pixel values."""
        ps = 14
        pixels = np.arange(1 * 14 * 14 * 3, dtype=np.float32).reshape(
            1, 14, 14, 3
        )
        result = navit_patchify(pixels, ps)
        assert result["pixel_values"].shape == (1, 3, 14, 14)
        reconstructed = result["pixel_values"][0].transpose(1, 2, 0)
        np.testing.assert_array_equal(reconstructed, pixels[0])


# ---------------------------------------------------------------------------
# normalize tests
# ---------------------------------------------------------------------------


class TestNormalize:
    """Tests for the normalization function."""

    def test_zero_pixel(self) -> None:
        """Black pixel with mean=0.5, std=0.5 should become -1.0."""
        x = np.zeros((1, 1, 1, 3), dtype=np.uint8)
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std_inv = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        out = normalize(x, mean, std_inv)
        np.testing.assert_allclose(out, -1.0, atol=1e-6)

    def test_max_pixel(self) -> None:
        """White pixel (255) with mean=0.5, std=0.5 should become ~1.0."""
        x = np.full((1, 1, 1, 3), 255, dtype=np.uint8)
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std_inv = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        out = normalize(x, mean, std_inv)
        np.testing.assert_allclose(out, 1.0, atol=1e-2)

    def test_mid_pixel(self) -> None:
        """Pixel 127 with mean=0.5, std=0.5 should be near 0."""
        x = np.full((1, 1, 1, 3), 127, dtype=np.uint8)
        mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        std_inv = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        out = normalize(x, mean, std_inv)
        np.testing.assert_allclose(out, 0.0, atol=0.01)


# ---------------------------------------------------------------------------
# timestamp_as_str tests
# ---------------------------------------------------------------------------


class TestTimestampAsStr:
    def test_zero(self) -> None:
        assert timestamp_as_str(0.0) == "00:00:00.000"

    def test_with_millis(self) -> None:
        assert timestamp_as_str(1.5) == "00:00:01.500"

    def test_mm_ss_mode(self) -> None:
        assert timestamp_as_str(65.0, "mm:ss") == "01:05"

    def test_mm_ss_fff_mode(self) -> None:
        assert timestamp_as_str(65.123, "mm:ss.fff") == "01:05.123"


# ---------------------------------------------------------------------------
# KimiK2_5VisionProcessor tests
# ---------------------------------------------------------------------------


class TestKimiK2_5VisionProcessor:
    """Tests for the full vision processor pipeline."""

    @pytest.fixture()
    def real_image(self) -> Image.Image:
        return _download_image(KIMI_LOGO_URL)

    def test_single_image(self, real_image: Image.Image) -> None:
        """Process a real image and validate output shapes."""
        processor = KimiK2_5VisionProcessor()
        result = processor.preprocess([{"type": "image", "image": real_image}])

        assert "pixel_values" in result
        assert "grid_thws" in result
        assert result["pixel_values"].ndim == 4
        assert result["pixel_values"].shape[1] == 3
        assert result["pixel_values"].shape[2] == 14
        assert result["pixel_values"].shape[3] == 14
        assert result["grid_thws"].shape == (1, 3)
        assert result["grid_thws"][0, 0] == 1  # temporal dim for image

    def test_multiple_images(self, real_image: Image.Image) -> None:
        """Process multiple images, verify concatenated output."""
        processor = KimiK2_5VisionProcessor()
        small = real_image.resize((160, 120))
        medias = [
            {"type": "image", "image": real_image},
            {"type": "image", "image": small},
        ]
        result = processor.preprocess(medias)

        assert result["grid_thws"].shape == (2, 3)
        total_patches = sum(
            int(np.prod(result["grid_thws"][i])) for i in range(2)
        )
        assert result["pixel_values"].shape[0] == total_patches

    def test_empty_medias(self) -> None:
        processor = KimiK2_5VisionProcessor()
        result = processor.preprocess([])
        assert result == {}

    def test_video_chunk(self) -> None:
        """Process a video chunk from a real video."""
        processor = KimiK2_5VisionProcessor()
        frames = _download_video_frames(KIMI_VIDEO_URL, num_frames=4)
        result = processor.preprocess(
            [{"type": "video_chunk", "video_chunk": frames}]
        )

        assert result["grid_thws"].shape == (1, 3)
        assert result["grid_thws"][0, 0] == len(frames)
        total_patches = int(np.prod(result["grid_thws"][0]))
        assert result["pixel_values"].shape[0] == total_patches

    def test_pixel_value_range(self, real_image: Image.Image) -> None:
        """Normalized pixel values should be in [-1, 1] for mean=0.5, std=0.5."""
        processor = KimiK2_5VisionProcessor()
        result = processor.preprocess([{"type": "image", "image": real_image}])
        pv = result["pixel_values"]
        assert pv.min() >= -1.0 - 1e-6
        assert pv.max() <= 1.0 + 1e-6

    def test_callable_alias(self, real_image: Image.Image) -> None:
        """__call__ should produce the same result as preprocess."""
        processor = KimiK2_5VisionProcessor()
        medias = [{"type": "image", "image": real_image}]
        result_call = processor(medias)
        result_preprocess = processor.preprocess(medias)
        np.testing.assert_array_equal(
            result_call["pixel_values"],
            result_preprocess["pixel_values"],
        )
        np.testing.assert_array_equal(
            result_call["grid_thws"],
            result_preprocess["grid_thws"],
        )


# ---------------------------------------------------------------------------
# KimiK2_5Processor tests
# ---------------------------------------------------------------------------


class TestKimiK2_5Processor:
    """Tests for the top-level processor orchestrator."""

    def test_update_raw_text_no_placeholders(self) -> None:
        proc = KimiK2_5Processor()
        text = "Hello, world!"
        assert proc.update_raw_text(text, []) == text

    def test_update_raw_text_single_video(self) -> None:
        proc = KimiK2_5Processor()
        ph = KimiK2_5Processor.VIDEO_PLACEHOLDER
        text = f"Before {ph} After"
        result = proc.update_raw_text(text, ["[VIDEO_PROMPT]"])
        assert result == "Before [VIDEO_PROMPT] After"

    def test_update_raw_text_multiple_videos(self) -> None:
        proc = KimiK2_5Processor()
        ph = KimiK2_5Processor.VIDEO_PLACEHOLDER
        text = f"A {ph} B {ph} C"
        result = proc.update_raw_text(text, ["[V1]", "[V2]"])
        assert result == "A [V1] B [V2] C"

    def test_image_only_flow(self) -> None:
        """Full image-only flow through KimiK2_5Processor."""
        proc = KimiK2_5Processor()
        image = _make_rgb_image(224, 224)
        medias = [{"type": "image", "image": image}]
        result, text = proc(medias, text="Describe this image.")
        assert "pixel_values" in result
        assert "grid_thws" in result
        assert text == "Describe this image."


# ---------------------------------------------------------------------------
# HuggingFace comparison test (manual, requires network + torch)
# ---------------------------------------------------------------------------


def _to_numpy(tensor_or_array: torch.Tensor | np.ndarray) -> np.ndarray:
    """Converts a torch.Tensor or numpy array to numpy."""
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.numpy()
    return tensor_or_array


def _assert_vision_outputs_match(
    our_result: dict[str, np.ndarray],
    hf_result: dict[str, object],
    *,
    label: str = "",
    mae_threshold: float = 1e-4,
) -> None:
    """Asserts pixel_values and grid_thws match between our and HF outputs."""
    hf_pv = _to_numpy(hf_result["pixel_values"])
    hf_grid = _to_numpy(hf_result["grid_thws"])

    assert our_result["pixel_values"].shape == hf_pv.shape, (
        f"pixel_values shape mismatch{label}: "
        f"{our_result['pixel_values'].shape} vs {hf_pv.shape}"
    )
    assert our_result["grid_thws"].shape == hf_grid.shape, (
        f"grid_thws shape mismatch{label}: "
        f"{our_result['grid_thws'].shape} vs {hf_grid.shape}"
    )
    np.testing.assert_array_equal(
        our_result["grid_thws"],
        hf_grid,
        err_msg=f"Grid dimensions differ{label}",
    )

    mae = float(np.mean(np.abs(our_result["pixel_values"] - hf_pv)))
    abs_err = float(np.max(np.abs(our_result["pixel_values"] - hf_pv)))
    assert mae < mae_threshold, (
        f"MAE {mae:.8f} exceeds threshold {mae_threshold}{label}"
    )
    assert np.allclose(
        our_result["pixel_values"], hf_pv, rtol=1e-4, atol=1e-4
    ), f"Pixel values differ{label}. Max abs err: {abs_err:.8f}, MAE: {mae:.8f}"


class TestVisionProcessorVsHuggingFace:
    """Compare our processor against the HuggingFace reference.

    Only text-only and image inputs are tested here. Video tests are
    excluded because the HuggingFace reference requires the proprietary
    ``mecord`` package for video decoding, which is not publicly available.
    """

    @pytest.fixture()
    def processors(
        self,
    ) -> tuple[KimiK2_5VisionProcessor, Any]:
        hf = AutoImageProcessor.from_pretrained(
            "nvidia/Kimi-K2.5-NVFP4",
            trust_remote_code=True,
        )
        ours = KimiK2_5VisionProcessor(media_proc_cfg=hf.media_proc_cfg)
        return ours, hf

    def test_text_only_no_media_vs_hf(
        self, processors: tuple[KimiK2_5VisionProcessor, Any]
    ) -> None:
        """Text-only input (no media) returns empty from both processors."""
        ours, hf = processors

        our_result = ours.preprocess([])
        hf_result = hf.preprocess([])

        assert our_result == {}, (
            f"Expected empty dict for no media, got {our_result}"
        )
        assert hf_result == {} or len(hf_result) == 0, (
            f"Expected empty HF result for no media, got {hf_result}"
        )

    def test_single_image_vs_hf(
        self, processors: tuple[KimiK2_5VisionProcessor, Any]
    ) -> None:
        """Process the Kimi logo with both processors and compare."""
        ours, hf = processors
        image = _download_image(KIMI_LOGO_URL)
        medias = [{"type": "image", "image": image}]

        _assert_vision_outputs_match(
            ours.preprocess(medias),
            hf.preprocess(medias),
            label=" (kimi-logo)",
        )

    def test_multiple_images_vs_hf(
        self, processors: tuple[KimiK2_5VisionProcessor, Any]
    ) -> None:
        """Multiple copies of a real image at different scales."""
        ours, hf = processors
        image = _download_image(KIMI_LOGO_URL)
        small = image.resize((160, 120))
        medias = [
            {"type": "image", "image": image},
            {"type": "image", "image": small},
        ]

        _assert_vision_outputs_match(
            ours.preprocess(medias),
            hf.preprocess(medias),
            label=" (multi-image real)",
        )
