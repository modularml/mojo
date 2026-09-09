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

"""Unit tests for Gemma4VideoProcessor.

These tests validate shapes, position IDs, and padding without requiring
actual video files or torch.
"""

from __future__ import annotations

import contextlib
import io
from fractions import Fraction
from types import SimpleNamespace
from typing import IO

import av
import numpy as np
import pytest
from max.pipelines.architectures.gemma4 import video_processor as vp_module
from max.pipelines.architectures.gemma4.video_processor import (
    Gemma4VideoProcessor,
    VideoMetadata,
    _sample_frame_indices,
)

PATCH_SIZE = 16
POOLING_K = 3
VIDEO_SOFT_TOKENS = 70
MAX_PATCHES = VIDEO_SOFT_TOKENS * POOLING_K**2  # 630


def _make_synthetic_video_bytes(
    width: int = 320,
    height: int = 240,
    num_frames: int = 8,
    container_format: str = "mp4",
) -> bytes:
    """Create a synthetic video as raw bytes using PyAV."""
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format=container_format)
    stream = container.add_stream("libx264", rate=24)
    assert isinstance(stream, av.VideoStream)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    for i in range(num_frames):
        arr = np.full((height, width, 3), fill_value=i * 20, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return buf.getvalue()


class TestSampleFrameIndices:
    def test_fewer_frames_than_requested(self) -> None:
        indices = _sample_frame_indices(total_frames=3, num_frames=8)
        assert indices == [0, 1, 2]

    def test_exact_frames(self) -> None:
        indices = _sample_frame_indices(total_frames=8, num_frames=8)
        assert indices == list(range(8))

    def test_more_frames_uniform(self) -> None:
        indices = _sample_frame_indices(total_frames=100, num_frames=4)
        assert len(indices) == 4
        assert indices[0] == 0
        assert indices[-1] == 99
        # Uniformly spaced
        diffs = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
        assert all(d > 0 for d in diffs)


class TestGemma4VideoProcessor:
    @pytest.fixture
    def processor(self) -> Gemma4VideoProcessor:
        return Gemma4VideoProcessor(
            patch_size=PATCH_SIZE,
            max_soft_tokens=VIDEO_SOFT_TOKENS,
            pooling_kernel_size=POOLING_K,
            num_frames=4,
        )

    def test_invalid_max_soft_tokens(self) -> None:
        with pytest.raises(ValueError, match="max_soft_tokens"):
            Gemma4VideoProcessor(max_soft_tokens=42)

    def test_output_shapes(self, processor: Gemma4VideoProcessor) -> None:
        """Test that a synthetic video produces correct output shapes."""
        video_bytes = _make_synthetic_video_bytes(
            width=320, height=240, num_frames=8
        )
        pixel_values, position_ids, num_soft, metadata = processor(
            [video_bytes]
        )

        assert len(pixel_values) == 1
        assert len(position_ids) == 1
        assert len(num_soft) == 1
        assert len(metadata) == 1

        pv = pixel_values[0]
        pos = position_ids[0]

        # pixel_values: [num_sampled_frames, max_patches, patch_dim]
        num_sampled_frames = min(8, processor.num_frames)
        patch_dim = PATCH_SIZE * PATCH_SIZE * 3
        assert pv.shape == (num_sampled_frames, MAX_PATCHES, patch_dim)
        assert pv.dtype == np.float32

        # position_ids: [num_sampled_frames, max_patches, 2]
        assert pos.shape == (num_sampled_frames, MAX_PATCHES, 2)
        assert pos.dtype == np.int32

    def test_padding_uses_neg_one(
        self, processor: Gemma4VideoProcessor
    ) -> None:
        """Verify that padding patches have position (-1, -1)."""
        video_bytes = _make_synthetic_video_bytes(
            width=160, height=160, num_frames=4
        )
        _, position_ids, _, _ = processor([video_bytes])

        pos = position_ids[0]  # [num_frames, max_patches, 2]

        for f in range(pos.shape[0]):
            real_mask = pos[f, :, 0] >= 0
            n_real = int(real_mask.sum())
            assert n_real > 0, "No real patches found"
            assert n_real <= MAX_PATCHES
            # Padding positions should be -1
            padding_pos = pos[f, n_real:, :]
            if padding_pos.shape[0] > 0:
                assert np.all(padding_pos == -1)

    def test_pixel_values_in_zero_one(
        self, processor: Gemma4VideoProcessor
    ) -> None:
        """Pixel values should be rescaled to [0, 1]."""
        video_bytes = _make_synthetic_video_bytes(
            width=320, height=240, num_frames=4
        )
        pixel_values, _, _, _ = processor([video_bytes])
        pv = pixel_values[0]
        # Only check real (non-padding) values
        assert np.all(pv >= 0.0)
        assert np.all(pv <= 1.0)

    def test_num_soft_tokens_formula(
        self, processor: Gemma4VideoProcessor
    ) -> None:
        """Verify num_soft_tokens = patches_per_frame // k² (per-frame)."""
        video_bytes = _make_synthetic_video_bytes(
            width=320, height=240, num_frames=6
        )
        _, position_ids, num_soft, _ = processor([video_bytes])

        pos = position_ids[0]

        # All frames have the same spatial layout after resize
        real_mask = pos[0, :, 0] >= 0
        patches_per_frame = int(real_mask.sum())
        expected = patches_per_frame // (POOLING_K**2)
        assert num_soft[0] == expected

    def test_multiple_videos(self, processor: Gemma4VideoProcessor) -> None:
        """Processing multiple videos returns per-video results."""
        v1 = _make_synthetic_video_bytes(width=320, height=240, num_frames=4)
        v2 = _make_synthetic_video_bytes(width=160, height=160, num_frames=8)
        pixel_values, position_ids, num_soft, metadata = processor([v1, v2])

        assert len(pixel_values) == 2
        assert len(position_ids) == 2
        assert len(num_soft) == 2
        assert len(metadata) == 2


class TestBoundedDecode:
    """The decode keeps only sampled frames, never the whole clip.

    Regression tests for the decode-all-frames-then-sample implementation,
    whose peak memory was the entire video as uncompressed PIL rasters.
    """

    @pytest.fixture
    def processor(self) -> Gemma4VideoProcessor:
        return Gemma4VideoProcessor(
            patch_size=PATCH_SIZE,
            max_soft_tokens=VIDEO_SOFT_TOKENS,
            pooling_kernel_size=POOLING_K,
            num_frames=4,
        )

    def test_samples_expected_indices(
        self, processor: Gemma4VideoProcessor
    ) -> None:
        """Timestamps identify which source frames were kept."""
        video_bytes = _make_synthetic_video_bytes(
            width=320, height=240, num_frames=8
        )
        pixel_values, _, _, metadata = processor([video_bytes])

        meta = metadata[0]
        assert meta.fps is not None
        sampled = [round(t * meta.fps) for t in meta.timestamps]
        assert sampled == [0, 2, 4, 7]
        assert pixel_values[0].shape[0] == 4

        # Source frames are constant-fill with brightness increasing per
        # index, so the kept frames' means must strictly increase — catches
        # off-by-one in which frames the streaming decode retained.
        means = pixel_values[0].mean(axis=(1, 2))
        assert np.all(np.diff(means) > 0)

    def test_no_declared_frame_count(
        self, processor: Gemma4VideoProcessor
    ) -> None:
        """A container without a frame count uses the counting-decode fallback."""
        video_bytes = _make_synthetic_video_bytes(
            width=320, height=240, num_frames=8, container_format="mpegts"
        )
        # Premise: MPEG-TS carries no frame count. If PyAV starts reporting
        # one, this test no longer exercises the fallback — update it.
        with av.open(io.BytesIO(video_bytes)) as container:
            assert container.streams.video[0].frames == 0

        pixel_values, _, _, metadata = processor([video_bytes])
        assert pixel_values[0].shape[0] == 4
        assert len(metadata[0].timestamps) == 4

    def test_wrong_declared_frame_count(
        self,
        processor: Gemma4VideoProcessor,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A lying container header falls back to the actual decoded count."""
        real_open = vp_module.open_video_container
        # The first open is the metadata probe; hand it a synthetic container
        # declaring 1000 frames. Later opens (the decode passes) get the
        # real container.
        lying_probe = SimpleNamespace(
            streams=SimpleNamespace(
                video=[
                    SimpleNamespace(average_rate=Fraction(24, 1), frames=1000)
                ]
            )
        )
        probe = iter([contextlib.nullcontext(lying_probe)])

        def fake_open(source: str | IO[bytes]) -> object:
            return next(probe, None) or real_open(source)

        monkeypatch.setattr(vp_module, "open_video_container", fake_open)
        video_bytes = _make_synthetic_video_bytes(
            width=320, height=240, num_frames=8
        )
        pixel_values, _, _, metadata = processor([video_bytes])

        meta = metadata[0]
        assert meta.fps is not None
        # Indices must come from the true count (8), not the declared 1000.
        sampled = [round(t * meta.fps) for t in meta.timestamps]
        assert sampled == [0, 2, 4, 7]
        assert pixel_values[0].shape[0] == 4


class TestVideoMetadata:
    """Tests for VideoMetadata returned by the processor."""

    @pytest.fixture
    def processor(self) -> Gemma4VideoProcessor:
        return Gemma4VideoProcessor(
            patch_size=PATCH_SIZE,
            max_soft_tokens=VIDEO_SOFT_TOKENS,
            pooling_kernel_size=POOLING_K,
            num_frames=4,
        )

    def test_metadata_has_fps(self, processor: Gemma4VideoProcessor) -> None:
        """VideoMetadata should contain the source video fps."""
        video_bytes = _make_synthetic_video_bytes(
            width=320, height=240, num_frames=8
        )
        _, _, _, metadata = processor([video_bytes])

        meta = metadata[0]
        assert isinstance(meta, VideoMetadata)
        assert meta.fps is not None
        assert meta.fps > 0

    def test_metadata_timestamps_count(
        self, processor: Gemma4VideoProcessor
    ) -> None:
        """Timestamps list length should match the number of sampled frames."""
        video_bytes = _make_synthetic_video_bytes(
            width=320, height=240, num_frames=8
        )
        pixel_values, _, _, metadata = processor([video_bytes])

        num_sampled = pixel_values[0].shape[0]
        assert len(metadata[0].timestamps) == num_sampled

    def test_metadata_timestamps_monotonic(
        self, processor: Gemma4VideoProcessor
    ) -> None:
        """Timestamps should be monotonically non-decreasing."""
        video_bytes = _make_synthetic_video_bytes(
            width=320, height=240, num_frames=8
        )
        _, _, _, metadata = processor([video_bytes])

        ts = metadata[0].timestamps
        for i in range(1, len(ts)):
            assert ts[i] >= ts[i - 1]

    def test_metadata_timestamps_start_at_zero(
        self, processor: Gemma4VideoProcessor
    ) -> None:
        """First frame timestamp should be 0.0."""
        video_bytes = _make_synthetic_video_bytes(
            width=320, height=240, num_frames=8
        )
        _, _, _, metadata = processor([video_bytes])

        assert metadata[0].timestamps[0] == 0.0


class TestVideoProcessorPreprocessParams:
    """Tests for configurable preprocessing parameters."""

    def test_do_rescale_false(self) -> None:
        """With do_rescale=False pixel values stay in [0, 255]."""
        proc = Gemma4VideoProcessor(
            patch_size=PATCH_SIZE,
            max_soft_tokens=VIDEO_SOFT_TOKENS,
            pooling_kernel_size=POOLING_K,
            num_frames=2,
            do_rescale=False,
        )
        video_bytes = _make_synthetic_video_bytes(
            width=320, height=240, num_frames=4
        )
        pixel_values, _, _, _ = proc([video_bytes])
        pv = pixel_values[0]
        assert pv.max() > 1.0, "Expected unscaled pixel values > 1.0"

    def test_custom_rescale_factor(self) -> None:
        """A custom rescale_factor should scale pixel values accordingly."""
        video_bytes = _make_synthetic_video_bytes(
            width=320, height=240, num_frames=4
        )
        default_proc = Gemma4VideoProcessor(
            patch_size=PATCH_SIZE,
            max_soft_tokens=VIDEO_SOFT_TOKENS,
            pooling_kernel_size=POOLING_K,
            num_frames=2,
        )
        custom_factor = 1.0 / 128.0
        custom_proc = Gemma4VideoProcessor(
            patch_size=PATCH_SIZE,
            max_soft_tokens=VIDEO_SOFT_TOKENS,
            pooling_kernel_size=POOLING_K,
            num_frames=2,
            rescale_factor=custom_factor,
        )
        default_pv, _, _, _ = default_proc([video_bytes])
        custom_pv, _, _, _ = custom_proc([video_bytes])
        ratio = custom_factor / (1.0 / 255.0)  # ≈ 1.992
        # Non-zero real patches should differ by the factor ratio
        mask = default_pv[0] != 0
        if mask.any():
            np.testing.assert_allclose(
                custom_pv[0][mask],
                default_pv[0][mask] * ratio,
                rtol=1e-4,
            )

    def test_do_normalize(self) -> None:
        """With do_normalize=True, values should be shifted by mean/std."""
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        proc = Gemma4VideoProcessor(
            patch_size=PATCH_SIZE,
            max_soft_tokens=VIDEO_SOFT_TOKENS,
            pooling_kernel_size=POOLING_K,
            num_frames=2,
            do_normalize=True,
            image_mean=mean,
            image_std=std,
        )
        video_bytes = _make_synthetic_video_bytes(
            width=320, height=240, num_frames=4
        )
        pixel_values, _, _, _ = proc([video_bytes])
        pv = pixel_values[0]
        real_mask = pv.any(axis=-1)
        real_vals = pv[real_mask]
        # Normalized values: (v/255 - 0.5) / 0.5 lies in [-1, 1]
        assert real_vals.min() >= -1.1
        assert real_vals.max() <= 1.1

    def test_do_resize_false(self) -> None:
        """With do_resize=False, frame dimensions stay as-is."""
        width, height = 48, 48  # must be divisible by patch_size
        proc = Gemma4VideoProcessor(
            patch_size=PATCH_SIZE,
            max_soft_tokens=VIDEO_SOFT_TOKENS,
            pooling_kernel_size=POOLING_K,
            num_frames=2,
            do_resize=False,
        )
        video_bytes = _make_synthetic_video_bytes(
            width=width, height=height, num_frames=2
        )
        _, position_ids, _, _ = proc([video_bytes])
        expected_patches = (height // PATCH_SIZE) * (width // PATCH_SIZE)
        pos = position_ids[0]
        real_mask = pos[0, :, 0] >= 0
        assert int(real_mask.sum()) == expected_patches
