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

"""Gemma4 video processor using pure numpy/PIL (no torch).

Processes video bytes into per-frame patchified tensors with 2D position IDs,
following the same conventions as the image processor but with per-frame
output and video-specific pooling (``video_seq_length`` soft tokens per frame).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import numpy.typing as npt
from max.pipelines.context import open_video_container
from PIL import Image

from .processing_utils import (
    SUPPORTED_SOFT_TOKENS,
    aspect_ratio_preserving_resize,
)


@dataclass
class VideoMetadata:
    """Per-video metadata returned alongside pixel data.

    Mirrors the ``video_metadata`` objects produced by the upstream
    HuggingFace ``BaseVideoProcessor``.

    Args:
        fps: Source video frame rate (frames per second).
        timestamps: Per-sampled-frame timestamps in seconds.
    """

    fps: float | None = None
    timestamps: list[float] = field(default_factory=list)


def _sample_frame_indices(total_frames: int, num_frames: int) -> list[int]:
    """Uniformly sample *num_frames* indices from *total_frames*."""
    if total_frames <= num_frames:
        return list(range(total_frames))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    return indices


def _decode_frames_at(
    video_bytes: bytes, indices: list[int]
) -> tuple[dict[int, Image.Image], int]:
    """Decodes one pass, keeping only the frames at ``indices``.

    Frames outside ``indices`` are dropped as they stream, so peak memory is
    the kept frames plus one in-flight frame rather than the whole clip.

    Returns:
        A ``(kept, actual_total)`` tuple: the kept frames by index, and how
        many frames the decode actually produced.
    """
    wanted = set(indices)
    kept: dict[int, Image.Image] = {}
    actual = 0
    with open_video_container(io.BytesIO(video_bytes)) as container:
        for idx, frame in enumerate(container.decode(video=0)):
            if idx in wanted:
                kept[idx] = frame.to_image().convert("RGB")
            actual = idx + 1
    return kept, actual


class Gemma4VideoProcessor:
    """Pure numpy/PIL video processor for Gemma4.

    Decodes video bytes into frames, applies aspect-ratio-preserving resize,
    patchifies each frame, and computes 2D position IDs. Padding patches use
    ``(-1, -1)`` positions.

    Args:
        patch_size: The size (resolution) of each patch in pixels.
        max_soft_tokens: Maximum number of soft tokens per video frame after
            pooling. Must be one of ``SUPPORTED_SOFT_TOKENS``.
        pooling_kernel_size: Spatial pooling kernel size applied after
            patchification.
        num_frames: Number of frames to sample from each video.
        do_sample_frames: Whether to sample frames from the video.
        do_resize: Whether to resize frames via aspect-ratio-preserving resize.
        do_rescale: Whether to rescale pixel values by ``rescale_factor``.
        rescale_factor: Multiplicative factor applied to each pixel value when
            ``do_rescale`` is True.
        do_normalize: Whether to normalize pixel values using ``image_mean``
            and ``image_std`` after rescaling.
        image_mean: Per-channel mean used for normalization.
        image_std: Per-channel standard deviation used for normalization.
    """

    def __init__(
        self,
        patch_size: int = 16,
        max_soft_tokens: int = 70,
        pooling_kernel_size: int = 3,
        num_frames: int = 32,
        do_sample_frames: bool = True,
        do_resize: bool = True,
        do_rescale: bool = True,
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = False,
        image_mean: tuple[float, ...] = (0.0, 0.0, 0.0),
        image_std: tuple[float, ...] = (1.0, 1.0, 1.0),
        **unused_kwargs,
    ) -> None:
        if max_soft_tokens not in SUPPORTED_SOFT_TOKENS:
            raise ValueError(
                f"`max_soft_tokens` must be one of {SUPPORTED_SOFT_TOKENS}, "
                f"got {max_soft_tokens}."
            )
        self.patch_size = patch_size
        self.max_soft_tokens = max_soft_tokens
        self.pooling_kernel_size = pooling_kernel_size
        self.num_frames = num_frames
        self.do_sample_frames = do_sample_frames
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)
        self.max_patches = max_soft_tokens * pooling_kernel_size**2

    def _aspect_ratio_preserving_resize(
        self, image: Image.Image
    ) -> Image.Image:
        """Resize preserving aspect ratio to fit within ``max_patches``."""
        return aspect_ratio_preserving_resize(
            image,
            patch_size=self.patch_size,
            max_patches=self.max_patches,
            pooling_kernel_size=self.pooling_kernel_size,
        )

    def _frame_indices(self, total_frames: int) -> list[int]:
        if self.do_sample_frames:
            return _sample_frame_indices(total_frames, self.num_frames)
        return list(range(total_frames))

    def _decode_video(
        self, video_bytes: bytes
    ) -> tuple[list[Image.Image], VideoMetadata]:
        """Decode video bytes into uniformly-sampled PIL frames + metadata.

        Sampling indices are chosen up front (from the container-declared
        frame count, or a counting decode when the header omits it) so only
        the sampled frames are ever materialized as PIL images. Decoding the
        whole clip first peaks at the full video as uncompressed rasters --
        ~100 GB for a 10-minute 1080p clip -- in the API server process.

        Returns:
            A tuple of (sampled_frames, metadata) where metadata contains
            the source fps and per-frame timestamps in seconds.
        """
        with open_video_container(io.BytesIO(video_bytes)) as container:
            stream = container.streams.video[0]
            avg_rate: Fraction | None = stream.average_rate
            fps = float(avg_rate) if avg_rate else None
            total_frames = stream.frames or 0

        if total_frames <= 0:
            # The header omits a frame count; count with a decode that
            # retains nothing.
            with open_video_container(io.BytesIO(video_bytes)) as container:
                total_frames = sum(1 for _ in container.decode(video=0))
        if total_frames <= 0:
            raise ValueError("Video contains no decodable frames.")

        indices = self._frame_indices(total_frames)
        kept, actual_total = _decode_frames_at(video_bytes, indices)
        if actual_total != total_frames:
            # The container header lied about the frame count; resample from
            # the count the decode actually produced.
            if actual_total <= 0:
                raise ValueError("Video contains no decodable frames.")
            indices = self._frame_indices(actual_total)
            kept, _ = _decode_frames_at(video_bytes, indices)

        effective_fps = fps or 24.0
        timestamps = [idx / effective_fps for idx in indices]

        return (
            [kept[i] for i in indices],
            VideoMetadata(fps=fps, timestamps=timestamps),
        )

    def __call__(
        self, videos: list[bytes]
    ) -> tuple[
        list[npt.NDArray[np.float32]],
        list[npt.NDArray[np.int32]],
        list[int],
        list[VideoMetadata],
    ]:
        """Process a list of video byte arrays into per-video frame patches.

        Returns:
            A 4-tuple ``(pixel_values_list, position_ids_list,
            num_soft_tokens_per_video, video_metadata)`` where:

            * ``pixel_values_list[i]``: float32 array of shape
              ``[num_frames_i, max_patches, patch_size² * 3]``, pixel values
              in ``[0, 1]``. Padding patches are zero-filled.
            * ``position_ids_list[i]``: int32 array of shape
              ``[num_frames_i, max_patches, 2]`` with ``(x, y)`` grid
              coordinates. Padding patches use ``(-1, -1)``.
            * ``num_soft_tokens_per_video[i]``: per-frame soft token count
              (= ``patches_per_frame // k²``).
            * ``video_metadata[i]``: ``VideoMetadata`` with source fps and
              per-frame timestamps in seconds.
        """
        patch_size = self.patch_size
        k = self.pooling_kernel_size

        all_pixel_values: list[npt.NDArray[np.float32]] = []
        all_position_ids: list[npt.NDArray[np.int32]] = []
        all_num_soft_tokens: list[int] = []
        all_metadata: list[VideoMetadata] = []

        for video_bytes in videos:
            frames, metadata = self._decode_video(video_bytes)
            if not frames:
                raise ValueError("Video has no frames after decoding.")

            # Determine target size (from first frame).
            first_rgb = frames[0].convert("RGB")
            if self.do_resize:
                first_rgb = self._aspect_ratio_preserving_resize(first_rgb)
            target_w, target_h = first_rgb.size

            num_frames = len(frames)
            patch_height = target_h // patch_size
            patch_width = target_w // patch_size
            num_patches = patch_height * patch_width
            patch_dim = patch_size * patch_size * 3

            # Pre-allocate padded arrays
            video_patches = np.zeros(
                (num_frames, self.max_patches, patch_dim), dtype=np.float32
            )
            video_positions = np.full(
                (num_frames, self.max_patches, 2), -1, dtype=np.int32
            )

            # Compute position IDs (same for all frames since spatial layout
            # is identical after resizing to the same target)
            grid_x, grid_y = np.meshgrid(
                np.arange(patch_width, dtype=np.int32),
                np.arange(patch_height, dtype=np.int32),
            )
            real_positions = np.stack([grid_x, grid_y], axis=-1).reshape(
                num_patches, 2
            )

            for f_idx, frame in enumerate(frames):
                frame_rgb = frame.convert("RGB")
                if self.do_resize:
                    frame_rgb = frame_rgb.resize(
                        (target_w, target_h), Image.Resampling.BICUBIC
                    )
                img_array = np.array(frame_rgb, dtype=np.float32)
                if self.do_rescale:
                    img_array = img_array * self.rescale_factor
                img_array = np.transpose(img_array, (2, 0, 1))  # CHW
                if self.do_normalize:
                    img_array = (
                        img_array - self.image_mean[:, None, None]
                    ) / self.image_std[:, None, None]

                c, _, _ = img_array.shape
                patches = (
                    img_array.reshape(
                        c, patch_height, patch_size, patch_width, patch_size
                    )
                    .transpose(1, 3, 2, 4, 0)
                    .reshape(num_patches, patch_dim)
                )

                video_patches[f_idx, :num_patches, :] = patches
                video_positions[f_idx, :num_patches, :] = real_positions

            num_soft = num_patches // (k**2)

            all_pixel_values.append(video_patches)
            all_position_ids.append(video_positions)
            all_num_soft_tokens.append(num_soft)
            all_metadata.append(metadata)

        return (
            all_pixel_values,
            all_position_ids,
            all_num_soft_tokens,
            all_metadata,
        )
