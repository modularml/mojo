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

"""Kimi K2.5 vision processor.

Equivalent to the HuggingFace ``KimiK25VisionProcessor`` from
``nvidia/Kimi-K2.5-NVFP4``, re-implemented using only PIL and NumPy.
Handles both image and video media.

Reference files:
- ``kimi_k25_vision_processing.py``  (KimiK25VisionProcessor)
- ``kimi_k25_processor.py``          (KimiK25Processor)
- ``media_utils.py``                 (navit_resize_image, navit_patchify, etc.)
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import numpy.typing as npt
from max.pipelines.context.exceptions import InputError
from max.support.math import ceildiv
from PIL import Image, UnidentifiedImageError


@dataclass
class MediaProcConfig:
    """Configuration for Kimi K2.5 vision media processing."""

    in_patch_limit: int = 16384
    patch_size: int = 14
    image_mean: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    image_std: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    merge_kernel_size: int = 2
    fixed_output_tokens: int | None = None
    patch_limit_on_one_side: int = 512
    in_patch_limit_each_frame: int = 4096
    in_patch_limit_video: int | None = None
    sample_fps: float = 2.0
    max_num_frames_each_video: int | None = None
    temporal_merge_kernel_size: int = 4
    timestamp_mode: str = "hh:mm:ss.fff"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MediaProcConfig:
        """Creates a config from a dictionary, ignoring unknown keys."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------


def _to_pil(data: bytes | Image.Image) -> Image.Image:
    """Converts raw bytes or a PIL Image to RGB.

    URL fetching, base64 decoding, and local file reads are handled
    upstream by the serving layer (see ``resolve_image_from_url`` in
    ``max.serve.router.openai_routes``), so this helper only needs to
    accept already-resolved bytes or PIL Image objects.
    """
    if isinstance(data, Image.Image):
        return data.convert("RGB")
    if isinstance(data, bytes):
        try:
            return Image.open(io.BytesIO(data)).convert("RGB")
        except (
            UnidentifiedImageError,
            OSError,
            Image.DecompressionBombError,
        ) as e:
            raise InputError(
                "Invalid image input: image bytes could not be decoded as a "
                "supported image."
            ) from e
    raise ValueError(f"Unsupported data type: {type(data)}")


def ensure_media_type(media: dict[str, Any]) -> dict[str, Any]:
    """Ensures image/video_chunk fields contain PIL Images."""
    if media["type"] == "image":
        media["image"] = _to_pil(media["image"])
        return media
    if media["type"] == "video_chunk":
        media["video_chunk"] = [_to_pil(f) for f in media["video_chunk"]]
        return media
    raise ValueError(f"Unsupported media type: {media['type']}")


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def timestamp_as_str(
    timestamp: float,
    timestamp_mode: str = "hh:mm:ss.fff",
) -> str:
    """Formats a float timestamp as a human-readable string."""
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    millis = f".{int((timestamp % 1) * 1000):03d}"
    if timestamp_mode == "hh:mm:ss.fff":
        return dt.strftime("%H:%M:%S") + millis
    if timestamp_mode == "mm:ss.fff":
        return dt.strftime("%M:%S") + millis
    if timestamp_mode == "mm:ss":
        return dt.strftime("%M:%S")
    raise ValueError(f"Invalid timestamp mode: {timestamp_mode}")


# ---------------------------------------------------------------------------
# Resize / patch helpers (ported from media_utils.py)
# ---------------------------------------------------------------------------


def navit_resize_image(
    width: int,
    height: int,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit: int,
    patch_limit_on_one_side: int,
    fixed_output_tokens: int | None,
) -> dict[str, int]:
    """Calculates target dimensions and padding for NaViT-style resizing.

    The image is scaled so the total patch count stays within
    ``in_patch_limit`` and neither dimension exceeds
    ``patch_limit_on_one_side * patch_size``.  Padding is added so both
    dimensions are divisible by ``merge_kernel_size * patch_size``.
    """
    s1 = math.sqrt(
        in_patch_limit
        / (max(1.0, width // patch_size) * max(1.0, height // patch_size))
    )
    s2 = patch_limit_on_one_side * patch_size / width
    s3 = patch_limit_on_one_side * patch_size / height
    scale = min(1.0, s1, s2, s3)

    new_w = min(
        max(1, int(width * scale)), patch_limit_on_one_side * patch_size
    )
    new_h = min(
        max(1, int(height * scale)), patch_limit_on_one_side * patch_size
    )

    factor = merge_kernel_size * patch_size
    merge_sq = merge_kernel_size * merge_kernel_size

    if fixed_output_tokens is None:
        # The pre-pad scaling bounds patches by in_patch_limit, but padding
        # to the merge-kernel boundary can push the post-pad count above
        # the cap by up to one row plus one column. Work in merge-cell units
        # (what the encoder actually sees) and trim from the longer side
        # until under cap.
        cells_w = ceildiv(new_w, factor)
        cells_h = ceildiv(new_h, factor)
        max_tokens = in_patch_limit // merge_sq
        while cells_w * cells_h > max_tokens:
            if cells_w >= cells_h and cells_w > 1:
                cells_w -= 1
            elif cells_h > 1:
                cells_h -= 1
            else:
                break
        new_w = min(new_w, cells_w * factor)
        new_h = min(new_h, cells_h * factor)

    pad_width = (factor - new_w % factor) % factor
    pad_height = (factor - new_h % factor) % factor

    if fixed_output_tokens is not None:
        num_tokens = fixed_output_tokens
    else:
        token_height = (new_h + pad_height) // factor
        token_width = (new_w + pad_width) // factor
        if token_height * merge_kernel_size > patch_limit_on_one_side:
            raise ValueError(
                f"token_height ({token_height}) * merge_kernel_size"
                f" ({merge_kernel_size}) exceeds patch_limit_on_one_side"
                f" ({patch_limit_on_one_side})"
            )
        if token_width * merge_kernel_size > patch_limit_on_one_side:
            raise ValueError(
                f"token_width ({token_width}) * merge_kernel_size"
                f" ({merge_kernel_size}) exceeds patch_limit_on_one_side"
                f" ({patch_limit_on_one_side})"
            )
        num_tokens = token_height * token_width

    return {
        "num_tokens": num_tokens,
        "new_width": new_w,
        "new_height": new_h,
        "pad_width": pad_width,
        "pad_height": pad_height,
        "sampled_nframes": 1,
    }


def navit_resize_video(
    width: int,
    height: int,
    nframes: int,
    avg_fps: float,
    sample_fps: float,
    patch_size: int,
    merge_kernel_size: int,
    in_patch_limit_each_frame: int,
    patch_limit_on_one_side: int,
    in_patch_limit_total: int | None,
    max_num_frames_each_video: int | None,
    fixed_output_tokens_each_frame: int | None,
) -> dict[str, int]:
    """Calculates video resize dimensions, then delegates to image resize."""
    sample_fps = min(sample_fps, avg_fps)
    sampled_nframes = max(round(nframes * sample_fps / avg_fps), 1)
    if max_num_frames_each_video is not None:
        sampled_nframes = min(sampled_nframes, max_num_frames_each_video)

    if in_patch_limit_total is not None:
        in_patch_limit_each_frame = min(
            round(in_patch_limit_total / sampled_nframes),
            in_patch_limit_each_frame,
        )

    ret = navit_resize_image(
        width,
        height,
        patch_size,
        merge_kernel_size,
        in_patch_limit_each_frame,
        patch_limit_on_one_side,
        fixed_output_tokens_each_frame,
    )
    ret["sampled_nframes"] = sampled_nframes
    return ret


def real_sample_fps_and_max_num_frames(
    type_name: str,
    sample_fps: float,
    max_num_frames_each_video: int | None,
) -> tuple[float, int | None]:
    """Returns effective (sample_fps, max_num_frames) for a media type."""
    if type_name == "video":
        return sample_fps, max_num_frames_each_video
    if type_name == "video_chunk":
        return math.inf, None
    return math.inf, None


# ---------------------------------------------------------------------------
# NumPy image utilities
# ---------------------------------------------------------------------------


def image_to_np(
    image: Image.Image,
    resize_to: tuple[int, int] | None = None,
    mode: str = "resize",
) -> npt.NDArray[np.uint8]:
    """Converts a PIL Image to a uint8 NumPy array, optionally resizing."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL Image, got {type(image)}")
    if resize_to is not None:
        if mode == "resize":
            image = image.resize(resize_to, resample=Image.Resampling.BICUBIC)
        else:
            raise ValueError(f"Unsupported resize mode: {mode}")
    return np.asarray(image)


def normalize(
    x: npt.NDArray[Any],
    mean: npt.NDArray[np.float32],
    std_inv: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Normalizes uint8 pixels: ``(x / 255 - mean) * std_inv``."""
    out = (x / 255.0).astype(np.float32)
    out -= mean
    out *= std_inv
    return out


def navit_patchify(
    pixel_values: npt.NDArray[np.float32],
    patch_size: int,
) -> dict[str, npt.NDArray[Any]]:
    """Reshapes ``(T, H, W, C)`` pixels into NaViT patches.

    Returns ``pixel_values`` of shape ``(n_patches, C, ps, ps)`` and
    ``grid_thw`` of shape ``(3,)`` as ``[T, H//ps, W//ps]``.
    """
    t, h, w, c = pixel_values.shape
    if c != 3:
        raise ValueError(f"pixel_values must have 3 channels, got {c}")

    patches = pixel_values.reshape(
        t, h // patch_size, patch_size, w // patch_size, patch_size, c
    )
    patches = patches.transpose(0, 1, 3, 5, 2, 4)
    patches = patches.reshape(-1, c, patch_size, patch_size)
    grid_thw = np.array([t, h // patch_size, w // patch_size])
    return {"pixel_values": patches, "grid_thw": grid_thw}


# ---------------------------------------------------------------------------
# KimiK2_5VisionProcessor
# ---------------------------------------------------------------------------


class KimiK2_5VisionProcessor:
    """Vision processor for Kimi K2.5, free of torch/torchvision.

    Equivalent to ``KimiK25VisionProcessor`` from the HuggingFace
    ``nvidia/Kimi-K2.5-NVFP4`` repo.  Uses PIL + NumPy only.
    Handles both image and video media.
    """

    def __init__(
        self, media_proc_cfg: MediaProcConfig | dict[str, Any] | None = None
    ) -> None:
        if media_proc_cfg is None:
            self.cfg = MediaProcConfig()
        elif isinstance(media_proc_cfg, dict):
            self.cfg = MediaProcConfig.from_dict(media_proc_cfg)
        else:
            self.cfg = media_proc_cfg
        self.num_frames_per_chunk: int = self.cfg.temporal_merge_kernel_size
        self._image_std_inv = 1.0 / np.array(
            self.cfg.image_std, dtype=np.float32
        )
        self._image_mean = np.array(self.cfg.image_mean, dtype=np.float32)

    @staticmethod
    def make_chunk_prompt(timestamp_text: str) -> str:
        """Builds the text prompt for a single video chunk."""
        return (
            f"{timestamp_text}"
            "<|media_begin|>video<|media_content|>"
            "<|media_pad|><|media_end|>"
        )

    def media_tokens_calculator(self, media: dict[str, Any]) -> int:
        """Returns the number of visual tokens a media item produces."""
        media = ensure_media_type(media)
        return self.get_resize_config(media)["num_tokens"]

    def get_resize_config(self, media_input: dict[str, Any]) -> dict[str, int]:
        """Computes target dimensions for a single media item."""
        if media_input["type"] == "image":
            w, h = media_input["image"].size
            return navit_resize_image(
                w,
                h,
                self.cfg.patch_size,
                self.cfg.merge_kernel_size,
                self.cfg.in_patch_limit,
                self.cfg.patch_limit_on_one_side,
                self.cfg.fixed_output_tokens,
            )
        if media_input["type"] == "video_chunk":
            frame = media_input["video_chunk"][0]
            width, height = frame.size
            num_frames = len(media_input["video_chunk"])
            sample_fps, max_num_frames = real_sample_fps_and_max_num_frames(
                media_input["type"],
                self.cfg.sample_fps,
                self.cfg.max_num_frames_each_video,
            )
            in_patch_limit_each_frame: int | None = (
                self.cfg.in_patch_limit_each_frame
            )
            if in_patch_limit_each_frame is None:
                in_patch_limit_each_frame = self.cfg.in_patch_limit
            return navit_resize_video(
                width,
                height,
                num_frames,
                1.0,
                sample_fps,
                self.cfg.patch_size,
                self.cfg.merge_kernel_size,
                in_patch_limit_each_frame,
                self.cfg.patch_limit_on_one_side,
                self.cfg.in_patch_limit_video,
                max_num_frames,
                self.cfg.fixed_output_tokens,
            )
        raise ValueError(f"Unsupported media type: {media_input['type']}")

    def resize_image(
        self,
        image: Image.Image,
        new_width: int,
        new_height: int,
        pad_width: int,
        pad_height: int,
    ) -> npt.NDArray[np.uint8]:
        """Resizes and zero-pads an image to patch-aligned dimensions."""
        image_np = image_to_np(image, (new_width, new_height), "resize")
        if pad_height > 0 or pad_width > 0:
            image_np = np.pad(
                image_np,
                ((0, pad_height), (0, pad_width), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        return image_np

    def split_video_chunks(
        self, video_url: str | bytes
    ) -> list[dict[str, Any]]:
        """Splits a video into temporal chunks of frames.

        Requires the ``av`` (PyAV) package for video decoding.
        """
        try:
            from max.pipelines.context import open_video_container
        except ImportError as exc:
            raise ImportError(
                "Video processing requires the 'av' package. "
                "Install it with: pip install av"
            ) from exc

        src: str | io.BytesIO = video_url  # type: ignore[assignment]
        if isinstance(video_url, str) and video_url.startswith("data:video/"):
            src = io.BytesIO(base64.b64decode(video_url.split(",", 1)[1]))
        elif isinstance(video_url, bytes):
            src = io.BytesIO(video_url)

        container = open_video_container(src)
        stream = container.streams.video[0]
        fps: float = float(stream.average_rate or stream.base_rate or 30.0)
        num_frames: int = stream.frames or 0

        all_frames: list[Image.Image] = []
        for frame in container.decode(video=0):
            all_frames.append(frame.to_image().convert("RGB"))
        container.close()

        if num_frames <= 0:
            num_frames = len(all_frames)

        sample_fps = min(self.cfg.sample_fps, fps)
        sampled_nframes = max(round(num_frames * sample_fps / fps), 1)

        frame_inds = (
            np.linspace(0, num_frames - 1, sampled_nframes)
            .round()
            .astype(int)
            .tolist()
        )

        temporal_merge = self.cfg.temporal_merge_kernel_size
        sampled_frame_ids: list[int] = []
        chunk_timestamps: list[str] = []

        for i in range(0, len(frame_inds), temporal_merge):
            sampled_frame_ids.extend(frame_inds[i : i + temporal_merge])
            start_time = frame_inds[i] / float(fps)
            chunk_timestamps.append(
                timestamp_as_str(start_time, self.cfg.timestamp_mode)
            )

        pil_frames = [all_frames[idx] for idx in sampled_frame_ids]

        num_chunks = len(chunk_timestamps)
        chunks: list[dict[str, Any]] = []
        for chunk_id in range(num_chunks):
            start = chunk_id * temporal_merge
            end = start + temporal_merge
            chunks.append(
                {
                    "type": "video_chunk",
                    "video_chunk": pil_frames[start:end],
                    "prompt": self.make_chunk_prompt(
                        chunk_timestamps[chunk_id]
                    ),
                }
            )
        return chunks

    def preprocess(
        self,
        medias: list[dict[str, Any]],
    ) -> dict[str, npt.NDArray[Any]]:
        """Preprocesses media items into model-ready NumPy arrays.

        Returns:
            Dictionary with ``pixel_values`` of shape
            ``(total_patches, C, ps, ps)`` and ``grid_thws`` of shape
            ``(N, 3)``, or empty dict when no media is provided.
        """
        if not isinstance(medias, list):
            medias = [medias]
        if not medias:
            return {}

        patch_size = self.cfg.patch_size
        patchified: list[dict[str, npt.NDArray[Any]]] = []

        for item in medias:
            item = ensure_media_type(item)
            cfg = self.get_resize_config(item)
            new_w, new_h = cfg["new_width"], cfg["new_height"]
            pad_w, pad_h = cfg["pad_width"], cfg["pad_height"]

            if item["type"] == "image":
                img_np = self.resize_image(
                    item["image"], new_w, new_h, pad_w, pad_h
                )
                pixels = np.expand_dims(img_np, axis=0)
            elif item["type"] == "video_chunk":
                frame_arrays = [
                    self.resize_image(f, new_w, new_h, pad_w, pad_h)
                    for f in item["video_chunk"]
                ]
                pixels = np.stack(frame_arrays, axis=0)
            else:
                raise ValueError(f"Unsupported media type: {item['type']}")

            normed = normalize(pixels, self._image_mean, self._image_std_inv)
            patchified.append(navit_patchify(normed, patch_size))

        pixel_values = np.concatenate(
            [p["pixel_values"] for p in patchified], axis=0
        )
        grid_thws = np.stack(
            [p["grid_thw"] for p in patchified], axis=0
        ).astype(np.int64)

        return {"pixel_values": pixel_values, "grid_thws": grid_thws}

    def __call__(
        self, medias: list[dict[str, Any]]
    ) -> dict[str, npt.NDArray[Any]]:
        """Alias for :meth:`preprocess`."""
        return self.preprocess(medias)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(cfg={self.cfg})"


# ---------------------------------------------------------------------------
# KimiK2_5Processor  (orchestrator analogous to the HF processor)
# ---------------------------------------------------------------------------


class KimiK2_5Processor:
    """Orchestrates vision preprocessing and video chunk expansion.

    Equivalent to the HuggingFace ``KimiK25Processor`` from
    ``nvidia/Kimi-K2.5-NVFP4/kimi_k25_processor.py`` but without any
    torch or transformers dependency.
    """

    VIDEO_PLACEHOLDER = "<|kimi_k25_video_placeholder|>"

    def __init__(
        self,
        vision_processor: KimiK2_5VisionProcessor | None = None,
        media_proc_cfg: MediaProcConfig | dict[str, Any] | None = None,
    ) -> None:
        self.vision_processor = vision_processor or KimiK2_5VisionProcessor(
            media_proc_cfg
        )

    def update_raw_text(self, text: str, video_prompts: list[str]) -> str:
        """Replaces video placeholders with actual chunk prompts."""
        count = text.count(self.VIDEO_PLACEHOLDER)
        if count == 0:
            return text
        if count != len(video_prompts):
            raise ValueError(
                f"Mismatch: {count} placeholders vs"
                f" {len(video_prompts)} prompts"
            )
        parts = text.split(self.VIDEO_PLACEHOLDER)
        if len(parts) != len(video_prompts) + 1:
            raise ValueError(
                f"Expected {len(video_prompts) + 1} parts after split,"
                f" got {len(parts)}"
            )
        return (
            "".join(
                parts[i] + video_prompts[i] for i in range(len(video_prompts))
            )
            + parts[-1]
        )

    def preprocess_medias(
        self, medias: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Expands videos into chunks, collects per-video prompts."""
        expanded: list[dict[str, Any]] = []
        video_prompts: list[str] = []
        for media in medias:
            if media["type"] == "image":
                expanded.append(media)
            elif media["type"] == "video":
                chunks = self.vision_processor.split_video_chunks(
                    media["video"]
                )
                expanded.extend(chunks)
                video_prompts.append("".join(c["prompt"] for c in chunks))
            else:
                raise ValueError(f"Unsupported media type: {media['type']}")
        return expanded, video_prompts

    def __call__(
        self,
        medias: list[dict[str, Any]],
        text: str | None = None,
    ) -> tuple[dict[str, npt.NDArray[Any]], str | None]:
        """Preprocesses media and optionally updates text prompts.

        Returns:
            ``(preprocessed_dict, updated_text)`` where the dict has
            ``pixel_values`` and ``grid_thws`` (or is empty).
        """
        expanded_medias, video_prompts = self.preprocess_medias(medias)
        preprocessed = self.vision_processor.preprocess(expanded_medias)
        if text is not None:
            text = self.update_raw_text(text, video_prompts)
        return preprocessed, text

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(vision_processor={self.vision_processor!r})"
