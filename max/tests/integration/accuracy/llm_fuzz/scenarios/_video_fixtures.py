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
"""Synthetic video fixtures, built from the standard library alone.

llm-fuzz ships as a standalone tool that runs against a live endpoint, so it
carries no numpy, no PIL and no ``av`` -- the image fixtures hand-write PNG
from ``zlib`` and ``struct`` for that reason, and video has to clear the same
bar while producing something FFmpeg will actually decode.

The container is AVI carrying PNG-coded frames. Two properties decide it:

* Each frame is a complete PNG file, so ``_image_fixtures.png_bytes`` is the
  frame encoder unchanged -- including its prefix cache, which is what makes a
  600-frame fixture cost about as much to build as a single image.
* The AVI header carries a frame count. That matters more than it looks:
  ``video_preprocessing`` falls back to decord whenever container metadata
  reports no frame count (``_container_total_frames`` returning 0), and decord
  reads a much narrower set of formats than PyAV does. A container that
  declares its length keeps the server on the PyAV path these scenarios mean
  to exercise. An APNG -- otherwise the obvious choice, being pure PNG -- is
  rejected on exactly this point: PyAV decodes it, but reports 0 frames.

Sizes stay modest because the frames compress: the largest fixture here, 666
frames at 672x672 (the vendor pixel cap), is under 6 MiB on the wire.
"""

from __future__ import annotations

import base64
import struct
from typing import Any

from scenarios._image_fixtures import MERGED_PATCH, png_bytes, resized_dims

# --- MiniMax-M3 vendor limits (max_private/minimax_m3/vision/video_validation.py) ---

VIDEO_MAX_COUNT = 20
# w * h * num_frames, checked after the resize rules run.
VIDEO_MAX_TOTAL_PIXELS = 301_056_000
VIDEO_MAX_BYTES = 512 * 1024 * 1024
FPS_MIN = 0.2
FPS_MAX = 5.0

# ``detail`` -> long-side limit. The video tiers are much smaller than the
# image ones, and the default is 672 rather than 2016, so a max-size video
# frame needs ``detail="high"`` to reach 1288.
VIDEO_DETAIL_LONG_SIDE = {
    "low": 504,
    "default": 672,
    "high": 1288,
    "auto": 672,
}
VIDEO_DEFAULT_LONG_SIDE = VIDEO_DETAIL_LONG_SIDE["default"]
VIDEO_MAX_LONG_SIDE = VIDEO_DETAIL_LONG_SIDE["high"]

# The vision encoder groups this many sampled frames into one temporal patch,
# so the token count is per patch, not per frame
# (max_private/minimax_m3/vision/config.py).
TEMPORAL_PATCH = 2

# Frames within one fixture differ only by a PNG ``tEXt`` chunk. The pixel data
# -- and therefore the expensive zlib pass -- is shared through
# ``_image_fixtures``'s prefix cache, so frame count is nearly free to scale.
# Distinct bytes per frame still keep the container honest: every frame is its
# own keyframe and the demuxer never coalesces them.
_AVI_FOURCC = b"MPNG"


def _riff(fourcc: bytes, payload: bytes) -> bytes:
    """Wraps a payload in a RIFF chunk, padded to the required even length."""
    return (
        fourcc
        + struct.pack("<I", len(payload))
        + payload
        + (b"\x00" if len(payload) % 2 else b"")
    )


def video_bytes(
    width: int,
    height: int,
    frames: int,
    native_fps: float = 1.0,
    nonce: str = "",
) -> bytes:
    """Returns a decodable AVI carrying ``frames`` PNG-coded frames.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        frames: Number of frames in the container.
        native_fps: Frame rate written into the header. The server refuses a
            sampling ``fps`` above this (``validate_fps_within_native``), so it
            bounds how many frames a request can ask for.
        nonce: Mixed into every frame, so two fixtures of the same shape that
            differ only here decode identically but miss the server's
            byte-keyed preprocess cache.
    """
    payloads = [
        png_bytes(width, height, f"{nonce}-f{i}") for i in range(frames)
    ]
    largest = max(len(p) for p in payloads)

    movi_payload = b"movi"
    index = []
    for payload in payloads:
        # Offsets in idx1 are relative to the start of the 'movi' fourcc.
        index.append(
            b"00dc" + struct.pack("<III", 0x10, len(movi_payload), len(payload))
        )
        movi_payload += (
            b"00dc"
            + struct.pack("<I", len(payload))
            + payload
            + (b"\x00" if len(payload) % 2 else b"")
        )

    avih = _riff(
        b"avih",
        struct.pack(
            "<IIIIIIIIIIIIII",
            int(1e6 / native_fps),  # microseconds per frame
            0,
            0,
            0x10,  # AVIF_HASINDEX
            frames,  # dwTotalFrames -- what keeps the server off decord
            0,
            1,  # one stream
            largest,
            width,
            height,
            0,
            0,
            0,
            0,
        ),
    )
    # dwScale/dwRate carry the rate as a rational, so a fractional fps (the
    # 0.2 floor, say) survives the round trip that dwMicroSecPerFrame alone
    # would round away.
    strh = _riff(
        b"strh",
        b"vids"
        + _AVI_FOURCC
        + struct.pack(
            "<IHHIIIIIIiIhhhh",
            0,
            0,
            0,
            0,
            1000,
            round(native_fps * 1000),
            0,
            frames,
            largest,
            -1,
            0,
            0,
            0,
            width,
            height,
        ),
    )
    strf = _riff(
        b"strf",
        struct.pack(
            "<IiiHH4sIiiII",
            40,
            width,
            height,
            1,
            24,
            _AVI_FOURCC,
            width * height * 3,
            0,
            0,
            0,
            0,
        ),
    )
    hdrl = _riff(
        b"LIST", b"hdrl" + avih + _riff(b"LIST", b"strl" + strh + strf)
    )
    idx1 = _riff(b"idx1", b"".join(index))
    body = b"AVI " + hdrl + _riff(b"LIST", movi_payload) + idx1
    return b"RIFF" + struct.pack("<I", len(body)) + body


def data_url(
    width: int,
    height: int,
    frames: int,
    native_fps: float = 1.0,
    nonce: str = "",
) -> str:
    encoded = base64.b64encode(
        video_bytes(width, height, frames, native_fps, nonce)
    ).decode("ascii")
    return f"data:video/x-msvideo;base64,{encoded}"


def video_part(
    width: int,
    height: int,
    frames: int,
    native_fps: float = 1.0,
    nonce: str = "",
    *,
    fps: float | None = None,
    max_frames: int | None = None,
    detail: str | None = None,
    max_long_side_pixel: int | None = None,
) -> dict[str, Any]:
    """Builds one OpenAI ``video_url`` content part.

    The router reads ``fps``, ``max_frames``, ``detail`` and
    ``max_long_side_pixel`` from inside the ``video_url`` object
    (``openai_routes.py``), the same way it reads the image sizing hints.

    Note the two frame rates: ``native_fps`` is encoded into the container,
    while ``fps`` asks the server to *sample* at that rate. A request whose
    ``fps`` exceeds the container's rate is a clean 4xx, because the vendor
    never invents frames.
    """
    video_url: dict[str, Any] = {
        "url": data_url(width, height, frames, native_fps, nonce)
    }
    if fps is not None:
        video_url["fps"] = fps
    if max_frames is not None:
        video_url["max_frames"] = max_frames
    if detail is not None:
        video_url["detail"] = detail
    if max_long_side_pixel is not None:
        video_url["max_long_side_pixel"] = max_long_side_pixel
    return {"type": "video_url", "video_url": video_url}


def video_payload(
    model: str,
    parts: list[dict[str, Any]],
    prompt: str = "Describe these videos in one word.",
    max_tokens: int = 16,
) -> dict[str, Any]:
    """Wraps video parts into a chat-completions request body."""
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [*parts, {"type": "text", "text": prompt}],
            }
        ],
        "max_tokens": max_tokens,
    }


def sampled_frames(frames: int, native_fps: float, fps: float | None) -> int:
    """Frames the server keeps after sampling a clip at ``fps``.

    Mirrors ``sample_frame_indices``: frame 0, then one every ``1/fps``
    seconds of video time, plus the final frame. Worth stating plainly because
    the ``max_frames`` argument threaded through that call is *ignored* -- the
    cap was removed so long clips keep their final frame -- so nothing bounds
    this but the clip's own length.
    """
    if fps is None or fps >= native_fps:
        return frames
    step = native_fps / fps
    kept = int((frames - 1) / step) + 1
    # The walk always keeps the final frame, even when the stride overshoots it.
    return kept + (1 if (kept - 1) * step < frames - 1 else 0)


def estimate_tokens(
    width: int,
    height: int,
    frames: int,
    native_fps: float = 1.0,
    fps: float | None = None,
    detail: str | None = None,
) -> int:
    """Approximates the vision tokens one video part contributes.

    Each temporal patch of ``TEMPORAL_PATCH`` sampled frames is encoded at the
    post-resize frame size, and every 28x28 block within it becomes one merged
    token.
    """
    long_side = VIDEO_DETAIL_LONG_SIDE.get(
        detail or "default", VIDEO_DEFAULT_LONG_SIDE
    )
    rw, rh = resized_dims(width, height, long_side)
    per_frame = (rw // MERGED_PATCH) * (rh // MERGED_PATCH)
    kept = sampled_frames(frames, native_fps, fps)
    return per_frame * -(-kept // TEMPORAL_PATCH)


def frames_for_pixels(width: int, height: int, pixels: int) -> int:
    """Frame count whose post-resize pixel total lands just under ``pixels``."""
    rw, rh = resized_dims(width, height, VIDEO_DEFAULT_LONG_SIDE)
    return max(1, pixels // (rw * rh))
