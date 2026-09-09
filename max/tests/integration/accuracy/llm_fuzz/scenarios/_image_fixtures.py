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
"""Synthetic image fixtures for vision stress scenarios.

Encodes PNGs with nothing but ``zlib`` + ``struct`` so the fuzz binary keeps
its current dependency set (openai, jsonschema, pydantic -- no Pillow, no
numpy).

Two properties drive the design:

*Cheap uniqueness.* The server keys its preprocessed-image cache on
``hash_image(raw_bytes, size_tier)``, so replaying identical bytes exercises
only the cache-hit path -- and the production hang this feeds
(MXSERV-395) was last seen logging ``encoding N uncached image(s)``. Forcing
a miss by perturbing pixels would mean re-running zlib over a 38 MB buffer
per image. Instead the pixel payload is compressed once per size and cached,
and uniqueness comes from a ``tEXt`` chunk appended before ``IEND``: the
decoded image is bit-identical, the encoded bytes are not.

*Bounded transfer size.* Preprocessing cost scales with pixel count, not
encoded bytes, so the block pattern is deliberately coarse (and coarser as
images get larger). A 3584x3584 fixture carries the full 12.8M-pixel
preprocessing cost while staying in the tens of KB on the wire, which is
what makes a dozens-of-images, million-token request practical to send.
"""

from __future__ import annotations

import base64
import struct
import zlib
from typing import Any

# --- MiniMax-M3 vendor limits (max_private/minimax_m3/vision/video_validation.py) ---

IMAGE_MAX_COUNT = 200
IMAGE_MAX_TOTAL_PIXELS = 12_845_056
MIN_SHORT_SIDE_PIXEL = 112
MAX_LONG_SIDE = 3584

# ``detail`` -> long-side limit. Anything sent without ``detail`` is capped at
# the 2016 default tier, so a max-size image needs ``detail="high"``.
IMAGE_DETAIL_LONG_SIDE = {
    "low": 672,
    "default": 2016,
    "high": 3584,
    "auto": 2016,
}

# patch_size (14) * merge_size (2). Both image dimensions are resized to a
# multiple of this, and each 28x28 block becomes one merged vision token.
MERGED_PATCH = 28

# A square at the pixel cap -- the single largest token count one image can
# produce: (3584/28)^2 = 16384 tokens.
MAX_IMAGE_SIDE = 3584
MAX_IMAGE_TOKENS = (MAX_IMAGE_SIDE // MERGED_PATCH) ** 2

# The vision encoder batches images up to this many tokens per call before
# splitting into chunks (max_private/minimax_m3/model.py, via
# ``_vision_encoder_token_budget``). Straddling it exercises the chunk loop.
#
# The budget is the LM-side ``max_batch_input_tokens``, which defaults to 8192
# (``DEFAULT_MAX_BATCH_INPUT_TOKENS`` in
# max/python/max/pipelines/lib/pipeline_runtime_config.py). Both the kadabra
# release recipe and the llm-fuzz configs leave it at that default; the AMD and
# MTP recipes pin 4096 instead, and against those this axis straddles nothing.
# TODO(CENG-884): read the served budget from the deployment rather than
# assuming the default, so the axis follows the recipe it runs against.
VISION_CHUNK_TOKENS = 8192

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# Keyed by (width, height): signature + IHDR + IDAT, everything up to but not
# including the nonce and IEND.
_PREFIX_CACHE: dict[tuple[int, int], bytes] = {}


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def _scanlines(width: int, height: int) -> bytes:
    """Builds raw RGB scanlines as a coarse colour-block grid.

    The block edge scales with the image so large fixtures stay highly
    compressible; it never drops below one merged patch, so the pattern
    survives the server-side resize instead of averaging to flat grey.
    """
    block = max(MERGED_PATCH, min(width, height) // 8)
    # Distinct rows repeat with period ``block``, so only that many need
    # building; the rest is memcpy.
    rows: list[bytes] = []
    for band in range(max(1, height // block) if height >= block else 1):
        pixels = bytearray()
        for x in range(width):
            tone = ((x // block) + band) % 3
            pixels += bytes(
                (
                    40 + 70 * tone,
                    90 + 50 * ((band + tone) % 3),
                    200 - 60 * tone,
                )
            )
        rows.append(b"\x00" + bytes(pixels))
        if len(rows) >= 8:  # cycle after 8 distinct bands
            break
    return b"".join(rows[(y // block) % len(rows)] for y in range(height))


def _prefix(width: int, height: int) -> bytes:
    key = (width, height)
    cached = _PREFIX_CACHE.get(key)
    if cached is not None:
        return cached
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(_scanlines(width, height), 6)
    prefix = _PNG_SIGNATURE + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat)
    _PREFIX_CACHE[key] = prefix
    return prefix


def png_bytes(width: int, height: int, nonce: str = "") -> bytes:
    """Returns a decodable PNG; distinct ``nonce`` values give distinct bytes.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        nonce: Carried in a ``tEXt`` chunk. Two fixtures of the same size that
            differ only here decode identically but miss the server's
            byte-keyed preprocess cache.
    """
    body = _prefix(width, height)
    if nonce:
        body += _chunk(
            b"tEXt", b"fuzz-nonce\x00" + nonce.encode("latin-1", "replace")
        )
    return body + _chunk(b"IEND", b"")


def data_url(width: int, height: int, nonce: str = "") -> str:
    encoded = base64.b64encode(png_bytes(width, height, nonce)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def image_part(
    width: int,
    height: int,
    nonce: str = "",
    detail: str | None = None,
    max_long_side_pixel: int | None = None,
) -> dict[str, Any]:
    """Builds one OpenAI ``image_url`` content part.

    ``detail`` and ``max_long_side_pixel`` are MiniMax sizing hints; the
    router reads both from inside the ``image_url`` object
    (``openai_routes.py``) and an explicit pixel hint wins over ``detail``.
    """
    image_url: dict[str, Any] = {"url": data_url(width, height, nonce)}
    if detail is not None:
        image_url["detail"] = detail
    if max_long_side_pixel is not None:
        image_url["max_long_side_pixel"] = max_long_side_pixel
    return {"type": "image_url", "image_url": image_url}


def image_payload(
    model: str,
    parts: list[dict[str, Any]],
    prompt: str = "Describe these images in one word.",
    max_tokens: int = 16,
) -> dict[str, Any]:
    """Wraps image parts into a chat-completions request body."""
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


def resized_dims(width: int, height: int, long_side: int) -> tuple[int, int]:
    """Approximates the server-side resize.

    Mirrors ``apply_minimax_resize_rules``: the two rules are mutually
    exclusive, not sequential. An over-limit long side is downscaled and the
    short-side floor never fires -- which is why an extreme aspect ratio can
    legally land below the documented 112 px floor.

    Close enough to predict token counts for sizing test inputs, but not
    exact: the real rounding mode varies with size and aspect ratio (CENG-880
    puts the spread at roughly 5%). Never assert an exact token count on it.
    """
    w, h = float(width), float(height)
    if max(w, h) > long_side:
        scale = long_side / max(w, h)
        w, h = w * scale, h * scale
    elif min(w, h) < MIN_SHORT_SIDE_PIXEL:
        scale = MIN_SHORT_SIDE_PIXEL / min(w, h)
        w, h = w * scale, h * scale
    return (
        max(MERGED_PATCH, round(w / MERGED_PATCH) * MERGED_PATCH),
        max(MERGED_PATCH, round(h / MERGED_PATCH) * MERGED_PATCH),
    )


def estimate_tokens(width: int, height: int, detail: str | None = None) -> int:
    """Approximate merged vision-token count for one image. See ``resized_dims``."""
    long_side = IMAGE_DETAIL_LONG_SIDE.get(detail or "default", 2016)
    w, h = resized_dims(width, height, long_side)
    return (w // MERGED_PATCH) * (h // MERGED_PATCH)
