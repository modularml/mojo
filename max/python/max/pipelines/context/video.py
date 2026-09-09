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

"""Shared PyAV container-opening helper for video-decoding pipelines."""

from __future__ import annotations

from typing import IO

import av

from .exceptions import InputError

__all__ = ["open_video_container"]

# TODO(SDLC-4121): Remove once the pinned `av` build vendors an FFmpeg
# >= 8.1.2, which fixes CVE-2026-8461 (a heap out-of-bounds write in the
# MagicYUV decoder).
_BLOCKED_VIDEO_CODECS = frozenset({"magicyuv"})


def open_video_container(
    source: str | IO[bytes],
) -> av.container.InputContainer:
    """Opens a video container for reading via PyAV, rejecting known-vulnerable codecs.

    Video-decoding pipelines should open containers through this function
    rather than calling ``av.open`` directly, so the codec blocklist applies
    uniformly everywhere PyAV decodes untrusted video input.

    Args:
        source: A file path or file-like object containing the encoded video.

    Returns:
        The opened input container.

    Raises:
        InputError: If any video stream uses a blocked codec.
    """
    container = av.open(source, mode="r")
    assert isinstance(container, av.container.InputContainer)
    for stream in container.streams.video:
        codec_name = stream.codec_context.name
        if codec_name in _BLOCKED_VIDEO_CODECS:
            container.close()
            raise InputError(f"Unsupported video codec {codec_name!r}.")
    return container
