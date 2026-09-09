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

"""Tests for the shared PyAV container-opening guard."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import av
import pytest
from max.pipelines.context import InputError, open_video_container


def _make_container(codec_names: list[str]) -> MagicMock:
    container = MagicMock(spec=av.container.InputContainer)
    video_streams = []
    for codec_name in codec_names:
        stream = MagicMock()
        stream.codec_context.name = codec_name
        video_streams.append(stream)
    container.streams.video = video_streams
    return container


class TestOpenVideoContainer:
    def test_allowed_codec_passes(self) -> None:
        container = _make_container(["h264"])
        with patch(
            "max.pipelines.context.video.av.open", return_value=container
        ):
            result = open_video_container("fake_path")
        assert result is container
        container.close.assert_not_called()

    def test_blocked_codec_raises(self) -> None:
        container = _make_container(["magicyuv"])
        with patch(
            "max.pipelines.context.video.av.open", return_value=container
        ):
            with pytest.raises(InputError, match="magicyuv"):
                open_video_container("fake_path")
        container.close.assert_called_once()

    def test_opens_in_read_mode(self) -> None:
        container = _make_container(["h264"])
        with patch(
            "max.pipelines.context.video.av.open", return_value=container
        ) as mock_open:
            open_video_container("fake_path")
        mock_open.assert_called_once_with("fake_path", mode="r")
