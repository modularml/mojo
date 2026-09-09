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

"""Tests for per-architecture pixel-area context validators."""

from __future__ import annotations

import numpy as np
import pytest
from max.pipelines.context import (
    PixelContext,
    TokenBuffer,
    validate_flux2_max_pixel_area,
    validate_wan_max_pixel_area,
)
from max.pipelines.context.exceptions import InputError


def _make_context(width: int, height: int) -> PixelContext:
    return PixelContext(
        tokens=TokenBuffer(np.array([0], dtype=np.int64)),
        width=width,
        height=height,
    )


class TestFlux2PixelAreaValidator:
    def test_at_max_passes(self) -> None:
        validate_flux2_max_pixel_area(_make_context(1024, 1024))

    def test_non_square_below_max_passes(self) -> None:
        validate_flux2_max_pixel_area(_make_context(1280, 768))

    def test_above_max_raises(self) -> None:
        with pytest.raises(InputError, match="FLUX2") as excinfo:
            validate_flux2_max_pixel_area(_make_context(1536, 1536))
        msg = str(excinfo.value)
        assert "1536x1536" in msg
        assert "1048576" in msg


class TestWanPixelAreaValidator:
    def test_2048_square_passes(self) -> None:
        validate_wan_max_pixel_area(_make_context(2048, 2048))

    def test_ticket_baseline_resolutions_pass(self) -> None:
        # Resolutions explicitly iterated in wan_comparison.py per the ticket.
        for w, h in [(2048, 1152), (2048, 1536), (2048, 2048)]:
            validate_wan_max_pixel_area(_make_context(w, h))

    def test_above_max_raises(self) -> None:
        with pytest.raises(InputError, match="WAN") as excinfo:
            validate_wan_max_pixel_area(_make_context(2048, 2064))
        msg = str(excinfo.value)
        assert "2048x2064" in msg
        assert str(2048 * 2048) in msg
